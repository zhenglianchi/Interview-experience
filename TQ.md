# TransferQueue（TQ）完全指南

> **verl 生态的异步流式数据管理模块：Controller（控制面）× Client（API）× Sampler（消费策略）× StorageManager（可插拔数据面）+ Mooncake backend。**

TransferQueue（TQ，`Ascend/TransferQueue`）是用于高效 post-training 的高性能数据存储与传输模块，官方定位："an asynchronous streaming data management module for efficient post-training"（对应论文 [AsyncFlow](https://arxiv.org/abs/2507.01663)）。核心能力：

1. **细粒度（sub-sample 级）数据管理 + 负载均衡**：不是按"整批数据"管理，而是跟踪**每个训练样本**的生产/消费状态；
2. **解耦显式数据依赖**：作为"数据网关"把计算任务之间的数据依赖解耦，让算法控制器（如 verl 的 trainer）设计更简单；
3. **控制面与数据面分离**：`TransferQueueController` 做元数据管理（控制面，ZMQ），`StorageManager` 做数据存取（数据面，可插拔后端）。

TQ 是 verl 的数据平面（Framework 写轨迹、trainer 读轨迹），本项目（简历）用 TQ 的 **Mooncake backend** 做双机全异步 RL 训练中 node1（trainer）与 node2（rollout）之间轨迹的跨机传输（RDMA 高速）。本指南先逐个讲清核心组件，再讲整体架构设计（含 ZMQ 控制面协议 + 数据面传输协议），最后专门讲 Mooncake backend。

> 说明：本文基于 `Ascend/TransferQueue` v0.1.8（tag 检出）讲解；Mooncake 本身的架构见 `Mooncake.md`。

---

# 一、核心组件

## 1. TransferQueueController：控制面（数据生产/消费状态管理）

### 1. 现有问题：为什么需要 Controller

训练数据（如 RL 的 rollout 轨迹）是**分阶段产生**的：一个样本先有 prompt，rollout 引擎算完补上 responses/logprobs，reward 引擎补上 rm_scores——**不同字段由不同计算任务在不同时间写入**。下游任务（如 GRPO 训练）必须等"它需要的所有字段都齐了"才能消费。如果每个计算任务都自己判断"数据齐没齐、能不能取"，逻辑会散落各处且难以协调。

Controller 的定位是**控制面的"全景数据管理"**：跟踪每个训练样本的**生产状态**（哪些字段已写入）和**消费状态**（哪些任务已读过哪些字段），一旦某样本的所有必需字段就绪，下游任务才能消费它；同时记录每个计算任务（如 `generate_sequences`、`compute_log_prob`）的消费历史——**不同任务需要相同字段时，可以各自独立消费互不干扰**（比如生成和算 logprob 都要读 responses，但各自维护消费进度）。

### 2. 方法论：Controller 是怎么实现的

`TransferQueueController`（`transfer_queue/controller.py`）的核心数据结构与接口，逐步操作：

**（1）DataPartitionStatus（分区状态）**：TQ 支持**数据分区（partition）**（对应 verl 的 train/val/test 数据集隔离）。每个分区维护：`total_samples_num`（样本总数）、`total_fields_num`（字段总数）、`allocated_fields_num`（已写入字段数）、`allocated_samples_num`（已写入样本数）。

**（2）生产状态跟踪（`update_production_status`）**：当某个任务写入字段时，Controller 更新该样本的字段生产状态；`get_production_status(data_fields)` 返回"这些字段是否全部就绪"。

**（3）消费状态跟踪（`mark_consumed` / `get_consumption_status`）**：每个计算任务（task_name）维护自己的消费进度——`mark_consumed(task_name, global_indices)` 标记已消费的样本，`get_consumption_status(task_name)` 返回该任务还能消费哪些样本。**不同 task 的消费进度相互独立**，这是"同一字段被多个任务独立消费"的实现。

**（4）字段元数据（`FieldMeta` / `get_field_schema`）**：记录每个字段的 schema（形状、dtype），`generate_batch_meta` 根据生产状态和采样策略生成一批可消费样本的 `BatchMeta`（元数据，含 global_index 和字段 schema）。

**（5）采样器（Sampler）**：Controller 用可插拔的 `Sampler` 决定"消费哪些样本"（`register_sampler`）——`SequentialSampler`（顺序）、`GRPOGrouNSampler`（GRPO 组采样）、`SeqLenBalancedSampler`（按序列长度均衡）、`RankAwareSampler`（rank 感知）等。这是 PR#101"把数据检索逻辑从 Controller 解耦"的成果——用户可以自定义消费策略。

**（6）通信方式**：Controller 是**控制面服务**，通过 **ZMQ**（`_init_zmq_socket`）接收 storage unit 和 client 的请求——`ZMQRequestType` 枚举定义了控制请求类型：HANDSHAKE（握手）、PUT/GET/CLEAR_DATA（数据操作）、GET_META（元数据）等（详见第 4 点架构设计）。

### 3. 具体数值样例

逐步演算一个 RL 样本的"生产-消费"状态流转：

```text
场景：一个训练样本需要 3 个字段齐了才能被 GRPO 训练消费：prompts、responses、rm_scores。

t0：sample #0 的 prompts 写入 → production_status[prompts]=1
    （responses/rm_scores 未就绪）→ GRPO 任务还拿不到这个样本
t1：rollout 引擎写入 responses → production_status[responses]=1
t2：reward 引擎写入 rm_scores → production_status[rm_scores]=1
    → 三个字段全就绪 → Controller 标记 sample #0 "可被消费"
t3：GRPO 任务调用 get_consumption_status("grpo") → 返回 sample #0
    → 消费后 mark_consumed("grpo", [0])
    → 注意：如果同时有"算 logprob"任务也需要 responses，
       它的消费进度独立，sample #0 对它仍是可消费的
```

**关键理解**：Controller 管理的是"**元数据**"（谁写了、谁读了、齐没齐），**数据本身存在 StorageManager 里**——这是"控制面与数据面分离"的核心。

> **面试一句话总结**：TransferQueueController 是 TQ 的控制面——按"样本级"跟踪每个字段的生产状态和每个计算任务的消费状态（可插拔 Sampler 决定消费哪些样本），用 ZMQ 与存储单元/客户端通信；它管理元数据（谁写了/谁读了/齐没齐），数据本体存在 StorageManager，实现"控制面与数据面分离"。

---

## 2. TransferQueueClient：用户 API（生产/消费入口）

### 1. 现有问题：为什么需要 Client

Controller 是控制面服务、StorageManager 是数据面服务，但**用户（训练脚本/agent framework）需要一个简单统一的 API**——不用关心底层是 ZMQ 控制 + 什么存储后端，只需"put 数据 / get 数据 / 查状态"。Client 的定位就是"**面向用户的 API 层**"，同时支持**同步（`TransferQueueClient`）和异步（`AsyncTransferQueueClient`）**两种形态（verl 的 agent framework 用异步，训练脚本可用同步）。

### 2. 方法论：Client 是怎么实现的

`TransferQueueClient`（`transfer_queue/client.py`）的关键接口，逐步操作：

```python
class AsyncTransferQueueClient:
    """异步客户端：直接暴露 async 方法。"""
    def initialize_storage_manager(self, ...): ...   # 初始化数据面（StorageManager）
    def put(self, metadata: BatchMeta, data: TensorDict) -> None: ...
        # 写入一批样本（metadata 指明写哪些 global_index 的哪些字段）
    def get_data(self, metadata: BatchMeta) -> TensorDict: ...
        # 按 metadata 取回一批样本数据
    def get_meta(self, ...) -> BatchMeta: ...
        # 向 Controller 请求"下一批可消费样本的元数据"
    def check_production_status(self, data_fields, partition_id) -> bool: ...
    def check_consumption_status(self, task_name, partition_id) -> bool: ...
    def get_consumption_status(self, task_name, partition_id): ...
    def reset_consumption(self, partition_id, task_name=None): ...
    def clear_partition(self, partition_id): ...
    def kv_retrieve_meta(self, ...): ...   # KV 后端（如 Mooncake）的元数据检索
    def kv_retrieve_keys(self, ...): ...   # KV 后端取 keys

class TransferQueueClient(AsyncTransferQueueClient):
    """同步客户端：内部跑一个事件循环 + 把 async 方法包成同步（_make_sync）。"""
    def _start_loop(self): ...      # 后台事件循环
    def _bind_sync_methods(self): ...  # 把每个 async 方法绑成同步版本
```

**关键设计**：`put(metadata, data)` 的 `metadata`（BatchMeta）告诉 StorageManager"这批量数据对应哪些 global_index 的哪些字段"，`get_data(metadata)` 按同样的 metadata 取回——**metadata 是 Controller 和 StorageManager 之间的"索引"**；同步客户端内部用后台事件循环把 async 方法包成同步（`_run(coro)` + `_make_sync`），让训练脚本可以同步调用。

### 3. 具体数值样例

```text
场景：verl 的 agent framework 写轨迹，训练脚本读轨迹。

# 写入侧（uni-agent framework，见 Uniagent.md）：
fields = {prompts: tensor, responses: tensor, rm_scores: tensor, ...}
meta = controller.generate_batch_meta(...)      # 先要 metadata
await tq.async_kv_batch_put(keys, fields, tags, partition_id)   # 或 client.put(meta, fields)

# 读取侧（verl trainer）：
meta = client.get_meta(partition_id, fields=["prompts","responses","rm_scores"])
batch = client.get_data(meta)                    # 拿到这批样本的 TensorDict
# 消费后：
client.mark_consumed(meta)  # 或 reset_consumption
```

> **面试一句话总结**：TransferQueueClient 是面向用户的 API 层——`put(metadata, data)` 写入、`get_meta()` 向 Controller 要"可消费样本的元数据"、`get_data(metadata)` 按元数据取数，同步/异步两套形态（同步内部跑事件循环包装）；metadata（BatchMeta）是控制面与数据面之间的索引。

---

## 3. StorageManager：可插拔数据面（存储后端抽象）

### 1. 现有问题：为什么需要 StorageManager

数据本体（TensorDict 形式的训练样本）必须存到某个存储里，但**不同场景需要不同的存储**：单机调试用内存（SimpleStorage）、大规模跨机用分布式 KV（Mooncake / Yuanrong）——如果 TQ 把存储逻辑写死，就没法适配不同集群。StorageManager 的定位是"**数据面的可插拔抽象**"（官方 PR#66 "Storage backends are now pluggable"）：定义统一接口，**换存储后端 = 写一个 StorageManager 子类 + 注册**。

### 2. 方法论：StorageManager 是怎么实现的

`StorageManager` 基类（`transfer_queue/storage/managers/base.py`）定义核心 API（README 明确）：

```python
class StorageManager(ABC):
    """封装 TQ 系统内的核心交互逻辑。"""
    @abstractmethod
    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """按 metadata（global_index + 字段）写入数据。"""
    @abstractmethod
    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """按 metadata 取回数据。"""
    @abstractmethod
    async def clear_data(self, metadata: BatchMeta) -> None:
        """清除数据。"""
```

现有实现（`transfer_queue/storage/managers/`）：

| StorageManager | 存储后端 | 特点 |
|---|---|---|
| `SimpleStorageManager` | CPU 内存（ZMQ） | 零依赖开箱即用，适合单机/小规模 |
| `MooncakeStorageManager` | Mooncake Store（KV） | **RDMA 高速传输**，跨机大规模（本项目的 backend） |
| `YuanrongStorageManager` | 昇腾 Yuanrong 数据系统 | HBM/DRAM/SSD 分层存储 |
| `RayStorageManager` | Ray 对象存储 | 走 Ray 共享内存 |

**分层结构**（`storage/` 目录）：`managers/`（StorageManager 子类，封装 TQ 交互逻辑）→ `clients/`（底层存储客户端，如 `mooncake_client.py` 调 `MooncakeDistributedStore`）→ `bootstrap/`（启动逻辑，如 `mooncake_bootstrap.py` 自动起 Mooncake 服务）。**配置选择**（`config.yaml` 的 `backend.storage_backend` 字段）：`SimpleStorage` / `MooncakeStore` / `Yuanrong`。

### 3. 具体数值样例

```text
场景：单机调试 vs 双机大规模训练，只改一行配置换后端。

# config.yaml
backend:
  storage_backend: SimpleStorage    # 单机调试（零依赖）
  # storage_backend: MooncakeStore  # 双机训练（RDMA 跨机传输）

# 用户代码完全不变：put_data / get_data 接口相同
await storage_manager.put_data(fields, meta)
batch = await storage_manager.get_data(meta)
```

**关键理解**：换后端只改配置、代码不动——因为 `put_data/get_data/clear_data` 接口是统一的，Controller/Client/Sampler 都不感知具体后端。这是 TQ"数据面可插拔"设计的价值。

> **面试一句话总结**：StorageManager 是 TQ 的数据面抽象——统一 `put_data/get_data/clear_data` 接口 + 可插拔后端（SimpleStorage 内存 / MooncakeStore KV+RDMA / Yuanrong 分层 / Ray），换后端只改 config 不改代码；`managers/`（交互逻辑）→ `clients/`（底层客户端）→ `bootstrap/`（启动）三层结构。

---

# 二、整体架构设计

## 4. 架构设计：控制面（ZMQ）与数据面（存储）如何协作

### 1. 现有问题：Controller / Client / StorageManager 如何串成一个系统

前三个组件各自独立，但 TQ 的价值在于它们的**协作架构**：Client 是用户入口、Controller 是控制面（元数据）、StorageManager 是数据面（数据本体）——三者通过 **ZMQ（控制面）** 和**存储后端自身的传输机制（数据面）**通信。这一节讲清整体架构、每一步的传输协议、以及一次完整 put/get 的流程。

### 2. 方法论：整体架构是怎么组织的

TQ 的整体架构（README 的 tq_arch 图 + 源码）：

```text
┌────────────────────── 用户侧（训练脚本 / agent framework） ──────────────────────┐
│  TransferQueueClient（Async 异步 / Sync 同步）                                    │
│    put(meta, data) / get_meta() / get_data(meta) / check_production_status ...    │
└───────────────┬────────────────────────────────┬─────────────────────────────────┘
                │ ZMQ（控制请求）                  │ 数据面调用
                ▼                                ▼
┌───────────────────────────┐      ┌──────────────────────────────────────────┐
│  TransferQueueController   │      │  StorageManager（数据面）                  │
│  （控制面，ZMQ 服务）        │      │  ├─ SimpleStorageManager（CPU 内存 + ZMQ）│
│  · 生产状态 / 消费状态       │◀────▶│  ├─ MooncakeStorageManager（KV + RDMA）  │
│  · Sampler（消费策略）      │      │  ├─ YuanrongStorageManager（HBM/DRAM/SSD）│
│  · BatchMeta 生成           │      │  └─ RayStorageManager（Ray 对象存储）      │
│  · 数据分区管理             │      │  clients/（mooncake_client 等底层客户端）   │
└───────────────────────────┘      └──────────────────────────────────────────┘
```

**控制面协议（ZMQ）**：Controller 和 StorageManager 之间、Controller 和 Client 之间用 **ZMQ** 通信。`ZMQRequestType` 枚举定义所有控制请求类型：

- **HANDSHAKE**：StorageManager（storage unit）启动时向 Controller 注册（HANDSHAKE → HANDSHAKE_ACK），建立连接；
- **DATA_OPERATION**：`PUT` / `GET` / `CLEAR_DATA` 及对应的 `*_RESPONSE` / `*_ERROR`——Client 请求数据操作时，先经 Controller 协调（确认字段就绪、分配索引），实际数据传输由 StorageManager 完成；
- **META_OPERATION**：`GET_META` / `GET_PARTITION_META` 等——Client 向 Controller 要"下一批可消费样本的元数据"。

ZMQ 的 socket 模式（`utils/zmq_utils.py`）：Controller 是服务端，Client/StorageManager 用 `zmq.DEALER` 连接（`create_zmq_socket(context, zmq.DEALER, ...)`），支持异步多路请求。

**数据面传输协议（按后端）**：

| 后端 | 数据面协议 | 说明 |
|---|---|---|
| SimpleStorage | **ZMQ**（内存数据经 ZMQ 传递） | 单机、CPU 内存 |
| MooncakeStore | **RDMA（RoCE/IB）/ TCP** | 跨机 KV 存储，RDMA 高速（见第 5 点） |
| Yuanrong | 昇腾原生分层存储 | HBM/DRAM/SSD |
| Ray | Ray 共享内存 | Ray 集群内 |

**一次完整 put/get 的流程**（以 verl 训练为例，逐步操作）：

```text
【写入侧】（agent framework / rollout 引擎）
第 1 步：controller.generate_batch_meta(...) → 得到这批样本的 metadata
        （global_index + 字段 schema）
第 2 步：async_kv_batch_put(keys, fields, tags, partition_id)
        （或 client.put(metadata, data)）→ StorageManager.put_data
第 3 步：StorageManager 把数据写入后端（SimpleStorage 内存 / Mooncake KV），
        并通知 Controller 更新 production_status

【读取侧】（verl trainer / 下游计算任务）
第 4 步：client.get_meta(partition_id, fields=[...]) → ZMQ 问 Controller：
        "这些字段都就绪的可消费样本有哪些？"（走 Sampler 策略）
第 5 步：Controller 返回 BatchMeta（可消费样本的 global_index）
第 6 步：client.get_data(meta) → StorageManager.get_data 取回 TensorDict
第 7 步：消费后 client.mark_consumed(meta) → Controller 更新该任务的 consumption_status
```

**关键设计**：① 控制面（ZMQ）只传"元数据/状态"，数据本体走数据面（存储后端）——**控制流与数据流分离**；② 生产/消费状态是"字段级 × 任务级"的细粒度矩阵，不同任务可独立消费同一字段；③ 数据分区（partition）隔离 train/val/test。

### 3. 具体数值样例

假设双机 RL 训练（对应本项目简历：node1 trainer + node2 rollout），逐步演算 TQ 的完整数据流：

```text
环境：node1（trainer）+ node2（rollout 引擎），TQ 用 MooncakeStore 后端。

第 1 轮（node2 产生轨迹）：
  · agent 完成 32 个 rollout → framework 把轨迹转成 TQ 字段
    （prompts/responses/rm_scores/num_turns...）
  · async_kv_batch_put 写入（MooncakeStore，RDMA 跨机传输）→
    Controller 更新 production_status（32 个样本的字段就绪）

第 2 轮（node1 消费训练）：
  · verl trainer 每步 kv_batch_get → client.get_meta：
    Controller 用 GRPOGrouNSampler 选出可消费的样本（需 rm_scores 就绪）
  · client.get_data(meta) → MooncakeStorageManager 经 RDMA 拉回 32 个样本
  · GRPO 训练一步 → mark_consumed → Controller 更新消费状态

第 3 轮（重叠）：node2 采第 N+1 批时 node1 在训第 N 批——
  TQ 的"生产/消费解耦"让两侧互不阻塞（这就是双机全异步 -39% 的数据面基础）
```

> **面试一句话总结**：TQ 整体架构 = Client（用户 API）+ Controller（控制面，ZMQ 管理生产/消费状态 + Sampler）+ StorageManager（数据面，可插拔后端）；控制流走 ZMQ（HANDSHAKE/PUT/GET/GET_META），数据流走后端自身（SimpleStorage 走 ZMQ、Mooncake 走 RDMA/TCP）；一次 put/get = "generate_batch_meta → put_data → get_meta（Controller 按 Sampler 选样本）→ get_data → mark_consumed"，生产与消费完全解耦，是双机全异步训练的数据面基础。

---

## 5. Mooncake backend：TQ 的 KV 存储后端（重点）

### 1. 现有问题：为什么 TQ 需要 Mooncake backend

SimpleStorage（CPU 内存 + ZMQ）在**跨机大规模训练**时有三个瓶颈：① 数据在机器间要经 CPU 内存中转，带宽低、延迟高；② 内存容量有限，轨迹数据量大时放不下；③ 无法利用 RDMA 高速传输。Mooncake（见 `Mooncake.md`）提供分布式 KV 存储 + RDMA 高速传输——TQ 把 Mooncake 作为**可插拔的 KV 存储后端**，让轨迹数据跨机传输走 RDMA，这正是本项目简历"双机全异步 RL 训练（TQ MoonCake 存储后端）"的技术基础。

### 2. 方法论：TQ 的 Mooncake backend 是怎么实现的

TQ 的 Mooncake 集成在 `transfer_queue/storage/` 下三层：

**（1）`clients/mooncake_client.py`（底层客户端）**：`MooncakeStoreClient`（注册为 `MooncakeStoreClient`，实现 `StorageKVClient` 接口）直接调 `mooncake.store.MooncakeDistributedStore`（`from mooncake.store import MooncakeDistributedStore, ReplicateConfig`）。关键配置（`__init__`）：

```python
class MooncakeStoreClient(StorageKVClient):
    def __init__(self, config: dict[str, Any]):
        # 需要安装 mooncake-transfer-engine，否则 ImportError
        # 关键配置：
        self.local_hostname = config.get("local_hostname", "")       # 本机地址
        self.metadata_server = config.get("metadata_server", None)   # Mooncake HTTP 元数据服务
        self.master_server_address = config.get("master_server_address")  # Mooncake master RPC
        ...
    # 内部用 ThreadPoolExecutor + as_completed 批量读写（MAX_BATCH_WORKER_THREADS=4）
    # BATCH_SIZE_LIMIT=400、MAX_RETRIES=3（失败重试）
```

**（2）`managers/mooncake_manager.py`（StorageManager 子类）**：`MooncakeStorageManager(KVStorageManager)`——实现 `put_data` / `get_data` / `clear_data` 接口，内部调 `MooncakeStoreClient` 读写 KV；`KVStorageManager` 是 KV 型后端的基类（统一 KV 语义，如按 key 存 TensorDict）。

**（3）`bootstrap/mooncake_bootstrap.py`（启动）**：`MooncakeStoreBootstrap`——TQ 自动启动 Mooncake 元数据服务（config 里 `auto_init: true` 时，TQ 会自动起 Mooncake master/metadata server；**注意**：config 警告 `auto_init=true` 时会尝试终止已存在的 mooncake_master 进程）。

**config.yaml 的 MooncakeStore 配置**：

```yaml
backend:
  storage_backend: MooncakeStore    # 切到 Mooncake 后端
  MooncakeStore:
    auto_init: true                 # TQ 自动起 Mooncake 元数据服务
    # metadata_server / master_server_address 等（auto_init 时自动生成）
```

**传输协议**：TQ 的 Mooncake backend 数据面走 **Mooncake Transfer Engine 的 RDMA（RoCE/IB）**——轨迹 TensorDict 写入 Mooncake Store（KV），跨机读取时经 RDMA 高速传输（40 GB 数据在 8×400Gbps RoCE 下达 190 GB/s，见 `Mooncake.md`）；无 RDMA 时 Mooncake 自动降级 TCP。

### 3. 具体数值样例

**本项目的双机全异步训练**（简历"双机分离式全异步 RL 训练（TQ MoonCake 存储后端）：平均单步耗时由同步方案的 79.4s 降至 48.1s（约 -39%）"），逐步演算 Mooncake backend 的作用：

```text
环境：node1（trainer，1×4090 48G）+ node2（rollout 引擎，2×4090 24G）
配置：batch 32 / 并发 64 / MooncakeStore 后端

第 1 步：node2 的 agent 完成 32 个 rollout → framework 转 TQ 字段
        → tq.async_kv_batch_put → MooncakeStorageManager
        → MooncakeStoreClient 写 Mooncake Store（KV）
        → 数据经 RDMA（若配置）或 TCP 传输到存储
第 2 步：node1 的 verl trainer kv_batch_get → get_meta（Controller 确认字段就绪）
        → get_data → MooncakeStoreClient 经 RDMA 拉回 32 个样本 → GRPO 训练
第 3 步：采样与训练完全重叠（node2 采 N+1 批时 node1 训 N 批）
        → 单步 48.1s（vs 同步 79.4s，-39%）

排障点（本项目实际修过的 TQ+Mooncake 链路 bug）：
· num_turns 13B 写读类型不一致：verl padding 行 Python int 走 msgpack 13B
  vs 训练端 int64 8B 读 → "Buffer too small"（修复在 padding_utils）
· 空响应轨迹 0 字节 slice：max_trajectory_length 截断占位写入 Mooncake 被拒
  → framework 跳过空轨迹（见 Uniagent-Lighting.md）
```

> **面试一句话总结**：TQ 的 Mooncake backend = `MooncakeStorageManager`（StorageManager 子类）+ `MooncakeStoreClient`（调 MooncakeDistributedStore）+ `MooncakeStoreBootstrap`（自动起服务），数据面走 Mooncake Transfer Engine 的 RDMA 高速传输；本项目用它做双机全异步 RL 训练的数据平面（单步 79.4s→48.1s，-39%），并修复了 num_turns 13B、空轨迹 0 字节 slice 等链路 bug。

---

## 附：组件速查表

| 组件 | 代码位置（transfer_queue/） | 角色 | 关键接口 / 类 |
|---|---|---|---|
| Controller | `controller.py` | 控制面：生产/消费状态 + Sampler + 分区 | `TransferQueueController`、`DataPartitionStatus`、`FieldMeta`；ZMQ 服务 |
| Client | `client.py` | 用户 API（同步/异步） | `TransferQueueClient`、`AsyncTransferQueueClient`；`put/get_data/get_meta/mark_consumed` |
| Sampler | `sampler/` | 消费策略 | `SequentialSampler`、`GRPOGrouNSampler`、`SeqLenBalancedSampler`、`RankAwareSampler` |
| StorageManager | `storage/managers/` | 数据面抽象 | `StorageManager`（put_data/get_data/clear_data）、`KVStorageManager` |
| SimpleStorage | `storage/simple_storage.py` | CPU 内存后端 | `SimpleStorageUnit`、ZMQ 传输 |
| Mooncake backend | `storage/managers/mooncake_manager.py` + `clients/mooncake_client.py` + `bootstrap/mooncake_bootstrap.py` | KV 后端（RDMA） | `MooncakeStorageManager`、`MooncakeStoreClient`、`MooncakeStoreBootstrap` |
| ZMQ 协议 | `utils/zmq_utils.py` | 控制面通信 | `ZMQRequestType`（HANDSHAKE/PUT/GET/GET_META）、`ZMQServerInfo`、DEALER socket |

> **关联文档**：Mooncake 本身（Transfer Engine / Store / EP/PG / 传输协议）详见 `Mooncake.md`；uni-agent 的 Framework 如何写 TQ（`_trajectory_to_tq_field_and_tag`）见 `Uniagent.md`；本项目双机全异步训练见 `Uniagent-Lighting.md`。
