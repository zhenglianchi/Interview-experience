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

## 6. SimpleStorage 与 MooncakeStore 的内存分配（双机怎么分）

### 1. 现有问题：两种后端的内存"记账口径"完全不同

面试里问"TQ 吃多少内存、双机怎么分"，最难的地方在于 **SimpleStorage 和 MooncakeStore 根本不是同一套账**：

- **SimpleStorage 按"样本条数"记账**：配置项叫 `total_storage_size`（默认 100000），单位是**条样本**，不是字节；而且它是"准入配额"，**不是预分配**——真正吃多少 RAM 取决于每条样本的字段有多宽。
- **MooncakeStore 按"字节"记账，而且是 per client**：`global_segment_size`（默认 4 GiB）和 `local_buffer_size`（默认 1 GiB）的单位是字节，但注释明确写着 **per client**——**每个客户端进程各自挂一份**，集群总量是各 client 之和，不是全局一份。

由此带来三个具体的坑：

1. **单纯看默认值会误判**：`total_storage_size: 100000` 看着很安全，但如果每条样本 68 KiB，两台机器各 50000 条就是 **6.5 GiB** 量级的 CPU 内存，可能直接把节点打爆；
2. **Mooncake 的容量是硬约束**：TQ 起的 `mooncake_master` 带 `--eviction_high_watermark_ratio=1.0 --eviction_ratio=0.0 --allow_evict_soft_pinned_objects=false`（`storage/bootstrap/mooncake_bootstrap.py:91-94`），加上客户端侧 `with_hard_pin = True`（`storage/clients/mooncake_client.py:87`）与 `-default_kv_lease_ttl=999999`（同 bootstrap `:89`）——**eviction 实际上被关闭了**，TQ 把 Mooncake 当 KV 数据库用而不是当 cache 用。段写满不会淘汰老对象，**只会写失败**；
3. **"双机"到底分几份内存，取决于有几个 `tq.init()` 进程**，而不是有几台机器。这是最容易被面试官追问的点。

### 2. 方法论：两套分配机制是怎么落地的

#### （1）SimpleStorage：按条数切给 N 个 Ray actor，SPREAD 铺到各机器，数据懒分配

**第一步：按条数均分。** 启动时对总数做向上取整除法（`storage/bootstrap/simple_storage_bootstrap.py:45`）：

$$ \text{storage\_unit\_size} = \left\lceil \frac{\text{total\_storage\_size}}{\text{num\_data\_storage\_units}} \right\rceil $$

默认 `total_storage_size: 100000`、`num_data_storage_units: 2`（`config.yaml:27,30`）→ 每个 unit **50000 条**。config 原注释还给了选参建议（`config.yaml:28-29`）：

```yaml
    # Number of distributed storage units.
    # Recommended: >= 2 x number of nodes for load balancing.
    num_data_storage_units: 2
```

**第二步：每个 unit 是一个 Ray actor，放进 SPREAD 的 placement group。** 关键代码（`simple_storage_bootstrap.py:37-46`）：

```python
storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)

for storage_unit_rank in range(num_data_storage_units):
    storage_node = SimpleStorageUnit.options(
        placement_group=storage_placement_group,
        placement_group_bundle_index=storage_unit_rank,      # rank i ↔ 第 i 个 bundle
        name=f"TransferQueueStorageUnit#{storage_unit_rank}",
    ).remote(
        storage_unit_size=math.ceil(total_storage_size / num_data_storage_units),
    )
```

而 placement group 的定义只有 CPU、没有内存（`utils/common.py:41-42`）：

```python
bundle = {"CPU": num_cpus_per_actor}          # num_cpus_per_actor=1
placement_group = ray.util.placement_group([bundle for _ in range(num_ray_actors)], strategy="SPREAD")
```

**这两行决定了双机的内存分布**：`strategy="SPREAD"` 让 Ray 尽量把每个 bundle 放到**不同的节点**，`bundle_index=rank` 又把 unit 和 bundle 一一绑定，所以 **2 个 unit 在双机集群上正好一机一个**；`bundle` 只声明 CPU、不声明 memory，意味着 **Ray 不会为存储预留任何内存**——unit 所在节点的内存是被"顺带占用"的，可能和训练进程抢内存。

**第三步（最关键）：配额 ≠ 预分配。** `StorageUnitData` 内部就是一个嵌套 dict + 一个计数集合（`storage/simple_storage.py:63-69`）：

```python
def __init__(self, storage_size: int):
    # field_name -> {global_index: data} nested dict
    self.field_data: dict[str, dict] = {}
    # Capacity upper bound (not pre-allocated list length)
    self.storage_size = storage_size
    # Track active global_index keys for O(1) capacity checks
    self._active_keys: set = set()
```

源码注释直接写明 `Capacity upper bound (not pre-allocated list length)`——**`storage_unit_size` 只是计数上限，不是 mmap / 预申请**。超限的判定是"新 key 数 + 已有 key 数 > 配额"，**直接抛错、不淘汰**（`simple_storage.py:105-111`）：

```python
new_global_keys = [k for k in global_indexes if k not in self._active_keys]
if len(self._active_keys) + len(new_global_keys) > self.storage_size:
    raise ValueError(
        f"Storage capacity exceeded: {len(self._active_keys)} existing + "
        f"{len(new_global_keys)} new > {self.storage_size}"
    )
```

释放靠显式 `clear()`，注释是 `Remove data at given global index keys, immediately freeing memory.`（`simple_storage.py:125-126`）。**所以 SimpleStorage 的真实内存公式是：**

$$ \text{RAM} \approx \sum_{\text{field}} \left(\text{已存样本数} \times \text{单条 field 字节数}\right) $$

配额单位是"条"，内存单位是"字节"，**两者之间差一个"单条样本平均宽度"**，这就是所有误判的来源。

#### （2）MooncakeStore：按字节配，而且是"每个客户端进程各自一份"

**第一步：配置与代码默认值一致，都是 4 GiB / 1 GiB，注释点明 per client**（`config.yaml:49-52`）：

```yaml
    # Global memory segment size in bytes **per client** for mounting (default: 4GB)
    global_segment_size: 4294967296
    # Local buffer size in bytes **per client** (default: 1GB)
    local_buffer_size: 1073741824
```

对应客户端代码（`storage/clients/mooncake_client.py:62-63`）：

```python
self.global_segment_size = int(config.get("global_segment_size", 4096 * 1024 * 1024))
self.local_buffer_size = int(config.get("local_buffer_size", 1024 * 1024 * 1024))
```

**第二步：这两个值在 `setup()` 里被"分配 + 注册"成两块真实内存**，不是上限声明（`mooncake_client.py:86-98`）：

```python
self.replica_config = ReplicateConfig()
self.replica_config.with_hard_pin = True          # 对象硬 pin，不可被淘汰

self._store = MooncakeDistributedStore()
ret = self._store.setup(
    self.local_hostname,        # 本机地址（为空则取 Ray 节点 IP）
    self.metadata_server,       # HTTP 元数据服务
    self.global_segment_size,   # ← 数据段：本机分配并挂载给 master
    self.local_buffer_size,     # ← 传输缓冲：本机分配并注册
    self.protocol,              # tcp / rdma
    self.device_name,
    self.master_server_address,
)
```

下钻到 Mooncake 侧（`mooncake-store/src/real_client.cpp`）可以看到两者都是**真分配**：

- `local_buffer_size` → `ClientBufferAllocator::create(local_buffer_size, protocol, use_hugepage)` + `RegisterLocalMemory(...)`（`real_client.cpp:636-643`），即"本机中转缓冲"，RDMA 收发时的 staging 区；
- `global_segment_size` → 循环 `MountSegment(ptr, size, protocol)`（`real_client.cpp:702-707` 起），内存来自 `allocate_buffer_mmap_memory` / RDMA 下的 `allocate_buffer_numa_segments`（`:720-727`），**当段大于 `max_mr_size` 时自动拆成多个 mapped_shm**（注释 `:653-655`），RDMA 下还会按有网卡的 NUMA 节点分段以打满多张 NIC（`:684-700`）。

所以 Mooncake 的两块内存是 **setup 阶段就预分配并 pin 住的常驻内存**（不是用到才涨），而 `local_buffer_size` 若配 0 则跳过注册（`:649-651`）；顺带一提，Mooncake 的 C++ binding 自身默认只给 16 MB（`mooncake-integration/store/store_py.cpp:1560`），**4 GiB 是 TQ 显式指定的**。

**第三步：谁来当 client？——每个 `tq.init()` 进程一个。** 链路是：`init()` → `_maybe_create_tq_client(final_conf)`（`interface.py:214`）→ `_TQ_CLIENT.initialize_storage_manager(manager_type=backend_name, config=conf.backend[backend_name])`（`interface.py:62`）→ `StorageManagerFactory.create(...)`（`client.py:90`）→ `KVStorageManager.__init__` 里**只创建一个**底层客户端（`storage/managers/base.py:428`）：

```python
self.storage_client = StorageClientFactory.create(client_name, config)
```

而 `tq.init()` 的行为在 `interface.py` 的 docstring 里就写清了双机语义（`interface.py:137-146`）：

```python
>>> # In process 0, node A
>>> import transfer_queue as tq
>>> tq.init()   # Initialize the TransferQueue
>>> # In process 1, node B (with Ray connected to node A)
>>> import transfer_queue as tq
>>> tq.init()   # This will only initialize a TransferQueueClient and link with existing TQ
```

第一台机器上的 `tq.init()` 走完整初始化（起 controller + storage + 自己的 client，`interface.py:148-214`）；后续机器上的 `tq.init()` 命中 `_init_from_existing()`（`interface.py:100-116`）→ **只创建一个 client**（模块级全局 `_TQ_CLIENT`，`interface.py:40`）。**每个 client 都会在本机挂一份 4 GiB 段 + 1 GiB buffer**，因此：

$$ \text{集群 Mooncake 数据段总量} = N_{\text{client}} \times \text{global\_segment\_size} $$

`local_hostname: ""` 时用 Ray 节点 IP 自动探测（`mooncake_client.py:69-74`，`get_node_ip_address()`），保证**段挂在自己这台机器上而不是 master 那台**——这是"双机内存怎么分"的物理落点。

**第四步：对象粒度决定"段里能装多少条"。** key 的格式是 `"<global_index>@<field_name>"`（`base.py:445-448`）：

```python
keys_suffixes = ["@" + f for f in sorted_fields]
keys_prefixes = [f"{i}" for i in global_indexes]
return [pfx + sfx for sfx, pfx in itertools.product(keys_suffixes, keys_prefixes)]
```

value 是**单样本单字段**（嵌套 tensor 用 `unbind()` 拆开，`base.py:466-471`）：

```python
if isinstance(field_data, Tensor) and field_data.is_nested:
    results.extend(field_data.unbind())
```

于是 **KV 对象数 = 样本数 × 字段数**，这直接影响元数据规模与 put/get 的调用次数。

**第五步：释放只有一条路。** 因为 eviction 关闭 + 硬 pin，容量不会被自动回收，只能显式 `clear_data`，或在 `close()` 时 `remove_all()`（`interface.py:249-255`）：

```python
ret = _TQ_CLIENT.storage_manager.storage_client._store.remove_all()
```

**第六步：还有第三份"隐藏内存"——CUDA tensor 的 CPU 中转副本。** put 的时候要先把 tensor 搬到 CPU 再注册（`mooncake_client.py:473-484`）：

```python
# TODO: support gpu direct rdma and use different data paths.
#       For GPU, it's more reasonable to perform data copy since
#       The register overhead is much higher than CPU
if t.device.type == "cuda":
    t = t.cpu()
t = t.contiguous()
```

**GPU-direct RDMA 尚未支持**（源码 TODO），所以每次 put 都会产生一份 CPU 侧的临时拷贝，再加上 `register_buffer` 的开销。

#### （3）双机布局：段随进程走，容量是各 client 之和

以本项目双机（node1 trainer + node2 rollout）为例，假设两台机器各有一个进程调用 `tq.init()`：

| 位置 | 进程 | `global_segment_size` | `local_buffer_size` | 说明 |
|---|---|---|---|---|
| node1 | trainer（`tq.init()` 首个进程） | 4 GiB | 1 GiB | 同时起 Controller + storage + **mooncake_master** |
| node1 | `mooncake_master` + HTTP metadata | 只存元数据（对象目录、lease/pin 表） | — | 比数据段小几个数量级；注意 `auto_init=true` 会 **kill 已存在的 mooncake_master**（`config.yaml:38`、`mooncake_bootstrap.py:43-53`） |
| node2 | rollout（`tq.init()` 第二个进程） | 4 GiB | 1 GiB | 只建 client，段挂在本机 |
| **集群合计** | | **8 GiB 可被 RDMA 访问的 KV 容量** | 2 GiB 中转缓冲 | 若每台再起 N 个进程，则各乘 (N+1) |

**要点**：段不是"master 上的一大块内存池"，而是**每个 client 在本机挂一段**，master 只做目录与路由；跨机读 = RDMA 远程读对方段，所以 **node2 写进去的 32 条轨迹物理上躺在 node2 的 4 GiB 段里**，node1 训练时是远程拉取。

#### （4）两种后端对比

| 维度 | SimpleStorage | MooncakeStore |
|---|---|---|
| 配额单位 | **样本条数**（`total_storage_size`，默认 100000） | **字节**（`global_segment_size`，默认 4 GiB） |
| 是否预分配 | 否，懒分配 dict（注释 `not pre-allocated`） | **是**，`setup()` 时 mmap/NUMA 分配 + RDMA pin |
| 双机怎么分 | 按**条数**均分给 N 个 Ray actor，`SPREAD` 铺到各节点（`bundle={"CPU":1}`，不预留内存） | 每个 client 进程在**本机**挂一份段，容量 = 各 client 之和 |
| 数据粒度 | `field_data[field][global_index]` | KV 对象，key = `"<idx>@<field>"` |
| 超限行为 | 直接 `raise ValueError`（无淘汰） | 段满 + eviction 关闭 + hard pin → **写失败** |
| 释放方式 | `clear()` 立即释放（ZMQ 通知） | 显式 clear / `remove_all()`；lease & pin TTL = 999999 |
| 跨机传输 | ZMQ（CPU 拷贝，无 RDMA） | RDMA（`protocol: rdma`），无 RDMA 自动降级 TCP |
| 额外内存 | — | `local_buffer_size` 中转缓冲 + CUDA→CPU 拷贝副本 |

### 3. 具体数值样例

#### 例 A：SimpleStorage 双机的"条数配额"换算成真实内存

```text
配置：total_storage_size = 100000，num_data_storage_units = 2
第 1 步：unit_size = ceil(100000 / 2) = 50000 条

第 2 步：SPREAD 铺到双机
  node1 → TransferQueueStorageUnit#0（配额 50000 条）
  node2 → TransferQueueStorageUnit#1（配额 50000 条）
  两台都不会为它预留内存（bundle 只有 CPU=1）

第 3 步：假设单条样本的字段（RL 轨迹常见形态）
  prompts      int64[512]  → 512 × 8 = 4096 B
  responses    int64[512]  → 512 × 8 = 4096 B
  rm_scores    fp32[1]     → 1 × 4   =    4 B
  num_turns    int64[1]    → 1 × 8   =    8 B
  单条合计 = 8204 B ≈ 8.01 KiB

第 4 步：真实内存
  单 unit 装满 50000 条：50000 × 8204 B = 410,200,000 B ≈ 391 MiB
  双机合计（100000 条）：≈ 782 MiB
  → 配额"10 万条"听起来很大，实际只占 782 MiB，完全安全

第 5 步：反过来算——给定内存预算求配额
  若只允许 storage 用 4 GiB：4,294,967,296 / 8204 ≈ 523,500 条
  → total_storage_size 可以放到 500000（每 unit 250000 条）

第 6 步：超限行为（无淘汰）
  unit#0 已存 50000 条后再 put 1 条：
    50000 existing + 1 new > 50000 → ValueError: Storage capacity exceeded
  只能等下游 mark_consumed / clear() 释放后重试

第 7 步：换成长轨迹（responses 变长是内存爆炸的主因）
  responses int64[8192] → 65536 B
  单条 = 4096 + 65536 + 4 + 8 = 69,644 B ≈ 68.0 KiB
  单 unit 50000 条：50000 × 69,644 ≈ 3.48 GB ≈ 3.24 GiB
  双机合计 ≈ 6.5 GiB —— 同样是 10 万条配额，内存涨了 8 倍
  ⇒ 结论：配 total_storage_size 之前必须先估"单条平均字节数"
```

#### 例 B：MooncakeStore 双机的容量、对象数与"第几步写爆"

```text
配置（默认）：global_segment_size = 4 GiB（每 client），local_buffer_size = 1 GiB（每 client）

第 1 步：双机各一个 tq.init() 进程
  node1 段 = 4,294,967,296 B（4 GiB）→ 数据真身存在这里
  node1 buffer = 1,073,741,824 B（1 GiB）→ RDMA 收发中转
  node2 段 = 4 GiB，node2 buffer = 1 GiB
  集群 KV 容量 = 8 GiB；RDMA 注册内存合计 = 10 GiB

第 2 步：一个 batch 的对象数（key = "<idx>@<field>"）
  batch = 32 样本，字段 4 个（prompts/responses/rm_scores/num_turns）
  对象数 = 32 × 4 = 128 个 KV 对象，分布在 node2 的段里

第 3 步：一个 batch 的字节数（同例 A 的单条宽度）
  单样本 8204 B × 32 = 262,528 B ≈ 256.4 KiB

第 4 步：8 GiB 能存多少个这样的 batch
  8,589,934,592 / 262,528 ≈ 32,720 个 batch
  → 短轨迹 + 小 batch 时，默认 4 GiB 段绰绰有余

第 5 步：长轨迹 + 大 batch —— 什么时候会爆
  单样本 69,644 B（68.0 KiB）
  batch = 1024 → 1024 × 69,644 = 71,315,456 B ≈ 68.0 MiB / step
  单机 4 GiB 段：4,294,967,296 / 71,315,456 ≈ 60 step 就写满
  ⇒ 因为 eviction_ratio=0.0 + with_hard_pin=True + lease TTL 999999，
    第 61 步不会淘汰老对象，而是 put 失败
  ⇒ 训练端必须保证 mark_consumed → clear 及时释放，或显式调大段

第 6 步：本项目与性能测试的实际取值（都是显式放开的）
  仓库内验证脚本：global_segment_size = 8,589,934,592（8 GiB）
                  local_buffer_size  = 2,147,483,648（2 GiB）  ← 默认值的 2 倍
  perftest 配置：num_data_storage_units = 16
                global_segment_size = 86,294,967,296 B ≈ 80.4 GiB
  ⇒ 说明默认 4 GiB 只是"能跑起来"的保守值，高并发 / 长轨迹必须按 batch 显式调大
```

### 4. 配置建议与演进（对应"新版特性"小节）

**（1）选参与容量规划的三条经验规则。**

- `num_data_storage_units >= 2 × 节点数`：这是源码注释直接给的（`config.yaml:29`）。原因在 `SPREAD` + `bundle_index=rank`：unit 数少于节点数时会有机器分不到 unit，负载不均；unit 数 = 2× 节点数时每台正好 2 个，读写请求可以轮转。
- `global_segment_size` 不要顶满物理内存：同一进程里还有 `local_buffer_size`（1 GiB 起）、CUDA→CPU 的临时拷贝副本、以及训练本体的 CPU 内存。工程上段大小控制在可用内存的 60% 左右比较稳。
- **有 RDMA 才用 RDMA**：`protocol: tcp` 是默认（`config.yaml:48`）。上 RDMA 时把 `device_name` 显式指到网卡（如 `mlx5_0`），否则按注释交给 Mooncake 自动选（`config.yaml:53-55`）；RDMA 下 Mooncake 会按有 NIC 的 NUMA 节点切分段（`real_client.cpp:684-700`），这也是多网卡机器要显式指定的原因。

**（2）口径差异是"双机内存分配"这一问的题眼。** 三套后端其实是三种记账口径：SimpleStorage 记"条数"、MooncakeStore 记"字节 × client 数"、Yuanrong 记"共享内存段大小"（`config.yaml:86` 的 `worker_args: "--shared_memory_size_mb 8192"`，且注释提醒 >21 GB 共享内存需要大页 `--enable_huge_tlb` + `sysctl vm.nr_hugepages` + `ulimit -l unlimited`，`config.yaml:81-85`）。回答时先说清"按什么记账、记几份"，再算数字，就不会被绕进去。

**（3）演进方向：从 CPU 段走向 GPU 直连。** 目前 `mooncake_client.py:475-477` 的 TODO 明确写着 `support gpu direct rdma and use different data paths`，理由是"GPU 的 register 开销远高于 CPU"——也就是说当前设计是**刻意**先把 CUDA tensor 拷到 CPU 再注册。一旦支持 GPU-direct RDMA，`global_segment_size` / `local_buffer_size` 的物理落点会从主机内存迁移到显存，容量规划就要和 KV cache、模型权重一起算总账——这也是"分卡 / 共卡"话题会交汇的地方（见 `Colocate-vs-Disaggregate.md`）。

> **面试一句话总结**：TQ 两种后端两套账——SimpleStorage 是按**样本条数**配额（`total_storage_size` 默认 100000，用 `ceil(总数/unit 数)` 切给 N 个 Ray actor，placement group 用 `SPREAD` + `bundle={"CPU":1}` 铺到各机器，且**只做计数校验、不预分配内存**，真实 RAM = 已存样本数 × 单条字段字节数，超限直接 `ValueError` 不淘汰）；MooncakeStore 是按**字节**配额且**per client**（`global_segment_size` 默认 4 GiB + `local_buffer_size` 默认 1 GiB，`setup()` 时 mmap/NUMA 分配并 RDMA pin），**每个 `tq.init()` 进程各挂一份、段挂在本机**，所以双机两个进程 = 8 GiB KV 容量 + 2 GiB 中转缓冲；又因为 master 侧 `--eviction_ratio=0.0 --allow_evict_soft_pinned_objects=false` + 客户端 `with_hard_pin=True` + lease TTL 999999，**eviction 实际关闭、容量是硬约束**，写满只会失败不会淘汰，必须靠显式 clear/`remove_all()` 释放——所以配 `total_storage_size` 前先估单条平均字节数，配 `global_segment_size` 前先算"单 step 字节数 × 最长未清理步数"。

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
