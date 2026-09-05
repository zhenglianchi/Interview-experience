# Mooncake 完全指南（v0.3.10.post2：Master 控制面 × Client 双角色 × Transfer Engine 全架构）

> 对应简历"了解 Mooncake 分布式 KV Cache 存储传输架构"与实习"TransferQueue 的 MoonCake 存储后端接入、RDMA 高速传输"。本文**完整讲解 Mooncake 自身架构**（不是只讲与 TQ 的结合）：v0.3.10.post2 已拆成 monorepo（`mooncake-transfer-engine` / `mooncake-store` / `mooncake-common` / `mooncake-pg` / `mooncake-ep` 等），核心是两个子系统 + 一个总思想：
> **① Mooncake Store = 分布式 KV 缓存存储引擎**，架构上只有**两个关键组件——Master（控制面/协调者，管元数据与分配）和 Client（双角色：既是发起请求的客户端，又是贡献内存的存储节点）**；**② Transfer Engine = 数据面传输引擎**，用 (GPUDirect) RDMA 把数据从一方 DRAM/VRAM 零拷贝搬到另一方 DRAM/VRAM，并聚合多网卡带宽。
>
> 素材：本地 `kvcache-ai/Mooncake`（已 checkout **tag `v0.3.10.post2`**，HEAD `e1d6d6f`）+ 官方设计文档 `docs/source/design/{architecture,mooncake-store}.md` + `docs/source/deployment/mooncake-store-deployment-guide.md`。与 `TQ.md` 的关系：TQ 的 MoonCake 后端只是"用 Store 的 Client API 存轨迹"，本文讲的是它背后完整的东西。

---

# 一、总体架构：一张图看懂 Mooncake 的组件与角色

## 1. 两个子系统 + 元数据服务：组件全景

### 1. 现有问题

Mooncake 0.3.10 拆成 monorepo 后，新人最容易晕的是"一堆子项目到底谁是谁"。先钉死全景，后面每个组件展开。

### 2. 方法论（组件分层）

```text
┌────────────────────────────────────────────────────────────────┐
│ 上层应用：vLLM（KV transfer / MooncakeConnector）、verl、TQ、    │
│           SGLang、训练框架……（只认识 Client / TransferEngine API）│
├────────────────────────────────────────────────────────────────┤
│ ① Mooncake Store 的【控制面】= Master Service（mooncake_master 进程）│
│    职责：集群逻辑内存池编排、节点加入/离开、对象空间分配（PutStart）│
│         元数据维护（object key → slice → replica 位置）、驱逐策略  │
│    部署：单点 或 HA（etcd 选举 leader，leader 心跳监控 client）    │
├────────────────────────────────────────────────────────────────┤
│ ② Mooncake Store 的【数据面】= Client（双角色！）                 │
│    角色 A（客户端）：被上层应用调用，发 Put/Get/Remove 等请求      │
│    角色 B（store server）：贡献一段连续内存（global segment），    │
│          让它被其他 Client 读写——数据面是 Client↔Client 直传，    │
│          完全绕过 Master！                                       │
│    三种形态：嵌入式 / dummy-real（每 rank dummy + 每 instance    │
│            real）/ 独立 store 服务（python -m mooncake.mooncake_ │
│            store_service）                                      │
├────────────────────────────────────────────────────────────────┤
│ ③ 数据面传输 = Transfer Engine（lib，每节点/每进程内嵌）           │
│    init(metadata, name) → installTransport(rdma/tcp/...) →      │
│    registerLocalMemory(注册 GPU/CPU 内存) → openSegment(远端发现) │
│    → submitTransfer(批量 READ/WRITE，零拷贝)                    │
├────────────────────────────────────────────────────────────────┤
│ ④ 元数据服务（独立于 Master 部署！）：etcd / Redis / HTTP          │
│    Transfer Engine 用它做节点/segment 目录（openSegment 握手拿 rkey）│
│    Master 的 HA 也用 etcd（选举）                                │
└────────────────────────────────────────────────────────────────┘
```

**关键澄清（面试口径）**：官方术语里**没有叫 "controller" 的组件**——用户口中的 controller/控制面就是 **Master Service**（元数据 + 分配协调）；store 不是一个独立服务名，而是 **Client 的"store server"角色**（或独立 `mooncake_store_service`）；真正的数据搬运是 **Transfer Engine**。

**数据流总原则**：`Put`/`Get` 的对象元数据（放哪、有几个 replica）问 **Master**；数据的实际搬运用 **Transfer Engine** 在 **Client↔Client** 之间直传，**不经过 Master**——控制面与数据面分离，Master 不是吞吐瓶颈。

### 3. 具体数值样例

- 4 机集群：1 台跑 `mooncake_master`（控制面）+ 每台跑一个 Client（同时是 store 节点，各贡献一段 DRAM，如 64 GB global segment）；vLLM 每 rank 内嵌一个 dummy client，转发到本机的 real client；
- vLLM 的 KV transfer（decode 请求前缀缺失时）：decode 节点 Client 发 `Get(key)` → 问 Master 拿 replica 位置 → 用 TransferEngine RDMA 从 prefill 节点的 Client（store 角色）直接拉 KV——数据不经过 Master；
- 对比：如果所有数据都走 Master 转发，Master 的 RPC/网络会成为瓶颈——所以设计成"元数据问 Master、数据 Client 直传"。

> 面试一句话总结：**Mooncake = Master（控制面，管对象元数据与分配）+ Client（双角色：请求方 + 存储节点，数据面 Client↔Client 直传绕过 Master）+ Transfer Engine（零拷贝 RDMA 传输）+ 独立元数据服务（etcd/HTTP）；官方没有 "controller"，它就是 Master——控制面管"放哪"，数据面管"怎么搬"，两者解耦。**

---

# 二、控制面：Master Service（mooncake_master）

## 2. Master 的职责与双部署模式

### 1. 现有问题

分布式对象存储需要"谁知道对象在哪"的权威——谁维护 object→位置映射？谁决定新对象放哪、满了驱逐谁？节点挂了怎么发现？

### 2. 方法论（Master 是什么、怎么跑）

**Master Service 是独立进程**（源码 `mooncake-store/src/master.cpp`），对外暴露 **RPC 服务**（默认端口 50051）+ 可选 HTTP metadata server（8080，给 Transfer Engine 当节点目录）+ metrics（9003）。核心职责（官方设计文档）：

- **编排集群逻辑内存池**：管理所有 Client（store 角色）贡献的 global segment，处理节点加入/离开；
- **对象空间分配**：`PutStart` 时决定对象的每个 slice 放哪个 segment（分配策略）；
- **元数据维护**：`object key → slice list → replica 位置` 的映射，`Get` 前先查（`GetReplicaList`）；
- **驱逐策略**：水位触发（`--eviction_high_watermark_ratio` 默认 0.95）、TTL、soft pin（默认 30 分钟）——专门为 LLM 推理负载优化；
- **Leader 心跳监控 Client 健康**：client 崩溃快速发现、恢复自动重连。

**两种部署模式**（availability 取舍）：
1. **默认单点**：一个 master，部署简单，但 master 挂了整个系统停摆（单点故障）；
2. **HA 模式**：多个 master 通过 **etcd 协调选举 leader**（源码 `mooncake-store/src/ha/`：`standby_controller`/`standby_state_machine`/`leadership`/`oplog`/`snapshot`）——leader 挂了自动重选，元数据经 oplog/snapshot 持久化恢复。

**注意**：Transfer Engine 需要的 **metadata service（etcd/Redis/HTTP）不属于 Master**，要单独部署——Master 与 TE 元数据是两个东西（很多人混）。

### 3. 具体数值样例

启动单点 master（部署文档真实参数）：
```bash
mooncake_master --rpc_port=50051 --rpc_thread_num=64 \
  --enable_http_metadata_server=true --http_metadata_server_port=8080 \
  --allocation_strategy=free_ratio_first --eviction_high_watermark_ratio=0.95 \
  --metrics_port=9003
```
- HA 时 client 的 `master_server_entry = "etcd://ip1:2379;ip2:2379;..."`，leader 选举 + 心跳（client TTL 默认 10s）；
- 驱逐：`--eviction_high_watermark_ratio=0.95`——DRAM 用到 95% 触发，按 `--eviction_ratio=0.05` 批量驱逐（优先逐未 pin / 超 TTL 的）。

> 面试一句话总结：**Master = 独立的控制面进程（RPC 50051 + 可选 HTTP metadata 8080），管"对象元数据 + slice 放置 + 驱逐"；可单点或 HA（etcd 选举 leader + oplog/snapshot）；它的元数据服务和 Transfer Engine 的 etcd/HTTP 节点目录是两回事，别混。**

---

## 3. 对象分配与元数据：PutStart / GetReplicaList 的协议语义

### 1. 现有问题

对象是"大"的（KV cache 一个对象几百 MB），要拆成 slice 才能条带化用多网卡并行传；每个 slice 放哪个节点由谁定？replica 怎么保证不扎堆？

### 2. 方法论（MasterClient 的 RPC 语义，源码 `master_client.h`）

客户端（Client 内部的 MasterClient）与 Master 的交互核心方法：

```cpp
// mooncake-store/include/master_client.h —— 客户端问 Master 的 RPC
PutStart(key, slice_lengths, ...)          // ① Put 前：问 Master"把 N 个 slice 放哪"
UpsertStart(...)                           //     （Master 分配 segment，返回每 slice 的位置）
BatchPutStart(keys, ...)                   //     批量版
GetReplicaList(object_key)                 // ② Get 前：拿对象的 replica 位置列表
BatchGetReplicaList(keys)                  //     批量版
BatchQueryIp(client_ids)                   // ③ 拿位置后：查目标节点 IP（建 RDMA 连接用）
CalcCacheStats()                           //     统计
```

**分配/放置的保证**（官方设计文档"Replication Guarantees"）：
- 同一对象的**不同 slice 保证放不同 segment**（避免单节点故障全丢）；
- **不同对象**的 slice 可以共享同一 segment（提高内存利用率）；
- replica 是 **best-effort**（尽力而为）：空间不够时少放几个 replica 也能写成功，不阻塞；
- 分配策略可配：`random`（基线，最快）/ `free_ratio_first`（采样多个候选、选空闲率最高的，负载均衡好）。

**数据面为什么绕过 Master**：Master 只回答"元数据"（放哪），拿到位置后客户端之间直接用 Transfer Engine 传——Master 的 RPC 只出现在 Put/Get 的"询问"阶段，不在数据搬运热路径上。

### 3. 具体数值样例

- 一个 600 MB 的对象（如一段长上下文 KV）拆成 `slice_lengths = [100MB × 6]` 6 个 slice；
- `PutStart` → Master 按 `free_ratio_first` 从 3 个候选 segment 里挑 2 个空闲率最高的，把 6 个 slice 交错放（slice0/2/4 到 segA，slice1/3/5 到 segB，**保证同对象 slice 不同 segment**）；
- `GetReplicaList` 返回位置 → `BatchQueryIp` 拿 IP → Transfer Engine 6 路并行 RDMA 拉回（条带化聚合多网卡）；
- replica=2 时同样的分配再执行一遍到另两个 segment（best-effort）。

> 面试一句话总结：**对象按 slice 切分、Master 决定每个 slice 放哪个 segment（同对象 slice 强制不同 segment、跨对象可共享、replica best-effort、策略 random/free_ratio_first）；客户端只把 Master 当"元数据查询"，数据用 Transfer Engine 在节点间直传——控制面管布局、数据面管吞吐。**

---

# 三、数据面：Client（双角色 + 三种形态）

## 4. Client 的双角色与三种使用形态

### 1. 现有问题

"存储节点"到底是谁？是独立进程还是一个库？vLLM 进程里怎么嵌？——官方设计给了很优雅的答案：**Client 一个类、两种角色、三种用法**。

### 2. 方法论（源码 `client_service.h` / `real_client.h` / 设计文档）

**① 双角色**：
- **客户端角色**：被上层应用调用，发 `Put/Get/Remove`（对象级 API）——`Get` 把数据读进**预注册的本地 DRAM/VRAM slice**，`Put` 把本地 slice 写进分布式池；
- **store server 角色**：贡献一段**连续内存（global segment）**加入分布式 KV 池，让其他 Client 读写——**数据实际从"一个 Client 的 segment"传到"另一个 Client 的 slice"**，全程绕过 Master。

**开关控制纯角色**：
- `global_segment_size = 0` → 纯客户端（只发请求、不贡献内存）；
- `local_buffer_size = 0` → 纯存储节点（只提供内存、不允许本实例 Get/Put）。

**② 三种使用形态**：

| 形态 | 说明 | 适用 |
|---|---|---|
| **嵌入式** | 与 LLM 推理程序同进程（vLLM 实例），import 共享库；`global_segment_size>0` 时同时贡献内存 | 单实例简单场景 |
| **dummy-real** | 每个 **rank** 持嵌入式 **dummy** client（无资源）；每个 **instance** 一个 **real** client（持 global segment）；dummy 转发请求给同 instance 的 real，两者 RPC + 共享内存零拷贝 | TP=8：8 dummy + 1 real，避免每 rank 都开 RPC 端口/管内存 |
| **独立 store 服务** | `python -m mooncake.mooncake_store_service` 包装一个 client 提供 global 内存/SSD 池；嵌入式 client 设 `global_segment_size=0` 只出网卡 | 存储与推理引擎分离部署 |

**③ 客户端 API 签名**（`Init` + 对象级操作）：

```cpp
// 设计文档 + real_client.h
ErrorCode Init(local_hostname, metadata_connstring, protocol, protocol_args, master_server_entry);
tl::expected<void, ErrorCode> Get(object_key, std::vector<Slice>& slices);   // 读进预注册 slice
tl::expected<void, ErrorCode> Put(ObjectKey& key, std::vector<Slice>& slices,
                                  const ReplicateConfig& config);            // 从本地 slice 写进池
// Remove / RemoveByRegex / RemoveAll；slice 复制；SSD 持久化开关
```

### 3. 具体数值样例

- vLLM 8×TP：8 个 rank 各一个 dummy client（零资源、只转发）+ 1 个 real client（持 64 GB global segment + 8 个网卡）——KV transfer 时 decode rank 的 dummy 把 `Get` 请求经共享内存转发给 real，real 用 Transfer Engine 拉数据回本地 slice；
- 训练侧（本项目 TQ MoonCake 后端）：每个 TQ storage worker 内嵌一个 client（embedded），`global_segment_size` 按机器 DRAM 配额设——轨迹数据以对象形式 Put 进池，训练侧另一节点的 client Get 出来（RDMA 直传）。

> 面试一句话总结：**Mooncake 的数据面只有一个类 Client 但身兼两职：对上当"请求方"（Get/Put/Remove 对象），对下当"存储节点"（贡献 global segment 让别人读写）；三种形态（嵌入式 / dummy-real 每 rank 转发 / 独立 store 服务）覆盖"推理进程内嵌"与"存储分离部署"两种拓扑。**

---

## 5. Put / Get 完整流程（数据面怎么绕开 Master）

### 1. 现有问题

一次 `Put` 或 `Get` 到底经过哪些步骤、谁和谁通信？——这是把控制面/数据面讲透的最好抓手。

### 2. 方法论（逐步流程，源码/设计文档）

**Put（写入分布式池）**：

```text
① 应用把数据放进本地预注册 slice（DRAM/VRAM，registerLocalMemory 过）
② Client 调 MasterClient.PutStart(key, slice_lengths, ...)  → Master 分配：
   同对象 slice 分散到不同 segment（free_ratio_first / random）
③ Client 用 TransferEngine.submitTransfer（WRITE）把每个 slice
   RDMA 写到目标 Client（store 角色）的 global segment 对应 offset
   ——数据 Client↔Client 直传，不经过 Master
④ 每个 slice 写完后向 Master 确认（对象可见性）
   ——Put 成功即不可变，后续 Get 一定读到完整最新值（强一致）
⑤ replica>1：同样的写复制到另几个 segment（best-effort）
⑥ 若开持久化：异步把对象 offload 到 SSD（多层存储）
```

**Get（读回对象）**：

```text
① 应用准备本地预注册 slice（注意：不是 global segment，是本地专用内存）
② Client 调 MasterClient.GetReplicaList(key) → 拿 replica 位置列表
③ Client 调 MasterClient.BatchQueryIp(位置) → 拿目标节点 IP（建 RDMA 连接）
④ Client 用 TransferEngine.submitTransfer（READ）从远端 Client 的 segment
   并行拉回各 slice（条带化多网卡）
⑤ 数据齐 → 返回给应用；若开持久化且池里没有 → 回退从 SSD 读
   （Get 保证返回完整正确数据）
```

**为什么数据面绕过 Master 后还能强一致**：Master 只维护"已提交对象"的元数据；对象 Put 完成后不可变（KV cache 语义），Get 一定读完整版本——不需要分布式锁/事务，轻量。

### 3. 具体数值样例

- 200 MB KV 对象 Get：`GetReplicaList`（RPC ~ms 级）→ `BatchQueryIp` → 4 路 RDMA（4×200 Gbps 网卡聚合）并行拉 4 个 50 MB slice ≈ 数十 ms（vs 单路 ~4× 时间）；
- CPU 开销：GPUDirect RDMA 从远端 VRAM 直读本端 VRAM，CPU 只在 submit 与完成通知时参与（"negligible CPU"）；
- 一致性数值：Put 期间 Get 可能看到旧版本或失败（重试），Put 完成（Master 标记）后所有 Get 都读到最新——对象级强一致、无中间态。

> 面试一句话总结：**Put = 问 Master 分配（PutStart）→ Transfer Engine WRITE 直传目标 Client 的 segment → Master 确认可见；Get = 问 Master 位置（GetReplicaList + QueryIp）→ Transfer Engine READ 并行拉回本地 slice；数据全程 Client↔Client 零拷贝 RDMA、绕过 Master，对象 Put 后不可变保证强一致。**

---

# 四、数据面存储：分层与淘汰

## 6. DRAM 池 + SSD offload + 驱逐：给 KV 用的内存管理

### 1. 现有问题

KV cache 池主要放 DRAM（快但贵），放不下怎么办？对象多了怎么逐？——Mooncake 做了 LLM 推理场景特化的存储管理。

### 2. 方法论

- **多级存储**：DRAM（global segment，主池）→ SSD（`--root_fs_dir` 挂载的 DFS / NVMe，offload 层）——`Put` 带持久化时异步落 SSD，`Get` 池内没有时回退 SSD 读（源码 `file_storage.cpp`/`storage_backend.cpp`/`uring_file.cpp`）；
- **分配器**：`offset_allocator` 管理 segment 内空间，segment 由 client 注册（`registerLocalMemory` 的 CPU 内存或 GPU 内存）；
- **驱逐**：Master 侧统一决策（不是各节点本地）——`count_min_sketch` 统计热度、`eviction_strategy`（watermark 触发、TTL `--default_kv_lease_ttl` 默认 5000ms、soft pin `--default_kv_soft_pin_ttl` 默认 30min、`--allow_evict_soft_pinned_objects`）；
- **HA 下的健康**：Master leader 心跳监控所有 client，崩溃节点上的 segment 标记失效、对象从 replica 补。

### 3. 具体数值样例

- 节点 DRAM 128 GB，`global_segment_size=96 GB`；KV 对象累计到 91 GB（95% 水位）→ Master 触发驱逐，按热度逐 ~4.8 GB（5%）——优先逐"soft pin 过期 / 低热度"对象，KV 传输热点对象用 `pin` 保住；
- SSD offload：冷 KV（很久没被 decode 复用的前缀）异步落 `/mnt/nvme/mooncake_dfs`，命中率低的对象 Get 时走 `uring` 从 SSD 读——DRAM 只留热数据。

> 面试一句话总结：**存储 = DRAM 主池（Client 的 global segment + offset_allocator）+ SSD offload（DFS/NVMe，Put 异步落、Get 兜底读）；驱逐由 Master 统一按热度/水位/TTL/soft-pin 决策（count-min-sketch + eviction_strategy），是"为 LLM KV 复用模式调过"的内存管理。**

---

# 五、传输面：Transfer Engine

## 7. TransferEngine 核心 API：注册内存 → 发现远端 → 批量传

### 1. 现有问题

数据面的"怎么搬"由 Transfer Engine 负责。它的核心抽象是什么？一次传输怎么组织？

### 2. 方法论（源码 `transfer_engine.h`，逐步操作）

**TransferEngine 是每进程一个的实例（lib）**，对上层暴露统一 API：

```cpp
// mooncake-transfer-engine/include/transfer_engine.h —— 核心 API
int init(metadata_conn_string, local_server_name, ip_or_host_name, rpc_port);  // ① 连元数据 + 注册自己
Transport* installTransport(proto, args);        // ② 装传输（rdma / local / tcp / nvmeof ...）
int registerLocalMemory(addr, length, location, remote_accessible); // ③ 注册本地 GPU/CPU 内存为 segment
SegmentHandle openSegment(segment_name);         // ④ 发现远端节点/segment（拿 rkey，握手）
BatchID allocateBatchID(batch_size);             // ⑤ 开一个传输 batch
Status submitTransfer(batch_id, {TransferRequest}); // ⑥ 批量提交 READ/WRITE（异步）
Status getBatchTransferStatus(batch_id, status); // ⑦ 查询 batch 完成状态
int getNotifies(...) / sendNotify(...);          // ⑧ 通知（事件）
```

**一次传输的组织**（`Transport::TransferRequest`，transport.h）：

```cpp
struct TransferRequest {
    enum OpCode { READ, WRITE };
    OpCode opcode;          // READ = 从远端拉；WRITE = 推给远端
    void* source;           // 本地 buffer（已注册）
    SegmentID target_id;    // 远端 segment 句柄（openSegment 拿的）
    uint64_t target_offset; // 远端偏移
    size_t length;
    int advise_retry_cnt;   // 建议重试次数
};
```

- **批量（batch）**：一次 `submitTransfer` 提交一组 TransferRequest，引擎内部拆 slice、多网卡并行、异步完成（BatchID → BatchDesc 直接指针转换，热路径零查找，见 transport.h 注释）；
- **状态机**：WAITING → PENDING → COMPLETED/FAILED/TIMEOUT（`TransferStatusEnum`），带 transferred_bytes；
- **零拷贝**：READ/WRITE 的 source/target 都是**已注册内存**（GPU 显存可经 GPUDirect RDMA），数据不落 CPU bounce buffer；
- **registerLocalMemory 的 GPU 支持**：`location` 标识（kWildcardLocation），GPU 内存经 cuda/hip 注册拿 rkey，远端可直接 RDMA 写显存。

### 3. 具体数值样例

- 读 200 MB KV：`allocateBatchID` → 拆 4 个 TransferRequest（各 50 MB，分别指向 4 台远端 segment）→ 一次 `submitTransfer` → 4 网卡并行 RDMA READ → `getBatchTransferStatus` 轮询/通知完成；
- 同机传输：local transport 直接 memcpy（不占网卡）；`isTcpOnly()` 时同机优先本地拷贝而非 TCP 回环；
- 失败：slice 级 retry（`advise_retry_cnt`），`markSuccess/markFailed` 原子记账（`__atomic_fetch_add`），上层拿 transferred_bytes 判断。

> 面试一句话总结：**TransferEngine = 每进程一个的传输库：init 连元数据注册本节点 → installTransport 装协议 → registerLocalMemory 把 GPU/CPU 内存登记成可远程读写的 segment → openSegment 发现远端拿 rkey → submitTransfer 批量提交 READ/WRITE（拆 slice 多网卡并行、异步、slice 级重试）——上层只认"注册 + 发现 + 批量传"三件事。**

---

## 8. Transport 抽象与多协议：一个 Slice union 装下所有硬件

### 1. 现有问题

RDMA、NVMe-oF、CXL、HCCL、昇腾直传……每种传输的底层参数完全不同，统一接口怎么表达？

### 2. 方法论（源码 `transport/transport.h` 的 Slice union）

**Transport 是抽象基类**（每协议一个子类：rdma_transport / tcp_transport / local_transport / nvmeof / cxl / hccl / ascend_direct / ubshmem / efa / hip / intranode_nvlink ...），核心是一个 **union 把各协议的"传输私有参数"装在一起**：

```cpp
struct Slice {            // 一次传输的最小单元（一个请求拆成多个 slice）
    void* source_addr; size_t length;
    TransferRequest::OpCode opcode;
    SegmentID target_id; std::string peer_nic_path;
    SliceStatus status; ... 
    union {
        struct { uint64_t dest_addr; uint32_t source_lkey;
                 uint32_t dest_rkey; ... } rdma;      // RDMA：lkey/rkey/QP 深度
        struct { ... } ub;                            // UBSHMEM
        struct { void* dest_addr; } local;            // 同机：直接地址
        struct { uint64_t dest_addr; } tcp;           // TCP
        struct { uint64_t offset; int cufile_desc;
                 const char* file_path; } nvmeof;     // NVMe-oF / cuFile
        struct { void* dest_addr; } cxl;              // CXL
        struct { ... } hccl;                          // HCCL（昇腾集合通信）
        struct { ... } ascend_direct;                 // 昇腾 direct transport（ADXL）
        struct { uint64_t dest_addr; } ubshmem;       // UBSHMEM
    };
};
```

- **多传输选择逻辑**：同机 → local/NVLink（intranode_nvlink / cxl）；跨机 → 优先 RDMA（RoCE/IB，`rdma_transport`，自动多 NIC）；网络不可用 → TCP 兜底；昇腾环境 → hccl / ascend_direct；还有 EFA（AWS）等；
- **GPUDirect RDMA**：`rdma` 分支的 lkey/rkey 针对显存注册，网卡直接 DMA 到 GPU VRAM——零拷贝核心；
- **multi_transport.cpp**：多个 Transport 并存时按"目标可达性 + 拓扑"路由（`topology.cpp` 感知 NUMA/NIC 位置，优先同 NUMA 的网卡）。

### 3. 具体数值样例

- 同机双卡传输：选 `intranode_nvlink` 或 `local`（memcpy），不占 RDMA 网卡带宽；
- 跨机：`rdma_transport` 自动在 4 张 RoCE 网卡间条带化（slice 轮流分到各 NIC 的 QP），单对象吞吐接近 4×200 Gbps 聚合；
- 昇腾训练机（本项目实习环境）：TQ MoonCake 用 RDMA（RoCE）路径——`hccl`/`ascend_direct` 分支在 NPU 直传场景启用，参数对上层透明（一个 TransferRequest 统一表达）。

> 面试一句话总结：**Transport 抽象用一个 Slice union 把 rdma（lkey/rkey）/local/tcp/nvmeof/cxl/hccl/ascend_direct/ubshmem 等各协议的私有参数装在一起，多传输按拓扑/可达性路由（同机 NVLink/local、跨机 RDMA 多网卡聚合、TCP 兜底、昇腾走 hccl/ADXL）——上层永远只写一个统一的 TransferRequest。**

---

## 9. 元数据服务与节点发现：openSegment 的握手

### 1. 现有问题

TransferEngine 怎么知道"远端有哪些节点、每个节点注册了哪些内存段、rkey 是多少"？——靠独立的元数据服务。

### 2. 方法论

- **元数据后端可插拔**：etcd / Redis / HTTP（`init` 的 `metadata_conn_string` 决定）；部署文档里 Master 可选开 `--enable_http_metadata_server`（8080）直接当 TE 的 metadata；
- **节点注册**：每个 TransferEngine `init` 时把自己（`local_server_name` + RPC 端口）写进元数据；
- **`openSegment(segment_name)` 流程**：查元数据找到目标节点 → 与该节点的 TransportEngine RPC 握手（交换内存注册信息：地址、长度、**rkey**）→ 返回 `SegmentHandle`（含远端 rkey），之后 `submitTransfer` 直接用 rkey 发 RDMA，不再查表；
- **`syncSegmentCache`**：本地缓存远端 segment 信息，定期/手动刷新（节点增删后同步）。

### 3. 具体数值样例

- 4 节点集群 + 1 个 etcd：每节点 init 注册；vLLM decode 节点 `openSegment("prefill-node-0")` 一次握手拿 rkey（~ms），之后上千次 KV 传输零额外 RPC；
- Master 的 HA 也复用 etcd（leader 选举 + 心跳）——**etcd 同时服务 TE 元数据与 Store HA**。

> 面试一句话总结：**TE 的节点目录靠独立元数据服务（etcd/Redis/HTTP）：init 注册自己、openSegment 向目标节点握手交换 rkey 一次，之后所有 RDMA 传输直接用 rkey、不再查表——元数据只在"发现"阶段出现，不在传输热路径。**

---

# 六、整体架构串讲

## 10. 一次 KV 传输的全链路：Store 的 Client + Master + Transfer Engine 怎么协作

### 1. 现有问题

把三部分串成一条真实链路（以本项目最关心的"跨节点读一段 KV / 一条训练轨迹"为例）。

### 2. 方法论（全链路 7 步）

```text
① 应用（vLLM / TQ storage worker）需要远端对象：调 Client.Get(key)
② Client 内 MasterClient.GetReplicaList(key) → Master 返回 replica 位置（segment 列表）
③ BatchQueryIp → 拿目标节点 IP/RPC 地址
④ TransferEngine.openSegment(target)（首次：握手换 rkey；之后复用）
⑤ allocateBatchID + submitTransfer([READ × N slices]) → 引擎拆 slice、多网卡 RDMA 并行
   （数据从目标 Client 的 global segment 零拷贝到本地预注册 slice）
⑥ getBatchTransferStatus / 通知 → 完成 → Get 返回应用
⑦ 缓存：本地 TransferEngine 的 segment 缓存让后续同目标传输免握手
```

**与纯 Store 语义对照**：这就是一个分布式 KV Get；上层（vLLM 的 KV transfer、TQ 的 MoonCake backend、verl）看到的只是 `Put/Get` 对象——传输细节全被封装。

### 3. 具体数值样例

- 200 MB 对象、4×200 Gbps RoCE：`GetReplicaList`(~1ms) + 握手(首次 ~2ms) + 4 路并行 RDMA(~0.4s at 50% 线速) ≈ 0.41s；命中缓存的后续对象 ≈ 只花传输时间；
- 对比 TCP 单路：~4× 时间 + CPU 拷贝开销——这就是"RDMA + 多网卡 + 零拷贝"的全部意义（对应实习"RDMA 高速传输"）。

> 面试一句话总结：**一次 KV 传输 = Client 问 Master 位置 → 握手拿 rkey → TransferEngine 多网卡 RDMA 直传目标 segment → 完成通知；Master 只在开头当"导航"，数据面全程 Client↔Client——上层只认 Put/Get，这也是 TQ/vLLM 集成 MoonCake 时只看到 Client API 的原因（详见 `TQ.md` 的 MoonCake backend）。**

---

# 七、速查表

## 组件速查

| 组件（用户口径） | 官方名 | 职责 | 关键文件（v0.3.10.post2） |
|---|---|---|---|
| controller/控制面 | **Master Service**（mooncake_master） | 元数据、slice 分配（PutStart）、驱逐、HA leader | `mooncake-store/src/master.cpp`、`include/master_client.h` |
| store 数据节点 | **Client 的 store 角色 / mooncake_store_service** | 贡献 global segment、接收/提供对象 | `mooncake-store/src/real_client.cpp`、`client_service.cpp` |
| client 请求方 | **Client / dummy-real** | Put/Get/Remove 对象、数据直传 | `mooncake-store/include/client_service.h` |
| 传输引擎 | **Transfer Engine**（lib） | 注册内存、发现远端、批量 READ/WRITE | `mooncake-transfer-engine/src/transfer_engine.cpp`、`include/transport/transport.h` |
| 元数据 | etcd / Redis / HTTP | TE 节点目录 + Master HA 选举 | `mooncake-common/src/etcd/` |
| 存储分层 | DRAM global segment + SSD DFS | 热 KV 在 DRAM、冷数据 offload | `mooncake-store/src/{offset_allocator,file_storage,uring_file}.cpp` |

## 与外部系统关系（一句话定位，不展开）

| 集成 | 用的部分 | 详见 |
|---|---|---|
| TQ MoonCake 后端 | Store 的 Client（Put/Get 轨迹对象）+ TransferEngine RDMA | `TQ.md` |
| vLLM | TransferEngine（KV transfer 的读写）+ 可选 Store | `VLLM.md`、Mooncake 集成文档 |
| verl / 训练 | Client 存储轨迹/权重 | 实习 |

## 面试一句话总结（背诵版）

**"Mooncake（0.3.10）= 两个子系统：Store（分布式 KV 缓存）只有 Master（控制面：元数据 + slice 分配 + 驱逐，可 etcd HA）与 Client（双角色：请求方 + 存储节点，数据面 Client↔Client 直传绕过 Master）两个关键组件；传输靠 TransferEngine（注册内存 → openSegment 握手拿 rkey → submitTransfer 批量多网卡 RDMA，Slice union 支持 rdma/local/tcp/nvmeof/cxl/hccl/ascend）；对象 Put 后不可变保证强一致、slice 级复制 + 条带化、DRAM+SSD 分层——上层只看到 Put/Get，这是 TQ/vLLM 集成它的原因。"**
