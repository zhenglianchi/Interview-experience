# Mooncake 完全指南

> **Kimi（月之暗面）的 KV Cache 中心化分离式架构：Transfer Engine（传输引擎）× Mooncake Store（分布式 KV 存储）× EP/PG（弹性 MoE 执行），FAST'25 Best Paper。**

Mooncake 是 Moonshot AI（月之暗面）开源的大规模 LLM 推理/训练基础设施（`kvcache-ai/Mooncake`），官方定位"A KVCache-centric Disaggregated Architecture for LLM Serving"。核心思想：**以 KV Cache 为中心做分离式架构**——把 prefill 和 decode 集群分离，并利用 GPU 集群中原本闲置的 CPU / DRAM / SSD 资源构建**分离式 KV Cache 池**。真实负载下，Mooncake 让 Kimi 在满足 SLO 的前提下多处理 75% 的请求。

Mooncake 包含三大组件：**Transfer Engine（TE，传输引擎）**——高性能异构数据传输框架（核心）；**Mooncake Store**——基于 TE 的分布式 KV Cache / 模型权重存储；**Mooncake EP / PG**——弹性 MoE 执行与容错集合通信。本指南先逐个讲清这三个技术点，再讲整体架构设计与传输协议。

> 说明：本文基于 `kvcache-ai/Mooncake` 仓库（main 分支）讲解；本项目（简历）用 Mooncake 作为 TransferQueue 的存储后端（RDMA 高速传输训练轨迹），相关集成见 `TQ.md`。

---

# 一、核心组件

## 1. Transfer Engine（TE）：高性能异构数据传输引擎

### 1. 现有问题：为什么要传输引擎

Mooncake 要解决的根本问题是：**分离式架构（prefill 和 decode 分离、KV Cache 池化）要求数据在节点间高频、大容量、低延迟地传输——而传统 TCP 传输根本扛不住。** 具体瓶颈有三个：

1. **KV Cache 数据量巨大**：一个 LLaMA3-70B 模型 128k token 的 KV Cache 约 40 GB，PD 分离时 prefill 节点算完必须整包传给 decode 节点；用 TCP 传 40 GB 在 4×200 Gbps RoCE 上要很久，直接成为端到端延迟的瓶颈；
2. **异构网络/设备**：集群里同时有 RDMA（RoCE/IB）、TCP、NVLink、NVMe-oF、AWS EFA 等，还有 CUDA/MUSA/HIP/昇腾等多种加速器——需要统一接口屏蔽差异，否则每种组合写一套传输代码；
3. **带宽利用率低**：单条 RDMA 链路带宽有限，且传输路径不感知拓扑（可能走了绕路），失败时也没有自动切换——需要"多网卡带宽聚合 + 拓扑感知路由 + 自动故障转移"。

Transfer Engine 的定位就是"**统一的高性能数据传输框架**"：对上层提供统一的批量数据传输接口，内部支持多种传输协议、拓扑感知路由、多 NIC 带宽聚合、自动故障转移。

### 2. 方法论：Transfer Engine 是怎么实现的

Transfer Engine 的架构（`mooncake-transfer-engine/src/`）分四层，逐步操作：

**（1）统一接口层（`transfer_engine.cpp`）**：对上层暴露 `TransferEngine` 核心 API——`transfer_async`（异步批量传输）、`register_memory` / `unregister_memory`（注册可传输的内存段）、`get_numa_info` 等。数据以"段（segment）+ 元数据（transfer_metadata）"为单位，一次 `transfer_async` 把一批内存段从源搬到目的，支持批量（batched）传输。

**（2）多传输协议层（`transport/` 目录）**：每个协议一个子目录（`rdma_transport/`、`tcp_transport/`、`nvme_transport/`、`nvlink_transport/`、`efa_transport/`、`ascend_transport/`、`cxl_transport/` 等），实现统一的 `Transport` 接口（`multi_transport.cpp` 管理多协议调度）。**支持协议清单**（README 明确）：TCP、RDMA、AWS EFA、NVMe-oF、NVLink、HIP、Barex、CXL、Ascend 系。

**（3）拓扑感知路由层（`topology.cpp`）**：根据源/目的设备的 NUMA 亲和性、网卡归属、加速器位置选择最优传输路径——例如两个 GPU 在同一台机器走 NVLink，跨机器走 RDMA，避免绕路。**多 NIC 带宽聚合**：支持用多个 RDMA NIC 设备并行传输同一批数据，聚合带宽（4×200 Gbps 和 8×400 Gbps RoCE 下分别达 87 GB/s 和 190 GB/s）。

**（4）共享内存段层（`shared_segment/`）**：`mmap_shared_segment`（CPU 内存）、`ascend_shared_segment`（昇腾）等，管理可被远程直接访问的内存段——RDMA 需要注册内存段并暴露给对端。

**核心收益**（README 数据）：40 GB 数据（LLaMA3-70B 128k token 的 KV Cache 大小）在 4×200 Gbps 和 8×400 Gbps RoCE 网络上分别达到 **87 GB/s 和 190 GB/s**，比 TCP 快约 **2.4x 和 4.6x**。TE 已被 SGLang、vLLM、TensorRT-LLM、vLLM-Ascend、checkpoint-engine、NIXL 等广泛采用。

### 3. 具体数值样例

逐步演算一次 KV Cache 跨节点传输（PD 分离场景）：

```text
场景：prefill 节点算完一个请求的 KV Cache（40 GB），要传给 decode 节点。

方案 A（TCP）：40 GB / 单路 25 GB/s（200 Gbps TCP 实际）≈ 1.6s
方案 B（TE + RDMA 4×200Gbps）：40 GB / 87 GB/s ≈ 0.46s（快 2.4x）
方案 C（TE + RDMA 8×400Gbps）：40 GB / 190 GB/s ≈ 0.21s（快 4.6x）
```

**为什么 TE 能接近线性叠加带宽**：TE 把 40 GB 切分成多块，同时经 4 个（或 8 个）RDMA NIC 并行传输（多 NIC 带宽聚合），每块走拓扑最优路径（拓扑感知路由），任一 NIC 失败时自动切换备用路径（自动故障转移）——三个机制叠加，带宽从"单链路"变成"多链路聚合"。

> **面试一句话总结**：Transfer Engine 是 Mooncake 的核心——通过统一接口 + 多传输协议（TCP/RDMA/EFA/NVMe-oF/NVLink/CXL/昇腾等）+ 拓扑感知路由 + 多 NIC 带宽聚合 + 自动故障转移，实现异构环境下 KV Cache / 权重的高带宽传输（40 GB 数据在 8×400Gbps RoCE 下达 190 GB/s，比 TCP 快 4.6x）。

---

## 2. Mooncake Store：分布式 KV Cache / 权重存储引擎

### 1. 现有问题：为什么要分布式 KV 存储

分离式架构（PD 分离 + KV 池化）要求**KV Cache 能被跨节点共享和复用**，但传统的做法有三个问题：

1. **KV Cache 绑定推理引擎**：传统上 KV Cache 存在推理引擎的 GPU 显存里，引擎重启/升级/调度变化就丢失——无法在 prefill 和 decode 节点间流动；
2. **单机显存不够**：长上下文 + 高并发的 KV Cache 总量远超单机显存（128k token × 高并发 = 上百 GB），需要把 DRAM / SSD 也利用起来（多级缓存）；
3. **复用率低**：相同前缀的 KV 应该被多个请求共享（prefix caching），但没有一个统一的分布式存储来管理和复用。

Mooncake Store 的定位是"**基于 Transfer Engine 的分布式 KV Cache 存储引擎**"：把 KV Cache 和模型权重作为"对象"存储在分布式集群中，支持对象存储、复制、淘汰、高带宽传输，并且**与推理引擎解耦**（存储节点可动态增删，缓存数据不依赖引擎生命周期）。

### 2. 方法论：Mooncake Store 是怎么实现的

Mooncake Store 的架构（`mooncake-store/src/`），核心机制逐步操作：

**（1）对象模型**：KV Cache 和模型权重被抽象成"对象（object）"，每个对象有**放置策略（placement policy）**——副本数（replica counts）、首选段（preferred segments）、软钉（soft pin）/硬钉（hard pin）等。应用可以控制对象放哪、复制几份、是否钉住不被淘汰——保护重要 KV 和权重，指导复制/放置/淘汰行为。

**（2）分段存储（segment-based）**：存储节点把内存（DRAM / SSD/NVMe）划分成段，对象按段存放；**多级缓存层级**：DRAM + SSD/NVMe 两级，容量更大（把不常用的 KV 放到 SSD，常用留在 DRAM）。

**（3）大对象条带化（striping）**：大对象（如一个 40 GB 的 KV Cache）切成多个条带，并行 I/O + 端到端零拷贝传输，充分利用多 NIC 聚合带宽（复用 Transfer Engine）。

**（4）Master 与 Metadata 服务**：Store 集群有 master（管理节点注册/对象放置/复制）和 metadata server（对象元数据），客户端通过 `MooncakeDistributedStore` API 访问——这正好对应 TQ 的 `MooncakeStoreClient` 配置（`metadata_server` + `master_server_address`，见 `TQ.md`）。

**（5）与推理引擎集成**：Store 被 SGLang 的 Hierarchical KV Caching、vLLM 的 prefill serving、LMCache、TorchSpec、TransferQueue 等用作 KV Cache / hidden states / 权重的存储后端——**解耦推理/训练/RL 工作负载**（通过高效状态管理和异步数据移动）。

### 3. 具体数值样例

逐步演算"KV Cache 存入 Store → 另一节点读取复用"：

```text
场景：请求 A 的 prompt 包含 2000 token 的 system prompt（KV 已算好），
     请求 B 与 A 共享相同 system prompt。

第 1 步：prefill 节点算完 A 的 KV Cache → 作为对象写入 Mooncake Store
        （按放置策略选段，DRAM 优先；soft pin 保护这段高频复用的 system prompt）。
第 2 步：请求 B 到达 decode 节点 → 检查 Store 是否有相同前缀的 KV
        （prefix caching）→ 命中 → 经 TE 从 Store 拉取 2000 token 的 KV
        （假设 100 MB）→ decode 节点跳过这 2000 token 的 prefill，直接 decode。
第 3 步：某段时间 system prompt 不再被访问 → Store 按淘汰策略
        （LRU + pin 状态）把它的 KV 从 DRAM 降级到 SSD，腾出 DRAM 给热点。
第 4 步：存储节点故障 → master 检测 → 用副本（replica）恢复，
        缓存数据不因引擎重启而丢失（弹性/分离式存储）。
```

**收益**：跨请求的前缀 KV 复用省掉大量重复 prefill 算力；存储与引擎解耦让 KV 可以在引擎重启后继续使用；DRAM+SSD 多级缓存把 KV 容量从"单机显存"扩展到"整个集群的 DRAM+SSD"。

> **面试一句话总结**：Mooncake Store 是基于 Transfer Engine 的分布式 KV Cache/权重存储——把 KV 抽象成带放置策略（副本/pin/首选段）的对象，支持 DRAM+SSD 多级缓存、大对象条带化并行 I/O、端到端零拷贝传输，且存储与推理引擎解耦（节点弹性、数据不随引擎重启丢失）；被 SGLang/vLLM/TQ 用作 KV 存储后端，实现跨节点前缀复用。

---

## 3. Mooncake EP / PG：弹性 MoE 执行与容错集合通信

### 1. 现有问题：为什么要 EP 和 PG

大规模 MoE 推理（如 DeepSeek、Kimi 的 MoE 模型）面临两个工程难题：

1. **专家并行（EP）的容错**：MoE 的 expert 分布在多卡上，某张卡故障时，如果整个推理服务重启，代价巨大——需要"绕过故障 rank、用健康专家继续服务"的能力；
2. **集合通信的容错**：标准 `torch.distributed` 的集合通信（all_gather 等）在某个 rank 挂掉时整个通信失败——需要能检测故障、上报、恢复 rank 的 process group。

Mooncake EP / PG 的定位：**把 Mooncake 从"高性能数据移动"扩展到"容错分布式执行"**——EP 做专家并行的容错调度，PG 做容错的 PyTorch 集合通信后端。

### 2. 方法论：EP / PG 是怎么实现的

**Mooncake EP**（`mooncake-ep/`）：实现 DeepEP 风格的专家并行 dispatch/combine 操作，但加了 **`active_ranks` 感知**——dispatch 和 combine 时知道哪些 rank 是健康的，绕过故障 rank、只用健康专家继续服务。API 与 DeepEP 的 low-latency 模式基本一致，方便推理引擎直接采用而不重写 MoE 通信栈。

**Mooncake PG**（`mooncake-pg/`）：实现 PyTorch 分布式 **process group 后端**——注册为 `torch.distributed` 的 backend 后，标准集合 API（如 `all_gather`）底层走 Mooncake 通信，同时暴露**故障恢复原语**：peer-state polling（轮询对端状态）、rank recovery（替换进程重新加入现有 process group），让推理服务从部分故障中恢复而不整体重启。

**SGLang 集成**：Mooncake 的 collective backend + expert-parallel kernels 已集成进 SGLang，支持大 MoE 模型的容错专家并行推理（含弹性专家并行服务场景）。

### 3. 具体数值样例

```text
场景：MoE 模型 64 个 expert 分布在 8 卡（EP=8，每卡 8 expert），
     某卡（rank 3）故障。

Mooncake EP：dispatch 时发现 rank 3 不在 active_ranks 中 →
    绕过它，把原本要发给 rank 3 的 token 路由到健康 expert
    （如 rank 3 的 expert 有副本/或负载均衡到其他卡）→ 服务继续，
    只是少部分 expert 容量降级。

Mooncake PG：all_gather 时 rank 3 无响应 → 检测到故障 → 上报上层 →
    拉起替换进程 → peer-state polling 发现新 rank 3 就绪 → 重新加入
    process group → 集合通信恢复（无需重启整个服务）。
```

> **面试一句话总结**：Mooncake EP/PG 把 Mooncake 扩展到容错分布式执行——EP 用 active_ranks 感知让 MoE 专家并行绕过故障 rank 继续服务（DeepEP 兼容 API），PG 作为 torch.distributed 后端提供故障检测/上报/rank 恢复，支持弹性专家并行的大规模 MoE 推理。

---

# 二、整体架构：三大组件如何串起来

## 4. Mooncake 的整体架构与传输协议

### 1. 现有问题：三大组件如何组成一个 KV Cache 中心的分离式系统

前三个组件各自独立，但 Mooncake 的价值在于**它们组合成的整体架构**：以 KV Cache 为中心，把"推理"（prefill + decode）和"存储/传输"解耦，让整个集群的资源（GPU 显存 + CPU DRAM + SSD）都被高效利用。这一节把 Transfer Engine、Mooncake Store、EP/PG 串起来，讲清整体数据流和每一步的传输协议。

### 2. 方法论：整体架构是怎么组织的

Mooncake 的整体架构（README 官方定位 + 组件图）：

```text
┌────────────────────────── 推理集群 ──────────────────────────┐
│  ┌──────────────┐        ┌──────────────┐                    │
│  │ Prefill 集群  │──KV──▶│  Decode 集群  │  （PD 分离）        │
│  │（算力密集）    │  TE   │（访存密集）    │                    │
│  └──────────────┘        └──────────────┘                    │
│         │ KV 复用/换入换出           │ KV 换入                 │
│         ▼                           ▼                        │
│  ┌──────────────────────────────────────────┐                │
│  │        Mooncake Store（分离式 KV 池）       │                │
│  │  利用闲置 CPU DRAM + SSD/NVMe 构建多级缓存  │                │
│  └──────────────────────────────────────────┘                │
└───────────────────────────────────────────────────────────────┘
         ▲ 全部经 Transfer Engine（TE）传输
         │ 协议：RDMA(RoCE/IB) / TCP / NVLink / NVMe-oF / EFA ...
```

**完整数据流**（一个请求从 prefill 到 decode 的跨节点流动）：

```text
① prefill 节点：算完请求的 KV Cache（GPU 显存）
② 写 Store：KV 经 TE 从 GPU → Store 的 DRAM/SSD
   传输协议：本机走 NVLink/NVMe，跨机走 RDMA（RoCE/IB）或 TCP
③ decode 节点拉取：经 TE 从 Store 拉 KV → 本机 GPU 显存
   传输协议：跨机走 RDMA（多 NIC 聚合带宽），同机走 NVLink
④ 前缀复用：新请求命中相同前缀 → 直接从 Store 读已有 KV（跳过 prefill）
⑤ MoE 场景：EP/PG 做专家并行调度 + 故障容错
```

**传输协议矩阵**（面试重点：每一步用什么协议）：

| 传输场景 | 传输协议 | 说明 |
|---|---|---|
| GPU ↔ GPU（同机） | **NVLink** | 最高带宽，TE 检测同机走 NVLink |
| 节点间 KV 传输 | **RDMA（RoCE / InfiniBand）** | 主力协议，多 NIC 聚合带宽（4×200Gbps→87GB/s、8×400Gbps→190GB/s） |
| 节点间（无 RDMA） | **TCP** | 兜底，TE 自动降级 |
| GPU ↔ SSD/NVMe（Store 多级缓存） | **NVMe-oF / NVLink** | KV 换入换出 Store 的 SSD 层 |
| 云环境（无 RDMA） | **AWS EFA** | EFA 协议（RDMA 的云化替代） |
| 昇腾 NPU 环境 | **Ascend 系传输** | 昇腾加速器专用 |
| 跨设备内存拷贝 | **CXL / HIP / Barex** | 新硬件支持 |

**TE 的协议选择逻辑**（`multi_transport.cpp` + `topology.cpp`）：不是固定用某个协议，而是**按"源/目的设备位置 + 可用协议 + NUMA 亲和性"动态选最优路径**——同机 GPU 直连走 NVLink、跨机有 RDMA 走 RDMA（多 NIC 聚合）、没有 RDMA 走 TCP、传输失败自动切换备用路径。

### 3. 具体数值样例

以 Kimi 的生产部署为例（FAST'25 论文 + README），逐步演算整体架构如何工作：

```text
场景：PD 分离 + 大规模 EP，128 H200 GPU（Kimi K2 部署）。

第 1 步：1000 个并发请求到达 → prefill 集群计算（224k tokens/sec prefill 吞吐）。
第 2 步：每个请求的 KV Cache 经 TE 写/读到 Mooncake Store：
        - 长上下文 KV（如 128k token ≈ 40 GB）经 RDMA 多 NIC 聚合传输
          （8×400Gbps 下 190 GB/s，40 GB 约 0.21s）
        - 相同前缀的请求复用 Store 里的 KV（跳过重复 prefill）
第 3 步：decode 集群从 Store 拉取 KV 继续生成（288k tokens/sec decode 吞吐）。
第 4 步：MoE 的 expert 经 EP 调度，某卡故障时 EP/PG 容错继续服务。
```

**真实收益**（README）：Kimi 在满足 SLO 的前提下多处理 **75%** 的请求；K2 部署实现 224k tokens/sec prefill + 288k tokens/sec decode；checkpoint-engine（P2P Store 官方版）用 TE 在数千 GPU 上 20 秒更新 Kimi-K2（1T 参数）权重。

**与简历/TQ 的关系**：本项目用 Mooncake 做 **TQ（TransferQueue）的存储后端**——训练轨迹经 TQ 的 `MooncakeStoreClient` 写入 Mooncake Store（RDMA 高速传输），实现双机全异步 RL 训练中 node1（trainer）与 node2（rollout）之间轨迹的跨机传输（详见 `TQ.md` 的 Mooncake backend 章节）。

> **面试一句话总结**：Mooncake 的整体架构 = PD 分离推理集群 + 分离式 KV Cache 池（Mooncake Store，利用闲置 CPU DRAM/SSD）+ 统一传输层（Transfer Engine，RDMA/NVLink/TCP/EFA/昇腾多协议、拓扑感知路由、多 NIC 聚合）+ 容错执行（EP/PG）；传输协议按"源/目的位置 + 可用协议"动态选择——同机 NVLink、跨机 RDMA（聚合带宽）、无 RDMA 降级 TCP，让 Kimi 在 SLO 内多处理 75% 请求。

---

## 附：组件速查表

| 组件 | 代码位置 | 角色 | 关键机制 |
|---|---|---|---|
| Transfer Engine | `mooncake-transfer-engine/src/` | 统一异构数据传输 | `transfer_engine.cpp`、`multi_transport.cpp`、`topology.cpp`、`shared_segment/`；多协议 + 拓扑路由 + 多 NIC 聚合 + 故障转移 |
| Mooncake Store | `mooncake-store/src/` | 分布式 KV/权重存储 | 对象模型（副本/pin/首选段）、DRAM+SSD 多级缓存、大对象条带化、master + metadata server |
| Mooncake EP | `mooncake-ep/` | 弹性专家并行 | DeepEP 兼容 dispatch/combine + active_ranks 感知 |
| Mooncake PG | `mooncake-pg/` | 容错集合通信 | torch.distributed backend、peer-state polling、rank recovery |
| 传输协议 | `transport/` 各子目录 | 底层传输 | RDMA（RoCE/IB）/ TCP / NVLink / NVMe-oF / EFA / CXL / Ascend |

> **关联文档**：Mooncake 作为 **TransferQueue 的存储后端**（`MooncakeStoreClient`，RDMA 传输训练轨迹）的集成细节见 `TQ.md`；vLLM 的 PD 分离通过 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/` 使用 Mooncake，见 `VLLM.md` 第 16 点。
