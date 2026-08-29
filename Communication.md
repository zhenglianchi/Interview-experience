# Communication 完全指南：GPU/NPU 通信协议与互联技术

> 对应简历"推理加速：了解 Mooncake 分布式 KV Cache 存储传输架构"与实习"昇腾 NPU 环境单/双机全链路、RDMA 高速传输"知识体系，以及 `Parallel.md` 里"通信原语跑在什么物理链路上"的底层答案。一句话知识框架：
> **AI 集群通信分两层：机内（scale-up）用 NVLink/HCCS 这类专用高速总线做全互联，机间（scale-out）用 InfiniBand/RoCE 跑 RDMA；集合通信库（NCCL/HCCL）负责把算法（ring/tree）映射到物理拓扑上**——面试考的就是"每条链路是什么协议、多少带宽、为什么这么分层"。
>
> 素材来源：web 检索交叉验证（NVIDIA/华为官方文档、Meta SIGCOMM 2024、DCQCN SIGCOMM 2015 等，文末附链接）+ 既有知识；不确定的数字标"约"。配套阅读：`Parallel.md`（并行维度）、`Ray.md`（Ray 传输栈）、`Mooncake.md`（KV 传输）。

---

# 一、机内互联（Scale-Up）

## 1. NVLink：GPU 专用机内高速互联

### 1. 现有问题

- 多卡并行（TP/PP）的每次前向都伴随 all-reduce/all-gather：**8 卡间的通信量级是 GB/s 到 TB/s**，PCIe（32~64 GB/s）根本喂不饱；
- 显存之间互拷（P2P 读对端显存）需要高带宽 + 低延迟 + 缓存一致性。

### 2. 方法论

**NVLink 是 NVIDIA 自研的 GPU↔GPU（及 GPU↔CPU）点对点高速互联**：专用 SerDes 物理链路 + 自有链路层协议（**不跑 PCIe 包格式**），带缓存一致性/原子能力，可 P2P 直接读写对端显存。带宽 = 链路数 × 每链路带宽（双向/每卡）：

| 版本 | 代表卡 | 链路 × 每链路 | 双向带宽/卡 |
|---|---|---|---|
| NVLink 1.0 | P100 | 4 × 20 GB/s | 160 GB/s |
| NVLink 2.0 | V100 | 6 × 25 GB/s | 300 GB/s |
| NVLink 3.0 | A100 | 12 × 25 GB/s | 600 GB/s |
| NVLink 4.0 | H100 | 18 × 25 GB/s | **900 GB/s** |
| NVLink 5.0 | B200/GB200 | 18 × 50 GB/s | **1800 GB/s（1.8 TB/s）** |

**NVSwitch**：NVLink 交换机，把 N 卡端口汇聚成**无阻塞全互联**——DGX H100 用 8 个 NVSwitch 提供 8×900 = 7.2 TB/s 机内带宽；GB200 NVL72 用 36 个 NVLink Switch 把 72 GPU 组成单 NVLink 域（约 130 TB/s 官方总带宽）。

**NVLink-C2C**：芯片级一致性互联（GH200 的 Grace–Hopper），物理复用 PCIe 5.0 SerDes（32 GT/s）但跑 **C2C 一致性协议**：CPU/GPU 统一地址空间、硬件缓存一致性——GPU 可访问 CPU 的 LPDDR5X（约 546 GB/s），带宽 900 GB/s；B200 双 die 也经 C2C 互联。

**易错点**：**没有"NVLink over 以太网/fabric"**——NVLink 是板级/机架级互联，跨机架走 IB/RoCE。

### 3. 具体数值样例

- TP=8 的 8B 模型：每层 MLP all-reduce ≈ 2×hidden×batch×2B ≈ 8 MB/次（hidden=4096、batch 256），H100 上 NVLink 4.0（900 GB/s 单向 450 GB/s）耗时 ≈ 18 µs，PCIe 5.0（64 GB/s）≈ 125 µs——**7 倍差距**；
- 对比机间 400G 网卡（50 GB/s）：差 9~18 倍——这就是"机内管快、机间管大"的物理根源。

> 面试一句话总结：**NVLink 用专用 SerDes + NVSwitch 全互联把机内带宽做到 900 GB/s（H100）~1.8 TB/s（B200），NVLink-C2C 用 PCIe 物理层跑一致性协议实现 CPU-GPU 统一内存——只做机内/机架内，跨机架必须走 IB/RoCE。**

---

## 2. PCIe：通用兜底总线

### 1. 现有问题

GPU 与 CPU、网卡、SSD 的连接基础；也是 GPU 与外部世界的"保底通道"（NVLink 只连 GPU 之间）。

### 2. 方法论

**PCIe 是通用主板总线**：串行差分 lane，带宽 = lane 数 × 每 lane 速率 × 编码效率（4.0/5.0 用 128b/130b 编码，6.0 起引入 **PAM4**（每 symbol 2 bit）+ FEC 翻倍）：

| 版本 | 速率 | x16 单向带宽 |
|---|---|---|
| PCIe 4.0 | 16 GT/s | 约 32 GB/s（A100/H100 标配） |
| PCIe 5.0 | 32 GT/s | 约 64 GB/s |
| PCIe 6.0 | 64 GT/s（PAM4） | 约 128 GB/s（Blackwell 支持） |

- GPU 通常 x16 直连 CPU 或经 **PCIe Switch**（Broadcom PEX88000 系）扇出；
- **PCIe P2P**：同一 switch 下的设备间可直接 DMA（不经过 CPU），NCCL 的 PCIe P2P 路径依赖它；
- **NUMA 亲和**：GPU 与 NIC 是否同一 socket/switch 决定机间带宽能否打满（第 22 点）。

### 3. 具体数值样例

- 8 卡 DGX 服务器：每卡 x16 PCIe 4.0（32 GB/s）直连 CPU，卡间数据优先走 NVLink（900 GB/s）而非 PCIe；
- 网卡（400G ≈ 50 GB/s）若与 GPU 跨 NUMA，要走 UPI 互联（Intel 双路约 200 GB/s 总带宽）——共享带宽下机间吞吐直接减半，所以生产要求 **GPU-NIC 1:1 亲和**。

> 面试一句话总结：**PCIe 是通用兜底互联，5.0 x16 = 64 GB/s、6.0 PAM4 翻倍到 128 GB/s；GPU 与 NIC 是否同 PCIe switch/NUMA 侧决定机间带宽能否打满，PCIe P2P 让同 switch 设备间直接 DMA。**

---

## 3. CXL：内存池化的开放标准

### 1. 现有问题

- 显存/内存是"私有的"：一台机器用不完、另一台不够用；大模型训练想要**内存池化/容量扩展**；
- 但池化需要"一致性"（多设备共享同一内存视图）——PCIe 本身只提供 IO 语义。

### 2. 方法论

**CXL 是基于 PCIe 物理层（Alt Mode）的开放标准**（1.1/2.0/3.0），把内存变成可共享/可池化资源。三种语义：
- **CXL.io**：PCIe 包，IO/DMA（兼容现有设备）；
- **CXL.cache**：设备访问主机内存的缓存一致性；
- **CXL.mem**：主机访问设备内存。

设备类型：Type 1（纯 cache 加速器）、Type 2（带内存加速器，**GPU 属于此类**）、Type 3（纯内存扩展器）。

带宽：1.1/2.0 基于 PCIe 5.0 PHY，x16 每方向约 32 GB/s；3.0 基于 PCIe 6.0 PHY（PAM4），每方向约 64 GB/s（双向约 128 GB/s）。

**对 GPU 的意义**：CXL 2.0 引入 switch + 内存池化，3.0 引入 fabric 化 + 多主机共享；但**带宽比 NVLink 低一个量级、时延更高**——定位是容量扩展/池化，不是替代 NVLink 的性能互联。

### 3. 具体数值样例

- 一台 8 卡服务器插 4 个 CXL 内存扩展器（Type 3，各 256 GB）：显存/内存池从"卡内私有"变成"可弹性分配"；
- 对比：CXL 3.0 单方向 64 GB/s vs NVLink 4.0 900 GB/s——**差 14 倍**，所以 CXL 管容量、NVLink 管性能；
- 现状：Samsung/Micron CXL 内存模块已出货但生态落地慢，GPU 显存池化仍是远期方案。

> 面试一句话总结：**CXL 用 PCIe 物理层 + io/cache/mem 三协议做一致性内存共享，3.0 每方向 64 GB/s；对 GPU 是显存池化的远期方案（Type 2 加速器），近期定位是容量扩展而非性能互联——带宽比 NVLink 低一个量级。**

---

## 4. UVM / 统一内存

### 1. 现有问题

- CPU/GPU 各有独立内存，数据搬运要手动 cudaMemcpy，容易错、容易慢；
- 希望"CPU 侧 malloc 的内存 GPU 也能直接访问"。

### 2. 方法论

**UVM（Unified Virtual Memory）** = CUDA Managed Memory：CPU/GPU 共享同一**虚拟**地址空间，按页管理数据位置。底层原理：Pascal 起硬件**页错误（page fault）** + 按需迁移（经 PCIe 或 NVLink），配合 `cudaMemPrefetchAsync` 预取。

关键区分：
- UVM 是**软件/API 层概念**（虚拟内存统一），NVLink/C2C 是它的**高速硬件迁移通道**（900 GB/s vs PCIe 32~64 GB/s）；
- GH200 上靠 NVLink-C2C 缓存一致性实现"免拷贝"（页面可驻留任意侧，硬件维护一致），但**物理上 HBM 与 LPDDR5X 仍是两个池**——"统一虚拟内存"≠"统一物理内存"。

### 3. 具体数值样例

- 数据集 40 GB > HBM 80 GB 的一半：`cudaMallocManaged` 分配，GPU 按需取页、CPU 侧数据自动迁移，省去手动分块拷贝；
- 迁移速度：页迁移走 NVLink（900 GB/s）比 PCIe（32 GB/s）快 28 倍——UVM 在 PCIe 机器上慢、在 NVLink/C2C 机器上实用；
- 面试点：**"统一内存"是虚拟的**——GH200 的 C2C 让 CPU/GPU 免拷贝，但物理内存未合一。

> 面试一句话总结：**UVM 是按需分页的统一虚拟内存（软件机制），NVLink/C2C 是它的高速迁移/一致性通道；GH200 靠 NVLink-C2C 实现 CPU/GPU 免拷贝访问，但物理上 HBM 与系统内存仍是两个池。**

---

# 二、机间互联（Scale-Out）

## 5. InfiniBand：为 RDMA 而生的专用网络

### 1. 现有问题

- 跨节点集合通信需要：低延迟（微秒级）、高带宽、**内核旁路**（不让 CPU 参与每包拷贝）、可靠传输；
- TCP/IP 协议栈太慢（内核中断 + 拷贝 + 拥塞控制），以太网会丢包（RDMA 的 RC 重传一丢就崩）。

### 2. 方法论

**IB 是一套独立网络体系**（物理/链路/网络/传输层全自研，不是以太网）：

- **HCA**（主机侧网卡）：内核旁路（verbs 直接用户态访问）+ DMA，应用只碰 QP/CQ；
- **子网管理器 SM**：集中式管理——分配 16 位 LID、计算路径表（与以太网的"分布式学习"是最大差异）；
- **VL 虚拟通道**：QoS + 硬件信用流控（逐跳不丢包，**天然无损**）；
- 跨子网用 GRH 路由。

速率演进（4X 端口）：SDR 10G → DDR 20 → QDR 40 → FDR 56 → EDR 100 → HDR 200 → **NDR 400** → **XDR 800 Gbps**（2024 GTC，Quantum-X800 平台）。

### 3. 具体数值样例

- NDR 400G 端口 ≈ 50 GB/s 单向：1 GB 参数 all-reduce（TP=8 跨 2 机）≈ 20 ms 量级；
- IB 交换机（Quantum-X800）576 端口、支持 **SHARP 网内计算**（归约卸载到交换机）；
- 代价：IB 硬件贵、生态割裂（NVIDIA 私有）→ 催生 RoCE。

> 面试一句话总结：**IB 是"硬件无损 + 集中式 SM + 内核旁路"的专用 RDMA 网络，速率到 NDR 400G/XDR 800G，天然适合 AI 集群；缺点贵、生态封闭——RoCE 用"普通以太网 + 无损改造"提供了兼容替代。**

---

## 6. RoCE：以太网上的 RDMA（重点，昇腾/华为生态标配）

### 1. 现有问题

- IB 太贵且被 NVIDIA 垄断；数据中心已有海量以太网基础设施，希望"在以太网上跑 RDMA"；
- 但**以太网会丢包**：RDMA 的 RC 传输一旦丢包，网卡重传 + 保序导致吞吐雪崩（"一丢全崩"）——必须把以太网变成"无损网络"。

### 2. 方法论

**RoCE = RDMA over Converged Ethernet**：
- **RoCEv1**：ethertype 0x8915，仅 L2 同子网不可路由；
- **RoCEv2**（实际全是 v2）：封装 **UDP/IP**（端口 4791），可路由，用 GID 标识端点。

**无损网络三件套**（面试必答）：

```text
① PFC（802.1Qbb 优先级流控）：按 8 个优先级逐跳暂停
   → 保证零丢包，副作用：队头阻塞、死锁、PFC 风暴（全局暂停蔓延）
② ECN（显式拥塞通知）：交换机队列超阈值 → 报文打 CE 标记
   → 接收端感知拥塞（端到端信号，不暂停）
③ DCQCN（SIGCOMM 2015）：接收端收 CE → 发 CNP 控制包回报
   → 发送端量化降速（先降 50% 再逐渐恢复）
   = RoCE 拥塞控制的事实标准
```

现代增强：**sPFC**（选择性 PFC，减少全局暂停）、**多路径包喷洒**（Meta 用，避免 ECMP 哈希碰撞）、INT 遥测、AI 拥塞控制（Spectrum-X 动态路由）。

带宽：跟随以太网 100G/200G/400G/800G；时延约 1~3 µs 级。

### 3. 具体数值样例

- 400G RoCE 端口 ≈ 50 GB/s 单向，与 IB NDR 400G 同级——**RoCE 与 IB 的差距不在带宽而在拥塞控制成熟度**；
- PFC 风暴场景：一条链路拥塞 → 8 优先级里 1 个暂停 → 反压传播到整棵树上游 → 全局吞吐塌陷；ECN/DCQCN 提前减速避免触发 PFC；
- Meta（SIGCOMM 2024）：24k GPU 生产集群从 IB 迁到自研 RoCE（2 层 CLOS + 400G + 包喷洒 + sPFC）——**证明 RoCE 在万卡规模可替代 IB**。

> 面试一句话总结：**RoCEv2 = UDP 4791 上的 RDMA，可路由；性能靠"无损网络"：PFC 兜底防丢包（但会队头阻塞）、ECN 端到端标记拥塞、DCQCN 接收端回报+发送端降速——无损 ≠ 不拥塞，难点全在拥塞控制；Meta 24k GPU 集群证明 RoCE 可替代 IB。**

---

## 7. GPUDirect RDMA / GDS：绕开 CPU 的直通 IO

### 1. 现有问题

- 传统网卡收数据要先进 CPU 内存（pinned buffer）再拷到显存——多一次拷贝、多一次 CPU 参与；
- 存储（SSD）读数据进训练也要经过 CPU bounce buffer。

### 2. 方法论

- **GPUDirect RDMA（GDR）**：NIC 通过 GPU BAR1 窗口**直接 DMA 写显存**（P100+），绕 CPU/pinned buffer——省 CPU、降时延，是 **NCCL 机间打满带宽的前提**（`NCCL_NET_GDR_LEVEL` 控制）；
- **GPUDirect Storage（GDS）**：存储（NVMe/文件系统）→ GPU 显存直通，`cuFile`（libcufile）API，免 bounce buffer——数据加载/checkpoint 提速数倍。

### 3. 具体数值样例

- 无 GDR：400G 网卡收包 → CPU memcpy 到 pinned buffer（占 CPU + 内存带宽）→ cudaMemcpy 到显存，端到端 CPU 占用高、延迟多一跳；
- 有 GDR：网卡 DMA 直达显存，CPU 只做控制；`NCCL_NET_GDR_LEVEL=PIX` 表示 PIX 模式（网卡与 GPU 同 PCIe switch）启用；
- GDS：读 100 GB checkpoint 从 NVMe 直达显存，相比传统路径省一次内存拷贝，加载时间可减半。

> 面试一句话总结：**GDR 让网卡→显存一跳直达（DMA 到 GPU BAR1），GDS 让盘→显存直通（cuFile）——都是绕 CPU 拷贝的 IO 加速；GDR 是 NCCL 机间打满带宽的前提，配置位是 NCCL_NET_GDR_LEVEL。**

---

## 8. NVLink 机架级域（Domain）：从机内到机架

### 1. 现有问题

单机 8 卡全互联后，跨机通信掉到 IB/RoCE（差一个量级）；能不能把"机内高带宽"扩展到更大规模？

### 2. 方法论

**NVLink 域（domain）**：NVLink Switch 把机内域扩展到机架级：
- **DGX H100 SuperPOD**：32 节点 256 GPU 组成 NVLink 域；
- **GB200 NVL72**：72 GPU + 36 NVLink Switch，单机架 NVLink 域（官方约 130 TB/s），机架间再用 Quantum-X800 IB 或 Spectrum-X 800G RoCE。

面试正解表述：是 **"NVLink domain"**（域内全互联），不是"NVLink over fabric"——NVLink 物理上仍是板级/机架内互联，域间必须走 IB/RoCE。

### 3. 具体数值样例

- NVL72 内 72 卡两两通信走 NVLink（130 TB/s 域总带宽），域间 8 个 Quantum-X800（800G）上联；
- 大模型（如 1.8T MoE）的 TP 组放域内（NVLink 全互联）、EP 组跨域（IB/RoCE）——**并行维度与物理拓扑匹配**是集群设计核心。

> 面试一句话总结：**NVLink 域（domain）是机内全互联的上位概念：DGX SuperPOD 256 GPU、NVL72 的 72 GPU 都组成单域（约 130 TB/s），域间才走 IB/RoCE——TP 放域内、EP 跨域，并行维度贴物理拓扑。**

---

# 三、集合通信库

## 9. NCCL：NVIDIA 集合通信库

### 1. 现有问题

- 多卡训练的 allreduce 等集合操作，如果每框架自己实现，通信效率天差地别；
- 需要一个"拓扑感知 + 多算法 + 多传输"的通用库，让框架（PyTorch DDP/FSDP、Megatron、DeepSpeed）无感拿到接近线速的性能。

### 2. 方法论

**NCCL（NVIDIA Collective Communications Library）** 是 GPU 集合通信库（AllReduce/Broadcast/AllGather/ReduceScatter），PyTorch DDP/FSDP、Megatron、DeepSpeed 底层使用：

**算法**：
- **Ring**（默认）：带宽最优 O(N) 步、适合大消息——每卡只与前后邻居通信，数据绕环；
- **Tree**：深度 O(log N)、时延最优、适合小消息/跨机——树状归约；
- **NVLS**（NCCL 2.17+）：**归约卸载到 NVSwitch 网内计算**（SHARP 类似思想）；
- 每个操作拆到多个 channel 并行，`NCCL_ALGO` 可强制。

**传输选择（拓扑感知）**：启动时用 NVML/PXI 探测，优先级：
```text
机内 NVLink P2P → PCIe P2P → 共享内存 → 机间 IB/RoCE（GPUDirect RDMA）→ TCP socket 兜底
```

**ncclComm**：通信域对象，每进程每设备一个；`ncclCommInitRank/InitAll` 创建；框架负责 rank 间协调建连。

**常用环境变量**（面试要能说出几个）：`NCCL_IB_DISABLE`、`NCCL_P2P_DISABLE`、`NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`、`NCCL_IB_TC/NCCL_IB_SL`（映射 PFC 优先级）、`NCCL_IB_TIMEOUT/RETRY_CNT`、`NCCL_IB_QPS_PER_CONNECTION`、`NCCL_BUFFSIZE`、`NCCL_NET_GDR_LEVEL`、`NCCL_ALGO`、`NCCL_PROTO`（LL/LL128/Simple）、`NCCL_DEBUG`、`NCCL_TOPO_DUMP_FILE`。

### 3. 具体数值样例

- 8 卡 H100 全互联 + 2 节点：机内 allreduce 走 NVLink ring（8 卡带宽 900 GB/s），机间走 IB/RoCE（400G ≈ 50 GB/s）——**NCCL 自动分层**（先机内归约再机间传一次，即 hierarchical allreduce）；
- 排障：`NCCL_DEBUG=INFO` 看传输选择，`NCCL_TOPO_DUMP_FILE` 看拓扑图，`NCCL_IB_DISABLE=1` 强制走 socket 对比；
- 大消息（≥几 MB）ring 最优，小消息（< 几 KB）tree/低延迟路径更优。

> 面试一句话总结：**NCCL 把"拓扑感知 + ring/tree/NVLS 多算法 + NVLink/PCIe/IB/RoCE 多传输自动选择 + GPUDirect RDMA"全做进库，让框架无感拿到接近线速的集合通信——机内 NVLink、机间 IB/RoCE、自动分层聚合；NCCL_* 环境变量是排障与调优位。**

---

## 10. GLOO / MPI / UCX / Horovod：其他集合通信栈

### 1. 现有问题

NCCL 只管 GPU；CPU 集合通信、通用 HPC、跨框架传输抽象还需要别的栈；面试常问"它们是什么关系"。

### 2. 方法论

- **GLOO**（Meta）：PyTorch 的 **CPU** 集合通信后端（TCP/共享内存 ring-allreduce），慢；GPU 训练默认 NCCL；
- **MPI**：消息传递标准（OpenMPI/MVAPICH2），通用 HPC；**CUDA-aware MPI**（MVAPICH2-GDR、OpenMPI+UCX）能传 GPU buffer，但优化不如 NCCL；**Horovod** 用 MPI 管 rank + NCCL 传数据；
- **UCX**：点对点通信框架/传输抽象层（IB verbs、RoCE、共享内存、TCP、cuda_ipc、gdr_copy），被 OpenMPI/UCX-Py/Dask 使用——自动选传输、零拷贝；**面试点：UCX 是传输中间层，NCCL 是集合层**；
- **Horovod**（Uber 2017）：基于百度 ring-allreduce，支持 NCCL/GLOO/MPI/CCL 后端，提出**层级 allreduce**（机内 + 机间）；DDP 成熟前的主流，现在被取代但思想仍是考点。

### 3. 具体数值样例

- PyTorch DDP：GPU 默认 `backend="nccl"`，CPU 训练用 `backend="gloo"`（分布式数据加载/评估）；
- Horovod 的层级 allreduce：机内 8 卡 NVLink ring → 机间 1 次 IB 传输 → 机内广播——只传一次跨机数据，通信量从 O(N²) 降到 O(N)；
- UCX 自动选传输：同节点用共享内存/cuda_ipc（零拷贝），跨节点用 verbs（IB/RoCE）——应用无需感知物理链路。

> 面试一句话总结：**GLOO 管 CPU、NCCL 管 GPU、MPI 是通用 HPC 标准、UCX 是传输抽象层（NCCL 是集合层）、Horovod 提出层级 allreduce（机内+机间各一次）——它们不是替代关系而是分层关系。**

---

# 四、华为昇腾 NPU 生态（重点，实习相关）

## 11. HCCL：昇腾的集合通信库（对标 NCCL）

### 1. 现有问题

昇腾 NPU 需要自己的集合通信栈（不能用 NCCL）；接口、算法、调优要"对齐业界"方便迁移。

### 2. 方法论

**HCCL 是 CANN 的集合通信库**，对标 NCCL，接口一一对应（`HcclAllReduce`、`HcclCommInitRank`、`HcclComm`），MindSpore/torch_npu 多卡训练底层：

- **算法/拓扑感知**：与 NCCL 同思路——**机内 HCCS 全互联 P2P 优先，机间 RoCE（RDMA）**；支持 ring 类算法与**分层聚合**（先机内 HCCS 聚合再机间 RoCE 传输）；`HCCL_ALGO` 可指定；`HCCL_DETERMINISTIC` 确定性模式；
- **环境变量**（命名镜像 NCCL）：`HCCL_CONNECT_TIMEOUT`、`HCCL_EXEC_TIMEOUT`、`HCCL_SOCKET_IFNAME`、`HCCL_RDMA_IFNAME`（RoCE 网卡）、`HCCL_RDMA_TC/SL/GID_INDEX`（RoCE 流控与 GID）、`HCCL_INTRA_ROCE_ENABLE`（允许机内也走 RoCE，HCCS 不可用/带宽不足时的补充）、`HCCL_ALGO`、`HCCL_DETERMINISTIC`、`HCCL_OP_BASE`。

### 3. 具体数值样例

- 8 卡 910B 单机：allreduce 走 HCCS（约 392 GB/s 量级）而非 RoCE（200G ≈ 25 GB/s）——**快 15 倍**，HCCL 自动选择；
- 双机 16 卡：每机先 HCCS 机内归约（8 卡 → 1 份），再 RoCE 机间传 1 份（分层聚合），跨机通信量降为 1/8；
- 对应实习"昇腾 NPU 单/双机全链路"：`HCCL_RDMA_IFNAME` 指定 RoCE 网卡名、`HCCL_RDMA_TC` 映射无损网络优先级——这就是昇腾上"RDMA 高速传输"的配置位。

> 面试一句话总结：**HCCL 是昇腾的 NCCL——接口对齐、算法同构（机内 HCCS 优先 + 机间 RoCE 分层聚合），HCCL_RDMA_* 系列环境变量镜像 NCCL_IB_* 语义；熟悉 NCCL 可快速迁移到昇腾。**

---

## 12. HCCS：昇腾的高速缓存一致性总线（对标 NVLink）

### 1. 现有问题

昇腾多卡机内互联需要"高带宽 + 缓存一致性 + P2P 显存直访"，不能靠 PCIe。

### 2. 方法论

**HCCS（Huawei Cache Coherence System）**：华为自研芯片间**缓存一致性互联总线**，对标 NVLink：
- **底层原理**：板级 SerDes 总线 + 硬件缓存一致性（类似 NVLink/CXL.cache），NPU 间 P2P 直接访问对端显存、原子操作；
- **带宽与拓扑**：Atlas 800T A2（910B）单机 8 卡，**HCCS 全互联**（每卡 7 条链路，8 卡两两直连），每卡 HCCS 双向总带宽约 **392 GB/s**（资料口径约 300~400 GB/s，约为 H100 NVLink 的一半；面试说量级即可）；
- **对比 NVLink**：同为"机内全互联 + 缓存一致性 + P2P 显存直访"，HCCS 带宽量级约为 NVLink 一半；NVLink 有 NVSwitch 扩展到机架级域，**HCCS 目前基本限定单节点**。

### 3. 具体数值样例

- 8 卡 910B：任意两卡间直接 P2P 读写对端显存（走 HCCS，无需经过 CPU/内存），TP 并行通信无瓶颈；
- 392 GB/s vs 机间 8×200G RoCE（每卡 25 GB/s）≈ **15 倍差距**——与 NVIDIA"NVLink 900 GB/s vs 400G 50 GB/s"的 18 倍同构；
- 训练脚本常见配置：`HCCS_CONNECT=...` / 确认 `npu-smi info` 显示 HCCS 链路正常（实习排障第一步）。

> 面试一句话总结：**HCCS = 昇腾的 NVLink：8 卡全互联、约 400 GB/s 量级、硬件缓存一致性、P2P 显存直访，只覆盖机内；机间靠 RoCE——昇腾拓扑与 NVIDIA"NVLink 机内 + IB/RoCE 机间"完全同构。**

---

## 13. 昇腾机间 RoCE 与服务器拓扑

### 1. 现有问题

昇腾机间怎么组网？"单/双机全链路"跑通要理解物理拓扑。

### 2. 方法论

- **单机 8 卡**（Atlas 800T A2/9000 系）：机内 HCCS 全互联；机间每节点约 **8× 200GE RoCE 上联**（每卡 1 口，以华为官网为准），双平面/胖树接入；
- **Atlas 900 集群**（2019）：1024 节点 × 8 = **8192 个昇腾 910**，机内 HCCS + 机间 RoCE 无损网络（CloudEngine），当时号称全球最快 AI 训练集群；
- **Atlas 900 A3 SuperPoD**（2024 超节点）：机架内更大规模互联（灵衢超节点/全光），机架间 Spine-Leaf RoCE，对标 GB200 NVL72 分层思路；
- **昇腾机间标配 RoCE**：HCCL 的 RDMA 路径跑在 RoCE 上（`HCCL_RDMA_*` 配置 = 无损网络配置：PFC/ECN/GID）——**昇腾不支持 IB**（官方组网以 RoCE 为准，IB HCA 不在标准 Atlas 硬件清单）。

### 3. 具体数值样例

- 双机训练（实习场景）：每机 8 卡 HCCS 全互联 + 8×200G RoCE 上联；HCCL 分层聚合（机内 HCCS、机间 RoCE），双机通信只传"机内归约后的 1 份"；
- 对比 NVIDIA：DGX H100 机内 NVLink 7.2 TB/s + 8×400G CX-7 网卡（400G ≈ 50 GB/s）——同构；
- 面试表述：**昇腾机间用 RoCE（无损以太）+ RDMA 语义，不依赖 IB 生态**。

> 面试一句话总结：**昇腾拓扑 = 机内 HCCS 全互联（8 卡 7 链路）+ 机间 RoCE 胖树（8×200GE 级），与 NVIDIA"NVLink 机内 + IB/RoCE 机间"同构；昇腾机间标配 RoCE 而非 IB，HCCL_RDMA_* 就是无损网络的配置位。**

---

## 14. CANN 通信栈层次（昇腾软件栈）

### 1. 现有问题

昇腾上从 PyTorch 到硬件的通信路径有几层？对应 NVIDIA 的什么？

### 2. 方法论

四层（对应 NVIDIA 栈）：

```text
① 应用/框架层：PyTorch（torch_npu）、MindSpore、TensorFlow
② CANN 层：AscendCL（ACL，应用接口，≈CUDA API）
            HCCL（集合通信，≈NCCL）
            GE（图引擎，≈编译器/图优化层）
            Runtime（任务/内存管理，≈CUDA Runtime）
③ 驱动层（Ascend HDK）：内核+用户态驱动，管理 DMA/中断/RDMA（RoCE）路径
④ 硬件层：Da Vinci NPU、HBM、HCCS、RoCE 网卡
```

通信路径：应用调 HCCL API → HCCL 选路（机内 HCCS / 机间 RoCE）→ 驱动 → 硬件。

### 3. 具体数值样例

- 一个 `HcclAllReduce` 调用：torch_npu 的分布式后端 → HCCL → 选 HCCS（机内）或 RoCE（机间）→ 驱动下发 DMA/RDMA → NPU 硬件执行；
- 对应 NVIDIA：`torch.distributed` → NCCL → NVLink/IB/RoCE 选路 → 驱动 → GPU；
- 面试点：**HCCL 是独立通信层、对应用透明**——换硬件栈不需要改训练代码（verl 在昇腾上跑就是靠这层抽象）。

> 面试一句话总结：**昇腾通信栈是"框架 → CANN（ACL/HCCL/GE/Runtime）→ 驱动 → 硬件"四层，HCCL 对标 NCCL 且对应用透明——这就是 verl 训练代码能在昇腾上无缝跑通的抽象层。**

---

# 五、其他平台

## 15. AMD：Infinity Fabric 与 RCCL

- **Infinity Fabric（xGMI）**：芯片间互联（CPU die 间、GPU XCD 间），类似 NVLink-C2C；MI300X 用 8 个 XCD 经 xGMI 全互联，聚合带宽约 896 GB/s（约），HBM3 总带宽 5.3 TB/s（192 GB）；
- **RCCL**：AMD 版 NCCL（fork 演进），ring/tree、支持 GDR，PyTorch ROCm 版底层使用；
- 机间：Azure ND MI300X v5 为 8× 400G InfiniBand NDR。

> 一句话考点：**AMD 用 xGMI（Infinity Fabric）做机内芯片互联（MI300X 约 896 GB/s）+ RCCL 对标 NCCL，机间用标准 IB/RoCE——与 NVIDIA/华为三层同构。**

## 16. Google TPU：ICI

- **ICI（Inter-Chip Interconnect）**：TPU 芯片间专用互联，**torus 拓扑**：v2/v3 为 2D torus；v4 起配合 **OCS（光电路交换）** 把 4096 芯片组成 pod；
- 带宽：v4 每链路约 400 Gb/s 级，v5p 每芯片聚合约 4.8 Tb/s（约）、8960 芯片/pod，Trillium（v6）再翻倍；
- 面试点：**torus 省交换成本（只连邻居），代价是远跳数多、需维度序路由；OCS 补大规模灵活性**——与 NVIDIA"全互联 + 胖树"路线差异最大。

> 一句话考点：**TPU 用 ICI + torus 拓扑 + OCS 光交换组万卡，成本低但跳数多——用拓扑换成本，与 NVIDIA 的全互联+胖树路线相反。**

## 17. AWS EFA / SRD

- **EFA**：AWS 自研弹性网卡，RDMA 语义 + 内核旁路，支持 GPUDirect RDMA，云上分布式训练主力；
- **SRD（Scalable Reliable Datagram）**：**可靠、无序（out-of-order）交付 + 多路径（包喷洒）**，规避单路径拥塞热点；
- 带宽：每 EFA 100 Gbps 级；p4d.24xlarge 4×100G = 400 Gbps；p5.48xlarge 8×400G 约 3.2 Tbps（约）；
- 面试点：**SRD 证明"无序 + 多路径"比"有序单路径"更适配 AI 大象流**（与 Meta 包喷洒思想一致）。

> 一句话考点：**EFA/SRD 用"多路径包喷洒 + 无序可靠交付"规避单路径拥塞，是云上 AI 网络的关键创新——无序传输反而更适合 AI 通信模式。**

## 18. Meta RoCE 部署与 800G 时代

- **Meta（SIGCOMM 2024《RDMA over Ethernet for Distributed AI Training at Meta Scale》）**：把主力训练集群从 IB 迁到自研 RoCE，生产跑 **24k GPU** 规模（2 层 CLOS、400G 光互联）；关键技术：**包喷洒多路径负载均衡**（避免 ECMP 哈希碰撞）、**sPFC**（选择性 PFC）、自研拥塞控制/限速（配合 ECN）；
- **800GE（IEEE 802.3df，2024）**：800G 光模块（OSFP）；交换机芯片 51.2T 时代（Broadcom Tomahawk 5、NVIDIA Spectrum-4）；
- **Spectrum-X**：Spectrum-4 交换机 + BlueField-3 DPU + 拥塞控制/隔离的**平台化 RoCE**，对标自家 Quantum-X800 IB（XDR 800G、576 端口、SHARP 网内计算）；
- 格局：AI 集群 = scale-up（NVLink/HCCS/IF/ICI）+ scale-out（IB 或 RoCE 800G）；**800G 端口约 100 GB/s 仍只有 NVLink 的 1/9，分层不会消失**。

> 一句话考点：**Meta 24k GPU 用包喷洒+sPFC+ECN 的自研 RoCE 证明万卡级可替代 IB；800G 时代 scale-out 达到约 100 GB/s，但机内 NVLink 900 GB/s 的 9 倍差距让"机内管快、机间管大"的分层长期不变。**

---

# 六、网络层细节（RDMA 内核知识）

## 19. RDMA 队列模型与 verbs

### 1. 现有问题

RDMA 为什么省 CPU？应用怎么跟网卡交互？——答 QP/CQ/verbs。

### 2. 方法论

- **QP（Queue Pair）** = SQ（发送队列）+ RQ（接收队列）；**CQ（完成队列）**；**SRQ**（共享接收队列）；
- **内存注册 MR** 拿 lkey/rkey（PD 保护域隔离）；
- **verbs 操作**：`ibv_post_send`（SEND / RDMA_WRITE / RDMA_READ / ATOMIC）、`ibv_post_recv`（预发布接收缓冲）、`ibv_poll_cq`（轮询完成）；
- **两种语义**：two-sided（SEND/RECV 双方参与）；**one-sided（RDMA READ/WRITE：对端 CPU 完全不参与）**——RDMA 省 CPU 的核心；
- 面试点：**RDMA 把连接管理/重传/DMA 卸载到网卡，应用只碰 QP/CQ，内核旁路 + 零拷贝**。

### 3. 具体数值样例

- one-sided RDMA READ：接收方预注册 MR 并告诉对方 rkey，发送方直接 `ibv_post_send(RDMA_READ)` 从对方显存/内存拉数据——接收方 CPU 零参与；
- 对比 TCP：每包中断 + 拷贝 + 协议栈处理（µs 级×N），RDMA 单次操作延迟 ~1 µs 级且 CPU 占用接近 0。

> 面试一句话总结：**RDMA 用 QP/CQ 队列模型 + verbs（post_send/post_recv/poll_cq）把连接与 DMA 卸载到网卡，one-sided RDMA READ/WRITE 让对端 CPU 完全不参与——内核旁路 + 零拷贝是它比 TCP 快一个数量级的根源。**

## 20. 传输类型：RC / UC / UD

- **RC（可靠连接）**：1:1、ACK+重传、保序，**最常用（NCCL 机间默认）**；
- **UC（不可靠连接）**：有错误检测无重传；
- **UD（不可靠数据报）**：1:N、需 GRH、消息长度受限（约 MTU 级）、无序；
- **XRC**：多进程共享 QP（HPC 优化）。

> 一句话考点：**RC 可靠性靠网卡重传，所以 RoCE 必须无损——一丢包重传雪崩；NCCL 机间默认 RC，这是"无损网络"需求的直接来源。**

## 21. 无损网络与拥塞控制（三件套深挖）

- **PFC（802.1Qbb）**：按优先级逐跳暂停保零丢包——问题：队头阻塞、死锁、PFC 风暴（全局暂停蔓延）；
- **ECN**：队列超阈值打 CE 标记（端到端信号）；
- **DCQCN**：接收端收 CE → 发 CNP 回报 → 发送端**量化降速再恢复**——RoCE 事实标准；
- 趋势：sPFC、包喷洒多路径（Meta）、INT 遥测、AI 拥塞控制（Spectrum-X 动态路由）。

> 一句话考点：**无损 ≠ 不拥塞——PFC 是"刹车保命"（逐跳暂停防丢包但会队头阻塞），ECN/DCQCN 是"提前减速"（端到端标记+发送端降速）；现代趋势是用多路径+选择性 PFC 减少对全局暂停的依赖。**

## 22. NUMA 与带宽

- 双路 x86 上 PCIe 设备分属不同 socket（NUMA 节点），GPU 与 NIC 跨节点要走 **UPI（Intel）/Infinity Fabric（AMD）**，带宽有限；
- 调优：**GPU-NIC 1:1 亲和**（DGX GPU0↔NIC0）、`numactl` 绑核、网卡 IRQ 亲和；NCCL/HCCL 拓扑检测（`NCCL_TOPO_DUMP_FILE`）自动规避跨 NUMA。

> 一句话考点：**拓扑亲和性决定机间带宽能否打满——GPU 与网卡跨 NUMA 直接掉一半带宽，所以 DGX 等 AI 服务器都是 GPU-NIC 一一对应直连。**

---

# 七、AI 集群互联总体现状（架构串讲）

## 23. 为什么分层：scale-up 管快、scale-out 管大

### 1. 现有问题

为什么 AI 集群一定要"机内 NVLink/HCCS + 机间 IB/RoCE"两层？不能统一用一种？

### 2. 方法论（数字驱动）

根因：**单卡显存装不下大模型 → 多卡多机 → 集合通信流量巨大**。数量级对比：

```text
机内 NVLink 4.0：900 GB/s（H100） / HCCS：约 392 GB/s
机间 800G 端口：约 100 GB/s（RoCE/IB）
→ 带宽差 4~9 倍（NVLink 甚至 9~18 倍）
```

- 分层收益：**机内高带宽消化细粒度通信**（TP 每层 all-reduce、注意力），**机间只传必要数据**（层级 allreduce：机内归约后再机间传一次）；
- 配套：通信与计算重叠（Megatron/DeepSpeed 的通信调度）；
- MoE 例外：AlltoAll 通信量大且跨机，是机间带宽的主要压力来源（所以 MoE 训练常配 EP 组内全互联 + 机间高带宽）。

### 3. 具体数值样例

- 8B Dense TP=8 单机：all-reduce 全走 NVLink（900 GB/s），机间零通信——这就是"小模型单机训"快的原因；
- 1.8T MoE EP=64 跨 8 机：每 token 的 AlltoAll 要把激活发到多个专家所在机——机间带宽成为瓶颈，需 400G/800G × 每卡。

> 面试一句话总结：**带宽差一个量级决定分层——机内 NVLink/HCCS（900/392 GB/s）消化细粒度 TP 通信，机间 IB/RoCE（100 GB/s 级）只传必要的跨机数据（层级 allreduce）；MoE 的 AlltoAll 是机间带宽的主要压力来源。**

## 24. 8 卡机 vs 万卡集群：拓扑对比

### 1. 现有问题

从一台 8 卡机到万卡集群，网络拓扑怎么变？

### 2. 方法论

- **8 卡机**：全互联（NVSwitch/HCCS）+ 每卡 1 NIC（DGX 8×400G/200G），机内带宽是机间 10 倍以上；
- **万卡集群**：**胖树（CLOS）** 为主（leaf 接服务器、spine 汇聚）；Meta 24k GPU、华为 Atlas 900 的 8192 NPU、TPU v5p 的 8960 芯片；万卡瓶颈不是带宽而是**拥塞控制、路由/负载均衡、故障域、checkpoint IO、慢节点**；
- 拓扑四选：**全互联**（带宽最优但链路 O(N²) 成本爆炸，只用于机内 8 卡）；**胖树/CLOS**（无阻塞 bisection、可扩展万卡，机间主流）；**Torus**（只连邻居、成本最低但跳数多，TPU + OCS）；**Dragonfly**（组内全互联 + 组间稀疏，HPC 常用）。

### 3. 具体数值样例

- 万卡胖树：2048 端口 spine × 1024 leaf，服务器 8×400G 上联 leaf，leaf 4×400G 上联 spine——bisection 带宽约 1.6 PB/s（800G 时代翻倍）；
- 对比 torus：4096 TPU 的 2D/3D torus 只需 4~6 邻居链路，交换机成本极低但最远跳数 ~20+（v4 后 OCS 重构网络降低跳数）。

> 面试一句话总结：**8 卡机内全互联（NVSwitch/HCCS）、万卡机间胖树（CLOS）是主流；TPU 用 torus+OCS 换成本、Dragonfly 服务 HPC——拓扑选择 = 带宽/成本/可扩展性三角权衡，万卡真正的难点在拥塞控制与故障域。**

---

# 附：30 秒速查表（面试背诵版）

| 技术 | 一句话考点 |
|---|---|
| NVLink | 专用 SerDes + NVSwitch 全互联，H100 900 GB/s、B200 1.8 TB/s，只做机内/机架内 |
| NVLink-C2C | 复用 PCIe 5.0 SerDes 的 die 间一致性互联（GH200 900 GB/s），CPU-GPU 统一内存硬件基础 |
| PCIe | 通用兜底总线，5.0 x16 = 64 GB/s，6.0 PAM4 翻倍 128 GB/s |
| CXL | PCIe 物理层上 io/cache/mem 三协议，3.0 每方向 64 GB/s，远期显存池化 |
| UVM | 按需分页统一虚拟内存（软件），NVLink/C2C 是高速迁移通道 |
| InfiniBand | 专用 RDMA 网络，SM 集中管理 + 硬件无损，NDR 400G / XDR 800G |
| RoCE | UDP 4791 上的 RDMA，PFC 保底 + ECN/DCQCN 降速才无损 |
| GPUDirect RDMA/GDS | 网卡/盘直接 DMA 进显存，绕 CPU 拷贝 |
| NCCL | ring（带宽）/tree（时延）/NVLS（网内归约）+ 拓扑感知多传输，NCCL_IB_* 是调优位 |
| HCCL | 昇腾的 NCCL，机内 HCCS + 机间 RoCE，HCCL_RDMA_* 是调优位 |
| HCCS | 昇腾的 NVLink，8 卡全互联约 392 GB/s，缓存一致性，只覆盖机内 |
| EFA/SRD | 多路径 + 无序可靠传输，云上 AI 网络 |
| TPU ICI | torus + OCS 光交换低成本万卡 |
| 分层 | 机内管快、机间管大，带宽差一个量级决定架构 |

**面试金句**："AI 集群 = scale-up（NVLink/HCCS/ICI 机内全互联）+ scale-out（IB/RoCE 800G 机间胖树）；NCCL/HCCL 用拓扑感知把 ring/tree 算法映射到物理链路，机内 NVLink、机间 RDMA；RoCE 靠 PFC+ECN+DCQCN 变无损；昇腾是 HCCS（机内）+ RoCE（机间）+ HCCL（集合库）的同构生态。"

**参考来源**（subagent web 检索实际使用）：
- [NVIDIA NCCL 官方文档（环境变量）](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_242/nccl-developer-guide/docs/env.html)
- [华为昇腾社区：HCCL 常用工具与配置](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/hccl/hcclug/hcclug_000025.html)
- [华为：Atlas 800T A2 训练服务器用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100317202/f3dba488)
- [Meta：RDMA over Ethernet for Distributed AI Training（SIGCOMM 2024）](https://dlnext.acm.org/doi/epdf/10.1145/3651890.3672233)
- [DCQCN：Congestion Control for Large-Scale RDMA（SIGCOMM 2015）](https://dl.acm.org/doi/abs/10.1145/2829988.2787484)
- [NVIDIA：GB200 NVL72](https://www.nvidia.com/en-gb/data-center/gb200-nvl72/)
- [AMD ROCm：RCCL 官方文档](https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.0/what-is-rccl.html)
- [NVIDIA Quantum-X800：XDR 800G InfiniBand](https://introl.com/blog/infiniband-switches-quantum-x800-xdr-sharp-ai-supercomputer-2025)
- [IP Infusion：RoCEv2 无损以太原理](https://www.ipinfusion.com/technology/rocev2/#main-content)
