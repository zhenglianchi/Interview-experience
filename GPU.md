# GPU 硬件架构完全指南（AI Infra 面经版）

> **一句话定位**：GPU 是 AI Infra 的物理底座——本文回答"一块 GPU 由什么组成、哪些组件负责计算、数据怎么在芯片里流动、性能天花板卡在哪"，以及"面试官问硬件时到底在考什么"。

AI Infra 面试里硬件部分的特点是：**门槛不高但极易露怯**——说得出"SM/Tensor Core/HBM"这三个词的人很多，但能把"SM 里面到底有什么、哪些是计算单元、为什么 Tensor Core 快、显存带宽怎么算、MFU 为什么只有 30%"讲成一条链的人很少。本文按"从芯片到算子"的顺序拆成 **12 个技术点**（含 CUDA 并发与 GPUDirect），再用一节串讲把一次 GEMM 的完整数据通路走一遍，最后给一份高频题检查清单。

**范围与分工**（避免与仓库内其他文档重复）：

| 主题 | 本文 | 去哪看 |
|---|---|---|
| GPU 芯片内部结构、计算单元、存储层次、性能模型 | **✅ 主体（§1–§10）** | — |
| **CUDA Stream / 并发重叠 / Hyper-Q / CUDA Graphs** | **✅ §11** | — |
| **GPUDirect（P2P / RDMA / GDS）、门铃机制、pinned / zero-copy / UVA / UM** | **✅ §12（GPU 侧机制）** | 网络侧结论（GDR/GDS 的收益、`NCCL_NET_GDR_LEVEL`、`cuFile`）见 `Communication.md` 第 7 节 |
| NVLink / PCIe / CXL / InfiniBand / RoCE / HCCL 等**互联技术** | 只给带宽量级与"五个量"中的一项 | `Communication.md` |
| DP/TP/PP/EP/FSDP 等**并行切分** | 只讲硬件为什么支持 | `Parallel.md` |
| Kernel 级算子实现（PagedAttention / FlashAttention） | 只讲它们在硬件上做了什么 | `VLLM.md` / `Speculative-Decoding.md` |
| 显存怎么被训练/推理吃掉的**账** | 给公式与示例 | `Agent-lighting.md` 第 19 节 |

**事实来源与口径说明**（本文所有数字都标了来源或标了"推算"）：

- NVIDIA 官方产品页：[H100](https://www.nvidia.com/en-us/data-center/h100/)、[H200](https://www.nvidia.com/en-us/data-center/h200/)、[GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)（**页脚注明 Tensor Core 数字默认是 sparse，dense 为一半**）
- 架构代际与 SM 结构：[阿里云开发者社区《【AI系统】GPU 架构回顾（从2018年-2024年）》（ZOMI酱）](https://developer.aliyun.com/article/1642215)
- 主流型号参数对照与 datasheet 读法：[llm-infra-atlas《GPU 硬件参数：常用量与主流型号对照》](https://github.com/llm-infra-atlas/llm-infra-atlas.github.io)
- GPU 内存层次的带宽/延迟数字：[Triton 教材第 8 章《内存优化——Coalescing、Layout 与 Shared Memory》](https://github.com/Allen-C-Guan/Pytorch-Inductor-Tutorial)
- Roofline 模型与拐点计算：[EngineersOfAI《Roofline Model and Bottleneck Analysis》](https://engineersofai.com/docs/hardware-and-silicon/gpu-architecture/roofline-model-and-bottleneck-analysis)；原始论文 Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009
- 带宽计算公式：[NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- 面经考点清单：[牛客《C++/CUDA/AI-infra 面试经验总结》](https://www.nowcoder.com/discuss/851502759596523520)
- 昇腾侧：[ForceInjection/AI-fundamentals《昇腾硬件架构与 CANN 软件栈》](https://github.com/ForceInjection/AI-fundamentals)

> **注意**：本文标注"**推算**"的数字是按官方参数自己算出来的（推导过程都写出来了），标注"**实测**"的是上述来源里的实测值，其余为官方 datasheet 值。面试中把推算当成官方值说会被追问穿，所以务必分清。

---

# 一、核心硬件技术点

## 1. GPU 与 CPU 的本质区别：为什么 GPU 长成这样

### 1. 现有问题：为什么不能"用 CPU 堆核"来训大模型

面试最常见的第一问是"GPU 和 CPU 到底差在哪"。如果只答"GPU 核多"，会被立刻追问："那 CPU 也可以堆到 100 核，为什么还是不行？"真正的答案在于**两者优化的目标函数不同**：

- **CPU 优化的是"单个任务的延迟"**（latency-oriented）：为了让一条指令流尽快跑完，需要**大容量多级 cache**（隐藏访存延迟）、**复杂的乱序执行/分支预测/寄存器重命名**（提高指令级并行）、**少量但强大的核**（支持复杂指令）。这些控制逻辑会吃掉大量芯片面积。
- **GPU 优化的是"单位时间完成的总工作量"**（throughput-oriented）：单个线程很慢（顺序执行、无分支预测、无乱序），但**同时挂着几千个线程**，用"**线程多 + 切换零开销**"来掩盖访存延迟——某个 warp 等内存时，warp scheduler 立刻切到另一个 ready 的 warp，ALU 不停工。

**这就解释了一个反直觉的现象**：GPU 的访存延迟（HBM 约 500–800 cycles）比 CPU 的 DRAM 延迟还高，但带宽利用率却能打到很高——因为它靠**并发度**而不是靠**低延迟**取胜（latency hiding via TLP，Thread-Level Parallelism）。

### 2. 方法论：面积分配、执行模型与延迟隐藏的算术

**（1）面积分配决定一切。** GPU 把绝大部分晶体管给了 ALU 和寄存器堆，只留很小的面积做控制与 cache；CPU 反过来。这带来两个后果：

- GPU **能同时执行的线程数远超 CPU**：H100 一个 SM 最多 2048 线程、132 个 SM → 理论 27 万线程在飞；主流服务器 CPU 是"几十核 × 2 线程"级别；
- GPU **单线程性能弱**：分支、递归、指针追逐这类"控制密集"负载在 GPU 上会退化得很惨（这正是不建议把整个训练脚本的 Python 逻辑放到 GPU 上执行的原因）。

**（2）执行模型是"两级并行"**：粗粒度靠 Block 抢占 SM（一个 Block 只能在一个 SM 上跑完，**不能跨 SM 拆**），细粒度靠 Warp（32 线程）在 SM 内锁步执行并由硬件调度。

**（3）延迟隐藏的算术（这是理解 GPU 性能的关键公式）。** 要让 ALU 不空转，需要足够的并发线程：

$$
\text{需要的并发 warp 数} \approx \frac{\text{访存延迟（cycles）}}{\text{每次访存间隔能执行的指令数}} \times \text{发射槽位}
$$

用具体数字感受一下：如果一次 HBM 访问要等 **600 cycles**，而 ALU 每 cycle 能发射 4 个 warp 的指令，那么要填满这 600 cycles 的等待，就需要大约 $4 \times$ 很多个 warp 轮流上——这就是为什么 SM 要同时驻留 48~64 个 warp（Hopper SM 上限 **64 warps = 2048 线程**）。

**（4）由此推出 GPU 的三条设计哲学**，每一条都能对应到一个具体硬件：

| 设计哲学 | 硬件体现 | 面试怎么用 |
|---|---|---|
| 用并发换延迟 | 多 warp + warp scheduler 零开销切换 | 解释"为什么 GPU 不怕高延迟" |
| 用宽度换控制 | 寄存器堆极大（每 SM 256 KB）、控制逻辑极小 | 解释"为什么 GPU 单线程弱" |
| 用专用化换吞吐 | Tensor Core / TMA / Copy Engine / NVJPEG | 引出后面第 4 节（Tensor Core） |

### 3. 具体数值样例

```text
场景：为什么 600 cycle 的访存延迟不会拖垮 GPU

【CPU 的做法】乱序执行 + 大 cache
  核数 64，每核 2 线程 → 最大并发 128 线程
  靠 OoO window（几百条指令）+ L1/L2/L3（数十 MB）把延迟压到"看得见"
  → 单线程延迟低，但并发度上不去

【GPU 的做法】多线程 + 快速切换
  H100：132 SM × 最多 64 warp/SM × 32 线程 = 270,336 线程（理论上限）
  单个 warp 发出 HBM load 后要等 ~600 cycles
  warp scheduler 每 cycle 从 ready 队列里挑一个 warp 发射 → 等待被别的 warp 填满
  若 SM 里驻留 32 个 warp、每个 warp 平均每 4 cycle 能发射一条指令：
    可覆盖的延迟 ≈ 32 × 4 = 128 cycle（不够）→ 需要更大的 ILP 或更多 warp
  ⇒ 这解释了"occupancy（占用率）为什么重要"，也解释了
    "为什么 occupancy 高不等于快"（还要看 ILP 与访存模式）

【面积直觉（定性）】
  CPU：控制 + cache 占大头，ALU 占小头
  GPU：ALU + 寄存器堆占大头，控制 + cache 占小头
  因此 GPU 的 cache 容量（H100 L2 = 50 MB）远小于
  同价位 CPU 的 L3，但寄存器总量极大
```

### 4. 演进：GPU 正在"专用化"

从 Volta 到 Blackwell，GPU 增加的关键单元几乎都是**专用加速器**而不是通用 ALU：Volta 引入第一代 Tensor Core；Turing 加入 RT Core；Ampere 加 TF32/BF16 与结构化稀疏；Hopper 加入 **TMA（Tensor Memory Accelerator，专用张量搬运单元）**、Thread Block Cluster、DPX 指令；Blackwell 把 Tensor Core 推到 FP4/FP6 并引入双 die 封装。

**结论：GPU 的"通用性"在下降，"矩阵+搬运"这两件事被越来越强的专用硬件接管**——这也是为什么"会写 CUDA kernel"的价值在下降，而"懂数据通路与 roofline"的价值在上升。

> **面试一句话总结**：GPU 与 CPU 的本质区别不是"核多"，而是**优化目标从"单任务延迟"换成了"吞吐"**——CPU 用大 cache + 乱序执行 + 少量强核压延迟，GPU 用"每 SM 驻留 48~64 个 warp + warp scheduler 零开销切换"把 600 cycle 的 HBM 延迟用并发填掉；代价是单线程性能弱、控制密集负载会退化；由此衍生出三条设计哲学（并发换延迟、宽度换控制、专用化换吞吐），也解释了后面所有硬件（Tensor Core/TMA/Copy Engine）存在的理由。

---

## 2. 芯片级组成：GPC → TPC → SM，哪些组件用于计算

### 1. 现有问题："一块 GPU 由什么组成"该怎么答

"GPU 由什么组成"是硬件题的第一道筛子。答"由很多核心组成"等于没答；正确的答法是**按层级 + 按功能**两个维度同时给出，并明确指出**哪一个层级才是真正的计算单元**。

### 2. 方法论：层级结构（从大到小）与功能分类

**（1）层级结构**：一块 GPU die 自上而下是

$$
\text{GPU die} \rightarrow \text{GPC（图形处理集群）} \rightarrow \text{TPC（纹理处理集群）} \rightarrow \text{SM（流式多处理器）}
$$

- **GPC（Graphics Processing Cluster）**：芯片上的大区块，每个 GPC 含若干 TPC 和一个专用光栅引擎（Raster Engine，图形用）；
- **TPC（Texture Processing Cluster）**：每个 TPC 通常含 **2 个 SM**，外加纹理单元（Texture Unit）；
- **SM（Streaming Multiprocessor）**：**真正的计算单元**——所有 CUDA 线程、所有 Tensor Core 运算都发生在 SM 里。

用三个真实 die 把这条链走一遍（数据来源：ZOMI 架构回顾，见文首来源）：

| 架构 / 型号 | GPC | TPC | SM | CUDA Core | Tensor Core | 其他 |
|---|---|---|---|---|---|---|
| **Turing TU102**（完整 die） | 6 | 36 | 72 | 4608 | 576（2nd gen） | 72 RT Core、288 纹理单元、12× 32-bit GDDR6 控制器（384-bit） |
| **Ampere GA100**（完整 die） | 8 | 64 | 128 | 8192 | 512（3rd gen） | 6 个 HBM2 栈、12× 512-bit 内存控制器 |
| **Hopper GH100**（完整 die） | 8 | 66 | 132 | 16896 | 528（4th gen） | 50 MB L2、HBM3 |
| **Hopper H100 SXM**（产品） | — | — | 132（启用） | 16896 | 528 | 80 GB HBM3、5120-bit |

**注意最后两行的区别**：die 上有的 SM 不代表产品全开——GA100 die 有 128 个 SM，但 **A100 产品只启用 108 个**（这是良率与功耗权衡的常规做法）。面试里把 die 规格和产品规格混着说是常见扣分点。

**（2）按功能分类（这才是"哪些用于计算"的答案）**：

| 功能类别 | 具体组件 | 是否"计算" |
|---|---|---|
| **计算（标量/向量）** | SM 内的 FP32/INT32/FP64 CUDA Core、SFU（特殊函数单元） | ✅ 通用计算 |
| **计算（矩阵）** | SM 内的 **Tensor Core** | ✅ 矩阵乘加（AI 主力） |
| **计算（图形）** | RT Core、光栅引擎 | ❌ 与 AI 无关 |
| **访存** | LD/ST（加载存储单元）、**TMA**（Hopper 新增的专用搬运单元） | ❌ 只搬运 |
| **调度** | GigaThread Engine（全局把 Block 分派到 SM）、SM 内的 4 个 Warp Scheduler + Dispatch Unit | ❌ 只调度 |
| **片上存储** | 寄存器堆、L1/Shared Memory、L2 Cache | ❌ 只存 |
| **片外存储** | HBM 栈 + 内存控制器 | ❌ 只存 |
| **IO / 拷贝** | PCIe 控制器、NVLink 接口、**Copy Engine**（DMA，与 SM 异步）、NVJPEG/NVDEC/NVENC | ❌ 只搬 |

**这张表的用法**：面试问"哪些组件用于计算"时，答"**SM 里的 CUDA Core 和 Tensor Core**"；再补一句"**SFU 也算计算单元，但只做 sin/exp/rsqrt 这类特殊函数**"；然后主动指出"**LD/ST、TMA、Copy Engine 都是搬运不是计算，但现代 GPU 的性能瓶颈往往在它们身上**"——这一句能直接把话题引到后面第 5 节（存储层次）和第 8 节（roofline），是很好的主动引导。

### 3. 具体数值样例

```text
【把 H100 的 132 个 SM 数出来】
  8 GPC × 每 GPC ... → 官方给的是 66 TPC / 132 SM
  66 TPC × 2 SM/TPC = 132 SM ✓
  CUDA Core 核算：132 SM × 128 FP32 lanes = 16,896 ✓（与官方 CUDA 核心数一致）
  Tensor Core 核算：132 SM × 4 TC/SM = 528 ✓

【A100 的 die vs 产品】
  GA100 die：8 GPC × 8 TPC × 2 SM = 128 SM；128 × 64 FP32 = 8192 CUDA Core
  A100 产品：108 SM 启用 → 108 × 64 = 6912 CUDA Core
  ⇒ 面试若说"A100 有 128 个 SM"就错了（那是 die 规格）；
    说"A100 有 108 个 SM / 6912 个 FP32 核"才对

【为什么"A100 每 SM 64 个 FP32"而"H100 每 SM 128 个"】
  两者都分 4 个 processing block
  A100：每 block 16 FP32 → 4 × 16 = 64
  H100：每 block 32 FP32 → 4 × 32 = 128（翻倍）
  ⇒ 同样的 SM 数量下，Hopper 的 FP32 吞吐直接翻倍，这就是
    "Hopper FP32 Core 两倍于 Ampere"的硬件来源
```

### 4. 演进：层级结构在变，SM 的"重量"在变

- Turing→Ampere：SM 内 FP32 与 INT32 分开（Turing 每个 block 64 FP32 + 64 INT32），开始支持独立整数流水；
- Ampere→Hopper：**FP32 与 FP64 数据通路翻倍**（每 block 32 FP32 + 16 FP64），**新增 TMA**；
- Hopper→Blackwell：引入**双 die 封装**（2080 亿晶体管、两片 die 通过 10 TB/s 片间互连"拼"成一块 GPU），SM 数量与 Tensor Core 精度继续扩张（FP4/FP6），并出现 **NVL72 这种"机架即一块 GPU"** 的系统级形态。

**趋势**：GPC/TPC/SM 这套层级没变，但"一块 GPU"的边界在从 die 扩展到封装再到机架——面试里能说出"Blackwell 是双 die，靠 10 TB/s 片间互连伪装成一块 GPU"，是个明显的加分点。

> **面试一句话总结**：GPU die 的层级是 **GPU → GPC → TPC → SM**，其中**只有 SM 是真正的计算单元**（GPC 含光栅引擎、TPC 含纹理单元，都是图形遗留）；按功能分，**计算单元是 SM 内的 CUDA Core（标量/向量，H100 每 SM 128 个 FP32）与 Tensor Core（矩阵，每 SM 4 个）加 SFU**，LD/ST、TMA、Copy Engine 是搬运，GigaThread/warp scheduler 是调度，寄存器堆/L1-Shared/L2/HBM 是存储；要注意 **die 规格 ≠ 产品规格**（GA100 die 128 SM，A100 产品只启用 108 SM / 6912 FP32 核；GH100 die 144 SM，H100 用 132 SM / 16896 FP32 核 = 132×128）。

---

## 3. SM 内部结构：4 个处理块与六类功能单元

### 1. 现有问题：SM 里面到底装了什么

"SM 里有什么"是硬件题里**最能区分深浅**的一问。粗略答"有 CUDA Core 和 Tensor Core"及格；能按 processing block 拆开、把每类单元的数量和职责说清、并解释"为什么要分成 4 块"，才是面试官想听的。

### 2. 方法论：Hopper SM 的完整构成

**（1）SM 被均分为 4 个 processing block（也叫 SM sub-partition / SMSP）。** 每个 block 拥有独立的 warp scheduler 与 dispatch unit，以及一组自己的执行单元。分块的意义是**让 4 个 block 可以并行取指、发射、执行**——如果 4 个 block 共用一个调度器，warp 多了调度器就会成为瓶颈。

**（2）Hopper SM 的完整清单**（数据来源：ZOMI 架构回顾）：

| 单元 | 整个 SM 数量 | 每个 processing block | 相对 Ampere |
|---|---|---|---|
| **Warp Scheduler** | 4 | 1 | 相同 |
| **Dispatch Unit** | 4 | 1 | 相同 |
| **FP32 Core** | **128**（4×32） | 32 | **×2** |
| **INT32 Core** | 64（4×16） | 16 | 相同 |
| **FP64 Core** | 64（4×16） | 16 | **×2** |
| **Tensor Core（4th gen）** | **4**（4×1） | 1 | 升级到 4th gen |
| **LD/ST Unit** | 32（4×8） | 8 | 相同 |
| **SFU** | 16（4×4） | 4 | 相同 |
| **TMA（Tensor Memory Accelerator）** | 有 | — | **新增** |
| **寄存器堆** | 256 KB（64K × 32-bit 寄存器） | 64 KB | — |
| **L1 / Shared Memory** | 228 KB（可配置） | 共享 | — |

**（3）按职责给这六类单元"定性"**（面试可以直接背这一段）：

1. **FP32/INT32/FP64 Core = 标量运算主力**：每个 core 每 cycle 一条指令（FP32 支持 FMA，即 fused multiply-add，所以计 2 FLOP）；
2. **Tensor Core = 矩阵运算主力**：不在 CUDA Core 的"每 core 一条标量指令"模型里，它一次完成一个小矩阵乘加（详见第 4 节）；
3. **SFU（Special Function Unit）= 特殊函数**：`sin/cos/exp/log/rsqrt`，吞吐是 FP32 的 1/4~1/8，所以"自己写 exp 近似代替 `__expf`"这类优化才有意义；
4. **LD/ST = 访存**：负责把数据在寄存器和 L1/Shared/Global 之间搬，**吞吐有限**（每 block 8 个），访存密集 kernel 常卡在这里；
5. **Warp Scheduler + Dispatch = 调度**：每 cycle 从 ready warp 里挑一个发射（Hopper 每 scheduler 每 cycle 可发射 1 条指令）；
6. **寄存器堆 + L1/Shared = 片上存储**：寄存器是**每线程私有**、延迟接近 0；Shared Memory 是**block 内共享**、延迟约 20–30 cycles。

**（4）寄存器堆的容量怎么算**（面试常考的细节）：

- Hopper 每 SM 有 **65536 个 32-bit 寄存器** = $65536 \times 4\text{B} = 256\text{ KB}$；
- 每个线程能拿到的寄存器数决定**能同时驻留多少 warp**——寄存器是 occupancy 的第一约束（见第 7 节）；
- 寄存器是 GPU 上**最快但最稀缺**的资源之一，`__launch_bounds__` 和 `maxrregcount` 就是用来在"寄存器溢出到 local memory"和"occupancy 太低"之间做取舍的。

### 3. 具体数值样例

```text
【样例 A：从 SM 结构反推 H100 的 FP32 峰值（验证官方 67 TFLOPS）】
  H100 SXM：132 SM × 128 FP32 lanes/SM = 16,896 个 FP32 lane
  每个 lane 每 cycle 一次 FMA = 2 FLOP
  ⇒ 每 cycle：16896 × 2 = 33,792 FLOP
  取 boost 频率 ~1.98 GHz：
    33,792 × 1.98e9 = 6.69e13 FLOP/s ≈ 66.9 TFLOPS ✓（官方标 67 TFLOPS）
  ⇒ 这条推导说明：**FP32 峰值 = SM 数 × 每 SM FP32 lane 数 × 2 × 频率**，
    面试里能当场推出来，比背数字可信得多

【样例 B：为什么 FP32 峰值只有 BF16 Tensor Core 的 1/15】
  同代同频下，BF16 Tensor Core dense ≈ 989 TFLOPS，FP32 CUDA Core ≈ 67 TFLOPS
  比值 ≈ 14.8×
  ⇒ 差距来自"一次算一个标量"vs"一次算一个小矩阵"
    （第 4 节会把这个倍数拆开算）

【样例 C：occupancy 的第一个约束是寄存器】
  Hopper：65536 寄存器/SM
  某 kernel 每线程用 64 个寄存器，block = 256 线程（8 warp）
    每 block 占用：256 × 64 = 16,384 寄存器
    寄存器维度允许的 block 数：65536 / 16384 = 4 个
    ⇒ 4 × 256 = 1024 线程 = 32 warp（SM 上限 64 warp）→ 寄存器把 occupancy 压到 50%
  若降到 32 寄存器/线程：每 block 8192 → 8 个 block → 2048 线程 = 100%
  ⇒ 这就是"少用寄存器能提高并行度"的算术，代价可能是 spill（溢出到 local memory）
```

### 4. 演进：SM 的"计算密度"在提升，"搬运"被独立出来

- **FP32/FP64 通路翻倍**（Ampere→Hopper）：在 SM 数量不变的情况下提升通用算力；
- **Tensor Core 从"每 block 一个"变成"每 block 一个但一代比一代强"**：靠精度（FP8/FP4）而不是靠数量提吞吐；
- **TMA 成为独立单元**（Hopper 新增）：把"Global→Shared"的多维搬运从 LD/ST + 寄存器里解放出来，SM 用一条指令启动大块异步拷贝，**数据不经过寄存器**——这对 GEMM/Attention 的流水线（producer-consumer、warp specialization）至关重要；
- **Thread Block Cluster**（Hopper 新增）：允许跨 SM 的分布式 Shared Memory（DSMEM），是"把 SMEM 当分布式缓存用"的起点。

> **面试一句话总结**：SM 是 GPU 的计算单元，内部**均分为 4 个 processing block**，每个 block 有独立的 warp scheduler/dispatch 与 32 FP32 + 16 INT32 + 16 FP64 + 1 Tensor Core + 8 LD/ST + 4 SFU，整个 SM 共享 **256 KB 寄存器堆（65536 × 32-bit）与 228 KB 可配置 L1/Shared**，Hopper 还新增了 **TMA** 专用搬运单元；**计算单元只有 CUDA Core（标量/向量）和 Tensor Core（矩阵）加 SFU**，LD/ST 与 TMA 是搬运、warp scheduler 是调度、寄存器与 L1/Shared 是片上存储；能用"SM 数 × 每 SM lane 数 × 2 × 频率"当场推出 H100 的 67 TFLOPS FP32，是这一节的加分动作（同时也能解释为什么 FP32 只有 BF16 Tensor Core 的 ~1/15）。

---

## 4. Tensor Core：矩阵单元为什么快、又为什么挑剔

### 1. 现有问题：Tensor Core 凭什么快一个数量级

面试问得最多的一句是"**Tensor Core 和 CUDA Core 有什么区别**"。答"Tensor Core 算矩阵、CUDA Core 算标量"只是描述现象；真正要回答的是：**为什么算矩阵就能快 10 倍以上？这种快有代价吗？**

### 2. 方法论：MMA 指令、数据复用与代价

**（1）Tensor Core 的本质是一条"矩阵级"指令（MMA）。** CUDA Core 的指令粒度是"一个标量 FMA"（$d = a \times b + c$，2 FLOP）；而 Tensor Core 执行的是

$$
D_{m \times n} = A_{m \times k} \times B_{k \times n} + C_{m \times n}
$$

一条指令完成 $2 \times m \times n \times k$ 个 FLOP。以 Volta/Ampere 代的 `m16n8k16`（BF16）为例：$2 \times 16 \times 8 \times 16 = 4096$ FLOP **一条指令**。

**（2）快的根本原因不是"更快的乘法器"，而是"数据复用"。** 矩阵乘法的瓶颈通常是**操作数供给**：如果每个乘加都要从寄存器/共享内存取两个操作数，访存带宽会立刻成为天花板。Tensor Core 的做法是：

- 把 $A$、$B$ 的小块**加载一次，参与 $m \times n \times k$ 次乘加**——复用度 = $O(k)$；
- 硬件内部是**二维乘加阵列 + 累加器**（脉动/Systolic 风格），数据在阵列里横向/纵向流动，**每个周期每个 PE 都在做乘加**，不需要给每个 PE 单独取数；
- 累加器 $C$ 常驻在寄存器里，跨很多个 MMA 指令累加，不需要每次写出。

**（3）代价一：整个 warp 协作完成一条 MMA，操作数按特定规则分布在 32 个线程的寄存器中。** 这就是为什么"自己写 Tensor Core kernel"很难，必须用 `mma.sync` 的**规定 layout**（哪个线程持有哪些元素），Triton 里的 `DotOperandEncodingAttr`/`NvidiaMmaEncodingAttr` 就是在编码这件事。也解释了为什么 Triton 要在 `dot` 操作前后插入 `convert_layout`（代价是 shared memory 中转或 warp shuffle）。

**（4）代价二：对 shape 挑剔。** MMA 指令的 $m,n,k$ 是**固定的几个合法组合**（如 16×8×16），所以：

- 张量维度不是这些值的整数倍时要 **padding**（浪费算力）；
- 小 batch / seq_len=1 的 decode 场景，$m$ 维只有 1，Tensor Core 的利用率天然很低——这是"decode 为什么是 memory-bound"的硬件级原因之一；
- 反过来说，**"把 batch 做大 / 把 GEMM 的 M 维做大"能立刻提高 Tensor Core 利用率**，这是推理侧 continuous batching 的硬件动机。

**（5）代价三：精度受限。** Tensor Core 支持的输入精度是一代一代加的：Volta FP16 → Turing INT8/INT4 → Ampere **TF32/BF16/FP64** → Hopper **FP8** → Blackwell **FP6/FP4**。而且**累加通常仍是 FP32**（保证数值稳定），所以"用 BF16 算、用 FP32 累加"是标准做法。

**（6）2:4 结构化稀疏：营销数字的 2× 从哪来。** Ampere 起支持 **2:4 structured sparsity**：每 4 个连续权重里恰好 2 个为零时，硬件可跳过零元、吞吐翻倍。**NVIDIA 官方页上的 Tensor Core 峰值默认是 sparse 口径**（页脚写明 "Specification in sparse. Dense is one-half sparse spec shown."），所以：

$$
\pi_{\text{dense}} = \frac{\pi_{\text{sparse}}}{2}
$$

**算 roofline 拐点和 MFU 时必须用 dense 值**——这一步搞反，所有结论都会偏一倍。这是硬件面试里最经典的"数字陷阱"。

### 3. 具体数值样例

```text
【样例 A：一条 MMA 指令 vs 一个 CUDA Core 指令】
  CUDA Core 一条 FMA：2 FLOP
  Tensor Core 一条 m16n8k16 BF16 MMA：2 × 16 × 8 × 16 = 4096 FLOP
  ⇒ 单指令计算量差 2048 倍
  但一条 MMA 要整个 warp（32 线程）协作、耗时若干 cycle，
  所以真实吞吐差距是"每 SM 每 cycle 的 FLOP"之比，不是 2048 倍

【样例 B：H100 的 BF16 Tensor Core 峰值是怎么来的（推算）】
  官方：BF16 dense 989 TFLOPS（sparse 1979）
  132 SM × 4 TC/SM = 528 个 Tensor Core
  989e12 / 528 = 1.873e12 FLOP/s per TC
  若频率 ~1.83 GHz：1873/1.83 ≈ 1024 FLOP/TC/cycle
  ⇒ 每个 Tensor Core 每 cycle 完成 1024 个 BF16 FLOP（= 512 FMA）
  对照 FP32：每 SM 每 cycle 128 lane × 2 = 256 FLOP
  比值（同 SM）：4 TC × 1024 / 256 = 16×
  ⇒ 这就是"BF16 TC 是 FP32 的 ~15 倍"的来源：
    4 个 Tensor Core，每个的矩阵吞吐是 128 个 FP32 lane 总和的 4 倍

【样例 C：一个 4096³ 的 BF16 GEMM 要多久（理想）】
  计算量 = 2 × 4096³ = 1.374e11 FLOP
  单卡 H100 BF16 dense = 989e12 FLOP/s
  理想时间 = 1.374e11 / 989e12 = 0.139 ms
  实测通常 60%~80% 峰值 → 0.17~0.23 ms
  （参考文献锚点：DeepGEMM 在 H800 上 FP8 GEMM 达到 1550 TFLOPS，
    约为 H100/H800 FP8 dense peak 1979 TFLOPS 的 78%）

【样例 D：为什么 decode 用不满 Tensor Core】
  decode 时 batch=1：GEMM 的 M 维 = 1
  MMA 最小 m=16 → 利用率最多 1/16 = 6.25%（还没算 n/k 维）
  ⇒ 所以 decode 的优化方向是"增大 batch（continuous batching）"
    或者"绕过 Tensor Core 用 GEMV 优化"，而不是"优化 MMA"
```

### 4. 演进：四代 Tensor Core 与"精确度降级换吞吐"

| 代际 | 架构 | 新增精度 | 关键变化 |
|---|---|---|---|
| 1st | Volta（V100） | FP16 | 首次引入 Tensor Core |
| 2nd | Turing（T4/2080） | INT8/INT4/Binary | 面向推理量化；每 TC 每 clock 64 FP16 FMA |
| 3rd | Ampere（A100） | **TF32、BF16、FP64** | 引入 **2:4 结构化稀疏**；"FP32 精度、FP16 速度"的 TF32 让 AMP 更顺手 |
| 4th | Hopper（H100） | **FP8**（E4M3/E5M2） | 配合 **Transformer Engine** 做逐层动态 FP8/FP16 混合；新增 TMA；官方称 FP8 训练比 Ampere FP16 快 6× |
| 5th | Blackwell（B200） | **FP6、FP4（NVFP4）** | 双 die 封装；第二代 Transformer Engine；官方称能效是 Hopper 的 25× |

**趋势**：每一代都不靠"更大的 SM"提吞吐，而是靠**降低数值精度 + 扩大矩阵块**。这带来两个面试可延伸的话题：① **数值精度与收敛的关系**（FP8 需要 per-tensor/per-block scaling，BF16 不需要）；② **量化的收益边界**（当 kernel 已经 memory-bound 时，把 FP16 换成 FP8 只能省带宽，不能让算力"变快"——回到第 8 节的 roofline）。

> **面试一句话总结**：Tensor Core 是一条**矩阵级指令**（MMA，$D_{m\times n}=A_{m\times k}B_{k\times n}+C$，一条 `m16n8k16` BF16 指令 = 4096 FLOP，而 CUDA Core 一条 FMA 只有 2 FLOP），快的根本原因是**操作数复用 $O(k)$ + 二维乘加阵列 + 寄存器内累加**，代价是**整个 warp 按固定 layout 协作**、**只支持固定的 $m,n,k$ 组合（所以要 padding，decode 的 M=1 利用率极低）**、**精度受限（每代新增 FP16→INT8→TF32/BF16→FP8→FP6/FP4，累加仍为 FP32）**；另外务必记住 **2:4 结构化稀疏让官方峰值默认是 sparse 口径，算 roofline/MFU 必须除以 2 换成 dense**（H100 BF16 是 989 dense / 1979 sparse）。

---

## 5. 存储层次：寄存器 / Shared+L1 / L2 / HBM

### 1. 现有问题：为什么 kernel 优化永远先看访存

"GPU 存储层次"如果只答"寄存器最快、显存最慢"是拿不到分的。面试官真正想确认的是三件事：① 每一层的**容量、带宽、延迟**大概是什么量级；② 数据在层与层之间搬运时**效率由什么决定**（这引出 coalescing 与 bank conflict）；③ 拿到一个慢 kernel，**怎么判断它卡在哪一层**。

### 2. 方法论：四层结构 + 两个效率规则

**（1）四层结构（含片上与片外）。** 以 H100 为例（容量/带宽/延迟数字来源：Triton 教材第 8 章，见文首来源）：

| 层次 | 容量 | 带宽（量级） | 延迟（cycles） | 可见范围 | 谁管理 |
|---|---|---|---|---|---|
| **寄存器堆（Register File）** | 256 KB / SM | ~16 TB/s（每 SM） | ~0（无延迟） | 线程私有 | 编译器（寄存器分配） |
| **Shared Memory / L1** | 228 KB / SM（可配置切分） | ~4 TB/s（每 SM） | ~20–30 | **Block 内共享**（L1 是自动缓存） | 程序员显式管理 SMEM；L1 硬件管理 |
| **L2 Cache** | 50 MB（全芯片） | ~12 TB/s | ~200 | 全 GPU 共享 | 硬件 |
| **HBM（显存）** | 80 GB | 3.35 TB/s | ~500–800 | 全 GPU + 可被 NVLink/RDMA 访问 | 硬件 + 驱动 |

**（2）关键认知一：延迟差了两个数量级，带宽只差几倍。** 从寄存器到 HBM，延迟放大约 500~800 倍，但带宽只降几倍。这说明：

- **不是"显存慢"，而是"显存远"**——所以要靠**大量并发线程**掩盖（第 7 节），靠 **SMEM 复用**减少访问次数（GEMM 的 tiling），靠**异步拷贝**（`cp.async`/TMA）把搬运与计算重叠；
- **L2 的带宽（~12 TB/s）远高于 HBM（3.35 TB/s）**——所以"提高 L2 命中率"（数据复用、分块、持久化 L2 窗口）是实打实的收益，这也是 FlashAttention、fused kernel 有效的硬件原因。

**（3）关键认知二：访存效率由两条规则决定。**

**规则 A：合并访问（Coalesced Access）。** 一个 warp 的 32 个线程在同一条访存指令里，如果访问的是**连续地址**，硬件可以把它们合并成少数几个 **cache line 粒度的事务**（32B/64B/128B），带宽利用率接近 100%。如果线程访问的是**跨步（strided）或随机**地址，每个线程可能触发独立事务，有效带宽骤降。

$$
\text{事务数} = \left\lceil \frac{\text{warp 覆盖的地址跨度}}{128\text{B}} \right\rceil \quad (\text{理想情况下，每线程 4B 连续})
$$

**规则 B：Bank Conflict（仅 Shared Memory）。** Shared Memory 被组织为 **32 个 bank，每个 bank 宽 4 字节**。同一 warp 内，如果多个线程访问**同一个 bank 的不同地址**，访问会被**序列化**：N 个线程冲突到同一 bank 就是 **N-way bank conflict，带宽降到 1/N**。经典的反例是**按列访问一个按行存储的二维数组**（矩阵转置）；标准解法是 **padding**（把 `[32][32]` 改成 `[32][33]`）或者 **swizzle**（对地址做 XOR 置换，让相邻线程落到不同 bank）。

**（4）把三条优化路径钉在层次上**（面试可以直接这样组织回答）：

| 优化手段 | 作用在哪一层 | 原理 |
|---|---|---|
| Tiling / 分块 | HBM → SMEM | 把数据搬进 SMEM 复用，减少 HBM 访问次数（提高算术强度） |
| 合并访问 / 向量化（float4） | 寄存器 ↔ L1/HBM | 减少事务数，提高每事务有效字节 |
| Padding / Swizzle | SMEM 内部 | 消除 bank conflict |
| Fusion（算子融合） | HBM 访问次数 | 中间结果不进 HBM，直接留在寄存器/SMEM |
| `cp.async` / TMA | HBM → SMEM | 搬运与计算重叠（不占寄存器、可流水线） |
| L2 persistence | L2 | 把热点数据钉在 L2 |

### 3. 具体数值样例

```text
【样例 A：合并访问的带宽差异（同一份数据，两种访问模式）】
  32 个线程，每线程读 4 字节（FP32）
  ① 连续：线程 i 访问 base + 4i → 覆盖 128 字节 = 1 个 128B cache line
     事务数 = 1，有效带宽 = 100%
  ② 跨步 32（stride = 32 × 4B = 128B）：线程 i 访问 base + 128i
     每个线程落在不同的 128B cache line → 32 个事务，每个只用到 4/128 = 3.1%
     有效带宽 ≈ 3.1%（理论最差）
  ⇒ 这就是"矩阵按列访问比按行访问慢 10 倍以上"的原因

【样例 B：bank conflict 的算术（32×32 FP32 矩阵转置）】
  SMEM 声明为 smem[32][32]，按行主序存
  warp 里线程 (r, c) → 读 smem[c][r]（转置读）
    地址 = c × 32 + r（单位：4B）
    bank = 地址 mod 32 = (c × 32 + r) mod 32 = r mod 32
    同一 warp 里若 r 固定、c 变化 → 32 个线程的 bank 全都是 r → 32-way conflict！
    序列化 32 次 → 有效带宽 1/32
  改成 smem[32][33]（padding 一列）：
    地址 = c × 33 + r；bank = (33c + r) mod 32 = (c + r) mod 32
    c 变化时 bank 依次错开 → 无冲突
  ⇒ 一行 padding 换来 32× 的 SMEM 吞吐，这是最经典的 SMEM 优化

【样例 C：算一次 attention 的 HBM 流量（解释 FlashAttention 为什么快）】
  朴素实现（每个 head、每层都写回中间矩阵 S = QK^T 和 P = softmax(S)）：
    读写量 ≈ O(seq²) 级别的中间矩阵
  设 seq = 4096，head_dim = 128，FP16：
    S 的大小 = 4096 × 4096 × 2B = 33.5 MB（单个 head！）
    多层多头累加 → 远超前向本身需要的 Q/K/V/O 读写量
  FlashAttention：把 Q/K/V 分块搬进 SMEM，在片上算完 softmax 再写 O
    HBM 流量从 O(seq²) 降到 O(seq × head_dim)
  ⇒ 这就是"同一硬件、换 kernel、训练时间 -38%"的硬件解释
```

### 4. 演进：容量在涨，但"搬"的方式在变

- **L2 容量**：A100 40 MB → H100 50 MB → B200 126 MB（双 die 合计）；L2 变大直接提高复用类算子的命中率；
- **SMEM 可配置比例**：从固定切分变成"按 kernel 需求在 L1 与 SMEM 之间重新划分"（Hopper 上限 228 KB），让 GEMM 能开更大的 tile；
- **TMA 改变搬运路径**：传统路径是 `Global → 寄存器 → SMEM`，TMA 让 `Global → SMEM` 成为**一条指令的异步大块拷贝**，SM 的寄存器和 LD/ST 被解放出来做计算——这是 Hopper 上 warp specialization（生产者 warp 管搬运、消费者 warp 管 MMA）的基础；
- **分布式 Shared Memory（DSMEM）**：Hopper 的 Thread Block Cluster 让一个 cluster 内的多个 SM 能互相访问 SMEM，等于把 SMEM 当成"SM 间的小型共享缓存"，减少走 L2/HBM 的往返。

> **面试一句话总结**：GPU 存储层次是**寄存器堆（256 KB/SM，延迟 ~0）→ Shared Memory/L1（228 KB/SM 可配，延迟 20–30 cycle）→ L2（50 MB，~200 cycle，带宽 ~12 TB/s）→ HBM（80 GB，~500–800 cycle，3.35 TB/s）**，延迟跨两个数量级而带宽只差几倍，所以优化永远是"**减少搬运次数 + 掩盖延迟**"；两条效率规则必须背下来——**合并访问**（warp 32 线程访问连续地址时合并成 128B 级事务，跨步访问会掉到几十分之一）与 **bank conflict**（SMEM 32 个 bank × 4B，N 路冲突 → 带宽 1/N，用 padding `[32][33]` 或 swizzle 解决）；三条常见优化路径（Tiling、Fusion、`cp.async`/TMA）分别作用在"HBM→SMEM 复用""中间结果不落 HBM""搬运与计算重叠"上，而 FlashAttention 的本质就是**把 O(seq²) 的 HBM 流量降到 O(seq×head_dim)**。

---

## 6. HBM：显存容量与带宽，以及"为什么它是第一瓶颈"

### 1. 现有问题：为什么大家都在喊"显存墙"

训练和推理都会遇到"显存不够/带宽不够"，但这是**两个不同的墙**：容量墙（$M$）决定"能不能装下"，带宽墙（$\beta$）决定"跑得多快"。面试里问"70B 模型要几张卡"（容量）和"decode 为什么只有几十 token/s"（带宽）是两道完全不同的题，混答就废了。

### 2. 方法论：HBM 是什么、带宽怎么算、容量怎么分

**（1）HBM 是什么。** HBM（High Bandwidth Memory）用 **3D 堆叠 + TSV（硅通孔）+ 硅中介层（Interposer）** 把多颗 DRAM die 垂直堆起来，再用**极宽但相对低速**的并行总线接到 GPU：

$$
\text{HBM 带宽} = \frac{\text{总线位宽（bit）}}{8} \times \text{每 pin 数据率（Gbps)}
$$

关键在于"**用位宽换速率**"：GDDR 靠高频率（单颗 32-bit 宽、速率高），HBM 靠超宽位宽（**每 stack 1024-bit**，H100 共 5120-bit），因此同等带宽下**功耗显著更低**——这是数据中心卡用 HBM、消费卡用 GDDR 的根本原因。

**（2）代际演进**：HBM2（V100）→ HBM2e（A100、**昇腾 910B3**）→ HBM3（H100）→ HBM3e（H200、B200）。每代主要抬**带宽与容量**；而计算架构（Volta→Ampere→Hopper→Blackwell）是**另一条独立的演进线**——所以会出现 **H200 这种"算力不变、只换显存"** 的 SKU（见第 9 节）。

**（3）容量墙：显存里的账怎么分。**

$$
M_{\text{训练}} \approx \underbrace{N \cdot s_{\text{param}}}_{\text{参数}} + \underbrace{N \cdot s_{\text{optim}}}_{\text{优化器}} + \underbrace{N \cdot s_{\text{grad}}}_{\text{梯度}} + \underbrace{\text{activations}}_{\text{激活}}
$$

$$
M_{\text{推理}} \approx \underbrace{N \cdot s_{\text{param}}}_{\text{权重}} + \underbrace{2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot B \cdot s}_{\text{KV cache}}
$$

其中 $N$ 是参数量、$s$ 是每参数字节数、$L$ 是层数、$H_{kv}$ 是 KV 头数（GQA 后远小于注意力头数）、$S$ 是序列长度、$B$ 是并发数。**注意 KV cache 对 $S$ 和 $B$ 都是线性的**——这是长上下文与高并发推理最贵的部分。

**（4）带宽墙：decode 的天花板由 $\beta / M_{\text{weights}}$ 决定。** 生成一个 token 需要把**整份权重扫一遍**（batch=1 时），所以：

$$
\text{tokens/s 上限} \approx \frac{\beta_{\text{hbm}}}{M_{\text{weights}}}
$$

这条公式极其重要：它说明**decode 的瓶颈不是算力而是显存带宽**，也说明为什么"量化权重"能直接提速 decode（$M_{\text{weights}}$ 变小）、为什么"H200 只加带宽就能让长上下文推理更快"。

### 3. 具体数值样例

```text
【样例 A：验证 A100 的显存带宽（推算，对照官方 2.04 TB/s）】
  A100 80GB：HBM2e、总线 5120-bit、每 pin 约 3.2 Gbps（HBM2e 规格档）
  (5120 / 8) B × 3.2e9 /s = 640 × 3.2e9 = 2.048e12 B/s = 2.048 TB/s ✓
  反推 H100：3.35 TB/s ÷ 640 B = 5.23e9 → 每 pin 约 5.2 Gbps（HBM3 位宽同为 5120-bit）
  反推 H200：4.8 TB/s ÷ (6144/8 = 768 B) = 6.25 Gbps（HBM3e，6144-bit）
  反推 B200：7.7~8.0 TB/s ÷ (8192/8 = 1024 B) ≈ 7.5~7.8 Gbps（HBM3e，8192-bit）
  ⇒ 面试里能"从位宽和数据率推带宽"，就不需要死记数字

【样例 B：容量墙——70B 模型要几张卡（纯权重，不算 KV/激活）】
  70B，BF16（2 B/参数）：70e9 × 2 = 140 GB
    H100 80GB → 装不下，至少 2 张（TP=2）才有位置放 KV 和激活
    H200 141GB → 刚好装权重，几乎没有 KV 余量 → 实际必须量化（FP8 = 70 GB）
  70B，训练态（BF16 参数 + BF16 梯度 + Adam fp32 两动量 + fp32 master）：
    2 + 2 + 8 + 4 = 16 B/参数 → 1.12 TB → 至少 14~16 张 H100（还要加激活）
  ⇒ 这就是 ZeRO/FSDP 存在的理由：把优化器状态切到多卡上

【样例 C：带宽墙——70B BF16 单卡 decode 的理论上限】
  权重 140 GB，H100 带宽 3.35 TB/s
  tokens/s 上限 ≈ 3350 / 140 ≈ 23.9 → 约 24 token/s（batch=1，理想值）
  实际还能更低（KV cache 读取、kernel 效率、采样开销）
  若权重换 FP8（70 GB）：3350 / 70 ≈ 47.9 → 直接翻倍
  ⇒ 这就是"量化能提速 decode"的算术证明，与算力无关

【样例 D：KV cache 有多大（决定长上下文能不能开）】
  Llama-3-70B 量级：L = 80 层，GQA 后 H_kv = 8，d_head = 128，BF16（2 B）
  单序列 8K 上下文：
    2 (K,V) × 80 × 8 × 128 × 8192 × 2 B
    = 2 × 80 × 8 × 128 = 163,840（每 token 的 KV 元素数）
    × 8192 = 1.342e9 元素
    × 2 B = 2.68e9 B ≈ 2.68 GB / 序列
  并发 32 条 → 85.9 GB 的 KV cache！
  ⇒ 这就是为什么"长上下文 + 高并发"必须靠 GQA / PagedAttention / KV 量化 / 前缀复用
```

### 4. 演进：HBM 与"内存墙"的三个趋势

- **容量与带宽代际**：HBM2 → HBM2e → HBM3 → HBM3e，H200 用 141 GB/4.8 TB/s、B200 用 ~180-192 GB/8 TB/s；GB200 NVL72 整柜 13.4 TB HBM3e、聚合带宽 576 TB/s（官方数据）；
- **算力比带宽涨得快**：A100→H100，BF16 算力涨 3.2×（312→989 TFLOPS）而带宽只涨 1.6×（2.04→3.35 TB/s）→ **roofline 拐点右移**（153→295 FLOPs/byte，见第 8 节），越来越多算子掉到"带宽受限"一侧；
- **H200 这类"带宽特供"SKU 出现**：算力不变、只把 $\beta$ 从 3.35 拉到 4.8 TB/s（拐点从 295 退回 206），直接受益的正是 decode 与长上下文 KV 读取。

> **面试一句话总结**：HBM 用"3D 堆叠 + 硅中介层 + 超宽总线（每 stack 1024-bit，H100 共 5120-bit）"换来高带宽低功耗，带宽公式是 **(位宽/8) × 每 pin 数据率**（A100：5120/8 × 3.2 Gbps = 2.048 TB/s，可当场验证官方 2.04 TB/s）；显存有两堵**不同**的墙——**容量墙**（训练要装参数+梯度+优化器，70B 训练态约 16 B/参数 = 1.12 TB；推理要装权重+KV cache，KV 对序列长与并发数都是线性的：Llama-70B 单序列 8K 就要 2.68 GB）与**带宽墙**（decode 每 token 要扫一遍全部权重，所以 tokens/s 上限 ≈ $\beta / M_{\text{weights}}$，70B BF16 在 H100 上理论约 24 token/s，换 FP8 直接翻倍）；HBM 代际是 HBM2→HBM2e→HBM3→HBM3e，而"算力比带宽涨得快"导致拐点持续右移，这正是 fusion/量化/FlashAttention 这些省带宽手段value 越来越高的根本原因。

---

## 7. SIMT 执行模型：warp、分支发散与 occupancy

### 1. 现有问题：为什么"线程越多越好"是错的，为什么 if/else 会变慢

CUDA 编程模型把线程分成 Grid → Block → Thread 三层，硬件执行时又按 **warp（32 线程）** 锁步推进。面试里围绕这一层的三个必问点是：① **为什么 Block 不能跨 SM**；② **warp divergence 的代价是什么、怎么量化**；③ **occupancy 是什么、越高越好吗**。

### 2. 方法论：三级映射、SIMT 与两个权衡

**（1）三级映射与"为什么 Block 不能跨 SM"。** Grid 是一维到三维的 Block 集合，Block 是一维到三维的 Thread 集合。硬件把**整个 Block 调度到一个 SM 上**（Block 内的线程可以通过 `__syncthreads()` 和 SMEM 通信，所以必须同 SM）。这就是"Block 不能跨 SM 拆"的根本原因——**不是硬件做不到，而是语义要求**：SMEM 和 barrier 只在一个 SM 内有效。

**（2）SIMT 与 SIMD 的区别。** SIMD 是"一条指令作用在**一个寄存器向量**上"（程序员显式写向量操作）；SIMT 是"一条指令作用在**32 个线程各自的寄存器**上"（程序员写标量代码，硬件把 32 个线程的标量合并成一次宽发射）。SIMT 的好处是**编程模型简单**（不用手写向量化），代价是**当 32 个线程想走不同分支时要串行化**。

**（3）Warp Divergence（分支发散）的代价。** 如果一个 warp 里的线程走到不同的分支，硬件**无法同时执行两条路径**，只能：

$$
\text{发散后的时间} = t_{\text{path A}} + t_{\text{path B}} \quad (\text{两条路径依次执行，各自只激活一部分 lane})
$$

也就是说**发散让"并行度换成了时间"**：32 个线程里 16/16 分叉，两次串行后总时间 ≈ 2 倍，平均 lane 利用率 50%。推论：

- `if (threadIdx.x < 16) ... else ...` 这种**按 lane 边界对齐的分支**是好的（一个 warp 内不分裂，或只分裂两侧都很小）；
- **数据相关的分支**（比如"按 token 长度决定是否循环"）在同一个 warp 里会严重发散——**这就是为什么变长序列要在 batch 维度上尽量把长度接近的样本放在一起**（seqlen 分桶）。
- **Volta 起引入 Independent Thread Scheduling**：warp 内的线程可以有独立的程序计数器，硬件能在更细粒度上交错两条路径，缓解（但**没有消除**）发散代价。

**（4）Occupancy（占用率）与两个权衡。** Occupancy 定义为

$$
\text{Occupancy} = \frac{\text{SM 上实际驻留的 warp 数}}{\text{SM 支持的最大 warp 数}}
$$

它由四个资源约束共同决定（取最严格的那个）：**寄存器/线程、SMEM/block、Block 数上限、warp 数上限**。Hopper 的上限是 **64 warp/SM（2048 线程）**。

**关键结论（面试高频）**：**occupancy 高 ≠ 快**。原因是延迟隐藏有两个来源：

| 来源 | 说明 | 影响 |
|---|---|---|
| **TLP**（线程级并行） | 更多 warp 轮流发射 | occupancy 高 → TLP 好 |
| **ILP**（指令级并行） | 单个 warp 内部有多条独立指令可发射 | 寄存器多 → ILP 好，但会压低 occupancy |

所以"用更多寄存器换 ILP"和"用更少寄存器换 occupancy"是一次**真实的取舍**，最优解取决于 kernel 是访存密集（偏好 TLP/occupancy）还是计算密集（偏好 ILP/低 occupancy）。实践中常用 `__launch_bounds__`、`maxrregcount`、`cudaOccupancyMaxPotentialBlockSize` 来调。

### 3. 具体数值样例

```text
【样例 A：分支发散的量化代价】
  warp 32 线程，执行：
    if (x[i] > 0) { A（10 条指令） } else { B（30 条指令） }
  假设 warp 内 16 个线程走 A、16 个走 B：
    硬件串行执行 A（只有 16 lane 有效）→ 再执行 B（另外 16 lane 有效）
    总耗时 ≈ 10 + 30 = 40 个指令时隙
    若所有线程走同一条路径（取较长的 B）：30 个时隙
  ⇒ 发散让这个 warp 多花 33% 的时间，且两个分支的有效 lane 利用率都只有 50%

【样例 B：occupancy 的完整计算（Hopper SM）】
  约束 1（warp 上限）：64 warp/SM = 2048 线程
  约束 2（寄存器）：65536 寄存器/SM
  约束 3（SMEM）：最大 228 KB/SM
  约束 4（Block 数）：硬件上限（Hopper 每 SM 最多 32 个 Block）

  设 kernel：block = 256 线程（8 warp）、每线程 40 寄存器、每 block 用 32 KB SMEM
    寄存器维度：65536 / (256 × 40 = 10240) = 6.4 → 6 个 block
    SMEM 维度：228 / 32 = 7.1 → 7 个 block
    Block 维度：32
    warp 维度：64 / 8 = 8 个 block
    取最小值 = 6 个 block → 1536 线程 = 48 warp
    Occupancy = 48 / 64 = 75%
  若把每线程寄存器压到 32：65536/(256×32=8192) = 8 个 block → 与 warp 上限持平
    Occupancy = 64/64 = 100%（但可能因寄存器不足产生 spill，反而变慢）

【样例 C：为什么"Block 不能跨 SM"能被观察到】
  一个 block 用 __syncthreads() 做 block 内同步
  如果硬件把 block 拆到两个 SM，__syncthreads 就无法实现（barrier 是 SM 级的）
  ⇒ 所以 block 大小直接决定"最小调度粒度"：
    block 太小 → 调度开销占比高、SMEM/寄存器分配浪费
    block 太大 → 一个 SM 只能放一个，尾部效应（tail effect）严重
    经验值：128~512 线程/block
```

### 4. 演进：从"锁步"到"warp 内独立调度"再到"warp 专用化"

- **Volta：Independent Thread Scheduling**——warp 内线程有独立 PC，配合 `__syncwarp()` 做 warp 级同步，缓解发散（但寄存器与调度是按 warp 分配的，所以并没有变成"32 个独立线程"）；
- **Hopper：Thread Block Cluster + warp specialization**——允许把不同 warp 分工成"生产者（搬数据）"和"消费者（算 MMA）"，配合 TMA 与 `mbarrier` 做流水线，这是现代 GEMM/Attention kernel 的标准结构；
- **正则化实践**：现代框架（Triton/Inductor）把"选择 block size、num_warps、num_stages"当作**自动调参**问题，正是因为 occupancy/TLP/ILP 的最优点依赖 kernel。

> **面试一句话总结**：CUDA 的三级模型是 Grid→Block→Thread，硬件执行时按 **warp（32 线程）锁步**推进，**Block 必须整个调度到一个 SM**（因为 SMEM 与 `__syncthreads` 是 SM 级语义，不是硬件限制而是语义要求）；SIMT 让程序员写标量代码、硬件合并宽发射，代价是 **warp divergence**——同一 warp 走不同分支时要串行执行两条路径，$\text{时间} = t_A + t_B$、lane 利用率减半，所以变长序列要分桶、Volta 后的 Independent Thread Scheduling 只是缓解而非消除；**occupancy = 驻留 warp / 最大 warp（Hopper 上限 64 warp/SM = 2048 线程）**，由寄存器、SMEM、Block 数、warp 上限四者取最严格约束决定，但**occupancy 高不等于快**——它只代表 TLP 好，真正的延迟隐藏还依赖 ILP，二者通过寄存器用量此消彼长（示例：256 线程/block、40 寄存器/线程、32 KB SMEM → 6 个 block → 75% occupancy）。

---

## 8. Roofline 与 MFU：怎么判断"算力到底打满没有"

### 1. 现有问题：MFU 只有 30%，是算力问题还是带宽问题

这是 AI Infra 面试的**分水岭题**。答"看看 nvidia-smi 利用率"是错的——`nvidia-smi` 显示的 GPU 利用率只是"有 kernel 在跑"的时间占比，**98% 利用率 + 28% MFU 是完全正常的组合**（来源：Roofline 章节开篇的 2AM debug 故事）。要正确回答"瓶颈在哪"，需要 roofline 模型。

### 2. 方法论：算术强度、两道天花板、拐点、MFU

**（1）算术强度（Arithmetic Intensity）。**

$$
I = \frac{\text{FLOPs}}{\text{Bytes accessed (DRAM)}}
$$

单位是 FLOPs/byte。**分母只算 DRAM（HBM）流量**——从 L2/SMEM 复用来的数据不计入，这正是"提高复用 = 提高算术强度"的原因。

**（2）两道天花板与拐点。** 硬件给两个上限：峰值算力 $\pi$（FLOP/s）与峰值带宽 $\beta$（B/s）。于是

$$
P_{\text{可达}} = \min\left(\pi,\; I \times \beta\right)
$$

- 当 $I \times \beta < \pi$：**memory-bound**，性能 = 算术强度 × 带宽（对数坐标下是斜线）；
- 当 $I \times \beta > \pi$：**compute-bound**，性能 = 峰值算力（水平线）；
- 二者交点就是**拐点（ridge point）**：

$$
I^* = \frac{\pi}{\beta}
$$

**（3）常见算子的算术强度**（这是面试里最值得背的一张"直觉表"）：

| 算子 | FLOPs | DRAM 字节 | $I$（FLOPs/byte） | 判定 |
|---|---|---|---|---|
| 逐元素（ReLU/GELU/add） | ~1 / 元素 | 4 读 + 4 写（FP32）= 8 B | **≈ 0.125** | 极度 memory-bound |
| LayerNorm | ~7H / token | 8H B | **≈ 0.875** | memory-bound |
| 向量点积 | 2N | 8N B | **≈ 0.25** | memory-bound |
| GEMV（矩阵×向量） | 2MN | ≈ 4MN B | **≈ 0.5** | memory-bound（decode 的主形态） |
| GEMM（$N \times N$ 方阵） | $2N^3$ | $12N^2$ B（FP32 三份） | **≈ N/6** | $N=4096$ → **683**，compute-bound |

**（4）拐点表（用官方参数推算，单位 FLOPs/byte）**：

| 卡 | $\pi$（BF16 dense） | $\beta_{\text{hbm}}$ | $I^* = \pi/\beta$ |
|---|---|---|---|
| A100 80GB | 312 TFLOPS | 2.04 TB/s | **≈ 153** |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | **≈ 295** |
| H200 SXM | 989 TFLOPS | 4.8 TB/s | **≈ 206** |
| B200 SXM | ~2500 TFLOPS | 7.7 TB/s | **≈ 325** |

**读法**：拐点在 150~330 FLOPs/byte 之间。**逐元素算子（0.125）差了三个数量级 → 永远 memory-bound**；**大 GEMM（683）在 A100 上刚过拐点、在 H100/B200 上仍然过**；**GEMV（0.5）永远 memory-bound**——这解释了为什么 decode 优化靠"增大 batch"和"省带宽"，而不是靠"换更强的 Tensor Core"。

**（5）MFU（Model FLOPs Utilization）。**

$$
\text{MFU} = \frac{\text{实测达到的 FLOPs/s}}{\text{硬件峰值 dense FLOPS} \times \text{卡数}}
$$

三个必须注意的口径：① **分母必须用 dense 峰值**（sparse 是 2×，会把 MFU 算低一半）；② **分子要用"模型所需的最小 FLOPs"**（训练常用 $6ND$：前向 $2ND$ + 反向 $4ND$，$N$ 参数量、$D$ token 数）；③ **MFU 高不等于训练快**（一个 MFU 低的实现可能因为算法更省 FLOPs 而整体更快，例如 MoE 的稀疏激活）。

### 3. 具体数值样例

```text
【样例 A：一次训练 step 的 MFU 怎么算（7B 模型，8×H100）】
  设一步处理 D = 4M（4×10^6）tokens，N = 7×10^9 参数
  训练 FLOPs ≈ 6 N D = 6 × 7e9 × 4e6 = 1.68e17 FLOP
  8×H100 BF16 dense 峰值 = 8 × 989e12 = 7.912e15 FLOP/s
  理想时间 = 1.68e17 / 7.912e15 = 21.2 s
  若实测一步 42 s → MFU = 21.2 / 42 = 50.5%
  若实测一步 70 s → MFU = 30.3%
  ⇒ 说"MFU 30%"时必须同时给 N、D、卡数、dense 峰值，否则无法核对

【样例 B：用算术强度判断一个算子该往哪优化】
  某 elementwise kernel 实测 200 GB/s（H100 峰值 3350 GB/s，利用率仅 6%）
    它的 I ≈ 0.125，远低于拐点 295 → 一定 memory-bound
    ⇒ 优化方向：减少 DRAM 访问（融合进前后算子、用 FP16 存中间结果、向量化 float4）
    ⇒ 不要去"优化计算"（加更多 ALU 无意义）
  某 GEMM 实测 500 TFLOPS（989 的 50%）
    它的 I ≈ 683 > 295 → compute-bound
    ⇒ 优化方向：更大的 tile（提高 SMEM 复用）、warp specialization、更大 K 维、
      或换 FP8（π 翻倍）
    ⇒ 减少 HBM 访问的收益有限（已经不是瓶颈）

【样例 C：为什么"升级到 H100 却没变快"（Roofline 章节的真实案例）】
  A100 → H100，BF16 算力 312 → 989 TFLOPS（3.2×），带宽 2.04 → 3.35 TB/s（1.6×）
  如果 kernel 的 I = 150（在 A100 上刚好是 compute-bound 边缘）：
    A100：min(312, 150 × 2.04 = 306) → 306 TFLOPS（贴着拐点）
    H100：min(989, 150 × 3.35 = 502) → 502 TFLOPS（memory-bound！）
  ⇒ 升级后只快了 1.64×，而不是算力的 3.2×
  ⇒ 因为拐点从 153 右移到 295，原本 compute-bound 的算子变成 memory-bound
  ⇒ 正确动作是"减少 HBM 流量"（FlashAttention、fusion、量化），而不是继续堆算力
```

### 4. 演进：为什么 roofline 越来越重要

- **算力增速 > 带宽增速**：A100→H100 是 3.2× vs 1.6×，拐点从 153 推到 295；Blackwell 继续在 ~300 量级 → **"memory-bound 的算子集合在持续扩大"**；
- **工具已经内置**：NVIDIA 从 2019 年起把 roofline 分析集成进 **Nsight Compute**，`ncu --set full` 会直接给出"这个 kernel 距 roofline 有多远"；
- **从 GPU 推广到通信**：同样的"两道天花板"思想可以推广成**通信 roofline**（$\alpha$-$\beta$ 模型、busbw 与峰值之比）——这就是 `Communication.md` 与 `Parallel.md` 里集合通信分析的数学基础；
- **注意 roofline 的局限**：它是**上界模型**，不考虑 L2 命中、访存模式、发散、kernel 启动开销；"实测远低于 roofline"只是说明"还有空间"，不说明"空间在哪"——定位仍需 profiling。

> **面试一句话总结**：判断"算力有没有打满"要用 **roofline** 而不是 `nvidia-smi` 利用率：算术强度 $I = \text{FLOPs}/\text{DRAM 字节}$，性能上界 $P = \min(\pi, I\beta)$，拐点 $I^* = \pi/\beta$（A100≈153、H100≈295、H200≈206、B200≈325 FLOPs/byte）；常见算子的 $I$ 必须记住——**逐元素 0.125、LayerNorm 0.875、GEMV 0.5、$N$ 阶 GEMM ≈ $N/6$（4096 时 683）**，所以**逐元素/GEMV 永远 memory-bound（优化方向是省带宽/加大 batch），大 GEMM 才是 compute-bound（优化方向是更大的 tile/FP8）**；**MFU = 实测 FLOPs/s ÷（dense 峰值 × 卡数）**，分母务必用 dense（sparse 会差 2×）、分子按 $6ND$ 估（示例：7B × 4M tokens × 8×H100 → 理想 21.2 s，实测 42 s 则 MFU≈50%）；最后记住那个经典反例——**A100 升 H100 只快了 1.6× 而不是 3.2×，因为拐点右移让原本 compute-bound 的算子掉到了 memory-bound 一侧**，此时正确动作是减少 HBM 流量而不是继续堆算力。

---

## 9. 主流型号对照与选型：五个量与五条铁律

### 1. 现有问题：面试问"这块卡什么水平/你选什么卡"

"说说 A100 和 H100 的区别"是硬件题的收尾必问。硬背参数容易记混、也容易被追问细节（"H800 和 H100 差在哪""H100 PCIe 和 SXM 一样吗"）。正确的做法是**只记五个量 + 五条读表铁律**，其余都能推。

### 2. 方法论：五本账 + 五条铁律 + 一个选型框架

**（1）一块卡先看五本账**（这是把硬件讲清楚的"索引"）：

| 量 | 单位 | 决定什么 |
|---|---|---|
| $\pi$ 峰值算力 | FLOP/s（**按精度、按 dense**） | compute-bound 算子的上界（第 8 节水平屋顶） |
| $M$ 显存容量 | GB | 单卡能放多少参数 / KV cache / 激活（第 6 节容量墙） |
| $\beta_{\text{hbm}}$ 显存带宽 | TB/s | memory-bound 算子的上界（第 8 节斜线屋顶） |
| $\beta_{\text{nvlink}}$ scale-up 互连 | GB/s | 机内 TP/EP 的通信屋顶（详见 `Communication.md`） |
| TDP | W | 功率密度、是否必须液冷、机柜能塞几张 |

**（2）读 datasheet 的五条铁律**（每一条都能救命）：

1. **sparse 先折半**：官方 Tensor Core 峰值默认是 2:4 sparse 口径，dense = 一半；算 roofline/MFU 必须用 dense；
2. **bit ≠ byte**：差 8 倍。`400 Gb/s` 的 IB 网卡 ≈ **50 GB/s** 单向；
3. **NVLink 官方数字默认双向合计**：H100 的 900 GB/s 是双向，单向约 450 GB/s；和 HBM、RDMA 比较时要统一方向；
4. **SXM ≠ PCIe ≠ NVL**：同一个芯片名，板型不同则带宽/功耗/算力都可能不同（H100 SXM 3.35 TB/s vs H100 PCIe 2.0 TB/s，差 1.7×）；
5. **出口 SKU 通常优先砍互连**：A800 相对 A100 只把 NVLink 从 600 砍到 400 GB/s；H800 同理（单卡 GEMM 看不出差别，**一开 TP=8 就露馅**）。

**（3）选型框架（按负载类型选账本）**：

| 负载 | 主要看 | 原因 |
|---|---|---|
| 大模型**训练** | $\pi$（按训练精度）+ $\beta_{\text{nvlink}}$（TP/EP）+ $M$（装优化器状态） | 大 GEMM 是 compute-bound；TP/EP 通信走 NVLink |
| 长上下文**decode** | $M$ + $\beta_{\text{hbm}}$ | decode 是 memory-bound，且 KV cache 吃掉大量显存（H200 的意义就在这） |
| 高并发**prefill** | $\pi$ + $\beta_{\text{hbm}}$ | prefill 是大 GEMM（compute-bound）但也要读长 prompt |
| 小模型推理 | 性价比 + 功耗，**别看 NVLink** | 不需要大 TP，L40S 这类无 NVLink 卡够用 |
| MoE 推理 | $M$ + $\beta_{\text{hbm}}$ + scale-up 域大小 | 专家要走 all-to-all，NVL72 把域从 8 扩到 72 就是为这个 |

### 3. 具体数值样例

```text
【主流数据中心 GPU 对照表】
（BT = BF16 Tensor Core；括号内为 sparse；NVLink 为双向合计；数据来源见文首）

型号          架构      显存 M          β_hbm      BT TC dense(sparse)  NVLink双向   TDP
V100 SXM2     Volta     32 GB HBM2     0.90 TB/s  125（无 2:4）        300 GB/s     300 W
A100 SXM      Ampere    80 GB HBM2e    2.04 TB/s  312 (624)            600 GB/s     400 W
A800 SXM      Ampere    80 GB HBM2e    ~2.0 TB/s  312 (624)            400 GB/s ←   400 W
H100 SXM      Hopper    80 GB HBM3     3.35 TB/s  989 (1979)           900 GB/s     700 W
H800 SXM      Hopper    80 GB HBM3     ~3.35 TB/s ≈ H100              400 GB/s ←   ~700 W
H100 PCIe     Hopper    80 GB HBM3     2.0 TB/s   ~756 (1513)          600 GB/s     350 W
H200 SXM      Hopper    141 GB HBM3e   4.8 TB/s   989 (1979)           900 GB/s     700 W
L40S          Ada       48 GB GDDR6    0.86 TB/s  362 (733)            无 ←         350 W
B200 SXM      Blackwell 180 GB HBM3E  7.7 TB/s   ~2500 (5000)         1.8 TB/s     1000 W
GB200(NVL72)  Blackwell 186 GB HBM3E  8.0 TB/s   ~2500 (5000)         1.8 TB/s     1200 W
⇒ 注意 A800/H800 的"←"：算力和带宽都对得起型号，**唯独互连被腰斩**；
  国内集群最常见的是 H800，一旦 TP=8 做 all-reduce，通信屋顶就比 H100 低一半

【推演 A：拿到"H100"三个字，必须先问板型】
  H100 SXM：80GB / 3.35 TB/s / NVLink 900 / 700W
  H100 PCIe：80GB / 2.0 TB/s / NVLink 600 / 350W
  带宽差 1.7× → 同一个训练任务，PCIe 版的 TP all-reduce 屋顶和 KV cache 吞吐都更低
  ⇒ 面试里追问一句"SXM 还是 PCIe"是专业度的体现

【推演 B：70B 模型训 vs 推，选卡逻辑完全不同】
  训练（全量微调，BF16 + Adam fp32）：≈ 16 B/参数 = 1.12 TB
    → 关注 π 与 NVLink：H100/H800 SXM 集群（TP/PP/DP + ZeRO）
    → M 也要够（否则激活都放不下）
  decode（70B BF16，长上下文）：≈ 140 GB 权重 + 每序列 2.68 GB/8K KV
    → H100 80GB 根本装不下权重（要 TP=2 以上）；H200 141GB 刚装下权重但没 KV 余量
    → 实际方案：FP8/INT4 量化（权重降到 35~70 GB）+ GQA + PagedAttention
    → 关注 M 与 β_hbm：H200（141GB/4.8TB/s）就是为这类负载设计的

【推演 C：一个 8 卡 HGX 节点 vs 一个 NVL72 机柜】
  HGX（H100/H200/B200）：1 节点 = 8 GPU 全互联（NVSwitch），跨节点靠 RDMA
  GB200 NVL72：1 机柜 = 72 GPU 在同一个 NVLink 域，聚合带宽 130 TB/s
  ⇒ 软件含义：**EP ≤ 72 时可以全程不掉出 NVLink 域**（MoE 的 all-to-all 收益巨大）
  ⇒ 代价：单卡 TDP 提到 1200W 量级，机柜几十~上百 kW，**必须液冷**
```

### 4. 演进：三个可以聊的趋势

- **拐点右移**：算力涨得比带宽快（A100→H100 是 3.2× vs 1.6×），导致 memory-bound 的算子越来越多；**H200 是一次"同算力、换带宽"的反向操作**（拐点从 295 退回 206），是理解 SKU 设计意图的好例子；
- **scale-up 域在膨胀**：NVLink 从 8 卡域（HGX）扩到 72 卡域（NVL72），"机架即一块 GPU"；而 scale-out（IB/RoCE）同期从 200G 到 800G，**两个域的落差没有收窄**（掉出 NVLink 域依然贵一个数量级）；
- **出口 SKU 的教训**：A800/H800 表明"算力可以对齐、互连可以锁"，因此**国内做大规模训练时，通信优化（DP 重叠、EP 域内转发、梯度压缩）的收益比国外同行更高**——这是很实用的面试观点。

> **面试一句话总结**：读一块 GPU 只记**五本账**（$\pi$ dense 峰值、$M$ 容量、$\beta_{\text{hbm}}$ 带宽、$\beta_{\text{nvlink}}$ 互连、TDP）和**五条铁律**（sparse 折半、bit≠byte 差 8 倍、NVLink 默认双向、SXM≠PCIe≠NVL、出口 SKU 优先砍互连）；对照记忆的锚点是 **A100 80GB/2.04TB/s/312 BF16、H100 SXM 80GB/3.35TB/s/989、H200 141GB/4.8TB/s（同算力换带宽）、H800 = H100 算力但 NVLink 砍到 400GB/s、B200 180GB/7.7TB/s/~2.5PFLOPS、NVL72 把 scale-up 域从 8 扩到 72 卡**；选型逻辑是"训练看 $\pi$+NVLink、长上下文 decode 看 $M$+$\beta_{\text{hbm}}$、小模型推理别看 NVLink"，且始终记得**同一个型号名可能对应完全不同的板型**。

---

## 10. 昇腾 NPU 结构对标：达芬奇 AI Core 与 CUDA 的差异

### 1. 现有问题：从 GPU 转到 NPU 时，心智模型怎么换

如果简历里有昇腾/华为相关经历，面试官一定会问"GPU 和 NPU 有什么不同"。答"都是加速卡"是负分。要点是：**达芬奇架构把计算"按类型"拆到不同硬件单元**，这与 NVIDIA 的"统一 SIMT + 专用 Tensor Core 子单元"是两条不同的设计路线。

### 2. 方法论：AI Core 三单元 + 四级存储 + CANN 栈

**（1）达芬奇架构的 AI Core 包含三种计算单元**（来源：ForceInjection/AI-fundamentals《昇腾硬件架构与 CANN 软件栈》）：

```text
┌─────────────────────────────────────┐
│              AI Core                │
│  ┌──────────┐ ┌────────┐ ┌───────┐  │
│  │  Cube    │ │ Vector │ │ Scalar│  │
│  │  Unit    │ │  Unit  │ │  Unit │  │
│  │ (矩阵运算)│ │(向量运算)│ │ (标量) │  │
│  └──────────┘ └────────┘ └───────┘  │
│  ┌──────────┐ ┌──────────────────┐  │
│  │ L1 Buffer│ │   L0 Buffer      │  │
│  └──────────┘ └──────────────────┘  │
│  ┌────────────────────────────────┐ │
│  │        Unified Buffer          │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

| 计算单元 | 职责 | CUDA 类比 |
|---|---|---|
| **Cube Unit** | 矩阵乘加（卷积/矩阵乘），占 AI 计算量约 90% | **Tensor Core** |
| **Vector Unit** | 向量运算：激活函数、归一化、逐元素操作 | CUDA Core（向量化） |
| **Scalar Unit** | 标量计算、控制流、地址计算 | CUDA Core（标量） |

**（2）与 SIMT 的本质差异（面试可以直接背这段）**：

- **NVIDIA**：所有线程走同一套 SIMT 流水线，Tensor Core 只是 SM 内的**一个子单元**，需要程序员/编译器**显式把 GEMM 映射到 Tensor Core**（否则就退化成 CUDA Core 跑）；
- **达芬奇**：计算**按类型静态拆分**到 Cube / Vector / Scalar 三个单元，由**图编译器（GE）**分析算子类型并分发。好处是矩阵运算的能效比极高（同等功耗吞吐更高）、坏处是**CUDA 代码不能 1:1 翻译**——自定义 CUDA kernel 必须改写成 Ascend C 或 TBE 算子；而**标准 PyTorch 算子（已有 NPU 适配）可以直接迁移**，因为它们落在 Cube Unit 的优化范围内。

**（3）存储层次四级**：`HBM → L2 Cache（芯片级共享）→ L1 Buffer（AI Core 内）→ L0 Buffer（计算单元独享）`。注意与 GPU 的对应关系：L0 更接近计算单元（类似寄存器/紧耦合缓冲），L1 类似 SMEM，L2 与 HBM 概念一致。

**（4）CANN 软件栈七层**（对标 CUDA 平台）：驱动层 → 运行时层（Runtime/TSC 任务调度）→ 算子层（内置算子 OPP / Ascend C / TBE）→ 图编译层（**GE 图引擎 + AOE 自动调优**）→ 应用层（AscendCL）→ 集合通信（HCCL，对标 NCCL）→ 框架适配（torch_npu / MindSpore）。

**与 CUDA 的一个关键结构差异**：**CANN 在工具链里内置了统一图编译器 GE**，而 CUDA 把图优化留给上层框架（TorchDynamo/XLA/TensorRT）——所以 CUDA Graphs 与 GE 不是同一层的东西。

### 3. 具体数值样例

```text
【Ascend 910B3 的实测参数（来源：昇腾架构文档，实测口径）】
  制程/架构：7 nm，达芬奇
  HBM：64 GB HBM2e，频率 1600 MHz
  实测显存带宽：约 1.54 TB/s（ascend-dmi --bw -t d2d 实测 1538 GB/s）
  卡间互联：HCCS，8 卡 full mesh（每两张卡都有专用链路直连，不经 PCIe/NUMA）
  典型功耗：空闲 ~90–97 W / FP16 满载 ~231 W / 训练满载实测 ~273 W / TDP 300 W
  ⇒ 对比 A100：容量大（64 > 40/80 GB 的 40GB 版）、带宽略低（1.54 vs 2.04 TB/s）

【为什么"HCCS 8 卡 full mesh"对训练很关键】
  GPU 侧：8 卡通过 NVSwitch 全互联（A100 600 GB/s、H100 900 GB/s 双向）
  NPU 侧：8 卡通过 HCCS 全互联，**任意两卡之间都是直连**
  ⇒ 好处：TP=8 的 all-reduce 不需要绕 PCIe/NUMA，
    避免了"跨 NUMA 节点带宽掉一半"这类抖动
  ⇒ 面试说法：昇腾的卡间拓扑是"训练友好"的，与 HGX 的 NVSwitch 全互联同构

【工具对照（背这一张就够用了）】
  设备管理   npu-smi        ↔ nvidia-smi
  Profiling  msprof         ↔ nsys / ncu
  模型转换   atc            ↔ trtexec
  算子开发   Ascend C / TBE ↔ CUDA C++
  编译工具   ccec (毕昇)     ↔ nvcc
  集合通信   HCCL           ↔ NCCL（hccl_test ↔ nccl-tests）
  图编译     GE             ↔ CUDA Graphs / TensorRT
  可见设备   ASCEND_RT_VISIBLE_DEVICES ↔ CUDA_VISIBLE_DEVICES
```

### 4. 演进与"迁移"的注意点

- **Cube Unit 对标 Tensor Core 但更"强制"**：GPU 上不用 Tensor Core 也能跑（退化成 CUDA Core），达芬奇上矩阵运算**默认就走 Cube**，所以"算子是否被 GE 正确分发到 Cube"直接决定性能；
- **算子开发心智不同**：CUDA 是"写 kernel"，昇腾是"优先用内置算子（OPP，1400+ 算子）→ 不够用 Ascend C 自定义 → 再不够用 TBE 深度优化"，三层抽象；
- **软件栈版本要分清商业版本与内部版本**（例如 CANN 商业版 8.0.1 对应 runtime/compiler/hccl 组件版本 7.6.0.2.220 系列）——查兼容性矩阵时这点很容易踩坑；
- **对我们项目的直接含义**：第 18 节讲过的"昇腾上 `torch.nested` 走慢速/回退路径导致 update_actor +15s"就是"图编译器 + Cube 分发"这套体系下的典型现象——**在 GPU 上便宜的操作，在 NPU 上可能因为落到 Vector/Scalar 而非 Cube 而变贵**。

> **面试一句话总结**：昇腾采用**达芬奇架构**，每个 **AI Core** 内含三类计算单元——**Cube Unit（矩阵，对标 Tensor Core，占约 90% AI 计算量）、Vector Unit（激活/归一化/逐元素）、Scalar Unit（标量/控制/地址）**，与 NVIDIA "统一 SIMT 流水线 + SM 内 Tensor Core 子单元"相比，达芬奇是**按计算类型静态拆分、由图编译器 GE 分发**，因此能效比高但 **CUDA 代码不能直接迁移**（标准 PyTorch 算子可直接迁移、自定义 kernel 要改写 Ascend C/TBE）；存储是 **HBM → L2 → L1 Buffer → L0 Buffer** 四级，910B3 实测 **64 GB HBM2e、约 1.54 TB/s、HCCS 8 卡 full mesh、TDP 300 W**；工具链与 CUDA 一一对应（npu-smi↔nvidia-smi、msprof↔ncu、HCCL↔NCCL、ccec↔nvcc、GE↔CUDA Graphs/TRT）。

---

## 11. CUDA Stream、并发与重叠：让拷贝和计算同时跑

### 1. 现有问题：单 stream 下 GPU 有一半时间在"等"

看 `nvidia-smi` 会发现一个奇怪现象：kernel 在跑的时候**拷贝引擎（Copy Engine）闲着**，而做 H2D 拷贝的时候**SM 全闲着**。更糟的是多进程共享一张卡时，两个进程的任务会莫名其妙**串行**。这两个问题的根源都是"并发没有被用起来"：

1. **同一 stream 内的命令严格有序**——上一个 kernel 跑完才启动下一个，拷贝也排队；
2. **CPU 侧发起太慢**——每个 kernel launch 要几微秒，几百个小 kernel 就把 GPU 喂不饱；
3. **假串行（false serialization）**——Fermi 时代主机到 GPU 只有**一条硬件工作队列**，即使你开了多个 stream，硬件层面也只能一个个来。

### 2. 方法论：三种引擎、两个"并发门槛"、一套同步原语

**（1）Stream 的本质是"主机侧的命令队列"。** `cudaStream_t` 就是一条 FIFO 队列，**同一队列内的 kernel/拷贝按提交顺序执行**，**不同队列之间默认没有任何顺序约束**——所以"能不能并发"取决于硬件资源与工作队列。

**（2）并发的物理基础是"芯片上有多个独立的执行引擎"**：

| 引擎 | 干什么 | 与 SM 的关系 |
|---|---|---|
| **Compute（SM 阵列）** | 跑 kernel | 主体 |
| **Copy Engine（DMA）** | H2D / D2H / D2D 拷贝 | **独立硬件，与 SM 并行** |
| NVJPEG / NVDEC / NVENC | 图像编解码 | 独立硬件 |
| **TMA / LD-ST** | 片内搬运（第 3、5 节） | 在 SM 内，但异步 |

**这就是"重叠"能成立的硬件前提**：拷贝引擎和 SM 是两块硬件，只要队列允许，它们可以同时工作。经典流水线是把大搬运切成 N 块，用 N 条 stream 交替发射，让"块 i 的计算"和"块 i+1 的搬运"重叠。

**（3）并发有两个门槛，缺一不可**：

- **门槛一：硬件工作队列数量。** Fermi 只有 1 条 → 多 stream 也会**假串行**；**Kepler（CC 3.5）起引入 Hyper-Q，提供 32 条硬件管理的工作队列**，允许来自**多个 stream、多个 MPI 进程、多个线程**的 work 同时在 GPU 上排队，NVIDIA 官方描述它消除的正是 "false serialization across tasks"；
- **门槛二：资源够不够。** 即使队列够，如果前一个 kernel 把 SM/寄存器/SMEM 占满了，后一个 kernel 也只能等——这就是"**并发 kernel 的收益取决于每个 kernel 的资源占用**"，小 kernel（资源占用低）叠加效果好，大 kernel 基本没法并存。

**（4）两个必须知道的"隐形串行"**：

- **Legacy default stream（NULL stream）的隐式同步**：在不使用 per-thread default stream 时，default stream 与所有 **blocking stream** 之间会互相等待——你以为在用多 stream，实际被 default stream 串起来了。解法：编译时加 `--default-stream per-thread`，或创建 stream 时带 `cudaStreamNonBlocking`；
- **pageable 内存的"假异步"**：H2D 从普通（pageable）内存发出时，驱动会**分块经内部 pinned 缓冲中转**，只有在数据量小到能一次放进内部缓冲时才真正异步；**D2H 更严格，必须等 kernel 结束**。所以**"重叠的前提是有 pinned memory"**（与第 5、6 节的 pinned 是同一条知识）。`cudaMemcpyAsync` 在没有 pinned 的情况下并不能兑现异步语义。

**（5）同步与计时原语（面试常混，需要分清）**：

| API | 语义 | 是否阻塞 host |
|---|---|---|
| `cudaDeviceSynchronize()` | 等设备上**所有**工作完成 | ✅ |
| `cudaStreamSynchronize(s)` | 等**某条 stream** 完成 | ✅ |
| `cudaStreamWaitEvent(s, e)` | 让 s **等事件 e**（跨 stream 建依赖） | ❌ 不阻塞 |
| `cudaEventRecord(e, s)` | 在 s 中打点 | ❌ |
| `cudaEventSynchronize(e)` | 等事件完成 | ✅ |
| `cudaEventElapsedTime()` | 两个事件间耗时（**测 GPU 时间**，比 wall clock 准） | — |

**关键用法**：跨 stream 的依赖必须用 `cudaStreamWaitEvent`（**非阻塞**），不要用 `cudaDeviceSynchronize()` 把整条流水线打断——这是"看起来用了多 stream、实际没重叠"的最常见原因。

**（6）两把"减小开销"的钥匙**：

- **CUDA Graphs**：把一串 kernel 启动**录制**成图，之后一次提交、重复执行，把每次 launch 的 ~几微秒开销摊薄。适合"很多小 kernel"的 launch-bound 场景（如 MoE 的小 expert GEMM、逐层小算子）；现代框架（vLLM/TensorRT/Inductor）都在用；
- **MPS（Multi-Process Service）**：让多个进程共享一张卡的 SM 资源，配合 Hyper-Q 让多进程的 work 真正并发（没有 MPS 时多进程不能同时在 SM 上跑）。HPC 上常用它提高小任务的"GPU farming"吞吐。

**（7）PyTorch 里的对应关系**（面试落到工程时有用）：`torch.cuda.Stream` / `torch.cuda.current_stream()` 对应上面的 stream；`torch.cuda.synchronize()` 对应 `cudaDeviceSynchronize`；`tensor.record_stream(s)` 告诉分配器"这块显存还要被 s 用"，**防止显存被提前复用**（多 stream 下的经典正确性坑）；`torch.cuda.graphs` / `torch.cuda.make_graphed_callables` 对应 CUDA Graphs；`torch.cuda.Stream` + `wait_stream` 用来手写流水线（通信与计算重叠就是这么做出来的）。

### 3. 具体数值样例

```text
【样例 A：拷贝与计算重叠的收益（H2D 100 MB + kernel + D2H 100 MB）】
  设定：有效 H2D/D2H 带宽 25 GB/s（PCIe Gen4 x16 实测量级）
        H2D = 100 MB / 25 GB/s = 4 ms；D2H 同理 4 ms；kernel = 10 ms

  串行（单 stream）：
    总时间 = 4 + 10 + 4 = 18 ms
    GPU 有效利用率：只有 10/18 = 56% 的时间 SM 在工作

  切成 4 块，2~4 条 stream 流水：
    每块：H2D 1 ms、kernel 2.5 ms、D2H 1 ms
    流水线总时间 ≈ 第一块 H2D(1) + 全部 kernel(4 × 2.5 = 10) + 最后一块 D2H(1)
                = 12 ms
    加速比 18/12 = 1.5×
  ⇒ 结论：**收益上界 = 拷贝时间能被计算完全吸收**；
    若 kernel 时间 ≫ 拷贝时间，重叠收益接近拷贝那部分的全部
    （本例拷贝共 8 ms，最多省到 8 ms；实际省 6 ms）

【样例 B：Hyper-Q 消除假串行】
  Fermi（1 条工作队列）：
    4 条 stream 各提交 1 个"只用 25% SM"的小 kernel
    硬件只能串行 → 总时间 = 4 × t
  Kepler+（32 条队列，Hyper-Q）：
    4 个小 kernel 可以同时在 SM 上共存（资源允许）
    总时间 ≈ t（理想）或介于 t 与 4t 之间（取决于资源）
  ⇒ 这就是 NVIDIA 说的 "false serialization across tasks" 被消除
  ⇒ 注意前提是"**每个 kernel 资源占用低**"；若每个 kernel 都要占满 SM，
    Hyper-Q 也救不了（第 7 节的 occupancy 与第 3 节的 SM 资源）

【样例 C：launch overhead 与 CUDA Graphs】
  一个 kernel launch 的开销约 5~10 µs（CPU 侧）
  1000 个小 kernel（每个算 2 µs）：
    纯计算 = 2 ms，但 launch 开销 = 5~10 ms → 总 7~12 ms，**launch-bound**
  改用 CUDA Graph 一次提交：
    launch 开销摊到 ~1~2 µs/kernel 甚至更低 → 总时间回到 ~3 ms
  ⇒ 判据：**单 kernel 时间 < launch 开销时，就该考虑 Graph**
    （MoE 的专家 GEMM、逐层 norm/激活都属于这一类）

【样例 D：为什么"用了多 stream 却没变快"】
  三个典型原因，按出现频率排：
    ① 跨 stream 用了 cudaDeviceSynchronize（把流水线打断）
       → 改用 cudaStreamWaitEvent
    ② H2D 的源内存不是 pinned（走了内部 staging + 同步路径）
       → 改用 cudaMallocHost / cudaHostAlloc
    ③ kernel 资源占用太高，硬件无法同时驻留两个 kernel
       → 减小 block/寄存器占用，或干脆别指望并发
```

### 4. 演进：从"排队"到"图"再到"程序化依赖"

- **Kepler：Hyper-Q（32 条硬件工作队列）+ Dynamic Parallelism** —— 解决假串行；
- **CUDA 10：CUDA Graphs** —— 解决 launch-bound，把"启动开销"从 O(n) 降到 O(1)；
- **CUDA 12：Programmatic Dependent Launch / 图条件节点** —— 允许后继 kernel 在**前驱还没完全结束**时就启动（用 `cudaGridDependencySynchronize` 精确控制依赖点），进一步压掉 kernel 之间的"边界气泡"；
- **与硬件异步化的合流**：`cp.async`（Ampere）、**TMA + mbarrier + warp specialization**（Hopper）把"异步"做进了 SM 内部，所以现代 kernel 的性能模型已经是"**多条流水线 + 显式依赖**"，而不再是"一条 stream 顺序跑"。

> **面试一句话总结**：CUDA 并发的物理基础是"**芯片上有多个独立引擎**"（SM 阵列跑计算、**Copy Engine** 跑拷贝、TMA/LDST 片内异步），而 stream 只是**主机侧的命令队列**——同 stream 有序、跨 stream 无序；要真正并发必须跨过两个门槛：**硬件工作队列数量**（Fermi 只有 1 条会"假串行"，**Kepler 起 Hyper-Q 提供 32 条硬件队列**，允许来自多 stream/多 MPI 进程/多线程的 work 同时排队）和**资源是否够**（小 kernel 才叠得起来，大 kernel 占满 SM 就无解）；还有两个隐形串行必须知道——**legacy default stream 会与所有 blocking stream 互相等待**（用 `--default-stream per-thread` 或 `cudaStreamNonBlocking` 解除）、**pageable 内存的 H2D/D2H 并非真异步**（尤其 D2H 必须等 kernel 结束，所以**重叠的前提是 pinned memory**）；跨 stream 依赖用**非阻塞的 `cudaStreamWaitEvent`** 而不是 `cudaDeviceSynchronize`，测时间用 `cudaEventElapsedTime`；当"单 kernel 时间 < launch 开销（5~10 µs）"时用 **CUDA Graphs** 摊薄（1000 个小 kernel 可从 7~12 ms 压到 ~3 ms），多进程共享卡用 **MPS + Hyper-Q**。

---

## 12. GPUDirect 与门铃机制：网卡/存储怎么直接读写显存

### 1. 现有问题：为什么"网卡直接读写显存"不是插上就行

跨机通信的传统路径是"网卡 → CPU 内存（pinned buffer）→ 显存"，多一次拷贝、多一次 CPU 参与、多占一份内存带宽。直觉上"让网卡直接 DMA 到显存"应该很简单，但实际会遇到一连串问题：

1. **网卡凭什么能访问显存？** GPU 显存不是 CPU 地址空间的一部分，PCIe 设备之间默认**不能互相访问**；
2. **为什么 `ibv_reg_mr` 注册显存会失败（EFAULT）？** RDMA 要求内存先被"注册/pin 住"，而显存的 pin 走的是一套**独立于 host 内存的机制**；
3. **GPU 怎么反过来去访问网卡的寄存器/队列？** 要发起通信，GPU 得能写网卡的硬件队列；
4. **"门铃（doorbell）"到底是什么、为什么必须有？**

> 本节只讲 **GPU 侧机制**；网络侧的协议与配置结论（GDR/GDS 能省什么、`NCCL_NET_GDR_LEVEL`、`cuFile`）见 `Communication.md` 第 7 节。

### 2. 方法论：四步打通 + 一个门铃 + 四类内存对照

**（1）GPUDirect 的三种形态**（面试先把这个分类说清）：

| 形态 | 谁 ↔ 谁 | 载体 | 典型用途 |
|---|---|---|---|
| **GPUDirect P2P** | GPU ↔ GPU（同机） | PCIe peer-to-peer DMA / NVLink | 机内 TP/EP、NVLink 之外的兜底路径 |
| **GPUDirect RDMA（GDR）** | RDMA NIC ↔ GPU | PCIe | 跨机集合通信（NCCL 机间打满带宽的前提） |
| **GPUDirect Storage（GDS）** | NVMe/文件系统 ↔ GPU | PCIe | 数据加载、checkpoint（`cuFile` API） |

**（2）"网卡怎么直接读写显存"——四步机制**（这是本题的核心答案）：

```text
① GPU 显存被映射到 PCIe 地址空间：BAR1 aperture
   GPU 把自己的显存通过一个 PCIe BAR（Base Address Register，叫 BAR1 或 peer aperture）
   暴露出去，使外部 PCIe 设备能把这段地址当作"对端内存"来访问。
   ⇒ BAR1 窗口大小限制了"一次能被 P2P 映射多少显存"，
     所以数据中心卡用 Resizable BAR / large BAR 让整块显存可映射。

② 内核模块 nvidia-peermem 把 GPU 页 pin 住并交给 RDMA 子系统
   RDMA 的规矩是"内存必须先注册成 MR（Memory Region）才能被网卡 DMA"。
   host 内存靠 ibv_reg_mr 直接 pin；GPU 显存需要 nvidia-peermem 这个
   内核模块出面，把 GPU 物理页交给 RDMA 栈。

③ 用户态创建 ThirdPartyP2P 对象（这一步是"为什么 GeForce 用不了 GDR"的答案）
   用户态用 CUDA VMM API（cuMemCreate / cuMemMap）分配显存，
   再通过 RM（Resource Manager）ioctl 创建 ThirdPartyP2P（NV503C）对象。
   ⇒ NVIDIA 通过"**软件分段**"把 GDR 限制在数据中心卡：
     GeForce 上 CUDA runtime 不会创建这个 NV503C 对象，
     nvidia-peermem 就找不到要 pin 的 BAR 页，ibv_reg_mr 直接 EFAULT。
     （已有开源补丁通过强制 BAR1 P2P + persistent P2P API + 用户态手工创建
       NV503C 来打通，属实验性质）

④ ibv_reg_mr 成功 → 网卡用 PCIe peer-to-peer DMA 直接读写 VRAM
   数据路径：NIC →（PCIe P2P DMA）→ GPU HBM，**完全绕开 host DRAM**
```

**（3）反向：GPU 怎么访问网卡的资源（寄存器、队列）？** 关键动作是**把 NIC 的 BAR（寄存器空间）mmap 到用户态**：

- 网卡的队列（SQ/RQ/CQ）、寄存器都是 **MMIO**（memory-mapped IO）；
- 用户态驱动把这部分物理地址 `mmap` 进自己的地址空间后，这些地址就是**普通的可读写内存地址**；
- 于是**GPU kernel 也能 load/store 它们**（对 GPU 而言就是一次 PCIe 写）——这一步打通之后，GPU 就能自己写网卡的**门铃寄存器**，从而"自己发起通信"。

**（4）门铃（doorbell）机制：为什么"写了队列"还要再敲一下门？**

发送一个消息的完整动作是"**填 WQE → 敲门铃**"：

```text
生产者（CPU 或 GPU）                     硬件（NIC）
  ① 把工作请求 WQE 写进发送队列 SQ  ──────────►（队列在内存里，硬件此时不知道）
  ② 写一个 doorbell 寄存器（MMIO 写） ────────► 硬件被"敲门"唤醒
                                              ③ 硬件去 SQ 取 WQE
                                              ④ 执行 DMA / 组包 / 发送
```

**为什么不能只写内存？** 因为硬件**不会持续扫描内存里的队列**（扫描要占用内部带宽和逻辑，代价高）。所以约定是：**内存写负责放数据，一次 MMIO 写（doorbell）负责通知**。这也是为什么：

- **门铃是"提交延迟"的关键路径**——小消息场景下总延迟 ≈ 构造 WQE + doorbell MMIO 写 + NIC 处理；
- 反过来，**接收侧可以不用门铃而用轮询（polling）**：消费方主动查 CQ（完成队列）而不是等中断，用 CPU/GPU 空转换低延迟（DPU/低延时交易场景常见）；`ibv_poll_cq` vs `ibv_get_cq_event` 就是这两种风格。

**（5）GPU-initiated communication：把"发起通信"也搬进 GPU。** 传统路径里 GPU 算完要**通知 CPU**、由 CPU post send——多一次 device→host 往返：

```text
传统（CPU 中介）：
  GPU kernel 算完 → (device→host 事件/中断 ~5~10 µs) → CPU 构造 WQE + 敲门铃 → NIC 发送

GPU-initiated（GPUDirect Async / NVSHMEM）：
  GPU kernel 内直接构造 WQE、直接写 NIC 的 doorbell → NIC 发送
  ⇒ 省掉 device→host→device 的一次往返，通信延迟与"每层都要通信"的开销显著下降
  ⇒ 这是 MoE all-to-all / TP all-reduce 能"藏在计算里"的机制基础
```

**（6）四类内存的适用场景与坑**（面试高频，务必分清）：

| 类型 | 怎么分配 | GPU 能直接访问 | 是异步拷贝前提吗 | 适合 / 坑 |
|---|---|---|---|---|
| **pinned（页锁定）** | `cudaMallocHost` / `cudaHostAlloc` | ❌（除非额外 mapped） | ✅ **是** | 高频 H2D/D2H、RDMA 的 MR；**分配过多会锁死物理内存**（`ulimit -l`） |
| **zero-copy（mapped pinned）** | `cudaHostAllocMapped` + `cudaHostGetDevicePointer` | ✅ 直接走 PCIe 读 | — | 只读一次、小数据、稀疏访问；**每次访问都过 PCIe，比 HBM 慢 1~2 个数量级** |
| **UVA（统一虚拟地址）** | 64-bit 系统自动 | ✅ | — | 简化指针管理（不用手动区分 host/device 指针），pinned 内存在 UVA 下可被 kernel 直接用 |
| **UM（managed / UVM）** | `cudaMallocManaged` | ✅（页错误驱动迁移） | — | 数据大于显存、访问模式不规则；**在纯 PCIe 机器上可能因页错误抖动而极慢**（详见 `Communication.md` 第 4 节） |

**一句话记忆**：**pinned 解决"能不能异步"，zero-copy 解决"能不能不拷贝"，UVA 解决"指针要不要分开写"，UM 解决"放不下怎么办"**。

### 3. 具体数值样例

```text
【样例 A：GDR 省掉了什么（100 MB 消息，400 Gb/s ≈ 50 GB/s 单向）】
  无 GDR（三次搬运 + CPU 参与）：
    NIC → host pinned buffer：50 GB/s → 2 ms（CPU 参与、占内存带宽）
    cudaMemcpy H2D：PCIe Gen4 x16 有效 25 GB/s → 4 ms
    合计 ≈ 6 ms，且 HBM 与 host DRAM 各被写一遍
  有 GDR（一跳直达）：
    NIC →（PCIe P2P DMA）→ GPU HBM：≈ 2~4 ms（受 PCIe 带宽约束，但省掉一跳与 CPU）
  ⇒ 收益：省一次拷贝、省一次 host 内存带宽占用、CPU 只做控制
  ⇒ 这也是 NCCL 机间打满带宽的前提（配置位 NCCL_NET_GDR_LEVEL）
  实测锚点（25GbE + ConnectX-4 Lx + RTX 3090，开源实验）：
    大消息带宽 ~2922 MiB/s ≈ 24.5 Gbps，**打满 25GbE 链路**，
    与"CPU 内存 RDMA"持平 → 说明 GDR 本身不是瓶颈

【样例 B：BAR1 窗口决定"能 P2P 多少显存"】
  查看：nvidia-smi -q | grep -i BAR1
  BAR1 太小 → 只能映射一部分显存做 P2P（大模型训练里表现为
    "某些 buffer 能用 GDR、某些不能"，或 reg_mr 失败）
  Resizable BAR / large BAR → 让整块显存落进 BAR1 窗口，P2P 覆盖更完整
  ⇒ 这是"同一张卡，换主板/BIOS 设置后 GDR 行为不同"的常见原因

【样例 C：门铃与"小消息延迟"的账】
  小消息（8 B payload）的端到端延迟大致由三段决定：
    ① 构造 WQE（写内存，~百 ns 级）
    ② **doorbell MMIO 写**（跨 PCIe，~几百 ns）
    ③ NIC 处理 + 线缆 + 对端处理
  host 内存 RDMA 的小消息延迟通常在 ~1~2 µs 量级
  ⇒ 结论：小消息场景**瓶颈不是带宽而是"提交/通知"路径**，
    所以优化手段是"批量 post（一次 doorbell 提交多个 WQE）""inline 小数据"
    "少发信号（减少 CQ 事件）"——这些和 RDMA verbs 层的调优一一对应

【样例 D：GPU-initiated 省下的往返（每层都要通信的场景）】
  传统路径每层多一次 device→host 通知（~5~10 µs）
  80 层 × 每层多次通信 → 光"通知 CPU"就可能累积到毫秒级
  GPU-initiated（kernel 内写 WQE + doorbell）省掉这段往返
  ⇒ 对 MoE all-to-all、TP all-reduce 这类"每层都通信"的负载，
    这项优化直接决定通信能否被计算掩盖（回到第 8 节的"通信墙"）
```

### 4. 演进：从"设备间能互相看见"到"GPU 主动通信"

- **GPUDirect P2P（CUDA 4 时代）**：让同机 GPU 之间通过 PCIe peer DMA 直接互访（NVLink 普及后机内主要走 NVLink，P2P 成为兜底）；
- **GPUDirect RDMA（CUDA 5.0 起，Tesla 卡）**：NIC 直连显存，成为 NCCL 跨机打满带宽的前提；
- **GPUDirect Storage（GDS，`cuFile`）**：NVMe/文件系统直通显存，数据加载与 checkpoint 免 bounce buffer；
- **GPUDirect Async / NVSHMEM（GPU-initiated communication）**：把"发起通信"从 CPU 搬到 GPU，配合 IBGDA（InfiniBand GPUDirect Async）让 kernel 内直接投递 WQE；
- **一个现实约束要记住**：NVIDIA 通过**软件分段**（是否创建 ThirdPartyP2P 对象）而非硬件能力来区分数据中心卡与消费卡——这就是"GeForce 也能 GDR，但要打补丁"的原因；面试里说"消费卡硬件不支持 GDR"是不准确的，准确说法是"**驱动/软件层面未开放**"。

> **面试一句话总结**：GPUDirect 分三种形态——**P2P（GPU↔GPU，PCIe peer DMA/NVLink）、RDMA（NIC↔显存，即 GDR，NCCL 跨机打满带宽的前提）、Storage（NVMe↔显存，`cuFile`）**；"网卡怎么直接读写显存"的四步是 **①显存经 BAR1 aperture 暴露到 PCIe 地址空间 → ②内核模块 `nvidia-peermem` 把 GPU 页 pin 住交给 RDMA 栈 → ③用户态用 CUDA VMM API 分配显存并经 RM ioctl 创建 ThirdPartyP2P（NV503C）对象（这一步就是 GeForce 上 `ibv_reg_mr` 报 EFAULT 的原因——NVIDIA 用软件分段限制，不是硬件不支持）→ ④NIC 通过 PCIe P2P DMA 直写 HBM、完全绕开 host DRAM**；反向让 **GPU 访问网卡**的办法是把 NIC 的 BAR（队列/寄存器）**mmap 到用户态**，于是 kernel 也能 load/store；**门铃（doorbell）机制**是"**填 WQE 写内存 + 一次 doorbell MMIO 写通知硬件**"——因为硬件不会持续扫描内存里的队列，所以必须"敲门"，它也是小消息延迟的关键路径（小消息瓶颈是提交/通知而非带宽，对应"批量 post、inline、少发信号"等 verbs 调优）；把门铃也交给 GPU 就是 **GPU-initiated communication（GPUDirect Async/NVSHMEM）**，省掉每层 device→host→device 的往返，是 MoE/TP 通信能被计算掩盖的基础；最后务必分清 **pinned（异步的前提）/ zero-copy（不拷贝）/ UVA（指针统一）/ UM（放不下怎么办）** 四类内存的适用场景与坑。

---

# 二、串讲：把一块 GPU 的十二个技术点串成一条数据通路

## 13. 一次 GEMM 的完整硬件旅程（十二个技术点如何协同）

### 1. 现有问题：十二个点都懂了，但讲不成一条线

面试的最后一问往往是开放式的："**从你写完一个 GEMM 到它跑在硬件上，数据经过了哪些地方？**"或者"**给你一台 8 卡机器，怎么榨干性能？**"。这两问考的不是某个知识点，而是**把硬件串成数据通路的能力**。这一节就是把第 1~12 节穿起来。

### 2. 方法论：六跳数据通路 + 三堵墙 + 一个 checklist

**（1）一次 GEMM 的六跳（从 HBM 到写回）**：

```text
① HBM（80 GB，3.35 TB/s）
      │  这是"容量墙 + 带宽墙"所在
      ▼
② L2 Cache（50 MB，~12 TB/s，~200 cycle）
      │  命中 L2 的数据不用去 HBM —— 复用类算子的第一层红利
      ▼
③ Shared Memory（228 KB/SM，~4 TB/s，20–30 cycle）
      │  由 TMA / cp.async 异步搬入（不经过寄存器，可与计算重叠）
      │  这里要处理 bank conflict（padding / swizzle）
      ▼
④ 寄存器堆（256 KB/SM，~0 cycle）
      │  MMA 的操作数按规定的 lane layout 分布到 32 个线程的寄存器
      │  （Triton 里由 DotOperandEncodingAttr 表达）
      ▼
⑤ Tensor Core（4 个/SM）
      │  一条 MMA 完成 m16n8k16 的矩阵乘加，累加器留在寄存器里
      ▼
⑥ 写回：累加结果 → 寄存器 →（可选经 SMEM 做 layout 转换）→ L2 → HBM
```

**这六跳对应到前面各节**：①→②→③ 是第 5 节（存储层次）；③ 的搬运由第 3 节的 **TMA** 负责；④→⑤ 是第 4 节（Tensor Core 与 layout 约束）；⑤ 的算力上界是第 8 节（roofline $\pi$）；①的带宽上界是第 6 节（$\beta_{\text{hbm}}$）；而"能给几个 SM 并行跑"由第 2/3 节（SM 数量）与第 7 节（occupancy）决定；跨卡则进入第 9 节的 $\beta_{\text{nvlink}}$（互联细节见 `Communication.md`）。

**（2）三堵墙与对应的三板斧**：

| 墙 | 硬件量 | 什么时候撞上 | 怎么办 |
|---|---|---|---|
| **内存墙** | $\beta_{\text{hbm}}$ | $I < I^*$（逐元素、GEMV、decode、norm） | 算子融合、提高复用（tiling）、量化、减少中间写回 |
| **算力墙** | $\pi$（dense，按精度） | $I > I^*$（大 GEMM、prefill） | 更大 tile、FP8/FP4、warp specialization、消除 padding 浪费 |
| **通信墙** | $\beta_{\text{nvlink}}$ / $\beta_{\text{rdma}}$ | TP/EP/DP 的集合通信 | 通信与计算重叠、拓扑感知切分、缩小跨域流量（见 `Parallel.md`/`Communication.md`） |

**（3）"榨干 8 卡机器"的标准答题框架**（面试实操题，按顺序答）：

```text
第 0 步：先量化目标 —— 定义"榨干"的指标
  · MFU（训练）或 tokens/s 与单卡 roofline 上限之比（推理）
  · 没有基线就没有优化：先跑一个 step，记录各阶段耗时（前向/反向/通信/数据加载）

第 1 步：判断瓶颈类型（roofline + profiling，不要猜）
  · 看 kernel 的算术强度落在拐点哪一侧（compute-bound 还是 memory-bound）
  · nsys（时间线：哪里有空泡）/ ncu（单 kernel：距 roofline 多远）
  · 关键指标：SM occupancy、访存吞吐（% of peak）、Tensor Core 利用率、
    L2 命中率、warp 停顿原因（stall reasons）

第 2 步：按瓶颈对症下药
  · memory-bound → 融合算子、提高 SMEM 复用、向量化、量化中间结果
  · compute-bound → 换精度（FP8）、更大 tile、减少 padding、warp specialization
  · 通信-bound → 通信重叠（DP 反向时的 all-reduce 重叠）、梯度压缩、
                拓扑感知的 TP/EP 切分、NVL72 这类更大的 scale-up 域
  · occupancy 低 → 降寄存器/SMEM 占用、调 block size
  · 数据加载瓶颈 → dataloader 并行度、预取、把预处理放到 GPU/NPU 上

第 3 步：检查常见"低级漏损"
  · 输入 padding 到 8 的倍数（Tensor Core 友好）
  · 变长序列分桶（减少 warp divergence 与 padding 浪费）
  · 避免不必要的 device-host 同步（.item()/print 会把流水线打断）
  · 算子切分是否让 GE 分发到了 Cube（NPU 侧尤其重要）

第 4 步：验证与回归
  · 每改一项单独测一次（不要一次改三处，否则归因不了）
  · 记录数值一致性（加速不能改变收敛）
```

### 3. 具体数值样例

```text
【把一次 7B 模型的训练 step 落到硬件上（8×H100）】
  前向：读权重 + 激活，做大量 GEMM（attention/FFN）
    算术强度高 → compute-bound → 吃 Tensor Core（989 TFLOPS/卡）
  反向：GEMM 数量约为前向 2×（dY = dX·W^T 与 dW = X^T·dY）
    同时要保存/重算激活 → 显存与计算双压力（这就是梯度检查点的动机）
  DP 的梯度 all-reduce：走 NVLink（900 GB/s 双向）
    在大 batch 下可与计算重叠；如果参数多、带宽不足，就会变成"通信墙"
  优化器：读 fp32 动量 + 更新参数 → 纯 memory-bound
    这一段的算术强度极低，所以 ZeRO/FSDP 把它切到多卡 + offload 到 host DRAM
  ⇒ 一个 step 里三种"墙"轮流出现，这就是为什么"只看一个指标"永远说不清瓶颈

【一份"能不能跑"的快速判断（训练）】
  N = 70B，BF16 训练态 16 B/参数 = 1.12 TB 状态
  8×H100（80 GB HBM）= 640 GB → **不够**
  ⇒ 必须 FSDP/ZeRO 切分（每卡 140 GB 状态还是超）→ 需要 offload 或更多卡
  ⇒ 结论：70B 全量微调至少要 16~32 张 H100 级别，或用 LoRA 把优化器状态打掉
```

### 4. 演进：数据通路本身在变

- **搬运路径**：`HBM → 寄存器 → SMEM`（老）→ `HBM → SMEM`（TMA，Hopper）→ 未来 `HBM → SMEM → Tensor Core 直读`（tcgen05/warp-specialized）——**寄存器的中间角色在弱化**；
- **计算单元**：从"CUDA Core 为主"变成"Tensor Core 为主"，Blackwell 上甚至出现"TC 有独立的内存语义（TMEM）"；
- **域的边界**：从"一块卡"扩到"NVL72 一个机柜 = 一个 scale-up 域"，软件上的 TP/EP 切分策略会随之改变。

> **面试一句话总结**：把硬件串起来的答案是"**六跳数据通路 + 三堵墙**"——数据从 **HBM（容量/带宽墙）→ L2（50 MB，复用红利）→ Shared Memory（TMA 异步搬运，需处理 bank conflict）→ 寄存器堆（MMA 操作数按固定 lane layout 分布）→ Tensor Core（一条 MMA 算 m16n8k16）→ 写回**，其中 HBM 对应内存墙（$\beta$，逐元素/GEMV/decode 撞它）、Tensor Core 对应算力墙（$\pi$ dense，大 GEMM/prefill 撞它）、NVLink/RDMA 对应通信墙（TP/EP/DP 撞它）；"榨干 8 卡机器"的正确答法是**四步框架**——先量化目标（MFU/roofline 上限）、再用 nsys+ncu 判断瓶颈类型（不要猜）、然后按"memory-bound 省带宽 / compute-bound 换精度+大 tile / 通信-bound 做重叠 / occupancy 低调 block / 数据瓶颈提并行度"对症下药、最后单变量验证并检查数值一致性；一个训练 step 里三种墙会轮流出现，所以"只盯一个指标"必然说不清瓶颈。

---

## 14. 硬件面试高频题检查清单（含答题框架）

### 1. 现有问题：怎么自测硬件部分准备到位了

硬件知识的"会"和"能答"之间差一次**口述练习**。下面的清单按主题整理，每题给"一句话答案 + 本文对应小节"，可以对着自测：能不看笔记答出 80% 以上，硬件部分就稳了。

### 2. 方法论：八个主题 + 三档难度 + 一个反问技巧

**主题 1：芯片与算力（对应 §1、§2、§3）**

| 问题 | 一句话答案 |
|---|---|
| GPU 和 CPU 的区别？ | 优化目标不同：CPU 压单任务延迟（大 cache + 乱序 + 少量强核），GPU 提吞吐（每 SM 驻留 48–64 warp 用并发掩盖延迟） |
| SM、Warp、SIMD/SIMT 分别是什么？ | SM 是计算单元、内部 4 个 processing block；warp 是 32 线程的锁步执行单位；SIMT 是"写标量代码、硬件把 32 个线程合并宽发射"，比 SIMD 编程简单但要处理发散 |
| Warp 分化（divergence）带来什么性能坑？ | 同一 warp 走不同分支时串行执行两条路径，时间 = $t_A + t_B$，lane 利用率减半 → 变长序列要分桶 |
| GPU 由哪些部分组成？哪些是计算单元？ | GPU→GPC→TPC→SM；**只有 SM 是计算单元**，其内 CUDA Core（标量/向量）+ Tensor Core（矩阵）+ SFU 负责计算 |
| 训练慢通常卡在哪些硬件指标？ | 算力（$\pi$ dense）、显存带宽（$\beta$）、显存容量（$M$）、互连带宽（NVLink/RDMA）、PCIe 主机通道 |
| MFU 是什么、怎么判断算力有没有用满？ | $\text{MFU} = \text{实测 FLOPs/s} \div (\text{dense 峰值} \times \text{卡数})$；用 roofline 判断 compute-bound 还是 memory-bound |

**主题 2：CUDA 编程模型与执行（对应 §7、§5）**

| 问题 | 一句话答案 |
|---|---|
| Grid/Block/Thread 三层为什么要分这么细？ | Block 是"能在 SM 上独立调度 + 内部可同步/共享 SMEM"的单位，Grid/Block/Thread 分别对应"整体任务/可独立调度的工作/并行数据元素" |
| Block 为什么不能跨 SM 拆？ | 因为 SMEM 与 `__syncthreads()` 是 SM 级语义（不是硬件做不到，而是语义要求） |
| `__global__`/`__device__`/`__host__`/`__shared__`/`__constant__`/`__restrict__`/`__managed__` 干嘛用？ | 分别指定"kernel 入口/设备函数/主机函数/共享内存/常量内存/无别名承诺/统一内存" |
| GPU 上常见几种内存，谁快谁慢？ | 寄存器（~0）> Shared/L1（20–30）> L2（~200）> Global/HBM（500–800）；另有 constant/texture 走缓存路径 |
| 合并访问（coalesced）是什么？不合并会怎样？ | warp 32 线程访问连续地址时合并成 128B 级事务；跨步/随机访问会退化成每线程一个事务，带宽掉到几十分之一 |
| 慢 kernel 的排查顺序？ | 先判 bound 类型（roofline）→ 看访存模式（coalescing/bank conflict）→ 看 occupancy → 看停顿原因（stall reasons） |
| Nsight/ncu 通常看什么指标？ | SM occupancy、访存吞吐 % peak、Tensor Core 利用率、L2 命中率、warp stall 原因 |
| **CUDA Stream 是干嘛用的？怎么把计算和拷贝叠起来？** | 见 §11：stream 是**主机侧有序命令队列**；并发靠"**多引擎（SM 阵列 + Copy Engine）+ 多 stream 分块流水**"，且**前提是 pinned memory**（pageable 的 D2H 必须等 kernel 结束） |
| **Hyper-Q 解决什么问题？** | 见 §11：Fermi 只有 **1 条**硬件工作队列 → 多 stream 也会**假串行（false serialization）**；**Kepler（CC 3.5）起 Hyper-Q 提供 32 条硬件工作队列**，允许来自多 stream/多 MPI 进程/多线程的 work 同时排队（**前提是每个 kernel 资源占用别太高**） |
| **CUDA Graphs 什么时候有用？** | 见 §11：当"**单 kernel 时间 < launch 开销（5~10 µs）**"时（大量小 kernel，如 MoE 专家 GEMM），把启动开销从 O(n) 摊成 O(1)（1000 个小 kernel 可从 7~12 ms 压到 ~3 ms） |
| **多进程共享一张卡怎么办？** | 见 §11：用 **MPS（Multi-Process Service）** 配合 Hyper-Q，让多进程的 work 真正并发在 SM 上 |

**主题 3：存储与带宽（对应 §5、§6）**

| 问题 | 一句话答案 |
|---|---|
| HBM 带宽怎么算？ | $(\text{位宽}/8) \times \text{每 pin 数据率}$：A100 = 5120/8 × 3.2 Gbps = 2.048 TB/s |
| HBM 和 GDDR 的区别？ | HBM 用 3D 堆叠 + 超宽位宽换低功耗（每 stack 1024-bit），GDDR 靠高频率；数据中心用 HBM |
| KV cache 怎么算？ | $2 \times L \times H_{kv} \times d_{head} \times S \times B \times s$；Llama-70B 单序列 8K ≈ 2.68 GB |
| 为什么 decode 是 memory-bound？ | 每生成一个 token 要扫一遍全部权重，$\text{tokens/s} \approx \beta/M_{\text{weights}}$ |
| pinned memory 为什么带宽更高？ | 页锁定内存不会被换出，DMA/异步拷贝可以真正重叠；但会占死物理内存 |

**主题 4：Tensor Core 与精度（对应 §4）**

| 问题 | 一句话答案 |
|---|---|
| Tensor Core 和 CUDA Core 区别？ | 前者是矩阵级 MMA 指令（一条 `m16n8k16` = 4096 FLOP），后者是标量 FMA（2 FLOP）；快的原因是操作数复用 $O(k)$ + 二维乘加阵列 |
| 为什么 Tensor Core 对 shape 挑剔？ | MMA 的 $m,n,k$ 只有固定合法组合，不整除就要 padding；decode 的 M=1 利用率极低 |
| TF32/BF16/FP8/FP4 的关系？ | 都是"降精度换吞吐"，指数位尽量保留（动态范围）、尾数位砍掉（精度）；累加仍用 FP32 |
| 结构化稀疏的 2× 是什么？ | 2:4 sparsity（每 4 个权重恰好 2 个非零）；**官方峰值默认 sparse 口径，dense = 一半** |

**主题 5：性能模型（对应 §8）**

| 问题 | 一句话答案 |
|---|---|
| 算术强度怎么算？ | $I = \text{FLOPs}/\text{DRAM 字节}$；逐元素 0.125、GEMV 0.5、$N$ 阶 GEMM ≈ $N/6$ |
| 拐点是什么？ | $I^* = \pi/\beta$；A100≈153、H100≈295、H200≈206、B200≈325 |
| 怎么判断是算力还是带宽瓶颈？ | 比 $I$ 与 $I^*$：小于则 memory-bound（省带宽），大于则 compute-bound（换精度/大 tile） |
| MFU 多少算正常？ | 大模型训练常见 35%~55%；低于 30% 通常有明确可定位的问题（数据加载/通信/低效 kernel） |

**主题 6：型号与选型（对应 §9）**

| 问题 | 一句话答案 |
|---|---|
| A100 和 H100 差多少？ | 算力 312→989 TFLOPS（3.2×）、带宽 2.04→3.35 TB/s（1.6×）→ 拐点 153→295 |
| H800 和 H100 差在哪？ | 算力/显存/带宽基本一致，**NVLink 从 900 砍到 400 GB/s**；单卡 GEMM 无感，TP=8 立现 |
| H200 的意义？ | 同算力、带宽 3.35→4.8 TB/s、显存 80→141 GB，专治长上下文 decode |
| H100 SXM 和 PCIe 能混用吗？ | 不能按同一基准算：PCIe 带宽 2.0 TB/s（差 1.7×）、功率 350W、NVLink 600 GB/s |

**主题 7：昇腾/国产（对应 §10）**

| 问题 | 一句话答案 |
|---|---|
| 达芬奇架构的核心设计？ | AI Core 内 Cube（矩阵）/Vector（向量）/Scalar（标量）按类型拆分，由 GE 图编译器分发 |
| Cube 和 Tensor Core 的差别？ | 都是矩阵单元；但 GPU 上"不用 TC 也能退化跑"，达芬奇上矩阵运算默认走 Cube，算子是否落到 Cube 直接决定性能 |
| 为什么 CUDA 代码不能直接迁移？ | 计算被静态拆分到不同单元，自定义 kernel 要改写 Ascend C/TBE；标准 PyTorch 算子可直接迁移 |

**主题 8：互联（只给量级，细节去 `Communication.md`）**

| 问题 | 一句话答案 |
|---|---|
| NVLink 比 PCIe 强在哪？ | 带宽高一个数量级（H100 NVLink 900 GB/s 双向 vs PCIe Gen5 ×16 128 GB/s），且经 NVSwitch 可做域内全互联 |
| NVSwitch 是什么？ | 把点对点 NVLink 收成全连接 fabric，任意两卡带宽对等（HGX 8 卡域） |
| 什么时候必须上 RDMA？ | 跨节点通信；且要注意 400 Gb/s ≈ 50 GB/s 单向，别和 NVLink 的双向数字直接比 |
| 为什么"掉出 NVLink 域"很贵？ | NVLink 单向 450 GB/s vs RDMA 单向 50 GB/s ≈ 9×（H800 上是 200 vs 50 = 4×） |
| **GPUDirect 有哪几种形态？** | 见 §12：**P2P**（GPU↔GPU，PCIe peer DMA/NVLink）、**RDMA / GDR**（NIC↔显存，NCCL 跨机打满带宽的前提）、**Storage / GDS**（NVMe↔显存，`cuFile`） |
| **网卡是怎么直接读写显存的？（高频）** | 见 §12 四步：**① 显存经 BAR1 aperture 暴露到 PCIe 地址空间 → ② 内核模块 `nvidia-peermem` 把 GPU 页 pin 住交给 RDMA 栈 → ③ 用户态用 CUDA VMM API 分配并经 RM ioctl 创建 ThirdPartyP2P（NV503C）对象 → ④ NIC 用 PCIe P2P DMA 直写 HBM、绕开 host DRAM**；GeForce 上 `ibv_reg_mr` 报 EFAULT 是 NVIDIA **软件分段**限制，不是硬件不支持 |
| **GPU 怎么访问网卡资源？门铃（doorbell）是什么？** | 见 §12：把 NIC 的 BAR（队列/寄存器）**mmap 到用户态**，于是 kernel 也能 load/store；发送动作 = "**填 WQE 写内存 + 一次 doorbell MMIO 写通知硬件**"——**硬件不会持续扫描内存里的队列**，所以必须"敲门"；小消息延迟瓶颈在提交/通知路径（对应"批量 post / inline / 少发信号"的 verbs 调优） |
| **GPU-initiated communication 省了什么？** | 见 §12：省掉每层 **device→host→device** 的往返（~5~10 µs/次），把"发起通信"也搬进 kernel（GPUDirect Async / NVSHMEM）——MoE all-to-all、TP all-reduce 能被计算掩盖的基础 |
| **pinned / zero-copy / UVA / UM 怎么选？** | 见 §12：**pinned 解决"能不能异步"**（分配过多会锁死物理内存）、**zero-copy 解决"能不能不拷贝"**（每次访问走 PCIe，比 HBM 慢 1~2 个数量级）、**UVA 解决"指针要不要分开写"**、**UM 解决"放不下怎么办"**（纯 PCIe 机器上可能因页错误抖动而极慢） |

**三档难度自测法**：

```text
L1（必须会）：说出五个量、SM 里有什么、roofline 两道天花板、TF32/BF16/FP8 关系
L2（区分度）：能当场推 H100 的 67 TFLOPS FP32 / 2.048 TB/s 带宽 / 拐点 295；
            能算 KV cache 与 decode 带宽上限；能说清 MMA 的 layout 约束；
            能说清 stream/Hyper-Q/CUDA Graphs 的适用场景与前提（pinned、32 队列、launch-bound）
L3（加分）：能解释"为什么升级 H100 只快 1.6×"（拐点右移）；
            能说 Blackwell 双 die 与 NVL72 对软件切分的影响；
            能讲清 GDR 打通显存的四步机制（BAR1 → peermem → ThirdPartyP2P → PCIe P2P DMA）
            与门铃为什么必需，以及它和 pinned/zero-copy/UVA/UM 的取舍；
            能对比达芬奇的静态拆分与 SIMT 的取舍
```

**一个反问技巧**：被问"H100 什么水平"时，先反问一句"**是 SXM 还是 PCIe？跑训练还是长上下文 decode？**"——这一问既体现你懂板型差异，也把问题收敛到你能展开的方向。

### 3. 具体数值样例

```text
【60 秒硬件总述模板（面试开场可用）】
"一块 GPU 的核心是 SM，H100 有 132 个 SM、每个 SM 128 个 FP32 lane
 和 4 个 Tensor Core。它同时受两道天花板约束：BF16 dense 算力 989 TFLOPS、
 HBM3 带宽 3.35 TB/s，两者的比值就是拐点约 295 FLOPs/byte——
 算术强度低于它的算子（逐元素、GEMV、decode）是访存受限的，
 优化方向是省带宽；高于它的大 GEMM 才是算力受限的，优化方向是换精度和更大 tile。
 卡间则受第三道墙约束，H100 的 NVLink 双向 900 GB/s，
 而 H800 只有 400，这就是国内做大规模训练时通信优化更值钱的原因。"

【自检清单（逐条确认能讲 2 分钟）】
  □ 能画出 GPU→GPC→TPC→SM 的层级，并指出 SM 是计算单元
  □ 能说出 SM 内 6 类单元及数量（Hopper：128 FP32 / 64 INT32 / 64 FP64 /
    4 TC / 32 LD-ST / 16 SFU + 256KB 寄存器 + 228KB L1-SMEM）
  □ 能当场推 67 TFLOPS FP32 与 989 TFLOPS BF16 的来源
  □ 能说清 MMA 为什么快、为什么挑 shape、warp 如何协作
  □ 能背四层存储的容量/带宽/延迟量级
  □ 能解释 coalescing 与 bank conflict，并给出 padding 解法
  □ 能算 HBM 带宽（位宽×数据率/8）与 KV cache 大小
  □ 能算 roofline 拐点并判断任意算子的 bound 类型
  □ 能算 MFU 并说清 dense/sparse 口径
  □ 能对比 A100/H100/H800/H200/B200 的关键差异
  □ 能说清达芬奇三单元与 SIMT 的取舍
  □ 能答"榨干 8 卡机器"的四步框架
```

### 4. 演进：硬件面试的重心在迁移

- **从"会不会写 kernel"到"懂不懂数据通路"**：随着 Triton/Inductor/编译器自动调参成熟，手写 CUDA 的边际价值下降，而"能判断瓶颈、能做取舍"的价值上升；
- **从"单卡"到"系统"**：NVL72、超节点、液冷、故障域这些系统级话题越来越多地出现在面试里（可延伸读 `Communication.md`、`K8S-KubeRay.md`）；
- **从"GPU"到"异构"**：昇腾/TPU/AMD 的架构差异、以及"同一份代码怎么跨硬件保持效率"，是越来越常见的加分方向。

> **面试一句话总结**：硬件部分按**八个主题**准备——芯片与算力（GPU vs CPU、SM 组成、MFU）、CUDA 执行模型（三层映射、Block 不能跨 SM、divergence、内存种类、coalescing、排查顺序）、存储与带宽（HBM 带宽公式、KV cache 公式、decode 的 $\beta/M$ 上限、pinned memory）、Tensor Core（MMA 原理、shape 挑剔、精度演进、2:4 稀疏）、性能模型（算术强度、拐点、bound 判定）、型号选型（A100/H100/H800/H200/B200 的关键差异与 datasheet 五条铁律）、昇腾（达芬奇 Cube/Vector/Scalar、GE 分发、Ascend C）、互联（NVLink/NVSwitch/RDMA 的量级与"掉域很贵"）；自测标准是**不看笔记答出 80%**，并能当场推三个数（H100 的 67 TFLOPS FP32、2.048 TB/s 带宽、拐点 295 FLOPs/byte）；最后记住那个**反问技巧**——被问"这块卡什么水平"时先确认"**SXM 还是 PCIe、训练还是长上下文 decode**"，这既是专业度体现，也能把话题引到你准备最充分的方向。

---

# 附：硬件速查表

## 一块 GPU 的五个量（选型/对比一眼看全）

| 型号 | 架构 | $M$ | $\beta_{\text{hbm}}$ | $\pi$ BF16 dense (sparse) | FP8 dense (sparse) | NVLink 双向 | TDP | $I^*$ |
|---|---|---|---|---|---|---|---|---|
| V100 SXM2 | Volta | 32 GB HBM2 | 0.90 TB/s | 125（无 2:4） | — | 300 GB/s | 300 W | ~139 |
| A100 SXM | Ampere | 80 GB HBM2e | 2.04 TB/s | 312 (624) | — | 600 GB/s | 400 W | ~153 |
| A800 SXM | Ampere | 80 GB HBM2e | ~2.0 TB/s | 312 (624) | — | **400 GB/s** | 400 W | ~153 |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 (1979) | 1979 (3958) | 900 GB/s | 700 W | ~295 |
| H800 SXM | Hopper | 80 GB HBM3 | ~3.35 TB/s | ≈989 | ≈1979 | **400 GB/s** | ~700 W | ~295 |
| H100 PCIe | Hopper | 80 GB HBM3 | 2.0 TB/s | ~756 (1513) | ~1513 (3026) | 600 GB/s | 350 W | ~378 |
| H200 SXM | Hopper | **141 GB HBM3e** | **4.8 TB/s** | 989 (1979) | 1979 (3958) | 900 GB/s | 700 W | **~206** |
| L40S | Ada | 48 GB GDDR6 | 0.86 TB/s | 362 (733) | 733 (1466) | 无 | 350 W | ~421 |
| B200 SXM | Blackwell | 180 GB HBM3E | 7.7 TB/s | ~2500 (5000) | ~5000 (10000) | 1.8 TB/s | 1000 W | ~325 |
| GB200 (NVL72 内) | Blackwell | 186 GB HBM3E | 8.0 TB/s | ~2500 (5000) | ~5000 | 1.8 TB/s | 1200 W | ~313 |

> 数据来源见文首；$I^*$ 列为按 $\pi/\beta$ **推算**。V100 无 2:4 稀疏，其 125 TFLOPS 为 dense。

## 存储层次速查（H100 口径）

| 层次 | 容量 | 带宽 | 延迟 | 可见范围 |
|---|---|---|---|---|
| 寄存器堆 | 256 KB / SM | ~16 TB/s（每 SM） | ~0 cycle | 线程私有 |
| Shared Memory / L1 | 228 KB / SM（可配） | ~4 TB/s（每 SM） | 20–30 cycle | Block 内共享 |
| L2 Cache | 50 MB（全芯片） | ~12 TB/s | ~200 cycle | 全 GPU |
| HBM3 | 80 GB | 3.35 TB/s | 500–800 cycle | 全 GPU（可被 NVLink/RDMA 访问） |

## SM 内部速查（Hopper）

| 单元 | 每 SM | 每 processing block | 类别 |
|---|---|---|---|
| Warp Scheduler / Dispatch | 4 / 4 | 1 / 1 | 调度 |
| FP32 Core | 128 | 32 | 计算 |
| INT32 Core | 64 | 16 | 计算 |
| FP64 Core | 64 | 16 | 计算 |
| Tensor Core (4th gen) | 4 | 1 | 计算（矩阵） |
| SFU | 16 | 4 | 计算（特殊函数） |
| LD/ST | 32 | 8 | 访存 |
| TMA | 有（新增） | — | 访存（异步大块） |
| 寄存器堆 | 256 KB（64K × 32-bit） | 64 KB | 片上存储 |

## 关键公式速查

| 公式 | 用途 |
|---|---|
| $\text{HBM 带宽} = \dfrac{\text{位宽(bit)}}{8} \times \text{每 pin 数据率(Gbps)}$ | 从位宽推带宽（A100：5120/8 × 3.2 = 2.048 TB/s） |
| $\pi_{\text{FP32}} = \text{SM 数} \times \text{FP32 lane/SM} \times 2 \times f$ | H100：132 × 128 × 2 × 1.98 GHz ≈ 67 TFLOPS |
| $\pi_{\text{dense}} = \pi_{\text{sparse}} / 2$ | 官方峰值口径换算（必须做） |
| $I = \text{FLOPs} / \text{DRAM 字节}$ | 算术强度 |
| $P = \min(\pi,\ I\beta)$，$I^* = \pi/\beta$ | roofline 与拐点 |
| $\text{MFU} = \dfrac{\text{实测 FLOPs/s}}{\pi_{\text{dense}} \times \text{卡数}}$ | 算力利用率 |
| $\text{训练 FLOPs} \approx 6ND$ | 估算一步训练量 |
| $M_{\text{KV}} = 2 L H_{kv} d_{head} S B s$ | KV cache 容量 |
| $\text{decode tokens/s} \lesssim \beta / M_{\text{weights}}$ | decode 带宽上限 |
| $\text{事务数} = \lceil \text{地址跨度}/128\text{B} \rceil$ | coalescing 代价 |
| $\text{SMEM bank} = \text{地址} \bmod 32$ | bank conflict 判定 |

## CUDA 并发与 GPUDirect 速查（§11–§12）

| 项 | 要点 |
|---|---|
| **Stream 本质** | 主机侧**有序命令队列**；同 stream 内有序、**跨 stream 默认无序** |
| **并发门槛（两个，缺一不可）** | ① 硬件工作队列数：Fermi **1 条** → 假串行；**Kepler(CC3.5)+ 32 条**（Hyper-Q）② 资源是否够：**小 kernel 才叠得起来**，大 kernel 占满 SM 就无解 |
| **两个隐形串行** | legacy **default stream** 与所有 blocking stream 互相等待（用 `--default-stream per-thread` / `cudaStreamNonBlocking` 解除）；**pageable 内存并非真异步**（H2D 走内部 staging，D2H 必须等 kernel 结束） |
| **重叠的前提** | **pinned memory**（`cudaMallocHost` / `cudaHostAlloc`） |
| **同步与计时** | 跨 stream 用 **`cudaStreamWaitEvent`（非阻塞）**；不要用 `cudaDeviceSynchronize` 打断流水线；计时用 `cudaEventElapsedTime` |
| **CUDA Graphs** | 当"**单 kernel 时间 < launch 开销（5~10 µs）**"时用（MoE 小 GEMM、逐层小算子）：1000 个小 kernel 可从 7~12 ms → ~3 ms |
| **MPS** | 多进程共享一张卡时，配合 Hyper-Q 让多进程 work 真正并发 |
| **GPUDirect 三形态** | **P2P**（GPU↔GPU）/ **GDR**（NIC↔显存，NCCL 跨机打满带宽前提）/ **GDS**（NVMe↔显存，`cuFile`） |
| **GDR 打通显存的四步** | ① 显存经 **BAR1 aperture** 暴露到 PCIe 地址空间 → ② 内核模块 **`nvidia-peermem`** pin 住 GPU 页交给 RDMA 栈 → ③ 用户态用 CUDA VMM API 分配并经 RM ioctl 创建 **ThirdPartyP2P（NV503C）** 对象（GeForce 报 EFAULT 的根因）→ ④ **NIC 用 PCIe P2P DMA 直写 HBM，绕开 host DRAM** |
| **GPU 访问网卡** | 把 NIC 的 BAR（队列/寄存器）**mmap 到用户态** → kernel 也能 load/store |
| **门铃（doorbell）** | 发送 = "**填 WQE 写内存 + 一次 doorbell MMIO 写**"；**硬件不会持续扫描内存里的队列**，故必须"敲门"；小消息瓶颈在**提交/通知**路径（→ 批量 post / inline / 少发信号；接收侧可改轮询降延迟） |
| **GPU-initiated communication** | kernel 内直接构造 WQE + 写 doorbell（GPUDirect Async / NVSHMEM），省掉每层 **device→host→device** 的 ~5~10 µs 往返 |
| **四类内存一句话** | **pinned 解决"能不能异步"、zero-copy 解决"能不能不拷贝"（走 PCIe，慢 1~2 个数量级）、UVA 解决"指针要不要分开写"、UM 解决"放不下怎么办"** |

## 三条"必须记住的反直觉"

1. **`nvidia-smi` 利用率 98% ≠ 算力打满**：它只表示"有 kernel 在跑"，MFU 可能只有 30%；要用 roofline 判断 bound 类型；
2. **occupancy 高 ≠ 快**：occupancy 只保证 TLP，真正的延迟隐藏还要 ILP，两者通过寄存器用量此消彼长；
3. **升级更快的卡可能只快一点**：拐点右移会让原本 compute-bound 的算子掉到 memory-bound 一侧（A100→H100 只快 1.6× 而非 3.2×），此时该做的是省带宽而不是继续堆算力。

> **关联文档**：互联技术（NVLink/NVSwitch/PCIe/CXL/InfiniBand/RoCE/HCCL/HCCS）见 `Communication.md`——其中**第 7 节**是 GPUDirect 的**网络侧**结论（GDR/GDS 的收益、`NCCL_NET_GDR_LEVEL`、`cuFile`），本文 §12 是它的**GPU 侧机制**（BAR1 / `nvidia-peermem` / ThirdPartyP2P / 门铃 / 四类内存）；并行切分（DP/TP/PP/SP/EP/FSDP 与集合通信原语）见 `Parallel.md`；显存怎么被训练与推理吃掉的完整账见 `Agent-lighting.md` 第 19 节；TQ/Mooncake 的存储分层见 `TQ.md`、`Mooncake.md`。

**§11 / §12 的来源**（除文首通用来源外）：

- CUDA Stream / Hyper-Q：[NVIDIA 官方对 Hyper-Q 的表述（经 SHARCNET 文档转述）](https://helpwiki.sharcnet.ca/wiki/index.php?title=Hyper-Q_/_MPS)——"32 simultaneous, hardware-managed connections (compared to the single connection available with Fermi)"、"false serialization across tasks"；MPS 文档见 NVIDIA *CUDA Multi-Process Service Overview*
- CUDA 并发/重叠与 pinned 语义：[NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)（异步拷贝、stream、event、Graphs 的官方口径）
- GPUDirect 的三形态与历史：[ACM Computing Surveys, *The Landscape of GPU-Centric Communication*](https://dl.acm.org/doi/full/10.1145/3813799)——"With the introduction of GPUDirect RDMA in CUDA 5.0..."
- **BAR1 P2P / `nvidia-peermem` / ThirdPartyP2P 的具体机制**：[mcornea/geforce-gpudirect-rdma](https://github.com/mcornea/geforce-gpudirect-rdma)（开源实验：GeForce 上打通 GDR 需要强制 BAR1 P2P、把 peermem 切到 persistent P2P API、并用 RM ioctl 手工创建 NV503C 对象；含 25GbE 实测 ~2922 MiB/s 打满链路）
- 低延迟 GPU 侧通信（门铃/轮询）实践参考：[eunomia-bpf/basic-cuda-tutorial 低延迟 GPU 包处理示例](https://github.com/eunomia-bpf/basic-cuda-tutorial)

> **口径提醒**：`Communication.md` 第 7 节写"P100+"，本文按资料来源写"**CUDA 5.0 起在 Tesla 卡上提供**"（GPUDirect RDMA 随 CUDA 5.0/Kepler 代引入）；两处不冲突但以"数据中心卡 + 软件分段限制"这个结论为准——**消费卡并非硬件不支持，而是驱动/软件层未开放**。

