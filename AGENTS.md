# 面经整理规范（长期记忆）

本目录下所有面经文档（VLLM.md / RL.md / SGLang.md / Mooncake.md / TQ.md / SmolVLA-VERL.md / Speculative-Decoding.md / Uniagent-Lighting.md / parallel.md 等）的整理，默认采用以下规范。用户只要说"整理 / 补全 / 写面经 / 生成新文档"，就按此规范执行。

## 文档整体结构（生成新面经文档时必选）

新文档的组织方式参考 `VLLM.md`：**先逐个讲核心技术点，再讲架构 / 后端，把技术点串起来**。具体分两大块：

1. **核心技术点部分**：把该主题拆成若干个独立技术点（每个技术点一个小节，用 `## N. 技术点名` 编号），按"从基础到进阶"或"从单点机制到系统机制"的顺序排列（如 VLLM 先 PagedAttention → Continuous Batching → … → CUDA Graph）；每个技术点用下面的**三段式**写法（可再加"演进 / 新版特性"小节）。
2. **架构 / 串讲部分**：放在技术点之后，专门用一个小节把前面所有技术点**串起来**讲——它们如何组合成一个完整系统、数据 / 控制如何在这些组件间流动、真实系统中它们如何协同工作（如 vLLM 的 Engine 分层架构把 PagedAttention / Scheduler / KV Cache Manager / Attention Backend / CUDA Graph 串成一条流水线；parallel.md 的"混合并行"把 DP/TP/PP/EP/SP 串成 3D / 4D 并行）。这一部分要有整体视角，可配分层图 / 组合图，回答"这些点是怎么一起工作的"。

典型文档结构示意（以 parallel.md 为例）：

```text
# 标题 + 引言（该主题的知识框架一句话）
# 一、核心并行技术点
## 1. Data Parallelism（DP）
## 2. Tensor Parallelism（TP）
## 3. Pipeline Parallelism（PP）
## 4. Sequence Parallelism（SP）
## 5. Expert Parallelism（EP）
## 6. Fully Sharded Data Parallel（FSDP）
# 二、并行架构：把所有 P 串起来
## 7. 3D 并行（DP + TP + PP）与 4D/多维混合并行
## 8. 混合切分的完整流程（如何选择并行维度、如何组合配置）
```

> 如果用户明确要求"每个 X 分开讲，最后讲 X 组合起来"（如"每个 P 分开讲，最后讲混合切分"），就严格按"先单点、后组合"的结构执行。

## 三段式结构

每个技术点按三个部分组织，**每部分用完整的几段话描述，不要分太多小标题**（每个技术点最多只用 3 个小标题分别标注三部分）：

1. **现有问题（为什么要提出这个东西）**：先讲清楚该方法要解决的痛点 / 瓶颈 / 背景，最好量化说明（如显存浪费、吞吐损失、延迟等），让读者先建立"没有它会有多糟"的认知。
2. **方法论（这个方法的实现是什么）**：讲清楚实现原理、核心机制、关键数据结构 / 算法流程，可配公式、伪代码、示意图；要讲"它到底是怎么工作的"，而不是罗列名词。
3. **具体样例（一个具体的数值样例讲解）**：用一个**具体的数值样例**逐步演算（例如：10 个 block、3 个请求、具体 token 数、具体显存大小），让读者能跟着数字走一遍，把抽象机制落到可感知的实例上。

## 其他要求

- 每个技术点结尾附一句"面试一句话总结"（用引用块），方便临场背诵。
- 技术点之间用 `---` 分隔。
- 保留原文中的标题层级（如 `## N. 技术点名`），只在标题下填充内容。
- 涉及数学公式用 `$$...$$`，涉及代码 / 数据结构示意用代码块。
- 内容要准确、深入，面向 AI Infra 推理面试，宁可详细不要敷衍。

## 核心代码小节（框架 / 项目类面经必须贴代码）

**整理框架类 / 项目类面经（如 Agent-Lighting、Uniagent、Uniagent-Lighting、verl 相关）时，每个组件的"方法论"部分必须贴出该组件的核心代码片段**，避免"眼高手低"（只懂概念、写不出 / 讲不出真实实现）。要求：

- **每个核心组件至少贴 1 段真实源码**（从仓库源码摘录，注明文件路径，如 `agentlightning/store/base.py`），选取最能代表该组件"怎么工作"的关键方法 / 数据结构 / 接口定义；
- 代码片段用代码块（```python ... ```），**保留关键行，可省略与主题无关的细节**（用 `...` 表示省略），但**方法签名、核心逻辑、关键参数不能编造**——必须与真实源码一致；
- 贴完代码后用 1~2 句话说明"这段代码关键在哪"（对应方法论里的哪个机制点），把代码和文字描述对应起来；
- 优先贴这些类型的代码：**① 接口 / 抽象类定义**（如 `LightningStore` 的接口方法、`Runner` 的基类方法）；**② 核心数据结构的字段**（如 `TrajectoryBuffer` / `Span` / `Attempt` 的 dataclass 字段）；**③ 关键算法流程**（如 `run_generation` 的 prepare/generate/commit/finalize、`_trajectory_to_tq_field_and_tag` 的转换步骤）；**④ 关键配置 / 调用示例**（如 runner 注册配置、`agl.VERL(config)` 的 dict）；
- 代码能讲清楚"这个组件到底是怎么实现的"，而不只是"这个组件是干什么的"。

> 示例：讲 LightningStore 时贴 `store/base.py` 的接口方法签名（enqueue/dequeue/add_otel_span/wait_for_rollouts 等），讲 Gateway 时贴 `run_generation` 的 prepare→generate→commit 骨架，讲本地 WSL 平台化时贴 `platform_local_agent.py` 的 TunnelForwarder 类。

## 新版特性小节（每个技术点末尾必须追加）

在每个技术点正文结束后（"面试一句话总结"之后、`---` 分隔线之前），追加一个 **"新版 vLLM 特性 / 演进"** 小节，介绍该技术点在新版 vLLM（重点 v0.25.0 及之后）中的变化，写法要求：

- 用 `### 4. 新版 vLLM 特性（v0.25.x 演进）` 这样的小标题（沿用三部分的编号延续）；
- 用几段话讲清楚：**旧实现 / 旧架构是什么 → 新版改成了什么 → 对你的理解有什么影响**；
- 必须基于真实版本事实，可参考 vLLM release notes（如 v0.25.0 移除 legacy PagedAttention、MRv2 成为 dense 默认路径、dynamic spec decode + full CUDA graphs、Mamba hybrid prefix caching、FP8 KVCache 等），不确定的不要编造；
- 典型写法示例：PagedAttention 在 v0.25.0 被移除的是 **2023 年的 legacy kernel 实现**，而**分页 KV 内存模型被保留**（由 FlashAttention 系 / FlashInfer 的 paged kernels 接管），因此"vLLM 用分页管理 KV cache"的说法依然成立——这体现了"思想保留、实现更替"的演进规律；
- 如果该点没有显著的新版变化，就写"该机制在 v0.25.x 中仍是核心路径，未见根本性变更"并简述现状。

## 详细程度标准（示例范本：VLLM.md 第 1 点 PagedAttention）

**方法论部分不能只讲"是什么"，要讲清"每一步怎么操作"**：

- 先定义核心数据结构（如 Block Table、`BlockTable = [7, 2, 19]`），说明每个字段 / 每一项的含义；
- 给出逐步操作流程：用伪代码或分步编号（第 1 步做什么、第 2 步做什么……），关键转换用公式表达（如 `logical_block_id = ⌊token_position / B⌋`、`block_offset = token_position mod B`）；
- 涉及映射 / 寻址 / 计算时，画出"输入 → 中间步骤 → 输出"的完整链路（如 `token position → logical block → BlockTable → physical block → block offset`）；
- 说明机制与上下游组件的关系（依赖什么、为谁打基础，如 PagedAttention 为 continuous batching 和 prefix sharing 提供基础）；
- 收益按条列出，每条对应一个具体的机制点（如分块增量分配 / 统一 allocator 降碎片 / block 共享）。

**具体数值样例部分必须"可跟着一步步演算"**：

- 先设定场景参数（block 数、block size、请求的 token 数、显存大小、batch 数等）；
- 然后逐步演算每一步的状态变化：每个请求 / 每个 block / 每个 step 前后的数据结构如何更新（Block Table 怎么变、refcount 怎么变、队列怎么变、显存怎么变），每步给出具体数字；
- 覆盖完整的生命周期：分配 → 增长 → 共享 / 冲突 → 释放，不要只演示一半；
- 结尾给出汇总数字（总收益、对比结果），并附"面试一句话总结"。
