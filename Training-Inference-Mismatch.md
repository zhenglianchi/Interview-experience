# Training-Inference Mismatch（训推不一致）完全指南

> **一句话知识框架**：在 LLM 强化学习里，"采样用的策略"和"训练用的策略"**从来就不是同一个策略**——rollout 由 vLLM/SGLang 在 fp8/bf16 下用一套 kernel 与并行布局采样，训练由 FSDP/Megatron 用另一套 kernel 重新算 log-prob，同一个权重、同一个 token 序列，两边算出的 log-prob 可以差出可观幅度。PPO/GRPO 的 on-policy 假设（$\pi_{\text{old}}$ 就是产生数据的行为策略）被悄悄破坏：轻则有效样本被大面积 clip、梯度方差激增，重则在 off-policy 比例较大时训练崩溃。修正手段分三层：**loss 层**（重要性采样 / 拒绝采样）、**架构层**（MoE 路由复放）、**数据层**（token 保真、不重新分词）。

> 素材来源：`verl-latest`（commit `483512d`）的 `docs/algo/rollout_corr.md`、`docs/algo/rollout_corr_math.md`、`verl/trainer/ppo/rollout_corr_helper.py`；`verl-latest/docs/advance/fully_async.md`；Polar 论文（arXiv:2605.24220）；ROLL Router Replay 官方文档；vLLM 官方博客《No More Retokenization Drift》；AReaL async 文档。配套阅读：`internship.md`（verl 训练全流程）、`RL.md`（GRPO/PPO 基础）、`VLLM.md`、`SGLang.md`。

---

## 1. 训推不一致的本质：on-policy 假设为什么必然被破坏

### 1. 现有问题

标准 PPO 的目标函数是

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_{(s,a)\sim \mu}\left[\min\left(r_t(\theta)A_t,\ \text{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)A_t\right)\right],
\qquad r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)}
$$

这里 $\mu$ 是**产生这批数据的行为策略**，也是 clip 的锚点。朴素实现里大家把 $\mu$ 直接当成 $\pi_{\text{old}}$——即"上一个 checkpoint 在训练引擎里算出的策略"。这在单机、同一引擎、同一精度的年代基本成立；但在 LLM RL 里它**必然不成立**，原因有三：

1. **推理引擎 ≠ 训练引擎**：rollout 走 vLLM/SGLang（PagedAttention、fp8 KV、可能 fp8 权重、TP/EP 切分），训练走 FSDP/Megatron（不同的 attention 实现、不同的 reduction 顺序、不同的 dtype）。同一个权重矩阵，两边前向数值不同。
2. **并行布局不同**：同一层的 logits 在不同 TP/EP 布局下经过不同的 all-reduce 顺序，浮点加法不满足结合律，结果有微小但系统性的差异。
3. **MoE 的路由是离散选择**：Router 的 top-k 由 logits 的 argmax 决定，两个引擎的 logits 只要有一点点差异，就可能在某个 token 上**选中完全不同的专家**——这不是"微小数值误差"，而是**结构性跳变**。

后果很直接：$r_t(\theta)$ 在第一步就偏离 1，clip 被大量触发，$\chi^2$ 散度（二阶矩）把方差放大；如果采样到的恰好是"高权重但其实是垃圾"的样本（toxic tail），梯度方向就被带偏。

> verl 官方文档把这个问题的定位写得很直白：**"Many LLM-RL implementations incorrectly apply PPO by ignoring the actual rollout policy $\pi_{\text{rollout}}$ and assuming the training reference policy $\pi_{\text{old}}$ is the behavior policy. This is mathematically incorrect when $\pi_{\text{rollout}} \neq \pi_{\text{old}}$."** 它同时强调 **"This is not PPO's fault"**——PPO 本身是对的，错的是把 $\pi_{\text{old}}$ 当成行为策略这个假设。

### 2. 方法论

**先用两个散度把问题的两面分开**（来自 rl-collapse 系列博客的分析框架）：

- **偏差（bias）**用**全变差距离（TV distance）**度量：$\mathrm{TV}(p,q)=\frac12\sum_x|p(x)-q(x)|$。它衡量"两个策略给出的分布能差多远"，决定梯度估计**方向**错多少。
- **方差（variance）**用 **$\chi^2$ 散度**度量：$\chi^2(p\|q)=\sum_x \frac{(p(x)-q(x))^2}{q(x)}=\mathbb{E}_q\left[\left(\frac{p}{q}\right)^2\right]-1$。注意它正好等于**重要性采样权重的二阶矩减一**——这就是"权重尾部越厚、方差越大"的数学根源。

**再区分两种漂移**（verl 三策略框架的命名）：

| 漂移 | 含义 | 谁来修 |
|---|---|---|
| **Drift 1：rollout → old** | 推理引擎采样时的策略 $\pi_{\text{rollout}}$ 与训练引擎此刻算出的 $\pi_{\text{old}}$ 不一致（精度/kernel/路由差异） | 重要性采样权重 $w_t=\pi_{\text{old}}/\pi_{\text{rollout}}$ |
| **Drift 2：old → current** | 同一个 batch 内，参数被更新了若干次，$\pi_\theta$ 已经偏离 $\pi_{\text{old}}$ | PPO 的 clip（软信任域） |

**关键洞察**：这两件事**必须分开处理**。PPO 的 clip 只能管 Drift 2；如果把 Drift 1 也丢给 clip，就会得到"数学上不成立的 PPO"。Decoupled PPO（Hilton et al., 2021）给出的正解是把两个角色拆开——**proximal policy**（clip 的锚点）和 **behavior policy**（要被 IS 修正的对象）不必是同一个策略。

### 3. 具体数值样例

设某个 token 上，rollout 侧算出 $\log\pi_{\text{rollout}}=-1.200$，训练侧重算得到 $\log\pi_{\text{old}}=-1.150$（差 0.05，属于"引擎差异"的正常量级）。则

$$
\log r=\log\frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}=0.050,\qquad r=e^{0.050}\approx 1.0512
$$

- 若这一 token 的 advantage $A_t=1$，PPO clip $\epsilon=0.2$ 不会触发（$1.05\in[0.8,1.2]$），看起来"没事"；
- 但把一整条 $T=2000$ token 的序列乘起来：$\prod_t r_t\approx e^{2000\times 0.05}=e^{100}$——**序列级 IS 权重会爆炸到 $10^{43}$**。这正是"token 级看起来温和、序列级方差致命"的原因。
- 加上安全界后（verl `SAFETY_BOUND = 20.0`）：$\log r$ 被 clamp 到 $[-20,20]$，单 token 权重被限制在 $[e^{-20},e^{20}]\approx[2.06\times10^{-9},4.85\times10^{8}]$，序列级同理——**这一步只防溢出，不解决方差**。

> 面试一句话总结：**训推不一致的根因是 rollout 引擎与训练引擎算出的 log-prob 不同（精度/kernel/并行布局/MoE 路由），导致 PPO 里"行为策略 = π_old"的假设失效；它同时带来偏差（TV 距离）和方差（χ² 散度 = IS 权重二阶矩 − 1），必须把 Drift 1（rollout→old）和 Drift 2（old→current）分开修正。**

---

## 2. 误差来源的六类分解

### 1. 现有问题

"训推不一致"是个笼统说法，面试里如果只会说"精度问题"，会被追问到底。实际生产里至少要能拆出五类来源，因为**每一类的修法不同**。

### 2. 方法论

| 类别 | 具体机制 | 典型表现 | 对应修法 |
|---|---|---|---|
| **① 数值精度** | rollout 用 fp8（权重/KV/激活）或 bf16，训练用 bf16/fp32；fp8 的 per-tensor/per-token scale 与训练侧不同 | 系统性偏移，$\log r$ 分布整体平移 | 精度对齐、IS 修正、或在 rollout 侧也开 fp8 训练 |
| **② 算子/kernel 差异** | attention 后端（FlashAttention-2 vs FlashInfer vs Triton）、KV cache 分页、softmax 归约顺序、logits 计算的 GEMM 分块 | 每个 token 都有小抖动，无明显偏向 | IS/TIS；或强制两侧同 kernel（代价大） |
| **③ MoE 路由** | Router top-k 是 argmax 离散选择，两侧 logits 微差 → 选中不同专家 → 输出概率**结构性跳变** | 大部分 token 一致、少数 token 剧变，IS 权重重尾 | **Router Replay（R3）**，架构层固定路由 mask |
| **④ 分词（retokenization）** | 推理返回字符串，训练侧重新分词 → token ID 切分不同 | 训练在"错的 token 序列"上算梯度 | **返回 token ids**（vLLM `return_token_ids`） |
| **⑤ 并行布局/批次组成** | TP/EP/SP 切分与 all-reduce 顺序不同；DP 下同一序列的 micro-batch 组成不同 | 与 batch 组成相关的抖动 | 布局对齐、Decoupled PPO 的 batch size invariance |
| **⑥ 采样语义（logprob 算的是哪个分布）** | 有的引擎返回 **temperature/top-k/top-p 处理之前**的 raw logprob，有的返回处理之后的；有的按**截断后词表**归一化，有的按**全词表**归一化 | **不是"数值小误差"，而是"算的根本不是同一个分布"** | sampling mask / distribution replay；对齐 `logprobs_mode` |

**为什么 ⑥ 是"另一类"问题**：前面五类是"同一个量算得略有不同"，⑥ 是"两边算的量定义都不同"。具体见下面专门展开。

**为什么 MoE 那一类最危险**：其余几类是"连续的小扰动"，MoE 是**离散跳变**。Router 选错专家后，被激活专家的权重、激活函数的非线性都会被放大，token 级 log-prob 差可能直接从 0.05 跳到 1.0 以上——这时 IS 权重 $e^{1.0}\approx 2.7$，序列级更容易爆。

#### 2.1 展开：logprob 语义不一致（最容易被忽略的一类）

这一类值得单独讲，因为**它不能靠 IS/TIS 修正**——IS 修正的前提是"两边算的是同一个量、只是数值不同"，而这里两边算的量定义不同。

**(a) raw logprob vs processed logprob**

vLLM 的默认配置是 `logprobs_mode="raw_logprobs"`（`vllm/vllm/config/model.py` 的枚举），即返回的是 **temperature / top-k / top-p 处理之前**的 logprob。vLLM 官方文档 `docs/usage/v1_guide.md` 自己承认：

> *"returned logprobs do not reflect the final adjusted probabilities used during sampling."*

也就是说：**模型采样时用的是"温度缩放 + top-p 截断 + 重新归一化"后的分布，而 API 返回的 logprob 是原始分布上的值**。拿这个值去和训练侧重算的 logprob 做 ratio，ratio 里混进了采样参数的差异。

**(b) 截断分布的 action space 不一致（sampling mask / distribution replay）**

更进一步：top-k/top-p 会在**截断后的小词表**上重新归一化，而训练侧算 logprob 通常是**全词表 softmax**——两者在"同一个动作上的概率"定义就不同。DeepSeek-V3.2 技术报告 §3.3 的 "**Keep Sampling Mask**" 就是针对这点：**把 rollout 时的采样 mask 保存下来，训练时在同一个 mask 上做 log_softmax**，从而让 action space 一致。vLLM 已把它做成显式特性（`docs/training/sampling_mask.md`）：

```bash
# vLLM：返回采样 mask，并用处理后的 logprob
--return-sampling-mask --logprobs-mode processed_logprobs
```

**(c) vLLM 与 SGLang 的默认语义本来就不同**

| 引擎 | 默认 logprob 语义 |
|---|---|
| **vLLM** | `raw_logprobs`：**temperature / top-k / top-p 之前**的 full-vocab logprob |
| **SGLang** | **post-temperature 但 full-vocab**（`SGLANG_RETURN_ORIGINAL_LOGPROB` 默认 False）；只有在开 `return_sampling_mask` 时才给出 nucleus 归一化后的值 |

**结论：在带 top-p/top-k 的请求下，vLLM 与 SGLang 返回的 logprob 语义天然不同。** 所以"我们用了 vLLM，训练也用 vLLM 的 logprob"并不自动解决问题。

**(d) 确定性对齐（把差异从源头压到最小）**

- **SGLang `--rl-on-policy-target fsdp`**：官方提供的"直接对齐 FSDP"方案，会自动打开 `--enable-deterministic-inference`，并且
  - 用 `log_softmax` 而不是 `log(softmax(x))` 来算 logprob（数值更稳）；
  - **故意做 bf16 round-trip**：`logits.bfloat16().div(temps).bfloat16()`；
  - 把 lm_head **钉死为 bf16**（即使开了 `--enable-fp32-lm-head`）。
- **`VLLM_BATCH_INVARIANT=1`**：vLLM 的 batch-invariance 模式，拒绝不支持 batch-invariance 的后端（用于保证"同一序列在不同 batch 组成下结果一致"）。
- **SGLang `--enable-deterministic-inference`**：固定 batch 内 reduction 顺序（只支持 FlashInfer/FA3/Triton）。

SGLang 官方 RL 文档对这件事有一句话总结，可以直接背：

> *"Even when weights are identical, token probabilities can drift, silently breaking the on-policy assumption. This is the training–inference mismatch problem."*

**(e) 一个必须记住的配置坑**：verl 里 `actor_rollout_ref.rollout.calculate_log_probs` **默认是 `False`**。忘了打开它，IS/RS 就没有输入数据——所有"指标看起来正常"都是假的。排查 mismatch 的第一步永远是这个开关。

### 3. 具体数值样例

**（a）真实测量：同一 checkpoint、同一个 token，两边 logprob 差多少**

ByteDance + UVA 的工作《Diagnosing Training Inference Mismatch in LLM Reinforcement Learning》（arXiv **2605.14220**，代码 `github.com/verl-project/vexact`）给出了一组逐字可引的测量——**Qwen3-8B、bf16、同一个 checkpoint**，在 token `"that"` 处：

| 量 | 值 |
|---|---|
| $\log\pi_{\text{rollout}}$（vLLM） | **−0.694** |
| $\log\pi_{\text{train}}$（训练引擎重算） | **−0.827** |
| $\delta = \log\pi_{\text{train}}-\log\pi_{\text{rollout}}$ | **−0.133** |
| 该位置的 argmax | **翻转**（trainer 的 top-1 是 `":\n\n"`，logprob −0.577） |

**两个关键观察**：

1. **per-batch 的平均 $\lvert\delta_t\rvert$ 很小，但 max $\lvert\delta_t\rvert$ 可达 ~1.0**——这就是"平均值骗人、尾部致命"的实证。和前面 $\chi^2$ 的判断完全一致。
2. **该位置的 argmax 会翻转**——说明不一致不只是"概率值不同"，而是"最高概率的 token 都换了"。

**（b）MoE 与 dense 的差距（可引用的硬数字）**

Ma et al.《Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers》（arXiv **2510.11370**，北大 + 小米）实测 Qwen3 系列训推之间的 K3 KL：

| 模型 | 类型 | 训推 K3 KL |
|---|---|---|
| Qwen3-30B-A3B | **MoE** | **1.535e-3** |
| Qwen3-8B | dense | **6.4e-4** |

**MoE 的路由分歧统计**（同文）：

- **约 10% 的 router（每 token 每层）选了不同的专家**；
- **94% 的 token 至少有一层选了不同专家**；
- 序列平均每个 token 约 **6 个 router** 不一致。

**最能说明问题的一条**：**同一个框架（Megatron）对同一批序列做两次前向，KL 也有 8.4e-4**——也就是说，这**不是"框架间差异"，而是 MoE 路由本身在不同 batch 组成/kernel 下就不可复现**。

**R3（路由复放）的效果**：K3 KL 从 1.5e-3 → **7.5e-4**（接近 dense 模型的 6.4e-4），极端 token 频率降一个数量级，rollout 阶段额外开销 **<3%**。

**（c）FP8 rollout 的退化曲线**

Jet-RL（arXiv **2601.14243**，NVIDIA/MIT/UC Berkeley）给出了精度组合的退化边界：

| 配置 | 结果 |
|---|---|
| BF16 训练 + BF16 rollout | 基准 |
| BF16 训练 + **FP8 rollout** | **8K 序列长度开始明显退化；16K 时仅 20 步就 accuracy collapse** |
| Qwen2.5-7B / MATH / 8K | "BF16-Train-FP8-Rollout **Did not converge**"；Jet-RL 平均 55.9 vs BF16 56.9 |
| Llama3.1-8B / 8K | BF16 avg 23.2 → BF16-train-FP8-rollout avg **13.0**（GSM8k 49.0→20.6），Jet-RL 25.2 |

**归因结论（很重要）**：主要误差来自**量化/反量化本身**，而**不是** GEMM kernel 实现——q/kernel 误差同量级（cuBLAS fp8 0.00068 vs DeepGEMM 0.00036~0.00065）。这正是"统一精度流"（让推理的计算图成为训练前向的子图）成为主流方案的原因。

**（d）一个"训练直接崩掉"的极端案例**

vLLM 博客《IsoExec: Unified Execution to Eliminate Trainer-Inference Mismatch in SkyRL》（2026-08）转述了 Fireworks 的报告：**GLM-5.2 的 train–inference KL ≈ 0.013 时，PPO clipping 丢掉了约 45% 的 token，reward 在约 step 20 崩溃；换成 bitwise-aligned 版本后 0 个 clipped token 且训练稳定。**

> 注：这条是 vLLM 博客的**转述**（未直接读到 Fireworks 原文），引用时宜标注来源。

**（e）把上面的数字代进"权重会爆"的推导**

取 MoE 场景实测的 $\lvert\log r\rvert\approx1.5\times10^{-3}$（K3 KL 量级）、$T=2000$ token：

$$
\sum_t \log r \approx 2000\times1.5\times10^{-3}=3.0
\ \Longrightarrow\ \rho_{\text{seq}}=e^{3}\approx20.1
$$

**序列级权重已经到 20**——远超典型的序列级阈值 2.0~10.0，也就是说**即使 MoE 场景下每个 token 的偏差只有千分之几，序列级 IS 权重照样会撞阈值**。这从数量上解释了为什么"序列级比 token 级敏感得多"，也解释了论文里那句"KL 不足以预警"（KL 是均值，而决策看的是尾部）。

> 面试一句话总结：**训推不一致要拆成六类来源——精度、算子/kernel、MoE 路由（离散跳变最危险）、分词 drift、并行布局、**采样语义（logprob 算的是哪个分布）**；MoE 与 dense 的训推 K3 KL 实测差一倍（1.535e-3 vs 6.4e-4），且同框架跑两次前向也有 8.4e-4 的 KL——说明这是 MoE 路由本身的不可复现性；FP8 rollout 在 8K 长度就退化、16K 时 20 步内崩，且误差主因是量化本身而非 GEMM kernel。**

---

## 3. 三策略框架与 Decoupled PPO

### 1. 现有问题

朴素实现的写法是：rollout 采样 → 用**同一个**模型算一次 `old_log_prob` → 训练时用 `exp(logp - old_logp)` 当 ratio。问题在于：

- 这个 `old_log_prob` 是**训练引擎**算的，而数据是**推理引擎**采的——两者不是一个分布；
- 一旦引入异步/流水（rollout 用旧 checkpoint），行为策略和 $\pi_{\text{old}}$ 差得更远；
- PPO 的 batch size 敏感性：把多个 worker 的数据聚成一个大 batch，等价于改变了"有效行为策略"，clip 的语义随之漂移。

### 2. 方法论

**三个策略，各司其职**（verl `docs/algo/rollout_corr_math.md` §2.1）：

| 策略 | 符号 | 含义 | 何时产生 |
|---|---|---|---|
| 行为策略 | $\pi_{\text{rollout}}=\mu$ | 真正采出这批数据的分布（vLLM/SGLang 那一侧） | rollout 阶段 |
| 近端策略 | $\pi_{\text{old}}=\pi_{\text{prox}}$ | clip 的锚点，控制更新幅度 | decoupled 模式：训练轮开始时 `actor.compute_log_prob()` 算一次；bypass 模式：直接令 $\pi_{\text{old}}=\pi_{\text{rollout}}$ |
| 当前策略 | $\pi_\theta$ | 正在被梯度更新的策略 | 训练中每步变化 |

**目标函数（Decoupled PPO）**：

$$
L_{\text{DecoupledPPO}}(\theta)=-\mathbb{E}_{(s,a)\sim\mu}\left[w_t\cdot\min\left(r_t(\theta)A_t,\ \text{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)A_t\right)\right]
$$

其中

$$
w_t=\frac{\pi_{\text{prox}}(a_t\mid s_t)}{\mu(a_t\mid s_t)}=\frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}
\quad(\text{修正 Drift 1}),\qquad
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{prox}}(a_t\mid s_t)}=\frac{\pi_\theta}{\pi_{\text{old}}}
\quad(\text{修正 Drift 2})
$$

**两个关键性质**：

1. $w_t$ **不需要 stop-gradient**——因为 $\pi_{\text{prox}}$ 在整轮训练里是冻结的，$w_t$ 是常数。这是 decoupled PPO 相比"在 REINFORCE 里套 IS"的一大工程优势。
2. **Batch size invariance**：clip 的锚点是 $\pi_{\text{prox}}$，与"数据从哪来、怎么聚合"无关，因此改变 batch 组成不会改变信任域语义。

**Bypass 模式（两策略）**是工程上的降本方案：直接令 $\pi_{\text{old}}=\pi_{\text{rollout}}$，**跳过 `actor.compute_log_prob()` 那次前向**。代价是不再区分 proximal 与 behavior；此时若用 PPO-clip，则 ratio $=\pi_\theta/\pi_{\text{rollout}}$ **本身就包含了 IS**，不需要额外权重；若用 REINFORCE，则需要显式 IS 权重 $\pi_\theta/\pi_{\text{rollout}}$（在 loss 里现算）。

> 官方配置对照（`rollout_corr.md` "Operation Modes" 表）：
> - **Decoupled**：`bypass_mode: false`，PPO loss，额外一次 `actor.compute_log_prob()`；
> - **Bypass + PPO-clip**：`bypass_mode: true`, `loss_type: ppo_clip`（默认），ratio 即 IS；
> - **Bypass + REINFORCE**：`bypass_mode: true`, `loss_type: reinforce`，显式 IS 权重、无 PPO clip。

### 3. 具体数值样例

设某 token：$\log\pi_{\text{rollout}}=-1.20$，$\log\pi_{\text{old}}=-1.15$，$\log\pi_\theta=-1.10$。则

$$
\log w=\log\pi_{\text{old}}-\log\pi_{\text{rollout}}=0.05\Rightarrow w=e^{0.05}=1.051
$$
$$
\log r=\log\pi_\theta-\log\pi_{\text{old}}=0.05\Rightarrow r=e^{0.05}=1.051
$$

- **Decoupled**：梯度被 $w\cdot r\cdot A$ 缩放，$1.051\times1.051\approx1.105$；
- **Bypass + PPO-clip**：$\log r'=\log\pi_\theta-\log\pi_{\text{rollout}}=0.10$，$r'=1.105$，clip 仍不触发，**结果与 decoupled 一致**（两步 IS 合并成一步）；
- **Bypass 省下的成本**：跳过 `compute_log_prob` 前向，按 7B 模型、序列 4k 估算，单次前向约占一个 PPO 迭代训练时间的 20~30%——这是 bypass 模式在生产里被默认开启的原因。

> 面试一句话总结：**正解是 Decoupled PPO 的三策略框架——π_rollout（行为）、π_old（clip 锚点）、π_θ（当前）；w=π_old/π_rollout 修 Drift 1（梯度上不传、是常数），PPO ratio r=π_θ/π_old 修 Drift 2；工程上常用 bypass 模式令 π_old=π_rollout，省掉一次 compute_log_prob 前向。**

---

## 4. 重要性采样修正：Token-TIS / Seq-TIS / IcePop

### 1. 现有问题

理论上的无偏 IS 权重 $w=\prod_t \frac{\pi_{\text{old}}(a_t)}{\pi_{\text{rollout}}(a_t)}$ 在长序列上方差爆炸（第 1 点的数值样例已经算过：$T=2000$、$|\log r|=0.05$ 时权重 $e^{100}$）。必须做截断或加权，但要**清楚每一步牺牲了什么**。

### 2. 方法论

**聚合层级（聚合维度的选择就是偏差-方差权衡）**：

- **Token 级**：$\rho_t=\frac{\pi_{\text{old}}(a_t)}{\pi_{\text{rollout}}(a_t)}$，每个 token 独立截断。**有偏、低方差**（每个权重都被单独限制住）。
- **序列级**：$\rho_{\text{seq}}=\prod_t \rho_t=e^{\sum_t \log\rho_t}$，整条序列一个权重。**无偏、高方差**（对离群 token 极敏感）。

**verl 的四步处理流水线**（`docs/algo/rollout_corr.md`「How IS Weights are Processed」+ `rollout_corr_helper.py:compute_rollout_correction_weights`）：

```python
# verl-latest/verl/trainer/ppo/rollout_corr_helper.py:522-657（节选）
def compute_rollout_correction_weights(
    log_ratio: torch.Tensor, response_mask: torch.Tensor,
    rollout_is: str = "token", rollout_is_threshold: str | float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute importance sampling weights to correct for off-policy distribution shifts."""
    rollout_is_threshold_upper, rollout_is_threshold_lower = _parse_rollout_is_threshold(rollout_is_threshold)
    use_icepop = rollout_is_threshold_lower is not None

    if rollout_is == "token":
        # Stage 1: safety bound（防溢出）
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)   # SAFETY_BOUND = 20.0
        raw_rollout_is_weights = torch.exp(log_ratio_safe)
    elif rollout_is == "sequence":
        log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        raw_rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)

    # Stage 2: padding 归零（保证聚合正确）
    raw_rollout_is_weights = raw_rollout_is_weights * response_mask

    # Stage 3: TIS 上截断 / IcePop 双侧置零
    if not use_icepop:
        rollout_is_weights = raw_rollout_is_weights.clamp(max=rollout_is_threshold_upper)      # TIS
    else:
        token_kept_mask = (raw_rollout_is_weights >= rollout_is_threshold_lower) & \
                          (raw_rollout_is_weights <= rollout_is_threshold_upper)
        rollout_is_weights = torch.where(token_kept_mask, raw_rollout_is_weights,
                                         torch.zeros_like(raw_rollout_is_weights))            # IcePop

    # Stage 4: detach（IS 权重改的是测度，不是目标函数）
    rollout_is_weights = rollout_is_weights.detach()
    ...   # 可选：batch normalize 到 mean=1.0（截断之后再归一化）
    return rollout_is_weights, metrics
```

**四步各自的语义**：

1. **安全界**：$\exp(\text{clamp}(\log r,-20,20))$ → 权重被限制在 $[2.06\times10^{-9},4.85\times10^{8}]$。**只防数值溢出，不解决统计问题**（上下界跨度仍有 17 个数量级）。
2. **截断（TIS, Truncated IS）**：`clamp(max=threshold)`，只截上界、不截下界——保留无偏性（小权重不该被砍）。典型阈值：token 级 1.5~5.0，序列级 2.0~10.0。
3. **padding 归零**：$\times$ `response_mask`，否则 padding 位置的权重会污染聚合。
4. **`detach()`**：这一行很关键——IS 权重是"测度变换系数"，必须当常数用；若让它带梯度，目标函数就被改了（verl 代码注释直接引用了数学文档 §3.2.2）。

**IcePop** 是 TIS 的一个变体：给一个 `"lower_upper"` 字符串（如 `"0.5_5.0"`），**区间外的权重直接置 0**（不是截断到边界）。区别在于：TIS 把超界样本拉回边界（保留样本、降低其影响），IcePop 把超界样本**丢弃**（连权重都不给）。IcePop **不改 `response_mask`**——它只改 IS 系数，这是它与"拒绝采样"的本质区别。

**Batch normalization**：`rollout_is_batch_normalize=True` 时把权重归一到 batch 内均值 1.0（token 级对所有 token 权重归一，序列级对每序列均值归一再跨序列平均），**且在截断之后做**，以保住截断语义；它只改权重值，不影响拒绝采样。

### 3. 具体数值样例

设一个 batch 有 4 个 token，$\log r$ 分别为 $[0.01,\ 0.5,\ 3.0,\ -0.2]$，阈值 token-TIS $=2.0$：

| token | $\log r$ | 原始 $e^{\log r}$ | TIS 后 | IcePop(`"0.5_2.0"`) 后 |
|---|---|---|---|---|
| 1 | 0.01 | 1.010 | 1.010 | 1.010 |
| 2 | 0.50 | 1.649 | 1.649 | 1.649 |
| 3 | 3.00 | 20.086 | **2.000**（截断） | **0.000**（区间外置零） |
| 4 | −0.20 | 0.819 | 0.819 | 0.819 |

- 均值（TIS）：$(1.010+1.649+2.000+0.819)/4=1.3695$；若不截断则为 $(1.010+1.649+20.086+0.819)/4=5.891$——**单个离群 token 把均值抬高 4.3 倍**，这就是"权重集中、有效样本数下降"的样子。
- **有效样本量（ESS）**：$\text{ESS}=\dfrac{1}{\overline{w^2}}$（权重归一后）。上面 TIS 权重归一化为 $[0.738,1.204,1.460,0.598]$，$\overline{w^2}=(0.545+1.450+2.132+0.358)/4=1.121$，$\text{ESS}=0.892$——健康（官方建议 $>0.3$）。
- 若不做 TIS，归一化后权重 $[0.171,0.280,3.409,0.139]$，$\overline{w^2}=(0.029+0.078+11.62+0.019)/4=2.937$，$\text{ESS}=0.340$——**已经逼近告警线**。一个 token 就能把 ESS 从 0.89 打到 0.34。

> 面试一句话总结：**IS 修正是"安全界 → padding 归零 → TIS 上截断（或 IcePop 双侧置零）→ detach →可选 batch normalize"五步；token 级低方差有偏、序列级无偏高方差；`detach()` 不能省，因为 IS 权重改的是测度不是目标函数。**

---

## 5. 拒绝采样：K1/K2/K3 距离与 seq_sum/mean/max

### 1. 现有问题

截断（TIS）有个隐患：**它假设"超界样本只是权重太大，信息仍有价值"**。但 rl-collapse 系列博客（Part 3「Toxic Tails and Length Traps」）指出，在 mismatch 严重的场景下，高权重样本往往是**垃圾样本或对抗样本**——把它截断到边界、照样参与训练，等于给错误方向配了一个放大系数。此时正确做法是**直接拒绝**（hard trust region），而不是软截断。

### 2. 方法论

verl 用**一族散度估计量**做拒绝判据（`rollout_corr_helper.py:compute_rollout_rejection_mask`，line 197–413），核心是三种对 $\log r$ 的变换：

$$
\underbrace{k_1 = -\log r}_{\text{KL 方向}},\qquad
\underbrace{k_2 = \tfrac12(\log r)^2}_{\chi^2/2},\qquad
\underbrace{k_3 = e^{\log r}-1-\log r}_{\text{K3 散度}}
$$

三者的性质差异：

| 估计量 | 无偏/非负 | 对 $- \log r$（rollout 更自信） | 对 $+ \log r$（训练更自信） | 适用 |
|---|---|---|---|---|
| $k_1=-\log r$ | 可正可负（KL 的直接估计） | 敏感 | 敏感但符号相反 | 双侧阈值 `"lower_upper"` |
| $k_2=\frac12(\log r)^2$ | **恒非负** | 对称敏感 | 对称敏感 | 只给上界即可 |
| $k_3=e^{\log r}-1-\log r$ | **恒非负**；期望上等于 KL | 增长快（指数） | 增长慢（二次） | 只给上界；**小 KL 更稳** |

**聚合方式**（`*` 代表 k1/k2/k3）：

- `token_k*`：逐 token 判据；
- `seq_sum_k*`：序列内**求和**（长序列天然更大，等价于"长序列更严"）；
- `seq_mean_k*`：序列内**求均值**（对长度不敏感）；
- `seq_max_k*`：序列内**取最大**（只盯最坏 token，最激进，只有 k2/k3 支持）。

判定逻辑：多个判据用逗号连接（如 `"token_k1,seq_mean_k3"`），**所有判据都必须通过**（logical AND），最后

```python
# verl-latest/verl/trainer/ppo/rollout_corr_helper.py:262-265, 412
log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
token_k1 = -log_ratio_safe
token_k2 = 0.5 * log_ratio_safe**2
token_k3 = torch.exp(log_ratio_safe) - 1.0 - log_ratio_safe
...
modified_response_mask = (response_mask * final_mask).to(dtype=response_mask.dtype)
```

注意最后一行：**RS 改的是 `response_mask`**（把越界 token 置 0），而 IS 改的是**权重**——这是两个正交机制，可以单独用、也可以叠加。

**为什么 K3 在小 KL 下更稳**：把 $k_3$ 在 $\log r=0$ 附近展开，$e^x-1-x=\frac{x^2}{2}+\frac{x^3}{6}+\cdots\approx k_2$；但当 $x$ 为负（rollout 比训练自信）时，$k_3$ 的增长比 $k_2$ 慢（指数项 $e^x\to0$，只剩 $-1-x$，线性），因此**对"训练侧低估"这一类不敏感**，不会误杀。这正是官方把 `decoupled_k3_rs()` 作为"small KL 更稳"预设的原因。

**「拒绝」vs「截断」怎么选**（官方建议的判据）：

- **Seq-TIS（只截断）**：最大化信息效率，从所有样本里榨信号；**数据干净、mismatch 中等**时用。
- **Seq-MIS（拒绝）**：最大化安全性，相当于一个硬信任域过滤器；**mismatch 严重、高权重样本大概率是垃圾**时用。

### 3. 具体数值样例

同一条序列 3 个 token，$\log r=[0.02,\ 0.30,\ 1.10]$：

| token | $k_1=-\log r$ | $k_2=\frac12(\log r)^2$ | $k_3=e^{\log r}-1-\log r$ |
|---|---|---|---|
| 1 | −0.02 | 0.0002 | 0.0002 |
| 2 | −0.30 | 0.045 | 0.0498 |
| 3 | −1.10 | 0.605 | **1.003** |

- `seq_sum_k1` = $-1.42$，配 `"lower_upper"` 阈值如 `0.5_2.0`（比值界）→ 通过；
- `seq_mean_k2` = $(0.0002+0.045+0.605)/3=0.2167$，若上界设 0.5 → 通过；
- `seq_max_k3` = $1.003$，若上界设 0.5 → **整条序列被拒**。

**阈值敏感性**：把阈值从 0.5 提到 2.0，第三条序列就会被保留；而 `seq_max_k*` 只看最坏 token，所以**阈值对它的影响最剧烈**——这也是官方把它归为"最激进"的原因。工程上通常先用 `seq_mean_k3` 起手（对长度和单点都不敏感），确认稳定后再考虑收紧。

> 面试一句话总结：**拒绝采样用 K1（−log r）/K2（½log²r）/K3（e^logr−1−logr）三种散度估计量做硬信任域，聚合可选 seq_sum/mean/max，多判据 AND；它改 response_mask 而 IS 改权重，两者正交；mismatch 严重时"拒绝"优于"截断"，因为高权重样本往往是垃圾。**

---

## 6. MoE 路由不一致与 Router Replay（R3）

### 1. 现有问题

MoE 模型每层的 Router 对每个 token 做 top-k 专家选择。RL 里同一份权重被三个角色使用（rollout policy / old policy / training policy），理想情况三者路由一致，但：

- **训练 vs 推理**：算子实现、精度、并行布局不同，即使权重相同，同一输入也可能选出**不同的 top-k 专家**；
- **多次梯度更新内部**：随着 mini-batch 不断更新，路由随权重漂移。

**危害被离散选择放大**：选中的 expert 不同 → 该 token 的输出概率出现明显偏差 → IS 比值被放大 → PPO/GRPO 的有效样本被严重 clip 或方差激增 → off-policy 比例大时训练崩溃。而且**这一层靠 IS/TIS 在 loss 层修不动**——因为偏差是结构性的、稀疏且巨大。

### 2. 方法论

**核心思路（ROLL 官方文档原文）**："与其在 loss 层做事后修正，不如在模型架构层固定路由 mask，让训练侧直接复用一份'参考路由'，从源头去掉这一差异。"

**复放公式**：训练 forward 时，把原本由 router logits 决定的 top-k mask 替换为外部传入的参考 mask $I_{\text{ref}}$，**再用训练侧的 logits $s_{\text{train}}$ 重新归一化**得到专家权重：

$$
g_i=\frac{I_{\text{ref},i}\cdot\exp(s_{\text{train},i})}{\sum_j I_{\text{ref},j}\cdot\exp(s_{\text{train},j})}
$$

两个要点：

1. **选哪几个专家由 $I_{\text{ref}}$ 决定**，不再由训练侧 argmax 决定；
2. **softmax 仍作用在训练侧 logits 上**，所以 router 权重的梯度仍能正常回传（这是"复放但不冻结梯度"的关键）。

**R2 vs R3**：

| 模式 | $I_{\text{ref}}$ 来源 | 解决什么 | ROLL 状态 |
|---|---|---|---|
| **R2（Vanilla Routing Replay）** | 训练引擎自己用 old policy 跑一次 forward 记录的路由 | 只解决"训练侧多次更新内部的路由漂移"（policy staleness）；第一个 mini-batch 时 $\theta=\theta_{\text{old}}$，等价 on-policy | **未实现**（`mode: R2` 抛 `NotImplementedError`） |
| **R3（Rollout Routing Replay）** | **推理引擎在 rollout 时记录的路由** | 同时消除 training-inference 路由差异 + 限制多次更新中的漂移 | **当前推荐**（SGLang 推理 + Megatron 训练） |

**R3 的端到端流程**：

```
┌─────────────────────┐                    ┌──────────────────────┐
│  SGLang Rollout     │                    │  Megatron Training   │
│  generate(...)      │                    │  forward()           │
│  └─ MoE Router top-k│                    │  └─ MoERouterReplay  │
│  └─ 导出索引        │ ───[seq,layers,k]──►│  └─ 用导出的索引      │
│                     │      batch data    │     参与 forward      │
│  返回 routed_experts│ ──────────────────►│  forward → backward  │
└─────────────────────┘                    └──────────────────────┘
```

1. **采样阶段**：SGLang 生成 token 时额外记录每一层 MoE 的 top-k 专家，形成 `[seq_len, num_layers, top_k]` 的 `routed_experts` 张量并随响应返回；
2. **数据搬运**：rollout 后处理把 `routed_experts` 挂到 batch 的每条样本上，沿标准数据通路（DP / mini-batch / micro-batch）流向训练 worker；
3. **训练阶段**：Megatron 侧 forward 时用复放 mask 替换 router 的 top-k 选择。

**已知限制**（ROLL 文档明确标注）：当前只实现 R3；**R3 与 sequence_packing 的组合暂不支持**，不要同时启用。

**独立的旁证**：ICML 2026 有一篇专门的工作《Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers》，说明这个问题在工业界已被系统性识别。

### 3. 具体数值样例

设某层 MoE 有 4 个专家，router logits 在两侧分别为

- 推理侧（SGLang）：$s_{\text{rollout}}=[2.0,\ 1.9,\ -1.0,\ -2.0]$ → top-2 = {E1, E2}
- 训练侧（Megatron）：$s_{\text{train}}=[1.8,\ 2.1,\ -1.1,\ -1.9]$ → top-2 = {E2, E1}（顺序不同，集合相同，**这种情况无差异**）

真正的危险是 logits 接近时的**翻转**：$s_{\text{rollout}}=[1.90,\ 1.85,\ 1.80,\ -2.0]$ → top-2 = {E1, E2}；$s_{\text{train}}=[1.88,\ 1.87,\ 1.95,\ -2.0]$ → top-2 = {E3, E1}——**E2 被 E3 换掉**。

**复放后**：$I_{\text{ref}}=\{E1,E2\}$，即使训练侧 $s_{\text{train}}$ 认为 E3 更高，也只允许 E1/E2 参与归一化：

$$
g_{E1}=\frac{e^{1.88}}{e^{1.88}+e^{1.87}}=0.5025,\qquad g_{E2}=\frac{e^{1.87}}{e^{1.88}+e^{1.87}}=0.4975,\qquad g_{E3}=0
$$

对照**不复放**的情形（$s_{\text{train}}$ 自己选 {E3,E1}）：$g_{E3}=\frac{e^{1.95}}{e^{1.95}+e^{1.88}}=0.5174$，$g_{E1}=0.4826$，$g_{E2}=0$——**E2 的贡献被完全抹掉、E3 拿到 51.7% 权重**。两个不同的专家子集，输出分布差异远大于 logits 的 0.07 差异，这正是"离散跳变"的量化含义。

**真实的量化规模**（arXiv 2510.11370 实测，比上面的构造例子更有说服力）：

| 指标 | 实测值 |
|---|---|
| 路由分歧比例（每 token 每层） | **约 10%** |
| 至少有一层选了不同专家的 token 比例 | **94%** |
| 每个 token 平均有多少个 router 不一致 | **约 6 个** |
| MoE K3 KL（Qwen3-30B-A3B） | 1.535e-3 |
| dense K3 KL（Qwen3-8B） | 6.4e-4 |
| **同框架两次前向的 KL** | **8.4e-4** |

最后一行是**决定性的证据**：同一框架、同一批序列、跑两次前向，KL 也有 8.4e-4——**说明 MoE 路由的不可复现性来自路由本身（kernel 的 tie-break、batch 组成导致的 launch grid 变化），而不是"两个框架实现不同"**。这也解释了为什么"用同一个框架"并不能解决 MoE 的训推不一致。

**R3 复放后的改善**：K3 KL 从 1.5e-3 降到 **7.5e-4**（已接近 dense 的 6.4e-4），极端 token 频率降一个数量级，而 **rollout 阶段的额外开销不到 3%**——这是"架构层修法"性价比很高的直接证据。

> 面试一句话总结：**MoE 路由是离散的 top-k 选择，两引擎 logits 微差会造成专家集合翻转、输出概率结构性跳变，IS/TIS 在 loss 层修不动；实测约 10% 的 router、94% 的 token 存在路由分歧，且同框架两次前向 KL 也有 8.4e-4（说明是路由本身不可复现）；解法是 Router Replay——把参考 mask I_ref 注入训练 forward，但 softmax 仍用训练侧 logits 以便梯度回传；R3（参考路由来自推理引擎）把 KL 从 1.5e-3 压到 7.5e-4，开销 <3%。**

---

## 7. Retokenization Drift 与 token 保真

### 1. 现有问题

Agent 框架（以及大多数 RL 的 rollout 采集）走的是 **OpenAI 兼容 API**（`/v1/chat/completions`），因为需要 chat template、role、tool calling、structured output 这些高层能力。但这类 API **历史上只返回字符串**。于是 RL 训练面临一个隐蔽的坑：

**推理时 detokenize → 训练时 retokenize，两套 token ID 可能不同，尽管字符串一模一样。**

### 2. 方法论

vLLM 官方博客《No More Retokenization Drift》（2025-10-22）给出了三条原因：

1. **非唯一分词**：一个词在生成时可能被切成两个 token（如 `H` + `AVING`），训练侧重新分词却得到另一种切分（如 `HAV` + `ING`）。**文本看着一样，ID 不同**，于是 learner 在"错的序列"上优化。
2. **特殊 token 处理**：chat template 注入的 special token 在 detokenize/retokenize 往返中可能被吞掉或改写。
3. **空白/换行的规范化差异**：两侧对空格、换行、Unicode 规范化的处理不完全一致。

**症状**：学习曲线不稳定，以及"你以为在优化的数据"和"模型实际采样的数据"之间存在难以调试的偏差。博客给出的对照实验很直观：**同设置下"存文本再重新分词"（红/蓝线）与"直接用推理引擎的 token"（黄线）曲线不一致**。

**解法**：让推理接口**直接返回 token ids**。vLLM 的 OpenAI 兼容端点现在支持 `"return_token_ids": true`，调用 `/v1/chat/completions` 或 `/v1/completions` 会在常规文本之外返回 `prompt_token_ids` 与 `token_ids`。

```json
// 请求
{"model": "...", "messages": [...], "return_token_ids": true}
// 响应中额外带回
{"choices": [{"prompt_token_ids": [...], "token_ids": [...]}]}
```

**与 Agent Lightning 的关系**：博客明确指出这个特性"与 Agent Lightning 完美搭配"——因为 Agent Lightning 的路线是**把每次模型调用视为独立的更新样本、不做拼接（without stitching）**，那么只要打开 `return_token_ids` 把 ID 记下来即可，天然规避 drift。

> 这个"不拼接"的路线与 Polar 的"前缀合并（prefix merging）"路线是**两种对立的设计选择**，第 8 点会展开对比——它是「轨迹拼接」专题的核心张力。

### 3. 具体数值样例

设模型生成文本 `"HAVING"`：

| 路径 | token 序列 | 长度 |
|---|---|---|
| 推理引擎实际采样 | `["H", "AV", "ING"]` → ID `[39, 1523, 4521]` | 3 |
| 训练侧重新分词 | `["HAV", "ING"]` → ID `[8812, 4521]` | 2 |

训练侧算的是 $\log\pi_\theta(\text{"HAV"}|\cdot)+\log\pi_\theta(\text{"ING}|\cdot)$，而行为策略实际采的是三步。两者**不是同一个事件**：前者在 token 空间里对应一条**从未被采样过的路径**，概率可以差几个数量级。

更糟的是，如果第一段就分叉，**后续所有 token 的 log-prob 全部错位**——一条 2000 token 的轨迹，从第 k 个 token 开始分叉，后面 $2000-k$ 个 token 的 advantage 与 log-prob 全部对不上。**这是"看不见的 on-policy 破坏"**：IS 权重看起来还很接近 1（因为字符串层面确实一样），但梯度已经在错的序列上算了。

> 面试一句话总结：**Retokenization drift 指"推理 detokenize、训练 retokenize"导致 token ID 切分不同（如 HAVING → H+AV+ING vs HAV+ING），字符串一样但序列不同，梯度在从未采样过的路径上计算；解法是让推理接口直接返回 token ids（vLLM `return_token_ids: true`），Agent Lightning 的"不拼接、每次调用独立成样本"路线天然规避它。**

---

## 8. 诊断指标体系

### 1. 现有问题

训推不一致的麻烦在于"没有报错"。训练不会崩在某一行的异常上，而是**指标慢慢漂**：clip fraction 上升、entropy 掉、grad_norm 抖、reward 卡住。所以需要一组**能提前报警的指标**，而不是等 loss 爆炸才回头查。

### 2. 方法论

verl 把这些指标统一挂在 `rollout_corr/` 前缀下（`rollout_corr.md`「Metrics」节），分四类：

**① 策略间差距（诊断 mismatch 大小）**

| 指标 | 公式 | 含义 |
|---|---|---|
| `kl` | $\overline{\log\pi_{\text{rollout}}-\log\pi_{\text{training}}}$ | KL(π_rollout‖π_training)；**可为负** |
| `k3_kl` | $\overline{e^{\log r}-1-\log r}$ | K3 散度，**恒 ≥ 0**，比直接 KL 稳 |
| `chi2_token` | $\overline{\text{ratio}^2}-1$ | token 级 χ²，IS 权重二阶矩 |
| `chi2_seq` | $\overline{(\prod_t\text{ratio}_t)^2}-1$ | 序列级 χ²，比 token 级更敏感 |
| `log_ppl_abs_diff` | $\overline{\lvert\log\text{ppl}_{\text{rollout}}-\log\text{ppl}_{\text{training}}\rvert}$ | 无方向的差距幅度 |
| `ppl_ratio` | $\exp(\overline{\log(\text{ppl}_{\text{train}}/\text{ppl}_{\text{rollout}})})$ | >1 表示训练侧更不自信 |

**② IS 权重分布**

- `rollout_is_mean`：应贴近 **1.0**（偏离说明系统性 shift）；
- `rollout_is_std`：越大越危险（官方阈值 >1.0 告警）；
- `rollout_is_min` / `rollout_is_max`：序列/几何级用**未截断的 log 空间**算真实极值，token 级用安全界后的值；
- `rollout_is_eff_sample_size`：$\text{ESS}=\frac{1}{\overline{w^2}}$，范围 $[0,1]$，**< 0.3 告警**。

**③ 截断/拒绝统计**

- `rollout_is_ratio_fraction_high` / `_low`：超过上/下阈值的比例；
- `rollout_rs_masked_fraction`：被拒绝采样 mask 掉的 token 比例；
- `rollout_rs_seq_masked_fraction`：至少有一个 token 被拒的序列比例。

**④ 健康度判据**（官方给出的经验阈值，可直接抄进面试答案）

```python
# 来自 rollout_corr.md 的 check_rollout_correction_health 示例
mean_weight < 0.5 or mean_weight > 2.0   # → 均值远离 1.0，存在系统性 shift
ess < 0.3                                 # → 有效样本量过低，权重过度集中
std > 1.0                                 # → IS 权重分布过散
abs(kl) > 0.1                             # → off-policy gap 显著
chi2_token > 1.0                          # → 分布偏移严重
```

### 3. 具体数值样例

假设连续 5 个 step 的 `rollout_is_mean` 与 `ess` 如下（数据为构造示例）：

| step | `rollout_is_mean` | `rollout_is_std` | `ess` | `kl` | 判断 |
|---|---|---|---|---|---|
| 1 | 1.002 | 0.08 | 0.94 | 0.004 | 健康 |
| 2 | 1.015 | 0.15 | 0.88 | 0.011 | 健康 |
| 3 | 1.080 | 0.42 | 0.66 | 0.038 | 关注 |
| 4 | 1.310 | 1.15 | 0.29 | 0.085 | **ESS 告警** |
| 5 | 1.870 | 2.40 | 0.14 | 0.160 | **三项同时告警** |

**排障顺序**（官方 Troubleshooting 节）：

1. **均值远离 1** → 先确认 `calculate_log_probs=True` 有没有真的打开、`rollout_log_probs` 有没有正确传下去（**最常见的原因就是配置漏了**）；
2. **权重过散 / ESS 低** → 把聚合层级从 `sequence` 换成 `geometric`（几何级 RS），并收紧阈值；
3. **确认两侧策略不是"差异过大"**（例如 rollout 用了完全没同步的旧权重、或 fp8 与 bf16 差距被放大）。

**推荐上线节奏**（官方 Example Workflow）：**先 metrics-only**（`rollout_is: null, rollout_rs: null`，但照样传 `rollout_log_probs`）跑 1~2 个 epoch 摸清 gap 量级，**再开 RS，最后开完整 IS**。

> 面试一句话总结：**训推不一致必须先能"看见"——用 rollout_corr/ 下的 KL/K3/χ² 看 gap 大小、用 is_mean/is_std/ESS 看权重健康度、用 rs_masked_fraction 看拒绝比例；健康判据是 mean∈[0.5,2]、ESS>0.3、std<1、|KL|<0.1；上线节奏是 metrics-only → RS → 完整 IS。**

---

## 9. 架构串讲：从 rollout 到 loss 的完整修正链

把前面 8 个技术点串起来，一次完整的修正链是：

```
① 采样（推理引擎）
   vLLM/SGLang 生成 token
   ├─ 打开 return_token_ids           → 解决 Retokenization Drift（第 7 点）
   ├─ 返回 rollout_log_probs          → IS/RS 的输入（必需）
   └─ [MoE] 导出 routed_experts       → Router Replay 的 I_ref（第 6 点）
                    │
                    ▼
② 数据通路
   token_ids / log_probs / routed_experts 随 batch 流向训练 worker
   （DP / mini-batch / micro-batch 切分；变长样本用 nested 组织）
                    │
                    ▼
③ 训练侧对齐
   ├─ Decoupled 模式：算 π_old 的 log_prob（多一次前向）
   └─ Bypass 模式：  令 π_old = π_rollout，跳过该前向（第 3 点）
   └─ [MoE] forward 时注入 I_ref 复放路由（第 6 点）
                    │
                    ▼
④ 计算 log_ratio = log π_train − log π_rollout
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
⑤ IS 权重（连续）            ⑥ RS 掩码（离散）
   clamp(±20) → ×mask        k1/k2/k3 → seq_sum/mean/max
   → TIS 截断 / IcePop 置零   → modified_response_mask
   → detach → 可选 batch norm      （第 5 点）
   （第 4 点）
        └───────────┬────────────┘
                    ▼
⑦ loss：w_t · min(r_t A_t, clip(r_t)A_t)  （w_t 修 Drift 1、r_t 修 Drift 2）
                    │
                    ▼
⑧ 监控：rollout_corr/{kl, k3_kl, chi2_*, is_mean, is_std, ess, rs_masked_fraction}
   → 触发告警则回到 ①③ 调精度/阈值（第 8 点）
```

### 各家方案的定位差异（面试对比用）

| 方案 | 主战场 | 做法 | 与上面的关系 |
|---|---|---|---|
| **verl** | loss 层修正最完整 | Rollout Correction 框架：IS（token/seq）+ RS（k1/k2/k3 × sum/mean/max）+ Decoupled/Bypass 模式 + 全套诊断指标 | 第 3/4/5/8 点的实现主体 |
| **ROLL（阿里）** | **架构层** | Router Replay R3：SGLang 记录 `routed_experts` → Megatron 复放 | 第 6 点的实现主体；与 verl 的 loss 层修正**互补**（ROLL 文档原话："让 IS 修正、TIS 等 loss 层补偿手段难以单独解决问题"） |
| **vLLM / Agent Lightning** | **数据层** | `return_token_ids` 消灭 retokenization drift；AGL 路线是"每次调用独立成样本、不拼接" | 第 7 点 |
| **AReaL** | 异步场景下的组合 | `use_decoupled_loss: true` + `recompute_logprobs: true`（官方注明：开 decoupled loss 时必须 recompute logprobs）+ `max_head_offpolicyness` 控制陈旧度 | 把第 3 点搬进异步流水 |
| **Polar（NVIDIA）** | agent harness 的 token 保真 | 网关代理捕获 token-level 数据；**assistant token 从推理响应拷贝、interstitial token 用 canonical prompt 分词、loss mask 只标 behavior-policy token** | 与第 7 点同源，但作用在任意 harness 上 |

### 一句话总纲

> **训推不一致 = 采样分布 ≠ 训练分布。** 修它有三层，缺一不可：**数据层**保证 token 保真（否则 IS 权重看着正常、梯度已经错了）；**架构层**固定 MoE 路由（否则离散跳变在 loss 层修不动）；**loss 层**用 Decoupled PPO + IS/RS 把残余的连续偏差修掉，并用 `rollout_corr/` 指标持续监控。

---

## 10. 深水区：九个反直觉的实证结论

这一节全部来自经过核实的论文/代码实测，每一条都**与"常识预期"相反**——面试里讲这些最能体现深度。

### 1. 根因可能就在 BF16 本身：换 FP16 就能消除

常识预期是"mismatch 来自 kernel 实现差异，所以要用更一致的 kernel"。但 Sea AI Lab + NUS 的工作《Defeating the Training-Inference Mismatch via FP16》（arXiv **2510.26788**）给出了一个更根本的结论：

- **BF16 = 8 位指数 + 7 位尾数；FP16 = 5 位指数 + 10 位尾数。** BF16 动态范围大（不易 overflow/underflow），但**精度只有 FP16 的约 1/8**——rounding error 更大，因此同一份权重在两次前向里被放大的差异也更大。
- 论文结论：**"its root cause lies in the floating point precision itself"，直接改回 FP16 就能有效消除 mismatch**（只需几行代码），且跨算法（GRPO/GSPO/TIS/MIS/PG）、跨模型（R1D/Qwen/OctoThinker）、跨形态（LoRA/14B dense/MoE）、**跨框架（verl 与 Oat）**一致有效。

**为什么这条重要**：它说明"训推不一致"不完全是系统工程的锅，**数值格式的选择本身是一阶因素**。

### 2. IS 修不了"部署 gap"（这是 IS 类方法的原理性局限）

同一个工作的式 (4) 指出：**IS 类修正只让 $\theta$ 对"训练引擎的 $\pi$"最优**：

$$
\arg\max_\theta \mathbb{E}_{y\sim\mu}[R] \neq \arg\max_\theta \mathbb{E}_{y\sim\pi}[R]
$$

**即：修正过后，模型仍然是在"训练引擎的分布"下最优的，部署时（用推理引擎）的性能损失修不掉。** 只有**消除 mismatch 本身**（FP16 / batch-invariant / 统一模型定义）才能解决。**这是"补丁派"与"消除派"最本质的分歧点。**

### 3. fp32 lm_head 是"必要但不充分"

直觉："logprob 精度问题，那把 lm_head 换成 fp32 不就行了？"（vLLM 确实有这个开关：`--hf-overrides '{"head_dtype":"float32"}'`，docstring 明写 "required for RL training-inference consistency"；SGLang 对应 `--enable-fp32-lm-head`。）

**但 rl-collapse 的消融实验表明：把 vLLM lm_head 改成 fp32 之后，训练仍然崩溃，vllm-kl 依旧急增。** 类似地，"禁用 chunked prefill"、"enforce_eager × free_cache_engine 四种组合"**全部无效**。

**结论**：局部补丁救不了系统性问题——必须组合使用（精度对齐 + 采样语义对齐 + 路由复放 + IS/RS）。

### 4. 减小 top-p 能降 KL，但会让训练**变慢**

直觉："top-p 截断让分布更尖锐，减小 top-p 应该能缓解 mismatch。"实测（rl-collapse §4.2.2，L20 上 on-policy GRPO，top-p ∈ {0.98, 0.99, 0.999}，**不加 IS 修正**）：

- 更小的 top-p **确实减少了 vllm-kl 的尖峰**；
- 但**同时增大了 vLLM 与 FSDP 分布之间的差异**（因为截断更狠），于是梯度**偏差**更大；
- 净效果：**top-p 越小，奖励提升越慢**。

**正确解读**：**减小 top-p 降的是方差，不是偏差**。这一条非常容易被误用，值得在面试里主动点出。

### 5. 硬件是一阶变量（同一份代码换机器就崩）

rl-collapse 记录：**同一份代码，vllm-kl 的量级是 H20 < L20 < A100**：

| 硬件 | vllm-kl 量级 | 可用性 |
|---|---|---|
| H20 | $5\times10^{-4}\sim10^{-3}$ | 可训 |
| L20 | $\sim10^{-3}\sim10^{-2}$ | 边缘 |
| **A100** | $10^{-2}\sim1$ | **训练不可行** |

文中还有一个很有说服力的实验：**在 L20 上崩溃的实验，把第 200 步的 checkpoint 换到 H20 继续训练就正常了**。

**工程含义**："我们复现不了"有可能不是代码问题，而是**硬件差异**——这解释了为什么"确定性对齐"这类方案在不同集群上收益差别巨大。

### 6. 一个真实的 attention bug：FA2 的 cascade attention 精度崩塌

rl-collapse 定位到：在 A100（及 L20）上特定 batch/seq 长度组合会触发 FlashAttention-2 kernel 的 `split_kv` 路径，**该路径错误转置了 LSE（log-sum-exp）布局**，导致 cascade attention **完全精度崩溃**。

**修法**：设 `disable_cascade_attn=True` → **A100 上 vllm-kl 从 $5\times10^{-2}\sim10^{-1}$ 降到 $\sim10^{-3}$**（两个数量级）。

**这类"kernel 级 bug"是最难查的一类 mismatch**——它不是精度问题、不是算法问题，而是**某个 kernel 在特定 shape 下算错了**。

### 7. 确定性推理的代价：平均慢 34%，逐配置 24~55%

"那我把推理做成确定性的不就行了？"——SGLang/TML 的实测给出代价（LMSYS blog）：

| 配置 | 确定性模式相对开销 |
|---|---|
| FlashInfer | +42.6% / +46.0% / +24.4% |
| FA3 | +27.2% / +30.2% / +35.7% |
| Triton | +55.1% / +44.6% / +44.8% |
| **平均** | **+34.35%**（TML 原始实现 +61.5%） |

CUDA graph 能拿回 2.79~2.93×，但仍不足以抵消。**这解释了为什么"确定性对齐"至今不是默认选项**——它是"用吞吐换正确性"的交易，只在 mismatch 已经导致崩溃时才划算。

### 8. sequence 级的"长度陷阱"：几何均值不只是个技巧

标准序列级 IS 有系统性长度偏差：设平均 per-token ratio 为 $1.1$，则

| 序列长度 $T$ | 序列级权重 $\rho=\prod_t \rho_t$ | 后果 |
|---|---|---|
| 10 | $1.1^{10}\approx2.6$ | 保留 |
| 100 | $1.1^{100}\approx13780$ | **必被拒绝** |

**后果是 Context Collapse**：模型发现"长答案一定被拒"，于是**只学短答案、拒绝长 CoT**。这正是 Agentic RL 里最致命的一类退化。

**Geo-RS（几何均值）修法**：$\rho_{\text{geo}}=\rho^{1/T}$——把 extensive 量变成 intensive 量。上表两行都变成 $1.1$，于是可以用**极窄的阈值** `"0.999_1.001"`（对应平均 per-token log 偏差 ≈0.1%）同时约束两者。**这不是"调参技巧"，而是修正了估计量的量纲。**

### 9. K3 散度在期望上就等于 reverse KL

一个很漂亮的推导（verl 数学文档 §3.3.5）：由 importance sampling 的基本性质

$$
\mathbb{E}_{\pi_{\text{rollout}}}[\rho]=1,\qquad
\mathbb{E}_{\pi_{\text{rollout}}}[\log\rho]=-\mathrm{KL}(\pi_{\text{rollout}}\|\pi_{\text{old}})
$$

（其中 $\rho=\pi_{\text{old}}/\pi_{\text{rollout}}$）。因此

$$
\mathbb{E}[k_3]=\mathbb{E}[\rho-1-\log\rho]=\underbrace{1-1}_{=0}+\mathrm{KL}(\pi_{\text{rollout}}\|\pi_{\text{old}})
$$

**$k_3$ 的期望恰好是 reverse KL**，且每个样本上恒非负。这解释了为什么 K3 比"直接算 $-\log\rho$"更稳（后者可以是负数、小样本下方差大、符号易被数值噪声翻转）。

**面试里写下这两行推导是很强的加分项**——它把"为什么 verl 提供 k1/k2/k3 三个估计量"从"经验选择"变成了"有数学依据"。

### 附：token 级 TIS vs sequence 级 MIS 的实验判决

这是该领域目前**最活跃的争论**，两派都有强论据：

| 派别 | 主张 | 论据 |
|---|---|---|
| Feng Yao 等（TIS 派） | **token 级 TIS 足够** | 简单有效，反例是"两个 system patch 都无效" |
| rl-collapse / Trust Region Masking 派 | **必须 sequence 级** | token 级 IS 只校正 action 分布、**没校正 state occupancy** $d_\mu\neq d_\pi$，且有 $O(T^2\Delta_{\max})$ 的偏差 |

**实验判决**（rl-collapse §4.2.1，L20 + Qwen3-14B-Base + TIR）：

- token 级 TIS 与 sequence 级 TIS **初期都能阻止梯度爆炸，但 token 级 TIS 后来仍然崩溃**；
- 在**更简单的 reasoning RL** 上 token 级 TIS 能防崩，但**测试性能没有提升、后期还下降**；
- **MIS > TIS**（峰值训练奖励与测试准确率都更高）；
- **token 级 MIS 也崩，只有 sequence 级 MIS 稳定**。

**结论（面试可以直接用）**：**越靠 sequence 级越稳，代价是丢样本；token 级便宜但在长序列上不够。** 还有一个补充证据来自 VeXact 论文：**KL 类指标不足以预警早期退化**——真正的早期信号是 advantage 加权贡献 $C(r)=-(r-1)A_t$ 的**符号不对称偏斜**。

### 再附：mismatch 会自己加速（复利效应）

Ring-1T（arXiv **2510.18855**）的 Theorem 1（Compounding Probability Discrepancy）给出

$$
\delta_{t+1}\ \ge\ \left(1+\frac{\eta\mu}{2}\right)\delta_t
$$

即**训推 KL 沿训练步几何增长**（$\mu$ 是步长、$\eta$ 是学习率相关量）。**这从理论上解释了为什么 mismatch 是"先慢后崩"而不是线性恶化**——它是正反馈：偏差让梯度更偏 → 参数走得更远 → 偏差更大。

同一个工作的 **IcePop** 方案也值得记：**双面校准 + 区间外置零**（"Discard All Noisy Gradient Updates"），即

$$
M(k)=\begin{cases}k & k\in[\alpha,\beta]\\ 0 & \text{otherwise}\end{cases},\qquad \alpha=0.5,\ \beta=5.0\ (\text{verl 默认})
$$

**IcePop 与 TIS 的本质差别**：TIS 把超界权重 `clamp` 到边界（**保留样本、降权**），IcePop **直接置 0**（**丢弃样本**）。verl 里有专门的单测验证"IcePop 置零"与"先按掩码过滤再加权 PPO"**严格等价**。

---

## 附：高频追问速答

**Q1：PPO 本身错了吗？**
没有。PPO 数学上是对的；错的是"把 π_old 当作行为策略"这个假设——在 LLM RL 里 π_rollout（vLLM）与 π_old（FSDP/Megatron 重算）不是一个分布。verl 文档原话："This is not PPO's fault."

**Q2：Token 级和序列级 IS 怎么选？**
Token 级低方差有偏、序列级无偏高方差。**默认从 token 级或几何级起手**；只有当确认数据干净、mismatch 中等时才上序列级（阈值也要放宽到 2~10）。

**Q3：TIS 和 IcePop 的区别？**
TIS 把超上界的权重**截断到边界**（保留样本、降权）；IcePop 给 `"lower_upper"` 区间，**区间外置 0**（丢弃）。IcePop 只改 IS 系数、不改 `response_mask`。

**Q4：IS 权重为什么要 detach？**
因为 IS 改的是**测度**（把 $\mathbb{E}_\mu$ 换成 $\mathbb{E}_{\pi}$），不是目标函数。让它带梯度就等于改了要优化的目标。verl 代码里有显式 `rollout_is_weights.detach()` 并注明引用数学文档 §3.2.2。

**Q5：为什么序列级 IS 权重会爆？**
它是逐 token 比值的**乘积**；即使单 token 只偏 0.05，$T=2000$ 时 $\prod r=e^{100}$。这就是"token 级温和、序列级致命"。

**Q6：MoE 为什么必须架构层修？**
因为 top-k 是**离散选择**，两侧 logits 微差会导致专家集合翻转，输出概率是结构性跳变而非小扰动——IS 权重会重尾，截断/拒绝只能砍掉这些样本（丢数据），不如从源头固定路由。

**Q7：Router Replay 会不会让梯度传不过去？**
不会。复放只替换 **top-k mask**，softmax 仍作用在**训练侧 logits** 上（$g_i\propto I_{\text{ref},i}\cdot e^{s_{\text{train},i}}$），所以 router 权重仍可训练。

**Q8：怎么判断当前训练有 mismatch？**
看三个数：`rollout_corr/rollout_is_mean`（应贴近 1）、`rollout_corr/rollout_is_eff_sample_size`（>0.3）、`rollout_corr/kl` 或 `k3_kl`（|KL|<0.1）。三项同时越界就是有问题。

**Q9：`recompute_logprobs` 是什么？**
AReaL 的配置项：训练时用训练引擎**重算** log-prob，而不是复用推理引擎返回的。开 `use_decoupled_loss` 时**必须**为 true，因为 decoupled loss 需要 $\pi_{\text{old}}$ 和 $\pi_\theta$ 都在训练引擎的数值域里。

**Q10：为什么 Agent Lightning 说"不用拼接"？**
因为它的路线是把**每次模型调用作为独立样本**（配合 `return_token_ids`），从设计上就绕开了"多轮拼接 + loss mask"的复杂度；代价是丢失跨轮次的连贯 credit assignment。这与 Polar 的 prefix merging 是两种取舍。
