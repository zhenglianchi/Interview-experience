# SmolVLA-Verl：分离式 VLA 在线强化学习（FlowGRPO）完全指南

> 对应简历项目（二）"分离式 VLA 在线强化学习"（2026.04-2026.05，RL Infra）。一句话知识框架：
> **本文回答"怎么把一个没有概率密度的流匹配 VLA（SmolVLA 0.45B）改造成可训练策略，并用分离式训推架构在 LIBERO 上跑通在线 RL"**——先讲 ODE→SDE 改造（算法前提），再讲 FlowGRPO 损失（算法本体），最后讲本地采集 ↔ 云端 serve ↔ 云端训练的分离式架构（工程落地）与 15 轮实验结论。
>
> 与本目录其他文档的关系：分离式思想的"母版"见 `Uniagent-Lighting.md`（本地 agent ↔ 云端 Gateway/TQ）；本文是它在**连续动作、多模态、机器人仿真**域的投影——reward 从"代码通过率"变成"LIBERO 二进制成功判定"，轨迹从"token 序列"变成"SDE 去噪状态序列（states + log-prob）"，数据面从 TransferQueue 变成 HTTP 端点 + 服务器内存 session。GRPO 基础见 `RL.md`；prefix KV cache 的 vLLM 侧对照见 `VLLM.md`。

---

# 一、核心技术点：从"流匹配 VLA 没有策略梯度"到"能训的 FlowGRPO"

## 1. 流匹配 VLA（SmolVLA）与"为什么 ODE 采样不能直接做 RL"

### 1. 现有问题

- **VLA（Vision-Language-Action）是什么**：输入机器人观测（摄像头图像 + 关节状态）+ 语言指令，输出**一段动作序列**。SmolVLA 0.45B 的构成是**冻结的 SmolVLM-2 视觉语言模型（VLM）+ 可训练的 flow-matching action expert**：VLM 负责把图像/语言/状态编码成条件，action expert 负责生成动作。
- **SmolVLA 的动作生成方式与 LLM 完全不同**：LLM 是自回归（逐 token 输出，天然有逐 token 概率）；SmolVLA 是 **flow matching（流匹配）**——从高斯噪声出发，**确定性 ODE 去噪 10 步**，一次性解出**整个动作 chunk**（例如 $10 \times 7 = 70$ 维动作矩阵）。它是"并行生成"而不是"逐位生成"。
- **致命问题：确定性 ODE 没有概率密度**。给定同一个噪声状态 $x_t$，ODE 的下一步 $x_{t-h}$ 是**唯一确定**的（退化分布 / delta 分布），`log p(x_{t-h} | x_t)` 无定义。而 PPO/GRPO 的策略梯度**必须**用新旧策略的 log-prob 之比（importance ratio）：
  $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} = \exp\left(\log p_\theta - \log p_{old}\right)$$
  没有 log-prob → 算不出 ratio → 梯度为零 / 无定义 → **flow-matching VLA 无法直接进 RL 训练管道**。这就是"流匹配模型明明生成效果很好，却一直用监督流匹配损失（SFT）训练、鲜有人做在线 RL"的根本原因。

### 2. 方法论

把问题拆成三步看，就能定位"缺什么"：

1. **ODE 流匹配采样路径**：SmolVLA 的时间约定是 $t=1$ 为高斯噪声、$t=0$ 为动作，去噪步从 $t$ 走到 $t-h$。线性流匹配定义噪声与动作之间的插值：
   $$x_t = t \cdot \text{noise} + (1-t) \cdot \text{action}, \qquad v = \text{noise} - \text{action}$$
   模型学的是**速度场** $v(x_t, t)$（一个去噪网络）。Euler 更新为 $x_{t-h} = x_t - h \cdot v$。因为给定 $x_t$ 时 $v$ 是网络输出的确定值，$x_{t-h}$ 也就是确定值——**这一步没有随机性，也就没有密度**。
2. **RL 需要的到底是什么**：策略梯度只需要"在当前策略下重估这条轨迹的对数似然"（重打分，rescoring）。所以只要能给每一步转移定义一个**有闭式解的条件高斯分布**，就能算 log-prob → ratio → 策略梯度。这正是第 2 点"边缘保持 SDE"做的事。
3. **为什么不能随便加噪声**：如果直接把 Euler 步替换成 $x_{t-h} = x_t - h v + \mathcal{N}(0, \sigma^2)$，虽然有了密度，但**每个噪声层的边缘分布 $p(x_t)$ 被改变**，生成的动作质量会退化。必须构造一种"**边缘分布保持**"的随机转移（第 2 点），才能既拿到密度又不损失生成质量。

> 一句话：**流匹配 VLA 生成的是连续动作的确定性去噪轨迹，没有逐转移的概率密度，因此无法计算 PPO/GRPO 的 importance ratio；出路是把确定性 ODE 采样改造成"边缘保持"的随机 SDE 采样，给每个去噪转移一个闭式高斯 log-density。**

### 3. 具体数值样例

设正式配置 `chunk_size=10`、`max_action_dim=7`（LIBERO 7 自由度机械臂）、`num_steps=10`：

- 一次 `sample_sde_chunk` 输出一个动作 chunk，形状 $(10, 7)$，即 $10 \times 7 = 70$ 个动作标量；
- 若要算策略梯度，需要**逐去噪步、逐动作维**的 log-prob：每 chunk $10 \text{ 步} \times 70 \text{ 维} = 700$ 个密度标量；
- 一条 LIBERO episode 平均约 $25$ 个 chunk（真实日志：第 1 轮 48 集共 1222 个 chunk，$1222/48 \approx 25.5$），即一条轨迹需要约 $25 \times 700 = 17500$ 个 log-prob 标量；
- ODE 采样下这 17500 个标量**一个都给不出来**（转移是 delta 分布）→ RL 训练无法启动。

这就是"先有密度、后有梯度"的量化含义：改造的目标就是让这 17500 个标量变得**可计算、可复现**。

---

## 2. 边缘保持 SDE：给确定性去噪装上闭式概率密度

### 1. 现有问题

- ODE 的 Euler 步 $x_{t-h} = x_t - h v$ 是确定性的 → 没有密度（第 1 点已述）；
- 朴素加噪会破坏边缘分布 → 生成质量退化；
- 还有一个工程上的坑：**所有概率计算必须用 float32**，而速度网络跑在 bf16 下（详见第 4 点数值一致性）。

### 2. 方法论

核心思路来自 FlowVLA-RL / Flow-GRPO（NeurIPS 2025，arXiv:2505.05470），本项目的实现完整落在 `verl-vla/src/verl_vla/models/smolvla/sde.py`。四个关键函数：

**① 从速度恢复 score**（线性流匹配的 score 有闭式解）：

```python
# verl-vla/src/verl_vla/models/smolvla/sde.py
def score_from_velocity(x_t: Tensor, velocity: Tensor, time: Tensor | float) -> Tensor:
    """For x_t = t * noise + (1-t) * action and v = noise - action, the
    score is -(x_t + (1-t) * v) / t."""
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have identical shapes")
    time_b = _broadcast_time(time, x_t)
    x_f = x_t.float()
    velocity_f = velocity.float()
    return -(x_f + (1.0 - time_b) * velocity_f) / time_b
```

**② 噪声调度**（有界调度 $g(t) = \eta\sqrt{t}$，$\eta$ 是用户给的 SDE 噪声强度）：

```python
def diffusion_scale(time: Tensor | float, eta: float, reference: Tensor) -> Tensor:
    """Return the bounded schedule g(t)=eta*sqrt(t) in float32."""
    if eta < 0:
        raise ValueError("eta must be non-negative")
    time_b = _broadcast_time(time, reference)
    return float(eta) * torch.sqrt(time_b)
```

**③ 边缘保持转移**（一次反向高斯转移的参数）：

```python
def marginal_preserving_transition(
    x_t: Tensor, velocity: Tensor, time: Tensor | float,
    step_size: float, eta: float,
) -> Transition:
    """Build p(x_{t-h}|x_t) for the marginal-preserving reverse SDE.
    mean = x_t - h * (v - 0.5*g(t)^2*score)  and  std = g(t)*sqrt(h)."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    score = score_from_velocity(x_t, velocity, time)
    g_t = diffusion_scale(time, eta, x_t)
    reverse_drift = velocity.float() - 0.5 * g_t.square() * score
    mean = x_t.float() - float(step_size) * reverse_drift
    std = g_t * math.sqrt(step_size)
    return Transition(mean=mean, std=std, score=score, reverse_drift=reverse_drift)
```

**④ 闭式高斯对数密度**（带 mask，padded 动作维/未执行位置不进入 ratio）：

```python
def gaussian_log_prob(sample, mean, std, *, mask=None, reduction="sum") -> Tensor:
    """Evaluate a diagonal Gaussian; mask is essential for SmolVLA: padded
    action dimensions and padded action positions must not enter the policy ratio."""
    sample_f, mean_f, std_f = sample.float(), mean.float(), std.float()
    elementwise = -0.5 * (
        ((sample_f.detach() - mean_f) / std_f).square()
        + 2.0 * torch.log(std_f)
        + math.log(2.0 * math.pi)
    )
    if mask is None:
        valid = torch.ones_like(elementwise, dtype=torch.bool)
    else:
        valid = torch.broadcast_to(mask.to(device=sample.device, dtype=torch.bool), sample.shape)
    elementwise = torch.where(valid, elementwise, torch.zeros_like(elementwise))
    if reduction == "none":
        return elementwise
    reduce_dims = tuple(range(1, elementwise.ndim))
    total = elementwise.sum(dim=reduce_dims)
    ...
    return total
```

**这段代码关键在哪**：① + ③ 合起来把"确定性 Euler 步"换成了"**均值沿 ODE 轨迹、方差由 $g(t)$ 决定**的高斯转移"——注意 `reverse_drift` 里 `- 0.5 g(t)^2 score` 正是 SDE 反向漂移修正项，它保证**各噪声层的边缘分布 $p(x_t)$ 与 ODE 版本完全相同**（"边缘保持"），所以动作质量与确定性采样等价；④ 给出逐元素闭式 log-density，并且**必须带 mask**——SmolVLA 的 padded 动作维和未执行动作位不能污染 ratio。全部计算强制 `.float()` 是因为 bf16 下高斯似然会溢出/漂移（第 4 点展开）。

采样一步的实现（`sample_transition`）还刻意 `.detach()`：**采样时保留图没有意义**，密度在 `gaussian_log_prob` 里单独评估，重打分阶段（训练）才会让梯度穿过网络。

### 3. 具体数值样例

取正式配置 `eta=0.05`、`num_steps=10`（步长 $h = 1/10 = 0.1$），手算**第一步反向转移**（$t=1.0$，单动作维）：

- 输入：$x_t = 0.3$，网络预测速度 $v = 0.7$；
- ① score：$\text{score} = -(x_t + (1-t)v)/t = -(0.3 + 0 \cdot 0.7)/1 = -0.3$；
- ② 噪声尺度：$g(1) = \eta\sqrt{1} = 0.05$，$g^2 = 0.0025$；
- ③ 反向漂移：$v - 0.5 g^2 \cdot \text{score} = 0.7 - 0.5 \times 0.0025 \times (-0.3) = 0.700375$；
- 均值：$\mu = x_t - h \cdot \text{drift} = 0.3 - 0.1 \times 0.700375 = 0.22996$；
- 标准差：$\sigma = g\sqrt{h} = 0.05 \times \sqrt{0.1} = 0.01581$；
- 于是 $p(x_{0.9} | x_{1.0}) = \mathcal{N}(0.22996,\, 0.01581^2)$。若采样得到 $x_{0.9} = 0.23$，其 log-density：
  $$-\tfrac12\left(\tfrac{0.23-0.22996}{0.01581}\right)^2 - \log 0.01581 - \tfrac12\log 2\pi \approx -3\times10^{-6} + 4.147 - 0.919 \approx 3.23 \text{ nats}$$
- 第二步（$t=0.9$）：$g(0.9) = 0.05\sqrt{0.9} = 0.0474$，$\sigma = 0.0474 \times 0.3162 = 0.0150$——噪声随 $t$ 减小而**衰减**，越接近动作（$t\to0$）越确定，保证最终动作质量。

对照：若 $\eta = 0$（纯 ODE），$\sigma = 0$，`gaussian_log_prob` 会因 `std <= 0` 直接报错——**这就是"要密度必须引入噪声"的代码级体现**；$\eta=0.05$ 是"小噪声"折中：边缘分布几乎不变（动作质量不退化），但每条轨迹有了完整密度。

> 面试一句话总结：**边缘保持 SDE 把确定性 ODE 去噪的每一步替换成一个"均值沿原轨迹、方差 $g(t)\sqrt{h}$"的高斯转移，闭式 log-density 让 flow-matching VLA 第一次有了可算的策略梯度，而边缘分布不变保证生成质量与 ODE 等价——用可调噪声强度 $\eta$ 换取"可训练性"。**

### 4. 演进与踩坑

- **算法来源**：本项目不是第一个做 flow-matching RL 的，参考了 FlowVLA-RL（MIT 协议移植）与 Flow-GRPO（NeurIPS 2025）；移植时**没有 patch LeRobot 原生策略**，而是镜像 `VLAFlowMatching.sample_actions` 的调用路径、只替换 Euler 更新为 SDE 转移核（`sde.py`），保证原生策略仍是"唯一事实源"。
- **噪声口径**：训练用 SDE（$\eta=0.05$），**评估用确定性 ODE**（官方协议）——两者口径刻意接近（0.05 很小），否则"训练的策略"与"评估的采样方式"之间会有分布偏移，这是实验结论里"持平"的一个重要前提（见第 12 点）。

---

## 3. prefix KV cache + chunk 级 SDE 采样：一次观测编码、多步去噪复用

### 1. 现有问题

- 一次 `sample_sde_chunk` 要跑 **1 次观测编码 + 10 次去噪前向**。SmolVLA 里 VLM（冻结）负责把图像 + 语言 + 状态编码成条件；**图像 token 是计算大头**（两张图就有几百个 patch token）。
- 如果每个去噪步都重新跑 VLM 前向，等于把视觉编码重复 10 遍——**10 倍浪费**，而且这 10 遍是**完全相同**的（观测在 chunk 内不变）。
- 采集是 12 实例并行的在线流程，每轮上千次 `/predict`，编码开销直接决定轮次耗时。

### 2. 方法论

实现见 `verl-vla/src/verl_vla/models/smolvla/sde_sampling.py`。核心是把"观测 → 条件"做成**一次性 prefix KV cache**，10 个去噪步全部复用：

```python
# verl-vla/src/verl_vla/models/smolvla/sde_sampling.py
def build_prefix_cache(model, images, image_masks, language_tokens, language_masks, state):
    """Build the frozen observation cache shared by all denoising steps."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state=state
    )
    attention = make_prefix_attention_mask(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=attention,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,          # 关键：一次前向填充 KV cache
    )
    return prefix_pad_masks, cache
```

`prepare_policy_prefix` 只是把"已经过 LeRobot preprocessor 归一化的 batch"（含 `observation.language.tokens` 等）喂给上面这个函数。采样主体 `sample_sde_chunk` 则**固定复用 prefix cache**、只更新去噪状态：

```python
def sample_sde_chunk(model, prefix_pad_masks, prefix_cache, *, action_dim, eta,
                     noise=None, generator=None) -> SmolVLATrajectory:
    ...
    step_size = 1.0 / model.config.num_steps
    state = noise.detach().float()
    states = [state]
    element_log_probs = []
    with torch.no_grad():
        for step in range(model.config.num_steps):
            time = 1.0 - step * step_size
            velocity = model.denoise_step(
                x_t=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=prefix_cache,     # ← 10 步复用同一个 cache
                timestep=time_tensor,
            )
            transition = marginal_preserving_transition(state, velocity, time, step_size, eta)
            next_state = sample_transition(transition, generator=generator)
            element_log_probs.append(gaussian_log_prob(
                next_state, transition.mean, transition.std, mask=mask, reduction="none"))
            states.append(next_state)
            state = next_state
    return SmolVLATrajectory(
        states=torch.stack(states, dim=1).detach(),
        element_log_probs=torch.stack(element_log_probs, dim=1).detach(),
        valid_action_mask=mask.detach(),
        eta=float(eta),
    )
```

配套的数据结构 `SmolVLATrajectory`（一条轨迹的全部"密度承载状态"，采集端记录、训练端重打分都靠它）：

```python
@dataclass(frozen=True)
class SmolVLATrajectory:
    states: Tensor              # (num_steps+1, chunk_size, max_action_dim)：噪声 + 10 步去噪状态
    element_log_probs: Tensor   # (num_steps, chunk_size, max_action_dim)：逐元素 log-prob
    valid_action_mask: Tensor   # padded 动作维掩码
    eta: float                  # 采集时的 SDE 噪声（重打分必须用同一个 eta）

    @property
    def actions(self) -> Tensor:
        return self.states[:, -1]   # t=0 的最终去噪状态即动作
```

**这段代码关键在哪**：`fill_kv_cache=True` 的 prefix 前向是"冻结观测只编码一次"的落点；`denoise_step(..., past_key_values=prefix_cache)` 让 10 个去噪步共享同一份 VLM cache——**编码成本从 $O(10 \times \text{prefix})$ 降到 $O(\text{prefix} + 10 \times \text{expert})$**；`SmolVLATrajectory` 把"采样产物"固定成 `states + element_log_probs + mask + eta` 四件套，采集端与服务端、服务端与训练端之间传输/落盘的就是这个结构（第 8 点）。

### 3. 具体数值样例

假设一次观测编码成 prefix 序列约 413 个 token（**估算**：2 张 $224\times224$ 图、patch 16 → 每张 $14\times14=196$ token；语言指令约 20 token；机器人状态 1 token；$196\times2+20+1=413$）：

- **无 cache**：10 个去噪步每步都重编码 413 token → 每 chunk 约 $10 \times 413 = 4130$ token 的 VLM 前向；
- **有 cache**：1 次 413 token 的 prefix 前向 + 10 次只跑 action expert 的去噪前向 → VLM 编码开销减少约一个数量级；
- 放大到训练规模：正式配置每轮 48 集，真实日志第 1 轮产生 **1222 个 chunk**（≈25 chunk/集）→ 每轮省下约 $1222 \times 9 \times 413 \approx 454$ 万 token 的重复视觉编码；
- 注意：**prefix cache 只服务"观测不变、去噪多步"这一个 chunk 内部**；换一个 chunk（观测变了）就要重新 `prepare_policy_prefix`——所以它是"chunk 内复用、chunk 间重建"。

> 面试一句话总结：**SmolVLA 每生成一个动作 chunk 要 1 次观测编码 + 10 步去噪，prefix KV cache 让冻结 VLM 只编码一次、10 步去噪全部复用，把每 chunk 的视觉编码开销从 10 倍降到 1 倍，是"单卡 4090 能跑起 12 实例在线采集"的前提之一。**

---

## 4. 逐 chunk 重打分与数值一致性（ratio≈1 守卫）

### 1. 现有问题

- 在线 RL 训练时，`old_log_prob` 来自**采集那一刻**的策略权重；`/train` 时要用**最新权重**重估每条轨迹的 log-prob（重打分）才能算 ratio。如果重打分和采集不在**同一条数值路径**上，ratio 会系统性偏离 1——而 PPO 式裁剪假设"新旧策略初始一致"（ratio 初始 = 1），偏离就直接毁训练。
- 两个典型破坏源，本项目都踩过：
  1. **批处理重打分**：把多个 chunk 拼成一个大 batch 一次前向，`valid_positions` / mask 与各 chunk 实际长度**错位** → 重打分 logp 与采集不一致（旧管道 R16 训练 16 轮后官方指标反而从 63% 掉到 55%，这就是头号根因）；
  2. **数值类型不一致**：采集用 bf16 autocast、训练用 fp32（或反过来），图像用半精度存储——同一轨迹 bf16 与 fp32 的 log-prob 会差约 **1.07 nats**，换算 ratio $e^{-1.07} \approx 0.34$，完全落在 clip 窗口 $[0.8, 1.2]$ 之外，**所有 ratio 被硬 clip 到边界，梯度失真且没有任何报错**（"静默退化"最可怕）。

### 2. 方法论

三条铁律，全部落在代码里：

**铁律 1：采集与重打分共用同一个 autocast 上下文**（bf16 on CUDA、fp32 on CPU）：

```python
# verl-vla/src/verl_vla/models/smolvla/sde_sampling.py
ROLLOUT_AUTOCAST_DTYPE = torch.bfloat16

def rollout_autocast(device):
    """Only enabled on CUDA: bf16 autocast semantics differ on CPU."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=ROLLOUT_AUTOCAST_DTYPE,
                          enabled=(dev.type == "cuda"))
```

**铁律 2：重打分必须逐 chunk 进行**（一次一个 chunk，与采集路径完全一致），且 `valid_positions` 参与掩码：

```python
def recompute_log_probs(model, prefix_pad_masks, prefix_cache, trajectory, *,
                        valid_positions=None) -> Tensor:
    """Rescore fixed sampled states under current or reference expert weights."""
    states = trajectory.states.detach()
    mask = trajectory.valid_action_mask
    if valid_positions is not None:
        mask = mask & valid_positions.to(device=states.device, dtype=torch.bool)[..., None]
    step_size = 1.0 / model.config.num_steps
    log_probs = []
    for step in range(model.config.num_steps):
        time = 1.0 - step * step_size
        velocity = model.denoise_step(x_t=states[:, step], prefix_pad_masks=prefix_pad_masks,
                                      past_key_values=prefix_cache, timestep=time_tensor)
        transition = marginal_preserving_transition(states[:, step], velocity, time,
                                                    step_size, trajectory.eta)
        log_probs.append(gaussian_log_prob(states[:, step + 1], transition.mean,
                                           transition.std, mask=mask))
    return torch.stack(log_probs, dim=1)
```

**铁律 3：把"静默退化"变成"显式失败"**——`/train` 第一个 chunk 先做一致性守卫，重打分 logp 与采集存储的 logp 偏差超过 0.05 nats 直接 raise：

```python
# src/smolvla_verl/trainer/grpo_offline.py
# Pass A: verify rescoring reproduces collection log-prob (first chunk)
with torch.no_grad(), sde_sampling.rollout_autocast("cuda"):
    pm, pc = sde_sampling.prepare_policy_prefix(policy, batch)
    old_lp = sde_sampling.recompute_log_probs(policy.model, pm, pc, traj, valid_positions=vpos)
    drift = (old_lp.detach() - mb[0]["stored_old"]).abs().max().item()
    if drift > RATIO_TOLERANCE:     # RATIO_TOLERANCE = 0.05
        raise RuntimeError(
            "rescored old logp drifts ... nats from the collection-time log-prob; "
            "collection and rescoring are on different numeric paths ...")
```

配套：**会话图像必须 float32 存储**（`serve_smolvla.py` 里 `batch_cpu = {k: v.cpu() ...}` 不动 dtype，注释明确写"half-precision storage silently breaks ratio=1"）；`grpo_libero.py` 里同样有**首前向 ratio_mean 守卫**（`drift = abs(ratio_mean - 1.0)`，>0.05 报错）。

### 3. 具体数值样例

- **数值路径不一致的代价**：假设采集（bf16）算得某 chunk 的 old logp $= -20.00$ nats，重打分（fp32）算得 $-21.07$ nats（差 1.07）→ ratio $= e^{-1.07} \approx 0.343$ < clip 下界 0.8 → 被强制 clip 成 0.8；而"正确"的 ratio 应该是 1.0。于是梯度信号整体失真，且 `ratio_mean` 指标显示 0.8 左右——**一眼就能看出漂移，但旧管道没有守卫，没人看**。
- **守卫生效的真实数据**（正式 run `work/logs/grpo_opt.log`）：15 轮训练每轮 `/train` 返回的 `ratio_mean` 全部落在 $[0.99999987,\, 1.00000005]$——逐 chunk 重打分 + 同一 autocast 路径 + float32 图像存储，让 ratio 初始值精确复现为 1.0，证明"采集与重打分是同一数值路径"。
- **修复前后的对照**：修复前旧管道 R16 官方指标 55.0%（< Base 63.0%），修复后 M3 恢复到 63.0%——数值一致性的修复把"白跑一天"变成了"可复现的训练"。

> 面试一句话总结：**flow-matching RL 的 ratio 必须从 1.0 起步，任何"采集与重打分数值路径不一致"（批处理错位、bf16/fp32 混用、半精度存图）都会让 ratio 系统性漂移并静默毁训练；解法是逐 chunk 重打分 + 共用 rollout_autocast + float32 存储，再用"首 chunk 偏差 >0.05 nats 即报错"的守卫把退化显式化——真实日志 ratio_mean 全程在 1.000000 ± 1e-7。**

---

## 5. Critic-free FlowGRPO：reset-matched 组优势 + 二进制奖励 + k3 KL 锚定

### 1. 现有问题

- 连续动作 RL 常见做法是 PPO：需要 **critic（价值网络）** 估计优势。但 VLA 场景 reward 极其稀疏（整条 episode 结束才有"成功/失败"一个数），critic 学不到东西，还多一套要训练的参数和分布（不稳定）。
- 流匹配策略的密度是"逐去噪步"的：一条轨迹有 `(episodes, chunks, denoise_steps)` 三个轴，奖励却只有一个标量——**如何把一个 episode 级信号分配到 700 个密度单元上**？
- 另外，纯二进制奖励 {0,1} 直接当目标，组间方差大、且容易策略漂移（M1 无锚定版本后期采集成功率掉到 45%）。

### 2. 方法论

GRPO（Group Relative Policy Optimization）正好是"critic-free + 组内相对"的设计，实现见 `verl-vla/src/verl_vla/models/smolvla/grpo.py`：

**① 组均值基线（reset-matched）**：同一个初始场景（同 task / init_state / env_seed）采样 $G$ 条 rollout（仅策略噪声 seed 不同），组内成功率的均值做基线。默认**不除以组标准差**（Dr.GRPO 约定，更稳）：

```python
def compute_group_advantages(rewards: Tensor, *, use_std: bool = False, eps: float = 1e-6) -> Tensor:
    """Input shape (number_of_groups, group_size). The default deliberately
    does not divide by standard deviation (the Dr.GRPO convention)."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (groups, group_size>=2)")
    advantages = rewards.float() - rewards.float().mean(dim=1, keepdim=True)
    if use_std:
        advantages = advantages / (rewards.float().std(dim=1, keepdim=True, unbiased=False) + eps)
    return advantages.detach()
```

**② k3 KL 估计（非负，锚定到固定 base 参考策略）**：

```python
def k3_kl_estimate(logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Non-negative k3 estimate of KL(policy || reference).
    With log_ratio = log p_ref - log p_policy, k3 is exp(log_ratio) - log_ratio - 1."""
    log_ratio = (ref_logp.detach().float() - logp.float()).clamp(-10.0, 10.0)
    return torch.exp(log_ratio) - log_ratio - 1.0
```

**③ 逐去噪步裁剪的 GRPO 损失**（`logp` 形状 `(episodes, control_chunks, denoise_steps)`，一个 episode 标量 advantage 广播到两个内轴；`sample_weights` 做 episode 均衡，见第 6 点）：

```python
def grpo_loss(logp, old_logp, ref_logp, advantages, *, clip_epsilon, kl_beta,
              valid_steps=None, sample_weights=None):
    """logp has shape (episodes, control_chunks, denoise_steps). One scalar episode
    advantage is broadcast across both inner axes. There is no transition probability
    between adjacent control chunks."""
    ...
    weighted_valid = valid_f * sample_w
    # Only guard against an exactly-zero denominator (floor of 1.0 would inflate
    # loss by ~1/sum(w) AND corrupt ratio_mean; see docs)
    denominator = weighted_valid.sum().clamp_min(1e-12)

    log_ratio = (logp.float() - old_logp.detach().float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    advantage = advantages.detach().float()[:, None, None]
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    pg_loss = -(torch.minimum(unclipped, clipped) * weighted_valid).sum() / denominator

    kl = k3_kl_estimate(logp, ref_logp)
    kl_loss = (kl * weighted_valid).sum() / denominator
    loss = pg_loss + float(kl_beta) * kl_loss
    ...
    return loss, metrics   # metrics 含 ratio_mean / clip_fraction / advantage_mean / advantage_std
```

**这段代码关键在哪**：`advantages.detach()[:, None, None]` 把 `(episodes,)` 的标量优势广播到 `(episodes, chunks, steps)` 全部单元——回答了"一个信号怎么分到 700 个密度单元"：**每单元共享同一个 episode 优势**（chunk 间没有转移概率，所以不需要跨 chunk 折现，折现由第 6 点的 reward-to-go 在损失外部实现）；`minimum(unclipped, clipped)` 是 PPO 的乐观下界；k3 KL 是"exp − 1 − x"形式的非负估计，比 `0.5·log_ratio²` 对负侧更宽容，且**锚定到固定 base（SmolVLA-LIBERO 预训练权重）而非轮起始权重**，防止策略随轮次累积漂移（M1 无锚定 45% 的教训，见第 11 点）。

组基线成立的前提是 **reset-matched**：`collect_remote.py` 里 `base_env.init_state_id = args.init_state_id` 把同组 4 条 rollout 钉在同一初始场景，只有 `torch.manual_seed(seed + episode_id * 7919)` 的策略噪声不同（详见第 9 点代码）。

### 3. 具体数值样例

- **组优势**：一组 4 条 rollout 成功情况 $[1,0,1,0]$（同场景，仅噪声不同）→ 组均值 $0.5$ → 优势 $[+0.5, -0.5, +0.5, -0.5]$（不除 std）。
- **广播**：取其中一条成功 episode（adv $=+0.5$），它有 3 个 chunk × 10 个去噪步 = 30 个密度单元，**每个单元都用 $+0.5$**。
- **ratio 手算**（某去噪步，70 维求和后的标量 logp）：old $=-2.00$，new $=-1.90$ → $r = e^{0.10} = 1.1052$，`clip_epsilon=0.2` 窗 $[0.8,1.2]$ 内未 clip → pg 项 $= -1.1052 \times 0.5 = -0.5526$（负号来自 $-\min(\cdot)$）。
- **clip 触发**：若 new $=-1.60$ → $r = e^{0.40} = 1.4918 > 1.2$ → clip 成 $1.2$ → pg 项 $= -0.6$（裁剪限制了单步更新幅度，这是 PPO 式稳定性的来源）。
- **KL 惩罚**：某单元 $\log p_{ref} - \log p_{new} = 0.1$ → $k_3 = e^{0.1} - 0.1 - 1 = 0.00517$ nats；`kl_beta=0.01` → 该单元 KL 项贡献 $5.17\times10^{-5}$。
- **全成功/全失败组**：$[1,1,1,1]$ 或 $[0,0,0,0]$ → 组均值等于每个成员，优势全 0 → **无组内信号，整组跳过不更新**（`grpo_libero.py` 里 `if all(s == successes[0] ...): continue`）。真实日志里每轮约 1/3 的组属于这种情况（第 12 点根因之一）。

> 面试一句话总结：**FlowGRPO 是"critic-free"的 GRPO：同初始场景采 G 条 rollout 做组均值基线（不除 std），二进制成功奖励减基线得优势，一个 episode 标量优势广播到全部 (chunk, 去噪步) 密度单元，配合逐步 clipped ratio 与非负 k3 KL 锚定到固定 base——无价值网络、无跨 chunk 折现，结构极简且稳定。**

---

## 6. episode 均衡加权 + reward-to-go：失败长轨迹不再主导梯度

### 1. 现有问题

- **失败轨迹天然更长**：失败的 episode 会一直执行到 `max_steps`（280 步上限），chunk 数多；成功的 episode 提前结束，chunk 数少。如果不加权，**失败的样本在数量上碾压成功样本**，损失被"怎么做都是错"的长失败轨迹主导。
- **credit 分配不合理**：每个 chunk 都拿到同一个 episode 优势，但一条轨迹里**只有最后几步真正决定了成败**（前期的动作只影响"能否走到终局"）——把 credit 均匀撒到所有 chunk，梯度信号被稀释，且容易让策略在无关早期动作上抖动。
- 隐含 bug 隐患：加权后每 chunk 权重之和远小于 1（约 0.01~0.3），若损失分母 `clamp_min(1.0)`，会把 loss/梯度放大 $1/\sum w$ 倍（3~10 倍），还污染 `ratio_mean` 指标——本项目第 4 号 bug 正是这个（第 11 点）。

### 2. 方法论

在 `grpo_offline.py` 里把每 chunk 权重算成**两项相乘**：

```python
# src/smolvla_verl/trainer/grpo_offline.py
# ① episode 均衡：每条 episode 总权重 = 1/E，与 chunk 数无关
episode_weights = [1.0 / (len(episodes) * max(len(ep["chunks"]), 1)) for ep in episodes]
...
# ② reward-to-go 折扣：越靠终局的 chunk 权重越大
"weight": float(chunk_discount) ** (n_chunks - 1 - cpos) / (n_episodes * n_chunks),
```

即第 $e$ 条 episode 的第 $c$ 个 chunk（共 $C_e$ 个，$c$ 从 0 数）权重为：
$$w_{e,c} = \frac{\gamma^{C_e - 1 - c}}{E \cdot C_e}, \qquad \gamma = \text{chunk\_discount} = 0.99$$

权重进 `grpo_loss` 的 `sample_weights`，先广播成 `(episodes,1,1)` 再与有效掩码相乘：

```python
# verl-vla/src/verl_vla/models/smolvla/grpo.py（grpo_loss 内）
if sample_weights is None:
    sample_w = torch.ones_like(valid_f[:, :1, :1])
else:
    sample_w = sample_weights.detach().to(device=logp.device, dtype=torch.float32)[:, None, None]
weighted_valid = valid_f * sample_w
# NOTE: with per-sample (episode-balanced) weights the weighted sum is far below 1
# (e.g. 0.01-0.3 per chunk), so a floor of 1.0 would silently inflate the loss by
# ~1/sum(w) AND corrupt the ratio_mean metric. Only guard against exactly-zero.
denominator = weighted_valid.sum().clamp_min(1e-12)
```

**这段代码关键在哪**：`1/(E·C_e)` 让**每条 episode 无论长短总权重相等**（长失败轨迹不再靠数量取胜）；$\gamma^{C_e-1-c}$ 让**终局 chunk 权重最大、首 chunk 最小**——对应"只有执行步才决定成败"的因果直觉（reward-to-go），credit 集中在真正导致成功/失败的决策上；分母只用 `clamp_min(1e-12)` 防除零，**刻意不设 1.0 下限**（代码注释明确记录了这个 bug 的教训）。

### 3. 具体数值样例

设一轮有 $E=2$ 条 mixed 组 episode：失败 ep A（10 个 chunk）、成功 ep B（3 个 chunk）：

- **不均衡的后果**：若按 chunk 均匀加权，A 占 $10/13 \approx 77\%$ 的梯度量——失败的"长"压过成功的"短"；
- **episode 均衡后**：A 每 chunk 权重 $1/(2\times10)=0.05$，B 每 chunk 权重 $1/(2\times3)=0.1667$；两条 episode 各自的总权重都是 $0.5$（各占一半梯度）；
- **reward-to-go 叠加**（$\gamma=0.99$）：A 的终局 chunk（$c=9$）权重 $0.99^{0}\times0.05 = 0.05$，首 chunk（$c=0$）权重 $0.99^{9}\times0.05 \approx 0.0457$；B 的终局 chunk 权重 $0.99^0 \times 0.1667 = 0.1667$——**成功 episode 的终局 chunk 拿到全场最大权重**，失败 episode 的早期 chunk 权重被双重压低；
- **分母 bug 的放大效应**：若不修分母，$\sum w \approx 0.01\sim0.3$，`clamp_min(1.0)` 会把 loss/梯度放大 $1/\sum w \approx 3\sim100$ 倍，且 ratio_mean 指标同步失真（曾制造"首前向 ratio 漂移"假警报）——修复后 denominator 只防除零，loss 量级正常（真实日志每轮 loss ≈ 160~340）。

> 面试一句话总结：**在线 RL 里失败轨迹更长，天然主导梯度；episode 均衡加权（1/(E·C)）让每条 episode 等权，reward-to-go 折扣（0.99^(C−1−c)）把 credit 集中到终局 chunk，同时把损失分母从 clamp_min(1.0) 修复为只防除零——既治"长失败主导"，又治"credit 稀释"。**

---

## 7. verl-vla 接入：SmolVLATrainableModel 与模型注册补丁

### 1. 现有问题

- verl-vla 官方仓库支持 act_torch / gaussian_actor / openvla_oft / pi0_torch / gr00t 等架构，**没有 SmolVLA**；
- SmolVLA 原生策略在 **LeRobot** 里（`lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy`），预处理/后处理管线也在 LeRobot；
- verl-vla 的模型层约定是"**每个架构一个包 + builder 统一注册 + TrainableModel 包装**"——要接入就得同时满足三件事：模型包、注册分支、训练/rollout hooks。

### 2. 方法论

**① 包装器**：`SmolVLATrainableModel` 继承 verl-vla 的 `TrainableVLAModelBase + SupportSFTTraining`，**原生策略仍是唯一事实源**（预处理/采样/checkpoint 全走 LeRobot），包装器只做两件事：把 verl-vla 的 `DataProto` 观测契约翻译成原生 batch 契约，以及暴露 RL 需要的 hook：

```python
# verl-vla/src/verl_vla/models/smolvla/trainable_model.py
class SmolVLATrainableModel(TrainableVLAModelBase, SupportSFTTraining):
    """SmolVLA = frozen SmolVLM-2 VLM + trainable flow-matching action expert.
    The native policy stays the source of truth for preprocessing, sampling and
    checkpoints; this wrapper only adapts the verl-vla DataProto observation
    contract to the native batch contract and forwards to the native APIs."""
    def __init__(self, policy, *, preprocessor, postprocessor, adapter_config=None):
        super().__init__(policy=policy)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        config = dict(adapter_config or {})
        self.flow_steps = int(config.pop("flow_steps", 10))
        self.sde_sigma = float(config.pop("sde_sigma", 1.0))
        self.sde_eta = float(config.pop("sde_eta", 1.0))
        ...
    # ---- rollout / inference ----
    def select_action(self, obs: DataProto, **kwargs) -> Tensor:
        batch = self.preprocessor(self._policy_batch(obs))
        action = self.policy.select_action(batch, **kwargs)
        return self.postprocessor(action)

    # ---- SFT contract（原生流匹配监督损失） ----
    def sft_loss(self, obs, tokenizer, actions, valids, action_mask=None, target_values=None) -> Tensor:
        ...
        loss, _ = self.policy.forward(batch)   # 原生 flow-matching loss

    # ---- FlowGRPO hooks（本项目新增的 RL 里程碑扩展点） ----
    def rollout_context(self):
        """Autocast context shared by collection and rescoring (bf16 CUDA / fp32 CPU)."""
        return rollout_autocast(self.policy.device if hasattr(self.policy, "device")
                                else next(self.policy.parameters()).device)

    @torch.no_grad()
    def sample_sde_chunk(self, obs: DataProto, *, eta=None, noise=None, generator=None) -> SmolVLATrajectory:
        batch = self.preprocessor(self._policy_batch(obs))
        prefix_pad_masks, prefix_cache = prepare_policy_prefix(self.policy, batch)
        return sample_sde_chunk_fn(self.policy.model, prefix_pad_masks, prefix_cache,
                                   action_dim=self.action_dim, eta=float(self.sde_eta if eta is None else eta),
                                   noise=noise, generator=generator)

    def flow_log_prob(self, obs, trajectory, *, valid_positions=None) -> Tensor:
        """Rescore a fixed trajectory under current weights (differentiable)."""
        batch = self.preprocessor(self._policy_batch(obs))
        prefix_pad_masks, prefix_cache = prepare_policy_prefix(self.policy, batch)
        return recompute_log_probs(self.policy.model, prefix_pad_masks, prefix_cache,
                                   trajectory, valid_positions=valid_positions)

    def flow_policy_loss(self, logp, old_logp, ref_logp, advantages, *,
                         clip_epsilon, kl_beta, valid_steps=None):
        return grpo_loss(logp, old_logp, ref_logp, advantages,
                         clip_epsilon=clip_epsilon, kl_beta=kl_beta, valid_steps=valid_steps)
```

注意 `sample_sde_chunk` 是 `@torch.no_grad()`（采集），而 `flow_log_prob` **不**带 no_grad（重打分必须让梯度穿过速度网络，这是策略梯度更新的通道）——注释里写得很清楚："rescoring must be differentiable; only collection-time sampling is no_grad."

**② 注册补丁**：`patches/apply_patch.sh` 把 `src/smolvla_verl/models/smolvla/` 拷进 verl-vla，再跑 `patch_registration.py` 打两个锚点——`builder.py` 增加 `architecture == "smolvla"` 分支（从 LeRobot checkpoint 构建 TrainableModel），`workers/config/model.py` 增加 `policy_type == "smolvla"` 映射并把 tokenizer/processor 置空（SmolVLA 的 tokenizer 在原生 preprocessor 里，verl worker 不需要）：

```python
# patches/patch_registration.py（builder.py 的 smolvla 分支，摘录）
if architecture == "smolvla":
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE
    from .smolvla import SmolVLAConfig, SmolVLAPolicy, SmolVLATrainableModel, load_smolvla_processors
    ...
    weights_path = model_path / SAFETENSORS_SINGLE_FILE
    if weights_path.is_file():
        policy = SmolVLAPolicy.from_pretrained(path)   # 原生 LeRobot 加载
    ...
    preprocessor, postprocessor = load_smolvla_processors(policy.config, model_path, ...)
    policy.to(dtype=torch_dtype)
    return SmolVLATrainableModel(policy, preprocessor=preprocessor,
                                 postprocessor=postprocessor, adapter_config=adapter_config)
```

`patch()` 是幂等的：`if new in s: SKIP`，可反复执行（对应 AGENTS.md 的"补丁幂等可重放"）。

**这段代码关键在哪**：接入层只做**契约翻译**（DataProto → 原生 batch + task 指令），不碰原生策略实现；SDE/GRPO 逻辑全部在模型包内部（sde.py / sde_sampling.py / grpo.py），而训练循环（第 8、9 点）通过 `sample_sde_chunk / flow_log_prob / flow_policy_loss` 三个 hook 使用它们——**算法、模型、训练循环三者解耦**，这也是为什么本项目可以"复用 verl-vla 模型基建、自研轻量训练循环"（见第 9 点）。

### 3. 具体数值样例

- **参数量与显存**：SmolVLA 0.45B（总参数约 4.5 亿），其中 **SmolVLM-2 VLM 冻结**、仅 flow-matching action expert 可训练（可训练参数只占小头）→ 单卡 RTX 4090 24GB 能同时承载 serve 推理 + GRPO 训练 + 12 实例并发的模型副本；
- **checkpoint 导出**：`export_policy` 走原生 `policy.save_pretrained` + 前后处理器，产出的 `model.safetensors` 是 **LeRobot 原生格式**，评估端 `lerobot_eval` 直接加载（第 10 点）——不需要任何格式转换；
- **补丁可重放**：`apply_patch.sh` 重新执行时 `patch_registration.py` 对已 patch 文件输出 `SKIP (already patched)`，保证 CI/换机重装不重复打补丁。

> 面试一句话总结：**verl-vla 接入 SmolVLA 遵循"原生策略是唯一事实源"：TrainableModel 包装器只做 DataProto→原生 batch 的契约翻译并暴露 sample_sde_chunk / flow_log_prob / flow_policy_loss 三个 FlowGRPO hook，builder 注册分支 + worker 配置由幂等补丁注入，采集/重打分/损失三者因此能在同一套模型基建上闭环。**

---

# 二、整体架构：分离式训推链路（本地采集 ↔ 云端 serve ↔ 云端训练 ↔ 评估）

> 从第 8 点开始，前面 7 个技术点被串成一条完整流水线。整体拓扑（来自 `docs/architecture.md`）：

```text
本地（8GB 4060 / WSL2）                服务器（RTX 4090 24GB）
┌─────────────────────┐   上传/SFTP   ┌──────────────────────────────┐
│ LIBERO 轨迹采集      │ ───────────▶ │ verl-vla（含 smolvla 扩展）    │
│ scripts/collect_*   │   HTTP/JSON   │ serve_smolvla.py（:8000）      │
│ 12 实例并行          │              │   /predict 记录 chunk 级轨迹    │
│ 环境仅执行+判定      │              │   /finish 记账组优势            │
└─────────────────────┘              │ grpo_offline.train_from_sessions│
        ▲                            │   /train 逐 chunk 重打分+GRPO    │
        │ 拉权重/评测结果             │ run_loop_opt.sh（48 集/轮×15 轮）│
        └────────────────────────────├──────────────────────────────┤
                                     │ eval_parallel.sh → 官方评估     │
                                     └──────────────────────────────┘
```

## 8. 分离式链路：collect_remote ↔ serve_smolvla ↔ grpo_offline（HTTP 端点与 chunk 级轨迹记录）

### 1. 现有问题

- 单机采集 + 训练串行跑，环境仿真（MuJoCo）和模型训练抢同一张卡，且**训练中断=全部重来**；
- 训练需要"每条轨迹的观测 + SDE 去噪状态 + 逐元素 log-prob"，采集端一次性算完再整体上传，数据量大、耦合深、无法增量；
- 需要"环境只负责执行与成功判定、模型只负责出动作、训练只负责更新"的解耦（简历亮点原话）。

### 2. 方法论

**服务端** `scripts/serve_smolvla.py`：一个 FastAPI 服务，内存里按 `SESSIONS[session_id][episode_id]["chunks"][chunk_id]` 记录每条轨迹。`/predict` 在**返回动作的同时**把 chunk 级轨迹落进 session（图像 float32 存储、SDE 状态与 logp 以 b64 序列化）：

```python
# scripts/serve_smolvla.py
@app.post("/predict")
def predict(req: PredictRequest):
    ...
    with torch.no_grad(), sde_sampling.rollout_autocast("cuda"):
        pm, pc = sde_sampling.prepare_policy_prefix(POLICY, batch)
        traj = sde_sampling.sample_sde_chunk(POLICY.model, pm, pc, action_dim=ACTION_DIM, eta=req.eta)
    actions = traj.actions[:, :, :ACTION_DIM]
    actions = POSTPROCESSOR(actions)
    n_exec = min(ACTION_STEPS, CHUNK_SIZE)          # 采样 10 步 chunk，只执行前 5 步
    # 记录 chunk（obs CPU + SDE trajectory）供后续训练；图像保持 float32：
    # rescoring 必须逐位复现采集数值路径（半精度存储会悄悄破坏 ratio=1）
    batch_cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with SESSIONS_LOCK:
        SESSIONS.setdefault(req.session_id, {})[req.episode_id] = ...
        SESSIONS[req.session_id][req.episode_id]["chunks"][req.chunk_id] = {
            "batch": batch_cpu,
            "states": _tob64(traj.states),                       # (11,10,7) float32
            "element_log_probs": _tob64(traj.element_log_probs), # (10,10,7) float32
            "valid_action_mask": _tob64(traj.valid_action_mask),
            "valid_positions": _tob64(torch.zeros((1, CHUNK_SIZE), dtype=torch.bool)
                                      .index_fill(1, torch.arange(n_exec), True)),
            "n_exec": n_exec, "eta": req.eta, "success": False,
        }
    return {"actions": actions[0, :n_exec].detach().cpu().numpy().tolist(), ...}
```

`/finish` 用客户端回传的**实际执行步数**掩掉终局 chunk 未执行的规划后缀（"规划 10 步只执行了 3 步就成功/截断"），并给组记账：

```python
@app.post("/finish")
def finish(req: FinishRequest):
    """Mark episode outcome; when a group completes, compute reset-matched advantages."""
    ...
    if req.executed_steps:
        for cid_str, n_exec in req.executed_steps.items():
            ch = ep["chunks"].get(int(cid_str))
            ...
            if n_exec < ch["n_exec"]:
                vp = ... .reshape(1, CHUNK_SIZE)
                vp[0, n_exec:] = False          # 掩掉终局 chunk 未执行的动作位
                ch["valid_positions"] = _tob64(torch.from_numpy(vp))
    ep["success"] = req.success
    GROUP_RESULTS.setdefault(req.group_id, {})[req.episode_id] = req.success
    return {"status": "ok"}
```

`/train` 直接把内存 session 交给离线训练器（第 6 点那个 `train_from_sessions`），`/clear` 清空 session 供下一轮 on-policy 采集。

**采集端** `scripts/collect_remote.py`：薄客户端，只跑环境。每组 4 条 rollout **钉同一初始场景**（reset-matched），逐 chunk POST 观测、执行返回的动作，episode 结束 POST `/finish`：

```python
# scripts/collect_remote.py
def run_one(episode_id: int, seed: int):
    torch.manual_seed(seed + episode_id * 7919)     # 仅策略噪声 seed 不同
    base_env.init_state_id = args.init_state_id     # 同组同初始状态（reset-matched）
    obs, _ = env.reset(seed=seed)
    ...
    while total < max_steps and not success:
        payload = {"session_id": ..., "episode_id": episode_id, "chunk_id": chunk_id,
                   "group_id": ..., "pixels": {k: _img_b64(v) for k, v in obs["pixels"].items()},
                   "robot_state": _conv(obs["robot_state"]), "task": task_desc, "eta": args.eta}
        r = _post(predict_url, payload)
        acts = np.array(r["actions"], dtype=np.float32).reshape(-1, 7)
        for a in acts:                              # 执行服务端返回的动作
            obs, reward, terminated, truncated, info = env.step(a.reshape(1, 7))
            ...
        executed_steps[chunk_id] = n_exec
        chunk_id += 1
    _post(finish_url, {"session_id": ..., "episode_id": episode_id, "group_id": ...,
                       "success": success, "executed_steps": executed_steps})
```

**这段代码关键在哪**：`/predict` 的返回里**同时**带 `actions`（给环境）和隐式记录（给训练）——一次 HTTP 往返完成"推理 + 轨迹采集"；`/finish` 是"mask 语义"的唯一入口（终局 chunk 实际执行步数）；采集端把"同组同初始场景"落实成 `base_env.init_state_id` 赋值，这是第 5 点组基线成立的物理前提。

### 3. 具体数值样例

- **每 chunk 的传输/存储量**（估算）：观测 2 张 $224\times224\times3$ uint8 图 $\approx 301$ KB，b64 后约 400 KB；SDE 状态 $(11,10,7)$ float32 = 3 KB + 逐元素 logp $(10,10,7)$ = 2.8 KB + 掩码若干；**单次 `/predict` 请求 ≈ 0.4 MB 上行**；
- **一轮的请求量**：48 集 × 平均 25.5 chunk/集 ≈ **1222 次 `/predict` + 48 次 `/finish`**（与真实日志第 1 轮 `chunks=1222` 完全吻合）→ 一轮上行 ≈ 0.5 GB；
- **chunk 语义**：`chunk_size=10` 采样 10 步动作，`n_exec = min(action_steps=5, 10) = 5`——每次只执行前 5 步；若中途成功/截断，`/finish` 再把终局 chunk 的 `valid_positions[3:]` 之类置 False，未执行动作不进损失（第 11 点 bug #3 的修复位置）；
- **角色解耦**：环境进程（12 个）不知道模型权重、训练进程不知道 MuJoCo——换模型只改 serve 端 checkpoint，换环境只改采集端配置，训练与推理完全分离（简历亮点：本地环境采集与云端 FastAPI 推理服务/训练解耦）。

> 面试一句话总结：**分离式链路 = 环境端只做"执行 + 成功判定"，服务端 `/predict` 在返回动作的同时把 chunk 级轨迹（观测 + SDE 状态 + 逐元素 log-prob，float32）记进内存 session，`/finish` 按实际执行步数掩码并记账，`/train` 消费 session 做离线 GRPO——一次 HTTP 往返同时完成推理与数据采集，训练与推理彻底解耦。**

---

## 9. 在线训练主循环：12 实例并行 on-policy rollout + 单权重夹覆盖 + serve 重启

### 1. 现有问题

- RL 是 **on-policy** 的：每轮必须用**最新权重**重新采集，否则"采集时策略"与"训练后策略"不一致，log-prob 对不上（第 4 点）；
- 单进程采集太慢：48 集/轮串行跑不现实，需要多实例并行；
- 训练要能**断点续训 + 时间截止**（服务器租用场景），且权重管理要简单可靠；
- 长驻进程连续多轮训练后会积累 CUDA 上下文污染（真实踩坑：连续 3 轮 `/train` 后 `/predict` 触发 device-side assert）。

### 2. 方法论

**编排脚本** `scripts/run_loop_opt.sh`（正式 run 用的就是它）：

```bash
# scripts/run_loop_opt.sh（摘录主循环）
for r in $(seq "$START_ROUND" "$ROUNDS"); do
  [ 到达 STOP_AT 时间 ] && break          # 时间截止
  PIDS=()
  for i in $(seq 0 $((INSTANCES-1))); do   # 12 实例并行采集
    task=${TASKS[$(( (r*INSTANCES + i) % 10 ))]}        # 任务逐轮轮换
    seed=$((20260901 + r*10000 + i*7))
    init_state=$(( (r + i) % 10 ))                       # 初始状态 0-9 轮换
    "$LOCAL_PYTHON" scripts/collect_remote.py \
      --server "$SERVER" --suite libero_spatial --task-id "$task" \
      --rollout-n 4 --group-id "r${r}_g${i}" --session-id "r${r}_s${i}" \
      --eta 0.05 --max-steps 280 --action-steps 5 --seed "$seed" --init-state-id "$init_state" &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done           # 等全部采集完
  n_ok=$(grep -hac "success=True" work/logs/opt_r${r}_i*.log | awk '{s+=$1} END{print s+0}')
  TRAIN_RESP=$(curl -s -X POST "$SERVER/train?lr=$GRPO_LR&steps=$GRPO_STEPS&batch_size=$BATCH_SIZE&chunk_discount=0.99")
  curl -s -X POST "$SERVER/clear" > /dev/null             # 清 session
  # 最佳权重备份（smolvla_best）+ 重启 serve 加载最新权重（on-policy + 规避 CUDA 污染）
  CHECKPOINT=/home/ubuntu/runs/smolvla_grpo bash .../serve_start.sh
done
```

**任务与初始状态轮换的动机**：评估测 10 个初始状态（0-9），训练若只覆盖部分初始状态就会过拟合训练分布（第 12 点根因）；`task=(r·12+i)%10`、`init_state=(r+i)%10` 让每轮覆盖不同 (task, init_state) 组合。**组内 reset-matched** 的完整四元组生成逻辑在多进程版 `grpo_libero.py` 里看得最清楚：

```python
# src/smolvla_verl/trainer/grpo_libero.py
for g in range(args.groups_per_round):
    task_id = task_ids[(rnd * args.groups_per_round + g) % len(task_ids)]
    init_state_id = (rnd + g) % args.init_state_count     # 初始状态轮换 0-9
    env_seed = args.seed + rnd * 1000 + g                 # 同组共享场景 seed
    for i in range(args.rollout_n):
        policy_seed = args.seed + 100_000 + rnd * 1000 + g * args.rollout_n + i
        tasks.append((task_id, init_state_id, env_seed, policy_seed))
# 组内 4 条 rollout 只有 policy_seed 不同 → reset-matched
...
for g, eps in grouped.items():
    successes = [float(e["success"]) for e in eps]
    if all(s == successes[0] for s in successes):
        continue          # 全成功/全失败组：无组内信号，跳过（第 5 点）
    advs = compute_group_advantages(torch.tensor([successes], dtype=torch.float32))[0].tolist()
    ...
```

**单权重夹**：`/home/ubuntu/runs/smolvla_grpo/model.safetensors` 每轮覆盖（`policy.save_pretrained`），配合 `last_round` 进度文件断点续训 + `smolvla_best` 最佳备份。**serve 每轮重启** = on-policy（采集端永远用最新权重）+ 规避长驻 CUDA 污染。

### 3. 具体数值样例

正式 run（`configs/grpo_formal.yaml` + `work/logs/grpo_opt.log` 真实数据）：

- **规模**：12 实例 × 4 rollout = **48 集/轮**，15 轮；`lr=5e-6, steps=1, batch_size=32`（配置值；实现里强制 **1 chunk/minibatch**，批处理重打分会破坏 logp，见第 4 点）、`clip_epsilon=0.2, kl_beta=0.01, chunk_discount=0.99, eta=0.05`；
- **每轮训练量**：`chunks` 在 **800~1590** 之间波动（取决于成功率与轨迹长度），loss ≈ 160~340，`ratio_mean` ∈ $[0.99999987, 1.00000005]$；
- **每轮采集成功率**：45.8% ~ 72.9%，15 轮均值 ≈ **62%**（真实逐轮：r1=62.5%, r2=70.8%, r5=45.8%, r9=72.9% …），无退化趋势、watchdog 未触发；
- **时间控制**：`STOP_AT` 到期自动停（日志里 `round 10/60` → 改 `ROUNDS=15` 续跑），`START_ROUND` 支持断点续训；
- **权重流**：每轮 `/train` 保存 → `smolvla_best` 备份 → serve 重启加载 → 下一轮采集用新权重 → **严格 on-policy**。

> 面试一句话总结：**在线 RL 主循环 = 每轮 12 实例并行采集 48 集（任务与初始状态逐轮轮换、组内 reset-matched）→ 统计成功率 → /train 离线 GRPO → /clear → 重启 serve 加载最新权重；单权重夹 + last_round 进度 + 时间截止实现断点续训，serve 每轮重启保证 on-policy 并规避 CUDA 上下文污染。**

---

## 10. 评估方法学：官方 10×10 协议、逐任务独立进程、确定性重测

### 1. 现有问题

- RL 训练结果必须用**与论文一致的官方协议**评估，否则数字不可比（SmolVLA 论文报 LIBERO-Spatial 90%，社区复评普遍只有 ~63%，协议/实现差异很大）；
- 单进程跑 100 集评估会偶发 CUDA 崩溃（真实踩坑：`CUDA error: misaligned address`），且一崩全毁；
- 100 集评估本身有 ±5% 噪声，单次运行结论不可信。

### 2. 方法论

官方协议：LIBERO-Spatial 每任务 10 trials × 10 任务 = 100 集；`seed=1000` 确定性；**评估用确定性 ODE 采样**（不是训练用的 SDE）；`n_action_steps=5`（与采集端一致的 chunk 回退执行，避免"评估极慢且口径漂移"）。

`scripts/eval_parallel.sh`：**逐任务独立进程、两波并行**，单任务失败可单独重跑：

```bash
# scripts/eval_parallel.sh（摘录）
run_task() {
  tid=$1
  "$ENV/bin/python" -m lerobot.scripts.lerobot_eval \
    --policy.path="$POLICY" --env.type=libero --env.task="$TASK" --env.task_ids="[$tid]" \
    --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false \
    --policy.device=cuda --policy.n_action_steps=5 --seed=1000 --output_dir="$TDIR"
}
for wave in 0 1; do
  for tid in $(seq $((wave*5)) $((wave*5+4))); do run_task $tid & done   # 每波 5 个独立进程
  wait
done
```

**确定性重测**：同一权重、同一 seed 重跑两次，结果**逐任务完全一致**（`grpo_final_pertask_run2` 与 run1 逐位一致）——用"可复现"换取可信度，抵消 100 集评估的随机噪声。评估结果解析用 `scripts/parse_eval_results.py`，逐任务输出 `task_$tid/` 目录并保留视频证据。

### 3. 具体数值样例

最终评估（2026-08-19，修复后 15 轮权重 vs 同日重测 Base）：

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 总体 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO-15轮 | 8/10 | 5/10 | 8/10 | 4/10 | 6/10 | 2/10 | 8/10 | 5/10 | 9/10 | 8/10 | **63/100 = 63.0%** |
| Base（同日重测） | 5/10 | 8/10 | 7/10 | 4/10 | 4/10 | 6/10 | 8/10 | 6/10 | 7/10 | 8/10 | **63/100 = 63.0%** |

- 两者**重测 ×2 完全一致**（seed=1000 确定性）；旧管道 R16（修复前）为 55.0%；
- 逐任务对比：GRPO 在 task0/2/4/8 提升（8 vs 5、8 vs 7、6 vs 4、9 vs 7），在 task1/5/7 下降（5 vs 8、2 vs 6、5 vs 6）——**任务间此消彼长，总体打平**，这正是"数据量不足 + 初始状态覆盖稀疏"的表现（第 12 点）；
- 官方论文基线 90% vs 本项目实测 63%：社区复评普遍低于论文（协议/实现差异），63.0% 是可复现的可信基线，且**训练/评估口径（n_action_steps=5、确定性采样）完全对齐**。

> 面试一句话总结：**评估走官方协议（10 任务 × 10 trials、seed=1000、确定性 ODE、n_action_steps=5 与采集一致），逐任务独立进程并行避免单进程 100 集崩溃，同权重重测 ×2 完全一致换取可信度——最终 GRPO 63.0% 与 Base 63.0% 打平，任务级此消彼长说明是数据/覆盖问题而非算法退化。**

---

# 三、排障与实验结论

## 11. 正确性排障：让"训练白跑一天"的 6 个 bug（M1 → M2 → M3）

### 1. 现有问题

训练迭代过程中，模型"跑是跑起来了"，但**官方指标不升反降**：M1（无锚定）后期采集成功率掉到 45%；M2（固定 base 锚定但未修正确性 bug）16 轮后官方指标 55.0% **低于 Base 63.0%**——训练白跑一天还退化了。这类问题最危险：**loss 在降、ratio 指标在动，但训练目标本身是错的**。

### 2. 方法论

排障思路 = **把"静默错误"逐个变成"显式检查"**。按"先正确性、后工程"排序，完整清单见 `docs/修改与补丁汇总.md`，6 个正确性 bug 如下：

1. **批处理重打分破坏 log-prob（头号根因）**：`/train` 对整批会话一次重打分，`valid_positions`/mask 与各 chunk 实际长度错位 → old logp 与采集不一致 → ratio 漂移。修复：**逐 chunk 重打分** + 首 chunk 一致性守卫（>0.05 nats 报错）。（第 4 点）
2. **组内不同初始场景**：同组 4 条 rollout 初始状态不一致 → 组均值基线失效。修复：reset-matched（同 task/init_state/env_seed）+ 按轮轮换 init_state 0-9。（第 5、9 点）
3. **终局 chunk mask 错误**：规划的 10 步 chunk 只执行了 5 步（甚至 3 步就成功/截断），未执行部分却计入损失。修复：客户端回传 `executed_steps`，服务端 `/finish` 把 `valid_positions[n_exec:]` 置 False。（第 8 点）
4. **`grpo_loss` 加权分母 bug**：分母 `clamp_min(1.0)`，而 episode 均衡后 $\sum w$ 只有 0.01~0.3 → loss/梯度被放大 $1/\sum w$ 倍，还污染 ratio_mean 指标、制造"首前向 ratio 漂移"假警报。修复：只防除零 `clamp_min(1e-12)`。（第 6 点）
5. **训练目标只应更新 mixed 组**：全成功/全失败组优势全 0，更新它们只产生噪声梯度（还叠 KL 惩罚）。修复：`all(s == successes[0]) → continue`；chunk 按 episode 均衡加权。（第 5、6 点）
6. **`final_info` 解析（dict 或 list）**：episode 成功信息两种格式解析不一致 → 漏判/崩训练。修复：`_success_from_step` 统一兼容（`dict.get("is_success")` / `final_info[0]` / `info` 兜底）。

另有两条**数值/工程**修复：**图像 float32 存储**（半精度存图悄悄破坏 ratio=1，第 4 点）与 **serve 每轮重启**（规避长驻进程 CUDA 上下文污染，第 9 点）。

### 3. 具体数值样例

- **M1 → M2 → M3 的三次实验**：M1（4 实例×4=16 集/轮、lr=1e-6、无锚定）R3 官方 62.0%，后期采集成功率掉到 45%（策略漂移）；M2（12 实例×4=48 集/轮、固定 base 锚定、**未修正确性 bug**）16 轮后官方 **55.0%**；M3（修复全部 6 个 bug + `chunk_discount=0.99` + `eta=0.05` + `steps=1`）15 轮后 **63.0%**；
- **守卫效果**：修复后 15 轮 `ratio_mean` ∈ [0.99999987, 1.00000005]（第 4 点），首 chunk 重打分偏差 < 0.05 nats 才放行——"静默退化"变成"显式失败"；
- **每轮跳过量**：约 1/3 的组是全成功/全失败组被跳过（48 集中约 16 集不产生梯度），这部分数据量是"提升不足"的原因之一（第 12 点）。

> 面试一句话总结：**训练"跑一天没效果"几乎都是目标函数的静默错误：批处理重打分错位、组内场景不一致、终局 mask 错误、加权分母 clamp、更新了无信号的 collapsed 组、final_info 解析歧义——逐个修复并把关键不变量（ratio≈1、重打分偏差<0.05）变成显式守卫后，官方指标从 55.0% 恢复到 63.0%，退化被消除。**

---

## 12. 实验结果与根因分析：63.0% vs 63.0%，为什么持平未超越

### 1. 现有问题

RL 训练的目标是**超越**基座；最终结果是与基座持平（63.0% = 63.0%）。面试官一定会问："你的 RL 到底学到东西没有？" 诚实且量化的回答比吹牛重要——本文档的结论是：**修复退化是确定的收益（55%→63%），但数据量与初始状态覆盖不足以产生统计显著的提升**。

### 2. 方法论

根因分析（`RESULTS.md` / `docs/训练评测分析.md` 四连）：

1. **初始状态分布不匹配**：评估测 10 个初始状态（0-9），修复前训练只覆盖部分（过拟合训练初始状态）；修复后按轮轮换已缓解，但**每轮每个初始状态只出现约 1 次**（48 集 / 10 状态 × 12 组 ≈ 每状态 4~5 条/轮），采样仍稀疏；
2. **数据量偏少**：48 集/轮 × 15 轮 = 720 集总量，且其中约 **1/3 是全成功/全失败组**（不产生梯度）——真正有学习信号的只有约 480 集；
3. **紧 KL 锚 + 低 lr**：防退化优先（`kl_beta=0.01`、`lr=5e-6`），单轮学习幅度小，15 轮不足以拉开差距；
4. **评估噪声**：100 集评估本身 ±5% 噪声（多次运行 62~64% 都在 base 附近波动），该数据量下"提升 1-2pp"与噪声不可区分。

### 3. 具体数值样例

- **训练过程 vs 官方指标**：训练采集成功率 15 轮均值 ~62%（45.8%~72.9% 波动），官方评估 63.0%——两者量级吻合，说明训练口径与评估口径一致（无隐藏的分布偏移）；
- **统计显著性**：100 集评估下成功率的二项分布标准误 $\sqrt{0.63\times0.37/100} \approx 4.8\%$，95% 置信区间半宽约 $\pm 9.5\%$；多次独立运行实测结果在 62%~64% 之间波动（±5%，`RESULTS.md` 口径），GRPO 63.0 vs Base 63.0 完全落在噪声带内——该数据量下无法区分"真实提升 1~2pp"与随机噪声；
- **逐任务信号**：GRPO 在 4 个任务提升、3 个任务下降（第 10 点表）——若真有稳定的策略改进，应该看到"多数任务同向"，此消彼长说明是任务间噪声而非系统性收益；
- **对照旧管道**：修复前 R16 = 55.0%（比 Base 低 8pp），修复后 63.0%（打平）——**"修复退化"本身是 8pp 的确定性收益**，这是本次实验最硬的结论。

> 面试一句话总结：**15 轮 GRPO 官方 63.0% 与 Base 持平（重测 ×2 完全一致）：修复正确性 bug 消除了 8pp 的退化（55%→63%），但受限于"每轮 48 集、初始状态覆盖稀疏、约 1/3 组无信号、紧 KL 低 lr、100 集评估 ±5% 噪声"，尚不足以产生统计显著的超 base 提升——下一步是初始状态全覆盖采样 + 加大每组 rollout 数 + 200 集评估降噪。**

---

# 附：组件速查表

## 组件 ↔ 文件 ↔ 作用

| 组件 | 文件 | 作用 |
|---|---|---|
| SDE 核（score/transition/logprob） | `verl-vla/src/verl_vla/models/smolvla/sde.py` | ODE→边缘保持 SDE，闭式高斯密度（第 2 点） |
| chunk 采样 + 重打分 + prefix cache | `verl-vla/src/verl_vla/models/smolvla/sde_sampling.py` | 一次编码多次去噪、逐 chunk 重打分、autocast 统一（第 3、4 点） |
| GRPO 损失 | `verl-vla/src/verl_vla/models/smolvla/grpo.py` | 组优势、k3 KL、clipped ratio + episode 加权（第 5、6 点） |
| 可训练包装 | `verl-vla/src/verl_vla/models/smolvla/trainable_model.py` | DataProto→原生 batch、rollout/sft/flow hooks（第 7 点） |
| 注册补丁 | `patches/apply_patch.sh` + `patch_registration.py` | builder + worker 配置注入，幂等可重放（第 7 点） |
| 服务端 | `scripts/serve_smolvla.py` | /predict 记录 chunk 轨迹、/finish 掩码记账、/train、/clear（第 8 点） |
| 采集端 | `scripts/collect_remote.py` | 薄客户端：环境执行 + reset-matched + 上传（第 8 点） |
| 离线训练器 | `src/smolvla_verl/trainer/grpo_offline.py` | 从 session 逐 chunk 重打分 + GRPO + 单权重夹（第 6、8 点） |
| 训练主循环（多进程版） | `src/smolvla_verl/trainer/grpo_libero.py` | 任务/初始状态轮换、collapsed 组跳过、ratio 守卫（第 9 点） |
| 编排脚本 | `scripts/run_loop_opt.sh` | 12 实例并行 + 时间截止 + 断点续训 + serve 重启（第 9 点） |
| 官方评估 | `scripts/eval_parallel.sh` + `lerobot_eval` | 10 任务两波并行、seed=1000、确定性 ODE（第 10 点） |

## 关键超参（正式 run `configs/grpo_formal.yaml`）

| 参数 | 值 | 含义 |
|---|---|---|
| `instances` / `rollout_n` | 12 / 4 | 12 实例并行 × 每实例 4 rollout = 48 集/轮 |
| `rounds` | 15 | 训练轮数（STOP_AT 可截断） |
| `chunk_size` / `action_steps` | 10 / 5 | 每 chunk 采样 10 步、执行前 5 步 |
| `eta` | 0.05 | SDE 噪声（评估用确定性 ODE，口径接近） |
| `lr` / `steps` / `batch_size` | 5e-6 / 1 / 32 | 优化（实现强制 1 chunk/minibatch） |
| `clip_epsilon` / `kl_beta` | 0.2 / 0.01 | 裁剪窗 / KL 惩罚系数 |
| `chunk_discount` | 0.99 | reward-to-go 折扣 |

## 简历亮点 ↔ 本文章节映射

| 简历亮点（项目二 4 条职责） | 对应章节 |
|---|---|
| 1) 分离式训练链路：本地 LIBERO 采集 ↔ 云端 FastAPI 服务/训练解耦，FastAPI 记录 chunk 级轨迹（观测、SDE 去噪状态、log-prob），云端统一 GRPO | 第 8 点（链路）、第 9 点（主循环） |
| 2) 确定性 ODE 流匹配采样 → 边缘保持 SDE，打通策略梯度路径 | 第 1、2 点（SDE 改造） |
| 3) critic-free FlowGRPO：episode 等权归一化 + reward-to-go 折扣 | 第 5、6 点（损失设计） |
| 4) 复用 VLM prefix KV cache + 逐 chunk 重打分 + 数值路径统一，ratio_mean 全程稳定 1.0 | 第 3、4 点（数值一致性） |
| 项目简介：15 轮完成，官方 10×10 通过率 63.0%（与基座持平） | 第 10、12 点（评估与根因） |
