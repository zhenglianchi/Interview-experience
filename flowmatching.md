# Flow Matching 完全指南：确定性 ODE 与边缘保持 SDE（从零一步步推导）

> 对应第二个项目（SmolVLA-Verl）的核心算法改造：**"把确定性 ODE 流匹配采样改造成边缘保持 SDE，为 flow-matching VLA 打通策略梯度路径"**。一句话知识框架：
> **Flow Matching 让模型学一个"速度场"，把高斯噪声"推"成数据；确定性版本（ODE）每一步走固定的路，没有概率密度 → RL 算不了 log-prob；边缘保持 SDE 在每一步加一点恰到好处的噪声（并用 score 修正 drift 让边缘分布不变），于是每步转移变成有闭式 log-density 的高斯 → RL 可训。**
>
> 本文是 `SmolVLA-VERL.md` 第 1、2 点的**完整展开版**：那里是"面试速答"，这里是"从零推导"。假设读者完全没接触过 flow matching / diffusion，每一步符号都解释。核心实现见项目源码 `verl-vla/src/verl_vla/models/smolvla/sde.py`（移植自 FlowVLA-RL / Flow-GRPO）。

---

# 一、先建立直觉：Flow Matching 是什么、解决什么问题

## 1. 生成模型的两种思路：自回归 vs "把噪声推成数据"

### 1. 现有问题（为什么要生成模型？）

机器人要输出动作，本质是**从一个分布里采样**：给定观察 $o$（图像+状态+语言），动作序列 $a$ 服从某个条件分布 $p(a|o)$。生成模型的任务就是学会从这个分布采样。两大流派：

- **自回归（LLM 的做法）**：把输出拆成 token 序列，一个接一个预测 $p(a_1|o), p(a_2|o,a_1)...$。每个 token 有概率，天然适合 RL。
- **连续/并行生成（diffusion / flow matching 的做法）**：把整个动作序列 $a$ 当成一个"向量"，先撒一把高斯噪声，再**一步步把噪声"推"成 $a$**。适合**连续动作**（机械臂关节角度、速度），因为动作本来就是连续向量，不需要 token 化。

SmolVLA 是流匹配 VLA：一次**并行生成**整个动作 chunk（如 $10$ 步 $\times$ $7$ 维 $=70$ 个动作标量），而不是逐 token 吐。

### 2. 方法论（核心比喻）

把生成过程想成**吹气球**：
- 开始（$t=1$）：一个高斯噪声球 $\epsilon \sim \mathcal{N}(0, I)$——什么都不是；
- 结束（$t=0$）：一个动作样本 $a$——我们要的东西；
- 中间：从噪声"变形"成动作的过程，每个中间时刻 $t$ 有一个"半成品" $x_t$。

模型学的是**怎么变形**：给一个"半成品" $x_t$ 和时刻 $t$，预测"下一步该往哪个方向推"——这个方向叫**速度场 $v(x_t, t)$**。学好了之后，从噪声出发沿着速度场一步步走，就走到了动作。

### 3. 具体数值样例（一维直观版）

假设动作是 1 维标量，真实动作 $a = 0.5$（这个样本在数据集里出现过一次）：
- 采样噪声 $\epsilon = -1.2$；
- 定义"变形路径"为最简单的一维线性插值：$x_t = t \cdot \epsilon + (1-t) \cdot a$；
- 那 $t=1$ 时 $x_1 = -1.2$（纯噪声），$t=0$ 时 $x_0 = 0.5$（动作），中间 $t=0.5$ 时 $x_{0.5} = 0.5\times(-1.2) + 0.5\times0.5 = -0.35$；
- **速度 = 位置对时间的导数**：$v = dx_t/dt = \epsilon - a = -1.2 - 0.5 = -1.7$（常数，因为线性插值的速度恒定）。

模型要学的就是这个速度：给它看 $x_t$ 和 $t$，让它猜"这条路径的速度是多少"。

> 面试一句话总结：**Flow Matching = 学一个速度场 $v(x_t,t)$，把高斯噪声通过"逐点推进"变成数据样本；SmolVLA 用它一次性并行生成整个动作 chunk，是连续动作生成的现代主流（相对自回归逐 token）。**

---

## 2. 时间约定与插值路径：把"方向"钉死

### 1. 现有问题

不同论文时间约定不同（有的 $t=0$ 噪声、$t=1$ 数据；SmolVLA 反过来）。不钉死约定，后面公式全乱。**本文全部采用 SmolVLA / 本项目代码的约定**。

### 2. 方法论（两个核心定义）

**① 线性插值路径**（噪声 $\epsilon$ 与动作 $a$ 之间"拉一条直线"）：
$$x_t = t \cdot \underbrace{\epsilon}_{\text{noise}} + (1-t) \cdot \underbrace{a}_{\text{action}}, \qquad t \in [0, 1]$$

- $t=1$：$x_1 = \epsilon$（高斯噪声）——**SmolVLA 里 $t=1$ 是噪声**；
- $t=0$：$x_0 = a$（动作）——$t=0$ 是动作；
- 去噪方向：从 $t=1$ 走到 $t=0$（时间**减小**），一步从 $t$ 到 $t-h$。

**② 速度场**（路径对时间的导数，因为 $x_t = t\epsilon + (1-t)a$）：
$$v = \frac{dx_t}{dt} = \epsilon - a$$

**为什么是这个约定**：这样"速度" $v = \epsilon - a$ 恰是"从动作指向噪声的向量"（反着看，$a = \epsilon - v$）。去噪时沿 $-v$ 方向走就是在"从噪声去掉噪声成分、逼近动作"。代码注释原文（`sde.py`）：

```python
# For x_t = t * noise + (1-t) * action and v = noise - action
```

**一个重要代数关系（后面推导 score 要用）**：把 $a$ 用 $x_t, v$ 表示出来——由 $x_t = t\epsilon + (1-t)a$ 与 $v = \epsilon - a$：
$$\epsilon = a + v, \qquad x_t = t(a+v) + (1-t)a = a + tv \quad\Rightarrow\quad \boxed{a = x_t - t\,v}$$
含义：**给定"半成品 $x_t$ + 模型预测的速度 $v$，就能反推出这条路径对应的完整动作 $a$**。这是 flow matching 能"一步到位去噪"（而非逐层）的理论基础。

### 3. 具体数值样例

沿用上例（1 维）：$\epsilon = -1.2$，$a = 0.5$，$v = -1.7$：
- $t=1$：$x_1 = 1\times(-1.2) + 0 = -1.2$；由 $a = x_1 - 1\times v = -1.2 - (-1.7) = 0.5$ ✓；
- $t=0.5$：$x_{0.5} = -0.35$；由 $a = x_{0.5} - 0.5\times v = -0.35 - 0.5\times(-1.7) = -0.35 + 0.85 = 0.5$ ✓；
- 无论从哪个中间时刻，用 $(x_t, v)$ 都能精确还原 $a = 0.5$——**这就是"学速度 = 学怎么一步到位去噪"的数值直觉**。

> 面试一句话总结：**SmolVLA 的时间约定是 $t=1$ 噪声、$t=0$ 动作，路径 $x_t = t\epsilon+(1-t)a$、速度 $v=\epsilon-a$，且 $a = x_t - tv$——学速度等价于学会"从任意中间态一步还原动作"。**

---

# 二、确定性 ODE 流匹配：训练学什么、采样怎么走

## 3. 训练：条件流匹配目标（模型到底学什么）

### 1. 现有问题

我们希望模型学"边缘速度场"：$v_\theta(x_t, t) \approx \mathbb{E}[\,\epsilon - a \mid x_t\,]$（给定半成品，期望速度）。但直接监督它有个大麻烦：**边缘分布 $p_t(x_t)$ 是"所有样本路径混合"的结果，写不出闭式**（$p_t(x) = \mathbb{E}_a[\mathcal{N}(x; (1-t)a,\, t^2 I)]$ 是对数据分布求期望，没法解析算）——那"边缘速度的真值"就没法算了。

### 2. 方法论（条件流匹配 CFM：绕开边缘、监督条件）

关键 trick：**不要监督"边缘速度"，监督"条件速度"**——每次采样时把动作 $a$（和噪声 $\epsilon$）都固定住，这条具体的"直线路径"的速度 $\epsilon - a$ 是确定的、好算的：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, \epsilon \sim \mathcal{N}(0,I),\, a \sim p_{\text{data}}}\Big[\big\|\, v_\theta(x_t,\, t) - (\epsilon - a) \,\big\|^2\Big], \qquad x_t = t\epsilon + (1-t)a$$

- 随机抽一个动作 $a$（训练样本）、一个噪声 $\epsilon$、一个时刻 $t$；
- 算出中间态 $x_t = t\epsilon + (1-t)a$；
- 让模型 $v_\theta(x_t, t)$ 去拟合这条路径的真实速度 $\epsilon - a$（**纯 MSE 回归**）；
- **理论保证**：优化 CFM 目标的最优解，恰好等于边缘速度场 $\mathbb{E}[\epsilon-a | x_t]$（可以证明两者有相同的梯度/最优解——这就是"条件化也没问题"的原因）。模型不需要显式知道 $p_t$，只需要"见到中间态会往哪推"。

**训练时模型只需要前向一次**（预测速度），不需要像 diffusion 那样同时预测噪声和数据——这是 flow matching 简洁的地方。

### 3. 具体数值样例

一个 mini-batch 的 3 条训练样本：
| 样本 | 动作 $a$ | 噪声 $\epsilon$ | 时刻 $t$ | 中间态 $x_t$ | 监督目标 $v=\epsilon-a$ |
|---|---|---|---|---|---|
| 1 | 0.5 | -1.2 | 0.7 | $0.7\times(-1.2)+0.3\times0.5=-0.69$ | $-1.7$ |
| 2 | -0.3 | 0.8 | 0.3 | $0.3\times0.8+0.7\times(-0.3)=0.03$ | $1.1$ |
| 3 | 0.9 | -0.5 | 0.5 | $0.5\times(-0.5)+0.5\times0.9=0.2$ | $-1.4$ |

模型输入 $(x_t, t)$ 输出预测 $\hat v$，loss = 预测与上表"监督目标"的 MSE。注意同一条样本在不同 epoch 会换 $\epsilon, t$ 重新抽——模型被迫学会"任意中间态该往哪推"，而不是背样本。

> 面试一句话总结：**训练用条件流匹配目标：随机抽 (动作, 噪声, 时刻) 构造中间态 $x_t$，用纯 MSE 让速度网络拟合这条确定性路径的真实速度 $\epsilon-a$——条件化的最优解等于边缘速度场，模型无需显式计算不可解的边缘分布。**

---

## 4. 采样（推理）：确定性 ODE 积分——以及它为什么"没有概率密度"

### 1. 现有问题

模型学好了速度场 $v_\theta$，推理时怎么从噪声得到动作？用**数值积分**一步步走。但走完之后发现一个问题：这条路是"确定的"，RL 没法用。

### 2. 方法论（确定性 ODE 的离散化）

把生成过程写成常微分方程（ODE）——"位置 $x$ 对时间 $t$ 的变化率等于速度场"：
$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$

SmolVLA 从 $t=1$（噪声）走到 $t=0$（动作），时间**减小**，用 Euler 法离散（步长 $h = 1/N$，$N$ 为去噪步数，如 $10$）：
$$x_{t-h} = x_t - h \cdot v_\theta(x_t, t), \qquad t = 1, \; 1-h, \; 1-2h, \dots$$

**为什么叫"确定性"**：给定起始噪声 $\epsilon$，每一步 $x_{t-h}$ 都由 $x_t$ 和 $v_\theta$ **唯一确定**——没有掷骰子。所以"从 $\epsilon$ 到 $a$"是一个**确定性映射**（同一个 $\epsilon$ 永远生成同一个 $a$）。

**为什么这对 RL 是致命的**：RL（PPO/GRPO）需要**逐转移的概率** $\log p(x_{t-h} | x_t)$ 来算 importance ratio。但确定性转移的概率是 **delta 分布**（"$x_{t-h}$ 只能是那一个值，其他值概率为 0"），delta 分布的 log-density 是 $-\infty$ / 无定义——**没有密度，算不出 ratio，梯度为零**。这就是"流匹配 VLA 生成很好却做不了在线 RL"的核心矛盾。

### 3. 具体数值样例（手算 10 步去噪）

假设一个 1 维 toy 速度场 $v_\theta(x,t) = x + 0.4$（瞎编的，只为演示），$N=10$，$h=0.1$，起点 $x_1 = 0.3$：

| 步 | $t$ | $x_t$ | $v_\theta$ | $x_{t-h} = x_t - 0.1\times v$ |
|---|---|---|---|---|
| 1 | 1.0 | 0.3 | 0.7 | $0.3 - 0.07 = 0.23$ |
| 2 | 0.9 | 0.23 | 0.63 | $0.23 - 0.063 = 0.167$ |
| 3 | 0.8 | 0.167 | 0.567 | $0.167 - 0.0567 = 0.1103$ |
| ... | ... | ... | ... | ... |
| 10 | 0.1 | $\approx 0.05$ | 0.45 | $0.05 - 0.045 = 0.005 \approx$ 动作 |

关键观察：**每一步 $x_{t-h}$ 是"算出来"的，不是"采出来"的**——同一张表每次跑都一样。要算 $\log p(x_{0.9} = 0.23 \mid x_1 = 0.3)$？在这个确定性过程中 $x_{0.9}$ **只能是 0.23**，取别的值概率为 0 → 密度无定义。

> 面试一句话总结：**推理 = 确定性 ODE 数值积分：从噪声 $x_1$ 出发按 $x_{t-h}=x_t-h\,v_\theta$ 走 $N$ 步到动作 $x_0$；因为每一步唯一确定，转移是 delta 分布、没有 log-density，RL 的 importance ratio 无从算起——这是"把 ODE 改成 SDE"的直接动机。**

---

# 三、从 ODE 到边缘保持 SDE：核心推导

## 5. 为什么"随便加点噪声"不行——边缘分布会被破坏

### 1. 现有问题

RL 需要密度，密度来自随机性。最朴素的想法：把确定性步 $x_{t-h} = x_t - h v$ 改成加噪声的步 $x_{t-h} = x_t - h v + \mathcal{N}(0, \sigma^2)$——这样就有密度了。**但这样改会破坏生成质量**，因为每个中间时刻的"边缘分布"变了。

### 2. 方法论（关键概念：边缘分布 $p_t$ 与"分布怎么演化"）

**什么是边缘分布 $p_t$**：想象训练数据里所有样本 $a$ 各配一个随机噪声 $\epsilon$，在时刻 $t$ 得到一堆"半成品" $x_t = t\epsilon + (1-t)a$。这些半成品的分布就是 $p_t(x)$。$t=1$ 时 $p_1 = \mathcal{N}(0,I)$（纯噪声），$t=0$ 时 $p_0 = p_{\text{data}}$（数据分布），中间是"噪声与数据的混合"。

**确定性 ODE 如何搬运分布**：ODE $dx/dt = v$ 把 $p_1$ 精确搬运成 $p_0$——中间每个 $p_t$ 由 ODE 唯一决定（密度跟随速度场"流动"，满足**连续性方程 / Liouville 方程**）：
$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \, v) = 0$$
（直观：粒子数守恒，$p_t$ 的变化 = 速度场造成的"流进流出"。）

**加了无脑扩散会怎样**：若在每一步加独立高斯噪声（等价于 SDE $dx = v\,dt + g\,dW$），分布的演化会多一项**扩散项**（Fokker-Planck 方程）：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t \, v) \;+\; \tfrac{1}{2} g(t)^2 \, \Delta p_t$$
多出来的 $\tfrac12 g^2 \Delta p_t$（$\Delta$ 是 Laplacian）就是"噪声把分布抹平/变宽"的项——**边缘分布 $p_t$ 不再等于 ODE 版本的 $p_t$**，生成时中间态整体"发虚"，动作质量退化。

### 3. 具体数值样例（一维抹平示意）

假设 $t=0.5$ 时 ODE 版本边缘是双峰（两个动作 $a=0.5$ 和 $a=-0.3$ 各一半概率）：$p_{0.5} = 0.5\delta(x-0.2) + 0.5\delta(x+0.55)$（粗略示意）。加噪后每个峰被高斯模糊成 $0.5\mathcal{N}(x-0.2,\sigma^2)+0.5\mathcal{N}(x+0.55,\sigma^2)$——**峰变矮变宽，甚至中间谷被填平**。采样时模型以为自己还在原分布上工作，实际分布已偏 → 生成偏离。

> 面试一句话总结：**边缘分布 $p_t$ 是"噪声+数据在时刻 t 的混合分布"，由 ODE 的速度场唯一搬运；直接加噪会在 Fokker-Planck 里多出扩散项 $\frac12 g^2\Delta p_t$，把 $p_t$ 抹平——所以"要随机性但不能破坏边缘"必须让扩散项被一个反向的 drift 修正精确抵消，这就是边缘保持 SDE。**

---

## 6. 边缘保持 SDE 的构造：加扩散 + score 修正 drift

### 1. 现有问题

要随机性（密度）→ 加扩散 $g(t)dW$；但扩散破坏边缘 → 必须同时改 drift。改多少？答案藏在 Fokker-Planck 方程里。

### 2. 方法论（核心推导：drift 修正 = 加 $\tfrac12 g^2 \nabla\log p_t$）

**目标**：找一个随机过程，它的边缘分布 $p_t$ 与确定性 ODE **完全相同**，但转移有随机性。

**推导**（只改 drift 为 $v + \text{修正项}$，加扩散 $g\,dW$）：
$$dx = \big(v(x,t) + c(x,t)\big)\,dt + g(t)\,dW$$

它的密度满足 Fokker-Planck：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot\big(p_t (v + c)\big) + \tfrac12 g^2 \Delta p_t = \underbrace{-\nabla\cdot(p_t v)}_{\text{ODE 项}} \underbrace{-\nabla\cdot(p_t c) + \tfrac12 g^2 \Delta p_t}_{\text{希望它}=0}$$

要让边缘保持（= 只留 ODE 项），需要：$-\nabla\cdot(p_t c) + \tfrac12 g^2 \Delta p_t = 0$，即
$$\nabla\cdot(p_t c) = \tfrac12 g^2 \Delta p_t$$

**猜一个 $c$**：令 $c = \tfrac12 g^2 \,\nabla\log p_t$（score 的倍数），验证：
$$\nabla\cdot(p_t \cdot \tfrac12 g^2 \nabla\log p_t) = \tfrac12 g^2\, \nabla\cdot\big(p_t \nabla\log p_t\big) = \tfrac12 g^2\, \nabla\cdot(\nabla p_t) = \tfrac12 g^2 \Delta p_t$$
（用恒等式 $p_t \nabla\log p_t = \nabla p_t$，因为 $\nabla\log p_t = \nabla p_t / p_t$。）——**两边相等，扩散项被精确抵消**。

**结论（正向时间 $t$ 增大的写法）**：
$$dx = \big(v(x,t) + \tfrac12 g(t)^2 \underbrace{\nabla\log p_t(x)}_{=\;s(x,t),\,\text{score}}\big)\,dt + g(t)\,dW$$
- 多出来的 drift 修正 $\tfrac12 g^2 s$ 把粒子**拉向高概率区**（score 指向概率上升方向）；
- 扩散 $g\,dW$ 把粒子**推散**；两者在 Fokker-Planck 里**精确抵消** → $p_t$ 与 ODE 完全一致 → **边缘保持**。

**SmolVLA 是去噪方向（$t$ 减小），drift 修正项变号**（反向时间下 $\nabla\log p$ 的贡献反号）：
$$x_{t-h} = x_t - h\Big(v - \tfrac12 g(t)^2 s\Big) + g(t)\sqrt{h}\,\xi, \qquad \xi \sim \mathcal{N}(0, I)$$
这正是 `sde.py` 的 `marginal_preserving_transition`：

```python
# mean = x_t - h * (v - 0.5*g(t)^2*score)  and  std = g(t)*sqrt(h)
score = score_from_velocity(x_t, velocity, time)
g_t = diffusion_scale(time, eta, x_t)
reverse_drift = velocity.float() - 0.5 * g_t.square() * score
mean = x_t.float() - float(step_size) * reverse_drift
std = g_t * math.sqrt(step_size)
```

### 3. 具体数值样例（一维验证"扩散项被修正抵消"）

1 维、某时刻 $t$：假设边缘 $p_t(x) = \mathcal{N}(x; \mu=0, \sigma^2=1)$（为手算方便假设成标准高斯），则 $\nabla\log p_t(x) = -x$（高斯 score 就是 $-x$），取 $g=0.05$：
- 扩散项贡献：$\tfrac12 g^2 \Delta p_t$（正的抹平）；
- drift 修正项：$\nabla\cdot(p_t \cdot \tfrac12 g^2 s) = \tfrac12 g^2 \Delta p_t$（完全一样）→ 相减为 0 ✓；
- 实际例子：$g^2 = 0.0025$，修正 drift $= \tfrac12\times0.0025\times(-x) = -0.00125x$——对 $x=0.3$ 是 $-0.000375$，很小但**恰好等于扩散在该点的反向补偿**，所以长期演化边缘不变。

> 面试一句话总结：**边缘保持 SDE = 在 ODE 的速度 $v$ 上加"扩散 $g\,dW$ + drift 修正 $\pm\frac12g^2s$"（$s=\nabla\log p_t$ 是 score）：Fokker-Planck 里扩散项被修正项精确抵消（用恒等式 $p_t\nabla\log p_t=\nabla p_t$ 验证），边缘分布 $p_t$ 与确定性 ODE 完全相同——随机性有了，质量不降。**

---

## 7. score 从哪来：线性流匹配的闭式解（含完整推导）

### 1. 现有问题

上一节的修正 drift 需要 score $s(x_t,t) = \nabla_{x}\log p_t(x_t)$——但 $p_t$ 是"数据分布的混合"，我们不是刚说它写不出闭式吗？怎么又要算 score？

### 2. 方法论（线性路径 + Tweedie 恒等式 → 闭式 score）

**关键**：虽然 $p_t$ 整体没有闭式，但**条件分布 $p(x_t | a)$ 是高斯**（因为 $x_t = t\epsilon + (1-t)a$，$\epsilon$ 高斯）：
$$p(x_t \mid a) = \mathcal{N}\big(x_t;\, (1-t)a,\; t^2 I\big)$$

**Tweedie 恒等式**（对指数族分布，条件 score = 条件期望的 score）：对 $p_t(x) = \mathbb{E}_a[p(x|a)]$，
$$\nabla_x \log p_t(x) = \mathbb{E}_{a}\big[\nabla_x \log p(x \mid a) \;\big|\; x\big]$$
（直观：混合分布的 score = 各成分 score 在"后验权重"下的平均——对高斯混合族成立。）

逐项算：$\nabla_x \log p(x_t|a) = -\big(x_t - (1-t)a\big)/t^2$，所以
$$s(x_t,t) = -\frac{x_t - (1-t)\,\mathbb{E}[a \mid x_t]}{t^2}$$

**代入 $a = x_t - t\,v$**（第 2 点推的代数关系，把 $\mathbb{E}[a|x_t]$ 用模型速度 $v_\theta$ 近似）：
$$s = -\frac{x_t - (1-t)(x_t - t\,v)}{t^2} = -\frac{t\,x_t + (1-t)t\,v}{t^2} = -\frac{x_t + (1-t)v}{t}$$

**这就是 `sde.py` 的闭式 score**：
```python
# score_from_velocity: score = -(x_t + (1-t) * v) / t
return -(x_f + (1.0 - time_b) * velocity_f) / time_b
```

**合理性检查（两个端点）**：
- $t=1$（纯噪声）：$s = -x_1/1 = -x_1$。而 $p_1 = \mathcal{N}(0,I)$ 的 score 确实是 $-x$ ✓；
- $t \to 0$（逼近数据）：$s = -(x_t + v)/t \to \infty$——因为 $p_t$ 收缩成数据的 delta 分布，score 发散，合理（代码要求 $t \in (0,1]$）。

**score 在项目 forward 里到底怎么算（完整计算链，贴合 `sde.py`）**——强调三点：score **不需要额外网络**、**不需要算梯度**、**只是速度 $v$ 的闭式函数**：

```text
输入：一个去噪步的 (x_t, t)
  ① model.denoise_step(x_t, t) ──► v_θ(x_t, t)          ← 唯一一次神经网络前向
  ② score_from_velocity(x_t, v_θ, t)                     ← 纯公式，无网络
        s = -(x_t + (1-t)·v_θ) / t                       ← 一行代公式
  ③ diffusion_scale(t, eta) → g = eta·√t
  ④ reverse_drift = v_θ − ½·g²·s                        ← score 修正 drift
  ⑤ mean = x_t − h·reverse_drift；std = g·√h
输出：p(x_{t-h}|x_t) = N(mean, std²)   （采样/求 logp 用）
```

- **为什么 score 不用学**：Tweedie 把"混合分布 score"归约为"各高斯成分 score 的后验平均"，而线性路径下这个平均恰好坍缩成只用 $v$ 的闭式——**score 的信息完全来自速度网络 $v_\theta$**（$v$ 是模型"看到 $x_t$ 该往哪推"的浓缩）；
- **为什么不用梯度**：$\nabla_x\log p_t$ 虽然写作"对 $x$ 求导"，但推导后它是 $x_t, v, t$ 的**代数表达式**，前向一次就算出，不需要反向传播（$\nabla_x\log p$ 里的"梯度"是数学记号，不是要 backprop 的计算图）；
- **float32**：公式里全 `.float()`（bf16 会引入漂移，破坏采集/重打分一致性）。

### 3. 具体数值样例（完整手算 score + 转移）

延续第 4 点的 toy：$t=1.0$，$x_t=0.3$，$v=0.7$，取 $\eta=0.05$（$g(t)=\eta\sqrt{t}$）、$h=0.1$：

1. **score**：$s = -(x_t + (1-t)v)/t = -(0.3 + 0\times0.7)/1 = -0.3$；
2. **扩散尺度**：$g(1) = \eta\sqrt{1} = 0.05$，$g^2 = 0.0025$；
3. **反向 drift（修正后）**：$v - \tfrac12 g^2 s = 0.7 - 0.5\times0.0025\times(-0.3) = 0.7 + 0.000375 = 0.700375$；
4. **转移均值**：$\mu = x_t - h\times\text{drift} = 0.3 - 0.1\times0.700375 = 0.22996$；
5. **标准差**：$\sigma = g\sqrt{h} = 0.05\times\sqrt{0.1} = 0.01581$；
6. 于是 $p(x_{0.9}\mid x_1 = 0.3) = \mathcal{N}(0.22996,\, 0.01581^2)$——对比 ODE 版的确定性 $0.23$：**均值几乎相同（$0.22996$ vs $0.23$，差 $3.8\times10^{-5}$），但现在是高斯分布、有密度了**；
7. 若采样得 $x_{0.9}=0.23$，其 log-density：
$$-\tfrac12\big(\tfrac{0.23-0.22996}{0.01581}\big)^2 - \log 0.01581 - \tfrac12\log 2\pi \approx -3\times10^{-6} + 4.147 - 0.919 \approx 3.23\ \text{nats}$$

> 面试一句话总结：**score 不用学也不用积分：因为条件分布 $p(x_t|a)$ 是高斯的，用 Tweedie 恒等式 + 代数关系 $a = x_t - tv$ 推出闭式 $s = -(x_t+(1-t)v)/t$——纯噪声端 $s=-x$、数据端发散，两个端点都对；代入修正 drift 后每一步转移变成"均值≈ODE 步、有高斯宽度"的可采分布。**

---

## 8. 反向转移与闭式 log-density：RL 需要的东西齐了

### 1. 现有问题

第 6-7 点给了"怎么采"（SDE 一步），RL 还需要"这条轨迹的对数概率"（重打分）。反向转移是高斯 → log-density 有闭式。

### 2. 方法论

一步反向转移（SmolVLA 从 $t$ 到 $t-h$）：
$$p(x_{t-h} \mid x_t) = \mathcal{N}\Big(x_{t-h};\, \underbrace{x_t - h\big(v - \tfrac12 g(t)^2 s\big)}_{\mu},\; \underbrace{g(t)^2 h}_{\sigma^2} I\Big)$$

对角高斯 log-density（$D$ 维，逐元素求再求和）：
$$\log p(x_{t-h}\mid x_t) = -\tfrac12\left(\frac{x_{t-h}-\mu}{\sigma}\right)^2 - \log\sigma - \tfrac12\log(2\pi)$$

对应 `sde.py` 的 `gaussian_log_prob`：
```python
elementwise = -0.5 * (((sample_f.detach() - mean_f) / std_f).square()
                      + 2.0 * torch.log(std_f) + math.log(2.0 * math.pi))
```

**一条轨迹的总 log-prob** = 各去噪步 log-prob 之和（每步只条件于上一步，Markov 链）：
$$\log p(\text{trajectory}) = \sum_{\text{denoise step}} \log p(x_{t-h} \mid x_t)$$

**重打分（rescoring）**：训练时换最新权重 $v_\theta$ 重新前向算每步的速度 → 得到新 $\mu,\sigma$ → 用**固定的采样状态** $x_{t-h}$（采集时留下的）重算 log-prob → 新旧 log-prob 之差就是 importance ratio 的来源：
$$\text{ratio} = \exp\big(\log p_{\text{new}} - \log p_{\text{old}}\big)$$

> 面试一句话总结：**反向转移 $p(x_{t-h}|x_t)=\mathcal{N}(\mu, g^2h)$ 是高斯的，log-density 有闭式（逐元素 $-0.5[(x-\mu)/\sigma]^2 - \log\sigma - \frac12\log2\pi$）；一条轨迹的 log-prob 是各去噪步之和（Markov 链），训练时固定采样状态、用新权重重算速度即可得新 log-prob → importance ratio → 这就是 flow-matching 策略梯度的全部基础。**

---

# 四、从论文到代码（对应项目实现）

## 9. sde.py 四个函数 ↔ 公式一一对应

### 1. 现有问题

公式懂了，代码怎么对应？项目把整套 SDE 逻辑收敛在 `sde.py` 四个纯函数里，逐一对上。

### 2. 方法论（函数 ↔ 公式映射表）

| sde.py 函数 | 对应公式 | 一句话 |
|---|---|---|
| `score_from_velocity(x_t, v, t)` | $s = -(x_t+(1-t)v)/t$ | 第 7 点闭式 score（Tweedie） |
| `diffusion_scale(t, eta, ref)` | $g(t) = \eta\sqrt{t}$ | 噪声调度：越靠近数据（$t\to0$）噪声越小 |
| `marginal_preserving_transition(...)` | $\mu = x_t - h(v-\tfrac12g^2s),\ \sigma=g\sqrt{h}$ | 一步反向转移（第 6 点） |
| `sample_transition(transition)` | $\mu + \sigma\xi,\ \xi\sim\mathcal{N}(0,I)$，结果 `.detach()` | 采样（不保留图；密度在 log_prob 里单独评） |
| `gaussian_log_prob(sample, μ, σ, mask)` | 第 8 点公式 | 闭式 log-density，mask 挡掉 pad/无效位 |

**实现细节（容易踩坑）**：
- **全部 `.float()`**：概率计算用 float32（bf16 高斯似然会漂移/溢出）——这是"采集与重打分数值一致性"的第 1 条铁律；
- **`eta`（代码里的 SDE 噪声强度）就是公式里的 $\eta$**：$g(t)=\eta\sqrt t$。$\eta=0$ 时 $\sigma=0$ → `gaussian_log_prob` 报错（"sampling with log-prob requires eta > 0"）——纯 ODE 没密度，代码层面强制；
- **`eta` 是超参**：本项目训练用 $\eta=0.05$（小噪声，动作质量接近 ODE、又有密度），与评估的确定性 ODE 口径尽量一致；
- **`mask` 参数**：SmolVLA 的 padded 动作维、未执行动作位必须挡在 ratio 外（`torch.where(valid, elementwise, 0)`）。

### 3. 具体数值样例（eta 的效应）

同一转移（$x_t=0.3, v=0.7, t=1, h=0.1$）：
| $\eta$ | $\sigma = \eta\sqrt{t h}$ | 转移形态 |
|---|---|---|
| 0 | 0 | delta（无密度，`gaussian_log_prob` 报错）|
| 0.05（本项目） | 0.0158 | 窄高斯，质量≈ODE、有密度 |
| 0.5 | 0.158 | 宽高斯，质量明显退化 |
| 1.0 | 0.316 | 几乎随机跳，动作全毁 |

> 面试一句话总结：**sde.py 四个纯函数把整套 SDE 公式落到代码：score/diffusion_scale/转移/log_prob 一一对应，全程 float32、eta>0 强制（eta=0 就是无密度的 ODE）、mask 挡无效位——eta=0.05 是本项目"质量与密度"的平衡点。**

---

## 10. 采样与重打分：sample_sde_chunk / recompute_log_probs（为什么要"同一条数值路径"）

### 1. 现有问题

RL 的 ratio 从 1 起步（新旧策略初始一致）。如果"采集时算 log-prob"和"训练时重打分算 log-prob"走了不同的数值路径（bf16 vs fp32、批处理 vs 逐条、图像半精度存储），两者会系统性偏差——实测 bf16/fp32 可差约 $1.07$ nats，ratio $\approx e^{-1.07}=0.34$，直接毁训练。

### 2. 方法论（两阶段必须逐位一致）

**采集（`sample_sde_chunk`，no_grad）**：给定观测，`prepare_policy_prefix` 建 prefix KV cache → 从噪声 $x_1$ 出发，对每个去噪步：算速度 → `marginal_preserving_transition` → `sample_transition` 采 $x_{t-h}$ → 记录 `element_log_probs`（`gaussian_log_prob` reduction="none"）→ 存下**全部采样状态** $x_t$（不是只存动作！）。

**重打分（`recompute_log_probs`，可微）**：训练时用最新权重，对**固定的采样状态序列** $x_1, x_{0.9}, \dots$ 重算每步速度 → 重算转移参数 → 用**同一个 $x_{t-h}$** 评 log-prob：
```python
# recompute_log_probs：state 固定，只有 velocity 由新权重给出
velocity = model.denoise_step(x_t=states[:, step], past_key_values=prefix_cache, ...)
transition = marginal_preserving_transition(states[:, step], velocity, time, step_size, trajectory.eta)
log_probs.append(gaussian_log_prob(states[:, step + 1], transition.mean, transition.std, mask=mask))
```

**一致性三铁律**（见 `SmolVLA-VERL.md` 第 4 点的完整版）：
1. 采集与重打分共用同一 autocast（bf16 on CUDA / fp32 on CPU，`rollout_autocast`）；
2. 逐 chunk 重打分（批处理会让 mask/valid_positions 错位）；
3. 图像 float32 存储 + 首 chunk 守卫（重打分偏差 >0.05 nats 即报错）。

### 3. 具体数值样例（为什么"只存动作"不够）

一条 trajectory 若只存最终动作 $a=x_0$，重打分时无法重建中间态 $x_{0.9}, x_{0.8}, \dots$（去噪路径依赖每步噪声 $\xi$）——所以 `SmolVLATrajectory` 必须存 `states`（$11$ 个时刻：噪声 + 10 步）与 `element_log_probs`（$10$ 步 $\times$ $10\times7$ 维）：
```python
states: Tensor              # (num_steps+1, chunk_size, max_action_dim) = (11, 10, 7)
element_log_probs: Tensor   # (num_steps, chunk_size, max_action_dim) = (10, 10, 7)
```
一个 chunk 就是 $11\times10\times7=770$ 个状态标量 + $700$ 个 log-prob 标量——**这些是"密度承载体"，是重打分唯一需要的输入**。

> 面试一句话总结：**采集用 SDE 采样并留下全部去噪状态与逐元素 log-prob（SmolVLATrajectory），训练时固定这些状态、用新权重重算每步高斯参数再评 log-prob——重打分必须与采集逐位同数值路径（同 autocast/逐 chunk/图像 float32），否则 ratio 从 1 漂走、训练静默退化。**

---

# 五、完整样例：一个 token 从噪声到动作（逐步演算 + 逐行对照 sde.py + 反向传播）

> 本部分是"越详细越好"的落地：**把真实流程（`num_steps=10`、chunk $10\times7$）压缩成"1 维 × 4 步"便于手算**（$h=1/4=0.25$，$\eta=0.2$ 取大一点让噪声可见；项目真实 $\eta=0.05$）。模拟设定：toy 数据动作 $a=0.5$、起始噪声 $x_1=\epsilon=0.8$、速度网络已学好（$v_\theta=0.3$ 恒定——因线性路径 $v=\epsilon-a$ 恒定）；每步采样噪声 $\xi$ 为给定模拟值。**所有数字按 `sde.py` 公式算出并经脚本校验自洽**；除维度外与项目代码逐行一致。

## 11. 样例设定与"每步到底发生什么"（逐行对照 `sample_sde_chunk`）

项目 `sample_sde_chunk` 的去噪主循环（`sde_sampling.py`）：

```python
step_size = 1.0 / model.config.num_steps          # 真实 1/10=0.1；本例 num_steps=4 → 0.25
state = noise.detach().float()                     # x_1 = ε = 0.8
for step in range(model.config.num_steps):         # 4 步
    time = 1.0 - step * step_size                  # t = 1.0, 0.75, 0.5, 0.25
    velocity = model.denoise_step(x_t=state, ...)  # ① 网络前向 → v_θ(x_t, t)（本例恒 0.3）
    transition = marginal_preserving_transition(state, velocity, time, step_size, eta)
    #   ②③④⑤：score → g → drift → μ, σ（sde.py 三行）
    next_state = sample_transition(transition)     # ⑥ 采样 x_{t-h} = μ + σ·ξ
    element_log_probs.append(gaussian_log_prob(...))  # ⑦ 记录这步 logp
    states.append(next_state); state = next_state
```

**一个去噪步的内部 7 步**（对照 `sde.py` 的 `marginal_preserving_transition` + `gaussian_log_prob`）：

```python
score = score_from_velocity(x_t, velocity, time)      # ② s = -(x_t+(1-t)v)/t
g_t = diffusion_scale(time, eta, x_t)                 # ③ g = η·√t
reverse_drift = velocity - 0.5 * g_t^2 * score        # ④ drift = v − ½g²s
mean  = x_t - step_size * reverse_drift               # ⑤ μ = x_t − h·drift
std   = g_t * sqrt(step_size)                         # ⑤ σ = g·√h
# sample_transition: x_{t-h} = μ + σ·ξ（ξ~N(0,1)，结果 .detach()）
# gaussian_log_prob:  -0.5[(x-μ)/σ]² − logσ − 0.5·log(2π)
```

## 12. 完整采样表：4 步全中间量逐步演算

参数：$x_1=0.8$，$v=0.3$（恒定），$\eta=0.2$，$h=0.25$，噪声 $\xi=[0.6,-0.5,1.2,-0.3]$。

| step | $t$ | $x_t$ | $v_\theta$ | score $s$ | $g=\eta\sqrt t$ | drift | $\mu$ | $\sigma$ | $\xi$ | $x_{t-h}$ | logp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.8000 | 0.3 | -0.8000 | 0.2000 | 0.3160 | 0.7210 | 0.1000 | 0.6 | 0.7810 | 1.2036 |
| 1 | 0.75 | 0.7810 | 0.3 | -1.1413 | 0.1732 | 0.3171 | 0.7017 | 0.0866 | -0.5 | 0.6584 | 1.4025 |
| 2 | 0.50 | 0.6584 | 0.3 | -1.6168 | 0.1414 | 0.3162 | 0.5794 | 0.0707 | 1.2 | 0.6642 | 1.0102 |
| 3 | 0.25 | 0.6642 | 0.3 | -3.5569 | 0.1000 | 0.3178 | 0.5848 | 0.0500 | -0.3 | 0.5698 | 2.0318 |

**逐步演算（step 1，$t=0.75$，完整展开）**：
1. 模型前向：$v_\theta(0.7810, 0.75) = 0.3$（本例恒值）；
2. score：$s = -(x_t+(1-t)v)/t = -(0.7810 + 0.25\times0.3)/0.75 = -0.8560/0.75 = -1.1413$；
3. 扩散尺度：$g = \eta\sqrt{t} = 0.2\times\sqrt{0.75} = 0.1732$，$g^2 = 0.03$；
4. 修正 drift：$v - \tfrac12 g^2 s = 0.3 - 0.5\times0.03\times(-1.1413) = 0.3 + 0.0171 = 0.3171$（比纯 $v$ 多 $+0.0171$，即 score 修正项）；
5. 转移均值：$\mu = x_t - h\times\text{drift} = 0.7810 - 0.25\times0.3171 = 0.7017$；
6. 标准差：$\sigma = g\sqrt{h} = 0.1732\times0.5 = 0.0866$；
7. 采样：$x_{0.5} = \mu + \sigma\xi = 0.7017 + 0.0866\times(-0.5) = 0.6584$；
8. logp：$\log\mathcal{N}(0.6584; 0.7017, 0.0866^2) = -0.5\big(\frac{0.6584-0.7017}{0.0866}\big)^2 - \log 0.0866 - 0.919 = 1.4025$。

**关键观察**：
- **score 的数值演化**：$t=1$ 时 $s=-x_1=-0.8$（纯高斯 score），越靠近数据 $s$ 越负越大（$-3.56$）——因为 $p_t$ 越窄、概率山脊越陡（$t\to0$ 发散）；
- **$\mu$ 贴着 ODE 走**：ODE 轨迹（每步 $-h\cdot0.3=-0.075$）为 $0.725\to0.650\to0.575\to0.500$；SDE 的 $\mu$ 为 $0.721\to0.702\to0.579\to0.585$——**每步均值都几乎等于 ODE 步**，这就是"边缘保持"在单条路径上的体现（随机性只在 $\sigma\xi$ 上）；
- **最终动作**：$x_0 = 0.5698$ vs ODE 的 $0.5000$——本例 $\eta=0.2$ 噪声偏大，累积偏移 $0.07$；项目 $\eta=0.05$ 时 $\sigma$ 小 4 倍，$x_0$ 与 ODE 几乎重合；
- **每步 σ 递减**（0.1→0.0866→0.0707→0.05）：$g(t)=\eta\sqrt t$ 随 $t\to0$ 收缩——越接近动作越确定，保证最终输出质量。

## 13. log-prob、轨迹总 logp、重打分与 ratio

- **每步 logp** 见上表；轨迹总 logp = 各步之和（Markov 链）：
$$\log p(\text{trajectory}) = 1.2036 + 1.4025 + 1.0102 + 2.0318 = 5.6481\ \text{nats}$$
- **重打分（RL 训练时）**：固定采集的 states $(0.8,\,0.7810,\,0.6584,\,0.6642,\,0.5698)$，只把某一步的 $v_\theta$ 换成新权重（模拟"策略更新了一点"）。例：**只有 step 2 的 $v$ 从 0.3 变成 0.28**（其余步不变）：
  - 重算 step 2：$t=0.5$，$x_t=0.6584$，$v'=0.28$ → $s'=-(0.6584+0.5\times0.28)/0.5=-1.5968$ → $\text{drift}'=0.28-0.5\times0.02\times(-1.5968)=0.29597$ → $\mu'=0.6584-0.25\times0.29597=0.5844$（原 $\mu=0.5794$）；
  - 新 logp：$\log\mathcal{N}(0.6642; 0.5844, 0.0707^2) = 1.0936$（原 1.0102，$\Delta=+0.0834$）；
  - 轨迹总 logp 变为 $5.7315$；**ratio $=\exp(5.7315-5.6481)=e^{0.0834}=1.087$**——这就是 GRPO 里"新策略相对旧策略这条轨迹被高估了 8.7%"的含义；
- **直觉**：$v$ 减小 → $\mu$ 减小 → 更接近实际采样值 $0.6642$ → 该状态在新策略下"更可能" → logp 上升 → ratio>1 → GRPO 提高这条轨迹的权重。

## 14. 反向传播：梯度如何从 logp 穿回速度网络

**两条梯度路径**：

**① CFM 训练（速度回归）**——简单直接：
$$\mathcal{L}_{\text{CFM}}=\tfrac12\big\|v_\theta(x_t,t)-(\epsilon-a)\big\|^2 \quad\Rightarrow\quad \frac{\partial L}{\partial\theta} = \big(v_\theta - (\epsilon-a)\big)\cdot\frac{\partial v_\theta(x_t,t)}{\partial\theta}$$
梯度 = 速度残差 × 网络 BP——标准监督学习。

**② RL 重打分（logp 对策略参数）**——链式穿过"去噪转移"：
$$\frac{\partial \log p_\theta}{\partial\theta}=\sum_t \frac{\partial\log\mathcal{N}(x_{t-h};\mu_t,\sigma_t^2)}{\partial\mu_t}\cdot\frac{\partial\mu_t}{\partial v_t}\cdot\frac{\partial v_\theta(x_t,t_t)}{\partial\theta}$$

逐项（$x_t$ 是**固定状态**，detach 不参与梯度，所以每步独立、无跨步二阶链）：
- $\dfrac{\partial\log\mathcal{N}}{\partial\mu_t}=\dfrac{x_{t-h}-\mu_t}{\sigma_t^2}$（高斯 logp 对均值）；
- $\dfrac{\partial\mu_t}{\partial v_t}=-h\Big(1+\tfrac12 g_t^2\,\tfrac{1-t_t}{t_t}\Big)$（因为 $\mu_t=x_t-h(v-\tfrac12g^2s)$ 且 $s=-(x+(1-t)v)/t$，score 项也依赖 $v$）；
- $\dfrac{\partial v_\theta(x_t,t_t)}{\partial\theta}$：速度网络自己的 BP（$x_t$ 固定 → 只穿网络层）。

**完整数值（本例 4 步的 $w_t = \partial\log p_t/\partial v_t$，脚本校验）**：

| step | $t$ | $x_{t-h}$ | $\mu_t$ | $\sigma_t$ | $\partial\log N/\partial\mu$ | $\partial\mu/\partial v$ | $w_t=\partial\log p_t/\partial v$ |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7810 | 0.7210 | 0.1000 | $6.000$ | $-0.25000$ | $-1.5000$ |
| 1 | 0.75 | 0.6584 | 0.7017 | 0.0866 | $-5.774$ | $-0.25125$ | $+1.4506$ |
| 2 | 0.50 | 0.6642 | 0.5794 | 0.0707 | $16.971$ | $-0.25250$ | $-4.2851$ |
| 3 | 0.25 | 0.5698 | 0.5848 | 0.0500 | $-6.000$ | $-0.25375$ | $+1.5225$ |

- **$w_t$ 的含义**：该步 logp 对"速度"的敏感度（$\Sigma=-2.812$）；最终梯度 $= \sum_t w_t \cdot \partial v_\theta(x_t,t_t)/\partial\theta$——**$|w_t|$ 大的步（step 2）主导训练**（它离 $\mu$ 最远、$\sigma$ 小，logp 对 $\mu$ 最敏感）；
- **为什么 $\partial\mu/\partial v$ 总在 $-0.25$ 附近**：$\mu=x_t-hv+\tfrac12 hg^2s$，第一项贡献 $-h=-0.25$，score 项因 $g^2$ 很小只微调（$<0.4\%$）——**速度每 +0.1，均值约 −0.025**；
- **符号直觉**：step 1 的 $x_{t-h}<\mu$（采样值在均值左边）→ 增大 $\mu$ 会远离采样值 → logp 降 → 要增大 logp 应减小 $\mu$ → 应减小 $v$（$\partial\mu/\partial v<0$）→ $w_t>0$（step 1/3）；step 0/2 相反（采样值在右边）→ $w_t<0$；
- **项目里梯度流向**：verl GRPO 的 policy loss（用 `flow_log_prob` 重打分得到 logp）→ `loss.backward()` 只穿过 **velocity 网络（action expert）** 与"$\mu\to$logp"这条闭式链——prefix cache 与采样状态都 detach，梯度不穿 VLM、不穿采样噪声。

> **样例一句话总结**：一个 token 的 SDE 生成 = 4（真实 10）步"前向出 $v$ → 闭式 score → 修正 drift → 高斯转移采样 → logp"，每步 $\mu$ 贴着 ODE、$\sigma$ 随 $t$ 收缩；轨迹 logp 是各步之和，重打分时固定状态只换 $v$ 得 ratio（例：step2 $v$ 变 0.28 → logp +0.083 → ratio 1.087）；反向传播 $\partial\log p/\partial\theta=\sum_t w_t\,\partial v_\theta/\partial\theta$，$w_t=\frac{x-\mu}{\sigma^2}\cdot\big[-h(1+\frac12g^2\frac{1-t}{t})\big]$ 就是"每步 logp 对速度的敏感度"，$|w_t|$ 大的步主导梯度——梯度只穿 velocity 网络，状态与 VLM 都冻结。

---

# 六、面试问答与速查

## 15. 高频追问速答

**Q1：ODE 和 SDE 到底差在哪？**
ODE 每步确定（$x_{t-h}=x_t-hv$，无密度）；SDE 每步加高斯噪声（$x_{t-h}\sim\mathcal{N}(x_t-h(v-\frac12g^2s),\,g^2h)$，有密度）。共同点：均值路径几乎一致 → 生成质量相当。

**Q2：为什么加噪声不会破坏生成质量？**
因为 drift 加了修正 $-\frac12g^2s$（去噪方向），Fokker-Planck 里它与扩散项精确抵消 → 边缘分布 $p_t$ 与 ODE 相同 → 模型仍在"正确的分布"上工作。

**Q3：score 是什么？怎么得到？**
score $=\nabla_x\log p_t(x)$（概率密度增长最快的方向）。线性流匹配下有闭式：$s=-(x_t+(1-t)v)/t$（Tweedie + $a=x_t-tv$），不用额外网络。

**Q4：eta 是什么？取多大？**
SDE 噪声强度（$g(t)=\eta\sqrt t$）。$\eta=0$=ODE（无密度，代码报错）；本项目取 $0.05$——足够小保证质量≈ODE，又>0 让每步有密度；评估用确定性 ODE，两者口径一致。

**Q5：为什么 RL 需要 log-prob？**
GRPO/PPO 的 importance ratio $=\exp(\log p_{\text{new}}-\log p_{\text{old}})$，没有每步转移密度就算不出 ratio；SDE 让每步转移是高斯、log-density 闭式。

**Q6：一条轨迹的 log-prob 怎么算？**
各去噪步 log-prob 之和（Markov 链）：$\sum_t \log\mathcal{N}(x_{t-h};\mu_t,\sigma_t^2)$。

**Q7：重打分是什么？**
训练时固定采集的采样状态，用最新权重重算每步速度→转移参数→log-prob；old（采集时）vs new（重打分）之差就是 ratio 来源。

**Q8：为什么必须"同一条数值路径"？**
bf16 vs fp32 差 ~1.07 nats → ratio≈0.34，超出 clip 窗直接毁训练；所以采集/重打分共用 rollout_autocast + 逐 chunk + float32 存图 + 0.05 nats 守卫。

**Q9：与 diffusion（DDPM）的区别？**
flow matching 学"速度"（路径是直线插值，一步可跳到端点），DDPM 学"噪声"（路径是 OU 过程/方差保持，逐层去噪）；flow matching 训练更简单（纯 MSE 回归速度），采样步数可更少（10 步 vs 1000 步）。

**Q10：这套东西在 RL 里的完整角色？**
把"不可训的确定性生成"变成"可训的带密度生成"——SDE 采样给轨迹密度，GRPO 用密度算 ratio 更新 action expert，prefix KV cache + remove-padding 让重打分高效（详见 `SmolVLA-VERL.md`）。

## 16. 面试一句话总结（背诵版）

- **Flow Matching**：学速度场 $v$，把噪声沿直线路径 $x_t=t\epsilon+(1-t)a$ 推到动作，训练=条件流匹配 MSE，推理=ODE 积分；
- **ODE 无密度**：$x_{t-h}=x_t-hv$ 确定性 → delta 分布 → RL 算不了 ratio；
- **边缘保持 SDE**：$x_{t-h}\sim\mathcal{N}\big(x_t-h(v-\tfrac12g^2s),\,g^2h\big)$，drift 修正 $-\tfrac12g^2s$ 让 Fokker-Planck 扩散项精确抵消 → 边缘分布不变、又有闭式 log-density；
- **score 闭式**：$s=-(x_t+(1-t)v)/t$（Tweedie + $a=x_t-tv$）；
- **落地**：sde.py 四函数 + SmolVLATrajectory 存全状态 + 采集/重打分同数值路径（eta=0.05）。

---

# 附：公式速查表

| 概念 | 公式 | 出处 |
|---|---|---|
| 插值路径 | $x_t = t\epsilon + (1-t)a$ | 第 2 点 |
| 速度 | $v = \epsilon - a$，且 $a = x_t - tv$ | 第 2 点 |
| 训练目标 | $\mathcal{L}=\mathbb{E}\left[\big\|\,v_\theta(x_t,t)-(\epsilon-a)\,\big\|^2\right]$ | 第 3 点 |
| ODE 采样 | $x_{t-h}=x_t-hv$ | 第 4 点 |
| Liouville | $\partial_t p_t + \nabla\cdot(p_t v)=0$ | 第 5 点 |
| Fokker-Planck | $\partial_t p_t=-\nabla\cdot(p_t f)+\tfrac12g^2\Delta p_t$ | 第 5、6 点 |
| 边缘保持 drift | $v \pm \tfrac12 g^2\nabla\log p_t$（去噪取 $-$）| 第 6 点 |
| 噪声调度 | $g(t)=\eta\sqrt{t}$ | 第 6/9 点 |
| score（闭式） | $s=-(x_t+(1-t)v)/t$ | 第 7 点 |
| 一步转移 | $p(x_{t-h}|x_t)=\mathcal{N}(x_t-h(v-\tfrac12g^2s),\,g^2h)$ | 第 8 点 |
| 高斯 log-density | $-\tfrac12[(x-\mu)/\sigma]^2-\log\sigma-\tfrac12\log2\pi$（逐元素求和）| 第 8 点 |
| 轨迹 log-prob | $\sum_t \log p(x_{t-h}|x_t)$（Markov 链）| 第 8 点 |
| ratio | $\exp(\log p_{\text{new}}-\log p_{\text{old}})$ | 第 8/10 点 |
