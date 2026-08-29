# GRPO/PPO 强化学习训练完全指南
> **从数学原理到工程实现的完整推导**

---

## 目录
1. [核心概念与符号系统](#1-核心概念与符号系统)
2. [GRPO 算法数学推导](#2-grpo-算法数学推导)
3. [损失函数完整解析](#3-损失函数完整解析)
4. [详细数值样例](#4-详细数值样例)
5. [Reward Model 深度解析](#5-reward-model-深度解析)
6. [Critic Model 深度解析](#6-critic-model-深度解析)
7. [训练循环工程实现](#7-训练循环工程实现)
8. [分布式训练与优化](#8-分布式训练与优化)
9. [常见问题与调试](#9-常见问题与调试)
10. [总结与对比](#10-总结与对比)
[附录A：完整配置模板](#附录a完整配置模板)
[附录B：术语表](#附录b术语表)
[附录C：GRPO核心补充与踩坑要点](#附录cgrpo核心补充与踩坑要点)

---

## 1. 核心概念与符号系统
### 1.1 基础符号定义
|符号|含义|类型|
|---|---|---|
|$s$|状态（State），即 Prompt 输入| - |
|$a$|动作（Action），即 Response 输出| - |
|$\pi_\theta$|当前策略（Actor Model），参数为 $\theta$|可训练|
|$\pi_{\theta_{\text{old}}}$|旧策略（Rollout 时的策略快照）|冻结|
|$\pi_{\text{ref}}$|参考策略（Ref Model，通常SFT模型）|冻结|
|$(r(s,a))$|奖励函数（Reward Model）输出|标量|
|$(V(s))$|状态价值函数（Critic Model）输出|标量|
|$G$|每个 Prompt 生成的 Response 数量（rollout.n）|超参数|
|$B$|Batch 中 Prompt 的数量|超参数|
|$N$|总样本数 $N = B \times G$|推导量|
|$\gamma$|折扣因子 $([0,1])$|超参数|
|$\lambda$|GAE 权衡参数 $([0,1])$|超参数|
|$\varepsilon$|PPO Clip 范围，通常 0.2|超参数|
|$c_{\text{ent}}$|熵奖励系数|超参数|
|$\beta_{\text{KL}}$|KL 惩罚系数|超参数|

### 1.2 核心概念：三元组（Triplet）
GRPO 训练的最小单元：
$$ \text{Triplet} = (\text{Prompt},\ \text{Response},\ \text{Reward}) $$

示例：一个 Prompt 生成 4 个 Response（$(G=4)$）

|#|Prompt|Response|Reward|
|---|---|---|---|
|1|`What is 3 + 5?`|`8`|1.0|
|2|`What is 3 + 5?`|`3 + 5 = 8`|0.9|
|3|`What is 3 + 5?`|`3 + 5 = 7`|0.0|
|4|`What is 3 + 5?`|`I don't know`|0.1|

> ⚠️ GRPO特性：**整条Response只有一个标量reward，无token‑level奖励；所有输出token共享同一个优势值**。

### 1.3 关键公式：概率比
这是 PPO/GRPO 最核心的数学工具：
$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s,\ a_{\lt t})}{\pi_{\theta_{\text{old}}}(a_t \mid s,\ a_{\lt t})}
$$

等价形式（工程代码更常用）：
$$
\rho_t(\theta) = \exp\Big(\log \pi_\theta(a_t \mid s,\ a_{\lt t}) - \log \pi_{\theta_{\text{old}}}(a_t \mid s,\ a_{\lt t})\Big)
$$

含义：衡量当前策略与旧策略在同一个动作上的概率比值；重要性采样核心。

---

## 2. GRPO 算法数学推导
### 2.1 数据收集阶段（Rollout）
对于每个 Prompt $s_i$：
从旧策略采样 $G$ 个 Response：
$$
a_{i,j} \sim \pi_{\theta_{\text{old}}}(\cdot \mid s_i), \quad j = 1,\ 2,\ ...,\ G
$$

收集经验数据集：
$$
\mathcal{D} = \big\{(s_i,\ a_{i,j},\ r_{i,j})\big\}_{i=1}^{B},\ j=1..G
$$
其中 $r_{i,j} = \text{RewardModel}(s_i,\ a_{i,j})$。

> ✅ 时序约束：必须使用 $\pi_{\theta_{\text{old}}}$ 做rollout，rollout阶段计算`old_log_prob`；actor更新后不可重新计算old_log_prob，否则$\rho\equiv1$训练失效。

### 2.2 GRPO 优势计算（核心创新）
PPO 需要 Critic 网络估计基线：
$$
\hat{A}_t^{\text{PPO}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k},\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GRPO**不需要Critic网络**，使用**同Prompt组内标准化**构造优势：

步骤1：组内均值（作为基线经验估计，替代Critic的$V(s_i)$）
$$
\mu_i = \frac{1}{G} \sum_{j=1}^{G} r_{i,j}
$$

步骤2：组内标准差
$$
\sigma_i = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r_{i,j} - \mu_i)^2 + \epsilon},\quad \epsilon=10^{-8}
$$

步骤3：标准化得到序列级优势
$$
\hat{A}_{i,j} = \frac{r_{i,j} - \mu_i}{\sigma_i}
$$

步骤4：广播到所有输出Token
$$
\hat{A}_{i,j,t} = \hat{A}_{i,j}
$$
> 注解：同一个回答的全部生成token共用同一个优势；prompt部分token通过`response_mask`屏蔽，不参与loss。

> 前提条件：每个prompt必须采样 $G\ge2$；$G$越大基线估计质量越高；奖励绝对数值丢失，只保留组内相对排序，不适合依赖奖励绝对值的任务。

### 2.3 PPO vs GRPO 优势计算对比
|维度|PPO|GRPO|
|---|---|---|
|优势公式|$\hat{A}_t = \sum_k (\gamma\lambda)^k \delta_{t+k}$|$\hat{A}_{i,j} = (r_{i,j} - \mu_i)/\sigma_i$|
|基线来源|$(V(s))$（Critic网络）|$\mu_i$（同Prompt组内平均奖励）|
|需要模型|Actor + Critic + Ref + Reward|Actor + Ref + Reward|
|显存占用|高（4套模型FSDP）|较低；但rollout样本量放大$G$倍|
|计算复杂度|$O(T^2)$（GAE递归）|$(O(G))$简单统计|
|单Prompt采样|支持$(G=1)$|必须$G\ge2$|
|奖励格式|支持token‑level /序列奖励|仅支持整条序列标量奖励|

---

## 3. 损失函数完整解析
### 3.1 损失函数全景图
```
┌───────────────────────────────────────────────────────────────────────────┐
 │          GRPO 训练中的损失函数                │
 ├───────────────────────────────────────────────────────────────────────────┤
 │                                   │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │ 1. 熵损失 (Entropy Loss)      → 监控指标，不参与梯度更新      │   │
 │  │    L_ent = mean(H(π_(\theta)))                      │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                                   │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │ 2. 参考策略 KL 散度 (Ref KL)     → 监控/损失项             │   │
 │  │    L_kl = mean(log π_(\theta) - log π_ref)               │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                                   │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │ 3. Critic 损失 (Value Loss)      → GRPO 中跳过            │   │
 │  │    L_critic = (V_φ - V_target)²                 │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                                   │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │ 4. Actor 损失 (Policy Loss)      → 更新 Actor（核心！）     │   │
 │  │    L_actor = -L_CLIP - c_ent·H + β_KL·KL_est           │   │
 │  │    ├── L_CLIP = min((\rho)·A, clip((\rho))·A)     → 限制更新幅度      │   │
 │  │    ├── -c_ent·H                 → 鼓励探索            │   │
 │  │    └── +β_KL·KL_est              → 防止偏离参考模型（采样近似）│  │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                                   │
 └───────────────────────────────────────────────────────────────────────────┘
```
> ⚠️ 注意：$\text{KL}_{\text{est}}=\mathbb{E}[\log\pi_\theta-\log\pi_{\text{ref}}]$ 是**蒙特卡洛采样一阶近似**，不等于真实KL散度 $\text{KL}(\pi_\theta\parallel\pi_{\text{ref}})$，Verl等工业框架均使用该近似。

### 3.2 CLIP 损失（核心）
#### 3.2.1 数学公式
$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\Big( \rho_t(\theta)\cdot \hat{A}_t,\ \text{clip}\big(\rho_t(\theta),1-\varepsilon,1+\varepsilon\big)\cdot \hat{A}_t \Big) \Big]
$$

#### 3.2.2 分段分析
当 $\hat{A} > 0$（好动作）：
$$
\min(\rho \cdot A,\ \text{clip}(\rho)\cdot A) =
\begin{cases}
\rho \cdot A, & \rho \le 1+\varepsilon \\
(1+\varepsilon) \cdot A, & \rho > 1+\varepsilon
\end{cases}
$$

当 $\hat{A} < 0$（坏动作）：
$$
\min(\rho \cdot A,\ \text{clip}(\rho)\cdot A) =
\begin{cases}
\rho \cdot A, & \rho \ge 1-\varepsilon \\
(1-\varepsilon) \cdot A, & \rho < 1-\varepsilon
\end{cases}
$$

> 🎯面试考点：**当A<0，$\rho>1$不会触发clip保护**；如果策略放大坏动作概率，会持续惩罚，clip只限制“对好动作的过度抬升”，不保护坏动作。

#### 3.2.3 图解说明
优势 A > 0（好动作）：鼓励增加 $\rho$，上限截断
```
L_CLIP = (\rho)·A        ← 正常鼓励
        ┌──────────────────┐
        │ L_CLIP = (1+(\varepsilon))·A │  ← 裁剪，停止鼓励
        └──────────────────┘
(\rho):     0.8        1.0    1.2     1.5
        └── clip 范围 ────┘
```

优势 A < 0（坏动作）：鼓励减小 $\rho$；$\rho>1$不裁剪，持续惩罚
```
┌──────────────────┐
│ L_CLIP = (1−(\varepsilon))·A │  ← 仅(\rho)过低时裁剪，停止过度打压
└──────────────────┘
        ┌──────────────────┐
        │ L_CLIP = (\rho)·A     │  ← (\rho>1)区间，正常惩罚，不保护
        └──────────────────┘
(\rho):     0.5        0.8    1.0     1.2
        └── clip 范围 ────┘
```

### 3.3 熵奖励（Entropy Bonus）
#### 3.3.1 数学公式
$$
H_t = -\sum_{a} \pi_\theta(a \mid s,a_{\lt t}) \log \pi_\theta(a \mid s,a_{\lt t})
$$
$$
L_{\text{entropy}} = \frac{1}{\sum_t m_t}\sum_t m_t \cdot H_t
$$
$$
L_{\text{entropy\_bonus}} = -c_{\text{ent}} \cdot \mathbb{E}_t\big[ H(\pi_\theta(\cdot\mid s,a_{\lt t})) \big]
$$
$m_t$为response token掩码，prompt token置0。

> ⚠️工程实践：GRPO对齐SFT之后的大模型，**通常设置 $c_{\text{ent}}=0$**；SFT模型本身熵足够，增加熵奖励容易产生无意义输出。文档样例0.01仅演示。

#### 3.3.2 为什么取负号？
优化目标：最大化熵（鼓励探索）；训练使用梯度下降最小化损失
$$
\min_\theta \big(-c_{\text{ent}} \cdot H\big) \iff \max_\theta \big(c_{\text{ent}} \cdot H\big)
$$

### 3.4 KL 惩罚（KL Penalty）
#### 3.4.1 数学公式
$$
L_{\text{KL\_penalty}} = \beta_{\text{KL}} \cdot \mathbb{E}_t \big[ \log \pi_\theta(a_t\mid s,a_{\lt t}) - \log \pi_{\text{ref}}(a_t\mid s,a_{\lt t}) \big]
$$

#### 3.4.2 两种使用模式
1. **use_kl_in_reward=True（工业GRPO更常用）**
KL项加到奖励侧，间接修改优势，loss端不再加入KL正则
$$
\tilde{r}_t = r_t - \beta_{\text{KL}} \cdot \big(\log\pi_\theta - \log\pi_{\text{ref}}\big)
$$

2. **use_kl_in_reward=False**
KL直接作为正则项加到actor损失：
$$
L_{\text{actor}} = -L^{\text{CLIP}} - c_{\text{ent}} \cdot H + \beta_{\text{KL}} \cdot \text{KL}_{\text{est}}
$$

> 经验：奖励侧加KL损失梯度更稳定；损失侧加KL容易出现梯度冲突。

### 3.5 总 Actor 损失（最小化损失范式）
$$
L_{\text{actor}}(\theta)
= -\mathbb{E}_t\Big[\min\big(\rho_t \hat{A}_t,\ \text{clip}(\rho_t)\hat{A}_t\big)\Big]
- c_{\text{ent}} \cdot \mathbb{E}[H(\pi_\theta)]
+ \beta_{\text{KL}} \cdot \mathbb{E}_t\big[\log\pi_\theta-\log\pi_{\text{ref}}\big]
$$

---

## 4. 详细数值样例
### 4.1 场景设定
```python
config = {
    "rollout.n": 4,                # 每个 Prompt 生成 4 个 Response
    "clip_ratio": 0.2,             # Clip 范围 [0.8, 1.2]
    "entropy_coeff": 0.01,         # 仅演示，真实GRPO一般置0
    "kl_loss_coef": 0.001,
    "use_kl_in_reward": False,
}
prompt = "What is 3 + 5?"
```

### 4.2 步骤1：Rollout生成与奖励
|#|Response|Reward $r$|含义|
|---|---|---|---|
|1|`8`|1.0|完美答案|
|2|`3 + 5 = 8`|0.9|正确但冗长|
|3|`3 + 5 = 7`|0.0|错误答案|
|4|`I don't know`|0.1|无效回答|

### 4.3 步骤2：熵损失计算
假设4个Response的Token概率分布（候选Token：`["8","7","?"]`）

|#|Token概率分布|熵 $H$|
|---|---|---|
|1|$([0.6,0.3,0.1])$|0.897|
|2|$([0.5,0.4,0.1])$|0.943|
|3|$([0.2,0.7,0.1])$|0.802|
|4|$([0.1,0.1,0.8])$|0.639|

Response1熵计算：
$$
H_1 = -\big(0.6\log0.6 + 0.3\log0.3 + 0.1\log0.1\big) = 0.897
$$

平均熵：
$$
L_{\text{entropy}} = \frac{0.897 + 0.943 + 0.802 + 0.639}{4}=0.820
$$

### 4.4 步骤3：GRPO优势计算
组内统计：
$$
\mu = \frac{1.0+0.9+0.0+0.1}{4}=0.5
$$
$$
\sigma = \sqrt{\frac{(1.0-0.5)^2+(0.9-0.5)^2+(0.0-0.5)^2+(0.1-0.5)^2}{4}} = 0.395
$$

优势计算：

|#|$r$|$\hat{A}=(r-\mu)/\sigma$|含义|
|---|---|---|---|
|1|1.0|$(1.0-0.5)/0.395 = +1.27$|远高于平均，强化|
|2|0.9|$(0.9-0.5)/0.395 = +1.01$|高于平均，强化|
|3|0.0|$(0.0-0.5)/0.395 = -1.27$|远低于平均，抑制|
|4|0.1|$(0.1-0.5)/0.395 = -1.01$|低于平均，抑制|

### 4.5 步骤4：概率比计算

|#|$\log\pi_{\text{old}}$|$\log\pi_\theta$|$\rho=\exp(\log\pi_\theta-\log\pi_{\text{old}})$|
|---|---|---|---|
|1|-1.0|-0.4|$(\exp(0.6)=1.82)$|
|2|-1.0|-0.5|$(\exp(0.5)=1.65)$|
|3|-1.0|-0.2|$(\exp(0.8)=2.23)$|
|4|-1.0|-1.8|$\exp(-0.8)=0.45$|

### 4.6 步骤5：CLIP损失计算
$\varepsilon=0.2$，clip区间$([0.8,1.2])$

|#|$\hat{A}$|$\rho$|$\text{clip}(\rho)$|$\rho\cdot\hat{A}$|$\text{clip}(\rho)\cdot\hat{A}$|$L_{\text{CLIP}}=\min(\cdot)$|裁剪？|
|---|---|---|---|---|---|---|---|
|1|+1.27|1.82|1.2|2.31|1.52|1.52|裁剪|
|2|+1.01|1.65|1.2|1.67|1.21|1.21|裁剪|
|3|-1.27|2.23|1.2|-2.83|-1.52|-2.83|**不裁剪**|
|4|-1.01|0.45|0.8|-0.45|-0.81|-0.81|裁剪|

示例1（好动作）：
$$
\rho\cdot\hat{A}=1.82\times1.27=2.31,\quad \text{clip}(\rho)\cdot\hat{A}=1.2\times1.27=1.52,\quad L_{\text{CLIP}}=\min(2.31,1.52)=1.52
$$

示例3（坏动作）：
$$
\rho\cdot\hat{A}=2.23\times(-1.27)=-2.83,\quad \text{clip}(\rho)\cdot\hat{A}=1.2\times(-1.27)=-1.52,\quad L_{\text{CLIP}}=\min(-2.83,-1.52)=-2.83
$$

平均CLIP损失：
$$
L_{\text{CLIP}}=\frac{1.52+1.21+(-2.83)+(-0.81)}{4}=-0.23
$$

### 4.7 步骤6：熵奖励与KL惩罚
熵奖励：
$$
L_{\text{entropy\_bonus}}=-c_{\text{ent}} \times L_{\text{entropy}} = -0.01 \times 0.820 = -0.0082
$$

KL惩罚（ref模型log概率）

|#|$\log\pi_\theta$|$\log\pi_{\text{ref}}$|$\text{KL}_{\text{est}}=\log\pi_\theta-\log\pi_{\text{ref}}$|
|---|---|---|---|
|1|-0.4|-0.5|0.1|
|2|-0.5|-0.6|0.1|
|3|-0.2|-0.3|0.1|
|4|-1.8|-2.0|0.2|

$$
L_{\text{KL}}=\frac{0.1+0.1+0.1+0.2}{4}=0.125
$$
$$
L_{\text{KL\_penalty}}=\beta_{\text{KL}} \times L_{\text{KL}} = 0.001 \times 0.125 = 0.000125
$$

### 4.8 步骤7：总损失
$$
L_{\text{actor}} = -L_{\text{CLIP}} - L_{\text{entropy\_bonus}} + L_{\text{KL\_penalty}}
$$
$$
L_{\text{actor}} = -(-0.23) - (-0.0082) + 0.000125 = 0.238
$$

### 4.9 训练效果总结
```text
训练前：
    P("8")             = 0.30
    P("3 + 5 = 8")     = 0.25
    P("3 + 5 = 7")     = 0.25
    P("I don't know")  = 0.20

GRPO优化后（梯度方向）：
    P("8")             = 0.35   ↑ (+0.05)  好答案增加
    P("3 + 5 = 8")     = 0.30   ↑ (+0.05)  好答案增加
    P("3 + 5 = 7")     = 0.15   ↓ (-0.10)  坏答案大幅减少
    P("I don't know")  = 0.20   → (0)      坏答案变化不大
```

---

## 5. Reward Model 深度解析
### 5.1 核心功能
Reward Model 的核心功能是评估 Response 的质量，输出一个标量分数。
> ⚠️ RM输出是相对打分，值域无约束，**GRPO训练阶段不要做sigmoid**，新手高频错误。

### 5.2 训练数据与损失
#### 5.2.1 Bradley‑Terry模型
人类偏好建模基础：
$$
P(\text{response}_1 \succ \text{response}_2) = \frac{\exp(r(s,a_1))}{\exp(r(s,a_1))+\exp(r(s,a_2))}
$$
$a_1\succ a_2$：$a_1$比$a_2$更受人类偏好。

#### 5.2.2 训练损失
$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(s,a_w,a_l)\sim\mathcal{D}} \Big[ \log\sigma\big(r(s,a_w)-r(s,a_l)\big) \Big]
$$
- $a_w$：Winning，偏好样本
- $a_l$：Losing，较差样本
- $\sigma$：Sigmoid函数

#### 5.2.3 梯度推导
$$
\frac{\partial \mathcal{L}_{\text{RM}}}{\partial r(s,a_w)} = -\sigma\big(r(s,a_l)-r(s,a_w)\big)
$$
$$
\frac{\partial \mathcal{L}_{\text{RM}}}{\partial r(s,a_l)} = \sigma\big(r(s,a_w)-r(s,a_l)\big)
$$

含义：提升winner分数、降低loser分数，拉大两者差距减小损失。

### 5.3 模型架构
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, response_mask=None):
        # 1.基础模型编码
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]

        # 2.token级别奖励头
        token_rewards = self.reward_head(last_hidden).squeeze(-1)

        # 3.聚合为序列标量奖励
        if response_mask is not None:
            # 对response部分取平均
            reward = (token_rewards * response_mask).sum(dim=1) / response_mask.sum(dim=1)
        else:
            # 取最后一个token
            reward = token_rewards[:, -1]
        return reward
```

### 5.4 RM训练流程
人类偏好数据 → $(a_w,a_l)$ → RM前向得到 $r_w-r_l$ → BCE‑like损失 → 梯度下降更新RM。

```python
def train_reward_model(model, dataloader, optimizer):
    for batch in dataloader:
        reward_win = model(batch.prompt, batch.response_win)
        reward_lose = model(batch.prompt, batch.response_lose)
        logits = reward_win - reward_lose
        loss = -torch.log(torch.sigmoid(logits)).mean()
        loss.backward()
        optimizer.step()
```

### 5.5 PPO/GRPO中RM使用
RM全程冻结，推理模式运行：
```python
with torch.no_grad():
    rewards = reward_model(prompts, responses)
```

---

## 6. Critic Model 深度解析
> GRPO训练不使用Critic；本章节为PPO‑LM对照。

### 6.1 核心功能
Critic Model 的核心功能是估计状态价值 $(V(s))$，作为计算优势的基线。

### 6.2 数学基础
#### 6.2.1 状态价值函数
$$
V^\pi(s) = \mathbb{E}_{a_t\sim\pi,\ s_{t+1}\sim P} \left[ \sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s \right]
$$

#### 6.2.2 Bellman方程
$$
V^\pi(s) = \mathbb{E}_{a\sim\pi}\Big[ (r(s,a)) + \gamma\mathbb{E}_{s'\sim P}V^\pi(s') \Big]
$$

#### 6.2.3 优势函数
$$
(\hat{A})\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

### 6.3 GAE（Generalized Advantage Estimation）
PPO使用GAE估计优势，平衡偏差‑方差。

#### 6.3.1 TD误差
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

#### 6.3.2 GAE公式
$$
\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}
$$

#### 6.3.3 递归实现
```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0
    # 从后往前递推
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns
```

> ⚠️GRPO：整条序列只有末尾单标量奖励，无中间token奖励，工程固定 $\gamma=1.0$。

#### 6.3.4 数值示例
```python
rewards = [0.1, -0.2, 0.3, 0.0]
values  = [0.5, 0.6, 0.4, 0.3, 0.2]
gamma = 0.99
lam = 0.95
```

TD误差：
$$
\delta_0 = 0.1 + 0.99\times0.6 - 0.5 = 0.194
$$
$$
\delta_1 = -0.2 + 0.99\times0.4 - 0.6 = -0.404
$$
$$
\delta_2 = 0.3 + 0.99\times0.3 - 0.4 = 0.197
$$
$$
\delta_3 = 0.0 + 0.99\times0.2 - 0.3 = -0.102
$$

GAE递推：
$$
\text{gae}_3 = \delta_3 = -0.102
$$
$$
\text{gae}_2 = 0.197 + 0.99\times0.95\times(-0.102) = 0.101
$$
$$
\text{gae}_1 = -0.404 + 0.99\times0.95\times0.101 = -0.309
$$
$$
\text{gae}_0 = 0.194 + 0.99\times0.95\times(-0.309) = -0.096
$$

|t|advantages|returns|
|---|---|---|
|0|-0.096|0.404|
|1|-0.309|0.291|
|2|0.101|0.501|
|3|-0.102|0.198|

### 6.4 Critic训练损失
#### 6.4.1 基础损失
$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_t\Big[ \big(V_\phi(s_t)-\hat{V}_t\big)^2 \Big]
$$
$\hat{V}_t$为GAE计算得到的returns目标价值。

#### 6.4.2 裁剪价值损失
```python
class ClippedValueLoss:
    def __init__(self, clip_range=0.2):
        self.clip_range = clip_range

    def compute_loss(self, values, old_values, returns):
        loss_unclipped = (values - returns) ** 2
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_range,
            self.clip_range
        )
        loss_clipped = (values_clipped - returns) ** 2
        loss = torch.max(loss_unclipped, loss_clipped)
        return loss.mean()
```

### 6.5 Critic与Actor交互（PPO）
```python
class PPOTrainer:
    def compute_advantage(self, batch):
        rewards = batch.batch["token_level_rewards"]
        values = batch.batch["values"]
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            gamma=self.config.gamma,
            lam=self.config.lam
        )
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns
        return batch

    def update_critic(self, batch):
        values = self.critic_model(batch["input_ids"])
        loss_fn = ClippedValueLoss(clip_range=0.2)
        value_loss = loss_fn.compute_loss(
            values=values,
            old_values=batch["old_values"],
            returns=batch["returns"]
        )
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        return {"critic_loss": value_loss.item()}
```

---

## 7. 训练循环工程实现
> 参考Verl架构；关键工程坑点已加入注释

### 7.1 GRPOTrainer核心伪代码
```python
class GRPOTrainer:
    def __init__(self, config):
        self.config = config
        self.global_steps = 0
        # 初始化模型
        self.actor_model = ActorModel(config.model_path)
        self.ref_model = RefModel(config.model_path)
        self.reward_model = RewardModel(config.reward_path)
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.actor_model.parameters(),
            lr=config.actor.optim.lr
        )
        # vLLM rollout引擎
        self.rollout_engine = vLLMEngine(
            model=self.actor_model,
            gpu_memory_utilization=config.rollout.gpu_memory_utilization
        )

    def _train_step(self, batch_dict):
        batch = DataProto.from_single_dict(batch_dict)
        metrics = {}

        # 阶段1：Rollout生成（使用π_(\theta)_old，生成同时计算old_log_prob）
        outputs = self.rollout_engine.generate(
            prompts=batch["prompts"],
            n=self.config.rollout.n,
            temperature=self.config.rollout.temperature,
            max_tokens=self.config.data.max_response_length
        )
        batch = self._construct_triplets(batch, outputs)

        # 阶段2：奖励计算，RM冻结no_grad
        with torch.no_grad():
            reward_scores = self.reward_model(
                input_ids=batch["sequences"],
                attention_mask=batch["attention_mask"],
                response_mask=batch["response_mask"]
            )
        batch.batch["token_level_scores"] = reward_scores

        # 阶段3：分布式对齐填充
        divisor = self.actor_world_size * micro_bsz
        batch, pad_size = pad_dataproto_to_divisor(batch, divisor)

        # 阶段4：rollout阶段计算old_log_prob，⚠️必须在actor参数更新之前！
        old_log_prob, _ = self._compute_old_log_prob(batch)
        batch = batch.union(old_log_prob)

        # 阶段5：参考策略log prob
        if self.use_reference_policy:
            ref_log_prob = self._compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        # 阶段6：去除填充
        batch = unpad_dataproto(batch, pad_size)

        # 阶段7：GRPO组归一化优势计算
        batch = self._compute_grpo_advantage(batch)

        # 阶段8：数据过滤
        keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
        batch = batch[keep_indices]
        mini_batch_size = self.config.actor.ppo_mini_batch_size
        n_remained = len(batch) // mini_batch_size * mini_batch_size
        batch = batch[list(range(n_remained))]

        # 阶段9：Actor网络更新
        actor_metrics = self._update_actor(batch)
        metrics.update(actor_metrics)

        # ✅关键：训练完成后同步权重到vLLM rollout副本
        self.sync_weights_to_rollout_replicas()
        self.global_steps += 1
        return metrics

    def _compute_grpo_advantage(self, batch):
        rewards = batch.batch["token_level_scores"]
        response_mask = batch.batch["response_mask"]
        total_rewards = (rewards * response_mask).sum(dim=-1)
        G = self.config.rollout.n
        N = len(total_rewards)
        advantages = torch.zeros_like(total_rewards)

        # 按prompt分组做组内归一化
        for i in range(0, N, G):
            group_rewards = total_rewards[i:i+G]
            mu = group_rewards.mean()
            sigma = group_rewards.std() + 1e-8
            group_advantages = (group_rewards - mu) / sigma
            advantages[i:i+G] = group_advantages

        # 广播到token维度，prompt位置mask置0
        token_advantages = advantages.unsqueeze(-1) * response_mask
        batch.batch["advantages"] = token_advantages
        return batch

    def _update_actor(self, batch):
        log_probs = self.actor_model.compute_log_probs(
            input_ids=batch["sequences"],
            attention_mask=batch["attention_mask"]
        )
        old_log_probs = batch.batch["old_log_probs"]
        ratio = torch.exp(log_probs - old_log_probs)
        advantages = batch.batch["advantages"]

        clip_eps = 0.2
        ratio_clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵项
        entropy = -(log_probs * log_probs.exp()).mean()
        entropy_bonus = -self.config.actor.entropy_coeff * entropy

        # KL正则
        ref_log_probs = batch.batch.get("ref_log_probs")
        if ref_log_probs is not None and not self.config.algorithm.use_kl_in_reward:
            kl_est = (log_probs - ref_log_probs).mean()
            kl_penalty = self.config.actor.kl_loss_coef * kl_est
        else:
            kl_est = torch.tensor(0.0)
            kl_penalty = 0.0

        total_loss = policy_loss + entropy_bonus + kl_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "actor/policy_loss": policy_loss.item(),
            "actor/entropy": entropy.item(),
            "actor/kl": kl_est.item(),
            "actor/total_loss": total_loss.item()
        }
```

### 7.2 old_log_prob计算工具函数
```python
def _compute_old_log_prob(self, batch):
    """rollout阶段计算旧策略log概率；不可在actor更新之后调用"""
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)
    tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
    output = self.actor_rollout_wg.compute_log_prob(batch_td)
    entropy = tu.get(output, "entropy")
    log_probs = tu.get(output, "log_probs")
    mfu = tu.get(output, "metrics")["mfu"]
    entropy = no_padding_2_padding(entropy, batch_td)
    log_probs = no_padding_2_padding(log_probs, batch_td)
    result = {
        "old_log_probs": log_probs.float(),
        "entropys": entropy.float()
    }
    return DataProto.from_tensordict(tu.get_tensordict(result)), mfu
```

> ⚠️工程坑点汇总：
> 1. 必须同步actor权重到vLLM rollout副本；权重不同步，old_log_prob与生成样本不匹配，训练崩坏
> 2. `response_mask`屏蔽prompt token，prompt部分不参与loss
> 3. padding位置mask置0，不参与损失计算
> 4. GRPO序列奖励模式，gamma固定等于1.0
> 5. GRPO务必开启梯度裁剪防止梯度爆炸

---

## 8. 分布式训练与优化
### 8.1 显存管理
```python
class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.replicas = []
        self.actor_offload = config.actor.fsdp_config.param_offload

    def sleep_replicas(self):
        """rollout副本休眠，释放显存"""
        for replica in self.replicas:
            if self.actor_offload:
                replica.model.to('cpu')
            replica.vllm_engine.release_kv_cache()
        torch.cuda.empty_cache()

    def wake_up_replicas(self):
        """唤醒rollout副本"""
        for replica in self.replicas:
            if self.actor_offload:
                replica.model.to('cuda')
            replica.vllm_engine.allocate_kv_cache()
```

### 8.2 权重同步
```python
async def update_weights(self, global_steps=None):
    """训练侧Actor权重同步到rollout副本，每一轮ppo step执行"""
    await self.abort_replicas()
    await self.release_kv_cache_replicas()
    self.build_process_group(self.rollout)
    ray.get(
        self.trainer.update_weights(global_steps=global_steps)
        + self.rollout.update_weights(global_steps=global_steps)
    )
    await self.resume_kv_cache_replicas()
    await self.resume_generation_replicas()
```

> 分布式关键点：
> - FSDP管理Actor/Ref；rollout使用独立vLLM副本；每step完成权重同步
> - param_offload：空闲时把模型参数卸载CPU，缓解显存压力
> - NestedTensor / remove‑padding：消除padding计算开销，大RL训练核心优化

---

## 9. 常见问题与调试
### 9.1 训练不稳定
|问题|可能原因|解决方案|
|---|---|---|
|Loss剧烈震荡|学习率过大；缺少warmup|降低lr，增加lr warmup|
|KL爆炸>0.1|$\beta_{\text{KL}}$过小；lr过高|增大kl系数；降低actor lr；减小clip_ratio|
|熵快速下跌|策略模式崩溃；探索不足|少量增大entropy_coeff；调高rollout温度；GRPO优先保持0|
|梯度爆炸|优势缩放异常；缺少梯度裁剪|开启clip_grad_norm；检查组归一化sigma是否接近1|
|reward不上涨但loss下降|old_log_prob与rollout策略不一致|检查权重同步时序，rollout后立刻计算old_log_prob|
|advantage全部趋近0|rollout.n(G)太小，组内奖励区分度不足|增大rollout.n至8/16|

### 9.2 显存不足
报错：`RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB`
- 降低`gpu_memory_utilization`（0.7 → 0.5~0.6）
- 开启`param_offload: true`
- 减小`ppo_micro_batch_size_per_gpu`
- 打开`enable_gradient_checkpointing: true`

### 9.3 Ray磁盘空间爆满
```bash
rm -rf /tmp/ray/session_*
rm -f /tmp/nccl_debug.log*
ray stop --force
```

### 9.4 必须监控关键指标
|指标|判断标准|风险信号|
|---|---|---|
|actor/kl|0.01~0.05合理|>0.05模型偏移；>0.1训练崩坏|
|actor/entropy|维持合理区间|快速持续下降→模式崩溃|
|reward_std|组归一化后advantage std≈1|偏离过大说明组归一化异常|

---

## 10. 总结与对比
### 10.1 PPO‑LM vs GRPO完整对比
|维度|PPO‑LM|GRPO|
|---|---|---|
|优势计算|$\hat{A}_t=\sum_k(\gamma\lambda)^k\delta_{t+k}$|$\hat{A}_{i,j}=(r_{i,j}-\mu_i)/\sigma_i$|
|Critic模型|需要|不需要|
|所需模型|Actor, Critic, Ref, Reward|Actor, Ref, Reward|
|显存占用|高（4套FSDP模型）|较低；rollout样本量放大$G$倍|
|训练复杂度|高，GAE、Critic调参繁琐|低，去掉Critic训练链路|
|单Prompt采样|支持$(G=1)$|强制$G\ge2$|
|奖励格式|token‑level /序列奖励均可|仅支持序列标量奖励|
|偏差方差|Critic引入偏差；GAE调参敏感|$G$有限引入估计方差；无Critic偏差|
|适用场景|机器人交互、token‑level奖励、通用RL|LLM偏好对齐、推理、代码校验、外部reward函数|

### 10.2 DPO / GRPO / PPO横向对比
|算法|On‑Policy|Rollout|RM|Critic|特点|
|---|---|---|---|---|---|
|PPO|✅|✅|可选|✅|通用强化学习，调参复杂|
|GRPO|✅|✅|✅|❌|PPO变体；支持任意外部reward函数；每个prompt多采样|
|DPO|❌|❌|❌|❌|离线偏好对；无法接入外部reward|

> 🎯面试一句话总结GRPO：
> GRPO是PPO的on‑policy变体，移除Critic网络；利用同一个prompt下多条生成样本做组归一化得到优势；牺牲单样本rollout能力，换取工程简化，专门适配大模型偏好对齐，支持自定义外部奖励（代码执行、数学校验等）。

### 10.3 关键公式速查
|公式|用途|
|---|---|
|$\rho_t = \exp(\log\pi_\theta-\log\pi_{\text{old}})$|新旧策略概率比|
|$\mu_i=\frac1G\sum_j r_{i,j}$|GRPO组内均值（基线）|
|$\sigma_i=\sqrt{\frac1G\sum_j(r_{i,j}-\mu_i)^2+\epsilon}$|GRPO组内标准差|
|$\hat{A}_{i,j}=\dfrac{r_{i,j}-\mu_i}{\sigma_i}$|GRPO标准化优势|
|$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$|TD误差（PPO）|
|$\hat{A}_t^{\text{GAE}}=\sum_k(\gamma\lambda)^k\delta_{t+k}$|GAE优势（PPO）|
|$L^{\text{CLIP}}=\mathbb{E}\big[\min(\rho A,\text{clip}(\rho)A)\big]$|PPO‑CLIP目标|
|$L_{\text{actor}}=-L^{\text{CLIP}}-c_{\text{ent}}H+\beta_{\text{KL}}\text{KL}_{\text{est}}$|GRPO Actor损失|

### 10.4 最终总结
GRPO通过**组内样本标准化**优雅替代PPO的Critic模型：
1. 节省资源：移除一套Critic模型，减少FSDP显存开销；代价是rollout样本量放大$G$倍
2. 简化链路：省去Critic训练调参，工程链路更短
3. 能力等价：在LLM偏好对齐、推理、代码任务效果与PPO‑LHF相当
4. 数学约束：奖励只保留组内相对排序；依赖每个prompt多条采样；不适合依赖奖励绝对值的任务

掌握整套原理，可以完成：GRPO参数调优、训练故障定位、自定义奖励函数、分布式RL训练工程落地。
