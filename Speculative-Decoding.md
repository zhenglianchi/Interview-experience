# Speculative Decoding 完全指南（EAGLE 系列 + 项目集成）

> 对应简历项目（一）亮点"集成 EAGLE-3 投机解码：生成吞吐 199→282 tok/s（+41.7%）、单 token 延迟降低 39.5%、全样本训练评测 83.23% 与 baseline 持平"。一句话知识框架：
> **投机解码 = "小草稿模型先猜 k 个 token，大模型一次前向并行验证，只重算被拒的 token"**——把 decode 从"每步 1 token 的访存串行瓶颈"摊平成"1 次验证换 k 个 token"；EAGLE 系列是其中最实用的一族（草稿质量高、可训练、被 vLLM/SGLang 原生支持）。
>
> 素材来源：SafeAILab/EAGLE 官方仓库（本地克隆 `C:\Users\HW\Desktop\简历投递\EAGLE`，commit `cb7e084`）、vLLM main 分支（本地 `vllm/v1/spec_decode/`）、本项目 verl + vLLM 0.11.1 集成（`uniagent-lighting/scripts/`）。并行/通信背景见 `Parallel.md`、`Communication.md`。

---

# 一、投机解码原理

## 1. 现有问题：decode 为什么慢——访存瓶颈 + 串行步数

### 1. 现有问题

- **decode 是"一步一个 token"的串行过程**：生成 1000 个 token 就要 1000 次模型前向，每次前向只多算 1 个 token 的注意力，但**要读全部权重**（比如 8B 模型 bf16 ≈ 16 GB）；
- **decode 是访存密集（memory-bound）**：单步计算量 = 2×模型参数 × 1 token 的前向（约 32 GFLOPs for 8B），而 H100 的算力是 989 TFLOPS——算力利用率不到 5%，瓶颈是**每步从 HBM 读 16GB 权重**（H100 HBM 带宽 3.35 TB/s → 单步 ≥ 4.8 ms 的纯访存时间）；
- 于是"吞吐"被锁死：单卡单请求 decode ≈ 200 token/s 量级。想提速只有两条路：① 增大 batch（分摊权重读取，但受显存和延迟约束）；② **减少串行步数**——这就是投机解码。

### 2. 方法论（瓶颈建模）

Roofline 视角：decode 单步时间 ≈ $\max(\text{访存时间}, \text{计算时间}) \approx \text{访存时间} = \frac{2P}{B}$（$P$=参数量，$B$=HBM 带宽）。所以 decode 提速的本质是**摊薄"每步读一次全量权重"的成本**：如果能一次前向产出多个 token（并行验证 k 个候选），等效"每步读一次权重、出 E 个 token"，吞吐直接乘 E。

### 3. 具体数值样例

- Qwen3-8B（bf16，$P=8\times10^9$，$2P=16$ GB），H100 HBM 带宽 $B=3.35$ TB/s：单步理论下限 $16\text{GB}/3.35\text{TB/s} \approx 4.8$ ms → 单请求上限 ≈ 208 token/s（实际 ~150-200，含 kernel 开销）；
- 若投机解码期望每步出 $E=2.5$ 个 token，等效每 token 访存成本降到 $16/2.5 = 6.4$ GB → 上限 ≈ 520 token/s——**瓶颈没有消失，但被"每步多产出"摊薄了**；
- 这就是为什么投机解码对**大模型、低 batch、长生成**场景特别有效（batch 大时权重读取已被多请求分摊，投机收益变小）。

> 面试一句话总结：**decode 是访存密集的串行过程，单步时间≈读一遍全量权重（2P/B），算力利用率常低于 5%；投机解码不改变单步成本，而是让"一次前向产出多个 token"，把每 token 的等效访存成本摊薄，从而突破单请求 decode 吞吐上限。**

---

## 2. 通用框架：草稿（draft）+ 验证（verify）+ 拒绝采样

### 1. 现有问题

- 让大模型"直接预测未来 k 个 token"（如 MTP 头）可以，但**大模型验证 k 个位置的前向成本与生成 1 个 token 几乎一样**（权重读取相同）——关键是"谁来猜、猜错了怎么办"；
- 猜错必须能**纠正且不破坏输出分布**：直接丢弃草稿重算，会引入分布偏差（生成的文本与正常解码不一致）。

### 2. 方法论

投机解码的标准流水线（Leviathan et al. / Chen et al.，2023）：

```text
每轮迭代：
1. DRAFT：用草稿模型（小模型）自回归地猜 k 个 token：x̂₁, x̂₂, ..., x̂ₖ
2. VERIFY：大模型一次前向，并行计算这 k 个位置的概率分布 p(x̂ᵢ | context)
3. ACCEPT/REJECT：对每个位置 i：
   - 若 x̂ᵢ 恰好是大模型分布 p 的 argmax → 接受（概率高）
   - 否则按概率 min(1, q(x̂ᵢ)/p(x̂ᵢ)) 接受（q=草稿分布，p=大模型分布）
   - 第一个被拒的位置 j：从 p 的"残差分布"重新采样一个 token 替换 x̂ⱼ
4. 已接受的 token 直接作为输出，下轮草稿从新位置继续
```

**关键性质（分布保持）**：只要拒绝时从残差分布 $p'(x) \propto \max(0, p(x) - q(x))$ 重新采样，最终输出分布**与直接用大模型自回归采样完全一致**——这是投机解码"无损"的理论保证（证明思路：接受概率和拒绝采样共同构造出 p 的精确采样）。

**接受率 α 的意义**：α = 草稿 token 与大模型分布一致的概率。α 越高，期望每步输出越多。对**完美草稿**（α=1）每步出 k+1 个 token；对**随机草稿**（α=1/vocab）几乎不加速。

**期望输出 token 数**（草稿 k 个，接受率 α）：
$$E[\text{tokens per step}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$
（若第 i 个位置被拒则只输出前 i-1 个 + 1 个重采样 token）

### 3. 具体数值样例

设 k=3（猜 3 个）、接受率 α=0.7：

- 期望每步输出：$E = (1-0.7^4)/(1-0.7) = (1-0.2401)/0.3 = 2.533$ 个 token；
- 每轮成本 = 草稿 3 步（设草稿单步成本 = 大模型单步的 c=0.15）+ 大模型验证 1 步 = $3\times0.15 + 1 = 1.45$ 个"大模型步"；
- 加速比 $S = E / (1 + kc) = 2.533 / 1.45 = 1.747$×；
- 若接受率掉到 α=0.5：$E = (1-0.5^4)/0.5 = 1.875$，$S = 1.875/1.45 = 1.29$×——**接受率是投机解码的生命线**；
- 若 α=0.9：$E = (1-0.6561)/0.1 = 3.44$，$S = 3.44/1.45 = 2.37$×。

> 面试一句话总结：**投机解码 = 草稿模型猜 k 个 token + 大模型一次并行验证，按 min(1, q/p) 概率接受、被拒处从残差分布重采样——分布保持无损；加速比 = 期望输出 / (1 + k·草稿成本比)，接受率 α 决定一切：α=0.7、k=3、c=0.15 时约 1.75×。**

---

## 3. 三条草稿路线：独立小模型 / 浅层自回归（EAGLE）/ 并行头（Medusa）

### 1. 现有问题

- 草稿模型的**质量（接受率 α）**与**成本（c）**直接决定加速比：独立小模型质量低（α 低）、成本高（要加载一套完整权重）；
- 理想草稿应该"**与大模型共享尽可能多的知识，只补一点点自回归能力**"。

### 2. 方法论（三条路线对比）

| 路线 | 代表 | 草稿怎么来 | 质量/成本 | 缺点 |
|---|---|---|---|---|
| 独立小模型 | Draft Model（vLLM 原生） | 单独训练一个小 LM（如 0.5B 猜 8B） | α 低、c 高（完整权重） | 分布不匹配，接受率差 |
| **浅层自回归（feature 外推）** | **EAGLE 系列** | 用目标模型**倒数第二层 hidden state** 喂一个 1~2 层自回归模块，预测下个 token 的 feature → 共享 LM head 出 token | **α 高、c 极低（1~2 层）** | 训练时需目标模型 feature（训练成本略高） |
| 并行 head | Medusa | 在最后一层加多个并行 head，各预测第 i 个 token | α 中、c 低 | head 之间**无自回归依赖**（只条件于真前缀），长草稿质量衰减快 |
| n-gram（零模型） | SGLang n-gram | 从上下文统计中猜 | α 场景相关 | 仅对重复文本有效 |

**关键洞察**：LLM 最后几层的 feature 已经"几乎决定下一个 token"——所以**直接外推 feature 比用小模型猜 token 准得多**，且 1~2 层模块成本 ≈ 主模型 5%~15%。这是 EAGLE 的核心思想（第 6 点展开）。

### 3. 具体数值样例

- 8B 模型，hidden=4096，单层 transformer 成本 ≈ 一个 attention + MLP ≈ 主模型一层；主模型 32 层 → c ≈ 1/32 ≈ 0.03（不含 embedding）~0.1（含）；
- 同一批 HumanEvalFix prompt 上实测（本项目）：Qwen3-8B 独立解码 199 tok/s，Qwen3-8B + EAGLE-3 草稿 282 tok/s（+41.7%）——对应反推接受率 α≈0.56~0.66（第 10 点演算），远高于典型独立小模型（α≈0.3~0.5）；
- Medusa 的并行头在 k≥2 后质量快速衰减（无自回归依赖），EAGLE 自回归浅层在 k=3 仍保持高接受率——这是 EAGLE 系在"长草稿"上胜出的原因。

> 面试一句话总结：**草稿模型三条路线——独立小模型（α 低 c 高）、EAGLE 式 feature 外推（用目标模型倒数第二层 feature 喂浅层自回归模块，α 高 c 极低，最优）、Medusa 式并行头（无自回归依赖，长草稿衰减快）；EAGLE 系是质量/成本比最优的一族。**

---

## 4. 树形草稿与批量验证：一次验证 k 个分支

### 1. 现有问题

- 简单投机是"一条链"（猜 k 个 token 一条线），第一个 token 猜错就全废——**链式草稿的期望输出受首 token 接受率限制**；
- 大模型验证时，草稿的每个 token 位置都要算注意力，链式草稿的"不同位置"之间没有并行分支，浪费验证算力。

### 2. 方法论

**树形草稿（tree draft）**：草稿阶段生成一棵树（多个分支），大模型一次前向**并行验证所有分支**（同一 batch 内不同序列长度 padding 到一致），然后**贪心选最长被接受的前缀**。数据结构在 EAGLE 仓库里就是显式的树：

```python
# EAGLE/eagle/modeling_eagle.py（EAGLE-3 的树形草稿数据结构）
class node:
    def __init__(self, parent=None, value=None, dict_key=None):
        self.parent = parent
        self.value = value
        if parent:
            self.depth = parent.depth + 1
            parent.children.append(self)
        else:
            self.depth = 0
        self.children = []
        self.dict_key = dict_key

    def all_index(self):
        """从根到本节点的完整 token 索引序列"""
        if not self.parent.parent:
            return [self.index]
        return self.parent.all_index() + [self.index]

class Tree:
    def __init__(self, tree_list):
        """tree_list: 所有分支路径（如 [[a], [a,b], [a,c], [a,b,d], ...]），
        按长度排序后逐条建节点，共享前缀自动合并成树"""
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root = node()
        self.node_dic = {}
        for tree_node in sorted_tree_list:
            cur_value = tree_node[-1]
            if len(tree_node) == 1:
                cur_node = node(parent=self.root, value=cur_value, dict_key=tuple(tree_node))
            else:
                cur_parent = self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value, dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()
```

**这段代码关键在哪**：`node` 的 `all_index()` 定义了"从根到某节点的路径"（验证时就是这条路径的 KV 位置）；`Tree` 把多个分支路径按"共享前缀自动合并"建树——这就是"**一批候选路径、一次并行验证**"的数据结构基础：vLLM/SGLang 验证时按树展开成 batch，被接受的最长前缀对应树上深度最大的被接受节点。

**验证策略**（vLLM 侧）：
- 贪心验证：每个位置若草稿 token == 大模型 argmax 则接受，否则从该位置截断（`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` 的 `_verify`）；
- 概率验证（温度>0 时）：`min(1, q/p)` 接受 + 残差重采样，保持分布一致。

### 3. 具体数值样例

- 草稿宽度 2、深度 3 的树：分支 `[t1]`、`[t1,t2a]`、`[t1,t2b]`、`[t1,t2a,t3]`、`[t1,t2b,t3']`——共 5 条路径，最大深度 3；
- 大模型一次前向验证 5 条路径（长度 1~3，padding 到 3），若 `[t1,t2a,t3]` 全接受（α 高），输出 3 个 token——**链式草稿此时只能输出 1 个**（t1 之后只能走一条线）；
- 树形 vs 链式：同样 k=3 预算，树形草稿的期望输出 ≈ 1.3~1.8 倍于链式（分支覆盖了"第二个 token 的多种可能"）；
- 代价：树的分支数越多，验证 batch 越大（padding 浪费），所以树宽/深要按 GPU 算力与接受率调（vLLM 的 `num_speculative_tokens` + 树的形状配置）。

> 面试一句话总结：**树形草稿把"一条链"变成"一棵树"，大模型一次前向并行验证所有分支、贪心选最长被接受前缀——用验证算力换期望输出（约 1.3~1.8× 于链式），代价是树越宽 padding 浪费越大，需按接受率调树形。**

---

# 二、EAGLE 系列算法（SafeAILab/EAGLE 仓库）

## 5. EAGLE-1：feature 外推（Extrapolation Algorithm for Greater Language-model Efficiency）

### 1. 现有问题

- 独立小模型草稿：与目标模型分布不匹配，接受率低（α≈0.3~0.5），还要额外加载一套权重（显存 + 带宽双贵）；
- Medusa 并行头：只条件于真实前缀、无自回归，k≥2 质量差；
- 洞察：**目标模型倒数第二层（top-layer）的 hidden state 已经"浓缩"了下一个 token 的全部信息**（最后一层只是把它投影到词表）——直接在这个 feature 上做自回归外推，比任何独立小模型都准。

### 2. 方法论

EAGLE-1（arXiv:2401.15077）的草稿网络是一个**轻量自回归模块**：输入 = **上一个 token 的 embedding ⊕ 目标模型倒数第二层 hidden state**，输出 = **预测的倒数第二层 feature**，再经目标模型共享的 **LM head** 出 token 概率：

```python
# EAGLE/eagle/model/cnets.py —— EAGLE-1 草稿层（cnet）的核心 forward
class LlamaDecoderLayeremb(nn.Module):
    """EAGLE draft layer: 输入是 token embedding 与上一层 hidden state 的拼接"""
    def __init__(self, config, last=True):
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_emb, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        # ★ 核心：把"下一个 token 的 embedding"与"当前 hidden state"拼接
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        # 过一个 attention + MLP（与正常 decoder layer 相同结构）
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache)
        hidden_states = residual + hidden_states
        ...
        return outputs
```

**这段代码关键在哪**：`torch.cat((input_emb, hidden_states), dim=-1)` 是 EAGLE 的全部秘密——**草稿层把"已知的下一个 token 的 embedding"和"当前预测的 feature"拼接**，attention+MLP 输出预测的 feature；由于目标模型的 LM head 是共享的（`EaModel` 里直接复用），草稿层预测的 feature 过 LM head 即得 token 分布。训练时用目标模型的真实 feature 做 teacher forcing（下个 token 的 embedding + 真实 feature → 预测 feature，损失 = 预测 feature 与真实 feature 的 L2 + token 交叉熵）。

- **为什么接受率高**：草稿不是在"猜 token"，而是在"外推 feature"——feature 空间比 token 空间平滑得多，外推误差小；
- **为什么成本低**：cnet 只有 1 层（可扩到 2-3 层），c ≈ 主模型单层成本 ≈ 3%~10%；
- **训练成本**：1~2 天（8×RTX 3090 可训），数据 = 目标模型在普通语料上的输出 feature（无需强化学习）。

### 3. 具体数值样例

- 8B/32 层模型：cnet 1 层 → 草稿单步成本 c ≈ 1/32 ≈ 0.03（忽略 embedding/LM head 时）；
- 官方数据（Vicuna-13B）：EAGLE-1 比 vanilla 快 **2.7~3×**、比 Medusa 快 1.6×；gpt-fast 上 2×；
- 关键对比：k=3、c=0.1 时，α=0.7 的 EAGLE 加速 1.75×（第 3 点演算），而独立小模型 α=0.4 时只有 $E=(1-0.4^4)/0.6=1.65$，$S=1.65/(1+0.3)=1.27$×——**同样的草稿预算，EAGLE 的接受率优势直接转化为加速优势**。

> 面试一句话总结：**EAGLE-1 用"目标模型倒数第二层 feature 外推"代替独立小模型：草稿层输入 = 下个 token 的 embedding ⊕ 当前 feature，输出预测 feature 再过共享 LM head——feature 空间比 token 空间平滑，接受率高（α≈0.7），而草稿只有 1~2 层（c≈0.03~0.1），质量/成本比全面优于独立小模型与 Medusa。**

---

## 6. EAGLE-2：置信度感知的动态树形草稿

### 1. 现有问题

- EAGLE-1 是固定深度/宽度的树：**草稿长度与接受率不匹配**——接受率高时浪费（可以多猜），接受率低时浪费（猜了也白猜）；
- 树形草稿的形状（每层的分支数）应该**随"当前 token 的确定性"动态调整**：模型很确定的下一个 token 给宽分支，不确定的给窄分支。

### 2. 方法论

EAGLE-2（arXiv:2406.16858）用**置信度分数**（草稿模型输出的 top-1 概率）近似接受率，动态调整每层分支数：

- 草稿阶段每层选出 **top-k 候选**（不止 top-1），k 由该层的置信度决定：置信度高 → 少分支（或只 1 个），置信度低 → 多分支覆盖更多可能；
- 验证阶段仍是一次前向验证整棵树，贪心选最长接受路径；
- 收益：**同样的树预算（验证成本）下，期望接受长度更长**——把"验证算力"花在最可能被接受的分支上。

### 3. 具体数值样例

- 某一层 top-1 概率 0.9（很确定）→ 只保留 1 个分支；某层 top-1 概率 0.4（不确定）→ 展开 4 个分支；
- 对比固定树：固定 3 层 × 4 宽 = 13 个节点；EAGLE-2 动态树平均 ~8 个节点（预算内自适应），验证 batch 更小、期望接受长度反而更长；
- 官方数据：EAGLE-2 比 EAGLE-1 快 1.4×，比 vanilla 快 4×（13B）。

> 面试一句话总结：**EAGLE-2 把树形草稿做成"置信度感知"：用草稿的 top-1 概率近似接受率，确定性高的层少分支、不确定性高的层多分支——同样的验证预算下期望接受长度更长，比 EAGLE-1 再快 1.4×。**

---

## 7. EAGLE-3：free-style 训练 + 多层特征融合

### 1. 现有问题

- EAGLE-1/2 训练时要求草稿预测的 feature **等于目标模型的真实 feature**（teacher forcing 硬约束）——推理时草稿的输入 feature 是"自己预测的"，与训练时的真实 feature 有**分布偏移（train-inference mismatch）**；
- 只用"倒数第二层"一个 feature：LLM 不同层携带不同粒度的语义（低层=语法/词形，中层=语义，高层=下一个 token 预测），只用高层丢失信息。

### 2. 方法论

EAGLE-3（arXiv:2503.01840）两个改动：

1. **free-style（去除 feature 预测约束）**：训练时用 **training-time testing** 模拟推理——草稿自回归时把"上一步预测的 feature"（而非真实 feature）作为下一步输入，让草稿**学会在自身预测的 feature 流上工作**；损失直接是**token 交叉熵**（不再要求 feature 精确相等）。推理时草稿与训练一致，没有分布偏移；
2. **多层特征融合**：草稿输入不只用倒数第二层，而是**融合低/中/高层的 hidden state**（vLLM 集成里对应 `eagle3_utils.py` 的 `set_eagle3_aux_hidden_state_layers`——EAGLE-3 需要主模型在指定层**额外导出 aux hidden state**）：

```python
# vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py
def set_eagle3_aux_hidden_state_layers(layers):
    """EAGLE-3 需要在目标模型里额外指定若干层导出 aux hidden state，
    供草稿网络融合低/中/高层语义特征"""

def get_eagle3_aux_layers_from_config(config):
    """从模型配置读取 EAGLE-3 的 aux 层设置"""
```

**这段代码关键在哪**：EAGLE-3 不是"只读倒数第二层"，而是让主模型在**多个指定层**导出 hidden state（aux layers），草稿融合这些多粒度特征——低层语法 + 高层语义共同决定下一个 token；同时 free-style 训练消除 train/inference 偏移，接受率在长草稿下不掉。

### 3. 具体数值样例

- 官方（Vicuna-13B）：EAGLE-3 比 vanilla 快 **5.6×**、比 EAGLE-1 快 1.8×；k 可拉到 4~6 仍保持高接受率；
- 本项目（Qwen3-8B + EAGLE-3 drafter，vLLM）：k=3 时吞吐 +41.7%（第 10 点），与官方"13B 上 5.6×"的量级差异主要来自**模型规模、batch 大小、硬件与实现**——本项目是单卡 4090、offline 批处理，投机收益被 batch 分摊了一部分；
- EAGLE-3 的实际部署需要：① 训练好的 drafter 权重（`Qwen3-8B-speculator.eagle3`）；② 推理框架支持导出 aux hidden state（vLLM `speculative_config` 里 `"method": "eagle3"`）。

> 面试一句话总结：**EAGLE-3 用 free-style 训练（training-time testing 让草稿在"自己预测的 feature 流"上自洽，消除 train/inference 偏移）+ 多层特征融合（主模型在多个指定层导出 aux hidden state），接受率与草稿长度上限都显著提升——官方 13B 上比 vanilla 快 5.6×，是当前 EAGLE 系列最新最强版本。**

---

# 三、推理框架集成（vLLM / verl）

## 8. vLLM 的投机解码架构：Proposer + Rejection Sampler

### 1. 现有问题

- 投机解码要侵入 decode 主循环：草稿步、验证步、接受/拒绝、与 CUDA Graph、调度、KV 管理协同——框架必须把它做成"一等公民"；
- 多种草稿方案（EAGLE/Medusa/ngram/draft model）要统一抽象，且要能在**批量多请求**下工作（不同请求各自接受长度不同 → 验证 batch 是 ragged 的）。

### 2. 方法论（vLLM main 分支，`vllm/v1/spec_decode/`）

vLLM 把投机解码拆成 **Proposer（草稿侧）+ Rejection Sampler（验证侧）**：

```text
vllm/v1/spec_decode/
├── llm_base_proposer.py       # 无投机时的基线 proposer
├── draft_model.py             # 独立小模型草稿（DraftModelProposer）
├── eagle.py                   # EAGLE 系草稿（EagleProposer）
├── medusa.py                  # Medusa 并行头
├── ngram_proposer.py          # n-gram 草稿
├── gemma4.py / dflash.py / suffix_decoding.py / step3p5.py  # 更多方案
├── metadata.py                # SpecDecodeMetadata：每请求的草稿/验证信息
└── metrics.py                 # spec_decode 指标（num_drafts / num_accepted 等）

vllm/v1/worker/gpu/spec_decode/
├── rejection_sampler.py       # RejectionSampler：验证 + 接受/拒绝 + 残差重采样
└── eagle/eagle3_utils.py      # EAGLE-3 aux hidden state 导出
```

**EagleProposer**（`eagle.py`）：

```python
# vllm/v1/spec_decode/eagle.py
class EagleProposer(SpecDecodeBaseProposer):
    """EAGLE 草稿：用主模型中间 feature + 共享 LM head 生成草稿 token 树"""
    def __init__(self, ...):
        ...
```

**RejectionSampler**（`rejection_sampler.py`）的核心验证逻辑：

```python
# vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
class RejectionSampler:
    def __init__(self, ...):
        ...
    def _verify(self, ...):
        """对每个草稿位置：草稿 token == 大模型 argmax 则接受，否则截断"""
    def _verify_in_chunks(self, ...):
        """分块验证（省显存/适配 CUDA graph 尺寸）"""
    def __call__(self, ...):
        """入口：输入草稿 tokens + 大模型 logprobs → 输出接受后的 tokens"""
```

- **与 CUDA Graph 协同**：验证步的大模型前向是**固定形状的 CUDA Graph**（batch = 草稿树展开后的 token 数，按最大形状捕获），`_verify_in_chunks` 分块让验证适配 graph 尺寸；
- **配置入口**（项目用的 vLLM 0.11.x 的 offline API）：`LLM(..., speculative_config={"method": "eagle3", "model": draft_path, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1})`；
- **指标**：`vllm:spec_decode_num_drafts` / `vllm:spec_decode_num_accepted_tokens`，接受长度 = 1 + accepted/drafts（项目 A/B 脚本直接读这两个指标算 `mean_accept_len`）。

### 3. 具体数值样例

- 批 16 个请求、每请求草稿 3：验证 batch = 16 请求 × 树展开 ≤ 16×13 = 208 token（若树 13 节点），走固定 CUDA graph（如 256 token 档）；
- 各请求接受长度不同（2/3/1/...），RejectionSampler 输出 ragged 的接受结果，下一轮各请求从自己的新位置继续草稿；
- 指标实测（项目 A/B 脚本）：`num_drafts` / `num_accepted` 来自 `llm.get_metrics()`，`mean_accept_len = 1 + num_accepted/num_drafts`——这是判断"草稿质量 + 集成是否正常"的第一手指标。

> 面试一句话总结：**vLLM 把投机解码抽象成 Proposer（草稿侧：EAGLE/Medusa/ngram/draft-model 统一接口）+ RejectionSampler（验证侧：argmax 接受 + 分块验证适配 CUDA Graph），配置入口是 speculative_config 一个 dict，指标 num_drafts/num_accepted 可直接观测接受率——投机解码在 vLLM 里是一等公民。**

---

## 9. 项目集成：verl + vLLM 0.11.1 + EAGLE-3（含踩过的坑）

### 1. 现有问题

- 训练侧的投机解码比纯推理难：verl 的 rollout 引擎（vLLM）由训练框架拉起，投机配置要**穿透 verl 的 engine_kwargs** 传进 vLLM；
- 三个工程坑（本项目全部踩过并解决）：
  1. **vLLM 0.11.1 的 EAGLE + dp>1 死锁**；
  2. **LoRA 训练 × 投机解码互斥**（草稿/主模型权重同步语义冲突）；
  3. **独立 drafter 不参与权重同步**：训练中主模型权重每轮更新，drafter 保持静态 → 接受率随训练漂移。

### 2. 方法论（集成代码）

**① 配置穿透**：verl 启动参数里把 speculative_config 以 JSON 字符串塞进 `actor_rollout_ref.rollout.engine_kwargs.vllm`（`run_grpo_dual_async_mooncake_ucloud.sh`）：

```bash
# uniagent-lighting/scripts/run_grpo_dual_async_mooncake_ucloud.sh（摘录）
SPEC_ON=${SPEC_ON:-1}
SPEC_DRAFT=${SPEC_DRAFT:-/home/ubuntu/models/Qwen3-8B-speculator.eagle3}
SPEC_TOKENS=${SPEC_TOKENS:-3}     # num_speculative_tokens（官方编码基准 k=3）
...
if [ "$SPEC_ON" = "1" ]; then
  EXTRA_ARGS+=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{\"method\": \"eagle3\", \"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": $SPEC_TOKENS, \"draft_tensor_parallel_size\": 1}'"
  )
fi
```

**② 三个坑的解法**：
- **dp>1 死锁**：`run_grpo_dual_async_mooncake_ucloud.sh` 注释明确——"投机解码 EAGLE-3 开启（SPEC_ON=1）：**独立引擎 dp=1 单节点**，避开 vLLM 0.11.1 EAGLE+dp>1 死锁，恢复单机 spec 的吞吐收益（+41.7%）"；双机全异步架构里，投机引擎只放一个 dp 副本（吞吐收益集中在 decode 侧）；
- **LoRA×SD 互斥**：`run_grpo_humanevalfix_ucloud.sh` 注释"SPEC_ON=1（EAGLE-3 drafter，需 LORA_MERGE=1）"——LoRA 训练时 drafter 无法感知 adapter，必须先把 LoRA **merge 成全量权重**再同步给引擎（`spec_train_run.sh` 的 LORA_MERGE=1）；
- **drafter 静态漂移**：`run_grpo_humanevalfix_ucloud.sh` 注释——"独立 drafter（非 MTP）不走权重同步（verl `_iter_all_models` 只同步 actor+MTP）；drafter 保持静态，训练中 LoRA 漂移会导致接受率下降（**step1 vs step5 各记录一次**）"——所以训练脚本专门在 step1/step5 记录接受率，量化漂移。

### 3. 具体数值样例

- 配置：Qwen3-8B 主模型 + `Qwen3-8B-speculator.eagle3` drafter，`num_speculative_tokens=3`，`draft_tensor_parallel_size=1`，`temperature=0.8`、`top_p=0.95`、`max_tokens=512`、`n=4`（A/B 脚本参数）；
- 训练链路（正式）：双机全异步 GRPO + Mooncake + EAGLE-3 + 白盒，25 步 7:11:40，评测 **83.23%（134/161）**，与 baseline 83.2% 持平——**投机解码加速了训练吞吐，未引入训练质量损失**；
- 接受率监控：A/B 脚本 `mean_accept_len = 1 + num_accepted/num_drafts`；训练中 step1 与 step5 各记录一次，对比 LoRA 漂移前后的接受率变化。

> 面试一句话总结：**verl + vLLM + EAGLE-3 的集成 = 一个 JSON speculative_config 穿透 engine_kwargs 即可，但有三个工程坑：EAGLE+dp>1 死锁（投机引擎 dp=1 单节点）、LoRA 与投机互斥（先 merge 全量权重）、独立 drafter 不随训练同步（接受率随 LoRA 漂移，step1/step5 各记一次量化）——都解决后 25 步训练 83.23% 与 baseline 持平，加速未损质量。**

---

## 10. 实测数据与自洽性演算：+41.7% 是怎么来的

### 1. 现有问题

- 简历数字（199→282 tok/s、+41.7%、-39.5%）要能**自圆其说**——面试官会问"为什么是这个数"；
- 两个口径（吞吐 vs 单 token 延迟）不互逆，需要解释清楚。

### 2. 方法论（数据 + 反推）

A/B 实验（`spec_bench_ab.py`）：同批 HumanEvalFix prompt（32 prompts × n=4 = 128 响应）、同采样参数（temperature=0.8, top_p=0.95, max_tokens=512），只切换 `speculative_config=None → {"method":"eagle3", ...}`：

| 指标 | spec off | spec on | 变化 |
|---|---|---|---|
| output tok/s | 199 | 282 | **+41.7%** |
| E2EL p50（每 token 延迟） | — | — | **-39.5%** |
| 官方评估（25 步训练） | 83.2%（baseline） | 83.23% | 持平 |

**自洽性反推**（k=3、草稿成本比 c≈0.15 假设）：
- 吞吐比 1.417 → 期望每步输出 $E = 1.417 \times (1+3\times0.15) = 2.055$ → 由 $(1-\alpha^4)/(1-\alpha)=2.055$ 解得接受率 $\alpha \approx 0.56$；
- 延迟比 0.605（-39.5%）→ $E = (1+0.45)/0.605 = 2.40$ → $\alpha \approx 0.66$；
- 两个口径反推的 $\alpha \in [0.56, 0.66]$，落在 EAGLE-3 的合理接受率区间——**数据自洽**；吞吐比延迟保守是因为吞吐口径含调度/排队开销（128 响应并发）。

### 3. 具体数值样例（一句话讲清"为什么 +41.7% 而不是 1.75×"）

- 理论加速 1.75×（第 3 点，α=0.7、单请求、c=0.15）是**单请求上限**；
- 实测 +41.7%（1.417×）低一截，原因：① batch 4 并发分摊了权重读取（投机收益被 batch 稀释）；② 4090 单卡 + offline 批处理，草稿/验证的 kernel 开销占比高；③ 实际接受率 α≈0.56~0.66 低于理想 0.7；
- 面试话术：**"理论单请求上限约 1.75×，批处理下被分摊到 1.42×；用吞吐/延迟两个口径反推接受率都在 0.56~0.66，数据自洽；训练评测 83.23% 与 baseline 持平，说明投机只换吞吐、不损质量。"**

> 面试一句话总结：**项目实测 EAGLE-3 把生成吞吐从 199 提到 282 tok/s（+41.7%）、单 token 延迟降 39.5%；用吞吐和延迟两个口径分别反推接受率 α≈0.56 与 0.66，数据自洽；低于理论 1.75× 是因为 batch 分摊与 4090 单卡开销——且 25 步训练 83.23% 与 baseline 持平，加速未损质量。**

---

# 四、最新进展：Block 并行投机（DFlash / DSpark，2026 DeepSeek V4 系）

> 素材：vLLM speculators 官方文档（`dflash.md` / `dspark.md`）、DFlash（arXiv:2602.06036，Z Lab）、DSpark（arXiv:2607.05147，DeepSeek）、本地 vLLM `vllm/v1/spec_decode/dflash.py` + `qwen3_dflash` 模型、本地 sglang `srt/speculative/` 的 `dflash_worker_v2.py` 与 `dspark_components/`。前 10 点（EAGLE 系列 + vLLM 集成）是"自回归草稿"路线；本部分是 2026 年的**"块级并行草稿"路线**——理解它能答"投机解码下一步往哪走"。

## 11. DFlash：diffusion-LLM 块级草稿（非因果 mask token + 对 target hidden states 做 cross-attention）

### 1. 现有问题

- EAGLE-1/2/3 的草稿是**自回归**的：draft 网络逐 token 出草稿（即便 EAGLE-2/3 用树形并行化验证，**草稿生成本身仍是逐 token 串行**）——草稿成本 = 逐 token 成本 $\times$ 草稿数，k 拉长草稿开销线性涨；
- 能不能"**一次前向直接预测一整块 token**"？Medusa 试过并行头，但它只条件于真实前缀、无自回归、无 target 中间特征，长草稿质量衰减快（前文第 3 点）——需要一个"既块级并行、又吃到 target 全部上下文特征"的草稿。

### 2. 方法论

**DFlash**（"Block Diffusion for Flash Speculative Decoding"，Z Lab，arXiv:2602.06036）用**小 diffusion-LLM 草稿模型**：单次前向预测一个 token 块，条件 = **target 模型的 hidden states**（EAGLE 系思想 + 块级并行的结合）：

```text
① target 前向一次：产出 fused context features（target hidden states）
   （vLLM：pass_hidden_states_to_model=True，把 target 中间层 hidden 传给 draft）
② 组装 draft 输入：context hidden states ⊕ mask token embeddings（占位）
   （vLLM 注释："Only next_token_ids and mask tokens are query tokens,
     all other context is K/V"——query 只有 1+block 个，其余全是 K/V）
③ draft layers（Qwen3 式 transformer 层，但用【非因果 bidirectional attention】
   + KV cache）：每个 query 同时 attend verifier hidden states 与 mask token
   embeddings → 一次前向并行产出整块 token 的 logits
   （vLLM 的 use_non_causal / dflash_has_any_non_causal 检测模型是否非因果）
④ 输出经 target 共享的 LM head → 词表 logits → 采样整块草稿
```

**Anchor Point 机制**（块验证怎么组织）：

```text
1. Select anchors：在序列里选锚点位置
2. Predict blocks：每个锚点单次前向预测一个 token 块
3. Verify blocks：target 并行验证各块
4. Accept：取最长有效前缀
```

**`sample_from_anchor` 开关**（两种采样语义，DFlash 与 DSpark 的关键分界）：
- `False`（DFlash 默认）：**锚点是 bonus token**，只有 mask token 位预测 → 每块产出 `block_size - 1` 个投机 token（slot 0 不训练）；
- `True`（DSpark 默认）：**锚点与所有 mask 位都采样预测未来** → 每块产出 `block_size` 个投机 token（与 DSpark 论文一致）。

**为什么快**：草稿从"k 次串行前向"变成"1 次块级前向"（块内并行、非因果 attention 一次算完），同步请求下官方称比 EAGLE-3 大 **2-3×** 的加速；草稿模型是 Qwen3 式层但**可配任意 target**（verifier 无关）。

**代码锚点**（vLLM `dflash.py` 的 `DFlashProposer`）：

```python
# vllm/v1/spec_decode/dflash.py —— DFlash 的关键设计点
pass_hidden_states_to_model=True,      # target hidden states 传给 draft（cross-attn 的 context K/V）
# "Only next_token_ids and mask tokens are query tokens, all other context is K/V"
self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
# DFlash 用 input embeddings 嵌入 mask token（占位 query）
# 非因果 attention：dflash_has_any_non_causal(hf_config) → use_non_causal
```

### 3. 具体数值样例

- `block_size = 8`、`max_anchors = 4`：每锚点一次前向出 7 个投机 token（sample_from_anchor=False），4 锚点共 4 条候选路径并行，target 一次验证（树形）；
- 对比 EAGLE-3（k=3 自回归草稿）：DFlash 单块 8 token 的草稿生成 = 1 次 draft 前向 vs EAGLE-3 的 3 次串行 draft 前向——**草稿阶段延迟从 $O(k)$ 降为 $O(1)$（块内并行）**；
- 代价：非因果 attention 的 KV cache 更大、draft 层要容纳"mask query + context"同时 attend，显存略增；训练要用 diffusion-LLM 式的块目标（一次性学整块）。

> 面试一句话总结：**DFlash = 块级并行的 EAGLE：小 diffusion-LLM 草稿把 target hidden states 当 context K/V、mask token 占位当 query，用非因果 attention 一次前向预测一整块 token（anchor 机制 + 最长有效前缀接受），草稿成本从 O(k) 串行降到 O(1) 块级，同步请求比 EAGLE-3 快 2-3×。**

---

## 12. DSpark：Markov head + Confidence head（置信度调度的半自回归投机）

### 1. 现有问题

- DFlash 的纯块并行有个内在缺陷：**块内 token 之间没有依赖**（每个 mask 位只条件于 target 上下文与锚点，不条件于"块内前一个预测的 token"）——所以块越长、越靠后的位置接受率越低（接受率沿块衰减）；
- 另一个问题：**不知道"这块值得验证到第几个位置"**——验证太浅浪费草稿，太深浪费 target 前向；
- DeepSeek 在 V4 上给出的答案：给 DFlash 加两个头，把"块内依赖"和"验证深度"都学出来。

### 2. 方法论

**DSpark**（"Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"，DeepSeek，arXiv:2607.05147）在 DFlash 的 block 并行主干上**加两个头**（draft 模型继承 DFlash，架构与训练其余不变，可配任意 verifier）：

```text
① Markov head（恢复块内依赖 → "半自回归"）：
   低秩 logit bias B = W1 @ W2：
     W1 把"块内前一个 token"（verifier 词表）嵌入到 markov_rank=256 维
     W2 投影回 draft 词表 → 加到 DFlash logits 上
   三变体：vanilla（只由前 token）/ gated（按 backbone hidden 门控）/ rnn（块内跨位置递归）
   （--markov-rank 0 关闭 = 纯 DFlash；块内每个位置现在"知道"前一个预测的 token）
② Confidence head（预测每位置接受概率 → 调度验证深度）：
   线性头从 backbone hidden state（+ markov 前 token embedding，
   --confidence-head-with-markov）预测该位置将被 target 接受的概率；
   用 BCE 训练（--confidence-head-alpha 加权）
③ 置信度调度：接受概率低的块位置 = "再验证也白搭" → 动态决定
   这块验证到第几个位置（对应 sglang 的 verify-window/topk 规划）
```

- **为什么"半自回归"**：块内用 Markov bias 只恢复"前一个 token"的一阶依赖（不是完整自回归）——成本远低于逐 token 自回归，但已足够抑制块末端接受率衰减；
- **`sample_from_anchor=True`**（DSpark 默认）：锚点与 mask 都预测 → 每块 `block_size` 个投机 token；
- **服务端接入**：vLLM `--speculative-config '{"method": "dspark", ...}'`；sglang 为 DeepSeek-V4 加了 `DSparkWorkerV2`（`srt/speculative/dspark_components/`：`dspark_planner.py` 的 VerifyWindow/topk 规划、`dspark_verify.py` 的 accept_and_finalize、`dspark_kv_inject.py` KV 注入、`dspark_disaggregation.py` PD 分离支持、`dspark_sps.py`/`dspark_sts.py` 校准表）；llama.cpp、NVIDIA NeMo AutoModel 也有实现。

**代码锚点**（sglang DSpark 组件——真实工程面）：

```python
# sglang/srt/speculative/dspark_components/dspark_planner.py
#   VerifyWindow / ScheduleVerifyLensTopk / compute_sort_survival —— 按置信度规划验证窗口
# dspark_worker_v2.py
class DSparkWorkerV2(BaseSpecWorker):
    def carries_confidence(self) -> bool: ...   # 是否携带置信度（调度依据）
    def forward_batch_generation(self, ...): ... # draft 一次出整块
# dspark_verify.py
class TargetVerifyExecutor:
    def accept_and_finalize(self, ...): ...     # 按置信度截断的块验证
# dspark_disaggregation.py
def build_dspark_disagg_draft_input(...): ...   # PD 分离下的草稿（配合 Mooncake 类 KV 传输）
```

### 3. 具体数值样例

- 一个 block_size=8 的块，Markov bias 让第 $i$ 位条件于第 $i-1$ 位：末位接受率从纯并行的 $\approx \alpha^8$ 恢复到接近 $\alpha$（每步条件接受率）——**块内依赖是"接受率不随深度衰减"的关键**；
- Confidence head 输出每位置接受概率，如 $[0.95, 0.9, 0.85, 0.6, 0.3, 0.1, \dots]$ → 调度器决定**只验证前 4 个位置**（第 4 位后概率 < 阈值），省下 target 对"注定被拒位置"的前向；
- 训练：DFlash 块目标 + Markov bias（交叉熵）+ confidence head（BCE，$\alpha$ 加权）多任务联合。

> 面试一句话总结：**DSpark = DFlash + 两个头：低秩 Markov head（$B=W_1W_2$ 把块内前一个 token 的 bias 加到草稿 logits，恢复"半自回归"依赖、抑制块末端接受率衰减）+ confidence head（预测每位置接受概率、用置信度动态调度验证深度）——把"块级并行"从"赌整块"变成"知道该信多远"，DeepSeek V4 生产使用。**

---

## 13. 定位与速答：DFlash/DSpark vs EAGLE vs Medusa vs MTP

### 1. 现有问题

新方法很多，面试要能一句话定位彼此关系。

### 2. 方法论（草稿路线的三代数 + 对比表）

| 方法 | 草稿形态 | 是否自回归 | 是否用 target 特征 | 加速特点 |
|---|---|---|---|---|
| 独立 Draft Model | 单 token | 是 | 否 | 简单但 α 低 |
| EAGLE-1/2/3 | 单 token / 树形 | 是（浅层） | 是（倒数第二层 / 多层 feature） | α 高、c 低（本项目方案） |
| Medusa | 并行头 | 否 | 否（只条件真前缀） | k≥2 质量衰减 |
| **DFlash** | **token 块（非因果）** | **否（块内并行）** | **是（target hidden 当 K/V）** | **草稿 O(1)、比 EAGLE-3 快 2-3×** |
| **DSpark** | **token 块 + Markov bias** | **半自回归（一阶）** | **是（+ confidence head）** | **块末端 α 不掉 + 置信度调度** |
| MTP（DeepSeek） | 训练时多头 | 否 | 主模型内部 | 无额外模型、随主模型同步 |

**三个必答点**：
1. **EAGLE vs DFlash/DSpark**：EAGLE 是"自回归草稿 + target feature"，草稿生成串行；DFlash/DSpark 是"块级并行草稿 + target feature + 非因果 attention"，草稿生成一次出块——**同一"target feature 条件化"思想，一个串行一个并行**；
2. **Medusa vs DFlash**：都是并行出多 token，但 Medusa 只条件真实前缀、无 target 特征、无块间依赖；DFlash 的每个 query 同时 attend target hidden states 与 mask token（信息量大得多）；
3. **DSpark 与 EAGLE-2 的"置信度"区别**：EAGLE-2 用 draft 输出的 top-1 概率近似接受率调树宽；DSpark 用**专门训练的 confidence head** 预测逐位置接受概率来调度验证深度——一个是启发式、一个是学出来的。

### 3. 具体数值样例（定位我们的项目）

- 本项目用 EAGLE-3（k=3）：自回归浅层草稿，实测 +41.7%（第 10 点）——是"串行草稿"路线的成熟选择；
- 若换 DSpark（block_size=8）：草稿从 3 次串行变 1 次块前向 + Markov bias 保末端接受率，单请求理论加速窗口更大，但需要训练新 speculator 且同步请求收益才明显（batch 大时被分摊）——**选型看场景：兼容/成熟选 EAGLE-3，极限单请求吞吐/DeepSeek V4 系选 DSpark**。

> 面试一句话总结：**草稿三代数：单 token 自回归（Draft/EAGLE）→ 树形（EAGLE-2/3）→ 块级并行（DFlash/DSpark）；EAGLE 与 DFlash 共用"target feature 条件化"，区别在草稿串行 vs 块并行；DSpark 又用 Markov head（半自回归）与 confidence head（学出来的验证深度调度）补上块并行的两个短板——Medusa 无 target 特征、MTP 在模型内部，各有定位。**

---

# 五、面试问答与进阶

## 14. 高频追问：何时投机无效 / MTP vs EAGLE / 与 CUDA Graph

### 1. 现有问题

面试官常见的延伸问题，先想清楚再答。

### 2. 方法论（四个必答点）

**① 何时投机解码无效 / 收益变小？**
- batch 已经很大时：权重读取被多请求分摊，decode 不再是纯访存瓶颈 → 投机收益趋近于零甚至为负（草稿开销 > 收益）；
- 草稿接受率低时：如主题分布与训练分布差异大、采样温度极高（分布平坦，草稿猜不中）→ 看 `num_accepted/num_drafts` 指标，接受率 < 0.4 基本无收益；
- 已用 CUDA Graph + 大 batch 优化到接近算力边界时。

**② EAGLE（独立 drafter） vs MTP（主模型多头）？**
- MTP（如 DeepSeek-V3 的 Multi-Token Prediction）：主模型内部多个**预测头**，训练时与主模型一起优化，权重**随主模型同步**——verl 的 `_iter_all_models` 同步 actor+MTP 就是这个原因；优点是无额外模型、训练一体化；缺点是草稿质量受主模型容量限制；
- EAGLE：独立 drafter（可单独训练/换），质量上限更高，但**权重不随主模型同步**（本项目"drafter 保持静态"的根因）——训练中主模型漂移会让接受率下降，LoRA 场景尤其明显；
- 面试结论：**训练侧（verl）选 MTP 更省心（同步天然），推理侧选 EAGLE 更灵活；本项目用 EAGLE-3 是因为它是已训练好的独立 drafter、开箱即用。**

**③ 与 CUDA Graph 的关系？**
- 验证步必须走 CUDA Graph（固定形状）才有收益；草稿步通常 eager（浅层、形状多变）；`_verify_in_chunks` 把验证分块以适配 graph 尺寸；
- vLLM v0.25+（MRv2）支持 **dynamic spec decode + full CUDA graphs**——投机解码与 CUDA Graph 深度耦合，树形草稿的验证 batch 形状动态选择 graph 档位。

**④ 本项目为什么不影响训练质量？**
- 投机解码是"分布保持"的（第 2 点证明），采样分布与正常解码一致 → on-policy RL 训练的数据分布不受影响 → 83.23% 与 baseline 83.2% 持平是**理论保证 + 实证一致**；
- 前提：拒绝采样实现正确（argmax 接受 + 残差重采样），且验证步数值精度与正常解码一致（本项目 A/B 用同一 `rollout_autocast` 路径）。

### 3. 具体数值样例

- batch=1、单请求长生成（如 2048 token）：EAGLE-3 收益最大（理论 1.75×）；batch=128：收益趋近 1.0×（权重读取已被分摊）；
- 温度 0.8 vs 0.0：温度越高草稿接受率越低（分布越平），本项目训练用 temperature=0.8 仍拿到 0.56~0.66 接受率，说明 EAGLE-3 的 feature 外推在高温下也稳定；
- vLLM 0.25+ 的动态投机：`speculative_config` 支持按请求/按步动态调 `num_speculative_tokens`（接受率高自动加长），是"自适应投机"的框架级实现（SGLang 也有 `adaptive_spec_params`）。

> 面试一句话总结：**投机解码在"大 batch 或低接受率"时无效（权重读取被分摊/草稿猜不中，看 num_accepted/num_drafts 判断）；EAGLE 是独立 drafter（不随主模型同步，训练侧需注意漂移）而 MTP 是主模型多头（天然同步）；验证步必须走 CUDA Graph；分布保持的拒绝采样保证 on-policy 训练数据分布不变——这就是项目投机加速 41.7% 而训练质量 83.23% 持平的理论依据。**

---

# 附：速查表

| 概念 | 一句话 | 关键文件 |
|---|---|---|
| 期望输出 token 数 | $E=(1-\alpha^{k+1})/(1-\alpha)$ | — |
| 加速比 | $S=E/(1+kc)$，c=草稿成本比 | — |
| EAGLE-1 | feature 外推：token embedding ⊕ hidden state → 预测 feature → 共享 LM head | `EAGLE/eagle/model/cnets.py` |
| EAGLE-2 | 置信度感知动态树 | EAGLE 仓库 |
| EAGLE-3 | free-style 训练 + 多层 feature 融合（aux hidden state） | `EAGLE/eagle/modeling_eagle.py`、`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` |
| DFlash | 块级 diffusion-LLM 草稿：非因果 mask token query + cross-attn target hidden states，一次前向出整块 | `vllm/v1/spec_decode/dflash.py`、`qwen3_dflash` 模型、sglang `dflash_worker_v2.py`；vLLM speculators `dflash.md` |
| DSpark | DFlash + Markov head（低秩 bias 半自回归）+ confidence head（置信度调度验证深度） | sglang `srt/speculative/dspark_components/`、vLLM `"method": "dspark"`；arXiv:2607.05147 |
| 树形草稿 | 多分支一次验证，贪心选最长接受前缀 | `EAGLE/eagle/modeling_eagle.py`（node/Tree） |
| vLLM 抽象 | Proposer（eagle.py/medusa.py/ngram_proposer.py）+ RejectionSampler | `vllm/v1/spec_decode/`、`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` |
| 项目集成 | verl engine_kwargs → speculative_config JSON；dp=1 避死锁；LoRA 需 merge；drafter 静态 | `uniagent-lighting/scripts/run_grpo_dual_async_mooncake_ucloud.sh`、`spec_bench_ab.py` |
| 实测 | 199→282 tok/s（+41.7%）、E2EL -39.5%、α≈0.56~0.66、83.23% 持平 | 简历项目（一）亮点 2 |

**面试金句**："投机解码把 decode 的串行步数从 k+1 降到 1 次验证；EAGLE 系列用 feature 外推拿到高接受率 + 低成本；本项目 EAGLE-3 在 4090 单卡批处理下 +41.7% 吞吐且训练质量持平——分布保持的拒绝采样保证了 on-policy 训练数据不变。"
