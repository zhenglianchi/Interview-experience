# Agentic RL 轨迹拼接（Trajectory Stitching）完全指南

> **一句话知识框架**：单轮 RL 里"一条 response = 一个训练样本"天经地义；到了 agentic RL，一条轨迹里**混杂了三种 token**——模型的输出（要训练）、工具/环境的返回（不能训练）、以及 harness 自己插入的模板文本（也不能训练）。于是"轨迹拼接"要解决三件事：**哪些 token 进 loss**（loss mask）、**多轮对话怎么切成样本**（per-call 独立成样本 vs 前缀合并成长轨迹）、**reward 怎么摊到各轮**（credit assignment）。这三件事决定了训练信号是否 "token-faithful"——只要拼接错一步，你以为在优化的序列就不是模型真实采样过的序列。

> 素材来源：`agent-lightning/agentlightning/adapter/triplet.py`（Agent Lightning 官方具现，1028 行，本文件直接引用）；Polar 论文（arXiv:2605.24220，NVIDIA，`github.com/NVIDIA-NeMo/ProRL-Agent-Server`）；vLLM 官方博客《No More Retokenization Drift》（2025-10-22）；`internship.md`（verl/TQ 的 triplet 构造）。配套阅读：`Training-Inference-Mismatch.md`（token 保真的另一半）、`Uniagent.md`、`Agent-lighting.md`。

---

## 1. 为什么多轮轨迹不能直接喂给 PPO/GRPO

### 1. 现有问题

一条典型的 agent 轨迹长这样（以 coding agent 为例）：

```
[system prompt]  [user: 修这个 bug]
  ↓ assistant 输出（模型采样）
  "我需要先看文件" + tool_call(read_file, path="a.py")
  ↓ tool 返回（环境产生，不是模型生成）
  "<file content ... 3000 tokens ...>"
  ↓ assistant 输出（模型采样）
  "问题在 42 行" + tool_call(edit_file, ...)
  ↓ tool 返回
  "OK"
  ↓ assistant 输出（模型采样）
  "已修复"
```

把它当成一条序列直接做 PPO，会踩三个坑：

1. **工具返回的 token 被当成模型输出**：`<file content>` 那 3000 个 token 不是模型生成的，如果对它们算 `log π_θ` 并施加 advantage，等于在训练模型"预测文件内容"——纯噪声，甚至会把模型推向复读工具输出。
2. **advice 错位**：如果整条轨迹只有一个 outcome reward（任务成功=1），那么"第 2 轮误读了文件"和"第 5 轮正确修复"会拿到**同一个 advantage**，模型无法区分哪一步做对了。
3. **长度爆炸与 credit 稀释**：coding agent 轨迹平均有 **104 条消息、51 个 assistant turn**（Polar 论文实测 SWE-Gym 数据集），长尾超过 200 turn。整条序列动辄几万 token，一个标量 reward 摊到几万 token 上，梯度信号被稀释到几乎为零。

### 2. 方法论

**先把"轨迹"这个连续概念离散化成"可训练单元"**。业界的共同抽象是：一条轨迹 = 若干次 **LLM 调用**（LLM call / completion），每次调用天然给出一个干净的 `(prompt_tokens, response_tokens)` 对。Agent Lightning 把它叫 **Triplet**（三元组），Polar 叫 **Trace**，verl 的 agent loop 叫 **sample**。

- **Triplet 的字段定义**（`agent-lightning/agentlightning/types/core.py` + `adapter/triplet.py`）：

```python
# agent-lightning/agentlightning/adapter/triplet.py:65-81
class Transition(BaseModel):
    """A single transition within a reinforcement learning trajectory."""
    state: List[int]        # Token identifiers describing the model input state（prompt tokens）
    action: List[int]       # Token identifiers representing the model output（response tokens）
    response_id: Optional[str]
    agent_name: str
    reward: Optional[float]
```

**关键点：`state` 和 `action` 都是 token id 列表，不是字符串。** 这是整个拼接正确性的地基——只有拿到真实 token id，才能保证"训练在模型真实采样过的序列上"。

- **三种 route 的取舍**（这是本专题的核心张力，第 2 点展开）：

| 路线 | 代表 | 做法 | 优点 | 代价 |
|---|---|---|---|---|
| **不拼接**（per-call） | Agent Lightning | 每次 LLM 调用独立成一个样本，`state → action` | 简单、无 mask 问题、天然 token-faithful | 丢失跨轮 credit assignment；样本碎片化 |
| **拼接**（stitching） | Polar `prefix_merging` | 把 append-only 的调用链合并成长轨迹 | 样本数大幅减少、跨轮语义连贯 | 需要严格的 token 前缀校验 + 精细 loss mask |
| **折中** | verl agent loop / 自研 | 按 turn 组装，observation 段 mask 掉 | 兼顾 | 实现复杂度中等 |

### 3. 具体数值样例

同一条 51-turn 的 coding 轨迹，两种路线的样本数差异：

- **不拼接**：51 次调用 → 51 个样本，每个样本平均 prompt 已累积到几千 token（因为每次都要带上完整历史），**总 token 量 = 51 × 平均 prompt 长度**，随轮数**二次增长**；
- **拼接**：合并成 1~3 条长 trace，**总 token 量 ≈ 轨迹实际长度**（线性）。

Polar 论文的实测数据（SWE-Gym，3 个训练 step、同 workload）：

| 指标 | `per_request` | `prefix_merging` | 倍数 |
|---|---|---|---|
| trainer 收到的 update 数 | 1185 | 218 | 5.44× 减少 |
| 墙钟时间 | 189.5 min | 35.2 min | **5.39× 加速** |
| rollout GPU 平均利用率 | 20.4% | **87.7%** | 4.3× |

**这就是"拼接"的经济价值**：per-request 把 GPU 的利用率压到 20%（trainer 被海量碎样本淹没，rollout 侧空转），拼接后拉回 87.7%。

> 面试一句话总结：**多轮轨迹不能直接喂 PPO，因为工具/环境的 token 不是模型生成的（要 mask 掉）、outcome reward 无法区分各轮对错（credit assignment）、长轨迹会把梯度稀释掉；业界把轨迹离散成"每次 LLM 调用"这个可训练单元（Triplet / Trace），再选择"per-call 独立成样本"还是"前缀合并成长轨迹"。**

---

## 2. 两条根本路线：per-call 独立样本 vs 前缀合并

### 1. 现有问题

"每次 LLM 调用是一个样本"听起来很干净，但**它有个致命前提**：单次调用必须自带足够的学习信号。现实中：

- **单轮 RL（数学题、代码补全）**：reward 直接来自这一轮的答案，per-call 完全成立；
- **多轮 agent**：reward 是**轨迹级**的（任务最终成功与否），早期轮次拿不到任何 reward。per-call 切分后，**早期轮次的样本没有 reward**，只能靠 `final_reward` 挂到最后一轮（Agent Lightning 的做法）或干脆丢弃。

所以两条路线的本质分歧是：**要不要为"跨轮 credit assignment"付出"拼接复杂度"的代价**。

### 2. 方法论

**路线 A：不拼接（Agent Lightning）**

Agent Lightning 的 `TraceTree.to_trajectory()` 把 trace tree 转成 **Triplet 列表**——每次 LLM 调用一个 Triplet，**不做任何合并**：

```python
# agent-lightning/agentlightning/adapter/triplet.py:702-758（节选）
def to_trajectory(
    self,
    llm_call_match: str = r"openai\.chat\.completion",
    agent_match: Optional[str] = None,
    exclude_llm_call_in_reward: bool = True,
    dedup_llm_call: bool = True,
    reward_match: RewardMatchPolicy = RewardMatchPolicy.FIRST_OCCURRENCE,
    final_reward: Optional[float] = None,
) -> List[Triplet]:
    """Convert the trace tree into a trajectory of Triplet items."""
    # 1) 找出所有 LLM 调用 span
    llm_calls = self.find_llm_calls(...)
    # 2) 每个 span → 一个 Triplet（prompt token ids + response token ids）
    id_transitions = []
    for llm_call, agent_name in llm_calls:
        triplet = self.span_to_triplet(llm_call.span, agent_name)
        ...
        id_transitions.append((llm_call.id, triplet))
    # 3) 按策略把 reward 匹配到某个 LLM 调用上
    rewards = self.match_rewards(reward_match, [call for call, _ in filtered_llm_calls])
    transitions = [t.model_copy(update={"reward": rewards.get(id, None)}) for id, t in id_transitions]
    # 4) 可选：把 final_reward 挂到最后一条
    if final_reward is not None and len(transitions) > 0:
        transitions[-1] = transitions[-1].model_copy(update={"reward": final_reward})
    return transitions
```

配套的两个工程细节很值得说：

**(a) reward 匹配策略**（`RewardMatchPolicy`）——决定轨迹级 reward 落到哪一轮：

- `FIRST_SIBLING`：用当前 trace 子树里的**第一个 reward span**；
- `FIRST_OCCURRENCE`：按时间顺序，用当前 LLM 调用**之后遇到的第一个 reward**。

**(b) 层级修复 `repair_hierarchy()`**——真实世界的 trace 树往往是坏的：

```python
# agent-lightning/agentlightning/adapter/triplet.py:478-488（函数文档）
def repair_hierarchy(self) -> None:
    """Repair missing parent-child relationships introduced by mixed tracing systems.

    Some agent frameworks emit spans via multiple subsystems, which can cause LLM completion
    spans to float directly under the root span instead of being nested under the correct agent.
    The method re-parents those spans to the closest ancestor that fully envelopes the child in
    time.
    """
```

即：混合追踪系统（OpenAI Agents SDK + AgentOps + LangGraph + Weave…）会让 span 挂错父节点，于是用**时间包含关系**（`node.start_time <= child.start_time and node.end_time >= child.end_time`，取时间余量最小的那个祖先）把它重新挂回去。**这是"工程现实"的典型体现**：框架给的 trace 不是干净树，得自己修。

**(c) Agent Lightning 其实也提供了"拼接"档位**——上面是 transition 级（不拼接），它还有 trajectory 级。`agentlightning/verl/daemon.py` 的 `get_train_data_batch()` 里用 **`ids_startswith` 做前缀判定**，然后把后续轮次增量拼进来：

```python
# agent-lightning/agentlightning/verl/daemon.py:947-1037（节选，trajectory 级）
is_prefix, diagnostic = ids_startswith(
    trace["prompt_ids"] + trace["response_ids"], current_context, self.tokenizer, ...)
...
for turn_index in current_merged_trace_idx[1:]:
    trace = sample_info["trace_list"][turn_index]
    new_prompt_length = len(trace["prompt_ids"]) - len(response_ids) - prompt_length
    response_ids += trace["prompt_ids"][-new_prompt_length:]   # 增量 prompt（= 工具结果）
    response_ids += trace["response_ids"]                      # 本轮模型生成
    response_mask += [0] * new_prompt_length                   # ★ 工具/环境 → 0
    response_mask += [1] * len(trace["response_ids"])          # ★ 模型生成 → 1
```

**注意 `ids_startswith` 与 Polar 的 `p_{m+1}[1:|p_m|]=p_m` 是同一件事**——两家独立实现了"前缀判定才能拼接"这个必要条件。区别在**判定失败后的行为**：AL 走 `is_drop` 标记（在 advantage 计算里处理，**不在这一步丢**，注释注明"为了 advantage 计算的正确性"），Polar 则**分裂成新的 chain**。

**(d) 两家的 credit assignment 策略正好相反**：

- **Agent Lightning：identical assignment**（官方文档原话）——
  > *"The final scalar reward … is **propagated to all preceding triplets** following the identical assignment strategy. This ensures that each triplet receives an identical reward signal and can be **independently optimized as a valid RLHF trajectory**."*
  即"整条轨迹的 reward 原样复制给每一个 transition"。
- **Polar：拒绝广播**——论文实测 `per_request` + outcome 广播"显著 reward hacking"（见第 5 点）。

**这两条的差异本质是"样本切分粒度"决定的**：切得越碎（per-call），广播就越危险；切得越整（一条长 trace），广播才安全。**这是本专题最值得在面试里点出的因果关系。**

**路线 B：前缀合并（Polar `prefix_merging`）**

Polar 的做法是：**先把 completion 分成若干条"append-only 链"，再把每条链合并成一条长 trace**。判定一条新 completion 能否接进已有链，靠**两个条件**：

1. **消息级分组键**（normalized message-level grouping key）判定它是"候选续写"；
2. **严格 token 前缀关系**必须成立：

$$
p_{i_{m+1}}[\,1:|p_{i_m}|\,] = p_{i_m}
$$

（第 $m+1$ 次调用的 prompt，前 $|p_{i_m}|$ 个 token 必须与第 $m$ 次的 prompt 逐 token 相同。）

**为什么子 agent / 并行分支会自然分裂成多条链**：subagent、并行 branch、context compaction、prompt rewriting、独立的 tool-mediated conversation——这些都会破坏"前缀相同"这个条件，于是 Polar **把它们分成不同链**，而不是硬塞进一条全局 trace。这个设计非常关键：**它不假设整条 session 是一条对话**。

**合并公式**：设一条链 $(C_{i_1},\dots,C_{i_K})$，第 $m$ 次调用的 prompt 为 $p_m$、原始采样响应为 $a_m$、response log-prob 为 $\ell_m$，end-of-turn token 记为 $e$。两次相邻调用之间定义**canonical tail**：

$$
t_m = p_{m+1}[\,|p_m|+1:\,]
$$

在 $t_m$ 里定位第一个 $e$：若 $a_m$ 已经以 $e$ 结尾，则 interstitial $u_m$ 是那个 $e$ **之后**的后缀；否则 $u_m$ 从那个 $e$ 开始（保证 assistant turn 在下一个 prompt 上下文之前先闭合）。整条链的 token 序列是

$$
z^{(j)} = p_1 \,\|\, a_1 \,\|\, u_1 \,\|\, a_2 \,\|\, u_2 \,\|\cdots\|\, a_K
$$

发出的 trace 是：**$p_1$ 作为 trace 的 prompt，$a_1\|u_1\|\cdots\|a_K$ 作为 trace 的 response**。

### 3. 具体数值样例

设一条链有 3 次调用，token 级别如下（简化）：

| 调用 | prompt $p_m$（长度） | 采样响应 $a_m$（长度） |
|---|---|---|
| $C_1$ | `[SYS, U]` (2) | `["我想", "看文件"]` (2) |
| $C_2$ | `[SYS, U, 我想, 看文件, <e>, TOOL_RESULT]` (6) | `["问题", "在42行"]` (2) |
| $C_3$ | `[SYS, U, 我想, 看文件, <e>, TOOL_RESULT, 问题, 在42行, <e>, TOOL_RESULT2]` (10) | `["已修复"]` (1) |

**前缀校验**：

- $p_2[1:2] = [SYS, U] = p_1$ ✓ → $C_2$ 可接进链；
- $p_3[1:6] = [SYS, U, 我想, 看文件, <e>] = p_2$ ✓ → $C_3$ 可接进链。

**canonical tail 与 interstitial**：

- $t_1 = p_2[3:] = [<e>, TOOL_RESULT]$。$a_1$ 不以 `<e>` 结尾 → $u_1$ 从 `<e>` 开始 → $u_1 = [<e>, TOOL_RESULT]$；
- $t_2 = p_3[7:] = [<e>, TOOL_RESULT2]$。$a_2$ 不以 `<e>` 结尾 → $u_2 = [<e>, TOOL_RESULT2]$。

**合并结果**：

$$
z = \underbrace{[SYS,U]}_{p_1} \| \underbrace{[我想,看文件]}_{a_1} \| \underbrace{[<e>,TOOL_RESULT]}_{u_1} \| \underbrace{[问题,在42行]}_{a_2} \| \underbrace{[<e>,TOOL_RESULT2]}_{u_2} \| \underbrace{[已修复]}_{a_3}
$$

总长 = 2+2+2+2+2+1 = **11 token**，其中 **loss mask = 1 的只有 $a_1,a_2,a_3$ 共 5 个 token**（`我想,看文件,问题,在42行,已修复`），$u_1,u_2$ 的 4 个 token 和 $p_1$ 的 2 个 token **mask=0**。

**对照 per-request 路线**：3 次调用 → 3 个样本，prompt 总长 = 2+6+10 = **18 token**，且每次都要重算一遍历史。合并成 1 条 trace 后 token 量从 18 降到 11，**样本数从 3 降到 1**。

> 面试一句话总结：**两条路线——Agent Lightning 的"per-call 独立成样本"（简单、token-faithful，但丢跨轮 credit、样本碎片化，实测 trainer update 数 1185 vs 218、GPU 利用率 20.4% vs 87.7%）；Polar 的"前缀合并"（按严格 token 前缀关系把 append-only 链拼成长 trace，subagent/compaction 等自然分裂成多条链，canonical tail 定位 interstitial）。**

---

## 3. Token 保真：拼接的前提是拿到真实 token id

### 1. 现有问题

拼接（和 per-call 切分）都建立在"我知道模型当时采了哪些 token id"之上。但 agent 框架走的是 **OpenAI 兼容 API**，而这类 API **历史上只返回字符串**。于是训练侧必须重新分词，产生 **Retokenization Drift**：

> vLLM 官方博客原话：**"tokens are detokenized during inference and subsequently retokenized during training; the two sets of tokens may differ even though their corresponding strings are identical."**

### 2. 方法论

**解法只有一个方向：让推理接口直接返回 token ids。**

vLLM 的 OpenAI 兼容端点支持 `"return_token_ids": true`：

```json
// 请求 /v1/chat/completions 或 /v1/completions
{"model": "...", "messages": [...], "return_token_ids": true}
// 响应中额外带回
{"choices": [{"prompt_token_ids": [...], "token_ids": [...]}]}
```

**Agent Lightning 在 adapter 层做了"多源兼容"**——`span_to_triplet()` 会依次尝试一串属性名，把不同版本 vLLM / OpenAI SDK 放 token id 的位置都覆盖到：

```python
# agent-lightning/agentlightning/adapter/triplet.py:637-658（节选）
prompt_token_ids = (
    _attributes_get_ids_multiple(
        span.attributes,
        [
            "prompt_token_ids",
            "agentlightning.operation.output.prompt_token_ids",  # Weave tracer
        ],
    ) or []
)
response_token_ids = (
    _attributes_get_ids_multiple(
        span.attributes,
        [
            "response_token_ids",
            "agentlightning.operation.output.response_token_ids.0",                          # Weave tracer
            "agentlightning.operation.output.choices.0.token_ids",                            # Weave tracer with newer vLLM
            "agentlightning.operation.output.choices.0.provider_specific_fields.token_ids",   # new vLLM + new OpenAI client SDK
        ],
    ) or []
)
```

最后那一行 `provider_specific_fields.token_ids` 就是"新版 vLLM + 新版 OpenAI SDK"的落点——**这段代码本身就是 retokenization drift 演进的考古层**：每加一个版本就多一个属性名。

**Polar 的对应设计**（论文 §3.2「Capture token-level data」）：网关代理在转发请求时**主动加上 `logprobs=true`**，然后记录一份 completion record，包含 request messages、response messages、**prompt token IDs、sampled response token IDs、finish reason，以及推理后端返回的 log probabilities**。

**Polar 的 token 保真不变式**（论文原文）：

> *"Every trainable token matches the behavior policy during rollout, and any non-generated tokens are masked out."*

这句可以当作整个"轨迹拼接"领域的**第一性原理**。

### 3. 具体数值样例

同一条轨迹，两种取 token 的方式：

| 方式 | token 序列 | 训练在什么上 |
|---|---|---|
| 存文本 + 重新分词 | `["HAV", "ING"]` | **一条从未被采样过的路径** |
| 取推理返回的 token ids | `["H", "AV", "ING"]` | 模型真实采样路径 ✓ |

一条 51-turn、平均每轮 response 200 token 的轨迹，response 端总 token ≈ $51\times200=10200$。如果**每轮的分词边界都有一次翻转**，那么"错位"会从翻转点开始污染该轮剩余的所有 token——按平均 3 个 token 一次边界分歧估算，**约有 $10200/3\times\frac12\approx1700$ 个 token 的 log-prob 是错的**（约 17%）。

**这个比例的梯度是纯噪声**。而它不会有任何报错、IS 权重看起来也正常（因为字符串层面确实一致）——这就是 retokenization drift 最阴险的地方。

> 面试一句话总结：**拼接正确的前提是 token 保真——推理时必须拿到真实 token id（vLLM `return_token_ids: true` 返回 `prompt_token_ids`/`token_ids`），否则训练侧重新分词会产生 drift（如 HAVING → H+AV+ING vs HAV+ING），梯度被算在一条从未采样过的路径上；Agent Lightning 的 adapter 同时兼容多种属性名，正是这个问题的工程考古层。**

---

## 4. Loss mask 的构造：哪些 token 参与训练

### 1. 现有问题

合并后的长 trace 里，token 分三类，**只有第一类能训练**：

| 类别 | 来源 | 能训练？ | 原因 |
|---|---|---|---|
| **assistant token** | 模型采样（`a_m`） | ✅ | 这是行为策略产生的，advantage 才有意义 |
| **interstitial token** | harness 插入的模板/canonical 渲染（`u_m`） | ❌ | 不是模型生成的，算 log-prob 没有意义 |
| **prompt token** | 初始 prompt（`p_1`） | ❌ | 同上，且它们是条件不是动作 |

### 2. 方法论

**Polar 的 mask 规则（精确到 token）**：

- loss mask = **1**：来自采样响应 $a_m$ 的 token；
- loss mask = **0**：来自 canonical interstitial $u_m$ 的 token；
- **真实的 response log-prob 只对 $a_m$ 拷贝**；interstitial 位置放**合成的 log-prob 占位**，目的是让 `response_logprobs` 与 `response_ids` **保持长度对齐**，可训练性由 `loss_mask` 单独控制。

这个"**logprob 占位 + mask 控制**"的设计很值得注意：它把"数据对齐"和"是否训练"解耦了——所有位置都有 logprob（便于张量对齐），但只有 mask=1 的位置参与 loss。

**verl 侧的对应构造**（`internship.md` 记录的 TQ 路径）：从 agent 轨迹构造 `input_ids / attention_mask / position_ids / loss_mask / response_mask / token_level_scores / uid / num_turns`，其中：

- `loss_mask`：标记哪些位置是"模型生成的 response token"；
- `response_mask`：response 段的 mask（拒绝采样也是改它，见 `Training-Inference-Mismatch.md` 第 5 点）；
- `token_level_scores`：reward 只放在 **response 最后一个 token** 的位置（GRPO 的 token 级奖励语义）。

**一个容易踩的坑**：`loss_mask`（哪些 token 算 loss）与 `response_mask`（哪些是 response）**不是一回事**。前者是"是否可训练"，后者是"是否属于 response 段"。在拼接场景里，interstitial token 属于 response 段（`response_mask=1`）但不可训练（`loss_mask=0`）——**这个区别在面试里是加分项**。

### 3. 具体数值样例

继续第 2 点的 11-token 例子：

| 位置 | token | 来源 | `response_mask` | `loss_mask` | `logprob` |
|---|---|---|---|---|---|
| 0 | `SYS` | $p_1$ | 0 | 0 | 真实（prompt） |
| 1 | `U` | $p_1$ | 0 | 0 | 真实（prompt） |
| 2 | `我想` | $a_1$ | 1 | **1** | 真实（采样） |
| 3 | `看文件` | $a_1$ | 1 | **1** | 真实（采样） |
| 4 | `<e>` | $u_1$ | 1 | **0** | 占位 |
| 5 | `TOOL_RESULT` | $u_1$ | 1 | **0** | 占位 |
| 6 | `问题` | $a_2$ | 1 | **1** | 真实 |
| 7 | `在42行` | $a_2$ | 1 | **1** | 真实 |
| 8 | `<e>` | $u_2$ | 1 | **0** | 占位 |
| 9 | `TOOL_RESULT2` | $u_2$ | 1 | **0** | 占位 |
| 10 | `已修复` | $a_3$ | 1 | **1** | 真实 |

- `response_mask.sum() = 9`（位置 2~10）；
- `loss_mask.sum() = 5`（位置 2,3,6,7,10）；
- **归一化分母的选择会直接改变梯度尺度**：若用 `response_mask.sum()` 归一化，分母 9；若用 `loss_mask.sum()`，分母 5——**同样的 loss 会差 1.8 倍**。这就是为什么必须明确"batch_num_tokens 用哪个 mask 求和"。

**一个真实的 token 级实测（uni-agent Gateway，`docs/任务剖析-内部形态.md` 的本地实测记录）**：

| 量 | 值 |
|---|---|
| `prompts` 长度 | 1288 |
| `responses` 长度 | 1004 |
| `input_ids` 长度 | **2292**（= 1288 + 1004） |
| `response_mask` = `loss_mask` | **411×1 + 593×0**（共 1004） |
| `rm_scores` | 长度 1004，**末位 = 1.0**（其余 0） |

**关键比例**：

- **模型生成的 token 只占 response 的 $411/1004=40.9\%$**；
- **只占整条序列的 $411/2292=17.9\%$**。

也就是说：**一条 agent 轨迹里，有 82% 的 token 是"陪跑"的**（prompt + 工具返回 + 模板）。如果不做 mask、按全序列算 loss 并按 token 平均，**58% 的梯度信号来自根本不是模型生成的内容**——这比"梯度被稀释"更严重，是"梯度方向被污染"。

mask 在序列里的**分段结构**（同文档给出的 7 段实例）：`26×1 / 253×0 / 320×1 / 30×0 / 35×1 / 310×0 / 30×1`——**交替的"模型生成段 / 工具返回段"**，这正是多轮 agent 轨迹的典型形状。

**UA 侧对应的构造代码**（`vendor/uni-agent/uni_agent/gateway/session/session.py`）：模型生成时 append `mask=1`、工具结果增量 append `mask=0`，两个方向都写进同一个 buffer：

```python
# session.py:272-291  模型生成 → mask=1
encoded.buffer.response_ids.extend(response_ids)
encoded.buffer.response_mask.extend([1] * len(response_ids))
if encoded.sampling_params.get("logprobs", False):
    if len(log_probs) != len(response_ids):
        raise RuntimeError("backend logprobs must align with token_ids: ...")
    encoded.buffer.response_logprobs.extend(log_probs)

# session.py:451-478  工具结果 / 增量上下文 → mask=0
incremental_ids = self._codec.encode_incremental(incremental_messages, ...)
buffer.response_ids.extend(incremental_ids)
buffer.response_mask.extend([0] * len(incremental_ids))
context_ids = buffer.prompt_ids + buffer.response_ids   # 本轮完整上下文
```

最后落在训练张量上（`framework.py:925-993`）：`input_ids = cat([prompts, responses])`、`attention_mask` 全 1（变长不 padding）、`rm_scores[-1] = reward`、**`loss_mask = response_mask`**（UA 里两者等价）。

> 面试一句话总结：**拼接后的 trace 里只有"模型采样的 assistant token"能训练（loss_mask=1），harness 插入的 interstitial 和初始 prompt 都要 mask 掉（loss_mask=0）；实测一条 agent 轨迹里模型 token 只占 response 的 40.9%、全序列的 17.9%（7 段交替的 mask 结构），不做 mask 等于让 58% 的梯度来自非模型内容；Polar 的工程细节是给 interstitial 放合成 logprob 占位以保持长度对齐、可训练性交给 loss_mask；注意 loss_mask ≠ response_mask，归一化分母用哪个直接决定梯度尺度（例子里 9 vs 5，差 1.8 倍）。**

---

## 5. Reward 传播与 credit assignment

### 1. 现有问题

拼接/切分完之后，reward 怎么发？三种粒度：

1. **outcome reward**：整条 session 一个 0/1（SWE-bench 的 pass/fail）；
2. **per-trace reward**：每条合并 trace 一个（session 有多个 trace 时）；
3. **process reward**：每一步一个（需要 PRM 或规则）。

Polar 论文报告了一个**重要的负面结果**：

> *"We also tried `per_request` with outcome-reward broadcasting to every emitted trace, but observed significant reward hacking. The issue is noisy credit assignment: request-level traces can receive session-level credit without proper session normalization or an advanced process reward model."*

即：**把 session 级 outcome reward 广播到每条 per-request trace → 严重 reward hacking**。原因正是 credit assignment 噪声——每条 request 都拿到 session 级分数，模型会去刷那些"容易被判成功"的局部行为。

### 2. 方法论

**Polar 的两级策略**：

> *"An outcome reward can be broadcast to every trace, whereas tasks with process rewards may need per-trace assignment."*

- session 只有**一条** trace（前缀合并成功）→ outcome reward 广播是安全的（因为本来就一一对应）；
- session 有**多条** trace（subagent / compaction 分裂）→ 需要 per-trace 分配，或做 **session normalization**。

**Agent Lightning 的对应机制**是两个 `RewardMatchPolicy`（第 2 点已引）：`FIRST_SIBLING` / `FIRST_OCCURRENCE`——都是"把 reward 挂到**某一个** LLM 调用上"，而不是广播。再加一个 `final_reward` 参数把轨迹级 reward 显式挂到**最后一条** transition：

```python
# agent-lightning/agentlightning/adapter/triplet.py:755-757
if final_reward is not None and len(transitions) > 0:
    # Add the final reward to the last transition
    transitions[-1] = transitions[-1].model_copy(update={"final": final_reward})
```

**GRPO 侧的另一种做法（组内归一化 + episode 等权）**：

- **outcome reward → 组内 z-score**：同一个 prompt 采 $n$ 条，算 $\frac{r_i-\text{mean}(r)}{\text{std}(r)}$（或 Dr.GRPO 只减均值不除 std）；
- **episode 等权归一化**：多轮轨迹按 episode 等权，**避免失败的长轨迹在 loss 里占比过高**（长轨迹 token 多，如果按 token 平均，一条 200-turn 的失败轨迹会主导梯度）；
- **reward-to-go / 折扣**：只在**实际执行步**给折扣，把梯度信号集中到真正产生动作的位置。

**uni-agent 的做法（一个很干净的工程答案）**：把"多轮拼接出的同一 session 的多个 output"当成一组，**只在 session 的最后一个 output 上算 GRPO advantage，再广播回该 session 的所有 output**：

```python
# vendor/uni-agent/verl/trainer/ppo/v1/utils.py:133-202（节选）
def compute_advantage_for_multi_trajectories(data, batch_keys, adv_estimator, ...) -> DataProto:
    """... only the final output in each ``{uid}_{session_id}`` group participates
    in advantage computation, and the result is broadcast to the other outputs ..."""
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return compute_advantage(...)          # GAE 等原样透传
    for i, key in enumerate(batch_keys):
        uid, session_id, index = key.rsplit("_", 2)
        session_key = f"{uid}_{session_id}"
        if session_key not in final_sessions or final_sessions[session_key][0] < index:
            final_sessions[session_key] = (index, i)     # ★ 只留最大 index（最后一个 output）
    final_data = compute_advantage(data.select_idxs(final_indices), ...)
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]
    scores = final_scores[row_to_local_index].unsqueeze(-1) * data.batch["response_mask"]  # ★ 只在 mask=1 位置铺开
    data.batch["advantages"] = scores
    data.batch["returns"] = scores
```

**为什么这个设计合理**：多轮拼接出的多个 output 共享**同一个 outcome reward**，如果每个 output 各算一次组内归一化，等于把"同一个 session"重复计数、放大噪声；只在最后一个（包含完整信息的）output 上算一次、再广播，等价于"以 session 为 GRPO 的组单位"。**注意最后一行 `* response_mask`——广播也只在可训练位置生效。**

配套的 GRPO 组内归一化（`verl/trainer/ppo/core_algos.py`）：

```python
scores = token_level_rewards.sum(dim=-1)              # ★ outcome-only：token 维加总成一个标量
if len(id2score[idx]) == 1:
    id2mean[idx] = 0.0; id2std[idx] = 1.0             # ★ 组内只有 1 条 → advantage ≡ 0（不产生梯度）
else:
    id2mean[idx] = mean; id2std[idx] = std
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)   # norm_adv_by_std_in_grpo
scores = scores.unsqueeze(-1) * response_mask
```

注意 `gamma`/`lam` 在 GRPO 分支里**根本没被使用**（只透传给非 GRPO 的 GAE 路径）——这从代码上印证了"GRPO 是 outcome-only、不做逐轮折扣"这一事实。**如果面试被问"agent 多轮怎么做 reward-to-go"，诚实答案是：主流实现里没有逐轮折扣，跨轮 credit 靠的是 mask + 广播 + 组内归一化。**

### 3. 具体数值样例

**组内归一化**：同一个 SWE 问题采 4 条轨迹，outcome reward 为 $[1, 0, 0, 0]$：

$$
\text{mean}=0.25,\quad \text{std}=\sqrt{\frac{(0.75)^2+3\times(0.25)^2}{4}}=\sqrt{\frac{0.5625+0.1875}{4}}=\sqrt{0.1875}\approx0.433
$$

z-score：$[+1.732,\ -0.577,\ -0.577,\ -0.577]$。若用 Dr.GRPO（只减均值）：$[+0.75,\ -0.25,\ -0.25,\ -0.25]$——**两者差 2.3 倍**，这就是"除不除 std"的实际影响。

**episode 等权的必要性**（构造示例）：两条轨迹，一条成功（10 turn，1000 token，$r=1$），一条失败（100 turn，10000 token，$r=0$）：

- **按 token 加权**：失败轨迹贡献 $10000/11000=90.9\%$ 的梯度权重；
- **按 episode 等权**：两条各 50%。

**在 GRPO 里"减均值"这一步就把两组抵消了**，所以按 token 加权的危害主要体现在**组内长度不均衡时的方差**上——这也是 reward-to-go 折扣与 episode 等权被引入的原因。

> 面试一句话总结：**reward 传播有三档（outcome / per-trace / process）；Polar 实测"把 session 级 outcome reward 广播到每条 per-request trace"会导致严重 reward hacking（credit assignment 噪声）；正确做法是——单 trace 可广播、多 trace 要 session normalization 或 per-trace 分配，Agent Lightning 用 FIRST_SIBLING/FIRST_OCCURRENCE 把 reward 挂到某一次调用上；GRPO 侧再叠组内 z-score（或 Dr.GRPO 减均值）+ episode 等权 + reward-to-go 折扣。**

---

## 6. 变长样本的组织与批处理

### 1. 现有问题

拼接出来的 trace 长度差异极大：SWE-Gym 的轨迹平均 104 条消息、51 turn，**长尾超 200 turn**。同一批里既有 500 token 的短 trace、也有 50k token 的长 trace，怎么组织成张量？

### 2. 方法论

**三种组织方式**：

| 方式 | 做法 | 代表 | 代价 |
|---|---|---|---|
| **Padded** | 补到 batch 内最大长度，加 attention mask | 早期 verl DataProto | 长尾样本导致 padding 占比高、显存浪费 |
| **Nested / jagged** | 变长张量（`torch.nested` / jagged），不 padding | TQ 路径（`internship.md`） | 某些算子走慢速回退路径 |
| **Packing** | 把多条短样本首尾相接填满一条固定长度序列，用 attention mask 隔离 | 现代做法 | 需要正确的 position_ids 与 mask |

**关键工程约束**（来自 `internship.md` 的实测）：

- `map_size` / batch 规模要能被 **DP × micro_bsz** 整除（verl 0.8.0 的 `prepare_micro_batches` 断言）；
- **rollout.n 展开**：一条 prompt 采 $n$ 条 response → 样本数变成 $n$ 倍，agent 场景下更复杂（agent 已经把 $n$ 条 response 展开成独立 triplet，所以 `_update_actor` 里**不能再乘一次 rollout.n**）；
- **DP 负载均衡补齐**：按序列长度重排，让各卡 token 数接近；补齐样本要标记为 padding（不能污染 loss）。

**Polar 的异步 staging 直接影响 batch 组装**：gateway 用 `INIT / READY / RUNNING / POSTRUN` 四个 worker pool + 一个有界 `READY` buffer，把"CPU 密集的 runtime 准备"和"GPU 密集的 agent 执行"解耦；`POSTRUN` 阶段才做轨迹重建与评估。这样 trainer 拿到的 batch 是**已经重建好的 trace**，不需要在训练关键路径上做拼接。

### 3. 具体数值样例

设一个 batch 有 4 条 trace，长度 $[500, 12000, 800, 45000]$：

- **Padded**：补到 45000 → 总 padding = $(45000-500)+(45000-12000)+(45000-800)+(45000-45000)=135700$ token，**有效占比 = $\frac{58300}{180000}=32.4\%$**——近 70% 算力浪费在 padding 上；
- **Packing**：如果按 8192 长度打包，4 条 trace 需要 $\lceil500/8192\rceil+\lceil12000/8192\rceil+\lceil800/8192\rceil+\lceil45000/8192\rceil=1+2+1+6=10$ 个 pack，**有效占比提升到 $\frac{58300}{81920}=71.2\%$**；
- **Nested**：零 padding，但要注意 kernel 支持（昇腾 NPU 上 `torch.nested` 曾被怀疑走慢速回退路径）。

**DP 补齐的连锁效应**（真实工程坑）：假设 DP=16、micro_bsz=4 → per-GPU 需要 $4\times$ 的整数倍。若某次 agent 只产出 100 条，朴素的 `floor(100/32)*32=96` **会丢掉 4 条真实样本**；正确做法是 **upsample/补齐到 128**（补齐样本标 padding），而不是丢弃。`internship.md` 记录了 verl 0.6.1→0.8.0 升级时这个断言踩的坑（pad 除数从 16 变 64、`mini_batch` 不能再乘 rollout.n）。

**一个很巧的工程细节：补齐样本必须"伪装成独立 session"，否则会污染 GRPO 组**。uni-agent 侧的做法是给每条 pad 样本生成**独立的 uid**：

```python
# vendor/uni-agent/verl/trainer/ppo/padding_utils.py:160-176（节选）
pad_uid = f"pad{uuid.uuid4().hex}"
pad_keys.append(f"{pad_uid}_{local_idx}_0")     # ★ 每行独立 session_id → 不会被算进真实 GRPO 组
template_tag.update(is_padding=True, prompt_len=1, response_len=1, seq_len=2)
# "zero out reward-related fields so they do not contribute to PPO, entropy, or KL losses"
```

**为什么必须这么做**：如果 pad 样本复用了某个真实 prompt 的 uid，它就会被算进那个 prompt 的 GRPO 组——**凭空给组里加了一个 reward=0 的样本，改变该组的均值和标准差**，直接污染 advantage。这是"补齐"和"负载均衡"必须一起考虑的原因。

配套的负载均衡（同文件）：

```python
global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
```

即按**各序列 token 数**（不是条数）做均衡分片——否则一个 DP rank 拿到 3 条 50k 的 trace、另一个拿到 30 条 1k 的，同步点会被最慢的卡拖死。

> 面试一句话总结：**变长 trace 的组织有 padded / nested / packing 三条路——padded 在长尾场景浪费近 70% 算力（500 vs 45000 的例子里有效占比仅 32.4%），packing 能拉到 71%，nested 零 padding 但受 kernel 支持限制；批处理还要处理 rollout.n 展开（agent 已展开则不能再乘）、DP 整除断言（补齐而非丢弃）、以及按 token 数做负载均衡。**

---

## 7. 架构串讲：从 LLM API 流量到训练 batch

把前面 6 点串起来，一条完整链路是：

```
① 捕获（Capture）
   harness 正常运行，LLM 调用被网关/代理截获
   ├─ 打开 return_token_ids        → 拿到 prompt_token_ids / token_ids（第 3 点）
   ├─ 加 logprobs=true            → 拿到采样 log-prob（IS/RS 的输入）
   └─ [MoE] 记录 routed_experts    → 与 Training-Inference-Mismatch 第 6 点联动
                    │
                    ▼
② 组装（Assemble）——先修树，再切单元
   ├─ from_spans() → TraceTree（缺 root 就合成 virtual root）
   ├─ repair_hierarchy()          → 按时间包含关系重挂父节点（第 2 点）
   └─ 每个 LLM 调用 → Triplet(prompt_ids, response_ids)（第 1 点）
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
③A 不拼接（Agent Lightning）   ③B 拼接（Polar prefix_merging）
   每次调用 = 1 个样本            按严格 token 前缀关系聚成链
   reward 用 FIRST_SIBLING /     每链合并 z = p1‖a1‖u1‖…‖aK
   FIRST_OCCURRENCE 匹配          loss_mask=1 仅 a_m，u_m 放占位 logprob
   （第 2/5 点）                  （第 2/4 点）
        └───────────┬────────────┘
                    ▼
④ Reward 分配
   ├─ 单 trace：outcome 广播安全
   └─ 多 trace：必须 per-trace 或 session normalization（否则 reward hacking，第 5 点）
                    │
                    ▼
⑤ 批处理
   padded / nested / packing；DP 整除补齐（不丢弃）；按 token 数均衡（第 6 点）
                    │
                    ▼
⑥ 训练
   GRPO/PPO loss（分母用 loss_mask 还是 response_mask 要明确）
   + Rollout Correction（见 Training-Inference-Mismatch.md）
```

### 各家方案定位（面试对比用）

| 方案 | 集成边界 | 拼接策略 | 关键设计 |
|---|---|---|---|
| **Agent Lightning** | tracer / LLM-client 回调 + SDK | **不拼接**，每次调用一个 Triplet | `TraceTree.repair_hierarchy()` 修混合追踪的坏树；`RewardMatchPolicy` 匹配 reward；adapter 兼容多版本 token id 属性名 |
| **Polar（NVIDIA）** | **Provider API 网关代理**（最深、最黑盒友好） | `per_request` 或 `prefix_merging` | 严格 token 前缀校验 + canonical tail 定位 interstitial；loss mask 只标采样 token；rollout-as-a-service（INIT/READY/RUNNING/POSTRUN） |
| **rLLM** | tracked client + decorator + proxy | 框架自有 workflow 抽象 | 论文把它和 Polar 并列：都"降低集成成本但仍要求 agent 遵从既定接口" |
| **SkyRL-Agent** | Gymnasium 风格环境 + agent 抽象 | 完整栈内自建 | 强在"任务已进入它的抽象后训练效率高" |
| **verl agent loop** | 框架内 agent loop | 按 turn 组装 + mask | 与 verl 训练栈同源，`loss_mask`/`response_mask` 直接对接 PPO/GRPO |
| **uniagent-lighting** | 自定义 Gateway + TQ | triplet 落 TQ（每样本一 key） | 轨迹异步入库、变长 nested、KVBatchMeta 零拷贝分发 |

**Polar 论文对"集成边界"的总结特别值得记**：

> *"For many coding and terminal agents, the most reliable interface is not an SDK callback graph but the provider API endpoint already used by the harness. This choice is narrower than general observability instrumentation, but it is robust to harnesses implemented as command-line programs, package-managed tools, or binaries."*

即：**SDK 回调（Agent Lightning / rLLM 路线）要求 agent 用 Python 且愿意接 SDK；API 网关（Polar 路线）对"编译好的二进制 harness"也有效**——这是两条路线最本质的工程分歧。

### 一句话总纲

> **轨迹拼接 = 把"一条多轮交互"重新离散成"可训练的 token 序列"，并且保证每个可训练 token 都来自行为策略。** 三件事必须同时做对：**token 保真**（拿真实 token id，不然序列是假的）、**loss mask**（只训练采样 token，工具返回与模板文本全 mask）、**样本切分与 reward 分配**（per-call 简单但丢 credit，前缀合并省算力但要严格校验；多 trace 时 reward 不能乱广播）。

---

## 附：高频追问速答

**Q1：为什么不能把整条 agent 轨迹当一个样本？**
工具/环境返回的 token 不是模型生成的，对它们算 log-prob 并施加 advantage 是纯噪声；且 outcome reward 无法区分各轮对错，几万 token 会把梯度稀释掉。

**Q2：`loss_mask` 和 `response_mask` 有什么区别？**
`response_mask` 标"哪些 token 属于 response 段"；`loss_mask` 标"哪些 token 参与 loss"。interstitial token 属于 response 段但不可训练（`response_mask=1, loss_mask=0`）。归一化分母用哪个直接决定梯度尺度（例子里 9 vs 5，差 1.8 倍）。

**Q3：Polar 的前缀合并怎么判断两条 completion 能接上？**
两个条件：消息级分组键判定是候选续写；且**严格 token 前缀关系** $p_{m+1}[1:|p_m|]=p_m$ 成立。

**Q4：subagent / context compaction 怎么办？**
它们**天然破坏前缀关系**，所以 Polar 把它们分裂成**不同的链**（不同 trace），而不是硬塞进一条——这正是"不假设整条 session 是一条对话"的设计。

**Q5：interstitial token 为什么还要放 logprob？**
为了让 `response_logprobs` 与 `response_ids` **长度对齐**（张量层面必须等长），可训练性由 `loss_mask` 单独控制。

**Q6：per-request 为什么会导致 reward hacking？**
把 session 级 outcome reward 广播到每条 request trace，等于每条 request 都拿 session 分数——credit assignment 噪声让模型去刷"容易被判成功的局部行为"（Polar 论文实测）。需要 session normalization 或 PRM。

**Q7：为什么 agent 场景下 `_update_actor` 不能再乘 rollout.n？**
因为 agent 已经把 $n$ 条 response 展开成独立 triplet 了，再乘一次会导致 batch 组成翻倍、触发整除断言或重复训练。

**Q8：变长样本用 padded 还是 packing？**
长尾场景（500 vs 45000）padded 的有效占比只有 32.4%，packing 能到 71.2%；nested 零 padding 但受 kernel 支持限制（昇腾上曾怀疑 `torch.nested` 走慢速回退路径）。

**Q9：SDK 回调 vs API 网关两条集成路线的本质区别？**
SDK 回调（Agent Lightning / rLLM）要求 agent 是 Python 且愿意接 SDK；API 网关（Polar）把边界放在 provider API，**对命令行程序、包管理工具、编译好的二进制 harness 都有效**。

**Q10：怎么验证拼接是正确的？**
用 Polar 的不变式自检：**"每个可训练 token 都必须在 rollout 时来自行为策略，任何非生成 token 都必须被 mask 掉。"** 具体可断言：`loss_mask` 非零位置的 token id 必须等于推理返回的 `token_ids` 的对应片段。
