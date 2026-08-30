# Uni-Agent（官方框架）完全指南

> **verl 官方社区开源框架 uni-agent 的深度解析：Gateway（协议处理 × 会话路由 × 轨迹物化）为核心，Agent × Sandbox × Framework × TransferQueue 各组件。**

Uni-Agent 是 verl 官方社区（`verl-project/uni-agent`）开源的 **Agent RL 训练编排层**（官方定位："a framework for training long-horizon agents"），构建在 verl 之上。它的三个核心能力（README Highlights）：

1. **接入任意 agent harness**：Claude Code、Mini-SWE-Agent，或任何能把 OpenAI/Anthropic 兼容模型端点指向 Uni-Agent Gateway 的 harness——"request string in, training tokens out"（请求进、训练 token 出）；
2. **解耦 agents / tasks / infrastructure**：用可复用的 `Agent` / `Tool` / `Task` / `Sandbox` 抽象构建白盒 agent，可独立定制 agent 逻辑、工具、任务环境、沙箱后端、reward，同时复用同一套评估与训练运行时；
3. **千级并发 session**：分布式 worker + 池化 Gateway session + 隔离沙箱 + 异步调度，支持 1000+ 长程有状态会话并发，每条轨迹/日志/reward 都与正确 session 关联。

**核心组件是 Gateway**：agent 跑在任意位置，模型调用统一指向 Gateway，Gateway 在服务模型的同时把 token 级轨迹（prompt/response/logprob/mask/reward，on-policy）物化下来，经 TransferQueue 进入 verl 完成 RL 训练。官方基准（SWE-Bench Verified 等）显示 Uni-Agent 在并行推理与 RL 训练上均优于 OpenHands 等基线（如 Qwen3-Coder-30B 在 SWE-Bench Verified 达 49.2 Avg@4；Qwen3.5-35B-A3B 经 GSPO 训练后 22.2→36.8）。

> 本文只讲**官方 uni-agent 组件本身**（Gateway 如何具体处理不同协议的请求、各组件如何工作、特性设计亮点），基于官方源码（commit `b139419`）讲解。

---

# 核心组件

## 1. Gateway（核心）：协议处理 + 会话路由 + 轨迹物化

### 1. 现有问题：为什么 Gateway 是 uni-agent 的核心

Uni-agent 要解决的根本问题是：**如何让"任意 agent"（白盒开源 harness、黑盒商业产品、外部自定义 agent）的交互过程变成"可训练的数据"？** 直接让 agent 往训练机里塞（"图省事"做法）会导致：agent 与训练强耦合、on-policy 无法保证（agent 调用的 logprob 可能不是当前策略模型的）、轨迹格式五花八门无法统一消费。Gateway 的定位是**统一入口**：agent 不需要改造（把 `base_url` 指向 Gateway 即可），Gateway 负责三件事——**协议处理**（把 OpenAI / Anthropic 等不同协议归一化成内部请求）、**会话路由**（多 session 隔离、多轮对话状态管理）、**轨迹物化**（记录 token 级 prompt/response/logprob/mask/reward，这是 on-policy 的保证）。

**为什么 on-policy 必须靠 Gateway 而非 agent 位置**：RL 训练要求轨迹的 logprob 来自"当前正在训练的策略模型"。如果 agent 在本地调一个外部 API，logprob 由外部生成、不可信也不可控；而 Gateway 云侧统一代理模型调用，logprob 由云侧 vLLM 直接产出并随响应返回——**轨迹的 token 真值（token-truth）在云侧生成，外部无法伪造**。这正是分离式架构的基石：agent 物理位置无关紧要，on-policy 的保证来自 Gateway 云侧物化。

### 2. 方法论：Gateway 是怎么实现的

Gateway 的代码位于 `uni_agent/gateway/`，分四块：**`gateway.py`（actor，HTTP 路由 + 生命周期）、`manager.py`（多 Gateway 实例路由）、`session/`（会话与轨迹，核心）、`adapters/`（协议适配）**。

**（1）协议适配层（`adapters/`）——不同形式的请求如何处理（核心）**。`openai.py` 和 `anthropic.py` 分别实现 OpenAI Chat Completions 和 Anthropic Messages 两种协议，核心是四个转换函数：

- **`openai_to_internal(payload, base_sampling_params, allowed_sampling_keys) → InternalGenerationRequest`**：把 OpenAI 请求降级为内部 canonical 格式（`messages` / `tools` / `sampling_params` 三个字段）。逐步操作：
  - **能力门控（capability gates）**：`n != 1` 直接拒绝（只支持 n=1）；`response_format` 拒绝；`tool_choice` 只支持 `"auto"` / `"none"`（指定具体 function 的 dict 拒绝）——**不支持的特性在入口就报 MalformedRequestError，避免半路出错**；
  - **消息归一化（`_normalize_message`）**：校验每条 message 的 role/content/tool_calls；`_normalize_tool_calls` 把 JSON 字符串形式的 function arguments **解析成 dict**（容错：解析失败保留原字符串）；content 为 None 时转空串；
  - **采样参数合并**：以 `base_sampling_params`（session 默认）为底，把请求里 `allowed_sampling_keys` 允许的字段（如 temperature/top_p）覆盖上去——**白名单机制防止请求注入非法参数**。
- **`anthropic_to_internal(...)`**：把 Anthropic Messages 请求降级为同一 internal 格式。逐步操作：
  - **system 处理**：把 `system` 字段转成 text 并作为第一条 `system` 消息插入（Anthropic 的 system 是顶层字段，OpenAI 的是消息）；
  - **`_fold_mid_list_system_into_user`**：处理 Anthropic 特有的"消息列表中间夹 system"兼容降级；
  - **工具转换 `_convert_tools` + `_apply_tool_choice`**：Anthropic 工具格式 → 内部工具格式；
  - **max_tokens 截断**：`min(max_tokens, GATEWAY_MAX_GENERATION_TOKENS)`（默认 8192）——**关键排障点**：claude-code 默认请求 max_tokens=32000，远超 vLLM 的 max_model_len（如 16384），会导致 vLLM 400 "maximum context length is negative"；这里截断到 8192 与 vLLM 协调（prompt 余量 = max_model_len − cap）；
  - `stop_sequences` → 内部 `stop`，`cache_control` 忽略。
- **`openai_build_response` / `anthropic_build_response`**：把 session 的 `GenerationOutcome` 序列化成对应协议的响应体；
- **`openai_stream_response` / `anthropic_stream_response`**：把 outcome 转成对应协议的 **SSE 流式响应**（OpenAI 的 `data: {...}` chunk、Anthropic 的 `message_start` / `content_block_delta` 等事件）。

**（2）Actor 层（`gateway.py` 的 `_GatewayActor`）**：基于 FastAPI 的 HTTP 服务，`_register_routes()` 注册三个端点：`POST /sessions/{session_id}/v1/chat/completions`（OpenAI）、`POST /sessions/{session_id}/v1/messages`（Anthropic）、`POST /sessions/{session_id}/reward_info`（reward 上报）。每个端点的处理流程（以 OpenAI 为例，`_handle_openai_chat_completions`）：

```python
async def _handle_openai_chat_completions(self, session_id, payload):
    session = self._sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    try:
        internal = openai_to_internal(          # ① 协议 → 内部 canonical
            payload,
            base_sampling_params=dict(session.sampling_params),   # session 默认采样参数
            allowed_sampling_keys=self._allowed_request_sampling_param_keys,  # 白名单
        )
        _validate_sampling_params(internal["sampling_params"])
    except MalformedRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    outcome = await session.run_generation(internal, self._backend)   # ② 会话处理（见下）
    model = str(payload.get("model") or "unknown")
    if payload.get("stream") is True:
        return openai_stream_response(outcome, model=model)           # ③ 流式
    return JSONResponse(openai_build_response(outcome, model=model))  # ③ 非流式
```

**这段代码关键在哪**：Actor 层是"不同协议请求"的统一入口——**两种协议都经过"`*_to_internal` 降级 → `session.run_generation` 处理 → `*_stream_response`/`*_build_response` 序列化"**，协议差异被适配器吸收，session 只面对统一的 `InternalGenerationRequest`（`messages` / `tools` / `sampling_params` 三个字段）。这保证了"任意 agent（OpenAI 或 Anthropic 协议）→ 同一套轨迹物化逻辑"。

**（3）会话层（`session/session.py` 的 `GatewaySession`）——轨迹的诞生地，最关键**。核心数据结构是 **`TrajectoryBuffer`**（一条轨迹的 token 级内容全在这里）：`prompt_ids`（初始 prompt）、`response_ids`（累积的响应 token）、`response_mask`（1=模型输出、0=续轮插入的上下文 token）、`response_logprobs`（与 response_ids 对齐，插入上下文处为 0.0）、`routed_experts`、`generation_versions`（每轮生成的权重版本跨度，供 off-policy 判定）。会话的状态机：**ACTIVE（可接收生成请求）→ FINALIZED（轨迹已返回、会话关闭）/ ABORTED（取消、不产生轨迹）**。每个生成请求的处理在 `run_generation()` 里走四个阶段：

**阶段 A（prepare：编码 + 选链 + 容量检查）**——`_prepare_generation_inputs()`，逐步操作：**①** 对 incoming 消息计算**消息前缀哈希序列**（每加一条消息算一个哈希）；**②** 调 `_select_chain()` 用这些哈希匹配已有 chain（详见下方 Chain 机制）；**③** 若匹配到已有 chain（续链），先 `_copy_trajectory_buffer` 复制该 chain 的 buffer 作为本次的起点（保留已累积的响应 token），再算 `incremental_start`（增量起点）；若需要回滚（rollback），则**从 `last_assistant_start` 截断 buffer**（删掉上一次 assistant 响应的 response_ids/mask/logprobs 及对应的 generation_versions 标记）；**④** 对增量消息做**增量编码**（`encode_incremental`），把新增的上下文 token 追加到 buffer 的 `response_ids`，**mask 标 0、logprob 填 0.0**（这些是"插入的上下文"不是模型输出）；**⑤** 容量检查（`max_trajectory_length`），超限则把 `sampling_params.max_tokens` 压到剩余容量、或直接标记 `capacity_exhausted`（产生空响应轨迹并关闭链）；**⑥** 快照 `last_assistant_start`（记录本次生成前 buffer 长度和消息长度，供回滚用）。

**阶段 B（generate：调后端并追加响应）**——调用 `backend.generate()`（即云侧 vLLM）生成响应 token ids；若请求了 logprobs，**校验 logprob 与 token 逐位对齐**（数量必须一致，否则报错）后，把 response_ids 追加到 buffer（mask 全标 1）、logprobs 逐位追加；`routed_experts`（若后端返回）记录最新值；`generation_versions` 追加一个 `(min_global_steps, max_global_steps)` 标记。

**阶段 C（commit：写入 chain）**——`_commit_generation_to_chain()`：把本次的 message_history（incoming + assistant 响应）和更新后的 buffer 写回 chain；若这是新 chain，分配 `chain_id`、记 `order_seq`（物化顺序）；若是续链，更新该 chain 的 message_history / buffer / tip hash。**一个请求对应一条 chain 的一次更新**。

**阶段 D（finalize：物化轨迹）**——会话结束时 `finalize()`：`_materialize_active_chains()` 把每条 active chain 用 `_build_materialized_trajectory()` 转成 `Trajectory`（含 prompt_ids / response_ids / response_mask / response_logprobs / reward_info / num_turns / routed_experts / multi_modal_data / extra_fields），按 `order_seq` 排序返回；`num_turns = 消息里 user/assistant 轮数 + 1`；`min/max_global_steps` 从各轮 `generation_versions` 折叠成轨迹级版本跨度。

**Chain 机制（多轮轨迹怎么组织，关键设计）**：一次会话可以有**多条 chain**（agent 分支/重试会产生新 chain），每条 chain 是"一条独立可训练轨迹的构建现场"。`_select_chain()` 的匹配逻辑：用 incoming 消息的前缀哈希与 active chain 的 `message_tip_hash` 比对——**若 incoming 消息是某 chain 消息历史的前缀延伸（续链），匹配到该 chain**；**若 incoming 消息与某 chain 的"最后一次 assistant 之后"的部分不一致（即 agent 重写/回退了上一次的 assistant 响应），触发 rollback**：截断上次 assistant 的 token（从 `last_assistant_start` 记录的截断点删起），从该点重新生成，被丢弃的 trainable token 计入 `rollback_dropped_trainable_tokens` 统计。这保证**被回滚的 assistant 输出不会残留在训练轨迹里**（否则会让"未采纳的中间尝试"也参与 loss）。

**子 agent 轨迹：官方不支持（重要澄清，面试易问）**——用户可能误以为 uni-agent 支持"子 agent 轨迹树"，实际**官方没有子轨迹的数据结构与机制**：

- **`Trajectory` 是扁平结构**（`gateway/session/types.py`）：只有 `prompt_ids / response_ids / response_mask / response_logprobs / reward_info / num_turns / extra_fields` 等，**没有 `children` / `parent` 字段**——不存在"子轨迹"的表达能力；
- **Gateway session 内是"线性链"而非树**（`session.py` 的 `ChainState`，docstring 明写 "One active **linear** trajectory chain"）：多条 chain（`chain_id`）是**轨迹断裂/回滚后分叉重开的线性链**（每条链物化一个 Trajectory），不是子 agent 嵌套；
- **verl agent_loop 无 sub-agent**（`verl/verl/experimental/agent_loop/` 的 `ToolAgentLoop`）：只有 `max_user_turns / max_assistant_turns`（多轮）、`max_parallel_calls`（**并行工具调用**）——全仓库搜 `multi_agent / orchestrator / sub_agents` 零命中；
- **官方 `claude_code` agent 默认禁用 Claude Code 的 subagent 能力**（`uni_agent/agents/claude_code/agent.py`）：`CLAUDE_CODE_FORK_SUBAGENT: "0"` + disallowedTools deny-list 含 Subagent；虽然配置了 `CLAUDE_CODE_SUBAGENT_MODEL`（pin 到主 model，因为 Claude Code 的 background/subagent 调用会打到 haiku+subagent 槽位，不 pin 则 vLLM 404）——**即使打开 fork，子代理调用也走主 session 的同一个 base_url（Gateway），作为主 session 内的事件记录，不产生独立子轨迹**；
- 结论：**"子 agent 轨迹"在官方 uni-agent 里没有落点**（数据结构扁平 + session 线性链 + 无多 agent 编排），子代理要么被禁（官方默认），要么其调用混入主 session 事件、无法作为独立轨迹训练——如果要支持多 agent/子代理轨迹，需要自研扩展（如每个子代理开独立 session + 元数据关联）。

**（4）MessageCodec（`session/codec.py`）——消息 ↔ token 的编解码**：`encode_full`（完整 prompt 编码成 token ids）、`encode_incremental`（增量消息编码，续轮用）、`canonicalize_message_for_prefix_comparison`（消息规范化用于前缀哈希比较）。codec 还负责**工具调用标记的解码**（`decode_response`：把模型输出的 token ids 解码成 assistant 消息，识别工具调用标记）和多模态数据提取（图片/视频 URL → 张量，`_codec.extract_multi_modal_data`）。

**（5）多模态支持**：`GatewaySession` 的 `_prepare_generation_inputs` 通过 `_codec.extract_multi_modal_data(messages)` 提取图片/视频数据，`TrajectoryBuffer` 和 `Trajectory` 都有 `multi_modal_data` 字段；`framework/multi_modal_postprocess.py` 的 `compute_multi_modal_inputs` / `compute_position_ids` 在转换 TQ 字段时处理多模态（对应 verl 的多模态输入格式）。

**（6）reward 接入**：`set_reward_info()` 通过会话的 reward-info 端点接收外部 reward（如沙箱执行结果），finalize 时写入所有 trajectory 的 reward_info。

**核心代码**（`gateway/session/session.py` 的 `run_generation`——Gateway 处理一个请求的真实骨架）：

```python
async def run_generation(self, request: InternalGenerationRequest, backend) -> GenerationOutcome:
    """Run one provider-normalized generation request and return its business outcome."""
    async with self.request_lock:
        encoded = await self._prepare_generation_inputs(request)   # 阶段A: 编码+选链+容量
        if encoded.capacity_exhausted:
            return GenerationOutcome(assistant_msg=empty_msg, finish_reason="length", ...)
        if encoded.chain_id is not None:
            self.reserved_chain_ids.add(encoded.chain_id)          # 预留链，防并发抢占

    output = await backend.generate(                               # 阶段B: 调 vLLM
        request_id=self.handle.session_id, prompt_ids=encoded.context_ids,
        sampling_params=encoded.sampling_params, ...)
    response_ids = list(output.token_ids)
    encoded.buffer.response_ids.extend(response_ids)
    encoded.buffer.response_mask.extend([1] * len(response_ids))   # 模型输出 mask=1
    if encoded.sampling_params.get("logprobs", False):
        log_probs = list(output.log_probs)
        if len(log_probs) != len(response_ids):
            raise RuntimeError("backend logprobs must align with token_ids: ...")
        encoded.buffer.response_logprobs.extend(log_probs)         # logprob 逐位对齐校验

    async with self.request_lock:
        assistant_msg, finish_reason = await self._codec.decode_response(response_ids, ...)
        self._commit_generation_to_chain(encoded, assistant_msg)   # 阶段C: 写入 chain
        return GenerationOutcome(assistant_msg=assistant_msg, finish_reason=finish_reason, ...)
```

**这段代码关键在哪**：`run_generation` 就是四阶段的代码落点——①prepare（`_prepare_generation_inputs`）在锁内做选链/容量检查；②generate 调 backend（vLLM）并**逐位校验 logprob 与 token 对齐**（数量不一致直接 RuntimeError，这是 on-policy 数据质量的硬保证）；③commit 写回 chain。`request_lock` 保证同一会话的请求串行化。

**核心数据结构**（`session/session.py` 的 `TrajectoryBuffer`——一条轨迹的 token 级内容全在这里）：

```python
@dataclass
class TrajectoryBuffer:
    """Mutable token buffer for the active trajectory under construction."""
    prompt_ids: list[int]                    # 初始 prompt token ids
    response_ids: list[int]                  # 累积的响应 token（含续轮插入上下文）
    response_mask: list[int]                 # 1=模型输出，0=续轮插入的上下文
    response_logprobs: list[float]           # 与 response_ids 对齐；上下文处为 0.0
    routed_experts: Any = None               # 后端返回的专家路由（MoE）
    generation_versions: list[tuple] = ...   # 每轮 (min_global_steps, max_global_steps)
```

**这段代码关键在哪**：`TrajectoryBuffer` 的字段直接对应"轨迹级训练样本"的组成——`response_mask` 区分模型输出（参与 loss）与插入上下文（不参与），`response_logprobs` 是 GRPO 的 on-policy 依据，`generation_versions` 是 off-policy 判定的版本跨度。这个 dataclass 就是前面所有阶段操作的对象。

### 3. Reward 分配：从沙箱判定到训练样本的完整路径

**现有问题：reward 是怎么"分配"到轨迹上的？** 一条多轮轨迹只有一个标量 reward（如沙箱 pytest 通过=1.0），但 verl 训练需要**token 级**的 reward 信号。这里的关键问题是：**这个标量 reward 如何变成 token 级分数、打在哪些 token 上、多轨迹/多轮时怎么分配？** 如果分配错了，训练信号就会错位（比如把 reward 打在插入的上下文 token 上、或让被回滚的中间尝试也吃到 reward）。uni-agent 的 reward 分配分四步，逐步操作如下。

**第 1 步（源头：沙箱/runner 产出标量 reward）**：agent 会话结束后，执行环境（沙箱）判定结果——如 SWE-bench 的 pytest 通过与否。runner 通过 HTTP `POST /sessions/{session_id}/reward_info` 把 `reward_info` 发布到 Gateway 会话（`reward` 字段为标量分数，可带 `finished` 等附加信息；`task_runner._post_reward_info` 负责发布）。

**第 2 步（Gateway 存到轨迹）**：`set_reward_info()` 把 reward_info 存到 session；`finalize()` 时 `replace(trajectory, reward_info=dict(self.reward_info))` 把 reward_info **复制到该会话物化的每一条 Trajectory**（同值分配——同一会话的所有轨迹共享同一份 reward_info）。

**第 3 步（Framework 打分：两种来源、一个优先级）**——`framework.py` 的 `_score_from_reward_info`（对应源码注释："Prefer the reward the runner posted to the session; otherwise defer to the RewardLoopWorker, else rm_scores stays 0"）：

- **来源 A（默认，runner 发布）**：从 `session_trajectories[-1].reward_info`（最后一条轨迹的 reward_info）里 pop 出 `reward` 字段 → `float(reward)` 作为 reward_score；其他字段（如 `acc`）作为 `reward_extra_info` 携带，`finished` 被丢弃（它是完成事实不是 reward 指标）。**每个轨迹各复制一份 `(score, extra_info)`**。
- **来源 B（兜底，RewardLoopWorker）**：若 runner 没发布 reward（`reward is None`），且配置了 `reward_loop_worker_handles`，则走 `_score_trajectories`：**只把会话的最后一条轨迹（final trajectory）送给 RewardLoopWorker 打分**，然后把 `(score, extra_info)` **广播（broadcast）给会话里的所有轨迹**（源码："only the final trajectory is dispatched to RewardLoopWorker; its score + reward_extra_info are then broadcast to every trajectory in the session"，对齐 verl 的 `AgentLoopWorkerTQ._agent_loop_postprocess`）。
- **都拿不到**：reward_score 保持 None，TQ 写入时 rm_scores 全 0（日志 warning "no reward available; rm_scores=0"）。

**第 4 步（Framework 落成 token 级 rm_scores）**：`_trajectory_to_tq_field_and_tag` 中：`rm_scores = zeros_like(responses)`，若 `reward_score` 非空则 **`rm_scores[-1] = reward_score`**——**reward 只打在响应序列（response_ids）的最后一个 token 上**，前面的 token 都是 0。这样写入 TQ 后，verl 训练侧 `data["rm_scores"].sum(dim=1)` 得到每个样本的序列级分数（只有一个非零位 = reward_score），再经 GRPO 的 advantage 计算（组内相对：当前样本分数 − 组内均值 / 组内 std）变成训练梯度信号。

**分配的关键设计**：① **同值广播**——一个会话（一个任务）只有一个 reward，分配给该会话所有轨迹/所有轮次（"identical assignment"思想，对应 Agent Lightning 的 `TracerTraceToTriplet` 同值分配策略，源于 arXiv:2508.03680）；② **末位落点**——reward 打在 response 序列最后一个 token，因为 GRPO 是"整条响应一个分数"，token 级分数在 loss 里通过 mask 只对模型输出 token 生效；③ **finished 语义**——未完成 episode（`finished=false`）时，若开启 `_mask_unfinished_episode`，整个 response_mask 置 0（该样本不参与 loss），避免半截轨迹污染训练。

**具体数值样例**：沿用前面的 3 次请求会话（1234-token 轨迹），假设沙箱判定测试通过 → reward=1.0：

```text
第 1 步：沙箱 pytest 全过 → runner POST /sessions/{sid}/reward_info
        reward_info = {"reward": 1.0, "finished": true}
第 2 步：finalize 时 reward_info 复制到会话的唯一一条 Trajectory：
        trajectory.reward_info = {"reward": 1.0, "finished": true}
第 3 步：_score_from_reward_info：
        reward = 1.0（从 reward_info 取出）
        reward_score = 1.0；reward_extra_info = {}；finished 被丢弃
        → trajectory.reward_score = 1.0
        （若 runner 没发布 reward，则 RewardLoopWorker 给 final trajectory
          打分后广播；都没有则 rm_scores 全 0）
第 4 步：_trajectory_to_tq_field_and_tag：
        rm_scores = zeros(1234,)
        rm_scores[-1] = 1.0   # reward 打在响应最后一个 token
        → TQ 字段 rm_scores = [0, 0, ..., 0, 1.0]（1234 个）
verl 侧：data["rm_scores"].sum(dim=1) = 1.0（该样本的序列分数）
        → GRPO advantage = (1.0 − 组内均值) / 组内 std
```

**对比负样本/多轨迹场景**：若一个会话物化出 2 条 Trajectory（如 agent 分支出两条完整路径），`_score_from_reward_info` 给**每条**都赋同一个 reward_score（同值广播）——两条轨迹分别作为独立训练样本，各打 1.0 或 0.0（取决于最终 reward）；RewardLoopWorker 场景则只给 final trajectory 打分再广播到所有轨迹。**被 rollback 截断的中间尝试（300 tokens）不产生轨迹，因此完全吃不到 reward**——这是"只有最终采纳的交互参与训练"的 reward 侧保证。

> **面试一句话总结**：reward 分配是"标量 → token 级"的转换：沙箱/runner 把标量 reward 经 session 的 reward_info 端点发布 → finalize 同值复制到会话所有轨迹 → Framework 优先取 runner 发布的 reward（否则 RewardLoopWorker 给最后一条轨迹打分后广播）→ 落成 `rm_scores[-1] = reward_score`（reward 打在响应最后一个 token）→ verl 侧 sum 得序列分数进 GRPO advantage；未完成 episode 和被 rollback 的中间尝试不参与训练。

### 4. 具体数值样例：逐请求演算 Gateway 如何物化一条轨迹

**逐请求演算**一个 agent 会话如何被 Gateway 逐步处理成轨迹。假设白盒 mini-swe-agent 处理一个 HumanEvalFix 任务，会话中调用了 3 次模型，每次调用就是一个"请求"（Gateway 的 `run_generation`），我们逐请求跟踪 `TrajectoryBuffer` 和 chain 的变化。

**请求 #1（agent 首次生成，prompt=题目 + 初始上下文，共 1273 tokens）：**

```text
【阶段 A - prepare】
  · 对 incoming 消息（system + user 题目）计算前缀哈希序列 H = [h_sys, h_user]
  · _select_chain：active_chains 为空 → 无匹配 → 新建 chain（chain_id=0）
  · 无增量消息可编码 → prompt_ids = 1273 tokens（编码后的题目上下文）
  · 容量检查：1273 < max_trajectory_length（如 20480）→ 通过
  · 快照 last_assistant_start（记录此时 buffer 长度）
【阶段 B - generate】
  · 调 vLLM 生成 300 tokens（模型输出，含工具调用标记如 <tool_call>）
  · 校验 300 个 logprobs 与 300 个 token 对齐 → 通过
  · buffer.response_ids  += [t_1..t_300]，mask 全标 1
  · buffer.response_logprobs += [lp_1..lp_300]
  · generation_versions += [(step_5, step_5)]   # 当前权重版本
【阶段 C - commit】
  · message_history = [system, user, assistant(300t)]
  · chain #0 建立：buffer=当前 buffer、tip_hash = H + h_assistant300
【agent 动作】执行工具（运行代码）→ 失败 → 准备第二轮，请求 #2 到来
```

此时 `chain #0.buffer`：`prompt_ids(1273) / response_ids(300) / mask=[1]*300 / logprobs(300)`。

**请求 #2（agent 带工具报错继续，incoming = 历史 + 工具报错 + "请修改"，message 数从 3 增到 5）：**

```text
【阶段 A - prepare】
  · 计算 incoming 前缀哈希 H' = [h_sys, h_user, h_assistant300, h_tool_err, h_fix_req]
  · _select_chain：H' 前 3 个哈希与 chain #0 的 tip_hash 延伸匹配
    → 匹配到 chain #0（续链），无 rollback
  · _copy_trajectory_buffer：复制 chain #0 的 buffer 作为起点
    （已有 1273 + 300 tokens 保留）
  · incremental_start = 3（消息从第 3 条之后是新增）
  · 对增量消息（tool_err + fix_req）encode_incremental → 150 tokens
  · 追加到 buffer.response_ids 尾部，**mask 标 0、logprob 填 0.0**
    （这些是"插入的上下文"，不是模型输出，不参与 loss）
  · 容量检查：1273+300+150 = 1723 < 20480 → 通过
【阶段 B - generate】
  · vLLM 基于完整上下文（1273 prompt + 450 已有 response）生成 400 tokens
  · buffer.response_ids += 400（mask 标 1），logprobs += 400
  · generation_versions += [(step_5, step_5)]
【阶段 C - commit】
  · message_history 更新为 6 条（新增 assistant(400t)）
  · chain #0 更新：buffer 现在 response_ids = 300+150+400 = 850 tokens
```

此时 `chain #0.buffer`：`response_ids(850) / mask=[1]*300 + [0]*150 + [1]*400`（**混合 mask**）、`logprobs=[lp]*300 + [0.0]*150 + [lp]*400`。

**请求 #3（agent 修正后再试，incoming = 历史 + 新工具报错 + "再改"）：**

```text
【阶段 A - prepare】
  · 前缀哈希匹配 chain #0 → 续链
  · 增量消息 encode_incremental → 117 tokens，追加（mask 0、logprob 0.0）
  · response_ids = 850 + 117 = 967
【阶段 B - generate】
  · vLLM 生成 267 tokens（mask 1，logprobs 267 个）
  · response_ids = 967 + 267 = 1234；generation_versions += [(step_5, step_5)]
【阶段 C - commit】chain #0 更新（message_history 再 +1 条 assistant）
```

**会话结束（沙箱判定 reward）：** `set_reward_info({finished: true})` → `finalize()` → `_materialize_active_chains()`：

```text
物化 chain #0 → 1 条 Trajectory：
  prompt_ids(1273) / response_ids(1234) / response_mask(1234，1/0 混合) /
  response_logprobs(1234，含 267 个 0.0) / num_turns=3 /
  extra_fields{min_global_steps:5, max_global_steps:5} /
  reward_info{finished:true}
```

**再加一个 rollback 场景**（体现"被回滚的中间尝试不残留"）：假设请求 #2 不是续链，而是 agent **重写**了请求 #1 的 assistant 响应（incoming 的前缀哈希在"最后一次 assistant 之后"与 chain #0 不一致）：

```text
【阶段 A - prepare】（rollback 分支）
  · _select_chain 检测到 incoming 与 chain #0 的 last_assistant 不一致
    → rollback_to_last_assistant = true
  · 从 last_assistant_start 截断：删掉请求 #1 的 300 个 response tokens
    （mask/logprobs 一并删，generation_versions 弹出最后一次标记）
    → buffer 回到只有 prompt_ids(1273) 的状态
  · rollback_dropped_trainable_tokens += 300（被丢弃的 trainable token）
  · 重新 encode_incremental → 生成新响应……
  · 最终物化的轨迹**不包含**被回滚的 300 tokens —— 只有新生成的
```

**关键理解**：3 次模型调用没有变成 3 条独立样本，而是**物化成 1 条 1234 token 的多轮 Trajectory**（mask 区分模型输出 1 与插入上下文 0，插入上下文不参与 loss）——这就是"轨迹级"训练样本：verl 对整条轨迹做 GRPO，reward 由沙箱的真实执行结果决定（Framework 转换时 `rm_scores[-1] = reward_score`），而不是按每轮拆开；同时 chain + rollback 机制保证**只有最终采纳的交互序列**进入训练数据，agent 重写的中间尝试被精确截断。

> **面试一句话总结**：Gateway 是 uni-agent 的核心——它把任意 OpenAI/Anthropic 兼容 agent 统一接入（协议适配 + 会话路由），并在服务模型的同时通过 session 的 chain 机制物化 token 级轨迹（prompt/response/mask/logprob/reward，on-policy 由云侧保证）；多轮交互被物化成一条带 mask 的 Trajectory，reward 经"标量→末位 token"分配后成为 verl 训练样本的直接来源。

---

## 2. Agent（官方抽象）：统一驱动任意 agent 的决策循环

### 1. 现有问题：为什么需要 Agent 抽象

Gateway 解决"轨迹怎么物化"，但**谁来产生这些交互**？——是 agent 本身。Uni-agent 面对的 agent 形态差异巨大：白盒 `mini-swe-agent`（可改源码、可注入 hook）、黑盒 `Claude Code`（闭源、只能通过 CLI 驱动、工具调用必须转发）、外部自定义 agent（任意 OpenAI 兼容 harness）。如果每种 agent 都单独写一套接入代码，工作量巨大且不可复用。Uni-agent 的 Agent 抽象解决"**用统一的方式驱动任意 agent**"：定义 agent 的决策循环（观察→推理→行动→观察…）与 Gateway 交互的标准方式，让"接入新 agent"变成"实现一个 runner"。

另一个关键问题是**黑盒 agent 的工具调用**：Claude Code 这类黑盒无法像白盒那样直接注入沙箱 API，它的 Bash/Read/Write/Edit 等工具调用发生在本地，必须被"转发"到云端沙箱执行——这需要一套工具转发机制（详见下文）。

### 2. 方法论：官方 Agent 抽象是怎么设计的

**`uni_agent/agents/`** 提供 **`base.py`**（Agent 基类与注册机制）和 **`registry.py`**，以及三个内置 agent：`mini_swe_agent/`（白盒 SWE agent）、`claude_code/`（黑盒 Claude Code）、`react/`（ReAct 风格的推理 agent，含 model 封装）。它们遵循统一接口：agent 的每次模型调用指向 Gateway（改 `base_url`），执行环境指向沙箱。

官方还定义了 **`AgentRunner` 协议**（`framework/base.py`）：一个 runner 是"驱动一种 agent 跑一条轨迹"的可调用对象，通过配置声明 `runner_fqn`（指向 runner 函数的全限定名）、`dispatch_mode`（如 `ray_task`，用 Ray 分发）、`max_concurrent_sessions`（并发会话上限）。**接入新 agent 的本质 = 实现一个 runner 函数 + 在配置里注册**。runner 函数的签名统一为 `async def runner(*, raw_prompt, session, sample_index, tools_kwargs, ...) -> None`——`session` 是 Gateway 的 `SessionHandle`（含 base_url 和 reward_info_url），runner 负责：驱动 agent 执行（模型调用指向 session.base_url）→ 完成后评估 reward → `POST session.reward_info_url` 上报。

**官方内置 runner 的通用流程**（以官方 mini_swe_agent / claude_code 为例）：**① 解析任务**（`extract_task`：从 raw_prompt / tools_kwargs 提取 issue、测试列表、instance_id）；**② 建沙箱**（`create_task_sandbox`：按 image 建执行环境）；**③ 跑 agent**（`build_agent_command` 构建调用命令，模型端点指向 Gateway）；**④ 评估 reward**（`evaluate_reward`：在沙箱内跑测试）；**⑤ 上报并清理**（POST reward_info → 停沙箱）。这五个步骤是官方 runner 的骨架，也是扩展 runner 的模板。

**核心代码**（`framework/base.py` 的 `AgentRunner` 协议——"接入新 agent = 实现一个 runner"的代码基础）：

```python
class AgentRunner(Protocol):
    """Protocol for a callable that drives one agent rollout."""

    async def __call__(
        self,
        *,
        raw_prompt: Any,              # 原始 prompt（str 或消息列表）
        session: SessionHandle,       # Gateway 会话句柄（base_url + reward_info_url）
        sample_index: int,            # 当前样本在 batch 中的下标
        tools_kwargs: dict | None = None,   # 任务配置（image/测试/文件等）
        **_: Any,
    ) -> None:
        """驱动 agent 执行 → 评估 reward → POST reward_info → 清理。"""
        ...
```

**runner 注册配置**（训练配置里声明用哪个 runner、怎么分发）：

```yaml
agent_runners:
  mini_swe_agent:
    runner_fqn: uni_agent_ext.agents.mini_swe_agent_runner.mini_swe_agent_runner
    dispatch_mode: ray_task          # 用 Ray 分发（支持并发）
    max_concurrent_sessions: 4       # 并发会话上限
    runner_kwargs:
      max_turns: 60
```

**这段代码关键在哪**：`AgentRunner` 是 Protocol（鸭子类型）而非抽象基类——**任何满足 `async def __call__(*, raw_prompt, session, sample_index, tools_kwargs, ...)` 签名的函数都能当 runner**，配合 `runner_fqn`（全限定名）注册，接入新 agent 就是"写一个函数 + 在 yaml 里声明"。`session` 参数把 Gateway 的 base_url / reward_info_url 交给 runner，是"agent 与 Gateway 交互"的唯一通道。

### 3. 具体数值样例

以官方黑盒 Claude Code runner 的接入方式为例，逐步演算"agent + Gateway + 沙箱"的交互：

```text
第 1 步：Claude Code 启动，模型调用指向 Gateway（Anthropic 协议端点）。
第 2 步：agent 决策要执行工具（如 Edit 修改代码）→ 工具调用被转发层
        捕获（官方场景：沙箱内直连，工具在沙箱本地执行）。
第 3 步：工具在云端沙箱执行（代码/测试），结果返回 agent 继续决策；
        期间每一次模型调用都经 Gateway 云侧物化轨迹（含工具调用标记
        mask=1、工具结果 mask=0）。
第 4 步：会话结束，沙箱判定 reward（如测试是否通过），
        finalize 物化整条轨迹 → TQ → verl 训练。
```

对比白盒：mini-swe-agent 是开源可注入 harness，直接用 E2B 环境类（如 `tencent_e2b`）attach 沙箱，无需工具转发层；黑盒多出的正是"工具转发"这一层工程——黑盒闭源、无法注入沙箱 API，它的 Bash/Read/Write/Edit 等工具调用必须被转发到沙箱执行，这是接入黑盒 agent 的核心难点（官方通过沙箱内直连解决，具体转发层实现可参考 `uni_agent/agents/claude_code/` 的 runner）。

> **面试一句话总结**：官方 Agent 抽象 = AgentRunner 协议 + 五步 runner 模板（解析任务 → 建沙箱 → 跑 agent → 评估 reward → 上报清理），接入新 agent 就是实现一个 runner 函数并在配置注册；白盒 harness 直接注入沙箱环境类，黑盒需要工具转发层。

### 4. Agent 部署位置：官方 in-sandbox vs 本项目 agent-outside（关键对比）

**官方 uni-agent 的黑白盒 agent 默认都在沙箱内部启动**（源码 + 官方文档双重确认）：

- **Agent 抽象语义**（`uni_agent/agents/base.py`）：`Agent.run(*, sandbox, messages)`——"Every Agent receives a **started Sandbox**"，agent 拿到已启动的沙箱、**在沙箱内**解决任务；官方概念文档（`docs/source/concepts/agent.md`）明写："Black-box Agent: an external harness owns the loop and **runs as a process inside the Sandbox**"、"Ensure the external CLI is installed **inside the Sandbox**. Launch the harness through `sandbox.exec()`"；
- **官方黑盒（Claude Code）**（`examples/blackbox_recipes/claude_code/claude_code_runner.py`）：sidecar 工具镜像（`@anthropic-ai/claude-code` npm 包，`FROM scratch` 根 `/opt/claude-code`）挂载进 SWE-bench 沙箱 → 沙箱内执行 `/opt/claude-code/bin/claude -p <task> --permission-mode bypassPermissions` → `ANTHROPIC_BASE_URL = http://127.0.0.1:<proxy_port>`（**沙箱内隧道**，`rewrite_gateway_url` 把 Gateway URL 重写为沙箱本地）→ claude 的 **Bash/Edit/Read/Write 工具就地 fork 执行**（agent 进程与文件系统/代码执行同处一个沙箱），不需要 MCP；
- **官方白盒（mini-swe-agent）**（`uni_agent/agents/mini_swe_agent/agent.py`）：注释明写 "black-box agent launched **inside the sandbox**. NOT YET IMPLEMENTED"——计划把 mini-swe-agent 装进沙箱 venv、`sandbox.exec()` 启动；
- **Gateway 的角色**：不是 agent 宿主，而是**模型端点代理 + 轨迹记录器**（session-scoped `base_url` 注入 `agent.model`）；agent 也可以不走 Gateway 直连外部模型 API（"External API inference bypasses the Gateway"）。

**本项目（uniagent-lighting）是反向改造：agent 在外部（用户侧/本地），沙箱仅负责执行**：

| 维度 | 官方 uni-agent（in-sandbox） | 本项目平台化（agent-outside） |
|---|---|---|
| agent 进程位置 | **沙箱内部**（claude 装沙箱 / 白盒计划装沙箱） | **用户侧/本地 WSL**（任意位置） |
| 黑盒工具执行 | claude 自带工具**就地执行**（无需 MCP） | 内置工具 `--disallowedTools` 禁用 + **手写 stdio JSON-RPC MCP 转发层**把 Bash/Edit 转发到云端沙箱 |
| 白盒工具执行 | harness 装沙箱，`sandbox.exec()` 启动 | harness 在本地，**环境类 `Sandbox.connect(attach_instance_id)` attach 沙箱**远程执行 |
| 模型调用 | 沙箱内隧道 → Gateway | 本地 → SSH 隧道（paramiko direct-tcpip）→ Gateway |
| 沙箱职责 | agent 宿主 + 执行环境（一个沙箱全包） | **仅执行**（代码/测试/reward 判定） |
| 一句话 | 官方管"agent 进沙箱、工具就地跑" | 我们管"agent 出沙箱、工具远程转发/attach" |

> 面试关键表述：**官方 uni-agent 的黑白盒 agent 默认在沙箱内部启动（Agent.run(sandbox) 语义 + claude_code_runner 沙箱内跑二进制 + 沙箱内隧道连 Gateway，工具就地执行不需要 MCP）；我们把 agent 挪到外部做"训练、推理、环境、奖励解耦"，因此白盒用 Environment attach 沙箱、黑盒手写 MCP 工具转发层把工具调用送回沙箱——"官方管 agent 进沙箱，我们管 agent 出沙箱"。**

---

## 3. Sandbox（官方抽象）：执行环境的多后端封装

### 1. 现有问题：为什么需要独立的沙箱

Agent 的"思考"（决策、生成）和"行动"（执行代码、跑测试）是两类完全不同的工作：思考需要模型算力，行动需要安全隔离的代码执行环境。如果让 agent 直接在训练机或本地执行任意代码，有**安全风险**（不可信代码可能破坏环境）和**耦合问题**（执行环境绑定 agent 位置）。官方 uni-agent 的定位是"**沙箱承载执行**"（代码执行、测试、reward 判定都在沙箱里完成），且官方默认把 agent 也放进沙箱（见第 2 点第 4 小节）；**本项目进一步把 agent 挪到外部**，让沙箱只负责执行、agent 在任意位置——这样 agent / 沙箱 / 训练可以独立扩缩，沙箱也可以换成任何后端。

### 2. 方法论：官方 Sandbox 抽象是怎么设计的

**`uni_agent/sandbox/`** 提供 **`base.py`**（`Sandbox` 基类 + 执行接口）、**`registry.py`**（后端注册 + `build_sandbox(config)` 工厂），以及多种后端实现：`docker.py`（Docker 容器）、`local.py`（本地执行）、`modal.py`（Modal 云）、`openyuanrong.py`、`vefaas.py`（云函数）等——统一抽象"在沙箱里执行命令/代码、读写文件"。创建方式是通过 `SandboxConfig(provider=..., image=..., runtime_timeout=..., sandbox_kwargs=...)` 交给 `build_sandbox` 工厂按 provider 实例化对应后端。

`Sandbox` 基类暴露的核心异步接口：`start()`（启动实例）、`exec_shell(command, timeout)`（执行命令）、`write_file(path, content)` / `read_file(path)`（文件读写）、`stop()`（销毁实例）。**换沙箱后端 = 换 provider + 实现同一套接口**——这是官方设计的扩展点：任何新的执行环境（云沙箱、函数计算等）只需实现 `Sandbox` 接口并在 `registry.py` 注册 provider 即可接入。

**核心代码**（`uni_agent/sandbox/base.py`——Sandbox 抽象 + 工厂，换后端只需实现这套接口）：

```python
class SandboxConfig(BaseModel):
    """沙箱配置：provider 决定用哪个后端，image 决定环境镜像。"""
    provider: str                      # docker / local / modal / vefaas / tencent_agent_runtime
    image: str                         # 环境镜像（如 swebench 实例镜像）
    runtime_timeout: float = 3600.0
    sandbox_kwargs: dict[str, Any] = {}

class Sandbox(abc.ABC):
    """Unified interface for sandbox execution environments."""

    async def start(self) -> None: ...        # 启动实例
    async def exec_shell(self, command: str, *, workdir=None,
                         timeout: int = 600) -> ExecResult: ...   # 执行命令
    async def read_file(self, path: str) -> bytes: ...            # 读文件
    async def write_file(self, path: str, content: bytes | str) -> None: ...  # 写文件
    async def stop(self) -> None: ...         # 销毁实例

    @classmethod
    def from_config(cls, config: SandboxConfig) -> Sandbox:
        """工厂：按 config.provider 实例化对应后端。"""
        return build_sandbox(config)   # registry 里按 provider 查找实现
```

**这段代码关键在哪**：`Sandbox` 用 `abc.ABC` 定义统一接口，`build_sandbox`（registry）按 `provider` 字符串实例化对应后端——**"换沙箱后端 = 实现同一套接口 + 注册 provider"** 是官方扩展机制（`registry.py` 维护 provider → 实现的映射）。`exec_shell` 返回 `ExecResult`（含 stdout/stderr/exit_code），reward 评估就是基于它解析 pytest 输出。

### 3. 具体数值样例

逐步演算一次"沙箱内执行并判 reward"的过程（SWE-bench 任务）：

```text
第 1 步：create_task_sandbox(image=sweb.eval.x86_64.django_1776_django-13447)
        → build_sandbox 按 provider 创建实例（如 docker 容器 / 云沙箱）。
第 2 步：沙箱内 /testbed 是题目仓库，swerex server 跑在 8000 端口。
第 3 步：agent 的每次工具调用（Edit/Write/Bash）落到沙箱执行。
第 4 步：会话结束，沙箱运行隐藏测试（test_patch 仅评估阶段使用，
        训练期 agent 只看题目——无测试泄露设计）。
第 5 步：测试通过 → reward=1.0；失败 → reward=0.0（或按通过用例比例）。
        该 reward 经 Gateway 的 set_reward_info 写入轨迹 → 训练信号。
```

> **面试一句话总结**：官方 Sandbox 抽象把"执行环境"封装成统一接口（start / exec_shell / read/write_file / stop），通过 `build_sandbox` 工厂按 provider 切换后端（Docker / 本地 / Modal / 云函数等），"换沙箱 = 实现一套接口"；沙箱只负责执行/测试/reward，与 agent 思考彻底解耦，且训练期无测试泄露。

---

## 4. Framework（官方编排层）：轨迹 → verl 训练样本的转换器

### 1. 现有问题：为什么需要 Framework 层

Gateway 物化出 Trajectory（token 级），但 **verl 训练消费的不是 Trajectory 对象，而是 TransferQueue 里的标准字段**（prompts / responses / input_ids / attention_mask / position_ids / response_mask / loss_mask / rm_scores / num_turns 等）。这中间有一层"翻译"：把 uni-agent 的轨迹格式转成 verl/TQ 的标准数据格式，同时处理各种边界情况（空响应轨迹、logprob 缺失、多模态输入、reward 定位、off-policy 版本标记）。Framework 层（`uni_agent/framework/`）就是这层翻译器——它把"agent 会话"编排成"verl 可消费的训练批次"。

Framework 还承担**任务编排**职责：`task_runner.py` 负责把训练 batch 的数据组织成 agent 任务、驱动 agent 执行、收集会话结果，形成"取 prompt → 跑 agent → 收轨迹 → 写 TQ"的完整循环（对应 verl 的 rollout 阶段在 agent 场景下的实现）。

### 2. 方法论：Framework 是怎么实现的

`uni_agent/framework/` 下：**`base.py`**（`AgentFramework` 基类 + `AgentRunner` 协议）、**`entry.py`**（入口组装）、**`framework.py`**（`OpenAICompatibleAgentFramework`，核心实现）、**`task_runner.py`**（任务循环）、**`multi_modal_postprocess.py`**（多模态后处理）。

`framework.py` 的核心方法 `_trajectory_to_tq_field_and_tag`（**轨迹 → TQ 字段**，逐步操作）：

**第 1 步（拼输入）**：`prompts = tensor(trajectory.prompt_ids)`、`responses = tensor(trajectory.response_ids)`；`input_ids = cat([prompts, responses])`（拼接成完整序列），`attention_mask = ones`。

**第 2 步（处理 mask）**：`response_mask` 来自轨迹（1=模型输出、0=续轮插入上下文）；若 `_mask_unfinished_episode` 且 reward 的 finished=false，则整个 response_mask 置 0（未完成 episode 不参与 loss）。

**第 3 步（position_ids）**：`compute_position_id_with_mask`（或带 processor 的多模态 `compute_position_ids`）；多模态时 `compute_multi_modal_inputs` 生成 multi_modal_inputs。

**第 4 步（logprob / 专家路由）**：若轨迹有 `response_logprobs` → 写入 `rollout_log_probs`（GRPO 需要）；若有 `routed_experts` → 对齐到 input_ids 长度写入。

**第 5 步（reward）**：`rm_scores = zeros_like(responses)`，若 `reward_score` 非空则 `rm_scores[-1] = reward_score`——**reward 打在响应序列最后一个 token 上**（详见第 1 点的 Reward 分配）。

**第 6 步（版本与元数据）**：`extra_fields` 写入 `min/max_global_steps`（权重版本跨度，off-policy 判定用），并弹掉 `materialization_reason` 等内部字段。

**核心代码**（`framework/framework.py` 的 `_trajectory_to_tq_field_and_tag`——轨迹 → TQ 字段的真实实现）：

```python
def _trajectory_to_tq_field_and_tag(self, *, trajectory, sample_fields,
                                    session_index, global_steps, uid):
    prompts = torch.tensor(trajectory.prompt_ids, dtype=torch.long)
    responses = torch.tensor(trajectory.response_ids, dtype=torch.long)
    source_response_mask = torch.tensor(trajectory.response_mask, dtype=torch.long)
    finished = trajectory.reward_info.get("finished")
    # 未完成 episode：整个 response_mask 置 0（不参与 loss）
    response_mask = (torch.zeros_like(source_response_mask)
                     if self._mask_unfinished_episode and finished is False
                     else source_response_mask)

    input_ids = torch.cat([prompts, responses], dim=0)       # 拼接完整序列
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)

    field = {
        "prompts": prompts, "responses": responses,
        "input_ids": input_ids, "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if trajectory.response_logprobs is not None:
        field["rollout_log_probs"] = torch.tensor(trajectory.response_logprobs, dtype=torch.float32)
    rm_scores = torch.zeros_like(responses, dtype=torch.float32)
    if trajectory.reward_score is not None and responses.numel() > 0:
        rm_scores[-1] = float(trajectory.reward_score)      # reward 打在最后一位
    field["rm_scores"] = rm_scores
    # extra_fields 携带 min/max_global_steps（off-policy 版本跨度）等
    field.update(dict(trajectory.extra_fields))
    return field, tag
```

**这段代码关键在哪**：`_trajectory_to_tq_field_and_tag` 就是"轨迹 → 训练样本"的代码落点——①`_mask_unfinished_episode` 分支对应"未完成 episode 不参与 loss"；②`rm_scores[-1] = reward_score` 对应"reward 打在响应最后一位"；③`input_ids = cat([prompts, responses])` + `position_ids` 就是 verl 消费的完整序列。这些字段（prompts/responses/rm_scores/num_turns 等）被 `tq.async_kv_batch_put` 写入 TransferQueue，verl 训练直接读取。

**写入 TQ**：`_write_session_trajectories_to_tq()` 遍历会话的所有轨迹，**跳过空响应轨迹**（`len(response_ids)==0`，如 max_trajectory_length 截断的占位——否则写入存储后端会因 0 字节 slice 被拒，导致整个 session 失败），重编号 index 保证 key 连续，然后 `tq.async_kv_batch_put(keys, fields, tags, partition_id)` 批量写入 TransferQueue。

**Reward 回填**：`_trajectory_to_reward_dataproto` 把轨迹转成 reward DataProto；`_score_from_reward_info` 从 reward_info 提取分数（见第 1 点）。

### 3. 具体数值样例

沿用第 1 点逐请求物化的轨迹（prompt 1273 tokens、response 1234 tokens、3 次请求累积、mask 混合），转成 TQ 字段：

```text
prompts    = tensor(1273,)          # 题目 + 上下文（请求 #1 的 prompt）
responses  = tensor(1234,)          # 3 次请求累积：300+150+400+117+267
                                   #   （含续轮插入的上下文 150+117）
input_ids  = cat = tensor(2507,)    # 1273 + 1234
attention_mask = ones(2507,)
position_ids   = compute_position_id_with_mask(input_ids) → tensor(2507,)
response_mask  = [1]*300 + [0]*150 + [1]*400 + [0]*117 + [1]*267
              # 逐请求的 mask 拼接：模型输出=1，续轮插入上下文=0
rollout_log_probs = tensor(1234,)  # GRPO 需要；插入上下文处为 0.0
rm_scores    = zeros(1234,); rm_scores[-1] = reward_score  # reward 打最后一位
num_turns    = tensor(3)            # int64，verl padding 消费
min/max_global_steps = (step_5, step_5)  # 3 次请求都在权重 step_5 生成

→ tq.async_kv_batch_put(key=f"{uid}_{session_index}_0", fields=..., tags=...)
```

verl 侧消费（`verl/trainer/ppo/v1/trainer_base.py`）：`kv_batch_get` 读 batch → `upsample_batch_to_divisible_size` 补 padding → `padding_utils.py` 构造 padding 行（这里有个已知的跨组件类型一致性问题：padding 行的 `num_turns` 以 Python int 序列化 vs 训练端按 int64 读取，需要保证 msgpack 类型一致）→ GRPO 训练。

> **面试一句话总结**：Framework 层把 Gateway 物化的 Trajectory 翻译成 verl/TQ 的标准字段——拼 input_ids、算 position_ids、组装 response_mask/rollout_log_probs、把 reward 打在 rm_scores 最后一位、写入 min/max_global_steps 版本标记，经 async_kv_batch_put 写入 TransferQueue；同时跳过空响应轨迹、处理多模态，是"轨迹 → 训练样本"的转换器。

---

## 5. TransferQueue（数据平面）：轨迹异步传输的管道

### 1. 现有问题：为什么需要 TransferQueue

verl 训练与 agent rollout 是**异步、可能跨机**的：agent（可能多台机器）持续产生轨迹，训练机（可能另一台机器）按 batch 消费。如果直接进程内传数据，无法跨机；如果走文件系统，延迟高且难管理。TransferQueue（TQ）是 verl 生态的数据平面：**一个 KV 存储 + 队列**，支持异步 put/get、跨进程/跨机共享（Ray 共享内存或网络存储）、以及 batch 消费。Uni-agent 用它做"轨迹 → 训练"的管道：Framework 写入（put）、verl trainer 读取（get），**生产和消费完全解耦**——agent 侧不用等训练、训练侧不用等 agent。

TQ 的**存储后端可插拔**：默认 SimpleStorage（内存/Ray 对象存储），也可以通过 `KVStorageManager` 接口接入其他 KV 后端（如 openYuanrong、Mooncake 等网络存储）——这是 TQ 的扩展点设计（官方文档："Storage backends are now pluggable"）。

### 2. 方法论：TransferQueue 是怎么接入的

**写入侧（uni-agent framework）**：`tq.async_kv_batch_put(keys, fields, tags, partition_id)`——key 形如 `{uid}_{session_index}_{index}`，fields 是拼好的 TensorDict（见第 4 点），tags 携带元数据（如 partition 隔离）。

**读取侧（verl trainer）**：`trainer_base.py` 用 `kv_batch_get` 读一批 key → 转成训练 batch → `upsample_batch_to_divisible_size` 补齐到 mini-batch 倍数 → 交给 GRPO 训练循环。

**关键字段**（verl 训练直接消费）：`prompts` / `responses`（输入输出 token ids）、`input_ids` / `attention_mask` / `position_ids`（拼接后完整序列）、`response_mask` / `loss_mask`（哪些 token 参与 loss）、`rollout_log_probs`（GRPO 需要）、`rm_scores`（reward，末位）、`num_turns`（轮次）、`min/max_global_steps`（权重版本跨度，off-policy 判定）。

**核心代码**（`framework/framework.py` 的 `_write_session_trajectories_to_tq`——写入侧真实实现）：

```python
async def _write_session_trajectories_to_tq(self, *, uid, session_index,
                                            trajectories, sample_fields,
                                            global_steps, partition_id):
    keys, fields, tags = [], [], []
    for trajectory in trajectories:
        # 空响应轨迹（如 max_trajectory_length 截断的占位）跳过：
        # 写入 Mooncake 会因 0 字节 slice 被拒，导致整个 session 失败
        if len(trajectory.response_ids) == 0:
            logger.warning("session %s: skip empty-response trajectory", uid)
            continue
        index = len(keys)
        field, tag = self._trajectory_to_tq_field_and_tag(
            trajectory=trajectory, sample_fields=sample_fields,
            session_index=session_index, global_steps=global_steps, uid=uid)
        keys.append(f"{uid}_{session_index}_{index}")     # key 连续编号
        fields.append(field); tags.append(tag)

    await tq.async_kv_batch_put(                             # 批量写入 TQ
        keys=keys, fields=_list_of_tq_fields_to_tensordict(fields),
        tags=tags, partition_id=partition_id,
    )
```

**这段代码关键在哪**：写入侧的核心是 `tq.async_kv_batch_put(keys, fields, tags, partition_id)`——key 形如 `{uid}_{session_index}_{index}`（唯一标识一条轨迹），fields 是 TensorDict（verl 直接读），tags 携带元数据、`partition_id` 做隔离；**空响应轨迹跳过**是关键设计（0 字节 slice 会被 KV 存储后端拒绝，导致整个 session 写入失败）。读取侧对称：verl 的 `trainer_base.py` 用 `kv_batch_get(keys)` 读回，`upsample_batch_to_divisible_size` 补齐后进 GRPO。

### 3. 具体数值样例

假设双机训练：node1（trainer）+ node2（rollout），batch 32。逐步演算 TQ 的数据流：

```text
第 1 步：node2 的 4 个 agent runner 并行执行 32 个任务（并发 64），
        每个任务物化出 1 条 Trajectory（平均 ~967 response tokens）。
第 2 步：framework._write_session_trajectories_to_tq 把 32 条轨迹
        转成 TQ 字段，tq.async_kv_batch_put 批量写入
        TransferQueue（partition_id 隔离），key 连续编号。
第 3 步：node1 的 verl trainer 每步 kv_batch_get 读 32 条 →
        upsample 补齐（32 已是 8 的倍数，无需补）→ GRPO 训练一步。
第 4 步：训练更新权重 → node2 的 vLLM rollout 引擎加载新权重 →
        下一轮 agent 用新模型采样 → 轨迹再入 TQ → 循环。
        （采样与训练完全重叠：node2 采第 N+1 批时 node1 在训第 N 批）
```

> **面试一句话总结**：TransferQueue 是轨迹异步传输的数据平面——Framework 侧 async_kv_batch_put 写入、verl trainer 侧 kv_batch_get 消费，生产与消费完全解耦；存储后端可插拔（默认 SimpleStorage，可接 KV 网络后端），字段（prompts/input_ids/rm_scores/num_turns 等）直接对应 verl 训练消费。

---

# 附：组件速查表

| 组件 | 代码位置（vendor/uni-agent） | 角色 | 关键类 / 方法 |
|---|---|---|---|
| Gateway（核心） | `uni_agent/gateway/` | 协议适配 + 会话路由 + **轨迹物化** | `_GatewayActor`、`GatewaySession.run_generation`、`_select_chain`、`_commit_generation_to_chain`、`finalize` |
| Agent | `uni_agent/agents/` | 官方 agent 抽象 + 内置三套 | `AgentRunner` 协议、`mini_swe_agent` / `claude_code` / `react` |
| Sandbox | `uni_agent/sandbox/` | 执行环境多后端封装 | `Sandbox` 接口、`build_sandbox` 工厂、docker/local/modal/vefaas |
| Framework | `uni_agent/framework/` | 轨迹 → TQ 训练字段 | `OpenAICompatibleAgentFramework`、`_trajectory_to_tq_field_and_tag`、`task_runner` |
| TransferQueue | verl 生态 | 异步数据平面 | `async_kv_batch_put` / `kv_batch_get`；SimpleStorage / 可插拔 KV 后端 |
