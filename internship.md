# Internship 完全指南：verl GRPO 训练全流程（TQ vs 无 TQ 双方案）

> 对应实习核心工作"升级 verl 0.6.1→0.8.0 以支持 TransferQueue（TQ）的 MoonCake 存储后端接入，实现数据 RDMA 高速传输"与"将训练入口由 RayPPOTrainer 迁移至 PPOTrainer"。一句话知识框架：
> **同一个 AgenticRL/GRPO 训练，两条数据通路：① master 分支（TQ 接入）用 TransferQueue 做"KV 数据仓库"——训练数据以变长 NestedTensor 按 key 存在 TQ 里，driver 只拿 KVBatchMeta（keys+tags），各阶段按需 `kv_batch_get/put` 读写；② verl-0.8.0（无 TQ）用 DataProto 做"整包搬运"——数据在 driver 手里整体 pad/unpad、随 RPC 发给 worker。两条路 rollout 与训练阶段的读写位置、字段布局、异步方式完全不同，性能差异（update_actor ±15s）根因就在字段布局（nested vs padded）。**
>
> 素材来源：本地 `agent-lightning`（master，TQ 接入，`AgentLightningTrainer(PPOTrainer)`）、`verl-v0.8.0`（baseline，`AgentLightningTrainer(RayPPOTrainer)`）、`transferqueue`（v0.1.8）、`agent-lightning/TQ_PERF_COMPARE_AND_FIX.md`、`agent-lightning/VERL_UPGRADE_ANALYSIS.md`。配套阅读：`Agent-Lighting.md`（三件套）、`TQ.md`（TransferQueue 组件）、`Mooncake.md`（存储后端）、`Parallel.md`（并行维度）。

---

# 一、两方案总览

## 1. 方案定义：master（TQ）vs verl-0.8.0（无 TQ）

### 1. 现有问题

同一个 AgentLightning 训练框架，两套分支用了**完全不同的 verl trainer 基类**，导致数据流的"形状"不同：

| 维度 | master（TQ 接入） | verl-0.8.0（无 TQ，baseline） |
|---|---|---|
| trainer 基类 | `verl.trainer.main_ppo_sync.PPOTrainer` | `verl.trainer.ppo.ray_trainer.RayPPOTrainer` |
| 数据载体 | **KVBatchMeta**（只有 keys/tags，数据在 TQ） | **DataProto**（数据整体带着走） |
| 数据存储 | TransferQueue（`kv_batch_get/put` 按 key 存取） | driver 内存 TensorDict/DataProto |
| 字段布局 | **NestedTensor（变长/jagged）** | **Padded 普通张量**（必要时 unpad） |
| worker 类 | `ActorRolloutRefWorker`（FSDP engine） | 相同（两边一致） |
| 数据从哪来 | agent daemon 执行后 `kv_batch_put` 写 TQ | agent daemon 返回 → driver 组装 DataProto |
| 训练入口 | `AgentLightningTrainer(PPOTrainer)` | `AgentLightningTrainer(RayPPOTrainer)` |

### 2. 方法论（为什么会有两套）

- verl 0.8.0 原生是 **DataProto 数据流**（`RayPPOTrainer`）：一次 rollout 产生一个 DataProto（含 input_ids/attention_mask/position_ids/response_mask/rm_scores 等全部字段），driver 持有它，每个计算阶段（old_log_prob/ref/advantage/update_actor）都是"整包 → RPC 给 worker → 结果 union 回来"；
- TQ 接入后（`main_ppo_sync.PPOTrainer`）：数据不再整体搬，而是**按 key 存进 TransferQueue**（partition=train/val），driver 只拿 KVBatchMeta（keys + tags），每个阶段用 `tq.kv_batch_get(keys, select_fields=...)` 按需拉字段、算完 `tq.kv_batch_put` 写回——**数据留存在 TQ（可落 Mooncake 存储后端、跨机 RDMA 传输），driver/worker 只搬运"元数据 + 需要的字段"**；
- 关键差异：TQ 存的是**变长 NestedTensor**（不 pad），而 DataProto 流是 **padded** 后再按需 unpad——这个布局差异是性能差异的根因（第 15 点）。

### 3. 具体数值样例（同一轮训练的数据形状）

- 一轮训练：dataloader 出 32 个 prompt（`train_batch_size=32`），`rollout.n=4` → 128 个 rollout 任务交给 agent 环境；
- agent 执行后有失败/超时 → 假设 120 个 triplet 成功回传；
- **baseline（agent-lightning 改造后，`upgrade/verl-0.8.0` 分支）**：120 → `pad_dataproto_to_divisor(120, divisor=lcm(dp_size, mini_batch))` 向上 pad（如到 128）→ 前向算 old_log_prob → unpad 回 120 → is_drop 过滤 10 条 → 110 → **向上补齐**：`if n_transition % per_gpu_divisor != 0: floor_pad_size = per_gpu_divisor - n_transition % per_gpu_divisor; pad_dataproto_to_divisor(batch, per_gpu_divisor)`（pad 到 `per_gpu_divisor = world_size × (mini_batch×n // world_size)` 的倍数，被 pad 样本的 `response_mask/token_level_rewards` 置 0，不参与 loss）→ `_update_actor`（override 后 mini_batch_size=32 不乘 n）；
- **master（TQ）**：120 个 triplet 直接以**变长 NestedTensor** 写 TQ（`daemon.get_train_data_batch` → `kv_batch_put`），不再 pad/unpad；`_balance_batch` 用 **upsample（合成 no-op 样本补齐到 batch_multiple=lcm(dp, mini_batch×n)）** 替代 pad 补零，`is_padding` 标记不进 PPO/entropy/KL，is_drop 在 tags 里标记、`_compute_*` 时过滤——**数据形状两方案差异巨大，但都是"向上补齐"而非丢弃**；
- ⚠️ 注意：`VERL_UPGRADE_ANALYSIS.md` 里的 "110 → `floor(110/32)*32` = 96 丢弃 14 条" 是 **原生 verl 0.8.0（agent-lightning 改造前）** 的 floor 行为，改造后两个分支都改为向上补齐（master upsample / baseline pad），**不存在"丢弃数据"**。

> 面试一句话总结：**master（TQ）与 verl-0.8.0（无 TQ）的差异 = 数据载体与字段布局：TQ 版把训练数据按 key 存进 TransferQueue（变长 NestedTensor），driver 只拿 KVBatchMeta 按需 kv_batch_get/put，省掉 pad/unpad 整包搬运；无 TQ 版用 DataProto 整包带着走、padded 布局——同一套 GRPO 逻辑，两条完全不同的数据通路。**

---

## 2. 数据流全景图（rollout → 训练 → 权重更新）

### 1. 现有问题

先给一张全景图，把"谁写、谁读、走什么协议"钉死，后面每一点都是它的展开。

### 2. 方法论（两方案全景）

**master（TQ 版）**：

```text
┌────────────────────────── driver 进程（AgentLightningTrainer）──────────────────────────┐
│ fit() → for batch in train_dataloader:                                                │
│   _train_step(batch_dict):                                                            │
│     ① gen：checkpoint_manager.wake_up_replicas()                                      │
│           → agent_mode_daemon.set_up_data_and_server(gen_batch, vllm 地址)            │
│                ├─ 把 32 prompt × n=4 任务入队（v1: store.enqueue_many_rollouts）        │
│                ├─ 给每个 rollout 写 TQ 运行屏障：kv_put(key, tag={global_steps,status:running})│
│                └─ agent 环境（远程）执行：agent → HTTP → LLMProxy → HTTP → vLLM 引擎    │
│           → replay_buffer.sample(partition_id="train", global_steps)  ← 阻塞等全部完成 │
│                └─ ReplayBuffer 后台线程每 1s kv_list() 轮询，等 status=running→success │
│           → agent_mode_daemon.clear_data_and_server()                                 │
│           → daemon.get_train_data_batch() → 120 triplet → NestedTensor → kv_batch_put │
│     ② reward：_compute_reward_colocate（规则 reward，写 TQ）                            │
│     ③ 过滤 is_drop（tags 标记）→ _balance_batch（upsample 到 128）                      │
│     ④ _compute_old_log_prob：kv_batch_get → worker compute_log_prob → 写回 TQ          │
│     ⑤ _compute_ref_log_prob：同上（ref 模型）                                           │
│     ⑥ _compute_advantage：kv_batch_get 拉字段 → GRPO 优势 → kv_batch_put 写回           │
│     ⑦ _update_actor：KVBatchMeta → worker 侧 tqbridge 从 TQ 拉数据 → FSDP 训练         │
│          → ppo_loss（nested→padded）→ backward → 梯度聚合 → 输出 metrics 写回 TQ       │
│     ⑧ update_weights：checkpoint_manager.update_weights（权重同步给 rollout 引擎）      │
└──────────────────────────────────────────────────────────────────────────────────────┘
     数据面：TransferQueue（partition=train，每 key 一个样本的 NestedTensor 字段）
             写入：daemon（rollout 结果）、_compute_*（old/ref/adv）
             读取：tqbridge（worker 训练）、_compute_*（driver 指标）
```

**verl-0.8.0（无 TQ 版）**：

```text
┌────────────────────────── driver 进程（AgentLightningTrainer(RayPPOTrainer)）─────────┐
│ fit() → for batch_dict in train_dataloader:                                          │
│   batch = DataProto.from_single_dict(batch_dict)      ← 32 prompt 的完整 DataProto    │
│   ① gen：gen_batch.repeat(n=4) → async_rollout_manager.generate_sequences()          │
│        （agent daemon 替代 vLLM 直出：任务 → agent 环境 → vLLM）→ 返回 DataProto        │
│        → batch = batch.union(gen_batch_output)        ← 生成结果 union 进 driver 的   │
│          batch（input_ids/attention_mask/position_ids/response_mask...全部字段）       │
│   ② reward：_compute_reward_colocate → batch.union(rm_scores)                        │
│   ③ pad 到 world_size 倍数 → old_log_prob：batch → to_tensordict → left_right_2_no_padding│
│        → worker compute_log_prob → no_padding_2_padding → union 回 batch               │
│   ④ ref / values：同样整包 union                                                   │
│   ⑤ advantage：driver 轻量计算（GRPO）→ union                                       │
│   ⑥ _update_actor：batch → to_tensordict → left_right_2_no_padding →                 │
│        worker.update_actor → 返回 metrics                                            │
│   ⑦ update_weights：checkpoint_manager.update_weights                               │
└──────────────────────────────────────────────────────────────────────────────────────┘
     数据面：driver 内存 DataProto 整包携带；worker 间走 Ray RPC（TensorDict 序列化）
```

### 3. 具体数值样例（读写点的对照）

| 阶段 | master（TQ）读写 | verl-0.8.0（无 TQ）读写 |
|---|---|---|
| rollout 结果 | daemon `kv_batch_put` 写 TQ（NestedTensor） | `generate_sequences` 返回 DataProto，`batch.union` |
| old_log_prob | `kv_batch_get` 读、worker 算、`kv_batch_put` 写回 | driver `to_tensordict+left_right_2_no_padding`，RPC 带数据，`union` 回来 |
| advantage | `kv_batch_get` 拉 7 个字段 → 算 → `kv_batch_put` 写回 | 数据都在 driver 的 batch 里，直接算 |
| update_actor | worker `tqbridge` 从 TQ 拉（`_meta_to_realdata`） | worker 收 RPC 里的 TensorDict |
| metrics | worker 输出写 TQ → driver `kv_batch_get` | worker 返回 TensorDict → `tu.get(output, "metrics")` |

> 面试一句话总结：**TQ 版是"数据仓库 + 按需读写"（写 TQ 一次、各阶段 kv_batch_get 按需拉字段、算完写回，数据不离开 TQ）；无 TQ 版是"driver 整包携带 + RPC 搬运"（union 拼字段、pad/unpad、每次 RPC 全量传）——同样一轮 GRPO，master 的 rollout/训练读写都发生在 TQ 上，baseline 都发生在 driver 内存与 RPC 上。**

---

# 二、rollout 阶段（agent 执行与轨迹回传）

## 3. 入口：_train_step 的 gen 阶段（daemon vs async_rollout_manager）

### 1. 现有问题

verl 原生 rollout 是"vLLM 引擎直出 token"；AgenticRL 是"agent 环境执行多轮工具调用"——rollout 不再是纯生成，而是"给 agent 派任务、等 agent 跑完、把轨迹收回来"。入口在哪？

### 2. 方法论（master 的 gen 阶段，`agentlightning/verl/trainer.py`）

master 的 `_train_step` 用 **AgentModeDaemon** 替代 verl 原生的 async_rollout_manager，gen 阶段 = "唤醒副本 → 派任务 → 等 TQ → 收数据"：

```python
# agentlightning/verl/trainer.py —— master 的 _train_step 第 ① 阶段
with _timer("gen", timing_raw):
    self.checkpoint_manager.wake_up_replicas()          # 唤醒 rollout vLLM 副本
    # ===== 新逻辑：TQ 双写 =====
    self.agent_mode_daemon.set_up_data_and_server(
        gen_batch.non_tensor_batch, self.llm_server_manager.get_addresses(),
        global_steps=self.global_steps,
    )                                                     # ① 派任务（内部写 TQ 运行屏障）
    batch = self.replay_buffer.sample(
        partition_id="train", global_steps=self.global_steps
    )                                                     # ② 等该 step 全部 rollout 完成（TQ 轮询）
    self.agent_mode_daemon.clear_data_and_server()        # ③ 清状态
    self.checkpoint_manager.sleep_replicas()              # ④ 副本休眠（省显存）
```

- `checkpoint_manager.wake_up_replicas/sleep_replicas`：rollout vLLM 引擎只在 gen 阶段唤醒（省显存），训练阶段休眠——**训练与 rollout 引擎复用同一批 GPU**（colocate 形态的优化）；
- `llm_server_manager.get_addresses()`：拿到 vLLM OpenAI 兼容端点地址，作为资源写进 store/代理；
- `replay_buffer.sample(partition_id="train", global_steps)`：**阻塞**直到该 global_steps 的所有 rollout 在 TQ 里都标记完成（第 7 点展开）。

**baseline（verl-0.8.0）对应**（`ray_trainer.py` fit）：

```python
gen_batch = self._get_gen_batch(batch)
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)  # 32×4=128
combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
self.checkpoint_manager.sleep_replicas()
...
batch = batch.union(gen_batch_output)      # 生成结果直接 union 进 driver 的 batch
```

### 3. 具体数值样例

- 32 prompt × n=4 = 128 个 rollout 任务；master 用 daemon 批量入队（v1 模式 `store.enqueue_many_rollouts` 一次入队 128 个），baseline 用 `repeat(interleave=True)` 复制成 128 条再 `generate_sequences`；
- master 的 `replay_buffer.sample` 返回的**不是数据**，是 `KVBatchMeta(keys=[128 个 key], tags=[...])`——数据还在 TQ 里等后续阶段按需拉；
- baseline 的 `generate_sequences` 返回**完整 DataProto**（128 条 × input_ids/attention_mask/position_ids/...），立即 `union` 进 batch——数据此刻就全在 driver 内存。

> 面试一句话总结：**gen 阶段两方案的分水岭：master 用 AgentModeDaemon 派任务 + ReplayBuffer 从 TQ 等结果，返回的是"只有 keys/tags 的 KVBatchMeta"；baseline 用 async_rollout_manager.generate_sequences，返回的是"包含全部字段的 DataProto"直接 union——一个"数据在 TQ、拿元数据"，一个"数据在手、整包带着"。**

---

## 4. HTTP 推送在哪里：任务派发 + Agent→LLMProxy→vLLM

### 1. 现有问题

rollout 阶段的 HTTP 到底发生在哪几个点？用户问"哪里去使用HTTP推送"——答案是**三个点**：① driver→store 的任务入队（v0 模式 HTTP / v1 模式 store API）；② agent 环境→LLMProxy 的推理调用（OpenAI 兼容 HTTP）；③ LLMProxy→vLLM 引擎（HTTP 转发）。

### 2. 方法论（`agentlightning/verl/daemon.py` 的 `_async_set_up`）

**① 任务派发（driver → agent 环境）**：`_async_set_up` 把"32 prompt × n 次 rollout"变成任务入队：

```python
# agentlightning/verl/daemon.py —— _async_set_up（v1 模式：批量入队到 LightningStore）
for i in range(num_samples):
    data_id = str(uuid.uuid4())
    original_sample = {key: data[key][i] for key in keys}
    for _ in range(rollouts_per_sample):          # rollout.n 次
        enqueue_rollout_requests.append(EnqueueRolloutRequest(
            input=_to_native(original_sample), mode="train",
            resources_id=resources_id,             # 资源 = LLMProxy/vLLM 地址
            config=RolloutConfig(unresponsive_seconds=..., timeout_seconds=...),
            metadata=task_metadata,                # 含 data_id/global_steps/max_prompt_length
        ))
if self.mode == "v1":
    rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)  # 批量入队
# ===== TQ 运行屏障：每个 rollout 一个 running 标记 =====
if global_steps is not None and is_train:
    for rollout_id in self._task_id_to_original_sample:
        tq.kv_put(key=rollout_id, partition_id="train",
                  tag={"global_steps": global_steps, "status": "running"})
```

**② Agent → LLMProxy → vLLM（推理 HTTP 链）**：
- `llm_server_manager.get_addresses()` 拿到 vLLM 引擎的 OpenAI 兼容地址（`verl/workers/rollout/llm_server.py` 的 LLMServerManager 管理多个 replica，`get_openai_http_addresses`）；
- 这些地址注册进 store 的资源（`resources = {"main_llm": LLM(endpoint=..., model=..., sampling_parameters=...)}`）；
- 远端 agent 从 store 取资源 → **agent 的每一次 LLM 调用走 HTTP → LLMProxy → HTTP → vLLM 引擎**（与 `Agent-Lighting.md` 第 17 点的通信矩阵一致）；LLMProxy 在此链路中负责**instrumentation**（把每次 LLM 调用的 prompt/response 记成 span 回传 store）；
- **v0 模式**：`self.server.queue_task(...)` 走老 HTTP server（每个任务一次 HTTP POST）；**v1 模式**：`store.enqueue_many_rollouts` 批量入队（走 LightningStore 接口）。

### 3. 具体数值样例

- 128 个 rollout 任务：v1 模式**一次** `enqueue_many_rollouts`（128 条请求批量），v0 模式 128 次 HTTP `queue_task`；
- 每个 rollout 内部 agent 平均 5 轮工具调用 → 每轮 1 次 LLM 推理 → 128×5=640 次 HTTP 推理请求（并发到 vLLM 引擎，`max_num_seqs` 限制并发）；
- TQ 屏障：128 个 `kv_put`（tag=status:running）在任务派发瞬间写入——训练侧 ReplayBuffer 靠这 128 个 tag 知道"该等谁"。

> 面试一句话总结：**rollout 的 HTTP 有三个点：driver→store 的任务入队（v0 逐任务 HTTP / v1 批量 store API）、agent→LLMProxy 的推理调用（OpenAI 兼容）、LLMProxy→vLLM 引擎（转发）；任务派发的同时给每个 rollout 写 TQ 运行屏障（kv_put status:running），训练侧据此等待完成——HTTP 负责"派活与推理"，TQ 负责"记账与取数"。**

---

## 5. 轨迹如何变成 triplet：span → store → adapter → RolloutLegacy

### 1. 现有问题

agent 执行产生的不是"token 序列"，而是**事件流（span）**——prompt、每次 LLM 调用、工具调用、奖励都要变成 (prompt, response, reward) 三元组才能训练。谁来做这个转换？

### 2. 方法论（`daemon._validate_data_v1`）

master（v1 模式）的 daemon 在等 rollout 完成时，对每个完成的任务做三件事：查 span → adapter 转 triplet → 组装 RolloutLegacy：

```python
# agentlightning/verl/daemon.py —— _validate_data_v1（v1 模式）
spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")  # ① 从 store 查该 rollout 的 span
triplets = self.adapter.adapt(spans)          # ② adapter（TracerTraceToTriplet）把 span 转 (prompt,response,reward)
final_reward = ...                            # ③ 从最后一个 triplet 提取最终奖励（向前搜索非 None）
task = Task(rollout_id=..., input=..., mode=..., resources_id=...)
result_rollout = RolloutLegacy(rollout_id=..., task=task, final_reward=final_reward, triplets=triplets)
```

- **span 从哪来**：agent 执行时，LLMProxy/Tracer 把每次 LLM 调用、工具调用、奖励记成 OpenTelemetry span 流式写回 store（`add_otel_span`）——**这就是"轨迹"的存储形态**；
- **adapter 是什么**：`TracerTraceToTriplet`（agentlightning 的默认 Adapter）把 span 序列转成 `(prompt, response, reward)` 三元组列表——一个 rollout 可能含多个 triplet（多轮工具调用 = 多个 transition）；
- **daemon 等待循环**（`_async_run_until_finished`）：每 5s `store.wait_for_rollouts(rollout_ids, timeout=0)` 拉一批完成的 rollout，逐个 `_validate_data_v1` 存入 `_completed_rollouts_v0`，直到 `len == _total_tasks_queued`；
- v0 模式：`server.retrieve_completed_rollouts()`（老 HTTP server 拉取）。

### 3. 具体数值样例

- 一个 rollout（SWE-bench 任务）：agent 5 轮 → 5 个 transition → adapter 出 5 个 triplet，每个 triplet = (prompt_ids, response_ids, reward=None 或最终奖励)；
- 128 个 rollout 全部完成后，`_completed_rollouts_v0` 有 128 个 RolloutLegacy，其中部分 triplet 为空（agent 无输出）→ 后续在 `get_train_data_batch` 里跳过；
- 奖励：`final_reward` 从最后 triplet 向前搜索；None 用 `_fillna_reward` 填默认值（`reward_fillna_value`）。

> 面试一句话总结：**轨迹转训练数据：agent 执行时的 LLM/工具调用被 Tracer/LLMProxy 记成 span 存 store → daemon 用 adapter（TracerTraceToTriplet）把 span 转成 (prompt,response,reward) 三元组 → 组装成 RolloutLegacy 收集到 _completed_rollouts_v0——多轮 agent 的一个 rollout 可以拆出多个 triplet（transition），这是"token 级轨迹"在 agent 场景的形态。**

---

## 6. 写入 TQ：get_train_data_batch（NestedTensor 构建 + kv_batch_put）

### 1. 现有问题

用户问"哪里去写入TQ"——答案是 **`daemon.get_train_data_batch`**：把收集到的 RolloutLegacy（triplet 列表）变成 TQ 里的变长 NestedTensor 字段，`kv_batch_put` 一次性写入。这是 TQ 方案 rollout 数据进训练管线的唯一入口。

### 2. 方法论（`daemon.get_train_data_batch`，核心代码）

**① 逐样本构建 unpadded 变长字段**（transition 模式：每个 triplet 一个样本）：

```python
# agentlightning/verl/daemon.py —— get_train_data_batch（核心）
for i in range(n_transition):
    prompt_ids = raw_prompt_ids_list[i]        # 未 pad 的变长 prompt
    response_ids = raw_response_ids_list[i]    # 未 pad 的变长 response
    seq_ids = prompt_ids + response_ids
    # token_level_scores：reward 放在 response 最后一个 token 位置
    token_level_scores_i = [0.0] * response_len
    token_level_scores_i[-1] = reward_list[i]
    field = {
        "prompts": torch.tensor(prompt_ids, ...),           # 变长
        "responses": torch.tensor(response_ids, ...),       # 变长
        "input_ids": torch.tensor(seq_ids, ...),            # 变长
        "attention_mask": torch.ones(seq_len, ...),
        "position_ids": torch.arange(seq_len, ...),         # unpadded 下等价 cumsum-1
        "token_level_scores": token_level_scores_tensor,
        "rm_scores": token_level_scores_tensor,
        "response_mask": torch.ones(response_len, ...),     # transition 模式全 1
        "loss_mask": torch.ones(response_len, ...),
        "uid": data_id_list[i], "num_turns": 1,
    }
    fields_list.append(field)
# list_of_dict_to_tensordict：变长 tensor → NestedTensor(jagged)，非 tensor → NonTensorStack
fields = list_of_dict_to_tensordict(fields_list)
# 生成唯一 key：{data_id}_{rollout_id}_{turn_index}
keys = [f"{data_id}_{rollout_id}_{turn}" for ...]
tags = [{"seq_len": plen+rlen, "is_drop": is_drop} for ...]
# ===== 写入 TQ =====
tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)
```

**② 关键设计点**：
- **不 pad**：直接保存变长序列，`list_of_dict_to_tensordict` 把它们包成 **NestedTensor（jagged）**——这是 TQ 能"零 pad 传输"的根本（对应 `TQ_PERF_COMPARE` 里"master 数据几乎全部 nested"）；
- **每样本一个 key**：`{data_id}_{rollout_id}_{turn_index}`，TQ 里每个 key 是一份完整训练样本（含 input_ids/attention_mask/position_ids/token_level_scores/rm_scores/response_mask/loss_mask/uid/num_turns）；
- **tags 携带元数据**：`seq_len`（长度，供 `_compute_metrics` 的 offsets 计算）、`is_drop`（prompt 过长标记，第 5 点过滤的依据）；
- **position_ids 免计算**：unpadded 序列 position_ids 就是 `arange(seq_len)`（M-RoPE 模型除外）；
- **token_level_scores**：reward 只放在 response 最后一个 token 位置（GRPO 的 token 级奖励语义）。

### 3. 具体数值样例

- 120 个 triplet → 120 个 key → `kv_batch_put` 一次写入 TQ 的 train partition；
- 每个 key 的字段：`input_ids`（变长，如 512 token）、`position_ids`（512）、`token_level_scores`（512，最后一位 = reward）、`rm_scores` 同、`response_mask`（256，response 长度）、`loss_mask`（256）、`uid`、`num_turns`；
- 数据量：120 × 平均 700 token × 8 字节 ≈ 0.67 MB unpadded（对比 baseline 的 padded 128 × 1024 × 8 ≈ 1 MB——**TQ 省掉 pad 的 ~30% 存储**，且可经 Mooncake 落盘/RDMA 传输）。

> 面试一句话总结：**写入 TQ 的唯一入口是 daemon.get_train_data_batch：把收集到的 triplet 逐样本构造成 unpadded 变长字段（input_ids/attention_mask/position_ids/token_level_scores/rm_scores/response_mask/loss_mask...），list_of_dict_to_tensordict 包成 NestedTensor，每个样本一个 key（{data_id}_{rollout_id}_{turn}）连同 tags（seq_len/is_drop）一次 kv_batch_put 进 train partition——rollout 数据从此以"变长 key-value"形态躺在 TQ 里，供训练各阶段按需拉取。**

---

## 7. 异步与开销：ReplayBuffer 轮询、daemon 等待、开销是否被 rollout 吃掉

### 1. 现有问题

用户问"是否异步，开销是否被rollout吃掉"——要分两个层面：**① rollout 与训练的并行性**（pipeline 重叠）；**② TQ 读写本身的开销**（有没有暴露成训练时间的增长）。

### 2. 方法论

**① 异步机制（三处）**：

```python
# verl/trainer/main_ppo_sync.py —— ReplayBuffer：后台线程轮询 TQ 元数据
class ReplayBuffer:
    def __init__(self, poll_interval: float = 1.0):
        self.poll_thread = threading.Thread(target=self._poll_from_transfer_queue, daemon=True)
        self.poll_thread.start()          # 后台线程：每 1s kv_list() 一次，把 tags 抄进本地字典

    def _poll_from_transfer_queue(self):
        while not self._stop_event.is_set():
            data = tq.kv_list()           # 只拉元数据（keys+tags），不拉数据本体
            if data is not None:
                for partition_id, items in data.items():
                    self.add(partition_id, items)   # 更新本地 partitions 字典
            self._stop_event.wait(self.poll_interval)

    def sample(self, partition_id, global_steps, ...) -> KVBatchMeta:
        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                # 等该 global_steps 的所有 key 从 running → success
                for key, tag in partition.items():
                    if tag["global_steps"] == global_steps:
                        if tag["status"] == "running": should_wait = True; break
                        elif tag["status"] == "success": keys.append(key)
                if not should_wait:
                    return KVBatchMeta(partition_id=..., keys=keys, tags=tags)
```

- **ReplayBuffer 后台轮询线程**：每 1s `kv_list()` 只拉元数据（开销极小），`sample()` 阻塞等待"该 step 全部 success"——**训练侧不需要知道 agent 怎么执行，只看 TQ 里的状态 tag**；
- **daemon 异步等待**：`_async_run_until_finished` 每 5s `store.wait_for_rollouts` 拉完成的任务，异步收集（不阻塞 agent 执行）；
- **agent 执行本身异步并行**：128 个 rollout 在远端 agent 环境并发执行（沙箱并行 + vLLM 并发），driver 的 `replay_buffer.sample` 只是"等结果"。

**② 开销是否被 rollout 吃掉（bench 实证，`TQ_PERF_COMPARE_AND_FIX.md`）**：

| 阶段 | master（TQ） | baseline（无 TQ） | 结论 |
|---|---|---|---|
| 生成 rollout | 基本相当 | 基本相当 | 相同（agent 执行占大头） |
| ref + old_log_prob | **快约 2s** | — | TQ 变长数据省 pad/unpad，infer 反而快 |
| **update_actor** | **慢约 15s** | — | **未被 rollout 掩盖**（串行训练阶段） |
| update_weights | 慢约 5s | — | CPU 竞争（TQ 栈 vs FSDP offload 传输） |

- **TQ 读取本身不是瓶颈**：`update_actor` 内 worker 侧 TQ 读取仅约 0.5s（已排除）；
- **开销没有被 rollout 吃掉的部分**：训练侧 `update_actor` 是**串行**的（agent 已全部完成、GPU 空闲等训练），TQ 数据读出来后 `ppo_loss` 的 nested→padded 转换（第 11/15 点）直接暴露成 +15s——**rollout 时间掩盖了 TQ 的写入（daemon 侧与 agent 执行并行），掩盖不了训练侧的转换开销**；
- **被 rollout 吃掉的部分**：daemon 的 `get_train_data_batch`（triplet→NestedTensor 构建 + kv_batch_put）发生在 agent 执行期间/完成后，其耗时被 agent 执行的墙钟掩盖（不额外增加 step 时间）。

### 3. 具体数值样例

- 一轮 step 总时长 ~7 分钟（agent 执行占 ~6 分钟）：TQ 写入（kv_batch_put 120 key ≈ 数十 ms）与 triplet 构建（~1s）都发生在 agent 执行期间 → **被 rollout 墙钟完全掩盖**；
- 训练阶段（agent 已完成）：`replay_buffer.sample` 返回后，old_log_prob/ref/advantage/update_actor 串行执行——其中 update_actor 的 nested→padded 转换 4 个 micro-batch 累计 ~15s **全额暴露**；
- 结论话术：**"TQ 的写入开销被 rollout 吃掉（并行发生），但训练侧 nested→padded 的转换开销是串行的、没被吃掉，这是 update_actor +15s 的来源；TQ 读取本身 0.5s 可忽略。"**

> 面试一句话总结：**异步有三层：ReplayBuffer 后台线程每 1s 轮询 TQ 元数据、daemon 每 5s 异步拉完成任务、agent 远端并发执行——训练只等"状态 tag"不看执行过程；开销上，TQ 写入/构建被 rollout 墙钟掩盖，但训练侧 ppo_loss 的 nested→padded 转换（每 micro-batch 重复）是串行的、没被掩盖，实测 update_actor 慢 15s 的根因就在这里。**

---

# 三、训练阶段（read → compute → write）

## 8. 从 TQ 读数据：KVBatchMeta + tqbridge（kv_batch_get）

### 1. 现有问题

用户问"update_actor 的时候刚开始要从哪里读取数据"——答案：**worker 进程里，tqbridge 装饰器把 KVBatchMeta 展开成真实数据（kv_batch_get）再喂给训练函数**。先讲清楚读取机制，再讲 update_actor 全流程。

### 2. 方法论

**① KVBatchMeta：只有元数据**（`verl/utils/tensordict_utils.py` 或 transferqueue 定义）：

```text
KVBatchMeta = { keys: [key1, key2, ...],   # 样本 key（对应 TQ 里的 key）
                tags: [{seq_len, is_drop, global_steps, status}, ...],
                partition_id: "train",
                fields: {input_ids, position_ids, ...},   # 需要的字段集合
                extra_info: {...} }         # 训练参数（mini_batch_size、epochs、temperature...）
```

**② tqbridge：把元数据变成数据**（`verl/utils/transferqueue_utils.py`）：

```python
# verl/utils/transferqueue_utils.py —— tqbridge 装饰器（worker 侧）
def tqbridge(dispatch_mode=None):
    def decorator(func):
        def inner(*args, **kwargs):
            batch_meta = _find_meta(*args, **kwargs)      # 从参数里找到 KVBatchMeta
            if batch_meta is None:
                return func(*args, **kwargs)              # 无 TQ 则原样调用（兼容）
            if not TQ_INITIALIZED:
                tq.init(); TQ_INITIALIZED = True          # worker 进程首次调用时初始化 TQ
            is_kv_batch_meta = isinstance(batch_meta, KVBatchMeta)
            if is_kv_batch_meta:
                tags = batch_meta.tags
                batch_meta = kv_batch_meta2batch_meta(batch_meta)   # KVBatchMeta → BatchMeta
            # ===== 关键：把 meta 展开成真实数据（从 TQ 拉取）=====
            args = [_meta_to_realdata(arg) if isinstance(arg, BatchMeta|KVBatchMeta) else arg for arg in args]
            kwargs = {k: _meta_to_realdata(v) if ... else v for k, v in kwargs.items()}
            output = func(*args, **kwargs)                # 调真实函数（compute_log_prob / train_mini_batch）
            # ===== 关键：输出写回 TQ（kv_batch_put）=====
            if put_data and need_collect:
                updated_meta = _update_meta_with_output(output, batch_meta, func.__name__)
                if is_kv_batch_meta:
                    updated_meta = batch_meta2kv_batch_meta(updated_meta)
                    updated_meta.tags = tags
                return updated_meta                       # 返回更新后的 KVBatchMeta
            return _postprocess_common(output, put_data, need_collect)
        return inner
    return decorator
```

- `_meta_to_realdata`：对 BatchMeta 调 **`tq.kv_batch_get(keys, partition_id, select_fields)`** 把 NestedTensor 字段物化成 TensorDict（变长）；
- `_update_meta_with_output`：把函数输出（如 log_probs/entropy）**写回 TQ**（kv_batch_put），只返回更新后的 meta——**worker 与 driver 之间传的是"元数据 + 变化量"，不是全量数据**；
- `tq.init()`：worker 进程首次调用时初始化 TQ 客户端（连接 controller/存储，见 `TQ.md`）。

### 3. 具体数值样例

- worker 侧 `compute_log_prob(batch_meta)`：tqbridge 收到 KVBatchMeta（128 keys）→ `kv_batch_get(select_fields=["input_ids","position_ids","attention_mask"])` 从 TQ 拉 128 份变长字段（≈0.7 MB）→ 前向算 log_probs → `_update_meta_with_output` 把 `log_probs/entropy` 写回 TQ（kv_batch_put）→ 返回 KVBatchMeta（带新字段标记）；
- 对比 baseline：worker 收 RPC 里的**完整 TensorDict**（128 × padded 1024 × 全部字段 ≈ 数 MB 序列化传输）——**TQ 版每阶段只传需要的字段 + 变长数据，RPC 版全量传 + padded**。

> 面试一句话总结：**从 TQ 读数据 = tqbridge 装饰器在 worker 侧把 KVBatchMeta（keys+tags）展开：kv_batch_get 按 select_fields 拉变长 NestedTensor 字段喂给训练函数，输出再 kv_batch_put 写回 TQ、只返回更新后的 meta——driver 与 worker 之间永远传"元数据+字段子集"，不传全量数据，这是 TQ 方案省传输的核心机制。**

---

## 9. update_actor 完整流程（master TQ 版，从读数据到收集指标）

### 1. 现有问题

用户的核心问题："update_actor 的时候刚开始要从哪里读取数据，然后经过哪些过程，比如前向传播反向传播，最后如何收集"。这里给出**端到端完整链路**（driver → worker → TQ → FSDP → 收集）。

### 2. 方法论（完整链路 8 步）

```text
① driver：_update_actor(batch: KVBatchMeta, metrics)
   └─ ppo_mini_batch_size = 32 × rollout.n(4) = 128
   └─ batch.extra_info 注入 {global_batch_size, mini_batch_size, epochs,
                             seed, dataloader_kwargs, temperature, calculate_entropy}
   └─ actor_rollout_wg.update_actor(batch)          ← Ray RPC，传 KVBatchMeta（元数据）
② worker：ActorRolloutRefWorker.update_actor(data)
   └─ self.actor.train_mini_batch(data)             ← FSDPEngine（tqbridge 装饰）
③ tqbridge：_meta_to_realdata → tq.kv_batch_get(keys, select_fields=全部训练字段)
   └─ 从 TQ 拉 input_ids/position_ids/response_mask/loss_mask/old_log_probs/
      ref_log_prob/advantages/returns/rm_scores/...（NestedTensor，变长）
④ train_mini_batch(data: TensorDict)
   ├─ maybe_fix_3d_position_ids(data)
   ├─ batch_size_per_dp = data.shape[0]（如 8 条/GPU）
   ├─ mini_batch_size_per_gpu = 128 ÷ dp_size(16) = 8
   ├─ make_iterator(data, mini_batch_size=8, epochs, shuffle)   # 迭代器按 mini-batch 切
   └─ for mini_batch_td in dataloader:
        ├─ tu.assign_non_tensor(mini_batch_td, global_token_num=..., 
        │                       update_lr_scheduler=..., compute_loss=True)
        └─ actor_output = self.train_batch(mini_batch_td)       # ← 前向+反向（第 10 点）
⑤ train_batch → engine.train_batch(data, loss_function=ppo_loss)
   └─ forward_backward_batch（prepare_micro_batches → forward_step → loss.backward）
   └─ optimizer_step（clip_grad_norm → optimizer.step）         # 反向与更新在 engine 内
⑥ tqbridge：_update_meta_with_output(output, batch_meta, "train_mini_batch")
   └─ output（loss/grad_norm/mfu 等 metrics）→ kv_batch_put 写回 TQ（或经 dispatch 收集）
   └─ 返回更新后的 KVBatchMeta
⑦ driver：actor_output = rename_dict(metrics, "actor/")
   └─ actor_metrics = reduce_metrics(output)
   └─ metrics.update(actor_metrics)                 # actor/loss、actor/grad_norm、perf/mfu/actor
⑧ driver：_compute_metrics（kv_batch_get 拉 prompts/responses/advantages/returns/rm_scores...）
   └─ compute_data_metrics / compute_timing_metrics / compute_throughout_metrics
```

**代码锚点**（全部真实）：

```python
# verl/trainer/main_ppo_sync.py —— driver 侧
def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
    ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n  # 32×4=128
    batch.extra_info.update({
        "calculate_entropy": calculate_entropy, "distillation_use_topk": ...,
        "global_batch_size": ppo_mini_batch_size, "mini_batch_size": ppo_mini_batch_size,
        "epochs": ..., "seed": ..., "dataloader_kwargs": {"shuffle": ...},
        "temperature": ...,})
    output: TensorDict = self.actor_rollout_wg.update_actor(batch)   # KVBatchMeta → worker
    output = rename_dict(output["metrics"], "actor/")
    actor_metrics = reduce_metrics(output)
    metrics.update(actor_metrics)
    return batch
```

```python
# verl/workers/engine_workers.py —— worker 侧（update_actor → train_mini_batch）
def update_actor(self, data: TensorDict) -> TensorDict:
    output = self.actor.train_mini_batch(data=data)     # actor = FSDPEngine
    return output.cpu() if output is not None else None
```

### 3. 具体数值样例（一条数据从 TQ 到梯度更新）

- 128 个 key（120 真实 + 8 upsample）→ tqbridge `kv_batch_get` 拉 8 个字段（input_ids/position_ids/response_mask/loss_mask/old_log_probs/ref_log_prob/advantages/returns）≈ 2 MB 变长数据 → 进 worker；
- `train_mini_batch`：`batch_size_per_dp=128/16=8`，`mini_batch_size_per_gpu=128/16=8` → **1 个 mini-batch（8 条）**，`ppo_epochs=1`（agent-lightning 配置）；
- `forward_backward_batch`：`prepare_micro_batches` 把 8 条按 `micro_batch_size_per_gpu=4` 切成 **2 个 micro-batch**，每个 micro-batch：前向（remove-padding 路径算 ppo_loss）→ `loss.backward()` → 梯度 FSDP all-reduce 聚合；
- `optimizer_step`：`clip_grad_norm_(1.0)` → `optimizer.step()` → 权重更新；
- 收集：worker 把 `loss/grad_norm/mfu` 写 TQ/返回 → driver `reduce_metrics` 得 `actor/loss=0.021, actor/grad_norm=0.87, perf/mfu/actor=0.31` 等。

> 面试一句话总结：**update_actor 完整链路：driver 注入 mini_batch 参数并发 KVBatchMeta 给 worker → worker 的 FSDPEngine.train_mini_batch 经 tqbridge 从 TQ kv_batch_get 拉变长字段 → 按 mini_batch/micro_batch 切分 → 每 micro-batch 前向算 ppo_loss + loss.backward()（FSDP 梯度聚合）→ optimizer_step（clip+step）→ 输出 metrics 经 kv_batch_put 写回 TQ → driver reduce_metrics 收集 actor/loss、grad_norm、mfu——数据"从 TQ 来、回 TQ 去"，driver 只做参数注入与指标聚合。**

---

## 10. 前向传播与反向传播细节（FSDPEngine 内部）

### 1. 现有问题

前向反向具体在 FSDPEngine 里怎么执行？micro-batch 怎么切？loss 怎么归一化？

### 2. 方法论（`verl/workers/engine/fsdp/transformer_impl.py`）

**forward_backward_batch（核心）**：

```python
# verl/workers/engine/fsdp/transformer_impl.py —— FSDPEngine.forward_backward_batch
def forward_backward_batch(self, data, loss_function, forward_only=False):
    tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)
    # ① loss 归一化分母：全 DP 的有效 token 数（loss_mask 求和 + all_reduce）
    batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
    torch.distributed.all_reduce(batch_num_tokens, op=SUM, group=self.get_data_parallel_group())
    tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
    tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())
    # ② micro-batch 切分（固定尺寸，per-GPU 整除断言）
    micro_batches, indices = prepare_micro_batches(
        data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True)
    output_lst = []
    ctx = torch.no_grad() if forward_only else nullcontext()   # 训练时保留梯度
    for micro_batch in micro_batches:
        with ctx:
            loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
            if not forward_only:
                loss.backward()              # ② 反向传播（scaler.scale(loss).backward() 若用 AMP）
        output_lst.append(meta_info)
    return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
```

**optimizer_step（更新）**：

```python
def optimizer_step(self):
    ...
    if isinstance(self.module, FSDP):
        grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)   # 梯度裁剪
    ...
    self.optimizer.step()      # 优化器更新（FSDP 分片状态）
```

**forward_step 内部**（`prepare_model_inputs` → 模型前向 → `loss_function`）：

- `prepare_model_inputs`（transformer_impl.py:923）：`use_remove_padding` 时把 padded input_ids 用 `remove_padding` 转成纯 token 序列（`unique_sequence_ids`/`indices`），`position_ids` 同样 unpad——**FSDP 的"remove-padding 前向路径"**，与 TQ 的变长数据天然契合（这也是 master infer 反而快 2s 的原因）；
- `loss_function` = `ppo_loss`（第 11 点）：在 logits 上算 policy ratio/clip/KL，返回 scalar loss；
- `prepare_micro_batches`（`verl/workers/engine/utils.py:91`）：断言 `per_GPU_batch % (force_group_size × micro_batch_size_per_gpu) == 0`——**这就是 0.8.0 升级时 ERROR 1 的来源**（第 16 点）。

### 3. 具体数值样例

- 每 GPU 8 条、`micro_batch_size_per_gpu=4`：`prepare_micro_batches` 断言 8 % (1×4) = 0 ✓ → 2 个 micro-batch；
- 每个 micro-batch 前向：`prepare_model_inputs` 把 4 条 padded → remove_padding 成实际 token 流（如 4 条 × 平均 700 token = 2800 token）→ 模型前向 → `ppo_loss`（loss_mask 掩码、batch_num_tokens 归一化）→ `loss.backward()`；
- 梯度：FSDP 下每层 backward 后 all-reduce 梯度（分片），`optimizer_step` clip 到 1.0 后更新——**与 baseline 完全相同的 FSDP 引擎，差异只在数据输入形态（nested vs padded）**。

> 面试一句话总结：**前向反向在 FSDPEngine.forward_backward_batch：先 all_reduce 全 DP 的 loss_mask 和得到归一化分母，prepare_micro_batches 按 micro_batch_size_per_gpu 切分（整除断言），每个 micro-batch 走 prepare_model_inputs（remove-padding 变长前向）→ loss_function（ppo_loss）→ loss.backward()；optimizer_step 做 clip_grad_norm + step——两方案共用同一引擎，差异全在喂进去的数据形态。**

---

## 11. ppo_loss 计算（losses.py：ratio / clip / KL / 归一化）

### 1. 现有问题

GRPO 的损失到底怎么算？为什么 TQ 版在这里慢 15s？——`ppo_loss` 的**字段布局转换**是答案。

### 2. 方法论（`verl/workers/utils/losses.py:85-91`）

```python
# verl/workers/utils/losses.py —— ppo_loss 的字段准备（TQ_PERF_COMPARE 的 15s 根因点）
def ppo_loss(data, ...):
    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()   # ★ nested → padded（每 micro-batch 一次）
    ...
    # 逐 token 的 ratio：
    #   log_ratio = log_probs - old_log_probs
    #   ratio = exp(log_ratio)
    #   pg_loss = -min(ratio*adv, clip(ratio, 1±ε)*adv) * response_mask 求和 / batch_num_tokens
    # KL 项：k3 = exp(log_ratio_ref) - log_ratio_ref - 1（ref_log_prob - log_probs）
    # loss = pg_loss + kl_beta * kl_loss
```

- **baseline（无 TQ）**：字段本来就是 **padded 普通张量**，`to_padded_tensor()` 是 **no-op**，每次 micro-batch 零成本；
- **master（TQ）**：从 TQ 读出的字段**几乎全是 NestedTensor**，`ppo_loss` 每个 micro-batch 都要 `select(...).to_padded_tensor()` **真转换**一次，加上 `index_select_tensor_dict`、`micro_batch.to(device)`、nested 求和都作用在 10+ 个 nested 字段上——昇腾 NPU 上 `torch.nested` 算子多为慢速/回退路径，4 个 micro-batch 累计 ≈ **15s**；
- 修复方案（`TQ_PERF_COMPARE_AND_FIX.md` 方案 A）：`train_mini_batch` 入口一次性把 loss 字段（response_mask/loss_mask/old_log_probs/ref_log_prob/advantages/returns/rm_scores/...）转 padded，`input_ids/position_ids` 保持 nested（remove-padding 前向需要）——**每个 update_actor 只转一次，而不是每 micro-batch 转 4 次**。

### 3. 具体数值样例

- 4 个 micro-batch × 每次 `to_padded_tensor`（10 个 nested 字段 × 每条 700 token）≈ 每次 3-4s → 4 次 ≈ 15s（实测 master update_actor 17s vs baseline 2s）；
- 修复后：`train_mini_batch` 入口转一次 ≈ 0.05s，`ppo_loss` 内变 no-op → update_actor 回落到 ~2-5s；
- 数值等价性：loss_mask 转 padded 后 `batch_num_tokens = loss_mask.sum()` 不变（padding 补 0）；ratio 只在 response_mask=1 的位置计算。

> 面试一句话总结：**ppo_loss 是 GRPO 的核心（ratio=exp(logp-old_logp)，clip 到 [1-ε,1+ε] 乘 advantage，加 k3 KL 惩罚，loss_mask 归一化）；TQ 版慢 15s 的根因就是它每 micro-batch 调一次 to_padded_tensor（nested→padded 真转换，昇腾回退路径），baseline 是 no-op——修复 = train_mini_batch 入口一次性转换，input_ids 保持 nested 走 remove-padding 前向。**

---

## 12. baseline（verl-0.8.0 无 TQ）的 update_actor：DataProto 整包流

### 1. 现有问题

无 TQ 版对应流程长什么样？——数据全在 driver 的 DataProto 里，padded 布局，每次阶段转换。

### 2. 方法论（`ray_trainer.py`）

**driver 侧 `_update_actor`**：

```python
# verl/trainer/ppo/ray_trainer.py —— baseline 的 _update_actor
def _update_actor(self, batch: DataProto) -> DataProto:
    batch_td = batch.to_tensordict()                  # DataProto → TensorDict（padded）
    batch_td = left_right_2_no_padding(batch_td)      # 左 pad prompt / 右 pad response → 无 padding
    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
    ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n  # ×n
    tu.assign_non_tensor(batch_td, global_batch_size=ppo_mini_batch_size,
                         mini_batch_size=ppo_mini_batch_size, epochs=..., seed=...,
                         dataloader_kwargs={"shuffle": ...}, compute_loss=True)
    actor_output = self.actor_rollout_wg.update_actor(batch_td)   # 完整 TensorDict 走 RPC
    actor_output = tu.get(actor_output, "metrics")                # 收集 metrics
    ...
```

**worker 侧**：与 TQ 版**完全相同**的 `ActorRolloutRefWorker.update_actor → FSDPEngine.train_mini_batch`——差异只在**数据怎么到 worker**：baseline 是 RPC 带完整 TensorDict（padded → left_right_2_no_padding 后是"紧凑 padded"），TQ 版是 worker 侧 `kv_batch_get` 拉变长 NestedTensor。

**其他阶段的差异**（`_compute_old_log_prob` baseline）：

```python
def _compute_old_log_prob(self, batch: DataProto):
    batch_td = batch.to_tensordict()                  # DataProto → TensorDict
    batch_td = left_right_2_no_padding(batch_td)      # padding → no-padding
    output = self.actor_rollout_wg.compute_log_prob(batch_td)   # RPC 前向
    log_probs = tu.get(output, "log_probs")
    log_probs = no_padding_2_padding(log_probs, batch_td)       # 转回 padded
    result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
    old_log_prob = DataProto.from_tensordict(tu.get_tensordict(result))
    return old_log_prob, mfu                    # 返回 DataProto，由 fit 里 batch.union 拼回
```

**关键点**：
- **pad/unpad 往返**：baseline 每阶段做 `to_tensordict → left_right_2_no_padding（unpad）→ 计算 → no_padding_2_padding（repad）→ union 回 batch`——**数据在 padded/no-padded 之间反复转换**；
- **union 拼接**：old_log_probs/ref_log_prob/advantages 都是"算完一块、union 一块"到 driver 的 batch 上——batch 字段随阶段递增；
- **数据永远在 driver**：任何阶段要数据，数据已在手（无 TQ 拉取），代价是**每次 RPC 全量序列化**（128 条 × 全部字段）。

### 3. 具体数值样例

- 补齐后的 batch（如 128 条，含 pad/upsample）进 `_update_actor`：`to_tensordict`（128×1024 padded）→ `left_right_2_no_padding`（保留实际 token）→ 塞 mini_batch_size=128 → RPC 到 16 worker → 每 worker 8 条 → `make_iterator` 断言 8 % 8 = 0 ✓（若 batch 数不足导致 per-GPU 非整除则触发 **ERROR 2**，见第 16 点）；
- 每阶段数据量：driver 内存中 batch 字段随 union 增长（input_ids+attention_mask+position_ids+response_mask+old_log_probs+ref_log_prob+advantages+returns+rm_scores ≈ 9 个字段 × 128 × 1024 × 8B ≈ 9 MB 常驻）；
- 对比 TQ：同 128 样本，TQ 里按需拉字段（每个阶段 2-8 个字段、变长），driver 常驻只有 KVBatchMeta（KB 级）。

> 面试一句话总结：**baseline 的 update_actor = DataProto 整包流：driver 反复 to_tensordict/left_right_2_no_padding（unpad）/no_padding_2_padding（repad）/union，数据常驻 driver 内存、RPC 全量传；worker 侧引擎与 TQ 版相同——差异本质是"数据在手、pad 往返、全量 RPC" vs "数据在 TQ、变长、按需拉取"。**

---

# 四、两方案读写对比

## 13. rollout 阶段对比（HTTP / TQ 写入 / 异步 / 开销）

### 1. 现有问题

把 rollout 阶段的读写差异浓缩成一张表，回答"哪里 HTTP、哪里写 TQ、是否异步、开销哪去了"。

### 2. 方法论（对比表）

| 维度 | master（TQ） | verl-0.8.0（无 TQ） |
|---|---|---|
| 任务派发 | `store.enqueue_many_rollouts` 批量入队（v1）/ HTTP `queue_task`（v0） | `async_rollout_manager.generate_sequences`（内部 agent-loop） |
| 推理 HTTP | Agent → HTTP → LLMProxy → HTTP → vLLM（LLMServerManager 地址注册为资源） | 同（agent-loop 里 agent 直接 HTTP 调 vLLM） |
| 轨迹存储 | span → store（add_otel_span）；triplet 由 daemon 构建 | span → store；triplet 由 daemon 构建（两边相同） |
| **写入数据** | **`tq.kv_batch_put`（NestedTensor，变长，120 keys）** | `batch.union(gen_batch_output)`（DataProto 拼进 driver） |
| 运行屏障 | `kv_put(key, tag={global_steps, status:running})` 128 个 | 无（generate_sequences 同步返回） |
| 等待方式 | `replay_buffer.sample` 阻塞（后台线程轮询 tag） | `generate_sequences` 阻塞（RPC 等待） |
| 异步性 | 三层异步（ReplayBuffer 轮询 / daemon 5s 拉取 / agent 并发执行） | agent 执行并发，但结果收集是同步 RPC |
| 写入开销 | triplet 构建 + kv_batch_put ≈ 秒级，**被 agent 执行墙钟掩盖** | union 拼接 ≈ 毫秒级（数据已在手） |
| 显存形态 | rollout 引擎 gen 阶段唤醒、训练阶段休眠（wake/sleep_replicas） | 同（checkpoint_manager 相同） |

### 3. 具体数值样例

- master：128 个 rollout 的 triplet 在 agent 执行期间陆续构建，`kv_batch_put`（120 key、~0.7 MB）在 agent 全部完成后一次写入（~50ms）——相对 6 分钟 agent 墙钟**完全被掩盖**；
- baseline：`generate_sequences` 返回 DataProto 后 `union`（内存拼接 ~10ms），同样被掩盖；
- **异步结论**：两边 rollout 都是"agent 并行执行 + driver 等待"，但 master 的等待基于 TQ 状态 tag（可与后续阶段解耦、支持未来 pipeline 重叠），baseline 的等待基于同步 RPC 返回值（强耦合）。

> 面试一句话总结：**rollout 阶段：两边都用 HTTP 走 Agent→LLMProxy→vLLM 推理、span 存 store、daemon 转 triplet；差异在写入——master 把 triplet 以 NestedTensor 变长 kv_batch_put 进 TQ 并写 128 个运行屏障，baseline 把生成结果 union 进 driver 的 DataProto；两者写入开销都被 agent 执行墙钟掩盖，但 master 的等待机制（TQ 状态 tag + 后台轮询）与训练解耦得更彻底。**

---

## 14. update_actor 阶段对比（读数据 / 字段布局 / 前向反向 / 收集）

### 1. 现有问题

把训练阶段（尤其 update_actor）的读写差异浓缩成表。

### 2. 方法论（对比表）

| 维度 | master（TQ） | verl-0.8.0（无 TQ） |
|---|---|---|
| 数据从哪读 | worker 侧 `tqbridge → kv_batch_get`（TQ，变长 NestedTensor） | driver 侧 `batch.to_tensordict + left_right_2_no_padding`（内存，padded） |
| 传输方式 | 元数据（KVBatchMeta）走 Ray RPC，数据按需拉 | 完整 TensorDict 走 Ray RPC（全量序列化） |
| 字段布局 | **nested（变长）**，`ppo_loss` 内转 padded | **padded**，阶段间 unpad/repad 往返 |
| mini_batch 切分 | `train_mini_batch`（make_iterator） | 相同（worker 侧共用） |
| 前向反向 | FSDPEngine.forward_backward_batch（共用） | 完全相同（共用同一引擎） |
| 归一化 | batch_num_tokens = loss_mask.sum + DP all_reduce | 相同 |
| 结果收集 | 输出 kv_batch_put 写回 TQ → driver reduce_metrics | worker 返回 TensorDict → `tu.get(output,"metrics")` |
| 常驻数据 | driver 只持 KVBatchMeta（KB 级） | driver 持 9+ 字段 DataProto（MB 级） |
| **update_actor 耗时** | **慢 ~15s**（nested→padded 转换，昇腾回退路径） | ~2s（to_padded_tensor no-op） |
| old_log_prob/ref | **快 ~2s**（变长免 pad/unpad，remove-padding 前向） | 慢 ~2s（pad 往返 + GPU unpad） |

### 3. 具体数值样例

- 同一批数据（baseline pad/补齐后 ~128 条 vs master 120 keys + upsample）：baseline 每阶段 pad/unpad 往返 2 次（128→120→补齐），master 全程变长只在 ppo_loss 里转 padded；
- 4 个 micro-batch × to_padded_tensor ≈ 15s（master 净增），2 个 infer 阶段（old_log_prob/ref）各省 ~1s（master 净省）；
- 结果：master update_actor 17s vs baseline 2s（+15s），ref+old_log_prob master -2s——**净 +13s/step，这是 TQ 方案当前的主要性能代价（有明确修复方案）**。

> 面试一句话总结：**update_actor 阶段两方案的差异高度集中：数据读取（TQ 变长 vs driver padded）、传输（元数据 vs 全量 RPC）、字段布局（nested vs padded）；前向反向与 FSDP 引擎完全共用——性能账 = update_actor 因 nested→padded 慢 15s，但 old_log_prob/ref 因变长免 pad 快 2s，净 +13s/step，修复点在 train_mini_batch 一次性转换。**

---

## 15. 性能差异根因与修复（TQ_PERF_COMPARE 的完整结论）

### 1. 现有问题

为什么 TQ 接入后训练反而慢？15s + 5s 的根因与修复，面试必问。

### 2. 方法论（根因链 + 修复方案）

**已确认/已排除的事实**（`TQ_PERF_COMPARE_AND_FIX.md`）：
1. trainer 基类不同（PPOTrainer vs RayPPOTrainer），但 worker 类/FSDP engine/超参完全一致；
2. batch 规模一致（245 左右 triplet，pad 后都是 128 的倍数）——排除；
3. **TQ 读取不是瓶颈**（worker 侧 TQ 读取仅 ~0.5s）——排除；
4. **根因 = 训练数据字段布局**：baseline `left_right_2_no_padding` 只把 input_ids/position_ids/loss_mask 转 nested，response_mask/old_log_probs/ref_log_prob/advantages/returns/... 都是 **padded 普通张量**；master 从 TQ 读出的**几乎全部是 nested**。

**慢 15s 的机制**（update_actor）：
- `ppo_loss`（losses.py:85-91）每个 micro-batch 执行 `data.select(fields).to_padded_tensor()`；
- baseline：字段已 padded → no-op；
- master：10+ 个 nested 字段真转换 + `index_select_tensor_dict`/`micro_batch.to(device)`/nested 求和 → 昇腾 NPU 的 `torch.nested` 是慢速/回退路径 → 4 个 micro-batch 累计 ~15s；
- 为什么 infer（old_log_prob/ref）反而快 2s：纯前向不吃 ppo_loss 的转换，master 变长数据还省掉 baseline 的 pad/unpad 与 GPU unpad 开销。

**慢 5s 的机制**（update_weights）：TQ 栈（controller + 8 storage unit actor + store server + llm_proxy + replay buffer 轮询线程 + 每步上百次临时线程/事件循环）与 FSDP `param_offload=true/optimizer_offload=true` 的 host 侧传输/汇聚**抢 CPU 和内存带宽**。

**修复方案**（方案 A，最小改动）：
- `train_mini_batch` 入口把 loss 字段（response_mask/loss_mask/old_log_probs/ref_log_prob/advantages/returns/token_level_rewards/token_level_scores/rm_scores/entropy）**一次性** `torch.nested.to_padded_tensor`，input_ids/position_ids 保持 nested（remove-padding 前向需要）；
- 效果：ppo_loss 的 to_padded_tensor 变 no-op，每 update_actor 只转一次（~0.05s vs 4×3-4s）；预期 update_actor 从 ~17s 回到 ~2-5s；
- 方案 C（update_weights -5s）：TQ controller/storage 限定 CPU、`SimpleStorage.num_data_storage_units` 8→1-2、`TQ_NUM_THREADS` 调低、关闭 `_update_tq_reward_async` 对照验证。

### 3. 具体数值样例

- 修复前后：update_actor 17s → 2-5s（-12~15s）；update_weights -5s；保持 infer -2s 收益 → **整体 step 时间从 +13s 变为 -2~-7s（TQ 方案反超）**；
- 验证闭环：`[BENCH-LOSS] to_padded_tensor` 应降到 ~0.000s；`[BENCH-TRAIN]` 每 micro-batch 回落；`actor/loss`、`actor/grad_norm` 数值一致（转换不引入偏差）。

> 面试一句话总结：**TQ 慢的根因不在传输而在字段布局：TQ 存的变长 NestedTensor 在 ppo_loss 每 micro-batch 都要转 padded（昇腾回退路径，4 次 ≈15s），baseline 本来 padded 是 no-op；修复 = train_mini_batch 入口一次性转换（input_ids 保持 nested 走 remove-padding），update_weights 慢 5s 是 TQ 栈与 FSDP offload 抢 CPU——修完后 TQ 方案在 infer 上还快 2s，整体反超。**

---

## 16. 0.6.1→0.8.0 升级的兼容性坑（VERL_UPGRADE_ANALYSIS）

### 1. 现有问题

升级 verl 不是换版本那么简单——0.8.0 引入了**固定尺寸 micro-batch 断言**与 **mini_batch_size 归一化位置变化**，agent 动态产出的 batch 大小直接触发断言失败。

### 2. 方法论（两个 ERROR + 两个修复）

**ERROR 1：`prepare_micro_batches` 整除断言**（`verl/workers/engine/utils.py:91`）：
- 0.8.0 新增 `per_GPU_batch % (force_group_size × micro_batch_size_per_gpu) == 0`，即总 batch 必须能被 `world_size × micro_bsz = 16×4 = 64` 整除；
- agent daemon 产出 14/50/100/120 条 → pad 后 16/64/112/128 → per-GPU 1/4/7/8 → 只有 64/128 通过（100 → 112 → 7 % 4 ≠ 0 ✗）；
- **修复**：`pad_dataproto_to_divisor(batch, world_size * micro_bsz)`（除数 16 → 64），forward-only 阶段多 pad dummy 代价可控。

**ERROR 2：`make_iterator` 断言**（`verl/utils/tensordict_utils.py:588`）：
- 0.8.0 的 `_update_actor` 把 `mini_batch_size = ppo_mini_batch_size × rollout.n`（32×4=128），per-GPU 128/16=8，要求总 batch 是 128 的倍数；agent 产出 96 → per-GPU 6 → 6 % 8 ≠ 0 ✗（0.6.1 的 dispatch 会 rebalance、0.8.0 的 TensorDict dispatch 不会）；
- **修复**：agent-lightning override `_update_actor`——**不乘 rollout.n**（agent 已经把 n 条 response 展开成独立 triplet）：

```python
# agentlightning/verl/trainer.py —— override _update_actor（关键：去掉 × rollout.n）
def _update_actor(self, batch: DataProto) -> DataProto:
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)
    # 关键：不乘 rollout.n（agent daemon 已展开 triplet）
    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size  # 32
    tu.assign_non_tensor(batch_td, global_batch_size=ppo_mini_batch_size,
                         mini_batch_size=ppo_mini_batch_size, epochs=...,
                         dataloader_kwargs={"shuffle": ...}, compute_loss=True)
    actor_output = self.actor_rollout_wg.update_actor(batch_td)
    ...
```

- 效果：`mini_batch_size_per_gpu = 32/16 = 2`，per-GPU 一定是 2 的倍数（agent-lightning 已保证总 batch 是 32 的倍数）→ 断言永远通过。

**数据流链条**（含原生 verl 与 agent-lightning 改造后的对比）：
```
原生 verl 0.6.1（改造前）：128(rollout) → 120(triplet) → pad16→128 → old_log_prob → unpad→120
       → is_drop→110 → floor32→96（丢弃 14 条）→ update_actor（dispatch rebalance 兼容）
原生 verl 0.8.0（改造前）：120 → pad64→128 → old_log_prob → unpad→120 → is_drop→110
       → floor32→96（丢弃 14 条）→ update_actor（原生 mini_bsz=128 → per-GPU 断言可能炸）
baseline（agent-lightning 改造后）：120 → pad(lcm(dp,mini))→128 → old_log_prob → unpad→120
       → is_drop→110 → 向上 pad 到 per_gpu_divisor（response_mask 置 0）→ update_actor（override 后 mini_bsz=32 ✓）
master(TQ)：120 triplet → NestedTensor 写 TQ → balance_batch upsample（合成 no-op 补齐）→ update_actor（无 pad 往返）
```
> 注意：`floor32→96` 只存在于**原生 verl 0.6.1/0.8.0（agent-lightning 改造前）**；改造后两个分支都是**向上补齐**（baseline pad dummy + response_mask 置 0、master upsample no-op 样本），**不丢弃任何真实 triplet**（丢弃的是 padding 标记，不参与 loss）。

### 3. 具体数值样例

- daemon 产出 100 条：原生 0.8.0 原版 pad 到 112 → per-GPU 7 → ERROR 1；修复后 pad 到 128 → per-GPU 8 ✓；
- daemon 产出 96 条：原生 0.8.0 原版 mini_batch_size=128 → per-GPU 6 → ERROR 2；override 后 mini_batch_size=32 → per-GPU 2 ✓（且 agent-lightning 对不足的 batch 是向上 pad 补齐，不再有 floor 丢弃）；
- 含义：**agent 场景 batch 大小动态波动（0~128），必须让"pad 除数"和"mini_batch 语义"都适配 agent 的 triplet 展开模型**——这是升级 verl 做 AgenticRL 的核心兼容性工作。

> 面试一句话总结：**0.6.1→0.8.0 的坑是两处整除断言：prepare_micro_batches 要求总 batch 是 world_size×micro_bsz 的倍数（pad 除数 16→64），_update_actor 的 make_iterator 要求是 mini_batch_size 的倍数（override 去掉 ×rollout.n，因为 agent 已把 n 条 response 展开成 triplet）——agent 动态 batch 大小在 0.8.0 的固定尺寸断言下必然踩雷，这正是实习"修复升级引入的数据流兼容问题"的内容。**

---

# 五、面试问答与速查

## 17. 高频追问速答

**Q1：TQ 方案数据到底存在哪？**
TransferQueue（partition=train/val），每 key 一个样本的 NestedTensor 字段；可落 Mooncake 存储后端（DRAM/SSD）并经 RDMA 跨机传输——数据不在 driver 内存、不在 worker 内存，在 TQ（`TQ.md` / `Mooncake.md`）。

**Q2：KVBatchMeta 和 DataProto 的区别？**
KVBatchMeta = keys + tags + fields 集合（元数据，KB 级）；DataProto = 完整字段数据（MB 级）。前者配 TQ 按需拉取，后者整包携带。

**Q3：update_actor 数据读取耗时？**
TQ 读取 ~0.5s（可忽略）；真正慢的是 nested→padded 转换（~15s，可修复为一次性转换）。

**Q4：rollout 和训练怎么并行？**
agent 执行并发 + ReplayBuffer 后台轮询 TQ tag + daemon 异步收集；训练在 rollout 全部完成后串行开始（当前同步 step，未来可 pipeline 重叠）。

**Q5：TQ 写入开销被 rollout 吃掉了吗？**
是——daemon 的 triplet 构建 + kv_batch_put 与 agent 执行并行，被墙钟掩盖；但训练侧 ppo_loss 转换是串行的、没被掩盖（15s 暴露）。

**Q6：为什么 infer（old_log_prob/ref）TQ 反而快？**
变长数据免 pad/unpad 往返 + remove-padding 前向天然契合；baseline 每阶段 GPU unpad 有开销。

**Q7：两方案 FSDP 引擎一样吗？**
完全一样（FSDPEngine.forward_backward_batch 共用），差异只在数据输入形态与传输方式。

**Q8：修复 15s 的落地方式？**
train_mini_batch 入口一次性 to_padded_tensor（input_ids/position_ids 保持 nested），ppotrainer 内转换变 no-op；验证 loss/grad_norm 数值一致。

**Q9：0.8.0 升级为什么报 14%8 != 0？**
0.8.0 的 _update_actor 把 mini_batch_size 乘了 rollout.n（128），而 agent 已展开 triplet——override 去掉乘法后 per-GPU mini_batch=2，永远整除。

**Q10：Mooncake 在 TQ 里的角色？**
TQ 的 StorageManager 后端之一（MooncakeStorageManager/MooncakeStoreClient），数据落 Mooncake Store（DRAM+SSD），RDMA 传输——训练数据跨机传输走 RDMA 而非 TCP（`Mooncake.md`）。

## 18. 面试一句话总结（背诵版）

- **两方案本质**：TQ = 数据仓库（变长 NestedTensor 按 key 存，driver 拿元数据按需读写）；无 TQ = driver 整包 DataProto（padded，pad/unpad 往返）；
- **rollout 读写**：agent 执行 + span 存 store + daemon 转 triplet + `kv_batch_put` 写 TQ（master）/ `union` 进 DataProto（baseline）；HTTP 在任务派发与 Agent→LLMProxy→vLLM 推理两处；
- **update_actor 读写**：tqbridge `kv_batch_get` 从 TQ 拉变长字段 → FSDP train_mini_batch → 前向 ppo_loss + backward → optimizer_step → 输出写回 TQ；baseline 是 driver 转换 + RPC 全量传；
- **性能账**：TQ 读取 0.5s 可忽略；nested→padded 转换 +15s（修复：入口一次性转换）；infer -2s；update_weights CPU 竞争 +5s；
- **升级坑**：0.8.0 两处整除断言（pad 除数 64、mini_batch 不乘 n）——agent 动态 batch 的兼容性修复。

---

# 附：速查表

## 关键文件 ↔ 作用（两方案共用/差异）

| 文件 | 作用 | 方案 |
|---|---|---|
| `agentlightning/verl/trainer.py` | `AgentLightningTrainer._train_step`（gen/reward/balance/compute/update 编排） | master |
| `agentlightning/verl/daemon.py` | `AgentModeDaemon`：派任务、等 rollout、`get_train_data_batch`（TQ 写入） | master |
| `verl/trainer/main_ppo_sync.py` | `PPOTrainer`：ReplayBuffer、`_compute_old_log_prob/_compute_advantage/_update_actor`（TQ 读写） | master |
| `verl/utils/transferqueue_utils.py` | `tqbridge`：KVBatchMeta ↔ TensorDict 桥（`kv_batch_get/put`） | master |
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer`：DataProto 整包流（`_compute_old_log_prob/_update_actor`） | baseline |
| `verl/workers/engine_workers.py` | `ActorRolloutRefWorker.update_actor` / `TrainingWorker.train_mini_batch` | 共用 |
| `verl/workers/engine/fsdp/transformer_impl.py` | `FSDPEngine.forward_backward_batch/optimizer_step`（前向反向） | 共用 |
| `verl/workers/utils/losses.py` | `ppo_loss`（nested→padded 转换点，15s 根因） | 共用 |
| `transferqueue/transfer_queue/` | TQ：controller/client/StorageManager/Mooncake backend | master |
| `agent-lightning/TQ_PERF_COMPARE_AND_FIX.md` | 性能对比与修复方案 | 文档 |
| `agent-lightning/VERL_UPGRADE_ANALYSIS.md` | 0.6.1→0.8.0 数据流与 ERROR 分析 | 文档 |

## 核心数字

- `train_batch_size=32`、`rollout.n=4` → 128 rollouts/step；triplet 120 → master upsample 到 128 / baseline pad 补齐到 per_gpu_divisor（都向上补齐，不丢弃真实样本；原生 verl 才是 floor32→96）
- `ppo_mini_batch_size=32`、`micro_batch_size_per_gpu=4`、DP=16 → per-GPU 8 条 → 2 micro-batch
- 性能：TQ 读取 0.5s；nested→padded 4×3-4s ≈ 15s（修复后 ~0.05s）；infer -2s；update_weights +5s（CPU 竞争）
- 断言：总 batch % 64 == 0（prepare_micro_batches）；per-GPU % mini_bsz_per_gpu == 0（make_iterator）

## 简历亮点 ↔ 本文章节映射

| 实习亮点 | 对应章节 |
|---|---|
| verl 0.6.1→0.8.0 升级（数据流兼容修复） | 第 16 点（两个 ERROR + override） |
| TQ MoonCake 存储后端接入、RDMA 高速传输 | 第 6、8 点（kv_batch_put/get）+ `Mooncake.md`/`TQ.md` |
| RayPPOTrainer → PPOTrainer 迁移 | 第 1、9 点（两 trainer 数据流差异） |
| 昇腾 NPU 单/双机全链路 | 第 15 点（torch.nested 回退路径）+ `Communication.md` |
| 训练/推理/环境/奖励解耦 | 第 2、4、7 点（TQ 数据面 + HTTP 推理面 + 异步） |
