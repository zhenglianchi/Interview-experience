# 从空泡到异步：Rollout 与 Training 的算力编排（问题驱动版）

> **一句话知识框架**：RL 训练里 GPU 大量时间花在"等"上——等 rollout 采完、等同步 barrier、等最慢的那条轨迹。这些"GPU 空转"的时间片就是**空泡（bubble）**。所有训练/推理编排技术（colocate、one-step-off、fully async、partial rollout、delta weight sync、staleness 控制、长尾治理）本质上都是在**用不同方式填这个空泡**，代价可以归到三笔账上：**显存账、通信账、off-policy 账**。这篇按"空泡从哪来 → 每种方法具体怎么做 → 代价记在谁头上"来讲。

> 素材来源：`verl-latest`（commit `483512d`）的 `docs/advance/fully_async.md`、`docs/advance/delta_weight_sync.md`、`docs/perf/rollout_kv_offload.md`；AReaL `docs/en/algorithms/async.md`；`uniagent-lighting` 的 vendored verl v1（`trainer_colocate_async.py` / `trainer_separate_async.py`）；Polar 论文（arXiv:2605.24220）；NeMo-RL `docs/guides/refit.md`、`docs/guides/async-grpo.md`。配套阅读：`Training-Inference-Mismatch.md`、`internship.md`、`Ray.md`、`K8S-KubeRay.md`。

---

## 1. 空泡是什么：GPU 在等，而不是在算

### 1. 问题

一个 RL step 的逻辑顺序是：**rollout（采样）→ 算 reward/advantage → train（前向反向）**。最朴素的实现是严格串行：采样阶段所有 GPU 都在跑推理，训练阶段所有 GPU 都在跑训练，看起来"没有空闲"。

但真实系统里 GPU 的空转发生在**三个更隐蔽的地方**：

| 空泡类型 | 现象 | 根因 |
|---|---|---|
| **长尾空泡** | rollout 阶段快样本早就生成完了，但必须等最慢的那条轨迹 | agent 轨迹长度方差极大（SWE 任务从 2 分钟到 40 分钟） |
| **切换空泡** | 从"推理态"切到"训练态"要换出权重/KV、重分片、重分配显存 | 共卡架构的固有成本 |
| **同步空泡** | 训练阶段 rollout 引擎完全闲置（反之亦然） | 同步 barrier：一个 step 内两种负载不重叠 |

还有一类容易被忽略：**批内空泡**——同一批样本长度差异大时，padding 或按最长样本分配资源，短样本那部分算力被浪费（这条在 `AgenticRL-Trajectory-Stitching.md` 里有专讲）。

### 2. 量化：空泡到底有多大

用一个具体 step 算账。设：rollout 需要 $T_r=100$ 单位时间，其中 90% 的样本 60 单位完成、**10% 的长尾要 400 单位**；train 需要 $T_t=60$ 单位；8 张卡。

**严格同步（先采完再训）**：

$$
T_{\text{step}} = \underbrace{400}_{\text{长尾决定}} + 60 = 460
$$

而"没有长尾"的理想值是 $100+60=160$。**460 里有 300 单位（65%）是空泡**——这 300 单位里，是一张卡在跑长尾、其余 7 张在等。

再算**利用率**：8 卡 × 460 = 3680 卡·单位的预算，有效工作量约为 $8\times60$（rollout 主体）+ $8\times60$（训练）= 960 卡·单位。**有效利用率只有 26%**。

Polar 论文给了行业实测——agentic harness 场景下 `per_request` 重建策略的 **rollout GPU 平均利用率只有 20.4%**，换成 `prefix_merging` 后是 **87.7%**。两者量级一致，说明"长尾 + 同步等待"确实能吃掉七八成算力。

### 3. 结论：三层解法

填空泡有三条路，按"改动从小到大"排列：

1. **让两种负载在时间上重叠**（colocate 时分复用 → one-step-off → fully async）：不减少空泡本身，而是**用另一种负载填进去**；
2. **把长尾从关键路径摘走**（partial rollout、长尾迁移、prefix KV 复用）：减少空泡的产生；
3. **降低切换与同步成本**（sleep/wake 优化、delta weight sync）：减少空泡的单价。

**下文第 2~8 节就是七种具体方法。**

> 面试一句话总结：**RL 训练里 GPU 的大头时间花在"等"上——等最慢的轨迹（长尾空泡）、等两种负载切换（切换空泡）、等同步 barrier（同步空泡）；一个 10% 长尾到 400 单位的 step，有效利用率只有约 26%（实测 agentic 场景 rollout 利用率 20.4%）；所有编排技术都是在用不同方式填这个空泡，代价记在显存/通信/off-policy 三笔账上。**

---

## 2. 方法一：Colocate / Hybrid —— 让两种负载时分复用同一批 GPU

### 1. 做法

**核心思想**：不新增卡，让**同一批 GPU 先跑推理、再跑训练**，把原本"一方空闲"的时间片填上。verl 里这是默认且唯一受支持的形态（`hybrid_engine=True`）。

**关键前提：显存也必须时分复用。** 推理引擎启动时会预分配 KV cache（往往占掉大半显存），训练侧还要放 activation 和 optimizer state，两者峰值叠加必然 OOM。所以 colocate 必须配套 **sleep / wake**。

**vLLM 的两个 sleep level**（官方文档原文）：

| Level | 行为 | 官方说明 |
|---|---|---|
| **Level 1** | 权重 offload 到 CPU + **丢弃 KV cache** | "Please make sure there's enough CPU memory" |
| **Level 2** | 权重和 KV **都丢弃** | "good for … update the model, where previous model weights are not needed, **e.g. RLHF weight update**" |

**RLHF 场景的标准四步**：

```python
engine.sleep(level=2)                 # 全丢，腾出最大显存给训练
engine.wake_up(tags=["weights"])      # 只先恢复权重
# → 在这里更新权重（resharding / delta sync）
engine.wake_up(tags=["kv_cache"])     # 再恢复 KV
```

**源码级真相**：`sleep_mode_backend.py` 里 `allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())`——**Level 2 就是空 tuple（全丢弃）**；真正的换出在 `cumem.py` 用 pinned CPU tensor + `cudaMemcpy`，然后在 C 侧 VMM `unmap_and_release`，**但保留虚拟地址**（`cuMemAddressReserve` 单独占着）。所谓"零拷贝"就是这个意思：**保留 VA，唤醒时把物理内存 map 回同一地址**。

**verl 的共卡四步 refit**（`engine_workers.py`，docstring 原文："before update_weights: rollout should be in sleep mode. after update_weights: rollout should be in wake_up mode."）：

```
resume(tags=["weights"]) → 权重更新 → resume(tags=["kv_cache"])
```

**一个成对实现的直观对照**（`uniagent-lighting` 的 vendored verl v1，共卡与分卡是两个 trainer 类）：

| | `trainer_colocate_async.py`（共卡） | `trainer_separate_async.py`（分卡） |
|---|---|---|
| 采样结束 | `on_sample_end` → `abort_replicas()` + `sleep_replicas()`（注释："sleep all replicas to **discard weights and kv cache**"） | — |
| 步末 | `on_step_end` → `update_weights()` + `resume_generation_replicas()` | — |
| 前置断言 | — | `assert checkpoint_engine.backend != "naive"` |

**分卡那条断言的注释把物理差异讲透了**：

> *"Colocate reward model … is not supported in separate async mode, because **the standalone rollout never pauses to free GPU memory**."*

即：**共卡之所以能省卡，正因为它愿意暂停并丢弃 KV 来换显存；分卡的 rollout 永不暂停，所以它不可能把显存让给同卡上的其他组件。**

### 2. 代价（记在哪笔账上）

- **显存账**：7B 模型 bf16 粗算，权重 14 GB + 梯度 14 GB + Adam fp32 动量与方差 56 GB ≈ **84 GB > 80 GB**——所以必须配合 `param_offload` / `optimizer_offload`（optimizer state 放 host），这又引入 H2D 拷贝；
- **切换账**：每次 sleep/wake 都是空泡（"切换空泡"的来源）；
- **稳定性**：显存竞争更激烈。实测 `colocate_async` 在 vLLM 0.11.1 上多轮生成 resume 时触发 **CUDA illegal memory access**（sleep/wake 竞态），只能退回分卡。

### 3. 两个必须知道的坑（否则会 OOM 或白做）

1. **vLLM DP > 1 时必须用 `engine.sleep()`，不能用 `collective_rpc("sleep")`**。verl 源码注释：`collective_rpc` 只到达单个 DP shard 内的 TP workers，**其他 DP shard 的权重没释放，会导致 FSDP 反向时 OOM**。
2. **verl 里 vLLM 的 `release_kv_cache` 是 TODO 空实现**——`CheckpointEngineManager.release_kv_cache_replicas()` 对 vLLM 实际无效，**只有 SGLang 真生效**。SGLang 的 `release_memory_occupation` 比 vLLM **多一个 `cuda_graph` tag**（三个 tag：`kv_cache` / `weights` / `cuda_graph`），且开头有断言 `assert self.is_fully_idle(), "release_memory_occupation should be called only when server is idle."`——**换出必须等引擎完全空闲**。

> 面试一句话总结：**colocate/hybrid = 同一批 GPU 上"算力和显存都时分复用"：推理态唤醒（带 KV）、训练态 sleep；vLLM 的 sleep level 1 换权重保 KV、level 2 全丢，RLHF 标准流程是 sleep(2) → wake_up(weights) → 更新 → wake_up(kv_cache)；代价是显存（7B 训练态 84GB > 80GB，必须 offload）、切换开销、以及 sleep/wake 竞态带来的稳定性风险。**

---

## 3. 方法二：One-step-off —— 最小改动的"错开一步"

### 1. 做法

**核心思想**：把 rollout 和 train 拆到**两组卡**上，让它们**错开一个 step**：trainer 训第 $N$ 步时，rollouter 已经在采第 $N+1$ 步。两种负载天然重叠，且 **staleness 固定为 1**，off-policy 程度可控。

**配置形态**（verl `one_step_off_ppo_trainer.yaml`）：

```yaml
actor_rollout_ref:
  rollout:
    free_cache_engine: False          # ← 分卡的标志：不需要换出显存
checkpoint_engine:
  backend: "nccl"                     # 权重必须真的送出去
```

**为什么"不需要 sleep/wake"是分卡的核心判据**：共卡必须换出（同卡显存要腾给训练），分卡各自有独立显存，**换出纯属浪费**。所以看到 `free_cache_engine: False` 基本可以断定是分卡形态。

### 2. 特性与限制

**这是一个"没有旋钮"的档位**：verl 的 one_step_off 路径里，`staleness` / `trigger_parameter_sync_step` / `require_batches` / `partial_rollout` **全部零匹配**——**lag 硬编码为 1**。想调异步度必须换到 `fully_async`。

**定位**：**改动最小、收益确定、风险最低的异步入门方案**。因为只错开一步，off-policy 程度和"训推不一致"本身同量级，不需要复杂 staleness 控制就能跑稳。

### 3. 收益上限

因为只有一步重叠，**长尾超过一个 step 时收益就封顶了**——trainer 还是要等。所以它是"入门方案"而不是"终极方案"。

> 面试一句话总结：**one-step-off = 把 rollout 和 train 拆到两组卡、错开一步执行，staleness 固定为 1；标志性配置是 `free_cache_engine: False`（分卡不需要换出显存）加一个真正的权重同步 backend（如 nccl，不能用 naive）；它是最好落地但收益上限有限的方案——lag 硬编码为 1，长尾超过一个 step 就等不到了。**

---

## 4. 方法三：Fully Async —— 多步异步 + 显式 staleness 控制

### 1. 做法

**核心思想**：干脆不要"一步"这个概念，把系统做成**生产者-消费者流水线**：rollouter 持续生产，trainer 攒够一批就训，两者节奏解耦。

**verl `fully_async_policy` 的四组件**：

```
① Rollouter  ──逐样本生产──►  ② MessageQueue  ──逐样本消费──►  ③ Trainer
     ▲                                                          │
     └──────────── ④ ParameterSynchronizer（NCCL）◄─────────────┘
```

| 组件 | 职责 | 关键设计 |
|---|---|---|
| **Rollouter** | 逐样本生成，放进队列 | 生产速度受 freshness 控制 |
| **MessageQueue** | 暂存样本 | **传输最小单位是"单条样本"**（streaming，不是整批） |
| **Trainer** | 逐样本取，攒够就训 | 训 `trigger_parameter_sync_step` 轮后触发一次权重同步 |
| **ParameterSynchronizer** | NCCL 权重同步 | 参考 MoonshotAI 的 `checkpoint-engine` |

**收益（官方实测）**：

> *"we achieved a **2.35x-2.67x** performance improvement when training the Qwen2.5-7B model with **128 GPUs**, without significantly affecting the results."*

### 2. 关键参数：staleness 怎么算

设 $S=\texttt{staleness\_threshold}$、$T=\texttt{trigger\_parameter\_sync\_step}$、$B=\texttt{require\_batches}$、$m=\texttt{ppo\_mini\_batch\_size}$：

- **同步（$S=0$）**：两次权重更新之间生产固定数量

$$
\text{rollout\_num} = T\times B\times m
$$

- **异步（$S>0$）**：最多生产

$$
\text{rollout\_num} = (1+S)\times(T\times B\times m) - \text{num\_staleness\_sample}
$$

其中 `num_staleness_sample` 是上一轮**超出**的陈旧样本数。

**代数字**：取 $T=4$、$B=2$、$m=32$ → 基准 $T\cdot B\cdot m=256$：

| $S$ | 最多生产 | 含义 |
|---|---|---|
| 0 | 256 | 同步 |
| 0.5 | 384 | 最多 50% 可能是上轮遗留 |
| 1.0 | 512 | 近似 one-step-off |

**官方调参建议（很重要）**：

> *"When rollout is fast enough, setting `staleness_threshold` to 1 is basically equivalent to one_step_off policy. To avoid too many expired samples affecting training accuracy, it is recommended to set this value to **less than 1**."*

**公平比较的公式**（否则"分卡更快"是假象）：

$$
\texttt{trigger\_parameter\_sync\_step} = \frac{\texttt{data.train\_batch\_size}}{\texttt{require\_batches}\times\texttt{ppo\_mini\_batch\_size}}
$$

即：要让两种形态**处理同样多的样本再同步一次权重**，比较才公平。

### 3. 另一个口径：AReaL 用"版本数"

| 框架 | 参数 | 语义 |
|---|---|---|
| **verl** | `async_training.staleness_threshold` | 允许的**陈旧样本比例**（建议 <1） |
| **AReaL** | `rollout.max_head_offpolicyness` | 允许落后多少个**版本步**（0=同步；典型 **2~8**） |

**两者语义不同，面试里别混。** AReaL 还要求 `use_decoupled_loss: true` 时必须 `recompute_logprobs: true`（因为 decoupled loss 需要 $\pi_{\text{old}}$ 和 $\pi_\theta$ 都在训练引擎的数值域里）。AReaL 给的收益参照：同步设置 "useful for debugging but typically **2x slower** than asynchronous training"。

### 4. NeMo-RL：staleness 不只是阈值，还是**采样策略**

NeMo-RL 把 staleness 做成**可插拔 sampler**（四选一）：

| sampler | 语义 |
|---|---|
| `in_order` | 严格按生成顺序消费 |
| `weight_fifo` | 按权重/新鲜度混合 |
| `ready_first` | 谁先就绪谁先训 |
| `windowed` | 固定窗口内任取 |

并给出完整 GRPO 修正推导：标准 GRPO 假设轨迹来自 $\pi_{\theta_{\text{old}}}$，异步下实际来自 $\pi_{\text{generator}}$，所以 loss 里要补 $\dfrac{\pi_{\text{training}}(x)}{\pi_{\text{generator}}(x)}$。样例口径很好记：**"v10 生成的轨迹可在 v10/v11 用，v12 丢弃"**（`max_trajectory_age_steps: 1`）。

> 面试一句话总结：**fully async = 生产者-消费者流水线（Rollouter + MessageQueue + Trainer + ParameterSynchronizer），逐样本 streaming 传输，NCCL 同步权重；verl 实测 128 卡 Qwen2.5-7B 提升 2.35~2.67×；核心旋钮 staleness_threshold（$S=0$ 同步、$S=1$ 近似 one-step-off、官方建议 <1），公式 rollout_num=(1+S)·T·B·m − 上轮遗留；AReaL 的 max_head_offpolicyness（版本数 2~8）是另一个口径；NeMo-RL 把 staleness 做成四种采样策略并给出 π_training/π_generator 修正。**

---

## 5. 方法四：Partial Rollout —— 同步点不杀进度，挂起续跑

### 1. 要解决的空泡

异步系统里还有隐蔽空泡：**权重同步时，正在跑的那批 rollout 怎么办？** 朴素做法是等它们跑完再换权重——**长尾在这里第二次伤害系统**（第一次是共卡的 step 阻塞）。一个任务跑 40 分钟、权重每 4 步同步一次，同步点就要等它。

### 2. 做法

**核心思想**：换权重时把进行中的生成**挂起（sleep）**，权重更新完再**恢复（resume）**继续生成。verl 官方描述：

> *"By adding `sleep() and resume()` logic, it **saves samples from ongoing rollouts and continues using them in the next rollout**, reducing the time spent waiting for ongoing tasks to finish during parameter synchronization."*

**关键约束**：

> *"`partial_rollout` **only actually takes effect when `staleness_threshold>0`**."*

即：**partial rollout 必然让一条轨迹跨版本，所以它天然制造 staleness**——必须和 staleness 控制一起用。

**AReaL 的对应描述**：*"a single trajectory can be **segmented across multiple policy versions**."*

### 3. 代价与边界

- **代价**：一条轨迹由多个版本的策略生成 → **本身就是 off-policy 混合体**，必须靠 IS 修正或 staleness 上限兜住；
- **SkyRL 的诚实说明**：容量不等式只在**总量**上约束，**不保证每条轨迹**都在 S 步内被训练；in-flight 语义由**推理引擎实现、harness 完全无感**（`/chat/completions` 被 block）；`pause_generation()` 会 abort 在途请求并保存部分结果，resume 时作为新请求重放。

> 面试一句话总结：**partial rollout = 权重同步时把进行中的生成 sleep 挂起、换完权重 resume 续跑，避免"等长尾跑完才能同步"；verl 明确它只在 staleness_threshold>0 时生效（因为它必然让一条轨迹跨多个策略版本）；代价是单条轨迹变成 off-policy 混合体，必须配 IS 修正；SkyRL 提醒这种容量约束只保证总量、不保证每条轨迹的时效。**

---

## 6. 方法五：权重同步优化 —— 把异步最大的一笔代价压下去

### 1. 为什么必需

分卡之后，**每次训练更新都必须把新权重送到 rollout 引擎**：

> *"In a disaggregated setup the trainer must broadcast its updated weights to the rollout engine **after every step**. By default this is a **full-weight broadcast whose cost grows with model size**."*

235B 模型全量广播一次是**几百 GB 通信**——实测 **246~266 秒/步**。**不优化这一项，异步省下的空泡收益会被同步成本吃回去。**

`uniagent-lighting` 的自报数据印证：**每步 48.1 秒里，NCCL 权重同步占约 27 秒**——超过一半。

### 2. 核心洞察：RL 的权重更新极度稀疏

> *"RL updates are highly sparse — under typical learning rates **over 99% of BF16 weight bytes are unchanged step-over-step**."*

**verl 的 `delta_sharded` 做法**（把差分**下沉到 all-gather 之前**）：

1. 每个 actor rank 只 pin **自己那个 FSDP shard** 的 CPU 快照；
2. 在本 rank 内做**字节级差分**，只把变化的 `(position, value)` 对 gather 到 rank 0；
3. gather 体量从"整个参数"降到"稀疏率"（约 1~3%），**且没有任何 rank 需要持有全模型快照**；
4. rank 0 只做"按 slot 拼接 + 分桶 + 广播"，**不需要知道任何布局知识**（`idx`/`val` 已是最终 HF 坐标）；
5. 接收端通过**同 GPU IPC** 原地更新（SGLang 用 `--custom-weight-loader` 钩子、vLLM 用 `WeightTransferEngine` + checkpoint patch API），**校验 checksum，不保留整份 rollout 模型副本**。

**正确性**：按位整数视图比较（bit-exact），无阈值、不漂移；每次 flush 带 checksum。**首次同步是 dense seed**（全量建基线），之后才走稀疏。

### 3. 实测收益（H100、GSM8K GRPO、verl V1 `separate_async`、SGLang rollout）

| 模型（布局） | `delta_sharded` | `nccl`（全量广播） | 加速 |
|---|---|---|---|
| Qwen2.5-7B（1+1 节点） | **3.9–4.9 s** | 5.5–6.0 s | ~1.3× |
| Qwen2.5-32B（2+2） | **11.2–11.9 s** | 17.7–18.1 s | 1.55× |
| Qwen2.5-32B（2+2，关 offload） | **6.2 s** | 14.2 s | 2.3× |
| Qwen2.5-72B（4+4，gen TP8） | **12.0–13.0 s** | 28.5–29.1 s | 2.3× |
| Qwen3-30B-A3B（ep8，1+1） | **7.1 s** | 32.2 s | **4.5×** |
| Qwen3-235B-A22B（ep8 × fsdp8，8+2） | **11.4–14.9 s** | 246–266 s | **~21×** |

**三条结论**：① delta 同步时间从 32B 到 235B 基本持平；② 全量广播随参数量线性增长；③ 稀疏率 dense 约 1~3%、**235B MoE 早期仅 0.02~0.05%**——MoE 越稀疏收益越大。

### 4. 其他家的方案

**NeMo-RL 的 transport 矩阵**：

| 部署形态 | Transport |
|---|---|
| **colocated** | **CUDA IPC**（vLLM = ZMQ+IPC；SGLang = Ray CUDA IPC） |
| non-colocated | NCCL broadcast / **`nccl_reshard`（>100B 大模型最优）** / sparse delta over ZMQ / sparse delta over S3 / **NIXL（UCX-RDMA）** |

**NIXL 的隐性显存成本**：`update_weights_bucket_memory_ratio=0.05 × 全卡显存`，且要两个 buffer → 80 GiB 卡上**默认每 engine 吃掉 8 GiB**。这类"同步方案的显存代价"很容易被忽略。

**共卡的"零搬运"特例**：verl 的 `naive` 后端 = `ColocatedCheckpointEngine`，`send_weights` **只是把 generator 存下来**、`receive_weights` **直接 yield**——**同卡时权重根本不搬运**。这正是分卡 trainer 要 `assert backend != "naive"` 的原因。

**一个必须知道的坑**：vLLM 的 IPC 传输文档要求——**FSDP/TP/PP/EP 下每个 trainer rank 都必须建 engine 并调 `send_weights()`**，因为"物化参数本身就是 collective，跳过某个 rank 会死锁"。这是权重同步最容易写出 hang 的地方。

> 面试一句话总结：**分卡后每步都要把权重送到 rollout 引擎，全量广播随模型规模线性增长（235B 实测 246~266 s/步，uniagent 项目里同步占单步 48.1s 中的约 27s）；核心洞察是 RL 更新极度稀疏（>99% 的 BF16 字节每步不变，dense 1~3%、235B MoE 早期仅 0.02~0.05%），所以用 delta 同步——把差分下沉到 all-gather 之前、每个 rank 只 pin 自己的 shard、只 gather 变化的 (position,value)、rank 0 只拼接分桶广播、接收端同 GPU IPC 原地更新并校验 checksum；实测 7B 1.3× 到 235B ~21×；NeMo-RL 另有 CUDA IPC/NCCL/nccl_reshard/sparse/NIXL 矩阵，NIXL 默认吃掉 2×5% 全卡显存。**

---

## 7. 方法六：Staleness 控制 —— 异步的刹车

### 1. 为什么必须刹车

异步省的是空泡，换来的代价是 **off-policy 程度变大**。Trust Region Masking 论文把 off-policy 分成三类来源：**backend discrepancies / MoE routing discontinuities / distributed staleness**——**第三类正是异步引入的**。

### 2. 量化：落后多少版本会出问题

设每次权重更新让 token 级 log-ratio 的标准差增加 $\sigma_0=0.01$，落后 $k$ 步的样本其 $\log r$ 标准差约 $\sigma_0\sqrt{k}$：

| 落后 $k$ | $\sigma(\log r)$ | 序列级（$T=2000$）$\rho=e^{\sigma\sqrt{T}}$ |
|---|---|---|
| 1 | 0.010 | $e^{0.45}=1.56$ |
| 4 | 0.020 | $e^{0.89}=2.44$ |
| 16 | 0.040 | $e^{1.79}=\mathbf{6.0}$ |

**落后 16 个版本时序列级 IS 权重已接近 6**，逼近典型阈值上界（2~10）。**这就是为什么 AReaL 定 2~8、verl 建议 <1。**

### 3. 三件配套的事

1. **算法修正**：decoupled PPO（$w=\pi_{\text{old}}/\pi_{\text{rollout}}$）+ IS/RS 掩码（见 `Training-Inference-Mismatch.md`）；
2. **超限处理必须显式选择**：verl/uniagent 提供 **`drop`**（丢超龄样本，浪费算力但训练干净）和 **`wait`**（等它，不浪费但拖慢）；Polar 的 demo 选 `drop`；uniagent 的 replay buffer 还有第三种——**按 `prompt_global_steps` 升序取最老的**（注释 "to reduce staleness"）；
3. **可观测**：uniagent 用 `trajectory_staleness = (global_steps − 1) − max_global_steps` 度量，并在轨迹上注入 `global_steps` / `min_global_steps` / `max_global_steps` 三个 tag 供采样器判断。

### 4. 一个理论警示

Trust Region Masking（arXiv 2512.23075）：经典 trust-region bound 对序列长度是 $O(T^2)$，取 $T=4096$、token 级 KL 上界 $10^{-4}$，**bound 可达 1677——reward 上限才 1**。即**理论 bound 完全 vacuous**。**含义**：不能指望 KL 惩罚自动约束住 staleness，**必须用显式上限 + 硬 mask（RS）**。

> 面试一句话总结：**异步必然引入 distributed staleness（off-policy 第三类来源）；量化上落后 16 个版本时序列级 IS 权重就到 6、逼近阈值，所以 AReaL 用 max_head_offpolicyness（2~8）、verl 用 staleness_threshold（<1）设上限；超限必须显式选 drop / wait / 取最老；理论 trust-region bound 在 T=4096 时可达 1677（reward 上限才 1，完全 vacuous），所以必须靠硬 mask 而不是靠 KL 惩罚。**

---

## 8. 方法七：长尾治理与资源配比

### 1. 长尾是空泡的源头

verl 文档定性得很清楚：

> *"**in the colocate case, using more resources for rollout cannot solve the idleness caused by long-tail samples.**"*

**给 rollout 加卡也治不好长尾**——瓶颈是"最慢那一条"，加卡只让快的更快。这是共卡的根本局限，也是必须走向分卡+异步的第一性理由。

### 2. 三个具体手段

**(a) 把长尾摘出关键路径**（分卡 + 异步 + partial rollout）：见第 3、5 节。效果量级：一个"90% 样本 60 单位、10% 长尾 400 单位"的 step，同步要 400+60，异步后等效降到接近 60。

**(b) prefix KV 下沉共享存储**（verl + Mooncake）：agentic RL 的特点是**所有轮次共享极长前缀**（system prompt + 工具历史 + `rollout.n` 条采样的公共前缀）。verl 把这些 prefix KV 块 offload 到 Mooncake：

> *"…so long shared prefixes get deduplicated across requests and rollout replicas. This also helps **long-tail load balancing**: when work migrates to idle rollout replicas, shared prefix KV reduces the **re-prefill cost**."*

**两个收益**：① 同一 prompt 的 $n$ 条采样共享前缀 KV；② **长尾任务迁移到空闲 replica 时不必重新 prefill 整个历史**。

**量化**：前缀 $P=8000$ token、32 个 prompt 各采 $n=8$ 条 → 不共享时 $32\times8\times8000=2.048\times10^6$ token；共享后 $32\times8000=2.56\times10^5$——**降到 1/8**。按 10k token/s 估算省约 179 秒 prefill。

**(c) 资源配比**：瓶颈在哪一侧就多分卡给哪一侧。**业界实际配比**：

| 系统 | train : rollout | 备注 |
|---|---|---|
| Polar + Slime 官方 example | **4 : 4** | 8 卡单机 |
| verl `one_step_off` | **6 : 2** | 实验配置另有 12 : 4 |
| SkyRL one-step-off | **4 : 4** | — |
| uniagent-lighting `separate_async` | **1 : 1** | 跨机 |

**注意 verl 自己就有 6:2 和 12:4 两种**——配比是**跟着"哪一侧更慢"调的**，不是常数。而 NeMo-RL 的 `rollout ≤ policy` 是 **NIXL paired 拓扑的实现约束**，不是性能最优——**这两类原因要分开说**。

> 面试一句话总结：**长尾是空泡的源头，而"给 rollout 加卡治不好长尾"（瓶颈是最慢那条）——这是走向分卡+异步的第一性理由；三个手段：① 分卡+异步+partial rollout 把长尾摘出关键路径；② prefix KV 下沉共享存储（rollout.n=8 时 prefill 量降到 1/8，且让长尾迁移到空闲 replica 时不必重新 prefill 历史）；③ 资源配比按两侧实际耗时定（实测从 4:4 到 12:4 都有），注意实现约束（如 NeMo-RL 的 paired 拓扑）与性能最优要分开说。**

---

## 9. 附：KV 复用与失效边界（一个容易踩的正确性坑）

### 1. 问题

填空泡的一个自然想法是"**复用上一轮的 KV cache**，省掉重复 prefill"。但 KV 是**上一个策略**算出来的——复用等于凭空制造 off-policy。

### 2. 各家做法差异很大（本身就是好素材）

| 系统 | 做法 |
|---|---|
| vLLM | `finish_weight_update(weight_version=...)` |
| SGLang | `record_weight_version_after_update` + 每次更新后 `flush_cache_after_weight_update` |
| NeMo-RL | `recompute_kv_cache_after_weight_updates` |
| **verl** | HYBRID 唤醒后 `reset_prefix_cache`；且**每次权重更新边界硬清 local + Mooncake 两侧** |
| **SkyRL** | `clear_kv_cache_on_weight_sync` **默认 false**——**故意复用 stale KV 省重算**，改用 `use_cache_salt` 按版本隔离 prefix cache |

**这是非黑即白的取舍**：**要么硬清**（干净但丢复用）、**要么用 salt 做版本隔离**（保留复用但实现复杂）。

**verl 的硬要求**：`rollout_kv_offload.md` 明确要求 **vLLM ≥ 0.22**，否则旧的 Mooncake master 里可能残留 stale KV。

### 3. 与共卡的关系

共卡下这条尤其关键：sleep/wake 会**丢弃** KV（level 2），唤醒后如果从共享存储里捞回旧 KV 就更危险。**"省下的 prefill"和"策略正确性"在这里正面冲突**，必须显式决定。

> 面试一句话总结：**"复用 KV 省 prefill"和"策略正确性"正面冲突——KV 是上一个策略算的，复用等于凭空制造 off-policy；各家分两派：硬清（vLLM/SGLang/NeMo-RL/verl，verl 还要求 vLLM≥0.22 以免 Mooncake 残留 stale KV）vs 用 salt 做版本隔离保留复用（SkyRL 默认不清）——能说出这两派的取舍，比只说"要清 KV"完整得多。**

---

## 10. 架构串讲：两种形态 + 选型

```
【形态 A：Colocate / Hybrid —— 填"同步空泡"，代价是显存与切换】
┌──────────────── 同一批 GPU ────────────────┐
│ ① wake_up(weights) → 更新权重 → wake_up(kv) │
│ ② rollouter 生成（吃 KV cache 显存）         │
│ ③ sleep(level=2)（换出权重/KV）              │
│ ④ 训练前向反向（optimizer state offload）    │
│ ⑤ 回到 ①                                    │
└─────────────────────────────────────────────┘
空泡：长尾直接阻塞 step；切换本身也是空泡
适用：小模型/单机/rollout 与 train 时长接近/无明显长尾

【形态 B：Disaggregate + Async —— 填"长尾空泡"，代价是通信与 off-policy】
┌──── Rollout 池 ────┐        ┌──── Train 池 ────┐
│ Rollouter 逐样本生成 │        │ 攒够 batch 就训    │
│  ├─ partial rollout │        │  ts 轮后触发同步   │
│  └─ 长尾挂起/迁移    │        │                  │
└─────────┬───────────┘        └────────┬─────────┘
          │  MessageQueue（逐样本 streaming）│
          └──────────────►────────────────┘
                         ▲
                         │ ParameterSynchronizer
                         │  delta sync（稀疏 1~3%）+ 同 GPU IPC apply
                         │  + staleness 上限 + 硬重置 KV
                         └────────────────────────
空泡：长尾被异步吸收
新增成本：权重同步 + 数据搬运 + staleness（需 IS/RS 兜底）
适用：长尾严重（SWE/agentic 多轮）、大模型（权重同步是瓶颈）
```

### 选型决策表

| 场景特征 | 建议 | 理由 |
|---|---|---|
| 单机、模型小（≤7B）、两侧时长接近、无明显长尾 | **Colocate** | 无通信成本，且显存够两态切换 |
| 长尾严重、rollout 远长于 train | **Disaggregate + 异步** | 长尾摘出关键路径，收益远大于同步成本 |
| 模型巨大（≥70B / MoE）、权重同步成瓶颈 | **Disaggregate + delta sync** | 全量广播 235B 要 250 s/步，delta 后 ~13 s |
| 需要任意 harness（二进制/CLI）、trainer 异构 | **Polar 型 rollout-as-a-service** | 网关代理 + 异步 service 边界，trainer 无关 |
| 稳定性要求极高、不能容忍 off-policy | **Colocate + 严格同步**（或 AReaL `max_head_offpolicyness=0`） | 代价是约 2× 的时间 |

**关于 Polar 的三点说准**（避免被追问）：① 它是 **rollout 框架**不是训练框架（"a rollout framework for scalable asynchronous RL over arbitrary agent harnesses"）；② **它自己不做权重同步**，只提供 `POST /admin/inference/pause|resume` 挂钩，且全仓无调用方（demo 里是 Slime 走 NCCL）；③ trainer 桥接**只有 Slime**，roadmap 里 NeMo-RL/verl **未勾选**。

### 一句话总纲

> **空泡有三种（长尾、切换、同步），填法有七种，代价记在三笔账上（显存、通信、off-policy）。** 判据是：**长尾是否显著、模型是否大到权重同步成为瓶颈、算法层能否消化 off-policy。** 演进路径通常是：**colocate（先解决显存）→ one-step-off（最小改动的重叠）→ fully async + delta sync + staleness 控制（真正填掉长尾空泡）**。

---

## 附：高频追问速答

**Q1：什么是"空泡"？**
GPU 空转的时间片，三类：**长尾空泡**（等最慢轨迹）、**切换空泡**（推理↔训练态切换）、**同步空泡**（一个 step 内两种负载不重叠）。一个 10% 长尾到 400 单位的 step，有效利用率只有约 26%（实测 agentic 场景 rollout 利用率 20.4%）。

**Q2：为什么"给 rollout 加卡"治不好长尾？**
瓶颈是**最慢那一条**，加卡只让快的更快。verl 原话：*"in the colocate case, using more resources for rollout cannot solve the idleness caused by long-tail samples."*

**Q3：共卡和分卡的真判据是什么？**
**共卡 ⇔ 必须有 sleep/wake**；**分卡 ⇔ 权重同步升级为一等系统问题**（`free_cache_engine: False` + 真 backend）。分卡 rollout 永不暂停，不可能把显存让给同卡组件。

**Q4：vLLM 的 sleep level 1 和 2 区别？**
Level 1 = 权重 offload 到 CPU + 丢弃 KV；Level 2 = 权重和 KV 都丢弃（RLHF 权重更新用这个）。源码里 level 2 就是 `offload_tags=()`（空 tuple）。

**Q5：`staleness_threshold` 和 `max_head_offpolicyness` 区别？**
前者是 verl 的**陈旧样本比例**（建议 <1），后者是 AReaL 的**落后版本数**（典型 2~8）。语义不同，别混。

**Q6：partial rollout 的代价？**
一条轨迹跨多个策略版本 → 本身变成 off-policy 混合体。且它**只在 `staleness_threshold > 0` 时生效**。

**Q7：权重同步为什么要做 delta？**
RL 更新极度稀疏（>99% 的 BF16 字节每步不变）。全量广播 235B 要 246~266 s/步，delta 后 11~15 s（~21×）。

**Q8：异步最快的收益来自哪？**
不是"算得更快"，而是**两种负载重叠 + 长尾被吸收**。verl 官方的话最准：*"the time for rollout and train may be longer than before (because fewer resources are used), but the overlap in their time consumption reduces the end-to-end time consumption."*

**Q9：怎么公平比较共卡和分卡？**
按公式对齐同步频率：`trigger_parameter_sync_step = train_batch_size / (require_batches × ppo_mini_batch_size)`，保证处理同样多的样本再同步一次。

**Q10：为什么"异步 + 硬清 KV"看起来自相矛盾？**
不矛盾——异步省的是**空泡**，KV 清的是**正确性**。省掉的 prefill 换的是 off-policy 程度；要么硬清（verl/vLLM/SGLang），要么用 `use_cache_salt` 做版本隔离（SkyRL 默认不清）。
