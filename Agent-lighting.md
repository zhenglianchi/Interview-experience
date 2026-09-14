# Agent Lightning 完全指南

> **Microsoft 的 agent 强化学习训推框架：Algorithm × Runner × TrajStore（LightningStore）三件套 + VERL 集成。**

Agent Lightning（`agentlightning`，简称 agl）是微软研究院开源的"**用强化学习训练任意 agent**"的框架，核心卖点是**对 agent 几乎零代码改动**（任何 agent 框架甚至纯 Python 都能接入）。它的架构围绕三个核心组件组织：**Algorithm（算法，系统的"大脑"）、Runner（执行器，系统的"工人"）、LightningStore（轨迹存储，系统的"数据库+消息队列"）**——三者构成一个持续循环：Algorithm 派发任务 → Runner 执行 agent 并流式回传轨迹（spans）→ Algorithm 从 store 取回轨迹、转成训练样本、更新模型。**我们实际采用三机分离部署：store、algo、agent（runner）分别部署在三台不同的机器，全部通过 store 的 HTTP 地址连接**——本指南全文都以此部署形态为语境（三机分离下，algo 机与 agent 机对 store 的一切访问都是 HTTP；只有 Runner 进程内部的 Agent 调用和算法内部的 Adapter 转换是实例直接调用）。本指南先逐个讲清这三个部分各自的职责与实现，再讲它们如何组成整体架构，最后重点讲与 **verl**（Volcengine 的开源 RL 训练框架）的集成关系；**第三部分（第 6~11 节）专门钻进轨迹本身**——一条轨迹是在哪一行代码被"抓住"的（三条记录通路 + 插桩细节）、`Span` 模型长什么样、怎么变成 HTTP 请求落到 store（OTLP protobuf 批量 vs 逐条 JSON，含 proxy 的子树缓冲专线）、到了 store 之后怎么从一堆扁平 span **重建成一棵树**（`TraceTree.from_spans` + `repair_hierarchy`）、怎么变成训练用的 triplet，以及分布式下的**有序性/幂等/静默失败**语义；**第四部分（第 12~18 节）讲共卡（colocate）下"推理 ↔ 训练"的全流程**——三种 `RolloutMode` 与"为什么共卡是解决空泡的 baseline"、参数到底在哪（`engine.to()` + `BaseEngineCtx._context_switch` + vLLM sleep level 1/2）、wake/sleep 的完整调用链与硬顺序、权重同步的两条路（`naive` 同卡直传 vs `checkpoint_engine` 跨卡 NCCL 多播 + bucket 切块）、项目 TQ 方案（`tqbridge` + `ReplayBuffer` + `KVBatchMeta`）的一次完整 `_train_step`、训练每一步的内部细节（micro-batch → forward → loss → backward → clip → step → lr），以及项目实测的空泡账与修复方案。

> 说明：本文基于官方仓库 `agent-lightning-official` 的 **v0.3.0** 分支（tag `v0.3.0`，commit 3b5d7338）讲解。**我们实际训练使用的是 v1 执行模式**（`AgentModeDaemon` 的 `mode` 参数默认即 `"v1"`）——v1 模式下任务的派发、轨迹的回收、资源的传递全部通过 LightningStore 完成（Runner 与算法完全解耦）；v0 模式（自起 Flask server）仅作为历史兼容保留，本文以 v1 为主线。**部署形态统一为三机分离（见第 4 点详述）**：store 单独一台机器跑 `LightningStoreServer`（`agl store --port 4747`），algo 机器和 agent 机器都作为 `LightningStoreClient` 通过 HTTP 连它。

---

# 一、三个核心组件

## 1. Algorithm（algo）：系统的"大脑"

### 1. 现有问题：为什么需要 Algorithm 这个抽象

Agent Lightning 要解决的第一个问题是：**"训练 agent"这件事本身需要一套统一的决策逻辑**——选哪些任务给 agent 做、从 agent 产生的数据里学到什么、如何更新被训练的资源（模型权重、prompt 模板等）。不同训练方法（RL / prompt 优化 / SFT）的"学习策略"千差万别，但如果每个方法都自己管理任务派发、数据回收、资源更新，代码会高度重复且无法复用。因此 Agent Lightning 定义了一个统一的 **`Algorithm`** 抽象：它是"**用来训练 agent 的策略（strategy）或调优器（tuner）**"（源码 docstring 原话："Algorithm is the strategy, or tuner to train the agent"）。

Algorithm 的核心职责有三条：**第一，决定跑什么任务**——把数据集（train/val）中的样本作为 rollout 任务 enqueue 进 store；**第二，从结果中学习**——等 rollout 完成后从 store 查询产生的 spans，用 Adapter 转成训练样本；**第三，更新资源**——根据学习信号更新模型或 prompt。这三条职责对应 RL 训练中"数据生成 → 学习 → 更新"的完整闭环，而 Algorithm 抽象把"学什么、怎么学"与"任务怎么跑、轨迹怎么存"解耦——前者由算法实现，后者交给 Runner 和 Store。

### 2. 方法论：Algorithm 是怎么实现的

**Algorithm 的完整接口**（`algorithm/base.py`，继承方必须实现的契约）：

| 方法 | 作用 |
|---|---|
| `run(train_dataset, val_dataset)` | **算法主入口**（唯一必须实现）：执行"派任务 → 收轨迹 → 学 → 更新资源"的完整逻辑；子类实现，基类抛 NotImplementedError |
| `is_async()` | 返回 `run` 是否为协程函数（判断算法同步/异步执行） |
| `set_trainer()` / `get_trainer()` | 注入/获取 Trainer（弱引用） |
| `set_llm_proxy()` / `get_llm_proxy()` | 注入/获取 LLMProxy（弱引用，可空） |
| `set_adapter()` / `get_adapter()` | 注入/获取 Adapter（弱引用，用于轨迹→训练样本） |
| `set_store()` / `get_store()` | 注入/获取 LightningStore（**强引用**，注释明确"其副本要在算法生命周期内维持"） |
| `set_initial_resources()` / `get_initial_resources()` | 注入/获取初始资源（NamedResources） |
| `__call__(*args, **kwargs)` | 语法糖：直接 `algorithm(...)` 等价于 `algorithm.run(...)` |
| `get_client()` | 获取与算法通信的 client（已废弃，未来移除） |

**核心代码**（`agentlightning/algorithm/base.py`，Algorithm 基类的骨架）：

```python
class Algorithm:
    """Algorithm is the strategy, or tuner to train the agent."""
    # 依赖通过注入持有：Trainer/LLMProxy/Adapter 用弱引用（防循环引用），
    # Store 用强引用（注释："its copy is meant to be maintained throughout
    # the algorithm's lifecycle"）
    _trainer_ref: weakref.ReferenceType[Trainer] | None = None
    _llm_proxy_ref: weakref.ReferenceType["LLMProxy"] | None = None
    _store: LightningStore | None = None
    _adapter_ref: weakref.ReferenceType[TraceAdapter[Any]] | None = None

    def is_async(self) -> bool:
        """Return True if the algorithm is asynchronous."""
        return inspect.iscoroutinefunction(self.run)   # run 是协程 → 异步算法

    def set_store(self, store: LightningStore) -> None:
        self._store = store    # 强引用：副本要贯穿算法生命周期

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)    # algorithm(...) == algorithm.run(...)

    def run(self, train_dataset=None, val_dataset=None):
        """Subclasses should implement this method to implement the algorithm."""
        raise NotImplementedError("Subclasses must implement run().")
```

**这段代码关键在哪**：① 依赖注入方式（弱引用 vs 强引用）决定了组件生命周期——Store 必须活到算法结束所以强引用，Trainer/LLMProxy/Adapter 用弱引用避免循环引用；② `is_async()` 用 `inspect.iscoroutinefunction` 反射判断，是"同步/异步算法统一接口"的机制；③ `__call__` 让 Algorithm 可以像函数一样被调用（`trainer.fit` 内部就是调 `algorithm(...)`）。

**VERL 的使用方式**（`algorithm/verl/interface.py`，一个 RL 算法的真实调用示例）：

```python
algorithm = VERL(config={
    "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
    "data": {"train_batch_size": 32, "max_prompt_length": 4096,
             "max_response_length": 2048},
    "actor_rollout_ref": {
        "rollout": {"tensor_model_parallel_size": 1, "n": 4,
                    "multi_turn": {"format": "hermes"}, "name": "vllm",
                    "gpu_memory_utilization": 0.6},
        "actor": {"ppo_mini_batch_size": 32, "optim": {"lr": 1e-6},
                  "fsdp_config": {"param_offload": True,
                                  "optimizer_offload": True}},
        "ref": {"fsdp_config": {"param_offload": True}},
        "model": {"path": "Qwen/Qwen2.5-1.5B-Instruct",
                  "enable_gradient_checkpointing": True},
    },
    "trainer": {"n_gpus_per_node": 1, "logger": ["console", "wandb"],
                "project_name": "AgentLightning", "save_freq": 64,
                "total_epochs": 2},
})
trainer.fit(algorithm, train_dataset=my_train_dataset)
```

**这段代码关键在哪**：VERL 算法本质是"把 verl 的 CLI 配置（Hydra overrides）包装成 dict 传给 Agent Lightning"——`adv_estimator=grpo`、`n=4`（每个 prompt 采 4 条响应）、LoRA/FSDP offload 配置都和 verl 原生一致，说明 Agent Lightning 的 VERL 是 verl 的薄包装（详见第 5 点）。

**执行框架**：Trainer 创建 Algorithm 并注入 Store/Adapter/LLMProxy → 调用 `trainer.fit(algorithm, dataset)` → fit 内部执行 `algorithm.run()` → run 内部用 store 派发任务、等结果、学习、更新资源。**三机分离下**：Algorithm 运行在 **algo 机器**，注入的 store 是指向 store 机器的 `LightningStoreClient`（HTTP），Adapter 是本地实例（实例调用）。

Agent Lightning 的 `algorithm/` 目录下有两个开箱即用的算法族：

- **`FastAlgorithm`**（`algorithm/fast.py`）：面向开发者工作流的轻量算法基类，优先"短反馈回路"——让 agent 开发者能快速跑小规模实验，不用等长训练任务。`Baseline` 就是它的参考实现（`_span_to_string` 把 span 格式化成可读日志）：把整个数据集流式灌进 rollout 队列，等每个 rollout 完成再继续——用于基线/调试。
- **`VERL`**（`algorithm/verl/interface.py`）：**真正的 RL 训练算法**，把训练委托给 verl 框架的 PPO runner（详见第 5 点）。它接收一个 dict 配置（镜像 verl CLI 的 overrides），通过 Hydra 与 verl 打包的默认配置合并后启动训练。此外还有 **`APO`**（`algorithm/apo/apo.py`，Automatic Prompt Optimization，自动提示词优化）——不更新模型权重、而是优化 prompt 的算法族。

算法侧还有两个配套组件：**`Adapter`**（`adapter/`）把 store 里的原始 spans 转成算法能消费的结构化数据（`TracerTraceToTriplet` 把 OpenTelemetry spans 转成 (prompt, response, reward) 三元组，这是 RL 微调的基本数据单元）；**`LLMProxy`**（`llm_proxy.py`）作为 agent 与模型之间的桥——所有 LLM 调用走它，它负责统一后端、自动插桩（把 LLM 调用也作为 span 记入 store）、以及**动态切换模型**（算法更新权重后只需换 proxy 的后端，agent 代码不用改）。

### 3. 具体数值样例

以 `VERL` 算法为例，逐步演算一次完整训练循环中 Algorithm 的职责（**三机分离语境：本段描述的是 algo 机器上的行为**）。假设要训练一个 SQL agent，训练集 1000 条、batch 32：

```text
Trainer.fit(VERL(config), train_dataset) 触发 VERL.run()（在 algo 机器上执行）
第 1 步：VERL 从 config 读取模型路径（如 Qwen2.5-1.5B），用 vLLM 启动一个
        chat completion 端点，并把它注册到 LLMProxy（作为 "main_llm" 资源，
        资源 key 固定为 main_llm）。
第 2 步：把 32 条训练样本作为 rollout 任务 enqueue 进 LightningStore
        （HTTP 到 store 机器；每条是一个 rollout，等待 Runner 领取）。
第 3 步：等待 Runner 执行完这 32 个 rollout —— 期间 agent 机器上的 Runner
        通过 LLMProxy（HTTP）调用 vLLM 生成、把轨迹 spans 写回 store
        （HTTP 到 store 机器）。
第 4 步：rollout 完成后，Algorithm 从 store 查询 spans（HTTP 到 store 机器），
        用 TracerTraceToTriplet（实例直接调用，纯本地）把每条轨迹转成
        多个 (prompt, response, reward) triplet —— 一条多轮轨迹（如 agent
        调了 3 次工具）会拆成 3 个 triplet，最终 reward 用 "identical
        assignment"（同值分配）赋给所有 triplet（源自 arXiv:2508.03680）。
第 5 步：把 triplets 转成 verl 的 DataProto（input_ids / position_ids /
        attention_mask / token_level_scores 等字段），交给 verl 的 PPO 训练循环，
        更新模型权重。
第 6 步：vLLM 端点加载新权重 → 下一轮 batch 的 agent 用新模型 → 循环。
```

**三机分离下 Algorithm 的边界**：Algorithm 只运行在 **algo 机器**上，它通过 `LightningStoreClient`（HTTP）访问 store 机器上的任务队列和轨迹；它不直接接触 agent——agent 在 agent 机器上由 Runner 驱动，两者唯一的交集就是 store（HTTP）。

**关键理解**：Algorithm 是"编排者"而非"执行者"——它不自己跑 agent，也不自己算注意力，它负责"派任务、收轨迹、转样本、更新模型"；agent 的执行和轨迹的产生完全由 Runner 承担，二者的解耦正是 Agent Lightning 能任意扩展 Runner 数量的原因。

> **面试一句话总结**：Algorithm 是 Agent Lightning 的"大脑"，通过统一的 `run()` 接口承担"派发任务 → 从 store 取回轨迹 → 用 Adapter 转成训练样本 → 更新资源（模型/prompt）"的完整学习闭环；`VERL` 算法把训练委托给 verl 的 PPO runner，`APO` 则做 prompt 优化，二者共享同一套 Algorithm 抽象。

---

## 2. Runner（执行器）：系统的"工人"

### 1. 现有问题：为什么需要 Runner 这个抽象

Agent Lightning 要解决的第二个问题是：**agent 到底由谁来跑、怎么跑、轨迹怎么被记录下来？** 如果算法直接调用 agent 函数，那么"任务调度、重试、轨迹采集、心跳上报、并行执行"这些工程问题会全部耦合进算法里，而且 agent 无法独立于算法横向扩展。Runner 的定位是"**长期运行的 agent 执行器**"（源码 docstring 原话："Abstract base class for long-running agent executors"）：它负责**从 store 领取任务（dequeue rollout）、协调 LitAgent 执行任务、把执行过程中产生的 spans 流式写回 store**。Runner 之于 Agent Lightning，相当于"执行引擎"之于训练框架——算法只负责决策，Runner 负责干活。

Runner 要解决的工程问题很多：**任务领取**（并发多 runner 时如何互不重复地拿任务）、**轨迹采集**（agent 的每次 LLM 调用、工具调用、中间奖励都要被记录）、**失败重试**（一次 rollout 失败/超时要能重试，对应 store 里的 Attempt 概念）、**存活上报**（runner 要持续心跳，让 store 的 watchdog 知道它还活着）、**并行扩展**（多 runner 同时跑，吞吐随 runner 数扩展）。这些如果都由算法实现，算法会变得极其臃肿——Runner 抽象把这些全部收拢。

### 2. 方法论：Runner 是怎么实现的

**Runner 的完整接口**（`runner/base.py`，继承方必须实现的契约）：

| 方法 | 作用 |
|---|---|
| `init(agent, **kwargs)` | 一次性初始化（所有 worker 共享一次，如创建 Tracer/Hooks），不是每 worker 一次 |
| `init_worker(worker_id, store, **kwargs)` | **每个 worker 各调一次**的 worker 本地初始化（注入 store） |
| `iter(event)` | **批量执行主循环**：持续轮询 store.dequeue_rollout 领取任务并执行，直到 event 置位 / 达到 max_rollouts / 无任务 |
| `step(input, resources, mode, event)` | **单任务执行**：绕过任务队列直接执行一个任务（在线/持续学习用），异常向上抛 |
| `teardown()` | 释放 init 获取的资源 |
| `teardown_worker(worker_id)` | 释放单 worker 资源 |
| `run_context(agent, store, hooks, worker_id)` | 上下文管理器：init + init_worker + yield + teardown（调试用） |

**核心代码**（`runner/base.py` 基类接口 + `runner/agent.py` 的 `iter()` 主循环）：

```python
# runner/base.py —— Runner 的"批量执行"与"单步执行"两个入口
class Runner(ParallelWorkerBase, Generic[T_task]):
    """Abstract base class for long-running agent executors."""
    def init(self, agent: LitAgent[T_task], **kwargs): ...      # 一次性初始化
    def init_worker(self, worker_id: int, store: LightningStore, **kwargs): ...
        # 每个 worker 各调一次，注入 store

    def run(self, *args, **kwargs):
        raise RuntimeError("The behavior of run() of Runner is undefined. "
                           "Use iter() or step() instead.")   # 旧入口已废弃

    async def iter(self, *, event=None):
        """Run the runner, continuously iterating over tasks in the store."""
        raise NotImplementedError()
    async def step(self, input, *, resources=None, mode=None, event=None) -> Rollout:
        """Execute a single task directly, bypassing the task queue."""
        raise NotImplementedError()
```

```python
# runner/agent.py —— LitAgentRunner.iter()：批量执行主循环（真实实现）
async def iter(self, *, event: Optional[ExecutionEvent] = None) -> None:
    store = self.get_store()
    stop_heartbeat = self._start_heartbeat_loop(store)   # 心跳线程先启动
    try:
        while not (event is not None and event.is_set()):
            next_rollout = await store.dequeue_rollout(worker_id=self.get_worker_id())
            if next_rollout is None:                      # 队列空 → 等 poll_interval
                await self._sleep_until_next_poll(event)
            else:
                await self._step_impl(next_rollout)       # 执行单个 rollout
    finally:
        await stop_heartbeat()
```

**这段代码关键在哪**：① `run()` 直接抛 RuntimeError 引导用 `iter()`/`step()`——**批量模式（训练）用 iter 轮询 store，单步模式（在线学习）用 step 绕过队列**；② `iter()` 的主循环就是"dequeue_rollout → 空则等 → 非空执行"，配合心跳线程，这就是 Runner 的完整生命周期。

**`_step_impl()` 的单 rollout 执行核心**（`runner/agent.py`）：

```python
async def _step_impl(self, next_rollout: AttemptedRollout, raise_on_exception=False):
    store = self.get_store(); agent = self.get_agent()
    rollout_id = next_rollout.rollout_id

    resources_update = await store.get_latest_resources()   # ① 取资源
    ...
    async with self._tracer.trace_context(                  # ② 进入 trace context
        name=rollout_id, rollout_id=rollout_id, attempt_id=next_rollout.attempt.attempt_id
    ):
        rollout_method = (agent.training_rollout_async if next_rollout.mode == "train"
                          else agent.validation_rollout_async)   # ③ 实例调用 agent
        ...
        result = await rollout_method(input, resources=..., rollout=next_rollout)
        # ④ agent 执行期间的 spans 由 Tracer 插桩自动捕获并写回 store
    # ⑤ 收尾：补 span + update_attempt(succeeded/failed)
```

**这段代码关键在哪**：`_step_impl` 就是 Runner 五步流程的代码体现——①取资源（`get_latest_resources`）→ ②进入 trace_context（注入 rollout_id/attempt_id）→ ③**实例直接调用** `agent.training_rollout_async`（注意是 `_async` 版本，说明 Runner 全程异步）→ ④Tracer 插桩自动捕获 spans → ⑤收尾；其中 `trace_context` 的 `attempt_id=next_rollout.attempt.attempt_id` 正是"一个 rollout 可多次 attempt、每次 attempt 的轨迹独立"的代码落点。

**`LitAgentRunner`**（`runner/agent.py`）是具体实现。Runner 的核心成员包括：**LitAgent**（要执行的 agent）、**Tracer**（轨迹采集器）、**Hooks**（生命周期回调）、以及通过注入持有的 **Store**（三机分离下是 `LightningStoreClient`，指向 store 机器）。**Runner 运行在 agent 机器上**，`iter()` 是主要入口——它的主循环逐步操作如下：

**第 1 步（领取任务）**：`iter()` 循环里轮询 `store.dequeue_rollout(worker_id)` 领取一个任务（**三机分离下是 HTTP 请求到 store 机器**，经 `LightningStoreClient` → `LightningStoreServer`），store 会为该 rollout 自动创建一次 **Attempt**（一次执行尝试），并把 worker 标记为 busy。

**第 2 步（获取资源）**：Runner 通过 `store.get_latest_resources()`（HTTP 到 store 机器）拿到当前资源（如 LLMProxy 的 URL 模板——在 VERL 场景中是带 rollout_id/attempt_id 占位符的 URL，用于让 proxy 精确记录这次尝试的流量）。

**第 3 步（进入 trace context 并执行）**：Runner 进入 `trace_context`，把 rollout_id / attempt_id 注入 Tracer，然后**实例直接调用** agent 的 `training_rollout` / `validation_rollout`（`agent.training_rollout(...)`——Agent 与 Runner 在同一进程、同一机器，永远不走网络）。**关键点：spans 不是 agent 主动发的，而是 Tracer 通过插桩（instrumentation）自动捕获的**——Tracer hook 住 agent 用到的关键方法（如 `openai.chat.completion`、`agent.execute`），每次调用完成就生成一个 OpenTelemetry span，通过 `store.add_otel_span(rollout_id, attempt_id, span)` 流式写回 store（**HTTP 到 store 机器**）。agent 还可以主动 emit 中间奖励（`emit_reward`）。

**第 4 步（收尾）**：agent 返回最终 reward 和额外 spans，Runner 再补一个 span、调用 `store.update_attempt(status)`（HTTP 到 store 机器）把 attempt 标记为 succeeded/failed，完成任务。

**第 5 步（心跳）**：Runner 的 producer/consumer 双线程持续上报系统快照（GPU 使用等）到 store 的 worker 记录（HTTP，`update_worker`），防止被 watchdog 判为 unresponsive。

Runner 还支持 **Hooks**（`on_rollout_start` / `on_trace_start` / `on_trace_end` / `on_rollout_end` 四个时机），用于自定义日志、资源准备/清理等。**并行扩展**通过 `Trainer(n_runners=N)` 配置——**三机分离下 agent 机器可以有多台、每台多个 Runner 进程**，每个 Runner 独立从 store 抢任务（HTTP），天然负载均衡。

### 3. 具体数值样例

假设训练一个 agent，**agent 机器**上配置 `n_runners=4`（可分布在多台 agent 机器），store 机器队列里一次 enqueue 了 32 个 rollout。逐步演算 Runner 的行为（**三机分离语境：Runner 的一切 store 访问都是 HTTP 到 store 机器**）：

```text
t0: 4 个 Runner 同时启动，都开始轮询 store.dequeue_rollout()（HTTP）。
    因为 store 的队列是原子的，4 个 Runner 各自拿到不同的 rollout
    （rollout #1~#4），不会重复领取。
t0~: 每个 Runner 各自执行自己的 rollout（进程内实例调用 agent）：
     - 进入 trace_context，Tracer 插桩 agent 的 LLM 调用；
     - agent 调用了 3 次 LLM（比如生成 SQL → 看到报错 → 改 SQL），
       Tracer 自动生成 3 个 span，每个都通过 add_otel_span 写回 store
       （HTTP），并且每个 span 都刷新 attempt 的 last_heartbeat_time；
     - agent 执行完，返回最终 reward（如 SQL 是否跑通：1.0/0.0）。
t1: 4 个 Runner 各自 update_attempt(succeeded)（HTTP），完成自己的 rollout，
    继续 dequeue 下一批（rollout #5~#8）……直到 32 个全部完成。
    总计 32/4 = 8 轮，4 个 Runner 并行执行，吞吐是单 Runner 的 4 倍。
```

**如果某个 rollout 失败**：Runner 捕获 agent 抛出的异常，`update_attempt(failed)`（HTTP）；store 检查 `RolloutConfig`（如 `max_attempts=3`、`retry_condition=["failed","timeout"]`），若还可重试则把 rollout 标记为 `requeuing` 并创建 attempt #2，Runner 之后会再次领取重试。**如果某台 agent 机器上的 Runner 崩溃**：它的心跳停止（不再 HTTP 上报），store 的 watchdog 在 `unresponsive_seconds` 后把该 attempt 标记为 unresponsive，其他 agent 机器上的 Runner 可以接管重试——这就是 Attempt 机制的意义：**一个 rollout 可以有多次 attempt，rollout 是"外部视角"，attempt 是"内部执行视角"**。

> **面试一句话总结**：Runner 是 Agent Lightning 的"工人"，负责从 store 领取 rollout、协调 LitAgent 执行、通过 Tracer 插桩自动把执行轨迹（spans）流式写回 store、上报心跳并支持失败重试；多个 Runner 通过 store 的原子队列天然并行，是系统横向扩展的执行侧。

---

## 3. LightningStore / TrajStore（轨迹存储）：系统的"数据库 + 消息队列"

### 1. 现有问题：为什么需要 LightningStore

Agent Lightning 要解决的第三个问题是：**算法和 Runner 分处不同进程/机器（甚至完全解耦），它们之间靠什么通信？** Algorithm 要派任务给 Runner、Runner 要把轨迹还给 Algorithm，如果直接互相调用，两者就强耦合了——算法换机器、Runner 扩容都会牵一发动全身。LightningStore 的定位是"**中央数据库和消息队列，作为系统的单一事实来源（single source of truth）**"（官方文档原话）：它存储任务（rollouts）、执行记录（attempts）、轨迹（spans）、版本化资源（resources）和 worker 元数据，并暴露一套 API 让 Algorithm 和 Runner 通信。

它的设计本质是**用"存储"解耦"生产"和"消费"**：Algorithm 只管 enqueue 任务、查询结果；Runner 只管 dequeue 任务、写入轨迹；两者**永不直接对话**，只通过 store 交换数据（官方强调：Runner 和 Algorithm 之间零直接通信）。这让两侧可以独立扩展（Runner 随便加，算法侧由 DeepSpeed/Megatron 等分布式训练框架负责扩展）。另外一个现实需求是**分布式一致性**：多 Runner 并发抢任务不能重复、多进程写 spans 必须有序——这些并发问题也由 store 统一解决。

### 2. 方法论：LightningStore 是怎么实现的

**LightningStore 的完整接口**（`store/base.py` 的抽象类，按功能分组；官方 docstring 定义它"协调训练 rollout 的持久化控制面"，每个方法都必须线程安全/异步安全）：

**（A）任务队列（Rollout 生命周期）**——算法派任务、Runner 领任务：

| 方法 | 作用 |
|---|---|
| `enqueue_rollout(input, mode, resources_id, config, metadata) → Rollout` | 持久化一个 `queuing` 状态的 rollout 入队（**不建 attempt**，等 Runner 来领）；生成唯一 rollout_id、默认 RolloutConfig |
| `enqueue_many_rollouts(rollouts) → [Rollout]` | 批量入队（verl 的 `AgentModeDaemon` 用这个一次入 32 条） |
| `start_rollout(input, ...) → AttemptedRollout` | **立即**建 rollout + 第一个 attempt（sequence_id=1，status=preparing），不走队列——给"在线/持续学习"用 |
| `dequeue_rollout(worker_id) → AttemptedRollout \| None` | Runner 认领最老的 queued rollout，原子地转为 preparing 并**自动创建 attempt**；多 Runner 并发调用不重（队列原子性） |
| `dequeue_many_rollouts(limit, ...)` | 一次认领最多 limit 个 queued rollout（不阻塞） |
| `start_attempt(rollout_id, worker_id) → AttemptedRollout` | 为已有 rollout **手动创建重试 attempt**（对应 retry 逻辑） |

**（B）Attempt 跟踪**——记录每次执行的状态/心跳：

| 方法 | 作用 |
|---|---|
| `update_attempt(rollout_id, attempt_id, status, worker_id, ...)` | 更新 attempt 状态（succeeded/failed）、worker 归属、心跳时间；驱动 worker 状态机（busy→idle） |
| `query_attempts(rollout_id)` | 返回该 rollout 的全部 attempt（按 sequence 升序） |
| `get_latest_attempt(rollout_id)` | 取 sequence_id 最高的 attempt |

**（C）轨迹采集（Span 摄入）**——Runner 写轨迹：

| 方法 | 作用 |
|---|---|
| `add_span(span) → Span` | 持久化一个**已构造好**的 Span（runner 显式构造的场景） |
| `add_many_spans(spans) → [Span]` | 批量持久化 span |
| `add_otel_span(readable_span, rollout_id, attempt_id, ...)` | **最常用**：把 OpenTelemetry `ReadableSpan` 归一化（`Span.from_opentelemetry`）后存储；**先领 sequence_id 保证有序** |
| `get_next_span_sequence_id(rollout_id, attempt_id) → int` | 分配 (rollout_id, attempt_id) 内的单调递增序号（span 排序用） |
| `get_many_span_sequence_ids(pairs) → [int]` | 批量分配序号 |

**（D）轨迹查询**——算法读轨迹：

| 方法 | 作用 |
|---|---|
| `query_spans(rollout_id, attempt_id, ...) → [Span]` | 返回某 rollout 的 spans（可按 attempt 限定、分页/排序/过滤） |
| `query_rollouts(status_in, rollout_ids, ...)` | 按状态/ID 过滤查询 rollout |
| `get_rollout_by_id(rollout_id)` | 取单个 rollout（不改状态） |
| `wait_for_rollouts(rollout_ids, timeout) → [Rollout]` | **阻塞等待**目标 rollout 达到终态或超时——算法等结果的核心方法 |

**（E）资源版本管理**——模型/prompt/端点快照：

| 方法 | 作用 |
|---|---|
| `add_resources(resources) → ResourcesUpdate` | 持久化一份**不可变**的资源快照并标记为 latest |
| `update_resources(resources_id, resources)` | 覆盖/扩展已有快照并标记为 latest |
| `get_latest_resources() → ResourcesUpdate` | 取全局最新快照（**Runner 领到任务后立即取这个**） |
| `get_resources_by_id(resources_id)` | 取指定快照 |
| `query_resources()` | 列出全部快照 |

**（F）Worker 心跳**：

| 方法 | 作用 |
|---|---|
| `update_worker(worker_id, ...)` | 记录心跳并刷新遥测（last_heartbeat_time、busy/idle 状态） |
| `query_workers()` | 查询所有 worker |
| `get_worker_by_id(worker_id)` | 取单个 worker |

**（G）能力声明**：`capabilities`（thread_safe / async_safe / zero_copy / otlp_traces）、`statistics()`（存储统计）、`otlp_traces_endpoint()`（若支持 OTLP 返回 `/v1/traces` 端点）。

**核心代码**（`store/base.py` 的接口定义——这是"调用方只面向接口编程"的代码基础）：

```python
class LightningStore:
    """Contract for the persistent control-plane that coordinates training rollouts.
    一个 LightningStore 调解算法与 runner 之间的每次交互：
    - Rollout lifecycle: 接受新 rollout、入队、创建 attempt、驱动状态机
    - Attempt tracking: 记录每次执行、心跳、重试、终态（timeout/unresponsive）
    - Span ingest: 接收 runner 的轨迹（Span 或 OpenTelemetry ReadableSpan）
    - Resource versioning: 管理不可变资源快照 + "latest" 快照
    """

    async def enqueue_rollout(self, input: TaskInput, mode=None, resources_id=None,
                              config=None, metadata=None) -> Rollout: ...
    async def dequeue_rollout(self, worker_id: str | None = None) -> AttemptedRollout | None: ...
    async def add_otel_span(self, readable_span, rollout_id, attempt_id, ...) -> Span: ...
    async def get_next_span_sequence_id(self, rollout_id, attempt_id) -> int: ...
    async def get_latest_resources(self) -> ResourcesUpdate | None: ...
    async def wait_for_rollouts(self, *, rollout_ids, timeout=None) -> List[Rollout]: ...
    async def query_spans(self, rollout_id, ...) -> List[Span]: ...
    async def update_attempt(self, rollout_id, attempt_id, status, ...) -> None: ...
```

**这段代码关键在哪**：LightningStore 的接口设计体现了"**用存储解耦生产与消费**"——算法侧只用 `enqueue_rollout`/`wait_for_rollouts`/`query_spans`，runner 侧只用 `dequeue_rollout`/`add_otel_span`/`update_attempt`，**两侧互不知道对方的存在**；`get_next_span_sequence_id` 保证 span 有序、`get_latest_resources` 让 runner 领到任务后立刻拿到当前模型/prompt 快照。

**三机分离下的 HTTP 化**（`store/client_server.py`——同一个接口，跨机器时变成 HTTP）：

```python
# 运行在 store 机器的 LightningStoreServer（FastAPI，暴露 /v1/agl/* 路由）
class LightningStoreServer(LightningStore):
    def __init__(self, store, host="0.0.0.0", port=4747):
        self.app = FastAPI()
        self._setup_routes()          # 把每个接口方法映射成 HTTP 路由

# 运行在 algo/agent 机器的 LightningStoreClient——实现同一套接口，
# 但内部把每个方法变成 aiohttp/httpx 的 HTTP 请求
class LightningStoreClient(LightningStore):
    def __init__(self, server_address: str):
        self._session_pool = ...      # 按 event-loop 的 aiohttp session 池
    async def dequeue_rollout(self, worker_id=None):
        async with session.post(f"{self._server_address}/v1/agl/dequeue_rollout",
                                json={"worker_id": worker_id}) as resp:
            ...
```

**这段代码关键在哪**：`LightningStoreClient` 实现了和 `LightningStoreServer` **完全相同的接口**（都是 `LightningStore` 子类），所以 Algorithm/Runner 代码**一行不用改**——三机分离时注入 Client（内部走 HTTP）、单机调试时注入内存 store（直接方法调用），对调用方完全透明。这正是"三机分离部署"的代码基础：store 机器跑 Server，algo/agent 机器注入指向它的 Client。

**存储的内容**（五类数据 + 一个队列 + 一个计数器）：`rollouts`（主键 `rollout_id`；任务单元，含输入/元数据/生命周期状态 queuing→preparing→running→succeeded/failed/requeuing/cancelled）、`attempts`（主键 `rollout_id + attempt_id`；每次执行尝试，一个 rollout 可多次 attempt，对应失败重试；状态 preparing→running→succeeded/failed，watchdog 可判 timeout/unresponsive）、`spans`（主键 `rollout_id + attempt_id + span_id`；轨迹事件，同一 rollout 内按 `sequence_id` 单调排序）、`resources`（主键 `resources_id`；版本化资源，如 prompt 模板 / LLM 端点）、`workers`（主键 `worker_id`；runner 心跳元数据）、一个 `rollout_queue`（FIFO 任务队列），以及一个 `span_sequence_ids` KeyValue（按 `rollout_id` 的单调计数器）。**Attempt 与 Rollout 的关系**：rollout 是"外部视图"，attempt 是"内部执行视图"——Runner 实际执行的是 attempt，rollout 状态是"最新 attempt 状态 + 排队/取消控制"的聚合。

**分层实现**（`store/` 目录，官方 classDiagram）：

1. **Collections 层**（`store/collection/`）：底层存储原语——`Collection`（带主键的索引存储：query/get/insert/update/upsert/delete）、`Queue`（FIFO：enqueue/dequeue/peek/size）、`KeyValue`（键值：get/set/inc/chmax/pop）。每个 backend 实现一套。**关键：`atomic()` 上下文管理器提供事务/锁语义**（InMemory 用排序锁防死锁，Mongo 用数据库原子性）。
2. **Store 层**（`store/collection_based.py`）：`CollectionBasedLightningStore` 在 collections 之上实现完整 LightningStore API，包括**业务逻辑：状态流转、watchdog 健康检查、重试策略**。
3. **包装层**：`LightningStoreThreaded`（加 mutex 线程安全，**同进程线程间直接用实例调用**）、`LightningStoreServer` / `LightningStoreClient`（**跨进程/跨机器的 HTTP 化**——Server 是 FastAPI 服务暴露 `/v1/agl/*` store API 和 `/v1/traces` OTLP 端点，Client 实现同一套 LightningStore 接口但内部把每个方法变成 aiohttp/httpx 的 HTTP 请求，带重试和按 event-loop 的 session 池）。**关键点：所有调用方（Algorithm/Runner）只面向 `LightningStore` 接口编程，同进程时是直接方法调用、跨进程时是 HTTP，对调用方透明**。

**开箱即用的实现**：`InMemoryLightningStore`（默认，零依赖，适合开发/CI/测试，锁模式可配 asyncio/thread）、`MongoLightningStore`（生产持久化、多进程安全、支持 `partition_id` 多 trainer 隔离）。通过 `capabilities` 属性（thread_safe / async_safe / zero_copy / otlp_traces）声明各自能力。

**spans 的分布式有序性**（关键设计）：分布式下不同进程产生的 span 时间戳可能乱序，store 强制每个 span 在写入前先 `get_next_span_sequence_id(rollout_id, attempt_id)` 领取单调递增的序号，保证**同一 rollout 内**的轨迹可稳定排序/合并。**注意计数粒度**：该方法的 `attempt_id` 只出现在签名里，计数器实际以 `rollout_id` 为键（`store/collection_based.py:1041-1049`，docstring 原话 "The number is strictly increasing for **each rollout**"），所以**重试产生的新 attempt 不会重置序号**，跨 attempt 的 span 必须靠 `attempt_id` 过滤来区分（详见第 11 节）。spans collection 的主键是 `["rollout_id", "attempt_id", "span_id"]`（`store/collection/memory.py:826`），这也是重复上传能被幂等丢弃的根据。OTEL span 经 `Span.from_opentelemetry()` 归一化后存储，且 store 暴露标准 OTLP `/v1/traces` 端点（`store/client_server.py:940-955`），任何 OpenTelemetry 兼容的 SDK/collector 都能直接灌 span——**这条 OTLP 路也是我们三机分离部署下实际走的传输路径**（详见第 8 节）。

### 3. 具体数值样例

**我们的三机分离部署**：`LightningStoreServer`（InMemory 底层）单独跑在 **store 机器**（`agl store --port 4747`），algo 机器和 4 个 Runner（在**其他机器**）都通过 `LightningStoreClient`（HTTP）连 store 机。逐步演算一次完整交互：

```text
第 1 步：algo 机 enqueue 8 个 rollout（HTTP 到 store 机）→ server 存入
        rollouts collection + rollout_queue 队列（队列里 8 个 rollout_id）。
第 2 步：4 个 Runner 各自调用 client.dequeue_rollout()（HTTP 到 store 机）→
        原子地从队列取 1 个 id，创建 attempt（status=preparing），
        返回 AttemptedRollout。因为队列原子，4 个 Runner 拿到的 id 互不重复。
第 3 步：Runner 执行时，每次 LLM 调用完成 → client.add_otel_span()（HTTP）：
        先领 sequence_id（KeyValue.inc()，如 #1→#2→#3），再存 span。
        3 次 LLM 调用 = 3 个 span，序列号 1、2、3，顺序确定。
        同时第一个 span 使 attempt 从 preparing → running。
第 4 步：Runner 完成 → client.update_attempt(succeeded)（HTTP）；
        若中途某 Runner 崩溃，心跳停止 → watchdog（在 server 每次变更前
        周期性调用）发现 last_heartbeat 超时 → attempt 标记 unresponsive；
        若 max_attempts=3 且 retry_condition 含 unresponsive，
        rollout → requeuing，之后其他 Runner 可重试。
第 5 步：algo 机 client.wait_for_rollouts(8 个 id)（HTTP）等到全部完成 →
        client.query_spans() 取回全部轨迹，交给 Adapter 转训练样本。
```

**对比两种 store 的取舍**：InMemory——零依赖、单机快，但进程间不能共享（除非包在 Server 里）；Mongo——持久化、天然多进程安全、partition_id 隔离，适合生产；Server/Client——把任意 store 变成 HTTP 服务，让 algo 和 Runner 跨机器访问，是"**store / algo / agent 三机分离**"部署的基础（store 单独一台机器当中心，其余机器全是 HTTP 客户端）。

> **面试一句话总结**：LightningStore（TrajStore）是 Agent Lightning 的中央数据库+消息队列，用"存储"解耦算法与 Runner——算法 enqueue/查询、Runner dequeue/写入，双方零直接通信；内部按 Collections（存储原语）→ Store（状态机+watchdog+重试业务逻辑）→ 包装层（线程安全/HTTP 化）三层实现，span 用单调序号保证分布式有序，支持 InMemory / Mongo / Server-Client 多种部署形态。

---

# 二、整体架构：三件套如何串成一个循环

## 4. Algorithm ↔ Runner ↔ Store 的完整闭环

### 1. 现有问题：为什么需要一个 Trainer 把三件套串起来

前面三个组件各自职责清晰，但**它们不会自己连起来**——谁创建 Algorithm？谁把 Store 注入 Runner？Runner 和 Algorithm 分别部署在哪台机器？异常时怎么互相通知终止？这些"组装和编排"问题需要一个总指挥，这就是 **Trainer**（`trainer/trainer.py`，官方定位"high-level orchestration layer that wires Algorithm <-> Runner <-> Store"）。Trainer 负责：**实例化 Algorithm 并注入 Store/Adapter/LLMProxy；创建一批 Runner（n_runners）并把 Store/Tracer/Agent 注入它们；通过 ExecutionStrategy 决定两个 bundle 的进程/线程布局；管理组件生命周期与优雅终止**。没有 Trainer，用户要手动接线几十个组件；有了它，一次 `trainer.fit(algorithm, dataset)` 就把整个循环跑起来。**三机分离下**：Trainer 的装配逻辑在 algo 机器和 agent 机器各跑一份（algo 机器装配 Algorithm bundle、agent 机器装配 Runner bundle），两边的 Store 注入的都是指向 store 机器的 `LightningStoreClient`。

另一个需要整体视角的原因是：**Agent Lightning 的可扩展性设计（runner 侧可无限扩、算法侧交给分布式训练框架）必须从"两个 bundle"的视角才能理解**。官方明确把系统分成两个可独立部署的 bundle：**Runner Bundle**（Runner + Tracer + Hooks + LitAgent）和 **Algorithm Bundle**（Algorithm + Adapter + LLMProxy），两者**唯一的交集就是 LightningStore 和 Trainer**。这个"零直接通信"的设计是理解整套架构的钥匙。

### 2. 方法论：整体架构是怎么组织的

**组件关系**（官方 mermaid 图的文字版）：Trainer **拥有**（has）Algorithm、Store、Adapter、LLMProxy、Runner、Tracer、Hooks；Store 被**注入**（injects）到 Algorithm、Runner、Tracer、Adapter、LLMProxy；Adapter 和 LLMProxy 被注入到 Algorithm；Tracer、LitAgent、Hooks 被注入到 Runner。依赖关系全部由 Trainer 统一装配，弱引用用于协调（Agent/Trainer 之间互相 reference）。

**核心数据流闭环**（一次完整训练循环，**标注了每条链路的通信方式**；**三机分离语境：图中 Algorithm 在 algo 机器、Runner 在 agent 机器、Store 在 store 机器，除 [②][④] 外全部跨机器走 HTTP**）：

```text
Algorithm ──enqueue_rollout──▶ Store（任务队列）      [①] HTTP
    ▲                            │ dequeue_rollout（自动建 attempt）[①] HTTP
    │                            ▼
    │                          Runner
    │                            │ 进入 trace_context，执行 agent     [②] 实例调用
    │                            ▼
    │                          Agent ──(LLM 调用)──▶ LLMProxy ──▶ vLLM/模型  [③] HTTP
    │                            │   Tracer 插桩自动捕获 spans
    │                            ▼
    │                    Store ◀──add_otel_span（带单调 sequence_id）  [①] HTTP
    │                            │ update_attempt(succeeded/failed)   [①] HTTP
    │                            ▼
Algorithm ◀──query_spans / wait_for_rollouts── Store   [①] HTTP
    │  Adapter: spans → triplets (prompt, response, reward)   [④] 实例调用
    │  训练（如 verl PPO）→ 更新模型权重
    └──▶ vLLM 加载新权重 → 下一轮
```

**通信方式全景（面试重点：哪些是 HTTP、哪些是实例直接调用）**：

- **[①] Store 相关调用（enqueue/dequeue/query_spans/add_otel_span/update_attempt）——可能是 HTTP，也可能是实例直接调用，取决于部署形态**：在 **shared-memory 模式**下，Algorithm/Runner 与 Store 在**同一进程**，这些是**直接 Python 方法调用**（async 方法，无网络）；在 **client-server 模式**下，调用方通过 `LightningStoreClient` 把这些调用变成 **HTTP 请求**（aiohttp/httpx）打到 `LightningStoreServer`（FastAPI，`/v1/agl/*` 路由）——**我们的三机分离部署（store/algo/agent 各一台机器）下，algo 和 agent 两侧都是经 `LightningStoreClient` 走 HTTP 连 store 机**，这是跨机器时唯一的通信方式。
- **[②] Runner → Agent（`agent.training_rollout()` / `validation_rollout()`）——总是实例直接调用**：Runner 持有 LitAgent 实例（同进程注入），直接调用 agent 的同步/异步方法（`training_rollout` / `training_rollout_async`），**不走网络**。Agent 和 Runner 永远在同一个进程（Runner Bundle 内）。
- **[③] Agent → LLMProxy → vLLM —— 总是 HTTP**：LLMProxy 是**独立的 FastAPI 服务**（带 RolloutAttempt/MessageInspection/StreamConversion 三个中间件），agent 的 LLM 调用以 **HTTP 请求**发到 proxy；proxy 再以 HTTP 转发到后端（vLLM 的 chat completion 端点 / 第三方 LLM 端点）。这条链路 HTTP 化的意义：proxy 能在中间拦截、记录每次 LLM 调用的轨迹（作为 span 写入 store）、统一鉴权/重试/限流、以及**动态切换后端模型**。
- **[④] Algorithm → Adapter —— 总是实例直接调用**：Adapter 是注入 Algorithm 的实例，`adapt(spans)` 是进程内方法调用（把 spans 转 triplets 是纯本地计算）。

**总结表（面试可直接背）**：

| 通信对 | shared-memory | client-server（三机分离：我们的部署） | 说明 |
|---|---|---|---|
| Algorithm ↔ Store | 实例直接调用 | **HTTP**（algo 机经 `LightningStoreClient` 连 store 机） | store 独立机器时必走 HTTP |
| Runner ↔ Store | 实例直接调用 | **HTTP**（agent 机经 `LightningStoreClient` 连 store 机） | 跨机器必走 HTTP |
| Runner → Agent | **实例直接调用** | **实例直接调用**（不变） | 同进程注入，永远不跨网络 |
| Agent → LLMProxy | **HTTP** | **HTTP** | proxy 是 FastAPI 服务 |
| LLMProxy → vLLM/模型 | **HTTP** | **HTTP** | 转发到后端 chat 端点 |
| Algorithm → Adapter | **实例直接调用** | **实例直接调用**（不变） | 纯本地转换 |
| 任意 → Store OTLP | — | **HTTP**（`/v1/traces`） | 兼容 OpenTelemetry 的 span 上报 |

**执行策略（ExecutionStrategy）**——决定 bundle 怎么部署，官方提供两种：

1. **SharedMemoryExecutionStrategy（共享内存）**：算法和 runner 作为**同一进程的线程**跑，Store 用 `LightningStoreThreaded` 包一层加锁。优点：共享 Python 堆、免序列化，适合轻量调试；缺点：不适合重 RL 训练或计算密集 agent。
2. **ClientServerExecutionStrategy（客户端-服务器）**：算法和 runner 分成**独立进程/机器**，通过 HTTP 通信。它用 `role` 参数声明本进程扮演的角色（`AGL_CURRENT_ROLE` 环境变量或直接传参）：**`role="algorithm"`**（本机起 `LightningStoreServer` 并执行算法 bundle）、**`role="runner"`**（用 `LightningStoreClient` 连已有 server 并执行 runner bundle）、**`role="both"`**（同机全跑，测试用）。`server_host` / `server_port`（默认 4747）指定 server 地址，`managed_store=True`（默认）时自动创建 client/server 包装器——**runner 侧 `LightningStoreClient(f"http://{server_host}:{server_port}")` 直接连远程 store**。

**我们的部署形态（三机分离）**：**store、algo、agent（runner）分别部署在三个不同的机器，全部通过 store 的 HTTP 地址连接**：

```text
┌─────────────┐   HTTP   ┌─────────────┐
│  algo 机器   │◀────────▶│  store 机器  │
│ Algorithm   │  /v1/agl │ Lightning   │
│ Adapter     │          │ StoreServer │
│ LLMProxy    │          │ (agl store) │
└─────────────┘          └──────┬──────┘
                                │ HTTP（http://<store机IP>:4747）
                    ┌───────────▼───────────┐
                    │  agent 机器（可多台）    │
                    │ Runner + Tracer + Agent│
                    │ role="runner"          │
                    └───────────────────────┘
```

- **store 机器**：单独启动 `LightningStoreServer`（`agl store --port 4747` 或嵌入算法进程），持有全部数据（rollouts/attempts/spans/resources/workers），是唯一的中心；
- **algo 机器**：执行 Algorithm bundle，通过 `LightningStoreClient`（或 `managed_store=False` 手动注入 client）连 store 的 HTTP 地址——注意 algo 侧默认 `role="algorithm"` 会本机起 server，**store 独立部署时要把 server 放在 store 机、algo 机只作为 client 连过去**；
- **agent 机器**：执行 Runner bundle（可多台、每台多进程），`role="runner"` + `server_host=store机IP` + `server_port=4747`，`LightningStoreClient` 经 HTTP 连 store。

**两个 bundle 的解耦价值**：因为 Runner 和 Algorithm 唯一交集是 Store，所以可以：Runner 铺 N 个进程/机器跑 agent（吞吐随 runner 线性扩展）、Algorithm 侧随时换训练后端（verl / APO / 自定义）、两者互不阻塞。Abort 处理也由 ExecutionStrategy 负责：算法正常退出 → 通知 runner 终止；runner 先退 → 不打扰算法（它可能还在处理已完成的 rollout）；失败/中断 → 两边都 abort，必要时强制终止。

### 3. 具体数值样例

用**我们的三机分离部署**（store / algo / agent 各一台机器）+ 4 个 runner 进程的实例，逐步演算整体架构如何协作（假设 batch 32、每个 agent 平均 3 次 LLM 调用、每次调用生成 ~500 token）。**这个部署下通信边界很清楚：algo 机和 agent 机与 store 机之间全部走 HTTP（经 `LightningStoreClient` → `LightningStoreServer`）；Runner 进程内部 Agent 是实例直接调用；agent 的 LLM 调用走 HTTP 到 LLMProxy**：

```text
部署：
  store 机器：LightningStoreServer（agl store --port 4747），持有全部数据
  algo 机器：Trainer + Algorithm + Adapter + LLMProxy + verl/vLLM
             （经 LightningStoreClient 连 store 机，HTTP）
  agent 机器 ×2：各 2 个 Runner 进程（共 4 Runner），role="runner"、
             server_host=store机IP、server_port=4747
             （Runner↔Store 是 HTTP；Runner 内部 Agent 是实例调用）

第 1 轮（派发）：Algorithm enqueue 32 个 rollout（HTTP 到 store 机）；
     4 个 Runner 各自 dequeue（HTTP 请求到 store 机，
     原子队列保证不重），各拿 8 个任务开始跑。

第 2 轮（执行 + 轨迹）：每个 Runner 内部：
     Runner → Agent 是实例直接调用（training_rollout）；
     agent 每次 LLM 调用发 HTTP 到 LLMProxy（algo 机上）→ proxy 再
     HTTP 转发到 vLLM；
     每个 rollout 平均 3 次 LLM 调用 → Tracer 插桩生成 3 个 span
     （带 sequence_id 1/2/3）经 HTTP 写回 store 机；
     4 Runner × 8 rollout × 3 span = 96 个 span 流入 store，
     总生成 token ≈ 4×8×3×500 = 48K token。
     期间每 5 秒每个 Runner 发一次心跳（update_worker，HTTP），
     保证 watchdog 不误判 unresponsive。

第 3 轮（学习）：32 个 rollout 全部完成（Algorithm 经 store 机
     的 wait_for_rollouts 等到，HTTP）；
     Algorithm 调 query_spans 取回 96 个 span（HTTP 到 store 机）；
     TracerTraceToTriplet（实例调用）把每条轨迹转成 ~3 个 triplet
     （共 ~96 个），最终 reward 同值分配给轨迹内所有 triplet；
     triplets → verl DataProto（batch 32 × 3 = 96 条训练样本）→
     verl PPO 训练一步 → 更新权重 → vLLM 加载新权重。

第 4 轮（下一 batch）：继续 enqueue 下 32 条 → 循环。
     （在线/持续学习模式下，Runner 可自发上报新任务，不依赖固定 dataset。）
```

**注意 shared-memory 模式下的差异**：若用 SharedMemoryExecutionStrategy（单机调试），则所有"HTTP 到 store"的调用都变成**同进程实例直接调用**（`LightningStoreThreaded` 加锁保护），只有 agent → LLMProxy → vLLM 这条 LLM 调用链**始终是 HTTP**（proxy 和 vLLM 本来就是独立服务）。

**性能视角**：4 Runner 把"agent 执行 + 轨迹产生"的吞吐放大了 4 倍，而 vLLM + verl 把"模型推理 + 训练"的算力集中在一侧——**执行侧和训练侧各自按需扩展，互不干扰**，这正是"用 store 解耦"带来的架构红利。

> **面试一句话总结**：Agent Lightning 整体架构 = Trainer 统一装配的 Algorithm Bundle（算法+Adapter+LLMProxy）与 Runner Bundle（Runner+Tracer+Agent）两个可独立部署的包，唯一交集是 LightningStore；数据流是"算法 enqueue → Runner dequeue 执行 → Tracer 自动写 spans → 算法 query+Adapter 转样本 → 训练更新资源 → 回到下一轮"，ExecutionStrategy（shared-memory / client-server）决定部署形态，runner 侧可横向扩展、算法侧交给分布式训练框架。

---

## 5. 与 verl 的关系：Agent Lightning 如何"借用" verl 做 RL 训练

### 1. 现有问题：Agent Lightning 为什么需要 verl

Agent Lightning 本身**不做模型训练**——它只负责"让 agent 跑起来、把轨迹存下来"，但"用这些轨迹更新模型权重"是 RL 训练框架的活。如果从零实现一个 PPO/GRPO 训练器，工程量巨大且不必要——业界已有成熟的 RL 训练框架。Agent Lightning 选择了 **verl**（Volcengine 开源的全栈 RL 训练框架，基于 Ray 做分布式、支持 vLLM rollout + FSDP/Megatron 训练、内置 GRPO/PPO 等算法），把它作为默认的 RL 训练后端。反过来看，**verl 本身是"为 RL 训练设计"的，但它假设 rollout 是标准的 RL 环境采样；agent 这种"多轮、工具调用、外部环境交互"的 rollout 形态 verl 原生不支持**——Agent Lightning 的价值正是把"agent 轨迹"翻译成 verl 能吃的"RL 轨迹"，两者互补。

这里有一个**关键的语义差异**（官方文档明确强调）：verl 的 RLHF 设置是"**每个 action 是一个 token**，state 是到该 token 为止的完整对话历史，reward 在末尾给出"；而 Agent Lightning 的 agent 场景是"**每个 action 是一段文本（一次 LLM 响应/工具调用）**"。所以 Agent Lightning 不能把 spans 直接喂给 verl，必须经过"轨迹 → triplet → DataProto"的转换（见下文）。

### 2. 方法论：Agent Lightning 与 verl 的集成是怎么实现的

集成代码分布在两个地方（官方文档注明"for historical reasons"）：**`agentlightning/verl/`**（legacy 集成目录，含 trainer/daemon/entrypoint/dataset/async_server，复用了一些旧的命名）和 **`agentlightning/algorithm/verl/`**（新算法接口的薄包装 `VERL(Algorithm)`，未来会合并）。核心组件与流程：

**（1）`VERL(Algorithm)`（`algorithm/verl/interface.py`）**：算法入口。构造时接收 dict 配置（镜像 verl CLI overrides），用 Hydra 与 verl 打包的默认配置（`config.yaml`）合并；`run()` 里启动 verl 的 PPO entrypoint。它有两种执行模式，**`AgentModeDaemon` 的 `mode` 参数默认就是 `"v1"`，实际训练（包括我们）都走 v1 模式**：v1 要求注入 `store` / `llm_proxy` / `adapter`（源码里 `assert store is not None`），任务、轨迹、资源全部通过 LightningStore 流转；v0 模式是历史遗留（自己起 Flask server 和独立 proxy 端口，不经过 store），仅用于兼容旧接口。

**（2）`entrypoint.py`**：`run_ppo()` 初始化 Ray（namespace 设为 `transfer_queue`，为 TQ 共享做准备）、构建角色→worker 映射（ActorRolloutRefWorker / TrainingWorker，按 `need_reference_policy` / `need_critic` / LoRA 配置决定）、用 `ResourcePoolManager` 分配资源池（global_pool + 可选 reward_pool/teacher_pool），创建 `AgentLightningTrainer` 并 `init_workers()` + `fit()`。

**（3）`AgentLightningTrainer(RayPPOTrainer)`（`verl/trainer.py`）**：**这是集成的核心**——它继承 verl 的 `RayPPOTrainer` 但为 agent 模式做了大幅简化（官方 docstring 列出 4 点差异）：使用 `AgentModeDaemon` 做服务器通信、简化数据流（去掉 pop/union 操作）、通过 agent daemon 直接批量处理、用 agent_mode 做流线化验证。它的 `_train_step` 关键逻辑：**把 batch 的非张量数据交给 `agent_mode_daemon.set_up_data_and_server()` → `run_until_all_finished()` 等 agent 跑完 → `get_train_data_batch()` 回收训练样本**（从完成的 rollout 重建 prompt/response/reward，padding 到 max_prompt/max_response_length，返回 DataProto）。它重写了 `_compute_reference_log_prob` 以支持 LoRA 场景（`ref_in_actor=True` 时由 actor_rollout worker 算 ref logprob 而不是独立 ref worker——对应 verl 0.6+ 的行为）。训练中复用 verl 的 `compute_advantage` / `apply_kl_penalty` / `agg_loss` 等核心算法，以及 `compute_data_metrics`（加 suffix 区分 post-processing 前后的指标：丢弃超长 prompt、batch 对齐到 mini PPO size 倍数）。

**（4）`AgentModeDaemon`（`verl/daemon.py`）**：**算法侧的"agent 服务器"**——它把训练 batch 的数据（prompt）提供给 agent 执行，等待执行完，把结果（rollout 的 triplets + final_reward）回收并转成训练 batch。**我们使用的 v1 模式**（`mode="v1"`，默认值）的具体流程：构造时复用 trainer 注入的 `store` / `llm_proxy` / `adapter`，并启动一个内部 event loop 线程（`_internal_loop_runner`）替代 v0 的 Flask server；`set_up_data_and_server()` 把 batch 数据经 **`store.enqueue_many_rollouts()` 批量入队**（带 `RolloutConfig`：`unresponsive_seconds` / `timeout_seconds` 都设为 `llm_timeout_seconds`）；`run_until_all_finished()` 通过 **`store.wait_for_rollouts(rollout_ids, timeout=0)` 轮询**完成状态；`get_train_data_batch()` 前先用 **`_validate_data_v1()`** 把每个完成 rollout 的轨迹取回——**`store.query_spans(rollout_id, attempt_id="latest")` 查 spans → `adapter.adapt(spans)`（TracerTraceToTriplet）转成 triplets → 从 triplets 反向搜索非 None 的 reward 作为 final_reward**——再重建 prompt_ids/response_ids、padding/截断到 max_prompt/max_response_length 生成训练 DataProto。核心方法：`set_up_data_and_server(data, server_addresses)`、`run_until_all_finished()`、`get_train_data_batch(max_prompt_length, max_response_length, ...)`、`get_test_metrics()`（验证集指标）。`_fillna_reward` 处理缺失 reward（默认填 0.0）。

**（5）`dataset.py` 与 `async_server.py`**：`AgentDataset(RLHFDataset)` 为 agent 场景定制（过滤超长 prompt、给 DataProto 加 fake_ids 张量占位、保留 index）；`PatchedvLLMServer` 继承 verl 的 `AsyncvLLMServer` 并 `instrument_vllm()` 打补丁——**让 vLLM 的 chat completion 也能产生 span 轨迹**（agent 的模型调用走 vLLM 时同样被插桩记录）。

**（6）资源与模型管理**：VERL 场景下 `main_llm` 资源是 `ProxyLLM`（LLM 子类），包含一个带 rollout_id/attempt_id 占位符的 **URL 模板**——每个 rollout 开始时用当前 id 格式化出唯一端点 URL，让 LLMProxy 能精确拦截和记录**这一次 attempt** 的流量（用于轨迹和负载均衡）。agent 通过 `@rollout` 装饰器时自动解析该模板（"auto-stripped"），类式 agent 需手动 `proxy_llm.get_base_url(rollout_id, attempt_id)`。

**（7）轨迹 → 训练样本的完整链路**（最重要的心智模型）：

```text
Runner/Tracer 写 spans ──▶ Store（add_otel_span）
Algorithm 查询 spans ──▶ Adapter（TracerTraceToTriplet）
    → (prompt, response, reward) triplet 列表
    → 最终 reward 用 identical assignment 赋给轨迹内所有 triplet
      （每条 triplet 都是可独立优化的 RLHF 轨迹）
    → 转成 verl DataProto：input_ids / position_ids / attention_mask /
      token_level_scores（token 级分数，reward 打在最后一个 token 上）
    → verl PPO/GRPO 训练（RayPPOTrainer + vLLM rollout + FSDP 训练）
    → 更新权重 → vLLM 加载 → 下一轮 agent 用新模型
```

**一个细节**：verl 的 `RayPPOTrainer` 原本自带 rollout 引擎（verl 的 vLLM rollout actor 负责采样）。在 agent 模式下，**采样不是 verl 的标准 rollout 做的，而是 agent 通过 LLMProxy 调 vLLM 完成的**——v1 模式下 `AgentModeDaemon` 把"要生成的 prompt"作为 rollout 入队 store，Runner 领走后 agent 自由发挥（多轮、调工具），产生的多轮轨迹写回 store 才是训练数据。verl 在这里主要承担"训练 + 部分推理服务"的职责，agent 的 rollout 语义由 Agent Lightning 自己定义。这就是官方强调的"verl 的 RL 是 token 级 action，agent 的 RL 是文本块级 action"的本质差别。**v1 模式相比 v0 的核心区别**：v0 是"daemon 自己起 server、任务直连"，v1 是"任务/轨迹/资源全部走 LightningStore"——v1 让 agent 的 rollout 生产与 verl 的训练消费通过 store 完全解耦，Runner 可以分布在任意机器上，这也是我们实际使用 v1 的原因。**三机分离下**：verl 训练跑在 **algo 机器**，Runner 跑在 **agent 机器**，两者只通过 **store 机器**的 HTTP 接口交换数据。

### 3. 具体数值样例

假设用 `agl.VERL(config)` 训练 SQL agent：config 里 `data.train_batch_size=32`、`max_prompt_length=4096`、`max_response_length=2048`、`actor_rollout_ref.rollout.n=4`（每个 prompt 生成 4 条响应）、`adv_estimator=grpo`、LoRA rank=8。逐步演算一个 batch 的完整数据流（**三机分离部署：verl 在 algo 机器、Runner 在 agent 机器、数据平面在 store 机器；除实例调用外全部 HTTP**）：

```text
第 1 步：VERL.run() → entrypoint.run_ppo() → Ray.init(namespace=transfer_queue)
        → 建 ActorRolloutRefWorker（global_pool）→ AgentLightningTrainer.init_workers()
        → trainer.fit() 进入训练循环（全部在 algo 机器）。
        （v1 模式：Trainer 已把 store / LLMProxy / Adapter 注入
        AgentModeDaemon，这些是实例直接调用；store 是连 store 机的
        LightningStoreClient。）

第 2 步（_train_step，v1 模式）：取 32 条 prompt（DataProto）→
        agent_mode_daemon.set_up_data_and_server() →
        把 32 条 prompt 作为 rollout 经 store.enqueue_many_rollouts()
        批量入队（**HTTP 到 store 机器**；每条带 RolloutConfig：
        timeout/unresponsive 都设为 llm_timeout_seconds，如 1200s；
        resources_id 指向 main_llm）。
        数据不是"挂到某个 server 端点"，而是进 store 的任务队列，
        由 agent 机器上的 Runner 来领。

第 3 步（agent 执行）：agent 机器上 4 个 Runner 从 store dequeue 这 32 个
        rollout（n_runners=4；**HTTP 到 store 机器**），Runner→Agent 是
        实例直接调用；每个 agent 通过 LLMProxy（**HTTP**，URL 模板带
        rollout_id）调 vLLM 生成 SQL；Tracer 记录每条轨迹的 spans
        （LLM 调用、工具调用、中间 reward），经 store.add_otel_span
        写回（**HTTP 到 store 机器**）；agent 执行完返回 final_reward
        （SQL 是否通过：1.0/0.0），Runner 调 update_attempt(succeeded)。
        n=4 意味着每个 prompt 实际生成 4 条独立响应 = 128 条轨迹。

第 4 步（回收，v1 模式）：algo 机器上 run_until_all_finished() 通过
        store.wait_for_rollouts(32 个 rollout_id, timeout=0) 轮询完成
        （**HTTP 到 store 机器**）→
        对每个完成 rollout 调 _validate_data_v1()：
        store.query_spans(rollout_id, attempt_id="latest") 查轨迹
        （**HTTP**）→ adapter.adapt(spans)（TracerTraceToTriplet，实例
        调用）转 triplets → 从 triplets 反向找非 None reward 作为
        final_reward → get_train_data_batch(4096, 2048)：
        - 每条轨迹拆成 triplet（prompt, response, final_reward 同值分配）；
        - padding/截断到 4096/2048，生成 token_level_scores；
        - 返回训练 DataProto（128 条样本）。

第 5 步（GRPO 训练，algo 机器）：verl 用 DataProto 算 advantage（grpo
        组内相对）、ref logprob（LoRA 时由 actor worker 算）、KL、loss；
        FSDP + LoRA 更新权重；指标经 compute_data_metrics 记录。

第 6 步：vLLM 加载新权重 → 下一 batch 循环。
```

**量化对比**：如果不用 Agent Lightning，直接用 verl 训练 agent，你需要自己实现"把 agent 多轮轨迹转成 verl 的 token 级 RL 轨迹"——这恰恰是 Agent Lightning 的 Adapter + AgentModeDaemon 替你做的事；反之如果只用 Agent Lightning 不用 verl，你就要自己写 PPO 训练器。**两者是"轨迹生产侧"和"训练侧"的分工**：Agent Lightning 管"agent 如何产生可训练轨迹"，verl 管"轨迹如何变成梯度"。

> **面试一句话总结**：Agent Lightning 与 verl 是"轨迹生产"与"RL 训练"的分工：`VERL(Algorithm)` 包装 verl 的 PPO entrypoint，`AgentLightningTrainer(RayPPOTrainer)` 继承 verl 训练器并用 `AgentModeDaemon` 让 agent 通过 LLMProxy 调 vLLM 完成"文本块级"的 agent rollout（区别于 verl 原生的 token 级 RL 采样）；轨迹经 Adapter 转成 (prompt, response, reward) triplets、reward 同值分配后，再转成 verl 的 DataProto 喂给 PPO/GRPO——agent 的多轮轨迹因此被"翻译"成 verl 能吃的 RL 训练样本。

---

# 三、轨迹（Trace / Span）的完整生命周期：产生 → 记录 → HTTP 传输 → 建树 → 转训练样本

前面讲了"谁负责什么"，这一部分专门钻进**数据本身**：一条轨迹在 Agent Lightning 里到底长什么样、它是在哪一行代码被"抓住"的、又是怎么变成 HTTP 请求落到 store 的、到了 store 之后怎么从一堆扁平 span 重建成一棵树、最后怎么变成 verl 能吃的 triplet。**这是面试里最容易被追问到底层的部分**（"你说 Tracer 自动记录，那它到底 hook 了什么？""span 怎么保证顺序？""父子关系断了两条 span 还算一棵树吗？"）。

## 6. 轨迹是怎么被记录的：三条记录通路

### 1. 现有问题：agent 代码零改动的前提下，轨迹从哪来

Agent Lightning 的卖点是"agent 几乎零代码改动"，但训练又需要**逐次 LLM 调用的 token 级信息**（`prompt_token_ids` / `response_token_ids` / `logprobs` / reward）。这两件事天然矛盾：**不改 agent 代码，凭什么知道它调了几次模型、每次用了哪些 token？** 答案是把"记录"这件事拆成**三条互不相同的通路**，各自负责一类信号：

1. **函数插桩（instrumentation）**——不需要 agent 配合，靠 monkey-patch 把第三方库（agentops / litellm / langchain / vllm）的关键函数换掉，在调用前后自动开 span、写属性。这是"自动记录"的主力；
2. **主动 emitter**——agent 自己知道"这一轮我得了多少分"，需要主动上报（`emit_reward` / `emit_message` / `emit_object` / `emit_exception`）。这是"语义记录"，插桩抓不到；
3. **LLMProxy 侧记录**——agent 的 LLM 调用全都经过 proxy，proxy 作为独立服务自己也能产生 span（它是 LiteLLM 的 OpenTelemetry 集成），而且**只有它知道 agent 用的是哪个 rollout/attempt**。

三条通路产生的 span 最终汇到同一个 `TracerProvider`，由同一个 `SpanProcessor` 统一落库——所以先要理解 **Tracer 的全局单例机制**，否则会看不懂"为什么 emit_reward 必须在 trace_context 里才不报错"。

### 2. 方法论：插桩、emitter、proxy 三路记录是怎么实现的

**（1）Tracer 抽象与"当前活跃 Tracer"的单例语义。** `Tracer` 是插件式基类（`tracer/base.py:27`），核心接口是 `trace_context()`（异步上下文管理器，进入时开 trace、退出时收口）、`create_span()`（凭空造 span，emitter 用）、`operation_context()`（记录一段操作，返回 `SpanRecordingContext` 可记异常/属性/状态）、`get_last_trace()`（取最近一次 trace 的 span 列表）。

关键机制在 `with_active_tracer_context` 装饰器 + 模块级全局变量（`tracer/base.py:22,257-285`）：

```python
_active_tracer: Optional[Tracer] = None

def set_active_tracer(tracer: Tracer):
    global _active_tracer
    if _active_tracer is not None:
        raise ValueError("An active tracer is already set. Cannot set a new one.")
    _active_tracer = tracer

class _ActiveTracerAsyncCM(AsyncContextManager[T]):
    async def __aenter__(self):
        set_active_tracer(self._tracer)      # 嵌套会直接抛错
        ...
    async def __aexit__(self, *args, **kwargs):
        try:
            return await self._inner.__aexit__(*args, **kwargs)
        finally:
            clear_active_tracer()
```

**这段代码关键在哪**：① `trace_context` 被 `@with_active_tracer_context` 包住，所以"进入 trace 上下文"= "把某个 Tracer 设为进程内的活跃 Tracer"；② `set_active_tracer` 遇到已有活跃 Tracer 会 `raise`，即**同进程不支持 trace 上下文嵌套**（多 runner 要各占一个进程/线程）；③ emitter 就是靠 `get_active_tracer()` 拿到这个单例来造 span 的——**这就是"emit_reward 必须在 trace_context 内"的根因**。

**（2）两条记录通路的差别：`OtelTracer` vs `AgentOpsTracer`。** 这是最容易答错的一点——两个 Tracer 的能力完全不同：

| Tracer | 能记录什么 | 源码自述 |
|---|---|---|
| `OtelTracer`（`tracer/otel.py:73`） | 只有 Agent Lightning **自己的信号**（emitter 产生的 reward/message/object/annotation span） | "You should be able to collect agent-lightning signals like rewards with this tracer, but **no other function instrumentations like `openai.chat.completion`**" |
| `AgentOpsTracer`（`tracer/agentops.py:32`） | OTel 全部能力 + 第三方库插桩（openai / litellm / langchain / vllm） | 继承 `OtelTracer`，额外调 `instrument_all()` |

`AgentOpsTracer._initialize_tracer_provider()` 的初始化顺序很关键（`tracer/agentops.py:70-96`）：

```python
def _initialize_tracer_provider(self, worker_id: int):
    if self.instrument_managed:
        self.instrument(worker_id)          # = instrument_all()，先插桩
    if self.agentops_managed:
        os.environ.setdefault("AGENTOPS_API_KEY", "dummy")     # 不需要真 key
        if not agentops.get_client().initialized:
            agentops.init(auto_start_session=False)            # 只初始化 SDK，不开 session
    span_processors = get_span_processors(self._get_tracer_provider(), LightningSpanProcessor)
    if len(span_processors) > 0:
        self._lightning_span_processor = span_processors[0]    # 复用已有的，避免重复注册
    else:
        self._lightning_span_processor = LightningSpanProcessor()
        self._get_tracer_provider().add_span_processor(self._lightning_span_processor)
```

**注意 `AGENTOPS_API_KEY` 默认塞的是字符串 `"dummy"`**——因为 Agent Lightning **不把数据发到 AgentOps 云端**，只用它的本地插桩能力。

**（3）"不发云端"的证据：`_patch_exporters` 把 exporter 换成可旁路的实现。** `instrumentation/agentops.py:49-58` 直接把 agentops 模块里的 exporter 和客户端换成 `Bypassable*` 子类：

```python
def _patch_exporters():
    agentops.sdk.core.AuthenticatedOTLPExporter = BypassableAuthenticatedOTLPExporter
    agentops.sdk.core.OTLPMetricExporter = BypassableOTLPMetricExporter
    if hasattr(agentops.sdk.core, "OTLPSpanExporter"):
        agentops.sdk.core.OTLPSpanExporter = BypassableOTLPSpanExporter
    agentops.client.api.V3Client = BypassableV3Client
    agentops.client.api.V4Client = BypassableV4Client
```

而 `BypassableV3Client.fetch_auth_token` 在服务未启用时**直接返回假的 token**（`instrumentation/agentops.py:292-297`）：

```python
def fetch_auth_token(self, *args, **kwargs) -> AuthTokenResponse:
    if _agentops_service_enabled:
        return super().fetch_auth_token(*args, **kwargs)
    else:
        return AuthTokenResponse(token="dummy", project_id="dummy")
```

`_agentops_service_enabled` 默认 `False`（`instrumentation/agentops.py:29`），也就是**默认纯本地模式**：不鉴权、不上报、不请求 agentops.ai。`BypassableAuthenticatedOTLPExporter` 更是同时继承 `LightningStoreOTLPExporter`（`instrumentation/agentops.py:248`），把导出目标**直接改指向 LightningStore**。

**（4）插桩到底"插"了什么：monkey-patch `handle_chat_attributes`。** agentops 内部用 `handle_chat_attributes(args, kwargs, return_value)` 把一次 LLM 调用的信息转成 OTel span 属性。Agent Lightning 把它整体替换，**追加 vLLM 特有的 token 字段**（`instrumentation/agentops.py:92-147`，按 agentops 版本走 `_patch_new_agentops` / `_patch_old_agentops` 两条分支）：

```python
_original_handle_chat_attributes = handle_chat_attributes

def _handle_chat_attributes_with_tokens(args=None, kwargs=None, return_value=None, **kws):
    attributes = _original_handle_chat_attributes(args=args, kwargs=kwargs, return_value=return_value, **kws)
    return_value = _unwrap_legacy_response(return_value)      # 处理 LegacyAPIResponse（LiteLLM / LangChain）

    if return_value is not None and hasattr(return_value, "prompt_token_ids"):
        attributes["prompt_token_ids"] = list(return_value.prompt_token_ids)
    if return_value is not None and hasattr(return_value, "response_token_ids"):
        attributes["response_token_ids"] = list(return_value.response_token_ids[0])
    ...
    if hasattr(first_choice, "logprobs") and first_choice.logprobs is not None:
        attributes["logprobs.content"] = json.dumps([lp.model_dump() for lp in first_choice.logprobs.content])
    return attributes
```

**这段代码关键在哪**：训练需要的 token id 和 logprob **不是 OTel 标准字段**，是靠这个补丁塞进 span attributes 的；没有它，后面 Adapter 拿不到 token，轨迹就是废的。

**（5）token id 的完整传递链条（六跳，面试可以直接背这条链）。** 这是理解"轨迹为什么能变成训练样本"的关键：

```text
① vLLM 引擎内部产生 prompt_token_ids / response_token_ids
     ↓  instrumentation/vllm.py: OpenAIServingChat.chat_completion_full_generator 被替换，
        用 _generate_inceptor() 包住 result_generator，边算边把 token ids 抓出来，
        再 response.model_copy(update={"prompt_token_ids":..., "response_token_ids":...})
     ↓
② 这两个字段被塞进 ChatCompletionResponse（vLLM 侧新增字段）
     ↓  注释：该插桩已上游化（"merged to upstream vLLM since v0.10.2"）
③ OpenAI SDK 客户端反序列化出对象（可能是 LegacyAPIResponse，
    需要 _unwrap_legacy_response() 调 response.parse()）
     ↓
④ agentops 的 handle_chat_attributes 读出 token ids（Agent Lightning 的补丁）
     ↓
⑤ 写进 OTel span 的 attributes["prompt_token_ids"] / ["response_token_ids"] / ["logprobs.content"]
     ↓
⑥ TracerTraceToTriplet.span_to_triplet() 读回 token ids → Triplet.prompt["token_ids"] / response["token_ids"]
     ↓
⑦ AgentModeDaemon 转成 verl DataProto 的 input_ids / response_ids → 真正进训练
```

**一个真实的坑**：`instrumentation/__init__.py:25-32` 里 vLLM 插桩的 import 被显式注释掉了，并留了一句全大写的警告：

```python
# MAGIC! DO NOT TOUCH THIS!
# vllm import will cause reward tracing function to fail and produce nothing.
# try:
#     from . import vllm
#     VLLM_INSTALLED = True
# except ImportError:
#     pass
```

也就是说 **v0.3.0 的默认组合里，vLLM 的 token-id 插桩不通过这条路径启用**（改由 verl 侧的 `PatchedvLLMServer` + `instrument_vllm()` 显式调，见 `verl/async_server.py`），否则会破坏 reward 的 tracer。

**（6）第二条通路：emitter 主动上报。** `emit_reward(1.0)` 的落点是 `emit_annotation`（`emitter/reward.py:204-206` → `emitter/annotation.py:35-67`）：

```python
def emit_annotation(annotation: Dict[str, Any], propagate: bool = True) -> SpanCoreFields:
    annotation_attributes = flatten_attributes(annotation, expand_leaf_lists=False)   # 嵌套 dict 展平
    check_attributes_sanity(annotation_attributes)
    sanitized_attributes = sanitize_attributes(annotation_attributes)

    if propagate:
        tracer = get_active_tracer()
        if tracer is None:
            raise RuntimeError("No active tracer found. Cannot emit annotation span.")
    else:
        tracer = DummyTracer()

    return tracer.create_span(name=AGL_ANNOTATION, attributes=sanitized_attributes,
                              status=TraceStatus(status_code="OK"))
```

**三个要点**：① `flatten_attributes` 把嵌套 dict 展平成点分键——多维 reward `{"task_completion": 1.0, "efficiency": 0.8}` 最终是 `agentlightning.reward.0.name` / `agentlightning.reward.0.value` 这样的扁平属性（`semconv.py:108-123`），**读取时再由 `_attributes_unflatten_multiple` 反展平**；② `propagate=False` 走 `DummyTracer`，只返回字段不上报（用于"只想拿字段给别的 span 用"的场景）；③ `get_active_tracer() is None` 时**抛 RuntimeError** ——这就是"必须在 `trace_context` 内 emit"的硬约束。

多维度 reward 的写法与约定（`emitter/reward.py:179-206`）：primary_key 指定的那维排第一，**读取时取列表第一个作为 scalar reward**（`emitter/reward.py:218-222`：`return reward_list[0].value`）。

**（7）第三条通路：proxy 侧记录。** LLMProxy 是 LiteLLM 起的独立服务，挂了一组中间件（`llm_proxy.py:536-620`），其中 `RolloutAttemptMiddleware` 保留了"记录轨迹"的必要上下文（详见第 8 节）：

```python
class RolloutAttemptMiddleware(BaseHTTPMiddleware):
    """Rewrites /rollout/{rid}/attempt/{aid}/... -> /...
    and injects x-rollout-id, x-attempt-id, x-sequence-id headers."""
    async def dispatch(self, request, call_next):
        path = request.url.path
        match = re.match(r"^/rollout/([^/]+)/attempt/([^/]+)(/.*)?$", path)
        if match:
            rollout_id, attempt_id = match.group(1), match.group(2)
            new_path = match.group(3) if match.group(3) is not None else "/"
            request.scope["path"] = new_path                  # 重写路径，下游看到干净的 OpenAI 路径
            store = get_active_llm_proxy().get_store()
            if store is not None:
                sequence_id = await store.get_next_span_sequence_id(rollout_id, attempt_id)
                request.scope["headers"] = list(request.scope["headers"]) + [
                    (b"x-rollout-id", rollout_id.encode()),
                    (b"x-attempt-id", attempt_id.encode()),
                    (b"x-sequence-id", str(sequence_id).encode()),
                ]
        return await call_next(request)
```

URL 的 `/rollout/{rid}/attempt/{aid}` 前缀是 `ProxyLLM.get_base_url()` 拼出来的（`types/resources.py:110-143`）：先去掉尾部 `/v1`，拼上 `/rollout/{rollout_id}/attempt/{attempt_id}`，再把 `/v1` 加回去——**这就是"每个 rollout 一个唯一端点"的实现**，proxy 靠它把请求归属到具体 attempt，并顺手分配 sequence id。

**（8）三路信号最终在哪落地：`LightningSpanProcessor.on_end()`。** 所有通路产生的 span 在"结束时"进同一个处理器（`tracer/otel.py:472-519`）：

```python
STORE_WRITE_TIMEOUT_SECONDS = 10.0

def on_end(self, span: ReadableSpan) -> None:
    if not span.context or not span.context.trace_flags.sampled:   # 未采样的 span 直接丢
        return

    if not self._disable_store_submission and self._store and self._rollout_id and self._attempt_id:
        try:
            with suppress_instrumentation():        # 别让"上传 span"这个动作自己又产生 span
                self._ensure_loop()
                uploaded_span = self._await_in_loop(
                    self._store.add_otel_span(self._rollout_id, self._attempt_id, span),
                    timeout=STORE_WRITE_TIMEOUT_SECONDS,
                )
                if uploaded_span is not None:
                    self._spans.append(uploaded_span)
        except TimeoutError:
            logger.warning("Timed out adding span %s to store after %.1f seconds. ...", span.name, ...)
            self._spans.append(Span.from_opentelemetry(span, ..., sequence_id=self._local_sequence_id))
        except Exception:
            logger.exception(f"Error adding span to store: {span.name}. The span will be store locally only.")
```

**四个关键设计**：① `on_end` 是**同步**接口（OTel SDK 在业务线程里调它），但 `add_otel_span` 是协程——于是 `LightningSpanProcessor` **自建一个名为 `otel-loop` 的守护线程跑私有 event loop**，用 `asyncio.run_coroutine_threadsafe(...).result(timeout=10)` 把协程同步化；② `suppress_instrumentation()` 防止"上传动作"被插桩后无限递归；③ 超时或异常时**只落本地 `_spans`**，不对调用方抛异常（注释：`on_end MUST NOT raise`）——代价是 **store 里永久缺这条 span**（第 11 节展开）；④ 提供一个"自死锁"保护（`tracer/otel.py:401-433`）：如果 `on_end` 恰好跑在 `otel-loop` 线程自己身上（GC 在 loop 线程里触发 `__del__` 的罕见情形），就改用 `call_soon_threadsafe` 发后不理，否则 `fut.result()` 会永久卡死。

### 3. 具体数值样例

假设一个 SQL agent 一次 rollout：**3 轮 LLM 调用 + 2 次工具执行 + 1 个 reward**，逐条演算 span 是怎么被造出来的：

```text
进入 trace_context(name=rollout_id, rollout_id="r-1", attempt_id="a-1")
  → OtelTracer.trace_context 里做两件事：
     · 若 store.capabilities["otlp_traces"] == True → _enable_native_otlp_exporter()
     · self._lightning_span_processor.with_context(store, "r-1", "a-1")   ← 给后续所有 span 打上归属

第 1 轮 LLM 调用（agent 调 openai chat completions，走 proxy）：
  · proxy 的 RolloutAttemptMiddleware 命中 /rollout/r-1/attempt/a-1/v1/chat/completions
    → 重写为 /v1/chat/completions，并分配 x-sequence-id=1，注入 3 个 header
  · LiteLLM 的 OTel 集成开 span: name="openai.chat.completion", parent=root
  · agentops 的 handle_chat_attributes 被 AGL 补丁替代 → 往属性里塞：
      prompt_token_ids   = [128000, 9707, ...]         (512 个 int)
      response_token_ids = [2675, 527, ...]            (180 个 int)
      logprobs.content   = '[{"token":"SELECT", ...}]' (180 条，JSON 字符串)
  · span.end() → LightningSpanProcessor.on_end()
      → 私有 otel-loop 线程执行 add_otel_span("r-1","a-1", span)  （10s 超时）
      → store 返回 span（带 sequence_id=1）→ 追加进本地 _spans

第 1 次工具执行（agent 自己调 execute_sql）：
  · 若 agent 用 @operation 装饰器包装 → operation_context 开一个 span
      name="agentlightning.operation", attributes["agentlightning.operation.name"]="execute_sql"
  · 若 agent 什么都不做 → 这个动作不会被记录（插桩只覆盖被 hook 的库）

第 2、3 轮 LLM 调用：同上，sequence_id = 2、3

reward 上报：
  · agent 代码显式调用：emit_reward(1.0)
  · emit_annotation → flatten → {"agentlightning.reward.0.value": 1.0}
  · tracer.create_span(name="agentlightning.annotation", attributes={...})
      → on_end → add_otel_span → sequence_id = 4

trace_context 退出：
  · _lightning_span_processor.__exit__ 清空 store/rollout_id/attempt_id
  · clear_active_tracer()

最终 store 里这个 attempt 有 5~6 条 span，sequence_id 1..4（+ 工具 span 若有）
```

**三个量化结论**：① **不是所有 agent 动作都能被记录**——插桩只覆盖被 hook 的库（openai/litellm/langchain/vllm）和显式 `@operation` 的代码，普通 Python 函数调用不会被记录；② **每条 span 一次 HTTP/一次 store 调用**，10s 是硬超时；③ `prompt_token_ids` 有 512 个整数，`logprobs.content` 是 180 条 JSON——**一条 LLM span 的属性体积远超 span 本身的结构字段**，这是第 8 节讨论传输开销的前提。

> **面试一句话总结**：Agent Lightning 的轨迹记录靠三条通路——① **函数插桩**（`AgentOpsTracer` → `instrument_all()` monkey-patch agentops 的 `handle_chat_attributes`，把 vLLM 的 `prompt_token_ids`/`response_token_ids`/`logprobs` 追加进 span attributes，并把 agentops 的 exporter 换成 `Bypassable*` 实现**纯本地、不发 agentops 云端**）；② **主动 emitter**（`emit_reward` → `emit_annotation` → `flatten_attributes` 展平成 `agentlightning.reward.0.value` → `get_active_tracer().create_span()`，因此**必须在 `trace_context` 内**，否则 `RuntimeError: No active tracer found`）；③ **proxy 侧记录**（`RolloutAttemptMiddleware` 重写 `/rollout/{rid}/attempt/{aid}/...` 并注入 `x-rollout-id`/`x-attempt-id`/`x-sequence-id`）；三路 span 统一在 `LightningSpanProcessor.on_end()` 落地，由专属 `otel-loop` 线程把协程 `add_otel_span` 同步化，**10s 超时、异常只落本地不抛**；`OtelTracer` 只收 AGL 自身信号、`AgentOpsTracer` 才带第三方库插桩——这一条最容易被问倒。

---

## 7. Span 数据模型：一条轨迹在 store 里到底长什么样

### 1. 现有问题：原生的 `ReadableSpan` 不能直接跨机存储

OTel 的 `ReadableSpan` 是**内存对象**：它有 `Resource`、`SpanContext`、看起来像 tuple 的 attributes、`status` 是枚举对象、时间戳是**纳秒整数**，还挂着 `span_processor` 之类的运行时引用。直接 `json.dumps` 必然失败，跨机器传输更不可能。同时训练侧还需要三类 OTel 里**根本不存在**的字段：**属于哪个 rollout / 哪个 attempt / 在一堆 span 里排第几**。所以 Agent Lightning 必须定义自己的规范 Span 模型：既要能无损装下 OTel 的语义字段，又要能安全序列化跨机，还要能挂上业务路由字段。

### 2. 方法论：`Span` 模型与 `from_opentelemetry` 的逐字段映射

**（1）`Span` 的字段分四组**（`types/tracer.py:251-306`，`model_config = ConfigDict(extra="allow")`）：

| 组 | 字段 | 含义 |
|---|---|---|
| **业务路由** | `rollout_id` / `attempt_id` / `sequence_id` | 归属哪个 rollout、哪次 attempt、attempt 内第几个（排序依据） |
| **OTel 身份** | `trace_id` / `span_id` / `parent_id` | 一次 trace 的 ID、本 span 的 ID、父 span 的 ID（**建树就靠 parent_id**） |
| **OTel 核心** | `name` / `status` / `attributes` / `events` / `links` / `start_time` / `end_time` | span 名称、状态（UNSET/OK/ERROR）、属性字典、事件列表、链接列表、起止时间（**秒**，浮点） |
| **OTel 结构** | `context` / `parent` / `resource` | `SpanContext` / 父 `SpanContext` / `OtelResource`（含 schema_url） |
| **额外字段** | `extra="allow"` | 所有未建模的 OTel 字段经 JSON 化后原样保留 |

**注意 `trace_id` 的注释**（`types/tracer.py:270`）：`# one rollout can have traces coming from multiple places`——**一个 rollout 可以有多条 trace**（比如 agent 自己的 trace + proxy 产生的 trace），所以 `trace_id` 不能当"轨迹 ID"用，**真正的分组键是 `(rollout_id, attempt_id)`**，`TraceTree.from_spans` 也确实是按传入的 span 列表（一个 attempt 的 span）建树的。

**（2）转换函数 `Span.from_opentelemetry`**（`types/tracer.py:308-371`）：

```python
@classmethod
def from_opentelemetry(cls, src: ReadableSpan, rollout_id: str, attempt_id: str, sequence_id: int) -> "Span":
    context = src.get_span_context()
    trace_id = context.trace_id if context else 0
    span_id = context.span_id if context else 0
    return cls(
        rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id,
        trace_id=trace_api.format_trace_id(trace_id),                       # 32 位 hex
        span_id=trace_api.format_span_id(span_id),                          # 16 位 hex
        parent_id=(trace_api.format_span_id(src.parent.span_id) if src.parent else None),
        name=src.name,
        status=TraceStatus.from_opentelemetry(src.status),                  # 枚举 → "OK"/"ERROR" 字符串
        attributes=dict(src.attributes) if src.attributes else {},
        events=[Event.from_opentelemetry(e) for e in src.events] if src.events else [],
        links=[Link.from_opentelemetry(l) for l in src.links] if src.links else [],
        start_time=convert_timestamp(src.start_time),                       # 纳秒 → 秒
        end_time=convert_timestamp(src.end_time),
        context=SpanContext.from_opentelemetry(context) if context else None,
        parent=(SpanContext.from_opentelemetry(src.parent) if src.parent else None),
        resource=OtelResource.from_opentelemetry(src.resource),
        **extract_extra_fields(src, ["name", "context", "parent", "resource", ...]),   # 未建模字段兜底
    )
```

**这段代码关键在哪**：① **三类 ID 全部格式化成 hex 字符串**（`trace_api.format_trace_id` / `format_span_id`），这样跨进程/跨机比较和 JSON 序列化都无损；② `parent_id` 只在 `src.parent` 存在时才有值，**它就是后面建树的唯一凭据**；③ 时间戳统一过 `convert_timestamp` 变成秒（浮点）；④ `extract_extra_fields` 是**向前兼容的兜底**：把 `ReadableSpan.__dict__` 里所有未建模字段 `json.dumps(default=str)` 再 `json.loads` 回来，保证"OTel 升级新增字段也不会丢"，同时保证"没建模也能序列化"。

`convert_timestamp` 的启发式值得单独记（`types/tracer.py:42-53`）：

```python
def convert_timestamp(timestamp: Optional[int]) -> Optional[float]:
    if not timestamp:
        return None
    return timestamp / 1_000_000_000 if timestamp > 1e12 else timestamp
```

**用"是否大于 1e12"来区分"纳秒"和"秒"**——1e12 秒是公元 33658 年，所以现实的秒级时间戳绝不会超过它，纳秒级则必然超过。这是个很典型的手写 heuristic。

**（3）三个构造入口，对应三条来源**：

| 构造函数 | 用途 | 谁会调 |
|---|---|---|
| `from_opentelemetry(src, rollout_id, attempt_id, sequence_id)` | 从真实 OTel span 转换 | `add_otel_span`（三路记录通路的落点） |
| `from_attributes(attributes, ...)` | 从裸属性**合成** span（自动生成随机 trace_id/span_id，默认 name=`agentlightning.virtual`） | `TraceTree.from_spans` 造虚拟父节点；OTLP 路径解析 |
| `from_core_fields(core, ...)` | 从 `SpanCoreFields` 构造 | emitter 返回的字段 → 真正的 span |

**（4）命名常量表**（`semconv.py`，面试可以背下来判断 span 类型）：

| 常量 | 值 | 含义 |
|---|---|---|
| `AGL_ANNOTATION` | `agentlightning.annotation` | annotation span（reward/tag/metadata 的载体） |
| `AGL_OPERATION` | `agentlightning.operation` | operation span（`@operation` 装饰的范围） |
| `AGL_REWARD` | `agentlightning.reward` | reward 的**属性前缀**（`agentlightning.reward.0.value`） |
| `AGL_MESSAGE` / `AGL_OBJECT` / `AGL_EXCEPTION` / `AGL_VIRTUAL` | `agentlightning.message` / `.object` / `.exception` / `.virtual` | 对应 emitter 与虚拟节点 |
| `LightningResourceAttributes.ROLLOUT_ID` | `agentlightning.rollout_id` | **resource 级**属性（OTLP 路由用） |
| `LightningResourceAttributes.ATTEMPT_ID` | `agentlightning.attempt_id` | 同上 |
| `LightningResourceAttributes.SPAN_SEQUENCE_ID` | `agentlightning.span_sequence_id` | 同上（十进制字符串） |

### 3. 具体数值样例

一个 LLM 调用 span 转成 `Span` 后的**真实结构**（字段名与类型都来自源码，具体值做了脱敏）：

```json
{
  "rollout_id": "r-7f3a1c",
  "attempt_id": "a-0b92e4",
  "sequence_id": 2,
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "parent_id": "9c1e4d2ab5f70c31",
  "name": "openai.chat.completion",
  "status": {"status_code": "OK", "description": null},
  "attributes": {
    "gen_ai.request.model": "Qwen2.5-1.5B-Instruct",
    "gen_ai.request.temperature": 0.7,
    "gen_ai.response.id": "chatcmpl-9x2K",
    "prompt_token_ids": [128000, 9707, 2675, "... 512 个"],
    "response_token_ids": [2675, 527, 1401, "... 180 个"],
    "logprobs.content": "[{\"token\":\"SELECT\",\"logprob\":-0.02}, ... 180 条]"
  },
  "events": [{"name": "exception", "attributes": {}, "timestamp": 1731250000.12}],
  "links": [],
  "start_time": 1731250000.10,
  "end_time": 1731250001.85,
  "context": {"trace_id": "4bf9...", "span_id": "00f0...", "is_remote": false, "trace_state": {}},
  "parent":  {"trace_id": "4bf9...", "span_id": "9c1e...", "is_remote": false, "trace_state": {}},
  "resource": {"attributes": {"service.name": "agentlightning"}, "schema_url": ""}
}
```

**逐字段解读三个面试常问点**：

```text
① 为什么 sequence_id=2 而 span_id 是一串乱码？
   span_id 是 OTel 随机生成的（RandomIdGenerator），用于表达"谁是谁的父节点"；
   sequence_id 是 store 分配的单调整数，用于表达"谁先谁后"。
   分布式的核心矛盾是"时钟不可信"，所以顺序必须靠序号而不是时间戳。

② 为什么既存 parent_id 又存 parent（SpanContext）？
   parent_id 是"建树用的轻量键"（TraceTree 只用它）；
   parent 是完整的父上下文（含 trace_id / trace_state），给需要跨 trace 关联的场景用。
   OTLP 协议里只有 parent_span_id，所以从 OTLP 解析时 parent 一律为 None（utils/otlp.py:212 注释）。

③ 为什么 attributes 里会混着 "logprobs.content" 这种 JSON 字符串？
   因为 OTel 的属性值只能是 string/bool/int/float 及其序列（AttributeValue 联合类型，
   types/tracer.py:76-85），嵌套结构必须自己序列化成字符串；
   AGL 因此成对提供了 flatten_attributes（写）/_attributes_unflatten_multiple（读）。
```

**存储侧的一个小但重要的实现细节**：`add_span` 会顺手把 store 的序号计数器**抬到不小于这个 span 的 sequence_id**（`store/collection_based.py:1056-1065` → `_sync_span_sequence_id` → `span_sequence_ids.chmax`）。作用是：**外部传入的乱序/较大序号不会导致后续分配重号**。

> **面试一句话总结**：Agent Lightning 用自研的 Pydantic `Span` 模型（`types/tracer.py:251`）替代 OTel 的 `ReadableSpan`，把字段分成四组——**业务路由**（`rollout_id`/`attempt_id`/`sequence_id`）、**OTel 身份**（`trace_id`/`span_id`/`parent_id`，全部格式化成 hex 字符串）、**OTel 核心与结构**（name/status/attributes/events/links/时间戳→秒/context/parent/resource）、**`extra="allow"` 兜底**（未建模字段 JSON 化保留，保证 OTel 升级不丢数据）；`from_opentelemetry` 负责逐字段映射，`convert_timestamp` 用 `> 1e12` 启发式区分纳秒与秒；`trace_id` **不是**轨迹分组键（一个 rollout 可能有多条 trace），真正的分组键是 `(rollout_id, attempt_id)`，建树唯一依赖 `parent_id`；三个构造入口 `from_opentelemetry` / `from_attributes`（合成虚拟节点）/ `from_core_fields`（emitter）分别对应三条来源。

---

## 8. HTTP 传输：Span 怎么从 agent 机器跨到 store 机器

### 1. 现有问题：轨迹是"流"，而 HTTP 是"一问一答"

三机分离部署下（agent 机器 → store 机器），轨迹上传有三个矛盾：① **span 是边跑边产生的流**，如果每条都同步阻塞 agent 线程，agent 会被网络拖慢；② **parent 和 child 不是同时结束的**，如果 parent 先结束就先上传，到达 store 的顺序天然是乱的；③ **单条 span 可能很大**（上一节的例子：512 个 prompt token id + 180 条 logprob 全在 attributes 里），逐条传的序列化和请求开销会被放大。所以 Agent Lightning 准备了**两条正式通路 + 一条 proxy 专线**，并按 store 的能力自动选路。

### 2. 方法论：三条传输路径与"按能力选路"机制

**（0）选路开关：`store.capabilities["otlp_traces"]`。** 各实现的声明（源码位置见括号）：

| store 实现 | thread_safe | async_safe | zero_copy | **otlp_traces** | 结果 |
|---|---|---|---|---|---|
| `InMemoryLightningStore`（`store/memory.py:149-153`） | 可配 | True | False | **False** | 走**逐条 JSON** |
| `MongoLightningStore`（`store/mongo.py:83-87`） | True | True | True | **False** | 走**逐条 JSON** |
| `LightningStoreServer`（`store/client_server.py:313-318`） | True | True | True | **True** | 走 **OTLP 批量** |
| `LightningStoreClient`（`store/client_server.py:1377-1388`） | True | True | True | **True** | 走 **OTLP 批量** |

**结论（对我们的三机分离部署很关键）**：store 机器上跑的是 `LightningStoreServer`，algo/agent 机器注入的是 `LightningStoreClient`，**两侧都声明 `otlp_traces=True`**，所以实际走的是 **OTLP protobuf 批量上传**。而**单机 shared-memory 调试（InMemory）反而走逐条 JSON**——和直觉相反。

选路代码在 `OtelTracer.trace_context`（`tracer/otel.py:157-173`）：

```python
if rollout_id is not None and attempt_id is not None:
    if store.capabilities.get("otlp_traces", False) is True:
        self._enable_native_otlp_exporter(store, rollout_id, attempt_id)   # 批量路
    else:
        self._disable_native_otlp_exporter()                              # 逐条路
    ctx = self._lightning_span_processor.with_context(store=store, rollout_id=rollout_id, attempt_id=attempt_id)
    with ctx:
        yield trace_api.get_tracer(__name__, tracer_provider=self._tracer_provider)
```

**注意 `_enable_native_otlp_exporter` 有个副作用**：它会把 `LightningSpanProcessor` **关掉**（`processor.disable_store_submission = True`，`tracer/otel.py:269`），并把 rollout/attempt **写进 TracerProvider 的 `_resource`**（`tracer/otel.py:255-262`）——因为既然走 OTLP 批量路，就不需要逐条上传了，归属信息改由 resource 属性携带。

**（1）OTLP 批量路（ftrace）：一次 POST 传整批 span。**

**出口侧**：`LightningStoreOTLPExporter` 继承官方 `OTLPSpanExporter`，只做一件事——**把 rollout_id / attempt_id 合并进每个 span 的 resource**（`utils/otlp.py:295-315`）：

```python
def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
    if self._rollout_id is not None and self._attempt_id is not None:
        for span in spans:
            span._resource = span._resource.merge(
                Resource.create({
                    LightningResourceAttributes.ROLLOUT_ID.value: self._rollout_id,
                    LightningResourceAttributes.ATTEMPT_ID.value: self._attempt_id,
                })
            )
        return super().export(spans)      # 官方 exporter：protobuf 序列化 + HTTP POST
    elif not self.should_bypass():
        return super().export(spans)      # 兜底：按普通 OTLP 行为
    else:
        return SpanExportResult.SUCCESS   # should_bypass() 默认 True → 直接吞掉
```

**`should_bypass()` 返回 `True`**（`utils/otlp.py:291-293`）意味着：**如果没有 rollout/attempt（即不是在为 store 上传），就什么都不发**——这正是"AGL 借 agentops 的插桩但不往 agentops 云端发数据"在传输层的最后一道阀门。

**入口侧**：server 的 `/v1/traces` 路由（`store/client_server.py:940-955`）：

```python
def _setup_otlp(self, api: APIRouter):
    async def _trace_handler(request: PbExportTraceServiceRequest) -> None:
        spans = await spans_from_proto(request, self.get_many_span_sequence_ids)
        await self.add_many_spans(spans)

    @api.post("/traces")
    async def otlp_traces(request: Request):
        return await handle_otlp_export(
            request, PbExportTraceServiceRequest, PbExportTraceServiceResponse, _trace_handler, "traces"
        )
```

路由前缀是 `API_V1_PREFIX` + `/traces`，所以完整路径是 **`POST /v1/traces`**（`otlp_traces_endpoint()` 返回 `f"{self.endpoint}/v1/traces"`，`client_server.py:320-322`）。

**协议处理 `handle_otlp_export`**（`utils/otlp.py:56-110`）的四条硬约束：

1. **只支持二进制 protobuf**：`Content-Type` 必须是 `application/x-protobuf`，否则 400（注释：`For brevity we only support binary protobuf here`）；
2. **支持 gzip 双向压缩**：请求头 `Content-Encoding: gzip` 则解压（`_read_body_maybe_gzip`），响应若 `Accept-Encoding` 含 gzip 则压缩返回；
3. **空 body 也返回 200**（OTLP 规范允许）；
4. 400 响应的 body 也是 protobuf 编码的 `google.rpc.Status`（OTLP/HTTP 规范要求）。

**`spans_from_proto` 的解析与批量领号**（`utils/otlp.py:113-226`）——三层循环 `resource_spans → scope_spans → spans`，关键是**归属信息的优先级**：

```python
# Resource 级属性
resource_attrs = _kv_list_to_dict(resource_spans.resource.attributes)
rollout_id_resource = resource_attrs.get(LightningResourceAttributes.ROLLOUT_ID.value)
sequence_id_resource = resource_attrs.get(LightningResourceAttributes.SPAN_SEQUENCE_ID.value)
...
# Span 级属性可以覆盖 Resource 级（"Override the resource-level attributes with the span-level attributes"）
rollout_id_raw = rollout_id_span if rollout_id_span is not None else rollout_id_resource
sequence_id = _normalize_sequence_id(sequence_id_raw)
...
# 没有 sequence_id 的 span 先标记 -1，最后一次性批量领号
if sequence_id is None:
    current_sequence_id = -1
...
bulk_issue_requests = [(s.rollout_id, s.attempt_id) for s in output_spans if s.sequence_id < 0]
bulk_sequence_ids = await sequence_id_bulk_issuer(bulk_issue_requests)     # 一次原子调用领 N 个号
```

**这是 OTLP 路最大的性能优势**：整批 span 只做**一次**批量领号（`get_many_span_sequence_ids`），而不是每条 span 一次；相比之下逐条路每条都要先领号。另外**缺失 rollout/attempt 的 span 会被直接丢弃**（`utils/otlp.py:174-182`，`logger.warning` 后 `continue`）——所以 resource 属性没打上就等于 span 丢了。

**（2）逐条 JSON 路：每条 span 两次 HTTP。** 客户端实现（`store/client_server.py:1912-1928`）非常直白：

```python
async def add_otel_span(self, rollout_id, attempt_id, readable_span, sequence_id=None) -> Optional[Span]:
    # unchanged logic, now benefits from retries inside add_span/get_next_span_sequence_id
    if sequence_id is None:
        sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)   # 第 1 次 HTTP
    span = Span.from_opentelemetry(readable_span, rollout_id=rollout_id,
                                   attempt_id=attempt_id, sequence_id=sequence_id)
    return await self.add_span(span)                                                 # 第 2 次 HTTP
```

而 `add_span` 发的是 **Pydantic JSON**（`store/client_server.py:1885-1887`）：

```python
async def add_span(self, span: Span) -> Optional[Span]:
    data = await self._request_json("post", "/spans", json=span.model_dump(mode="json"))
    return Span.model_validate(data) if data is not None else None
```

落到 server 侧的路由是 **`POST /v1/agl/spans`**（`API_V1_PREFIX` + `API_AGL_PREFIX="/agl"` + `/spans`，`client_server.py:74,438,784-786`，状态码 201）。

**关键差别**：`ReadableSpan` **从来没有被跨机传输**——它在客户端本地就被 `Span.from_opentelemetry` 转成了可序列化的 `Span`，网络上跑的是 **`Span` 的 JSON**。这一点常被误解成"把 OTel 对象 pickle 过去"。

**（3）proxy 专线：子树缓冲批量导出。** LLMProxy 自己也是 span 生产者，它用的是另一个 exporter——`LightningSpanExporter`（`llm_proxy.py:196-497`），设计目标是**"等一棵子树完整了再整体发"**：

```python
class LightningSpanExporter(SpanExporter):
    """Buffered OTEL span exporter with subtree flushing and training-store sink.

    Design:
    * Spans are buffered until a root span's entire subtree is available.
    * A private event loop on a daemon thread runs async flush logic.
    * Rollout/attempt/sequence metadata is reconstructed by merging headers from any span within a subtree.
    """
```

它的四个动作：

```python
# ① 找根：parent is None 的就是根
def _get_root_span_ids(self) -> Iterable[int]:
    for span in self._buffer:
        if span.parent is None:
            span_context = span.get_span_context()
            if span_context is not None:
                yield span_context.span_id

# ② 深度优先收集整棵子树的 span_id
def _get_subtrees(self, root_span_id: int) -> Iterable[int]:
    yield root_span_id
    for span in self._buffer:
        if span.parent is not None and span.parent.span_id == root_span_id:
            yield from self._get_subtrees(span.get_span_context().span_id)

# ③ 从 buffer 里"整棵摘走"
def _pop_subtrees(self, root_span_id: int) -> List[ReadableSpan]:
    subtree_span_ids = set(self._get_subtrees(root_span_id))
    ...  # 分裂成 subtree_spans / new_buffer

# ④ 合并子树内任意 span 携带的 requester_custom_headers，取出三个 ID
headers_str = span.attributes.get("metadata.requester_custom_headers")   # 字符串化的 dict
headers = ast.literal_eval(headers_str)                                  # 安全解析
rollout_id  = headers_merged.get("x-rollout-id")
attempt_id  = headers_merged.get("x-attempt-id")
sequence_id = headers_merged.get("x-sequence-id")                        # 必须是数字字符串
```

**注意两个设计细节**：① 三个 ID 是**从 HTTP header 还原**的（`RolloutAttemptMiddleware` 注入 → LiteLLM 插桩把 headers 存成 `metadata.requester_custom_headers` 字符串 → 这里 `ast.literal_eval` 解析回来），所以**"从任意一个 span 里 merge 出来"就够了**，不要求根 span 上有；② 拿到 ID 后，如果 store 支持 OTLP，就给**整棵子树**的每个 span 合并 resource 然后**一次性 `export(subtree_spans)`**；否则退化成对子树里每条 span 调 `store.add_otel_span`（`llm_proxy.py:412-439`）。

**（4）客户端的可靠性设计：重试、健康探测、session 按 event loop 隔离。**

```python
def __init__(self, ..., retry_delays: Sequence[float] = (1.0, 2.0, 5.0),
             health_retry_delays: Sequence[float] = (0.1, 0.2, 0.5)):
    self._sessions: Dict[int, aiohttp.ClientSession] = {}   # id(loop) -> ClientSession
```

`_request_json` 的重试语义（`store/client_server.py:1485-1541`）：

```python
attempts = (0.0,) + self._retry_delays        # 立即试一次，然后按 1s/2s/5s 退避
for delay in attempts:
    ...
    async with http_call(url, json=json, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()
except aiohttp.ClientResponseError as cre:
    if 400 <= cre.status < 500 and cre.status != 408:
        raise                                  # 4xx（除 408）是应用层错误 → 不重试
    ...                                        # 5xx → 先探 /health 再重试
except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError,
        aiohttp.ClientOSError, asyncio.TimeoutError) as net_exc:
    if not await self._wait_until_healthy(session):
        break                                  # 服务器不健康 → 不再重试
```

**为什么要"按 event loop 缓存 session"**（`store/client_server.py:1426-1441` 的注释写得很清楚）：同一个 client 会从**多个线程的多个 event loop** 被调用（uvicorn 的 loop、otel-loop、LightningSpanExporterLoop），而 `aiohttp.ClientSession` **不是 loop-agnostic 也不线程安全**，"Using it from another loop can hang on the first request"——所以用 `id(loop)` 做 key 各存一个 session。

**（5）server 侧的透明路由：`_call_store_method`。** 这是"同一套接口，同进程直调、跨进程走 HTTP"的实现（`store/client_server.py:1010-1047`）：

```python
async def _call_store_method(self, method_name: str, *args, **kwargs) -> Any:
    if self.store is not None and self.store.capabilities.get("zero_copy", False):
        return await getattr(self.store, method_name)(*args, **kwargs)     # zero-copy：直接调
    if os.getpid() == self._owner_pid:                                      # owner 进程：本地调
        if method_name == "wait_for_rollouts":
            return await getattr(self.store, method_name)(*args, **kwargs)  # 长阻塞，不持锁
        if self.store is not None and self.store.capabilities.get("thread_safe", False):
            return await getattr(self.store, method_name)(*args, **kwargs)  # 线程安全：不加锁
        else:
            with self._lock:
                return await getattr(self.store, method_name)(*args, **kwargs)
    if self._client is None:
        self._client = LightningStoreClient(self.endpoint)                  # 非 owner：建 HTTP client
    return await getattr(self._client, method_name)(*args, **kwargs)
```

**这段代码关键在哪**：Algorithm 和 Runner 调的是**同一个方法名**，`Server` 自己按"是不是 owner 进程 + store 能力"决定走本地调用还是 HTTP——**调用方完全无感**，这就是"三机分离不用改一行业务代码"的代码基础。

### 3. 具体数值样例

同一次 rollout（3 轮 LLM + 1 个 reward，共 4 条 span），对比两条通路：

```text
【逐条 JSON 路】（InMemory store / 单机 shared-memory）
每条 span 2 次 HTTPS：
  · POST /v1/agl/spans/next   ← 领 sequence_id（请求体 {"rollout_id","attempt_id"}，响应 {"sequence_id":N}）
  · POST /v1/agl/spans        ← 提交 Span 的 Pydantic JSON
4 条 span → 8 次 HTTP 请求
序列化：每条 span 做 1 次 model_dump(mode="json") + 1 次 model_validate
单条 span 的 JSON 体积粗估（按字段长度推算，非实测）：
  · response_token_ids 180 个 int，JSON 里每个约 6 字节        ≈ 1.1 KB
  · prompt_token_ids  512 个 int                              ≈ 3.1 KB
  · logprobs.content  180 条，每条约 45 字节 JSON              ≈ 8.1 KB
  · messages 原文（prompt + completion）                       ≈ 2~3 KB
  · 结构字段（id/status/timestamps/context/resource/events）    ≈ 1 KB
  ⇒ 一条带 logprobs 的 LLM span ≈ 15 KB 量级；不带 logprobs ≈ 6 KB
4 条 span ≈ 30~40 KB JSON

【OTLP 批量路】（Server/Client，即我们的三机分离部署）
一次 trace 生命周期内的 span 由 BatchSpanProcessor 攒批，整体 POST 一次：
  · POST /v1/traces   Content-Type: application/x-protobuf（可再 gzip）
  ⇒ 4 条 span = 1 次 HTTP 请求
序列化：protobuf（比 JSON 紧凑，且 int 数组有原生 packed encoding）
领号：整批一次 get_many_span_sequence_ids（1 次原子 inc(rollout_id, 4)）而非 4 次
server 侧：spans_from_proto → add_many_spans（内部主键去重 + 失败退化为逐条）
```

**量化的三条结论**：① **OTLP 路把 HTTP 请求数从 $2N$ 降到 $1$**（N 条 span），并且序列化从 JSON 换成 protobuf；② **span 体积几乎完全由 attributes 决定**（token ids + logprobs 占了 80% 以上），所以"轨迹带宽"本质上等于"token 数量 × 常数"，这直接决定了双机全异步训练里网络上要跑多少数据；③ `wait_for_rollouts` 在 client 侧有个刻意限制（`client_server.py:1940-1943`）：

```python
if timeout is not None and timeout > 0.1:
    raise ValueError("Timeout must be less than 0.1 seconds in LightningStoreClient to avoid blocking the event loop")
```

**即 client 侧的 `wait_for_rollouts` 单次等待不允许超过 0.1s**——这就是 `AgentModeDaemon` 必须自己写"轮询循环"（`timeout=0` 反复调）而不是"一次阻塞等到天荒地老"的原因。

> **面试一句话总结**：Agent Lightning 的 span 传输按 `store.capabilities["otlp_traces"]` 自动选路——**`LightningStoreServer`/`LightningStoreClient` 声明 True（我们的三机分离部署就走这条）**：出口用 `LightningStoreOTLPExporter` 把 rollout/attempt 合并进每个 span 的 resource，官方 `OTLPSpanExporter` 以 **protobuf（`Content-Type: application/x-protobuf`，支持 gzip）一次性 `POST /v1/traces`**，入口 `handle_otlp_export` → `spans_from_proto` 从 resource/span 属性还原归属并**整批一次领号**，再 `add_many_spans`；**`InMemoryLightningStore`/`MongoLightningStore` 声明 False**，退化为**每条 span 两次 HTTP**（`POST /v1/agl/spans/next` 领号 + `POST /v1/agl/spans` 提交 `Span` 的 Pydantic JSON，**`ReadableSpan` 从不跨机**）；proxy 另有专线 `LightningSpanExporter`——按 `parent is None` 找根、DFS 收集子树、**整棵子树攒齐才导出**，三个 ID 从 HTTP header（`x-rollout-id`/`x-attempt-id`/`x-sequence-id`）经 `metadata.requester_custom_headers` + `ast.literal_eval` 还原；客户端重试用 `(1.0,2.0,5.0)` 退避、4xx（除 408）不重试、5xx/网络错先探 `/health`，且 `aiohttp.ClientSession` **按 `id(loop)` 一 loop 一个**避免跨 loop 挂死。

---

## 9. Span 树是怎么构建出来的：从扁平列表到 `TraceTree`

### 1. 现有问题：store 里只有一张扁平的 span 表

`query_spans(rollout_id, attempt_id)` 返回的是**一个列表**，按 `sequence_id` 排序。但训练需要的语义是**树形**的："这次 LLM 调用属于哪个 agent？这个 reward 是给哪一步的？"——扁平列表回答不了。更麻烦的是**分布式现实会让树"断"**：

- **顶层 session span 可能丢失**（比如 agentops 的 session span 没被上传成功），于是子 span 的 `parent_id` 指向一个**列表里不存在**的 ID；
- **混合插桩系统会导致层级错乱**：一个 agent 同时用 openai 插桩和 langchain 回调，LLM span 可能**直接挂在 root 上**而不是挂在所属 agent 下面（源码原话见下）；
- **一次 attempt 可能有多个根**（agent 自己一个 trace + proxy 另一个 trace），而下游要的是"一棵树"。

所以必须有一套**容错建树算法**：先按 `parent_id` 建图，再补丢失的父节点，再把错位的节点按时间重新挂载。

### 2. 方法论：`TraceTree.from_spans` 的五步算法 + `repair_hierarchy` 重挂

**（1）数据结构**（`adapter/triplet.py:99-136`）：`TraceTree` 就是一个三元组 `{id, span, children}`，`id` 直接复用 `span.span_id`；`start_time` / `end_time` 是 `span` 的透传属性（**重挂父节点要靠它们**）。

**（2）`from_spans` 的五步**（`adapter/triplet.py:225-315`）：

```python
@classmethod
def from_spans(cls, spans: List[Span]) -> "TraceTree":
    if not spans:
        raise ValueError("No spans provided to create TraceTree.")

    # 第 1 步：按 span_id 建索引
    id_to_span = {span.span_id: span for span in spans}

    # 第 2 步：建"父 → 子列表"前向图，同时收集根
    forward_graph: dict[str, list[str]] = {}
    root_ids: list[str] = []
    for span in spans:
        span_id = span.span_id
        if span.parent_id is None:
            root_ids.append(span.span_id)          # 显式根：parent 为空
        else:
            if span.parent_id not in forward_graph:
                forward_graph[span.parent_id] = []
            forward_graph[span.parent_id].append(span_id)

    # 第 3 步：容错——父节点不在 span 列表里时，把它也当根
    # "Sometimes the top-level session span is lost."
    unfound_roots = set(forward_graph.keys()) - set(id_to_span.keys())
    for unfound_root in unfound_roots:
        root_ids.append(unfound_root)

    # 第 4 步：后序递归建树；缺失节点用 from_attributes 合成"虚拟 span"
    def visit(node_id: str) -> "TraceTree":
        children: list[TraceTree] = []
        if node_id in forward_graph:
            for child_id in forward_graph[node_id]:
                children.append(visit(child_id))

        if node_id not in id_to_span:
            assert len(children) > 0
            virtual_span = Span.from_attributes(       # name 默认 = AGL_VIRTUAL
                rollout_id=children[0].span.rollout_id,
                attempt_id=children[0].span.attempt_id,
                sequence_id=children[0].span.sequence_id,
                trace_id=children[0].span.trace_id,
                span_id=node_id,                       # 复用"幽灵父节点"的 id
                parent_id=None, attributes={},
                start_time=min(c.start_time for c in children if c.start_time is not None),
                end_time=max(c.end_time for c in children if c.end_time is not None),
            )
            return cls(node_id, virtual_span, children=children)
        else:
            return cls(node_id, id_to_span[node_id], children=children)

    # 第 5 步：多于一个根 → 再造一个 "virtual-root" 把所有根收进来
    if len(root_ids) > 1:
        root_spans = [visit(root_id) for root_id in root_ids]
        virtual_root = TraceTree(id="virtual-root", span=Span.from_attributes(..., name="virtual-root", ...),
                                children=root_spans)
        return virtual_root
    elif len(root_ids) == 0:
        raise ValueError("No root spans found in the trace.")
    else:
        return visit(root_ids[0])
```

**五个设计点，每个都能单独出面试题**：

| 步骤 | 设计 | 为什么必须这样 |
|---|---|---|
| 2 | 用**前向图**（`parent → [children]`）而不是后向指针 | 自顶向下递归建树需要 O(1) 取子节点；后向 `parent_id` 在 `find_llm_calls` 等查询里用不到 |
| 3 | `unfound_roots = forward_graph.keys() - id_to_span.keys()` | **父 span 丢失时不能直接丢子树**，必须把"幽灵父节点"提升为根，否则整条轨迹消失 |
| 4 | `visit` 是**后序**（先递归子节点再建自己） | 合成虚拟 span 需要知道子节点的 `start_time/end_time` 才能取 min/max；也就是"自底向上补时间" |
| 4 | 虚拟 span 用 `Span.from_attributes` 生成 | 它在 store 里**并不存在**，只是为了保持树结构完整；name 默认 `agentlightning.virtual`，`attributes={}` |
| 5 | 多根时套 `virtual-root` | 下游 `find_llm_calls`/`match_rewards` 都假设"有一个树根可以开始遍历" |

**（3）`repair_hierarchy()`：按"时间包含关系"重挂错位节点**（`adapter/triplet.py:478-520`）。源码 docstring 把动机说得很明白：

```python
def repair_hierarchy(self) -> None:
    """Repair missing parent-child relationships introduced by mixed tracing systems.

    Some agent frameworks emit spans via multiple subsystems, which can cause LLM completion
    spans to float directly under the root span instead of being nested under the correct agent.
    The method re-parents those spans to the closest ancestor that fully envelopes the child in
    time.

    If we don't, when we want to select the LLM completion span with agent as filter.
    We will never get the correct span underneath.
    """
```

算法本身很短：

```python
# 只有一个孩子时，递归修它自己就返回
# （因为 agentops.end_trace 会把所有 span 额外包一层合成根，例如 "run_one.session"）
if len(self.children) == 1:
    self.children[0].repair_hierarchy()
    return

nodes_to_repair = list(self.children)
for repair_node in nodes_to_repair:
    if len(self.children) == 1:
        break
    closest_parent = None
    closest_duration = float("inf")
    for node in self.traverse():
        if node.id == repair_node.id or node is self:
            continue
        # 候选条件：node 在时间上"完全包住" repair_node
        if node.start_time <= repair_node.start_time and node.end_time >= repair_node.end_time:
            duration_delta = node.end_time - repair_node.end_time + repair_node.start_time - node.start_time
            if duration_delta > 0 and duration_delta < closest_duration:
                closest_duration = duration_delta          # 取"包得最紧"的那个
                closest_parent = node
    if closest_parent is not None:
        self.children.remove(repair_node)
        closest_parent.children.append(repair_node)
```

注意 `duration_delta = (node.end - repair.end) + (repair.start - node.start)` 正好等于**父节点时长 − 子节点时长**，所以"`duration_delta` 最小的正数"= **时间上包得最紧的那个祖先**（`> 0` 排除了时长完全相等的退化情况）。

**（4）配套的查询与可视化工具**（都是后续 step 的依赖）：

| 方法 | 作用 |
|---|---|
| `traverse()` | 深度优先遍历返回所有节点（`match_rewards`/`repair_hierarchy` 依赖） |
| `find_id(id)` | 按 id 在子树里查找 |
| `agent_name()` | **从属性识别"这是哪个 agent 的 span"**（7 种框架规则，见下表） |
| `is_reward_span()` / `maybe_reward_dict()` | 判断这条 span 是否携带 reward |
| `find_llm_calls(...)` | 找出符合条件的 LLM 调用 span（第 10 节） |
| `names_tuple()` / `visualize()` / `to_json()` | 调试：嵌套名字元组 / Graphviz 画图（`dot.render(..., format="png")`）/ JSON 序列化 |

**`agent_name()` 的 7 条识别规则**（`adapter/triplet.py:317-365`）——这是"混合框架下如何定位 agent 子树"的答案，也是面试很好的加分项：

| 来源框架 | 判定依据（属性名） |
|---|---|
| OpenAI Agents SDK | `agent.name` |
| AgentOps `@agent` 装饰器 | `agentops.span.kind == "agent"` → 取 `operation.name` |
| Autogen team | `recipient_agent_type` |
| LangGraph | `langchain.chain.type` |
| agent-framework | `executor.id` |
| Weave | `type == "agent"` → 取 `agentlightning.operation.input.name` |
| Weave + LangChain | span 名以 `langchain.Chain.` 开头 → 取 `lc_name` |

### 3. 具体数值样例

**给 6 条 span（其中 1 条父节点丢失、1 条 reward），逐步演算建树过程**：

```text
输入 spans（同一 rollout="r-1" / attempt="a-1"）：
  s1_root   name="agent.run"              parent=None   [0.0, 10.0]
  s2_agent  name="agentops.agent"         parent=s1     [0.5,  9.5]  attr: operation.name="SqlAgent"
  s3_llm1   name="openai.chat.completion" parent=s2     [1.0,  2.5]  attr: gen_ai.response.id="resp-1"
  s4_tool   name="tool.execute_sql"       parent=s2     [2.6,  3.0]
  s5_llm2   name="openai.chat.completion" parent=sGHOST [3.2,  5.0]  attr: gen_ai.response.id="resp-2"
  s6_reward name="agentlightning.annotation" parent=s2  [5.1,  5.2]  attr: agentlightning.reward.0.value=1.0
  （sGHOST 不在列表里 —— 模拟"顶层 session span 丢失"）

第 1 步：id_to_span = {s1, s2, s3, s4, s5, s6}（6 项）

第 2 步：forward_graph 与 root_ids
  s1.parent is None           → root_ids = [s1]
  s2.parent=s1 → forward_graph[s1] = [s2]
  s3.parent=s2 → forward_graph[s2] = [s3]
  s4.parent=s2 → forward_graph[s2] = [s3, s4]
  s5.parent=sGHOST → forward_graph[sGHOST] = [s5]
  s6.parent=s2 → forward_graph[s2] = [s3, s4, s6]
  ⇒ forward_graph = { s1:[s2], s2:[s3,s4,s6], sGHOST:[s5] }

第 3 步：unfound_roots = {s1, s2, sGHOST} − {s1..s6} = {sGHOST}
  ⇒ root_ids = [s1, sGHOST]   ← 幽灵父节点被提升为根，s5 的子树得救

第 4 步：后序 visit(s1) → visit(s2) → visit(s3)/visit(s4)/visit(s6) → 回填
        visit(sGHOST) → visit(s5) → 合成虚拟 span（时长 = s5 的 [3.2, 5.0]）

第 5 步：len(root_ids) == 2 → 造 "virtual-root"
  virtual-root [0.0, 5.0]（start=root_spans[0].start, end=root_spans[-1].end）
    ├── s1_root            [0.0, 10.0]
    │     └── s2_agent     [0.5,  9.5]   [SqlAgent]
    │           ├── s3_llm1  [1.0, 2.5]  (resp-1)
    │           ├── s4_tool  [2.6, 3.0]
    │           └── s6_reward[5.1, 5.2]  (reward=1.0)
    └── sGHOST(virtual)    [3.2, 5.0]    ← 错位：它其实是 SqlAgent 内部的 span
          └── s5_llm2      [3.2, 5.0]  (resp-2)

第 6 步：repair_hierarchy()（根有两个孩子，不早退）
  修 s1_root：找"包住 [0.0,10.0] 的最紧节点"（排除自己与根）
     候选 s2_agent [0.5,9.5] 不含 0.0；sGHOST [3.2,5.0] 不含 → closest_parent=None → 不动
  修 sGHOST [3.2,5.0]：候选
     s1_root   [0.0,10.0] 含 → delta = (10.0−5.0)+(3.2−0.0) = 8.2
     s2_agent  [0.5, 9.5] 含 → delta = ( 9.5−5.0)+(3.2−0.5) = 7.2   ← 最小
     s3_llm1   [1.0, 2.5] 不含（2.5 < 5.0）
     s4_tool   [2.6, 3.0] 不含
     s6_reward [5.1, 5.2] 不含
     ⇒ closest_parent = s2_agent → 把 sGHOST 从 virtual-root 摘下，挂到 s2_agent 下

最终树（repair 后）：
  virtual-root
    └── s1_root
          └── s2_agent [SqlAgent]
                ├── s3_llm1   (resp-1)
                ├── s4_tool
                ├── s6_reward (reward=1.0)
                └── sGHOST(virtual)
                      └── s5_llm2 (resp-2)
```

**三个结论**：① **丢父节点不会丢数据**（虚拟 span 补位，第 3、4 步）；② **错位的层级能被时间关系修回来**（第 6 步，`sGHOST` 从根下移到 `SqlAgent` 下）；③ 修好之后 `find_llm_calls(agent_match="SqlAgent")` 才能把 `s3_llm1` 和 `s5_llm2` **都**捞到——**如果跳过 `repair_hierarchy`，`s5_llm2` 会因为不在 `SqlAgent` 子树里而被过滤掉，这条轨迹就少了一个训练样本**。这就是 `TracerTraceToTriplet(repair_hierarchy=True)` 这个开关的实际价值（`adapter/triplet.py:775-776, 825-847`）。

> **面试一句话总结**：Span 在 store 里是**扁平列表**，树是在 Adapter 里现建的——`TraceTree.from_spans`（`adapter/triplet.py:225`）五步走：① 按 `span_id` 建索引；② 按 `parent_id` 建 `parent → [children]` 前向图并收集 `parent_id is None` 的根；③ **容错**——把"在 `forward_graph` 里当爹但自己不在 span 列表里"的 ID 也提升为根（"Sometimes the top-level session span is lost"）；④ **后序**递归建节点，缺失节点用 `Span.from_attributes` 合成"虚拟 span"（name=`agentlightning.virtual`，时间取子节点的 min/max）；⑤ 多根时再套一个 `virtual-root`；之后 `repair_hierarchy()` 用**时间包含关系**把因混合插桩而上浮的 span 重挂到"包得最紧的祖先"（`duration_delta = 父时长 − 子时长` 取最小正数），`agent_name()` 用 7 条框架规则（`agent.name` / `agentops.span.kind` / `recipient_agent_type` / `langchain.chain.type` / `executor.id` / Weave `type` / `langchain.Chain.` 前缀）定位 agent 子树——**不修层级就会漏掉挂在 root 下的 LLM span，直接少训练样本**。

---

## 10. 从 Span 树到训练样本：reward 匹配与 triplet 抽取

### 1. 现有问题：一次 rollout 一个 reward，但有多次 LLM 调用

RL 训练要的是 `(prompt, response, reward)` 三元组。但 agent 场景里**reward 是整条轨迹末尾给的一个标量**（SQL 是否跑通），而 LLM 调用有 3 次。三个必须回答的问题：① 哪些 span 算"可训练的 LLM 调用"（工具调用/内部规划不算）；② 同一个 reward 分给哪一次调用；③ 最终 reward 怎么落到 token 上。分错就等于给了错误的信用分配（credit assignment）。

### 2. 方法论：`find_llm_calls` → `span_to_triplet` → `match_rewards` → `to_trajectory`

**（1）`find_llm_calls`：四个过滤维度**（`adapter/triplet.py:398-476`）。递归遍历树，同时传递四个状态：

```python
def find_llm_calls(self, *, llm_call_match, agent_match,
                   within_matching_subtree=None, within_reward=None,
                   within_llm_call=None, existing_llm_call_response_ids=None):
    llm_calls = []
    is_llm_call = True
    if within_matching_subtree is None or within_reward is True:
        is_llm_call = False                                  # ① 必须在"匹配的 agent 子树"内，且不在 reward 子树内
    if re.search(llm_call_match, self.span.name) is None:
        is_llm_call = False                                  # ② span 名要匹配 llm_call_match 正则
    if is_llm_call:
        response_id = _attributes_get_multiple(
            self.span.attributes, ["gen_ai.response.id", "agentlightning.operation.output.id"])
        if response_id is None and within_llm_call is True:
            is_llm_call = False                              # ③ 嵌套 LLM 调用：无 response_id 的不重复计
        if (response_id is not None and existing_llm_call_response_ids is not None
                and response_id in existing_llm_call_response_ids):
            is_llm_call = False                              # ④ 同一个 response_id 只算一次（去重）
        if is_llm_call:
            llm_calls.append((self, within_matching_subtree))
            ...existing_llm_call_response_ids.add(response_id)
            within_llm_call = True
    ...
    for child in self.children:                              # 递归传状态
        llm_calls.extend(child.find_llm_calls(..., within_matching_subtree=within_matching_subtree, ...))
```

四道过滤的语义：**② 名字正则**（默认 `r"openai\.chat\.completion"`）、**① agent 子树**（`agent_match` 命中后 `within_matching_subtree=agent_name`，其子树内的 LLM 调用才被认领）、**③ 嵌套去重**（外层 LLM 调用内部的嵌套调用若无 `response_id` 就不再计）、**④ response_id 去重**（同一响应只留一条）。

**（2）`span_to_triplet`：把 span 属性摊成 triplet**（`adapter/triplet.py:631-700`）。token id 因为要兼容多个 tracer，用了**候选键列表**：

```python
prompt_token_ids = _attributes_get_ids_multiple(span.attributes, [
    "prompt_token_ids",
    "agentlightning.operation.output.prompt_token_ids",              # Weave tracer
]) or []
response_token_ids = _attributes_get_ids_multiple(span.attributes, [
    "response_token_ids",
    "agentlightning.operation.output.response_token_ids.0",          # Weave tracer
    "agentlightning.operation.output.choices.0.token_ids",           # Weave + 新版 vLLM
    "agentlightning.operation.output.choices.0.provider_specific_fields.token_ids",  # 新 vLLM + 新 OpenAI SDK
]) or []
...
prompt_payload = {"token_ids": prompt_token_ids, "raw_content": prompt_raw_content, "image_urls": image_urls}
response_payload = {"token_ids": response_token_ids, "raw_content": completion_raw_content}
logprobs_content = span.attributes.get("logprobs.content")            # FIXME: Weave tracer 尚不支持 logprob
if isinstance(logprobs_content, str):
    response_payload["logprobs"] = json.loads(logprobs_content)
return Triplet(prompt=prompt_payload, response=response_payload, reward=None,
               metadata=dict(request=..., response=..., response_id=..., agent_name=...))
```

`Triplet` 的定义极简（`types/core.py`）：`{prompt: Any, response: Any, reward: Optional[float], metadata: Dict[str, Any]}`。

**（3）`match_rewards`：两种分配策略**（`adapter/triplet.py:522-575`）：

| 策略 | 算法 | 语义 |
|---|---|---|
| `FIRST_OCCURRENCE`（默认） | 把全树按 `start_time` 排序，边走边记"已出现的 LLM 调用（id, end_time）"；遇到 reward span 时**从后往前**找第一个 `end_time <= reward.start_time` 的调用，把 reward 给它 | "**时间上最后一次已完成的 LLM 调用**"拿到 reward |
| `FIRST_SIBLING` | 在每个节点内部，只看**直接子节点**里的 LLM 调用与 reward，同样从后往前匹配 | "**同一个父节点下、reward 之前最后完成的** LLM 调用"拿到 reward |

两者都用"从后往前"遍历，代码里那句注释解释了原因：

```python
for assign_to_id, assign_to_end_time in reversed(assign_to):
    # This reward happens before the end of the LLM call.
    if assign_to_end_time > item.start_time:
        continue                                  # reward 必须在这次调用结束之后才算数
    if assign_to_id in rewards:
        continue                                  # 已有 reward 的不覆盖
    rewards[assign_to_id] = agentops_output.get("value", None)
    break
```

**（4）`to_trajectory`：整条流水线的收口**（`adapter/triplet.py:702-758`）——找 LLM 调用 → 转 triplet → 过滤掉没有 token id 的 → 匹配 reward → 补 `final_reward`：

```python
if final_reward is not None and len(transitions) > 0:
    transitions[-1] = transitions[-1].model_copy(update={"reward": final_reward})   # 只打到最后一条
```

**（5）verl 侧最终怎么用这些 triplet**（`verl/daemon.py`）：

```python
# _validate_data_v1：final_reward 由"从后往前搜第一个非 None 的 triplet.reward"决定
final_reward: Optional[float] = None
if triplets:
    for triplet in reversed(triplets):
        if triplet.reward is not None:
            final_reward = triplet.reward
            break
```

```python
# get_train_data_batch：同一 rollout 的**所有 triplet 共用同一个 final_reward**
trace_list = [{"prompt_ids": t.prompt.get("token_ids", []),
               "response_ids": t.response.get("token_ids", []),
               "image_urls": t.prompt.get("image_urls", [])} for t in rollout.triplets]
info = {"reward": final_reward, "trace_list": trace_list, "data_id": original_sample["data_id"]}
...
# 在最后一个 token 位置写入 token 级分数
# "Create token-level scores by placing the final reward at the last token position"
scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(device)
```

**这就是"identical assignment（同值分配）"的代码证据**：一条 rollout 拆成 $K$ 个 triplet（$K$ = LLM 调用次数），**每个 triplet 都是一条独立训练样本，且都带同一个 `final_reward`**，reward 最终落在**该 triplet response 的最后一个 token** 上。源论文出处：arXiv:2508.03680。

### 3. 具体数值样例

沿用第 9 节修好层级后的那棵树（2 次 LLM 调用 + 1 个 reward=1.0），对比两种 reward 策略：

```text
树的 LLM 调用（agent_match="SqlAgent"，llm_call_match="openai\.chat\.completion"）：
  s3_llm1 [1.0, 2.5] response_id=resp-1
  s5_llm2 [3.2, 5.0] response_id=resp-2
reward span：s6_reward [5.1, 5.2] value=1.0

策略 A：FIRST_OCCURRENCE（默认）
  按 start_time 排序遍历：
    s1_root(0.0) → s2_agent(0.5) → s3_llm1(1.0)：命中 LLM → assign_to=[(s3,2.5)]
    s4_tool(2.6)：非 LLM → assign_to 不变
    sGHOST(3.2) → s5_llm2(3.2)：命中 LLM → assign_to=[(s3,2.5), (s5,5.0)]
    s6_reward(5.1)：是 reward → reversed 遍历：
        (s5, 5.0)：5.0 > 5.1 ? 否 → rewards[s5]=1.0，break
  结果：s3.reward=None，s5.reward=1.0
  → 只有"最后一次 LLM 调用"拿到 reward

策略 B：FIRST_SIBLING
  按节点遍历，item=s2_agent 时只看它的直接子节点 [s3, s4, s6, sGHOST]：
    s3 是 LLM → assign_to=[(s3,2.5)]
    s4 不是
    s6 是 reward → reversed：s3 的 2.5 > 5.1 ? 否 → rewards[s3]=1.0
  结果：s3.reward=1.0，s5.reward=None
  → "同父下的第一次 LLM 调用"拿到 reward（因为它与 reward 同属 SqlAgent 的直接子节点）

两者输出完全不同的 credit assignment！这正是 RewardMatchPolicy 存在的意义。

to_trajectory 的输出（策略 A + final_reward=1.0）：
  transition[0] = Triplet(prompt={token_ids:[512 个]}, response={token_ids:[180 个]}, reward=None)
  transition[1] = Triplet(prompt={token_ids:[700 个]}, response={token_ids:[150 个]}, reward=1.0)
  → 因为 final_reward 非空，transitions[-1] 被覆写为 reward=1.0（本来就是 1.0，无变化）

daemon 侧（_validate_data_v1）：
  从后往前搜第一个非 None → final_reward = 1.0（取自 transition[1]）
  trace_list = [ {prompt_ids:512, response_ids:180}, {prompt_ids:700, response_ids:150} ]
  info = {"reward": 1.0, "trace_list": [...], "data_id": <原样本 id>}

get_train_data_batch 侧：
  同一个 rollout 的 2 个 triplet → 2 条训练样本，**reward 都是 1.0**
  token_level_scores：在每条样本 response 的最后一个 token 位置写 1.0
  统计量：training/n_triplets += 2；training/n_truncated_triplets（被 max_response_length 截断的）

对比"如果没做 repair_hierarchy"：
  s5_llm2 因为挂在根下（不在 SqlAgent 子树内）被 find_llm_calls 过滤掉
  → 只剩 1 个 triplet → 这条 rollout 的训练样本数腰斩
```

**四个量化结论**：① **一条 rollout 的训练样本数 = LLM 调用次数 $K$**（不是 1，也不是 token 数）；② **reward 策略会让"谁拿 reward"完全改变**（策略 A 给 s5，策略 B 给 s3），而 `final_reward` 覆写最后一跳又会让"最后一次调用"必然带上 reward；③ verl 侧同一 rollout 的 $K$ 条样本**共享同一个 reward**（identical assignment）；④ 修层级能**把样本数从 1 恢复到 2**——这是 `repair_hierarchy` 的直接收益。

> **面试一句话总结**：Span 树 → 训练样本分四步——① `find_llm_calls` 用**四道过滤**（span 名正则 `openai\.chat\.completion`、必须在匹配的 agent 子树内、嵌套 LLM 调用去重、`gen_ai.response.id` 去重）找出可训练的 LLM 调用；② `span_to_triplet` 把每个调用摊成 `Triplet{prompt:{token_ids,raw_content,image_urls}, response:{token_ids,raw_content,logprobs}, reward, metadata}`，token id 用**候选属性键列表**兼容 OTel / Weave / 新旧 vLLM 四种写法；③ `match_rewards` 两种策略——`FIRST_OCCURRENCE`（全树按时间排序，reward 给"时间上最后一次已完成"的调用）与 `FIRST_SIBLING`（只在同父兄弟里找），两者**从后往前**匹配且要求 `llm_end <= reward_start`；④ `to_trajectory(final_reward)` 把 final_reward 打到最后一跳，verl 的 `AgentModeDaemon` 再"从后往前搜第一个非 None reward"作为该 rollout 的 `final_reward`，并让**该 rollout 的所有 triplet 共享它**（identical assignment，arXiv:2508.03680），最后在每条样本 response 的**最后一个 token** 位置写 token-level score。

---

## 11. 有序性、幂等与故障语义（深水区）

### 1. 现有问题：分布式下"顺序"和"重复"都是问题

三机分离 + 多 runner 的现实带来三个必须显式设计的语义：① **时钟不可信**——agent 机器和 store 机器的时钟可能差几百毫秒，`start_time` 排序在多机场景下不可靠；② **同一条 span 可能被上传两次**（重试、proxy 与 tracer 双写、`BatchSpanProcessor` 重发）；③ **上传失败不能拖垮 agent**——但"失败静默"又会导致训练数据悄悄缺一条。这三件事如果没有明确设计，训练侧的样本数就会**随机波动**，而且极难排查。

### 2. 方法论：store 分配序号 + 主键幂等 + 上传即心跳 + 静默降级

**（1）顺序：序号由 store 统一分配，且**键是 `rollout_id`**。** `store/collection_based.py:1008-1054`：

```python
async def _issue_many_span_sequence_ids(self, rollout_ids: List[str]) -> List[int]:
    """Issue a new span sequence ID for a given rollout."""
    request_counts: Dict[str, int] = defaultdict(int)
    for rollout_id in rollout_ids:
        request_counts[rollout_id] += 1                    # 同一次批量里同一 rollout 要几个号

    latest_values: Dict[str, int] = {}
    for rollout_id, count in request_counts.items():
        async with self.collections.atomic(mode="rw", snapshot=False, labels=["span_sequence_ids"]) as collections:
            latest_values[rollout_id] = await collections.span_sequence_ids.inc(rollout_id, count)   # 原子自增
    ...

async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
    """Get the next span sequence ID for a given rollout and attempt.
    The number is strictly increasing for each rollout.
    The store will not issue the same sequence ID twice.
    """
    ret = await self._issue_many_span_sequence_ids([rollout_id])
    return ret[0]
```

**这里有个值得单独指出的实现细节**：`get_next_span_sequence_id(rollout_id, attempt_id)` **接收** `attempt_id`，但**计数器只以 `rollout_id` 为键**——docstring 也明确写 "strictly increasing for **each rollout**"。含义是：**同一个 rollout 的第二次 attempt（重试）不会重置序号**，序号召集会跨 attempt 继续累加。所以：
- 同一 attempt 内的 span 靠 `sequence_id` 稳定排序（这是设计目标）；
- 跨 attempt 的 span **不能**只靠 `sequence_id` 区分，必须同时按 `attempt_id` 过滤——这正是算法侧要用 `query_spans(rollout_id, attempt_id="latest")` 的原因（`verl/daemon.py` 的 `_validate_data_v1`）；
- proxy 中间件里那句注释 "Allocate a monotonic sequence id per (rollout, attempt)"（`llm_proxy.py:564`）表达的是**意图**，实际计数粒度以 store 实现为准。

**外部传入的序号会被"抬升"而不是"覆盖"**（`store/collection_based.py:1034-1038`）：

```python
async def _sync_span_sequence_id(self, rollout_id: str, sequence_id: int) -> None:
    """Sync the span sequence ID for a given rollout from the input span sequence ID."""
    async with self.collections.atomic(mode="rw", snapshot=False, labels=["span_sequence_ids"]) as collections:
        await collections.span_sequence_ids.chmax(rollout_id, sequence_id)     # 取 max，不是 set
```

`chmax`（**ch**ange to **max**）保证：**即便有外部实体塞进来一个很大的序号，计数器也只会单向变大，绝不会回退导致后续分配重号**。

**（2）幂等：靠 `spans` collection 的主键去重 + 逐条退化重试。** `store/collection_based.py:1112-1147`：

```python
async def _insert_spans_with_fallback(self, spans: Sequence[Span]) -> Sequence[Span]:
    async def _add_span_fallback(collections, span: Span) -> bool:
        try:
            await collections.spans.insert([span])
            return True
        except DuplicatedPrimaryKeyError:                             # 重复 span
            logger.error(f"Duplicated span added for rollout={span.rollout_id}, "
                         f"attempt={span.attempt_id}, span={span.span_id}. Skipping.")
            return False

    successful_spans: List[Span] = []
    try:
        async with self.collections.atomic(mode="w", snapshot=..., commit=False, labels=["spans"]) as collections:
            await collections.spans.insert(spans)                     # 先批量插
        successful_spans.extend(spans)
    except DuplicatedPrimaryKeyError:
        for span in spans:                                            # 有一条重复 → 整体退化为逐条
            async with self.collections.atomic(mode="w", ...) as collections:
                if await _add_span_fallback(collections, span):
                    successful_spans.append(span)
    return successful_spans
```

**语义**：span 的主键（`rollout_id + attempt_id + span_id`）天然提供幂等——**重复上传不会产生两条 span，而是被丢弃并打 ERROR 日志**；批量插入一旦撞重复就退化成逐条插入，**其余不重复的 span 仍然入库**（不会因为一条重复丢掉整批）。源码里还有一句诚实的 FIXME：`Part of the insertion might complete though the full operation fails` —— 非事务性插入的边界情况。

**（3）"上传 span 就是心跳"：attempt/rollout 状态会被 span 顺带推进。** `add_span` 成功后走 `_post_add_spans` → `_on_attempt_heartbeat`（`store/collection_based.py:1181-1235`）：

```python
# Update attempt heartbeat and ensure persistence
attempt.last_heartbeat_time = time.time()
if attempt.status in ["preparing", "unresponsive"]:
    attempt.status = "running"                       # 第一次收到 span → attempt 进入 running
await collections.attempts.update([attempt], update_fields=["last_heartbeat_time", "status"])

# If the status has already timed out or failed, do not change it (but heartbeat is still recorded)

# Update rollout status if it's the latest attempt
if latest_attempt is not None and attempt.attempt_id == latest_attempt.attempt_id:
    if rollout.status in ["preparing", "queueing", "requeuing"]:
        rollout.status = "running"                   # 最新 attempt 在跑 → rollout 也 running
```

**这段代码关键在哪**：① **span 上传 = 心跳**，所以一个正常产生轨迹的 runner 不会因为"忘调 `update_worker`"而被 watchdog 判为 unresponsive；② 状态推进是**单向且保守**的——已经 `succeeded`/`failed`/超时的 attempt/rollout **不会被 span 改回 running**（注释明确写了），避免"迟到的 span 复活一个已终结的 rollout"。

**（4）故障语义：写 store 失败是"静默降级 + 本地留存 + 日志"，不阻断 agent。** 回看第 6 节的 `on_end`：超时（`STORE_WRITE_TIMEOUT_SECONDS = 10.0`）或任何异常，只在本地 `self._spans` 里留一份并 `logger.warning/exception`，**绝不向 agent 抛异常**（注释：`on_end MUST NOT raise`）。

**这意味着一个非常重要的工程事实（面试可以主动说）**：

$$ \text{store 里的 span 数} = \text{实际产生的 span 数} - \text{上传失败的 span 数} $$

而**上传失败是静默的**——训练侧只会看到"这个 rollout 的 triplet 少了几个"甚至"没有 triplet"，表现为**训练样本数莫名波动**。工程上的应对是：监控 `tracer/otel.py` 的两条日志（`Timed out adding span ...` / `Error adding span to store: ...`），并把 `_spans`（本地留存列表）在必要时落盘。**更隐蔽的是 token 相关的失败**：`to_trajectory` 会把"没有 token id 的 LLM 调用"过滤掉（`_skip_empty_token_spans`），所以 **token 插桩没生效的轨迹会"看起来跑成功了，但训练时被静默丢弃"**——这也是第 6 节那个 `# MAGIC! DO NOT TOUCH THIS!` 警告存在的原因。

**（5）OTLP 路的额外故障点：缺 ID 的 span 被直接丢弃。** `utils/otlp.py:174-182`：

```python
if rollout_id is None or attempt_id is None:
    logger.warning(
        "Both rollout_id and attempt_id must be present in resource attributes. "
        "Spans will not be able to log to the store because of missing IDs: rollout_id=%s, attempt_id=%s, sequence_id=%s",
        rollout_id, attempt_id, sequence_id,
    )
    continue
```

对比逐条路：逐条路是把 `rollout_id`/`attempt_id` 当**函数参数**传的，不可能丢；OTLP 路要**靠 resource 属性携带**，所以一旦 `_enable_native_otlp_exporter` 没生效（或 resource 合并失败），整批 span 会被静默丢弃（只有一条 warning）。**这是选路机制带来的新故障模式**，也是排查"OTLP 路上传成功但 store 里没数据"的第一个检查点。

**（6）v0.3.0 的边界（现状）**：OTLP 只实现了 traces，metrics 与 logs 端点**直接返回 501**（`store/client_server.py:957-968`）：

```python
# Other API endpoints are not supported yet
@api.post("/metrics")
async def otlp_metrics():
    return Response(status_code=501)

@api.post("/logs")
async def otlp_logs():
    return Response(status_code=501)
```

也没有实现 OTLP 的 `partial_success` 字段（`utils/otlp.py:94` 注释：`Partial success field is left unset`），即**服务端不告诉客户端"我丢了几条"**——所以"静默丢 span"在协议层面是允许的。

### 3. 具体数值样例

```text
场景：同一个 rollout r-1/attempt a-1，store 计数器初始为 0

【序号分配】4 条 span 声称的顺序 vs 实际分配
  proxy 中间件先为每次 LLM 请求领号（在第 3 节例子中为 1、2、3），
  emitter 的 reward span 随后领号：
    第 1 次 inc(r-1, 1) → 返回 1
    第 2 次 inc(r-1, 1) → 返回 2
    第 3 次 inc(r-1, 1) → 返回 3
    第 4 次 inc(r-1, 1) → 返回 4
  若改用 OTLP 批量路：4 条 span 一次 inc(r-1, 4) → 返回 4，客户端本地切分为 1、2、3、4
  ⇒ 请求数从 4 次原子操作降到 1 次（第 8 节的关键性能差异）

【乱序到达也不影响排序】
  假设 span#2 因为网络慢，在 span#3 之后才到 store：
    写入顺序：1 → 3 → 2 → 4
    store 里的 sequence_id 仍是 1、2、3、4
    查询时 ORDER BY sequence_id → 恢复正确顺序（这就是不用时间戳排序的意义）

【外部塞号被抬升】
  假设某 span 自带 sequence_id=100（来自 proxy 的 x-sequence-id）
    写入时 chmax(r-1, 100) → 计数器变成 100
    后续 inc(r-1, 1) → 返回 101（不会重号）

【重复上传】
  span#2 被上传两次：
    第 1 次：insert 成功 → successful_spans=[s2]
    第 2 次：批量 insert 抛 DuplicatedPrimaryKeyError
             → 退化为逐条：s2 抛重复 → return False（打 ERROR 日志，丢弃）
    ⇒ store 里仍然只有 4 条 span，幂等成立

【上传失败（最需要警惕的）】
  span#3 上传超时（>10s）：
    on_end 捕获 TimeoutError → logger.warning("Timed out adding span ... after 10.0 seconds. "
                                              "The span will be stored locally but it's not guaranteed to be persisted.")
    → 本地 _spans 里有 4 条；store 里只有 3 条
  下游影响（假设 span#3 是一次 LLM 调用）：
    轨迹的 LLM 调用数从 3 变成 2
    → 该 rollout 的训练样本从 3 条变成 2 条（少一条）
    → training/n_triplets 指标下降，但训练本身不会报错
  这就是"训练样本数随机波动"的最常见根因之一。

【跨 attempt 的序号连续性】
  attempt a-1 用了 sequence_id 1..4；agent 崩溃 → rollout 重试 → attempt a-2
    a-2 的第一条 span 领到的是 5（不是 1）
  ⇒ 如果算法侧错用 query_spans(r-1)（不限 attempt）拿到的会是 a-1+a-2 混在一起、且序号连续的数据
  ⇒ 正确做法：query_spans(r-1, attempt_id="latest")（verl daemon 的做法）
```

**五个量化结论**：① **序号是"分配"的而不是"比较"的**，所以乱序到达不影响正确性；② OTLP 批量路把**原子领号次数从 $N$ 降到 1**；③ 重复上传被主键幂等丢弃，**不会重复计数**；④ 上传失败是**静默的**，直接表现为训练样本数减少（最需要监控的指标）；⑤ 序号**跨 attempt 连续性**，所以查轨迹必须带 `attempt_id` 过滤。

> **面试一句话总结**：分布式轨迹有三个必须显式设计的语义——① **有序**：`sequence_id` 由 store 用 `span_sequence_ids.inc()` 原子分配，**键是 `rollout_id`（docstring: "strictly increasing for each rollout"，`attempt_id` 只出现在签名里）**，因此重试不会重置序号、跨 attempt 的 span 必须靠 `attempt_id` 过滤；外部塞入的序号用 `chmax`（取 max）单向抬升绝不回退；OTLP 路支持整批一次领号。② **幂等**：`spans` collection 用主键（`rollout_id + attempt_id + span_id`）去重，批量插入撞重复时**退化为逐条插入**，重复 span 记 ERROR 日志后丢弃，其余仍入库。③ **故障**：`on_end` 有 10s 超时且**绝不向 agent 抛异常**（`on_end MUST NOT raise`），失败只在本地 `_spans` 留存 + 打日志；并且"上传 span 即心跳"（`_on_attempt_heartbeat` 会刷新 `last_heartbeat_time` 并把 `preparing/unresponsive → running`，但**不会把已终结的状态改回 running**）；OTLP 路额外有个"缺 rollout/attempt 的 span 被静默丢弃"的故障模式，且 v0.3.0 的 `/v1/metrics`、`/v1/logs` 直接返回 501、响应也不带 `partial_success`——**"静默丢 span → 训练样本数波动"是最需要监控的坑**。

---

# 四、共卡（colocate）下「推理 ↔ 训练」的全流程：参数在哪、怎么 offload、每一步怎么训

前面三部分讲的是"轨迹怎么存、怎么传、怎么建树"。但真正吃掉 GPU 时间的是另一件事：**推理和训练怎么在同一批卡上轮流用、参数在切换时放在哪里、权重怎么从训练侧搬到推理侧**。这就是"训推解决空泡"的**第一个 baseline——共卡分时复用**：不加机器、不加卡，靠"把推理引擎睡下去、把训练引擎叫起来"来复用同一批显存。这一部分把这条流水线从 `_train_step` 一路拆到 `optimizer.step()`，每一步都给源码位置。

> **版本说明**：本部分基于 `verl-v0.8.0`（**engine 架构**：`workers/engine_workers.py` + `workers/engine/` + `checkpoint_engine/`，**没有** `fsdp_workers.py`，也**没有** `sharding_manager/`）+ 项目自己的 TQ 接入仓 `agent-lightning`（`AgentLightningTrainer(verl.trainer.main_ppo_sync.PPOTrainer)`）。注意与第 1~11 节讲的 `agent-lightning-official` v0.3.0（继承 `RayPPOTrainer`）**不是同一份代码**，两者的 trainer 基类、数据流（DataProto vs KVBatchMeta）完全不同。

## 12. 先定位：共卡是"解决空泡"的 baseline，三种 RolloutMode 决定一切

### 1. 现有问题：为什么"共卡"是绕不开的第一选择

RL 训练里 GPU 上有两类完全不同的负载：**推理（rollout）**要的是"大 KV cache + 高并发小 batch"，**训练**要的是"参数/梯度/优化器状态 + 大 batch 反向"。它们**不可能同时塞进同一张卡的显存**——以 1.5B 模型在 8 卡上为例（$P=1.5\text{B}$，bf16 参数/梯度 + fp32 Adam）：

| 项目 | 总量 | 单卡（8 卡 FSDP 分片） |
|---|---|---|
| 模型参数（bf16） | $2P = 3$ GB | 0.375 GB |
| 梯度（bf16） | $2P = 3$ GB | 0.375 GB |
| Adam 一阶+二阶动量（fp32） | $8P = 12$ GB | 1.5 GB |
| fp32 master weights | $4P = 6$ GB | 0.75 GB |
| **训练侧小计** | | **≈ 3 GB/卡**（还要加激活） |
| vLLM 权重（bf16，TP=1） | 3 GB | 3 GB/卡 |
| vLLM KV cache | $2 \times L \times H_{kv} \times d_{head} \times \text{tokens}$ | 由 `gpu_memory_utilization` 决定 |

**两边都是 3 GB 级**，而 24 GB 卡上还要留激活、通信 buffer、CUDA context——所以只有两条路：

1. **共卡（colocate / 时间复用）**：同一批卡上先跑推理、再跑训练，**切换时把一方的显存让出来**。优点：零额外机器、权重搬运在同卡/同机（便宜）、天然 on-policy（权重刚更新就能拿去采样）。缺点：**推理和训练不能重叠**，切换本身有开销；
2. **分卡（disaggregate）**：推理一组卡、训练一组卡，**真异步重叠**。优点：无切换、可重叠；缺点：多花卡、权重跨机传输贵、必须处理 off-policy（staleness）。

**共卡就是那个"不加机器"的 baseline**——所以面试里被问"你怎么解决空泡"，正确开场是："最基础的做法是共卡分时复用，让推理引擎和训练引擎轮流占用同一批显存，切换点用 sleep/wake + offload 交接；它的极限是推理与训练无法重叠。"

### 2. 方法论：三种 RolloutMode 是三条不同的协作路线

v0.8.0 把"推理和训练怎么共存"抽象成 `RolloutMode` 三态（`verl-v0.8.0/verl/workers/rollout/replica.py:54-68`，注释原样摘录）：

```python
class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"
```

| 模式 | 进程/卡关系 | 切换时要不要同步权重 | 典型场景 |
|---|---|---|---|
| **HYBRID** | 推理引擎与训练引擎**同进程**，共享同一批 GPU | **要**（训练更新了权重，推理必须重新加载） | on-policy RL —— **共卡 baseline** |
| **COLOCATED** | 同 placement group 但**分进程**，共享 GPU | **不要**（推理侧不训练，权重不变） | GRM（LLM as a judge） |
| **STANDALONE** | 独立 GPU 资源池，**分卡** | 不需要（跨机靠 checkpoint engine 传） | off-policy / 异步 |

**判定代码**在 worker 里——rollout 的 device mesh 直接从**训练侧自身的 `world_size`** 推导，这就是"共用同一批卡"的代码证据（`verl-v0.8.0/verl/workers/engine_workers.py:596-611`）：

```python
infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
infer_pp = rollout_config.pipeline_model_parallel_size
infer_world_size = infer_tp * infer_pp
dp = self.world_size // infer_world_size
assert self.world_size % infer_world_size == 0, (
    f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
)
rollout_device_mesh = init_device_mesh(
    get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
)
self.rollout = rollout_cls(config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh)
```

同一段代码里还有一句很能说明问题的注释（`engine_workers.py:631-632`）——**训练侧要主动把显存还给系统，否则同卡的 vLLM 通过 `cudaMemGetInfo` 看不到**：

```python
# Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
aggressive_empty_cache(force_sync=True)
```

**我们的项目落在哪一态？** 项目仓 `agent-lightning` 的 TQ 方案走的是 **HYBRID 的共卡路线**（同机多卡，训练进程 + rollout server 共处），并且把它推到"**训练侧几乎不持有数据，只持有 keys**"的程度（第 16 节）。而 `Colocate-vs-Disaggregate.md` 里讨论的 `separate_async` 属于 STANDALONE 路线——**两者是同一根轴的两端**。

### 3. 具体数值样例

```text
场景：单机 8 卡，Qwen2.5-1.5B，rollout.n=4，ppo_mini_batch_size=32

【共卡（HYBRID）时间线】——同一批 8 张卡
t0 ─ t1  推理阶段：vLLM 占 GPU（权重 3 GB/卡 + KV cache 由 gpu_memory_utilization=0.5 决定）
         训练侧：参数 offload 在 CPU（param_offload=true），GPU 上几乎为 0
t1        切换点①：rollout.sleep() → vLLM 把显存交还（权重 + KV cache 全释放）
t1 ─ t2  训练阶段：FSDP 参数/梯度/优化器状态回 GPU（≈3 GB/卡）+ 激活
         推理侧：sleep，显存占用 ≈ 0
t2        切换点②：actor 更新完 → update_weights()（把新权重灌回 vLLM）+ rollout.wake_up()
t2 ─ t3  回到推理阶段，下一轮 rollout 用新权重（on-policy）

GPU 利用率 =（推理时间 + 训练时间）/ 总时间 → 推理与训练**串行**，切换开销直接计入空泡

【分卡（STANDALONE）时间线】——推理 4 卡 + 训练 4 卡
推理卡：一直在跑 rollout（利用率可接近 100%）
训练卡：等 TQ 里的轨迹够了就训；权重要跨机传到推理卡
GPU 利用率更高，但：① 卡数翻倍；② 权重跨机传输（第 15 节）；③ 轨迹来自旧权重 → off-policy
```

**一个关键的量化直觉**：共卡下"每步时间 = 推理时间 + 切换开销 + 训练时间"，三项**相加**；分卡下"每步时间 ≈ max(推理时间, 训练时间)"。所以**只有当推理与训练时间接近时，分卡才有接近 2× 的收益**；如果一边远大于另一边，分卡的收益立刻塌掉——这就是为什么"要不要分卡"必须先看 `timing_raw` 里 `gen` 与 `update_actor` 两项谁大。

> **面试一句话总结**：解决空泡的第一层 baseline 是**共卡分时复用**——把推理引擎和训练引擎放在同一批 GPU 上轮流用，切换时用 `sleep/wake` 把显存交还、用 `update_weights` 把新权重灌回推理侧；v0.8.0 用 `RolloutMode` 把这件事抽象成三态：`HYBRID`（同进程共享 GPU，切权重，on-policy）、`COLOCATED`（同 GPU 分进程，不切权重，GRM）、`STANDALONE`（独立卡，分卡异步，off-policy）；共卡下每步时间 = 推理 + 切换 + 训练**串行相加**，分卡下 ≈ `max(推理, 训练)` 但要多花卡、要传权重、要处理 staleness——所以**共卡是 baseline，分卡是"用卡换重叠"**。

---

## 13. 参数在哪里：训练态与推理态的两套内存 + 四个开关

### 1. 现有问题：同一个进程里两套引擎，谁占哪块显存

共卡最反直觉的一点是：**同一个进程里同时存在两个"模型"**——FSDP 训练模型和 vLLM 推理引擎，它们各有一份权重，各自还有一堆附属状态。面试里"参数在哪"的回答必须**分状态、分设备、分时刻**，否则一定被追问穿。要回答四个问题：

1. **训练态**：参数、梯度、优化器状态、激活分别在 GPU 还是 CPU？谁决定？
2. **推理态**：vLLM 的权重和 KV cache 什么时候释放、释放到什么程度？
3. **切换时**：谁负责搬运？搬什么、不搬什么？
4. **有什么坑**：哪些开关实际是"死"的、哪些组合会冲突？

### 2. 方法论：`to()` + 自动 offload + sleep level 三件套

**（1）训练侧的 offload API 只有一个：`engine.to(device, model, optimizer, grad)`**（`verl-v0.8.0/verl/workers/engine/base.py:169-180`）：

```python
def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
    """
    Move model parameters, optimizer states, or both to the specified device.

    Args:
        device: Target device identifier.
        model: If True, move the model.
        optimizer: If True, move the optimizer states.
        grad: If True, move the gradient buffer.
    """
    if grad:
        assert model, "Gradient buffers must be moved to device along with model parameters"
```

**注意 `grad` 被硬绑到 `model`**——想"只 offload 梯度、不 offload 参数"是不允许的。

**（2）自动 offload：真正干活的是 `BaseEngineCtx._context_switch`**（`base.py:229-264`）：

```python
class BaseEngineCtx:
    def __init__(self, engine: BaseEngine, mode, **kwargs):
        self.mode = mode
        assert self.mode in ("train", "eval")
        self.disable_auto_offload = kwargs.pop("disable_auto_offload", False)

    def _context_switch(self, device):
        if self.disable_auto_offload:
            return                                                    # ① 显式关掉自动搬运
        if device != "cpu":
            if not self.engine.is_param_offload_enabled and not self.engine.is_optimizer_offload_enabled:
                return                                                # ② 没开 offload → 上卡是 no-op
        if self.mode == "eval":
            self.engine.to(device=device, model=self.engine.is_param_offload_enabled,
                           optimizer=False, grad=False)               # ③ 推理/前向只要参数
        elif self.mode == "train":
            self.engine.to(device=device,
                           model=self.engine.is_param_offload_enabled,
                           optimizer=self.engine.is_optimizer_offload_enabled,
                           grad=self.engine.is_param_offload_enabled)  # ④ 训练要参数+优化器+梯度

    def __enter__(self):
        self._context_switch(get_device_name())                       # 进入：搬上 GPU
        self.engine.mode = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu")                                   # 退出：搬回 CPU
        self.engine.mode = None
```

**这段代码就是"参数在哪里"的唯一答案**：参数/优化器/梯度的设备位置**不是静态的**，而是被 `with engine.train_mode():` / `with engine.eval_mode():` 这对上下文管理器按"进入/退出"自动搬运的。三个可直接引用的结论：

- **`grad` 的开关被硬绑为 `is_param_offload_enabled`**——FSDP 路径下"梯度 offload"不是独立开关，而是跟着参数走。配置里的 `grad_offload` 只有 **Megatron** 引擎读（`workers/engine/megatron/transformer_impl.py:96`），**FSDP 引擎完全没消费它**；
- **`disable_auto_offload=True` 是给"内层循环"用的**：`TrainingWorker.train_mini_batch` 在外层开一次 `train_mode`，内层每个 mini-batch 都显式传 `disable_auto_offload=True`，**避免每个 mini-batch 都来回搬一次**（`engine_workers.py:272-302`）：

```python
with (
    self.engine.train_mode(disable_auto_offload=disable_auto_offload),
    Timer(name="train_batch", logger=None),
):
    for batch_idx, mini_batch_td in enumerate(dataloader):
        tu.assign_non_tensor(
            mini_batch_td,
            global_token_num=NonTensorData(global_token_num),
            update_lr_scheduler=batch_idx == total_num_iterations - 1,
            disable_auto_offload=True,                     # ← 内层不搬，只在外层搬一次
        )
        actor_output = self.train_batch(mini_batch_td)
```

- **初始化完就立刻 offload**（`fsdp/transformer_impl.py:202-207`）——建完模型先搬到 CPU，等第一次 `train_mode()` 再上卡：

```python
self.to(
    device="cpu",
    model=self._is_offload_param,
    optimizer=self._is_offload_optimizer,
    grad=self._is_offload_param,
)
```

**（3）搬运的底层实现**（`verl-v0.8.0/verl/utils/fsdp_utils.py:166-247`；注意**不在** `workers/engine/fsdp/utils.py`，那个文件只有 3 个 mesh 相关函数）：

```python
@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2 or fsdp_version(model) == 0:
        offload_fsdp2_model_to_cpu(model, empty_cache)      # FSDP2：model.cpu()
        return
    assert isinstance(model, FSDP)
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
                and id(flat_param.data) != id(flat_param._local_shard)
                and flat_param.data.size() == flat_param._local_shard.size())
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)   # FSDP1：整个 flat_param 搬走
        flat_param._local_shard = flat_param.data                       # 修 _local_shard 的身份
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        get_torch_device().empty_cache()

@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)      # exp_avg / exp_avg_sq / step 全搬
```

**四个实现细节，每个都能单独出面试题**：

| 细节 | 说明 |
|---|---|
| FSDP1 vs FSDP2 | FSDP1 对每个 handle 搬 `flat_param`（分片后的扁平参数）并**修正 `_local_shard` 的身份**（那段 assert 是 FSDP1 的硬约束）；FSDP2 直接 `model.cpu()` / `model.to(device)`，让 DTensor 自己走 |
| **没有 pinned memory** | FSDP 的 offload 只用 `non_blocking=True`；**pinned 只出现在 FSDP2 的 `CPUOffloadPolicy(pin_memory=True)`**（`fsdp/transformer_impl.py:405-418`）、NIXL 收发 buffer、activation offload、Megatron 路径。所以"用了 `non_blocking` 就一定异步重叠"是**错的**——H2D/D2H 要真异步，目标内存必须 pinned |
| 优化器 state 全搬 | 不枚举 key，只判 `isinstance(value, torch.Tensor)`——Adam 的 `exp_avg` / `exp_avg_sq` / `step` 都会搬到 CPU |
| `offload_policy` 会"接管" | 一旦用 FSDP2 的 `CPUOffloadPolicy(pin_memory=True)`，引擎会主动把 `_is_offload_param/_is_offload_optimizer` 置 False，把设备管理权交还给 PyTorch |

**（4）推理侧的睡眠等级：level 1 vs level 2，语义由 vLLM 的 `CuMemAllocator` 定义。** vLLM 的 `sleep(level)` 最终落到 `allocator.sleep(offload_tags=...)`（`vllm/device_allocator/sleep_mode_backend.py:120-125`）：

```python
def suspend(self, level: int = 1) -> None:
    self._state = "SUSPENDED"
    allocator = get_mem_allocator_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())   # level 1 只备份 weights
```

而 `CuMemAllocator.sleep` 的语义是"**命中 tag 的备份到 pinned CPU，其余直接 unmap 释放**"（`vllm/device_allocator/cumem.py:229-294`）：

```python
def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
    """
    Put the allocator in sleep mode.
    All data in the memory allocation with the specified tag will be
    offloaded to CPU memory, and others will be discarded.
    """
    ...
    for ptr, data in self.pointer_to_data.items():
        ...
        if data.tag in offload_tags:
            backup_bytes += handle[1]
            cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8,
                                            device="cpu", pin_memory=PIN_MEMORY)     # ← pinned CPU
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
            data.cpu_backup_tensor = cpu_backup_tensor
        try:
            unmap_and_release(handle)                                               # 其余物理显存直接还
        finally:
            data.is_asleep = True
    logger.info(
        "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
        "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded directly.",
        total_bytes / 1024**3, backup_bytes / 1024**3, (total_bytes - backup_bytes) / 1024**3,
    )
```

**配合 vLLM 给两类分配打的 tag**（`vllm/v1/worker/gpu_worker.py:489` 是 `tag="weights"`，`:733` 是 `tag="kv_cache"`），就得到一张完全确定的表：

| level | `offload_tags` | 权重 | KV cache | 醒来（`wake_up`）要做什么 |
|---|---|---|---|---|
| **1** | `("weights",)` | **备份到 pinned CPU**（不丢） | **直接丢弃**（物理显存还回去） | 从 pinned CPU 拷回权重 + 重建 KV cache |
| **2** | `()`（空） | **也丢弃** | 丢弃 | 权重必须**重新灌**（靠 `update_weights`），代价更大但显存更干净 |

verl v0.8.0 里两种模式的等级选择不同（`vllm_async_server.py:625-634`、`:932-949`）：

```python
async def sleep(self):
    if self.node_rank != 0 or not self.config.free_cache_engine:
        return                                        # ← free_cache_engine=False 时 sleep 完全是 no-op
    if self.rollout_mode == RolloutMode.HYBRID:
        await self._sleep_hybrid()                    # level = 1 若 lora_as_adapter 或 NPU，否则 2
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.sleep(level=1)              # COLOCATED 硬编码 level=1
    elif self.rollout_mode == RolloutMode.STANDALONE:
        logger.info("skip sleep in standalone mode")
```

`_sleep_hybrid` 的 docstring 解释了两件重要的事（`vllm_async_server.py:932-949`）：

```python
async def _sleep_hybrid(self):
    """HYBRID sleep: lora adapters only need level=1; full weights need level=2.

    Uses engine.sleep() instead of engine.collective_rpc("sleep") to ensure
    that sleep is properly propagated to all data-parallel worker processes.
    collective_rpc only reaches the TP workers within a single DP shard,
    leaving other DP shards' weights unreleased, which causes OOM during
    FSDP training backward when DP > 1.
    """
    if self.lora_as_adapter or is_torch_npu_available(check_device=False):
        sleep_level = 1
    else:
        sleep_level = 2
```

**三个可直接背的结论**：① **HYBRID + 全量微调 → level 2**（权重直接丢，必须靠 `update_weights` 重新灌）；② **LoRA adapter 或 NPU → level 1**（权重保留在 pinned CPU，醒来快；NPU 是因为 vllm-ascend 还不支持 level 2）；③ **`free_cache_engine=False` 时 sleep/wake 全是 no-op**，共卡直接变"显存死锁"，只能靠 `param_offload` 让训练侧腾挪。

**（5）还有一个中间档：只放 KV cache、保留权重。** NCCL 权重同步需要"写进已有的权重 buffer"，所以专门有 `release_kv_cache_replicas` / `resume_kv_cache_replicas`（`verl-v0.8.0/verl/checkpoint_engine/base.py:452-467`）：

```python
@auto_await
async def release_kv_cache_replicas(self):
    """Release kv_cache of all rollout replicas before NCCL weight sync.

    Unlike sleep_replicas(), this only frees the kv_cache and leaves model
    weights untouched, so the NCCL transfer can write directly into the
    existing weight buffers.  Call resume_kv_cache_replicas() after sync.
    """
```

**（6）四个开关的完整组合表**（默认值来自 `workers/config/engine.py:89-96`、`workers/config/rollout.py:192,196`）：

| 开关 | 默认 | 作用域 | 打开后的效果 |
|---|---|---|---|
| `param_offload` | **False** | 训练引擎 | 每次进 `train_mode`/`eval_mode` 把 FSDP 参数（及梯度）GPU↔CPU 搬 |
| `optimizer_offload` | **False** | 训练引擎 | 同上，额外搬优化器 state（Adam 两个动量） |
| `grad_offload` | **False** | **仅 Megatron** | FSDP 路径**不消费**；FSDP 的梯度 offload 实际由 `param_offload` 决定 |
| `free_cache_engine` | **True** | 推理引擎 | 允许 `sleep/wake`；False → 变成 no-op，共卡下会 OOM |

而全局的 `gpu_memory_utilization` 默认 **0.5**（`workers/config/rollout.py:192`）——**这就是 vLLM 在共卡场景下的"自我限额"**：只敢占一半显存，剩下的留给 FSDP 训练。

### 3. 具体数值样例

```text
场景：单机 8 卡，1.5B 模型，bf16 训练，Adam(fp32)，rollout.n=4

【配置 A：全 offload（共卡标配，显存最省）】
  param_offload=true, optimizer_offload=true, free_cache_engine=true, gpu_memory_utilization=0.5
  推理阶段：vLLM 权重 3 GB/卡 + KV cache（限额 0.5×24 = 12 GB/卡）
           训练侧：参数/梯度/优化器 ≈ 0（全在 CPU）
  切换：sleep(level=2) → 释放权重 + KV cache → 显存回到 ~0
  训练阶段：FSDP 参数 0.375 + 梯度 0.375 + Adam 1.5 + master 0.75 ≈ 3 GB/卡（+激活）
  退出：train_mode 退出 → 全部搬回 CPU（D2H 拷贝 3 GB/卡）
  代价：每步多 2 次 H2D + 2 次 D2H 的 3 GB 级拷贝（无 pinned → 不重叠，纯串行）

【配置 B：不 offload（分卡/显存够用）】
  param_offload=false, optimizer_offload=false, free_cache_engine=false
  推理阶段：vLLM 权重 3 GB + KV cache 占着不放
  训练阶段：FSDP 3 GB + 激活，且 vLLM 显存**没还**
  ⇒ 两套模型同时常驻：3(训练) + 3(vLLM) + KV cache + 激活 → 24 GB 卡极易 OOM
  ⇒ 这就是"共卡必须 offload 或必须 sleep"的算术证明

【一个容易被忽略的开销】
  没有 pinned memory 时，cudaMemcpy 是同步的：
    3 GB / PCIe 4.0 x16 有效带宽 ~12 GB/s ≈ 0.25 s（单卡，单程）
    每步 2 程（上卡+回卡）→ ~0.5 s/步，且是**每卡各自串行**
  换成 pinned + non_blocking 后可与别的 compute 重叠
  → 这正是 FSDP2 用 CPUOffloadPolicy(pin_memory=True)、vLLM sleep 用 pin_memory=PIN_MEMORY 的原因
```

**一句话理解这张账**：offload 的本质是"**用 PCIe/NVLink 带宽换显存**"，共卡的本质是"**用串行换卡数**"；两者的代价都必须记在空泡账上（第 18 节）。

> **面试一句话总结**："参数在哪"由 `engine.to(device, model, optimizer, grad)` 一个 API 决定，而**调用它的是 `BaseEngineCtx._context_switch`**——`with engine.train_mode()` 进入即把参数（+优化器+梯度）搬上 GPU，退出即搬回 CPU，内层 mini-batch 用 `disable_auto_offload=True` 避免反复搬运；FSDP 路径下梯度 offload 被硬绑在 `param_offload` 上（`grad_offload` 只有 Megatron 读），底层是 FSDP1 的 `flat_param_to(non_blocking=True)` / FSDP2 的 `model.cpu()`，优化器 state 不枚举 key 全搬，**FSDP offload 路径没有 pinned memory**（pinned 只在 FSDP2 `CPUOffloadPolicy` 里）；推理侧由 `free_cache_engine` 控制是否允许 sleep，sleep level 1 = 权重备份到 pinned CPU + KV cache 丢弃、level 2 = 连权重一起丢（HYBRID 全量微调用 2、LoRA/NPU 用 1、COLOCATED 硬编码 1），`free_cache_engine=False` 时 sleep/wake 全是 no-op——**共卡下这四个开关的组合就是"显存够不够"的全部答案**。

---

## 14. 推理 → 训练的切换：wake/sleep 的完整代码路径与硬顺序

### 1. 现有问题：切换点到底是几个函数、谁调谁

"共卡要切换"这句话在代码里对应**一长串调用链**：训练脚本调一个方法 → 管理器 → 每个 rollout 副本 → HTTP server → vLLM 引擎 → 显存分配器。面试里如果只能说"调 sleep 让它睡"，深度不够；要把这条链走完，并说清**每一步做什么、为什么必须按这个顺序**。

### 2. 方法论：一条链 + 两个方向 + 严格顺序

**（1）入口：`CheckpointEngineManager` 提供四个"整体状态"操作**（`verl-v0.8.0/verl/checkpoint_engine/base.py:345-467`）。docstring 先用一张图讲清"谁跟谁连"（`base.py:346-366`）：

```python
class CheckpointEngineManager:
    """Checkpoint engine manager to coordinate weight synchronization between trainer and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In trainer, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.
    """
```

四个操作（都用 `asyncio.gather` 并发到所有副本）：

```python
@auto_await
async def sleep_replicas(self):
    """Sleep all rollout replicas: free weight and kv_cache device memory."""
    await asyncio.gather(*[r.sleep() for r in self.replicas])

@auto_await
async def wake_up_replicas(self):
    """Resume all rollout replicas: recover kv_cache and weights device memory."""
    await asyncio.gather(*[r.wake_up() for r in self.replicas])

@auto_await
async def release_kv_cache_replicas(self): ...   # 只放 KV，保留权重（NCCL 要写进原 buffer）
@auto_await
async def resume_kv_cache_replicas(self): ...    # 同步完恢复 KV
```

**注意 `@auto_await` 这个装饰器**：它让**同步调用方也能直接调这些协程方法**——训练主循环是同步的（`fit()` 里直接 `self.checkpoint_manager.sleep_replicas()`），底层却是异步的。这是共卡方案里"同步 trainer 驱动异步 rollout server"的粘合点。

**（2）中层：`RolloutReplica` 扇出到每个 server**（`verl-v0.8.0/verl/workers/rollout/replica.py:265-291`）：

```python
async def wake_up(self):
    """Wake up each rollout server."""
    await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

async def sleep(self):
    """Sleep each rollout server."""
    await asyncio.gather(*[server.sleep.remote() for server in self.servers])

async def release_kv_cache(self):
    """Release only the kv_cache GPU memory, keeping model weights in place."""
    await asyncio.gather(*[server.release_kv_cache.remote() for server in self.servers])
```

**（3）底层：vLLM server 的 `wake_up` / `sleep`**（`verl-v0.8.0/verl/workers/rollout/vllm_rollout/vllm_async_server.py:604-634`）：

```python
async def wake_up(self, tags: list[str] | None = None):
    if self.node_rank != 0:
        return
    if self.rollout_mode == RolloutMode.HYBRID:
        # engine.wake_up() broadcasts via the DP coordinator to ALL EngineCore
        # processes across all DP shards (unlike collective_rpc which only reaches
        # TP workers within a single shard).
        await self.engine.wake_up(tags=tags or self._get_wake_up_tags())
        await self.engine.reset_prefix_cache(**_RESET_PREFIX_CACHE_KWARGS)
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.wake_up(tags=self._get_wake_up_tags())
        await self.engine.reset_prefix_cache(**_RESET_PREFIX_CACHE_KWARGS)
    elif self.rollout_mode == RolloutMode.STANDALONE:
        logger.info("skip wake_up in standalone mode")
```

`wake_up` 里 **`reset_prefix_cache` 不能省**——注释说明它会 `reset_connector=True`，把挂在 vLLM 上的外部 KV store（例如 **MooncakeStoreConnector**）里"用旧权重算出来的"条目丢掉。**这条注释直接连到本项目**：共卡 + Mooncake 做 KV 存储时，权重一换，prefix cache 必须失效，否则复用旧 KV 会算错。

另外 `_get_wake_up_tags()` 默认返回 **`["kv_cache", "weights"]`**（`vllm_async_server.py:928-930`）——即"全都要醒"。

**（4）两个方向的完整周期，顺序是硬约束。** 顺序来自 `engine_workers.py:666-746` 的 `update_weights` docstring + `base.py:470-514` 的 Manager 实现：

```text
【阶段 1：推理】训练循环开始前/每轮 rollout 前 → wake_up_replicas()（权重 + KV cache 上 GPU）
【阶段 2：训练前】sleep_replicas()  → vLLM 释放权重/KV cache（level 2 时权重直接丢）
                 训练侧 train_mode() → 参数/优化器搬上 GPU（第 13 节）
【阶段 3：训练】forward_backward_batch → optimizer_step（第 17 节）
【阶段 4：训练后】update_weights() → 把新权重灌回 vLLM
【阶段 5：回推理】rollout 继续，用新权重（on-policy）
```

**`update_weights` 内部对顺序有明确要求**（`engine_workers.py:666-700` 的 docstring + 分流代码）：

```python
async def update_weights(self, global_steps: int = None, mode: str = "auto"):
    """Update weights from trainer to rollout.

    1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
       - before update_weights: rollout should be in sleep mode.
       - after update_weights: rollout should be in wake_up mode.
    2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.
    """
    effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

    # 0. send_weights only for async training with disaggregated trainer and rollout
    if effective_mode != "naive":
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
        return
    ...
```

**这段 if 是整节的题眼**：`update_weights(mode)` 用**一个 if 分叉**把"共卡"和"分卡"分开了——`mode="naive"` 是共卡（同进程直接内存交接），其它（`nccl` / `nixl` / `hccl` / `mooncake` / `kimi_ckpt_engine`）是分卡（走 checkpoint engine）。而 `checkpoint_engine.backend` 的**默认值就是 `"naive"`**（`workers/config/rollout.py:149`）——**默认就是共卡**。

### 3. 具体数值样例

```text
环境：8 卡共卡，1.5B，HYBRID（全量微调 → sleep level 2），param_offload=true

【显存时间线（单卡，粗算）】
t0  推理态：vLLM weights 3.0 GB + KV cache 10.0 GB（gpu_memory_utilization=0.5×24）
           训练侧 0.0 GB（offload 在 CPU）                        GPU 占用 ≈ 13.0 GB
t1  sleep_replicas()（level 2：offload_tags=()）
           → CuMemAllocator 日志：
             "sleep freed 13.00 GiB memory in total,
              of which 0.00 GiB is backed up in CPU and the rest 13.00 GiB is discarded directly."
           训练侧仍 0.0 GB                                          GPU 占用 ≈ 0.0 GB
t2  train_mode() → load_fsdp_model_to_gpu + load_fsdp_optimizer
           参数 0.375 + 梯度 0.375 + Adam 1.5 + master 0.75 ≈ 3.0 GB
           + 激活（remove-padding 后按 token 数）                    GPU 占用 ≈ 3.0+ GB
t3  optimizer_step() 后 train_mode 退出 → 全部搬回 CPU             GPU 占用 ≈ 0.0 GB
t4  update_weights()（naive）
           rollout.resume(tags=["weights"])     ← 先把权重 buffer 要回来
           get_per_tensor_param() 流式产出 (name, tensor)
           rollout.update_weights(per_tensor_param)  ← 同卡直灌（CUDA IPC）
           engine.to("cpu", model=True, optimizer=False, grad=False)
           aggressive_empty_cache(force_sync=True)
           rollout.resume(tags=["kv_cache"])    ← 再把 KV cache 要回来
                                                                    GPU 占用 ≈ 13.0 GB
t5  回到推理态，on-policy 采样

【如果 free_cache_engine=False（对照）】
t1  sleep_replicas() → **no-op**（第一行 `not self.config.free_cache_engine: return`）
    13 GB 仍然占着 → t2 训练要 3 GB + 激活 → 24 GB 卡几乎必然 OOM
  ⇒ 共卡场景下 free_cache_engine 与 param_offload 至少要开一个，通常两个都开
```

**三个顺序性的坑**（都是代码里显式写了的约束）：

1. **`update_weights` 必须在 rollout sleep 之后、wake 之前**（docstring 原文："before update_weights: rollout should be in sleep mode / after update_weights: rollout should be in wake_up mode"）——否则要么把权重写进已释放的 buffer，要么和推理抢显存；
2. **`set_expandable_segments(False)` 必须在 vLLM 醒来之前**（naive 路径 `engine_workers.py:702`）——`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 与 vLLM 的 `CuMemAllocator` 冲突；这也是 NCCL 的 `prepare()` 里要用 `cupy` 而不是 `torch` 分配 buffer 的原因（`nccl_checkpoint_engine.py:134-137` 有注释）；
3. **`release_kv_cache_replicas` 必须在权重同步之前**（`base.py:492-493` 注释："weights stay in place, so the NCCL transfer can write directly into the existing weight buffers"）——只放 KV、留着权重 buffer，避免重新分配。

> **面试一句话总结**：共卡的切换链是 `CheckpointEngineManager`（`sleep_replicas`/`wake_up_replicas`/`release_kv_cache_replicas`/`resume_kv_cache_replicas`，用 `@auto_await` 让同步 trainer 能直接调）→ `RolloutReplica`（`asyncio.gather` 扇出到每个 server）→ `vllm_async_server.wake_up/sleep`（HYBRID 必须用 `engine.wake_up/sleep` 广播到所有 DP shard，**不能用 `collective_rpc`**，否则 DP>1 时其他 shard 不释放权重、训练反向会 OOM）→ vLLM `CuMemAllocator.sleep`（按 tag 备份到 pinned CPU 或直接 unmap 释放）；**顺序是硬约束**：`sleep_replicas` → `train_mode` → 训练 → `update_weights`（`mode="naive"` 走同卡直灌，`backend` 默认就是 `naive`）→ `wake_up_replicas`；`wake_up` 里必须 `reset_prefix_cache(reset_connector=True)` 把用旧权重算的 KV（含 MooncakeStoreConnector）作废；`free_cache_engine=False` 会让 sleep/wake 全变 no-op，共卡下必须靠 `param_offload` 兜底。

---

## 15. 权重同步：naive（同卡直传）与 checkpoint_engine（跨卡多播）两条路

### 1. 现有问题：FSDP 的分片参数怎么变成 vLLM 能吃的权重

训练侧参数是 **FSDP 分片 + 扁平化**的（每个 handle 一个 `flat_param`），推理侧 vLLM 要的是**按名字组织的完整张量**。中间要做四件事：① 把分片**还原**成完整张量（`full_tensor()`）；② 按 vLLM 的命名约定**改名**；③ 决定**怎么运**（同卡直接写内存？跨机 NCCL？走 Mooncake？）；④ 控制**显存开销**（不能为了传输再复制一整份模型）。

### 2. 方法论：先有"流式生成器"，再分两条路

**（1）统一的源头：`get_per_tensor_param()` 返回的是生成器，不是字典。** `verl-v0.8.0/verl/workers/engine/fsdp/transformer_impl.py:794-871`：

```python
def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
    log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

    # FSDP2 CPUOffloadPolicy owns CPU<->GPU placement; calling model.to(device) here
    # leaves the module half-moved and crashes state_dict() below (#5995). The
    # per-DTensor .to(device).full_tensor() below still produces GPU tensors.
    if not self._uses_fsdp2_cpu_offload_policy:
        load_fsdp_model_to_gpu(self.module)
    ...
    else:
        params = self.module.state_dict()

    params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.module)          # ← 取完 state_dict 就立刻把模型搬回 CPU

    if peft_config is not None and base_sync_done:
        per_tensor_param = params.items()
    else:
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in params.items()
        )
    ...
    return per_tensor_param, peft_config_dict
```

**这段代码是"显存友好"的关键**：返回值是**惰性生成器**——每 yield 一个 `(name, tensor)` 才把那一个张量 `.to(device).full_tensor()` 物化出来，**全程只额外持有"一个张量"而不是整份模型**。这就是为什么"权重同步"不需要 2× 模型显存（除了 NCCL 自己的双 buffer，见下）。

**（2）路 A：naive（共卡，同进程直传）——8 个有序步骤**（`verl-v0.8.0/verl/workers/engine_workers.py:702-746`）：

```python
set_expandable_segments(False)                                      # ① 先关 expandable_segments
if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["weights"])                      # ② 把 vLLM 的权重 buffer 要回来
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=True)         # ③ 取流式权重
do_lora_base_sync = False
if not self.peft_merge and peft_config is not None:
    self.rollout.sleep_level = 1
    do_lora_base_sync = not self.base_sync_done
if do_lora_base_sync:
    await self.rollout.update_weights(per_tensor_param_base, peft_config=peft_config,
                                      base_sync_done=False, global_steps=global_steps)   # ④ LoRA 先灌基座
await self.rollout.update_weights(per_tensor_param, peft_config=peft_config,
                                 base_sync_done=True, global_steps=global_steps)          # ⑤ 再灌 adapter/全量
if self.actor.engine.is_param_offload_enabled:
    self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)                 # ⑥ 训练侧回 CPU
aggressive_empty_cache(force_sync=True)                                                   # ⑦ 收显存
if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["kv_cache"])                                          # ⑧ 恢复 KV cache
self.base_sync_done = True
set_expandable_segments(True)
```

**注意 ⑥ 只搬 `model`、不搬 `optimizer`**——因为在 `update_weights` 时优化器状态本来就已经在 CPU（第 13 节的 `train_mode` 退出时搬走了），这里只是把刚为同步而临时上卡的参数再搬回去。

**（3）路 B：checkpoint_engine（分卡，跨进程/跨机）——抽象接口 + 6 个后端。** 接口定义（`verl-v0.8.0/verl/checkpoint_engine/base.py:96-110`，docstring 本身就是用法说明）：

```python
class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from trainer to rollout.

    In trainer process:
    >>> trainer = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(trainer.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """
```

生命周期固定为 **`prepare` → `build_topology`(classmethod) → `init_process_group` → `send_weights`/`receive_weights` → `finalize`**，注册名与后端对应关系：

| backend | 文件 | 用途 |
|---|---|---|
| `naive` | `base.py:220` `ColocatedCheckpointEngine` | **共卡**：`send_weights` 只是把生成器存起来（`self.weights = weights`），`receive_weights` 再 `yield from` —— 零拷贝的内存交接 |
| `nccl` | `nccl_checkpoint_engine.py:102` | CUDA 多卡广播 |
| `hccl` | `hccl_checkpoint_engine.py:96` | 昇腾 NPU（**注意注册字符串写的是 `"nccl"`**，与 CUDA 版同名，疑似 bug） |
| `nixl` | `nixl_checkpoint_engine.py:238` | NVIDIA 的 NIXL 传输库 |
| `mooncake` | `mooncake_checkpoint_engine.py:34` | **Mooncake 传输**（和 TQ 的存储后端同源） |
| `kimi_ckpt_engine` | `kimi_checkpoint_engine.py:222` | Kimi 的传输实现 |

**Manager 侧的 8 步编排**（`base.py:470-514`，非 naive 路径）：

```python
# 1. abort and save all unfinished requests for partial rollout
await self.abort_replicas()

# 2. create a temporay worker group for all replicas
workers = []
for replica in self.replicas:
    workers.extend(replica.workers)
rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))

# 3. release kv_cache before weight sync (weights stay in place)
await self.release_kv_cache_replicas()

# 4. build process group
self.build_process_group(rollout)

# 5. update weights of all workers
ray.get(
    trainer.update_weights(global_steps=global_steps, mode=self.backend)
    + rollout.update_weights(global_steps=global_steps)
)

# 6. finalize all workers
ray.get(trainer.execute_checkpoint_engine(["finalize"] * trainer.world_size)
        + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size))

# 7. restore kv_cache after weight sync
await self.resume_kv_cache_replicas()

# 8. resume all unfinished requests for partial rollout
await self.resume_generation_replicas()
```

**第 1 步和第 8 步是"partial rollout"的落点**：同步权重前先把在飞的请求 abort 掉（但**保存**进度），传完权重再恢复生成——这样"正在跑的 rollout 不会读到半新半旧的权重"。这是分卡异步方案里最容易漏掉的一致性细节。

**（4）切块与重组：bucket 让显存可控。** `base.py:517-585` 提供一对生成器变换：

```python
async def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None], bucket_size: int
) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
    """Split the weight into chunks."""
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)              # 摊平成 uint8 字节流
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(name=name, shape=weight.shape, dtype=weight.dtype,
                                     chunk_offset=chunk_offset, chunk_size=chunk_size, offset=None)
            yield (tensor_meta, buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size
```

接收侧 `merge_weight_chunks` 按 `TensorMeta` 把块拼回原张量（**小张量直接过、大张量才开 buffer 累积**，`base.py:563-568` 有一段 `nbytes <= bucket_size` 的快路径）。bucket 大小的配置项是 **`update_weights_bucket_megabytes`，默认 2048（即 2 GB）**（`workers/config/rollout.py:151`，代码里 `<< 20` 转字节）。

**（5）NCCL 后端的两个硬事实**：**元数据走 ZMQ、数据走 broadcast**（`nccl_checkpoint_engine.py:80-90`）：

```python
def _run(self):
    # broadcast tensor meta via zeromq PUB/SUB
    if self.rank == 0:
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(self.metadata)
    else:
        self.socket.recv_string()
        self.metadata = self.socket.recv_pyobj()

    # broadcast tensor via NCCL
    collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)
```

拓扑是 **1:N 广播**而不是 all_gather（`nccl_checkpoint_engine.py:159-171`）：trainer 只有 rank0 参与发送（其余 rank 记 `-1`），rollout 从 rank1 开始；trainer 的非 0 rank 在 `send_weights` 里**只消费不发送**（`:240-246`，注释 `"Trainer workers other than rank 0 should not send weights."`）。**注意 README 表格里写的是 `all_gather+broadcast`，但代码里只有 broadcast**——这是文档与实现的偏差，面试时可以作为一个"我读过源码"的细节提。

而**显存开销是显式写死的 2×bucket**（`nccl_checkpoint_engine.py:104-108` docstring）：

```python
"""NCCL checkpoint engine with collective communication.

Args:
    bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
        two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
"""
```

对应 `prepare()` 里真的分配两个 buffer（`:134-144`），而且 master 侧**用 `cupy` 而不是 `torch`**，原因写在注释里：

```python
def prepare(self) -> MasterMetadata:
    # For master process, use cupy instead of torch to avoid memory register error
    # when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
    if self.is_master:
        self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
        self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
    else:
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
```

**（6）落地：rollout 侧最后一段走 CUDA IPC / 共享内存**（`verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:172-178`）：

```python
def _init_buffer(self):
    """build communication buffer"""
    buffer, shm = None, None
    if not self.use_shm:
        buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
        handle = reduce_tensor(buffer)
```

**（7）演进：带本地缓存的 engine（partial rollout 的进阶）。** `base.py:203-217`：

```python
class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """
    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
```

**"abort 请求 → 传权重 → 恢复请求"（v0.8 的做法）会被"权重先落到本地 cache，等旧请求自己跑完再切换"（Laminar 的做法）取代**——这是共卡/分卡方案从"打断式同步"走向"无打断同步"的方向。

### 3. 具体数值样例

```text
场景：1.5B 模型，bucket_size = 2048 MB（默认），NCCL backend，trainer 8 rank + rollout 8 rank

【权重张量数与切块数】
  1.5B 参数，按 HF 命名大约 300~500 个张量（q/k/v/o/gate/up/down × 层数）
  最大的张量：embed_tokens [151936, 1536] bf16 ≈ 0.43 GB，lm_head 同量级
  其余多为 [1536, 1536] bf16 = 4.5 MB 级
  bucket_size = 2 GB → 只有 embed/lm_head 这类会被切成 1 块（0.43 GB < 2 GB，不切）
  实际上 2 GB 的 bucket 远大于任何单个张量 ⇒ split_weight_chunks 基本"一桶一张量"
  若把 bucket 调成 32 MB：4.5 MB 的小张量仍不切；只有 >32 MB 的才切 2 段

【显存开销】
  NCCL 双 buffer：2 × 2 GB = 4 GB（每 rank，常驻）
  这在共卡场景下很贵 —— 所以共卡用 naive（零额外 buffer），
  分卡才用 NCCL（4 GB/rank 相对独立卡的显存可以接受）

【传输量（bf16）】
  全量：3 GB（= 1.5B × 2 B）
  若每次只传"变化的部分"？—— v0.8 **没有** delta/changed-ratio 机制（B 调研确认：
  checkpoint_engine/ 内无任何增量同步实现，send_weights 的 global_steps 形参在 NCCL
  实现体内未被使用）。所以每步都是**全量 3 GB**。
  时间估算：8 卡 NVLink/NVSwitch 有效带宽 ~200 GB/s 共享
    3 GB / 200 GB/s ≈ 15 ms（理想）
    实际含 ZMQ 元数据往返、chunk 切分、rank 同步、CUDA IPC 落地 → 常见 0.1~1 s 级
  ⇒ 对比：共卡 naive 走同卡内存/cuda ipc，量级更小（但省不掉"同步"这件事本身）

【一个可验证的顺序账】
  release_kv_cache_replicas()（放过 10 GB KV）
    → build_process_group（prepare 分配 2×2 GB buffer）
    → send/receive（3 GB 全量）
    → finalize（释放 buffer）
    → resume_kv_cache_replicas()（KV 重建 10 GB）
  峰值额外显存 = 2×bucket（4 GB）+ KV 重建过程中的临时占用
```

**三个结论**：① **每步都是全量 3 GB**（v0.8 无增量同步）——这是分卡方案的主要固定成本；② **bucket 越大吞吐越好但显存越贵**（2×bucket 常驻），共卡下根本不该用 NCCL（`naive` 零 buffer）；③ **共卡的 `naive` 路径本质上也是"全量 + 逐张量"**，靠的是"同卡内存直写"省掉网络——所以**共卡省的是带宽，不是拷贝**。

> **面试一句话总结**：权重同步的源头是 `engine.get_per_tensor_param()` 返回的**惰性生成器**（每个张量现取现 `.to(device).full_tensor()`，只额外持有一个张量的显存，不会复制整份模型）；共卡走 `mode="naive"`（`ColocatedCheckpointEngine` 只把生成器存下来、被消费时 `yield from`，零额外 buffer，`checkpoint_engine.backend` 默认就是 `naive`）；分卡走 `CheckpointEngineManager.update_weights()` 的 8 步——**abort 在飞请求 → 建临时 worker group → `release_kv_cache_replicas`（只放 KV 保留权重让传输直接写进原 buffer）→ `build_process_group` → trainer/rollout 并发 `update_weights` → `finalize` → 恢复 KV → 恢复生成**；传输本身是"命名张量 → `split_weight_chunks` 按 `bucket_size`（默认 2048 MB）切块 → uint8 字节流 → NCCL `broadcast(src_rank=0)` 一写多读（元数据走 ZMQ PUB/SUB）→ rollout 侧 CUDA IPC 落地"，**显存开销是 2×bucket_size**（docstring 明写），**且没有任何 delta/增量同步**，每步都是全量权重。

---

## 16. 项目 TQ 方案的一次完整 `_train_step`：从推理切到训练再切回来

### 1. 现有问题：TQ 化之后，训练循环长什么样

项目在 `verl-v0.8.0` 里加了一套 **TQ 原生训练循环**（`verl/trainer/main_ppo_sync.py`，1866 行，含 `ReplayBuffer` / `KVBatchMeta` / `AgentLoopWorkerTQ` / `AgentLoopManagerTQ`），然后 `agent-lightning` 仓的 `AgentLightningTrainer` **继承它并重写 `_train_step` / `fit`** 以插入 agent 模式。它和前面 1~11 节讲的 `RayPPOTrainer`（v0.3.0）**数据流完全不同**：

| 维度 | v0.3.0（`RayPPOTrainer`） | 项目 TQ 方案（`main_ppo_sync.PPOTrainer`） |
|---|---|---|
| 训练数据载体 | `DataProto`（张量全在 driver/worker 内存里） | **`KVBatchMeta`（只有 keys + tags，张量在 TQ）** |
| driver 持有的东西 | 完整 batch 的张量 | **只有 keys、tags、extra_info** |
| 中间结果（log_probs/entropy/advantage） | 留在 DataProto | **写回 TQ，driver 需要时再拉** |
| 等待 agent 完成 | `run_until_all_finished()` 轮询 store | **`ReplayBuffer` 轮询 `tq.kv_list()` + running barrier** |

`main_ppo_sync.py` 开头的 docstring 自己点明了这一点（文件中第 19 行附近）：`2. Use ReplayBuffer to sample data from TransferQueue.`

### 2. 方法论：`tqbridge`（元数据/数据分离）+ ReplayBuffer（屏障轮询）+ `_train_step`（九段式）

**（1）核机制：`tqbridge` 让 worker 方法"进 meta、出 meta"，真实张量走 TQ。** 装饰器在调用前后各做一次搬运（`verl-v0.8.0/verl/utils/transferqueue_utils.py:322-370`）：

```python
def decorator(func):
    pid = os.getpid()

    @wraps(func)
    def inner(*args, **kwargs):
        batch_meta = _find_meta(*args, **kwargs)          # ① 找到参数里的 BatchMeta/KVBatchMeta
        if batch_meta is None:
            return func(*args, **kwargs)                  # 没有 meta → 原样调用（不影响普通方法）
        else:
            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()                                 # ② worker 进程里首次使用时初始化 TQ 客户端
                TQ_INITIALIZED = True

            is_kv_batch_meta = isinstance(batch_meta, KVBatchMeta)
            if is_kv_batch_meta:
                tags = batch_meta.tags
                batch_meta = kv_batch_meta2batch_meta(batch_meta)
            t1 = time.time()
            args = [_meta_to_realdata(arg) if isinstance(arg, BatchMeta | KVBatchMeta) else arg for arg in args]
            kwargs = {k: _meta_to_realdata(v) if isinstance(v, BatchMeta | KVBatchMeta) else v for k, v in kwargs.items()}
            t2 = time.time()
            logger.info(f"Task {func.__name__} (pid={pid}) is getting len_samples={batch_meta.size}, cost time: {t2 - t1}")

            output = func(*args, **kwargs)                # ③ 真正执行（此时拿到的是张量）

            put_data = False
            if isinstance(output, TensorDict):
                if output.batch_size:
                    assert output.batch_size[0] == batch_meta.size, (
                        f"output batch size {output.batch_size} != meta size {batch_meta.size}")
                    put_data = True
            need_collect = _compute_need_collect(dispatch_mode, args) if dispatch_mode is not None else True
            if put_data and need_collect:
                updated_meta = _update_meta_with_output(output, batch_meta, func.__name__)   # ④ 输出写回 TQ
                if is_kv_batch_meta:
                    updated_meta = batch_meta2kv_batch_meta(updated_meta)
                    updated_meta.tags = tags
                return updated_meta
            return _postprocess_common(output, put_data, need_collect)
```

配合两个搬运函数（`transferqueue_utils.py:111-158`）：

```python
async def _async_meta_to_realdata(meta: BatchMeta | KVBatchMeta) -> TensorDict:
    if isinstance(meta, KVBatchMeta):
        meta = await async_kv_batch_meta2batch_meta(meta)
    meta_info = copy.deepcopy(meta.extra_info)
    tq_client = tq.get_client()
    tensordict = await tq_client.async_get_data(meta)         # ← 真正从 TQ 取张量
    for key, val in meta_info.items():
        if isinstance(val, (NonTensorData | NonTensorStack)):
            tensordict[key] = val
        else:
            tu.assign_non_tensor_data(tensor_dict=tensordict, key=key, val=val)
    return tensordict

async def _async_update_meta_with_output(output: TensorDict, meta: BatchMeta, func_name=None) -> BatchMeta:
    fields, meta_data = [], {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor | NonTensorStack):
            fields.append(k)                                   # 张量 → 进 TQ
        elif isinstance(v, NonTensorData):
            meta_data[k] = v.data                              # 非张量 → 留在 meta.extra_info 里传回
        ...
    meta = await tq_client.async_put(data=output.select(*fields), metadata=meta)
    meta.extra_info = meta_data
    return meta
```

**这就是"跨进程只传 keys、张量走 TQ"的实现**：`extra_info`（张量以外的配置，如 `mini_batch_size` / `epochs` / `seed` / `temperature` / `global_token_num`）随 meta 走控制面，**张量走数据面**。回忆第 4 节讲过的"控制流与数据流分离"——这里是它在**训练循环内部**的又一次应用。

**（2）下行：daemon 把 prompt 入队 + 写 `running` 屏障。** `agentlightning/verl/daemon.py:599-622`：

```python
if self.mode == "v1":
    # Enqueue all the tasks in a single batch
    rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)     # ① 任务进 LightningStore
    self._task_id_to_original_sample.update({...})
    self._total_tasks_queued += len(rollouts)

if global_steps is not None and is_train:
    import transfer_queue as tq
    for rollout_id in self._task_id_to_original_sample:
        tq.kv_put(
            key=rollout_id,
            partition_id="train",
            tag={"global_steps": global_steps, "status": "running"},                # ② TQ 里写"占位屏障"
        )
```

**"running 屏障"是很漂亮的设计**：trainer 不需要知道 agent 什么时候完成，只需要知道"TQ 里这批 rollout_id 的 status 什么时候不再是 running"。**等待状态本身也放进了数据面**，于是等待可以变成"带超时的轮询"而不是"一次阻塞"。

**（3）上行：daemon 把轨迹字段与 tags 批量写入 TQ。** `daemon.py:1108-1126`：

```python
tags.append({"seq_len": prompt_len + response_len, "is_drop": is_drop_list[i]})
tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)      # 轨迹 + tag 一起进 TQ
...
batch_meta = KVBatchMeta(
    keys=...,
    tags=...,
    partition_id=partition_id,
    ...
)
```

**tag 里塞了 `is_drop` 和 `seq_len`**——于是后面 `_train_step` 里"过滤超长 prompt"和"按长度做负载均衡"**都不用把张量拉回来**，只看 tags 就够了（`trainer.py:361-372`）：

```python
non_drop_mask = [not tag.get("is_drop", False) for tag in batch.tags]
if not all(non_drop_mask):
    valid_indices = [i for i, m in enumerate(non_drop_mask) if m]
    metrics["training/n_triplets_prompt_too_long"] = len(batch.keys) - len(valid_indices)
    batch = KVBatchMeta(
        keys=[batch.keys[i] for i in valid_indices],
        tags=[batch.tags[i] for i in valid_indices],
        partition_id=batch.partition_id,
        fields=batch.fields,
        extra_info=batch.extra_info,
    )
```

**（4）`ReplayBuffer`：后台线程轮询 `tq.kv_list()` + `sample()` 忙等屏障。** `verl-v0.8.0/verl/trainer/main_ppo_sync.py:194-294`：

```python
class ReplayBuffer:
    """Replay buffer periodically polls metadata from transfer queue."""

    def __init__(self, poll_interval: float = 1.0):
        # partition_id => {key: tags}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.poll_thread = threading.Thread(target=self._poll_from_transfer_queue, daemon=True)
        self.poll_thread.start()

    def _poll_from_transfer_queue(self):
        """Periodically poll metadata from transfer queue."""
        try:
            while not self._stop_event.is_set():
                data = tq.kv_list()                       # ← 只拉 key+tag，不拉张量
                if data is not None:
                    for partition_id, items in data.items():
                        self.add(partition_id, items)
                self._stop_event.wait(self.poll_interval)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error in _poll_from_transfer_queue: {e}")
                os._exit(1)                               # ← 轮询线程崩了直接退出进程（否则静默卡死）

    def sample(self, partition_id: str, global_steps: int = None, batch_size: int = None) -> KVBatchMeta:
        ...
        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                keys, tags = [], []
                should_wait = False
                partition = self.partitions[partition_id]
                for key, tag in partition.items():
                    if tag["global_steps"] == global_steps:
                        if tag["status"] == "running":
                            should_wait = True            # ← 还有没跑完的 → 继续等
                            break
                        elif tag["status"] == "success":
                            keys.append(key)
                            tags.append(tag)
                        else:
                            logger.debug(f"Unknown status {tag['status']} for key {key}")
                if not should_wait:
                    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
```

**这段 `sample()` 就是空泡的量尺**：`while True: time.sleep(poll_interval)` —— **训练进程在这里纯等**。项目把 `poll_interval` 设成 **3.0 秒**（`agentlightning/verl/trainer.py:490`：`self.replay_buffer = ReplayBuffer(poll_interval=3.0)`），意味着**平均要多等 1.5 s 才察觉"这批跑完了"**，这是最直接的空泡。

**（5）项目的 `_train_step` 九段式**（`agentlightning/verl/trainer.py:280-440`，逐行对应）：

```text
第 1 段 [gen]：① checkpoint_manager.wake_up_replicas()        ← 推理醒来（权重+KV 上 GPU）
              ② agent_mode_daemon.set_up_data_and_server(...)   ← prompt 入 store + 写 TQ running 屏障
              ③ replay_buffer.sample(partition_id="train", global_steps=...)  ← ★ 忙等，空泡在这里
              ④ agent_mode_daemon.clear_data_and_server()
              ⑤ checkpoint_manager.sleep_replicas()             ← 推理睡下，把显存让给训练
              （项目为这 5 步各打了一个计时器：gen_wake_replicas / gen_set_up /
                gen_replay_sample / gen_clear / gen_sleep_replicas）

第 2 段 [reward]：_compute_reward_colocate(batch)（colocate reward worker）

第 3 段：按 tags 过滤 is_drop（不拉张量）+ _balance_batch（upsample 复制样本做负载均衡，
         padding 样本打 is_padding tag）—— 替代了老版本的 pad → compute → unpad → floor_pad

第 4 段 [old_log_prob]：_compute_old_log_prob(batch, metrics)
第 5 段 [ref]：_compute_ref_log_prob(batch, metrics)（若 use_reference_policy）
第 6 段 [values]：_compute_values(batch, metrics)（若 use_critic）
第 7 段 [adv]：_compute_advantage(batch, metrics)
第 8 段 [update_critic] / 第 9 段 [update_actor]：_update_critic / _update_actor
```

**（6）第 4 段 `_compute_old_log_prob` 展示了 TQ 往返的"三段式"**（`verl-v0.8.0/verl/trainer/main_ppo_sync.py:1302-1361`）：

```python
# 1. compute log probs —— 把 meta 交给 worker，worker 通过 tqbridge 自己从 TQ 取数
batch.extra_info.update({"calculate_entropy": True, "compute_loss": False,
                         "temperature": self.config.actor_rollout_ref.rollout.temperature})
output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
assert len(output) == len(batch)

# driver 端再把需要的字段拉回来（只拉 3~5 个字段，不是整个 batch）
fields = ["entropy", "log_probs", "response_mask"]
if self.config.actor_rollout_ref.rollout.calculate_log_probs:
    fields.extend(["responses", "rollout_log_probs"])
data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

# 2. write old_log_probs and entropy back to TransferQueue
data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id,
                        fields=data.select("old_log_probs", "entropy"))

# 3. driver 端只为算指标把张量转成 padded
data = DataProto(batch=data.to_padded_tensor())
entropy_agg = agg_loss(loss_mat=data.batch["entropy"], loss_mask=data.batch["response_mask"], ...)
metrics.update({"actor/entropy": entropy_agg.detach().item()})
```

`response_from_nested` / `response_to_nested` 定义在 `verl-v0.8.0/verl/workers/utils/padding.py:196,215`——它们是"变长 nested 张量 ↔ padded 张量"的转换器。**注意这里埋了第 18 节那个 15 秒的坑**。

**（7）第 7 段 `_compute_advantage` 是"取—算—写"的完整闭环**（`main_ppo_sync.py:1405-1466`）：

```python
fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
response_mask = data["response_mask"]
data = DataProto(batch=data.to_padded_tensor())
data.batch["token_level_scores"] = data.batch["rm_scores"]
data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
# 1. KL penalty / 2. rollout correction（IS 权重、拒绝采样）/ 3. advantage
data = compute_advantage_for_multi_trajectories(data, batch_keys=batch.keys, adv_estimator=..., ...)
# 4. write nested advantages and returns back to TransferQueue
output = {}
for field in fields:                       # ["advantages", "returns", (+token_level_rewards/response_mask/...)]
    output[field] = response_to_nested(data.batch[field], response_mask)
batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
return batch
```

**注意 advantage 用的是项目自研的 `compute_advantage_for_multi_trajectories`（`main_ppo_sync.py:122`）而不是 verl 原生的 `compute_advantage`**——因为它要按 `batch_keys` 对多条轨迹（triplet）分组。

**（8）第 9 段 `_update_actor` 只往 `extra_info` 里塞超参、把 batch 原样交给 worker**（`main_ppo_sync.py:1490-1520`）：

```python
ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
extra_info = {
    "calculate_entropy": calculate_entropy,
    "global_batch_size": ppo_mini_batch_size,
    "mini_batch_size": ppo_mini_batch_size,
    "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
    "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
    "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
    "temperature": self.config.actor_rollout_ref.rollout.temperature,
}
batch.extra_info.update(extra_info)
output: TensorDict = self.actor_rollout_wg.update_actor(batch)      # ← worker 侧 tqbridge 自己取数
```

**（9）外层 `fit` 的完整顺序**（`agentlightning/verl/trainer.py:447-612`）：

```text
_load_checkpoint()
  → checkpoint_manager.update_weights()                  # ① 训练前把初始权重同步给推理侧
  → 创建 agent_mode_daemon（v1 模式，注入 store/llm_proxy/adapter）
  → self.replay_buffer = ReplayBuffer(poll_interval=3.0) # ② 启动后台轮询线程
  → tq.kv_list() → tq.kv_clear(...)                      # ③ 清理上次崩溃遗留的 TQ 残留数据
  → val_before_train（_validate 里也会 sleep/update_weights）
  → for epoch / for batch_dict:
        batch = self._train_step(batch_dict, metrics, timing_raw)   # ④ 上面那九段
        → bench_log(...)                                            # ⑤ 打印 20+ 个分阶段耗时
        → save_checkpoint（按 save_freq）
        → with _timer("update_weights"): checkpoint_manager.update_weights()   # ⑥ 同步新权重
        → _validate（按 test_freq）
        → _compute_metrics
        → tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)        # ⑦ 释放 TQ 里这一步的数据
        → global_steps += 1
```

**第 ⑦ 步的 `tq.kv_clear` 是最容易被忽略但最要命的一步**：TQ 用来存轨迹的对象**不会自己过期**（回忆 `TQ.md`：Mooncake 侧 eviction 被关掉、SimpleStorage 是计数配额），所以每步必须显式 `kv_clear`，否则**几十步内就会把存储写满**。

### 3. 具体数值样例

用项目自己打的计时器名，演算一个 step 的时间线（数字取自 `TQ_PERF_COMPARE_AND_FIX.md` 的量级与本次阅读到的代码结构，标注为**示意**）：

```text
环境：单机 16 卡（昇腾 NPU），ppo_mini_batch_size=32，rollout.n=4，
      batch ≈ 245 个 triplet（pad 后为 128 的倍数），micro_batch_size_per_gpu=4

t0.0  gen_wake_replicas      ≈ 1~3 s     vLLM 醒来（权重 + KV cache 上显存）
t0.1  gen_set_up             ≈ 1~2 s     32×4=128 个 rollout 入 store + 128 条 running 屏障写 TQ
t0.2  gen_replay_sample      = ？        ★ 忙等：agent 真正跑完 rollout 的时间
                                         （poll_interval=3.0 s，所以至少有 ~1.5 s 的探测延迟）
                                         这一步包含：agent 多轮 LLM 调用 + 工具调用 + 轨迹写 TQ
t0.3  gen_clear              < 0.5 s     clear_data_and_server
t0.4  gen_sleep_replicas     ≈ 1~2 s     vLLM 睡下（level 2 全丢 or level 1 备份权重）
      ⇒ gen（合计）= 上面 5 项相加 —— **其中 gen_replay_sample 是空泡主体**

t1    data_prep_total        ≈ 0.01 s    按 tags 过滤 is_drop + _balance_batch（**不碰张量**）
      reward                 ≈ 0.1~1 s   colocate reward
t2    old_log_prob           ≈ ?         worker 从 TQ 取数（~0.5 s）+ 前向 + 写回 old_log_probs/entropy
t3    ref                    ≈ ?         （项目实测比 baseline **快约 2 s**）
t4    values                 ≈ ?         若 use_critic
t5    adv                     小          取 7 个字段 → 算 advantage → 写回 advantages/returns
t6    update_critic          ≈ ?
t7    update_actor           ≈ ?         ★ 项目实测比 baseline **慢约 15 s**（见第 18 节根因）
t8    update_weights         ≈ ?         项目实测比 baseline **慢约 5 s**

【TQ 读写次数（一个 step，粗算）】
  下行：1 次 kv_put × 128（running 屏障）
  worker 侧取数：old_log_prob / ref / values / update_actor 各 1 次 kv_batch_get（每个 worker rank）
  中间结果写回：old_log_probs+entropy、advantages+returns、（可选）values → 3~5 次 kv_batch_put
  结束清理：1 次 kv_clear（128 个 key）
  ⇒ **一个 step 有 10 次以上的 TQ 批量往返**，每次都要跨进程/可能跨机

【每阶段的 GPU 归属】
  gen 段：GPU 归 vLLM（训练侧 offload 在 CPU）
  训练段：GPU 归 FSDP（vLLM 已 sleep）
  update_weights 段：两边都在动（vLLM 醒着收权重、训练侧还要取 state_dict）
  ⇒ 这就是共卡的"接力棒"模型，任何一段拖长都直接变成空泡
```

> **面试一句话总结**：项目 TQ 化训练循环的核心是 **`tqbridge`（元数据/数据分离）+ `ReplayBuffer`（屏障轮询）+ `KVBatchMeta`（只有 keys/tags）**——`tqbridge` 在 worker 方法调用前后用 `_meta_to_realdata` / `_update_meta_with_output` 把张量从 TQ 取来再写回，**跨进程只传 keys 和 `extra_info`**；下行时 daemon 用 `tq.kv_put(tag={"global_steps","status":"running"})` 写"running 屏障"，agent 完成后 `tq.kv_batch_put(keys, fields, tags)` 把轨迹和 `is_drop`/`seq_len` tag 一起写进 TQ；`ReplayBuffer` 用后台线程轮询 `tq.kv_list()`（只拉 key+tag），`sample()` 忙等到该 `global_steps` 的所有记录不再是 running——**这段 `while True: time.sleep(3.0)` 就是共卡方案里最直接的空泡**；`_train_step` 是九段式（wake → set_up → **sample 忙等** → clear → sleep → reward → 过滤/balance → old_log_prob/ref/values → adv → critic/actor），外层 `fit` 每步还会 `update_weights` 同步权重并 `tq.kv_clear` 显式释放 TQ 数据（**不 clear 会把存储写满**）。

---

## 17. 训练侧每一步的内部细节：micro-batch → forward → loss → backward → clip → step

### 1. 现有问题：`update_actor` 里面到底发生了什么

`_update_actor` 在 driver 上只是"塞超参 + 交给 worker"，真正的训练全在 worker 的 `TrainingWorker` 里。面试问"每一步训练细节"时，要能按顺序说出：**mini-batch 怎么分、micro-batch 怎么分、梯度累积靠什么、loss 怎么组合、clip 和 step 在哪、lr 什么时候走**。

### 2. 方法论：五层调用栈，逐层落实到行号

**第 1 层：`TrainingWorker.train_mini_batch` —— 按 mini-batch 循环，`epochs` 个 epoch**（`verl-v0.8.0/verl/workers/engine_workers.py:234-321`）：

```python
def train_mini_batch(self, data: TensorDict) -> TensorDict:
    maybe_fix_3d_position_ids(data)
    batch_size_per_dp = data.shape[0]
    disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
    mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
    num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
    epochs = tu.pop(data, key="epochs", default=1)
    seed = tu.pop(data, key="seed", default=42)
    ...
    if mini_batch_size is None:
        assert batch_size_per_dp % num_mini_batch == 0
        mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
    else:
        assert mini_batch_size % self.engine.get_data_parallel_size() == 0
        mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

    dataloader = tu.make_iterator(data, mini_batch_size=mini_batch_size_per_gpu,
                                  epochs=epochs, seed=seed + self.engine.get_data_parallel_rank(), ...)
    with (self.engine.train_mode(disable_auto_offload=disable_auto_offload), Timer(name="train_batch", logger=None)):
        output_lst = []
        total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs
        for batch_idx, mini_batch_td in enumerate(dataloader):
            ...
            tu.assign_non_tensor(mini_batch_td, global_token_num=NonTensorData(global_token_num),
                                 update_lr_scheduler=batch_idx == total_num_iterations - 1,
                                 disable_auto_offload=True)
            actor_output = self.train_batch(mini_batch_td)
            output_lst.append(actor_output)
```

**三个要点**：① **`mini_batch_size` 是全局值，要除以 `data_parallel_size` 才是每卡值**（`mini_batch_size_per_gpu`）；② **`train_mode` 在这一层进出**——所以 onload/offload 每个 `train_mini_batch` 只发生一次（内部都 `disable_auto_offload=True`）；③ **`update_lr_scheduler` 只在最后一个 iteration 为 True**——lr scheduler 每个 `update_actor` 只走一步，而不是每个 mini-batch 走一步。

**第 2 层：`TrainingWorker.train_batch` —— 一次 mini-batch 的训练 + 记录 lr 和耗时**（`engine_workers.py:325-377`）：

```python
with (self.engine.train_mode(disable_auto_offload=disable_auto_offload),
      Timer(name="train_batch", logger=None) as timer):
    output = self.engine.train_batch(data, loss_function=self.loss_fn)
delta_time = timer.last

update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
if update_lr_scheduler:
    lr = self.engine.lr_scheduler_step()          # ← lr_scheduler.step() 在这里发生
...
output["metrics"]["lr"] = lr
final_output = self._postprocess_output(output, global_token_num=global_token_num,
                                       delta_time=delta_time, forward_only=False, ...).cpu()
```

**第 3 层：`BaseEngine.train_batch` —— 四步骨架**（`workers/engine/base.py:112-131`）：

```python
def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
    maybe_fix_3d_position_ids(data)

    self.optimizer_zero_grad()                                          # ① 清梯度
    outputs = self.forward_backward_batch(data, loss_function, forward_only=False)   # ② 前向+反向
    grad_norm = self.optimizer_step()                                   # ③ clip + step
    if self.is_mp_src_rank_with_outputs():
        assert "grad_norm" not in outputs["metrics"]
        outputs["metrics"]["grad_norm"] = grad_norm
    return outputs
```

**第 4 层：`forward_backward_batch` —— micro-batch 切分 + 逐块前向反向（梯度累积在这里）**（`workers/engine/fsdp/transformer_impl.py:617-654`）：

```python
def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
    tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

    # compute num_tokens in global batch for loss normalization
    batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
    torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM,
                                 group=self.get_data_parallel_group())        # ① 全局 token 数（跨 DP）
    tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
    tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

    micro_batches, indices = prepare_micro_batches(
        data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True)   # ② 切 micro-batch

    output_lst = []
    ctx = torch.no_grad() if forward_only else nullcontext()
    scaler = getattr(self, "scaler", None)
    for micro_batch in micro_batches:
        with ctx:
            loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function,
                                                forward_only=forward_only)     # ③ forward + loss
            if not forward_only:
                if scaler is not None:
                    scaler.scale(loss).backward()                              # ④ AMP 路径
                else:
                    loss.backward()                                            # ④ 普通路径
        output_lst.append(meta_info)
    return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
```

**四个必须能说出口的细节**：

| 细节 | 说明 |
|---|---|
| **梯度累积靠什么** | **没有 `no_sync()`**——FSDP 路径里**不存在** `no_sync` 包裹（`no_sync` 只在 Megatron 路径出现），累积语义完全由"**micro-batch 之间不 `zero_grad`、只在 `train_batch` 开头清一次**"保证；每个 micro-batch 的反向都会触发 FSDP 自己的 reduce-scatter |
| **`batch_num_tokens` 为什么 all_reduce** | loss 归一化要用**全局**有效 token 数（跨所有 DP rank），否则各 rank 的 loss 尺度不一致，梯度会被错误加权 |
| **`sp_size` 为什么要 assign** | Ulysses 序列并行下，micro-batch 切分要按 SP 组对齐 |
| **`forward_only` 走 `torch.no_grad()`** | ref log prob / log prob 计算复用同一个 `forward_backward_batch`，靠 `ctx` 切换 |

**第 5 层：`optimizer_step` —— unscale → clip → step（含"梯度非有限就跳过更新"）**（`fsdp/transformer_impl.py:665-711`）：

```python
def optimizer_step(self):
    assert self.optimizer_config.clip_grad is not None
    scaler = getattr(self, "scaler", None)

    # Unscale gradients before clip so the clip threshold is applied to true gradient
    # magnitudes, not scaled ones. scaler.step() will skip the update if any grad is inf/nan.
    if scaler is not None:
        scaler.unscale_(self.optimizer)

    if isinstance(self.module, FSDP):
        grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)          # FSDP1
    elif isinstance(self.module, FSDPModule):
        grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)  # FSDP2
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)

    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()

    if scaler is not None:
        scaler.step(self.optimizer)                   # scaler 内部检查 inf/nan 并可能跳过
        scaler.update()
    else:
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()                # ← 非有限就**跳过这一步更新**
        else:
            self.optimizer.step()
    ...
    return grad_norm.item()
```

**lr scheduler 单独一步**（`fsdp/transformer_impl.py:713-719`）：

```python
def lr_scheduler_step(self):
    """Advance FSDP scheduler and return updated learning rate."""
    self.lr_scheduler.step()
    lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
    return lr
```

**loss 的组合在 `verl-v0.8.0/verl/workers/utils/losses.py:103-142` 的 `ppo_loss` 里**（由 `engine_workers.py:580-584` 用 `partial(ppo_loss, config=actor_config)` 绑成 `self.loss_fn`）：

```python
pg_loss, pg_metrics = policy_loss_fn(old_log_prob=old_log_prob, log_prob=log_prob,
                                     advantages=advantages, response_mask=response_mask,
                                     loss_agg_mode=loss_agg_mode, config=config,
                                     rollout_is_weights=rollout_is_weights)
policy_loss = pg_loss

# add entropy loss
if entropy is not None:
    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                            **config.global_batch_info)
    policy_loss -= entropy_coeff * entropy_loss

# add kl loss
if config.use_kl_loss:
    ref_log_prob = data["ref_log_prob"]
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode,
                       **config.global_batch_info)
    policy_loss += kl_loss * config.kl_loss_coef
```

以及那句 **`to_padded_tensor`**（`losses.py:85-91`）——它就是第 18 节 15 秒坑的现场：

```python
# select fields and convert to padded tensor
fields = ["response_mask", "old_log_probs", "advantages"]
if "rollout_is_weights" in data:
    fields.append("rollout_is_weights")
if "ref_log_prob" in data:
    fields.append("ref_log_prob")
data = data.select(*fields).to_padded_tensor()
```

### 3. 具体数值样例

```text
场景：ppo_mini_batch_size=32（全局），rollout.n=4，8 卡 DP，
      micro_batch_size_per_gpu=4，ppo_epochs=1，grad_clip=1.0，entropy_coeff=0

【一次 update_actor 的完整数字账】
mini_batch_size_per_gpu = 32 / 8 = 4 个样本/卡/次？
  注意：项目里传进来的 mini_batch_size 已经乘过 rollout.n：
    _update_actor: ppo_mini_batch_size = 32 * 4 = 128（全局）
    ⇒ mini_batch_size_per_gpu = 128 / 8 = 16（每卡每次 16 个 triplet）
micro_batch_size_per_gpu = 4
⇒ 每个 mini-batch 有 16 / 4 = 4 个 micro-batch（4 次前向 + 4 次反向）
epochs = 1（PPO 通常 1；若 epochs=2 则整个 mini-batch 序列跑两遍）

【执行序列（每个 mini-batch）】
1. engine.train_mode 已在最外层进入（onload 一次）
2. optimizer_zero_grad()                       ← 清一次
3. for micro_batch in 4 个:
     micro_batch → device
     forward_step:
       prepare_model_inputs（remove-padding 路径）
       module(**inputs, use_cache=False)       ← 明确禁用 KV cache
       prepare_model_outputs（算 logits → log_prob / entropy）
       loss_function = ppo_loss:
         to_padded_tensor(response_mask/old_log_probs/advantages/ref_log_prob)
         pg_loss = policy_loss_fn(...)          ← clip 在 policy_loss_fn 内
         （entropy_coeff=0 → 不算 entropy loss）
     loss.backward()                            ← 累加梯度，不清零（= 梯度累积）
4. optimizer_step():
     scaler.unscale_（有 scaler 时）
     FSDP1: module.clip_grad_norm_(1.0) / FSDP2: fsdp2_clip_grad_norm_(1.0)
     grad_norm 非有限 → zero_grad 跳过；否则 optimizer.step()
5. 最后一个 mini-batch 后：engine.lr_scheduler_step()   ← lr 走一步
6. 退出 train_mini_batch → train_mode.__exit__ → zero_grad + offload 到 CPU

【显存峰值估算（每卡）】
  参数 0.375 + 梯度 0.375 + Adam 1.5 + master 0.75 ≈ 3 GB
  + 激活（micro_batch=4，remove-padding 后按 token 数；设 4×2048 token）
  ⇒ 这就是 micro_batch_size_per_gpu 存在的意义：**用更小的 micro-batch 换激活显存**

【一个常见误解】
  "梯度累积 = 一次大 batch" —— 数值上近似，但：
  ① 每个 micro-batch 的 backward 都会触发 FSDP reduce-scatter（通信量不省）
  ② 没有 no_sync，所以 4 个 micro-batch = 4 轮 all-reduce 级通信
  ③ 真正省的是激活显存，不是通信
```

> **面试一句话总结**：训练侧是五层调用栈——`TrainingWorker.train_mini_batch`（按 mini-batch × epochs 循环，`mini_batch_size` 是全局值需除以 DP size，`train_mode` 在最外层进出一次、内层 `disable_auto_offload=True`，`update_lr_scheduler` 只在最后一次 iteration 为真）→ `TrainingWorker.train_batch`（计时 + 调 `engine.lr_scheduler_step()`）→ `BaseEngine.train_batch`（`optimizer_zero_grad` → `forward_backward_batch` → `optimizer_step`）→ `forward_backward_batch`（先 `all_reduce` 出全局 `batch_num_tokens` 供 loss 归一化，再切 micro-batch，逐块 `forward_step` + `loss.backward()`，**梯度累积靠"micro-batch 间不清零"，FSDP 路径没有 `no_sync`**）→ `optimizer_step`（`scaler.unscale_` → `clip_grad_norm_(grad_clip)` → **`grad_norm` 非有限就 `zero_grad` 跳过更新** → `optimizer.step()`）；loss 组合在 `workers/utils/losses.py` 的 `ppo_loss` 里（`pg_loss - entropy_coeff*entropy_loss + kl_loss*kl_loss_coef`），其中 `to_padded_tensor()` 那一行是下一节性能坑的现场。

---

## 18. 时间账：空泡在哪、TQ 方案的实测代价与修复

### 1. 现有问题：共卡方案的每一分代价都要落到具体阶段上

前面 12~17 节把机制讲完了，但**"到底哪里慢"必须用数字回答**。项目自己写了一份对比文档 `agent-lightning/TQ_PERF_COMPARE_AND_FIX.md`（181 行），把 TQ 化之后的训练循环与 **无 TQ 的 baseline**（`upgrade/verl-0.8.0` 分支，继承 `RayPPOTrainer`）逐阶段对比——这是本节全部数字的来源。

### 2. 方法论：先定位到阶段，再定位到行

**（1）对比设置**（`TQ_PERF_COMPARE_AND_FIX.md:5`）：分支 `master`（TQ 接入，`AgentLightningTrainer(PPOTrainer)`）vs 基线 `upgrade/verl-0.8.0`（无 TQ，`AgentLightningTrainer(RayPPOTrainer)`）；环境单机 16 卡（昇腾 NPU），TQ `0.1.6`，verl `release/v0.8.0`。

**（2）逐阶段实测差异**（`TQ_PERF_COMPARE_AND_FIX.md:9-16`）：

| 阶段 | 观测 |
|---|---|
| 生成 rollout | 基本相当 |
| ref + old_log_prob | master **快约 2s** |
| **update_actor** | master **慢约 15s** |
| update_weights | master **慢约 5s** |

**（3）先排除的四件事**（`TQ_PERF_COMPARE_AND_FIX.md:18-25`，这部分方法论很值得学）：

1. **trainer 基类不同**（`PPOTrainer` vs `RayPPOTrainer`）——但两边 worker 类（`ActorRolloutRefWorker`）、FSDP engine、`ppo_mini_batch_size=32`、`rollout.n=4`、`ppo_epochs` **完全一致**；
2. **batch 规模一致**——三元组数量两边相同（**245 左右**，pad 后都是 128 的倍数）；
3. **TQ 读取不是瓶颈**——`update_actor` 内 worker 侧 TQ 读取**仅约 0.5s**；
4. 于是**剩余最大差异 = 训练数据的字段布局**。

**（4）根因（已定位到具体行）**：`ppo_loss` 里的 `data.select(*fields).to_padded_tensor()`（`verl/workers/utils/losses.py:85-91`）：

- **baseline**：`left_right_2_no_padding` 只把 `input_ids / position_ids / loss_mask` 转成 nested，`response_mask / old_log_probs / ref_log_prob / advantages / returns / token_level_scores / rm_scores` 都是**普通 padded 张量** → `to_padded_tensor()` 是 **no-op，零成本**；
- **master**：从 TQ 读出的数据**几乎全部是 nested**（TQ 以变长 NestedTensor 存储，`_compute_*` 阶段用 `response_from_nested / response_to_nested` 写回）→ **每个 micro-batch 都要对多个 nested 字段真转换**，加上 `index_select_tensor_dict`、`micro_batch.to(device)`、nested 求和等操作都作用在 10+ 个 nested 字段上。

**原文的量化解释**（`TQ_PERF_COMPARE_AND_FIX.md:40`）：

> 在昇腾 NPU 上 `torch.nested` 算子多为慢速/回退路径，4 个 micro-batch（16 样本/卡，`micro_batch_size_per_gpu=4`）的累计开销就是 ~15s。

**并且它顺带解释了"为什么 infer 反而快 2s"**（`:42`）：

> old_log_prob/ref 是纯前向，不吃 `ppo_loss` 的 nested→padded 路径，master 的原始变长数据还省掉了 baseline 的 pad/unpad 与 `left_right_2_no_padding` 的 GPU unpad 开销。

**"update_weights 慢 5s"是另一个独立因素**（`:44`）：TQ 栈（controller + 8 个 storage unit actor + store server + llm proxy + replay buffer 轮询线程 + 每步上百次临时线程/事件循环）与 FSDP `param_offload=true / optimizer_offload=true` 的 **host 侧传输/汇聚抢 CPU 和内存带宽**。

**（5）修复方案（三条，A 是主方案）**。

**方案 A（推荐，最小改动）：把 loss 字段一次性转 padded**——在 `TrainingWorker.train_mini_batch` 入口（`maybe_fix_3d_position_ids` 之后）加一段，**每个 `update_actor` 只转一次**，而不是每个 micro-batch 在 `ppo_loss` 里转 4 次（`TQ_PERF_COMPARE_AND_FIX.md:118-143`）：

```python
def train_mini_batch(self, data: TensorDict) -> TensorDict:
    """Split a batch into N mini-batches run for multiple epochs"""
    maybe_fix_3d_position_ids(data)
    # ---- TQ perf fix: 一次性把 loss 相关 nested 字段转 padded ----
    # 保留 input_ids/position_ids 为 nested（引擎 remove-padding 前向路径需要），
    # 其余按行字段转成普通 padded 张量，避免 ppo_loss/value_loss 每 micro-batch 重复转换。
    for _key in (
        "response_mask", "loss_mask", "old_log_probs", "ref_log_prob", "advantages",
        "returns", "token_level_rewards", "token_level_scores", "rm_scores", "entropy",
    ):
        _val = data.get(_key, None)
        if isinstance(_val, torch.Tensor) and _val.is_nested:
            data[_key] = torch.nested.to_padded_tensor(_val, padding=0.0)
    # ----------------------------------------------------------------
    batch_size_per_dp = data.shape[0]
    ...
```

要点（`:145-150`）：① `input_ids / position_ids` **保持 nested**（`prepare_model_inputs` 的 `use_remove_padding` 前向路径需要）；② `loss_mask` 转 padded 后 `batch_num_tokens = data["loss_mask"].sum()` **结果不变**（padding 补 0）；③ `ppo_loss` 的 `to_padded_tensor()` 变成 no-op，且 `index_select_tensor_dict` / `micro_batch.to(device)` 都改为作用在普通张量上；④ 该函数**被 actor 和 critic 共用**，一处改动两边受益。

**方案 B（可选）：`update_actor` 只读训练必需字段**——借 `KVBatchMeta.fields` + `tqbridge` 的 `select_fields` 限定字段（`verl/utils/transferqueue_utils.py:270` 的 `kv_batch_meta2batch_meta` 已支持），减少每 worker 物化的 nested 字段数量。

**方案 C（update_weights 慢 5s 的缓解）**（`:161-166`）：
- TQ controller / storage unit 限定独立 CPU（placement group / `num_cpus`）；
- `SimpleStorage.num_data_storage_units` 从 8 降到 1~2（减少 ZMQ 序列化与 host 线程）；
- `TQ_NUM_THREADS` 调低（默认 8）；
- 对照实验：临时关掉 store 的 `_update_tq_reward_async` 与 `ReplayBuffer.poll_interval`，看 `update_weights` 是否恢复 baseline，确认是否 CPU 竞争。

**（6）预期收益与验证闭环**（`:168-181`）：`update_actor` 从 ~17s 回到 ~2-5s；`update_weights` 目标 -5s；同时保住 ref/old_log_prob 的 -2s 与 TQ 零拷贝传输优势。验证要用**同一套插桩**再跑 3~5 步，并确认 `[BENCH-LOSS] to_padded_tensor` 降到 ~0.000s、训练指标（`actor/loss`、`actor/grad_norm`、`global_seqlen/*`）与修复前一致——**防止转换引入数值偏差**。

### 3. 具体数值样例

```text
【TQ 方案的净账（16 卡，单 step）】
  推理 rollout：            ≈ 相当（没有明显变化）
  ref + old_log_prob：      -2 s   （变长数据省掉 pad/unpad，纯前向不吃 nested→padded）
  update_actor：           +15 s   （4 个 micro-batch × nested→padded 真转换，NPU 上 nested 走慢路径）
  update_weights：          +5 s   （TQ 栈的 host 线程 + FSDP offload 的 host 带宽竞争）
  ─────────────────────────────────
  合计：                   +18 s/step（TQ 化带来的净代价）

【修复后的目标】
  update_actor：17 s → 2~5 s（方案 A：一次性转 padded，省掉 4× 的重复转换）
  update_weights：-5 s（方案 C：把 TQ 栈的 CPU 占用与 offload 隔开）
  净值：约 -2 s（即 TQ 化后**比 baseline 还快 2s**，且保留零拷贝传输）

【空泡分类（共卡视角，对回 Colocate-vs-Disaggregate.md）】
  ① 长尾空泡：gen_replay_sample 里等"最慢的那个 rollout"
     → 项目用 poll_interval=3.0 s 轮询，平均多等 1.5 s；TQ 方案靠 _balance_batch
       的 upsample + is_padding tag 把长尾样本"复制补齐"，但**等待本身没消除**
  ② 切换空泡：gen_wake_replicas + gen_sleep_replicas（vLLM 醒/睡各 1~3 s）
     → 这就是共卡 baseline 的**固有成本**，只能靠减少切换次数（one-step-off / 部分 rollout）缓解
  ③ 同步空泡：update_weights（全量 3 GB，v0.8 **无增量同步**）+ 屏障轮询的探测延迟
     → 修复方向：delta 同步（checkpoint_engine 目前没有）、把 poll_interval 改成事件通知

【结论：共卡 baseline 的天花板在哪】
  共卡 = 推理 + 切换 + 训练 串行 ⇒ GPU 利用率上限 ≈ 1/(1 + 切换/计算)
  若切换开销占比 10% → 利用率上限 ~90%；占比 30% → ~77%
  要突破这个上限，就必须让"推理与训练重叠"⇒ 回到分卡/异步路线
  （这正是 Colocate-vs-Disaggregate.md 里 7 种方法要解决的问题）
```

**三个结论**：① **共卡是 baseline，不是终点**——它的固有成本是"推理与训练串行 + 每次切换的显存交接"；② **TQ 化的代价是可定位、可修复的**：`update_actor +15s` 的根因被精确到了 `losses.py:85-91` 的 `to_padded_tensor()` 与"TQ 存的是 nested 而 baseline 存的是 padded"这一字段布局差异，修复只需在 `train_mini_batch` 入口转一次；③ **排查方法论比结论更值钱**：先排除"基类/规模/TQ 读取"三个变量，再把差异收敛到"同一行代码在不同数据布局下的行为"，最后给出最小改动的修复与验证闭环（含数值一致性检查）。

> **面试一句话总结**：项目用 `TQ_PERF_COMPARE_AND_FIX.md` 做了一次教科书式的性能归因——先排除 trainer 基类、batch 规模（两边约 245 个 triplet）、TQ 读取（仅 0.5s）三个变量，把差异收敛到"**TQ 以 nested 存储而 baseline 以 padded 存储**"这一字段布局差异上，根因精确到 `verl/workers/utils/losses.py:85-91` 的 `to_padded_tensor()`：baseline 是 no-op、TQ 版每个 micro-batch 都要真转换 10+ 个 nested 字段，在 NPU 上 4 个 micro-batch 累计 **~15s**；`update_weights +5s` 则是 TQ 栈的 host 线程/带宽与 FSDP `param_offload` 的竞争；修复方案 A 是在 `train_mini_batch` 入口**一次性**把 loss 相关字段转 padded（保留 `input_ids/position_ids` 为 nested），预期 `update_actor` 回到 2~5s、整体比 baseline 还快 ~2s——**共卡的空泡分三类：长尾（等最慢 rollout，`poll_interval=3.0s` 的轮询延迟）、切换（wake/sleep 各 1~3s）、同步（全量 3 GB 权重无增量）**，而共卡的利用率上限就是 $1/(1+\text{切换}/\text{计算})$，想突破就必须回到分卡/异步路线。

---

## 附：组件速查表

| 组件 | 代码位置（v0.3.0） | 角色 | 关键接口 / 类 |
|---|---|---|---|
| Algorithm | `algorithm/` | 大脑：派任务、学数据、更新资源 | `Algorithm.run()`；`VERL` / `APO` / `Baseline` |
| Runner | `runner/` | 工人：领任务、跑 agent、写轨迹 | `Runner`；`LitAgentRunner` |
| LightningStore | `store/` | 数据库+队列：解耦两侧 | `enqueue/dequeue_rollout`、`add_otel_span`、`query_spans`、`update_attempt` |
| Tracer | `tracer/` | 轨迹采集：插桩自动捕获 spans | `Tracer`；`OtelTracer`（只收 AGL 信号）/ `AgentOpsTracer`（带第三方库插桩） |
| Span 模型 | `types/tracer.py:251` | 跨机可序列化的规范轨迹单元 | `Span`（`rollout_id`/`attempt_id`/`sequence_id`/`trace_id`/`span_id`/`parent_id`）；`Span.from_opentelemetry` |
| 插桩补丁 | `instrumentation/agentops.py` | 把 vLLM 的 token id / logprob 塞进 span 属性 | `_patch_new_agentops` / `_patch_old_agentops`（patch `handle_chat_attributes`）；`Bypassable*Exporter`（不发 agentops 云端） |
| vLLM 插桩 | `instrumentation/vllm.py` | 让 vLLM 响应带上 `prompt/response_token_ids` | `ChatCompletionResponsePatched`；`instrument_vllm()`（已上游化到 vLLM ≥ v0.10.2） |
| Span 落库 | `tracer/otel.py:308` | 每个 span 结束时上传 store | `LightningSpanProcessor.on_end()`（专属 `otel-loop` 线程，10s 超时，失败只落本地） |
| emitter | `emitter/` | agent 主动上报 reward / message / object / exception | `emit_reward` → `emit_annotation` → `get_active_tracer().create_span()`（须在 trace_context 内） |
| Span 树 | `adapter/triplet.py:99` | 扁平 span 列表 → 树（含容错与重挂） | `TraceTree.from_spans`（补丢失父节点 + `virtual-root`）、`repair_hierarchy()`、`agent_name()`（7 种框架识别） |
| Adapter | `adapter/` | 轨迹→训练样本 | `TraceAdapter`；`TracerTraceToTriplet`（`find_llm_calls` → `span_to_triplet` → `match_rewards` → `to_trajectory`） |
| LLMProxy | `llm_proxy.py` | agent↔模型的桥 + 插桩 + 换模型 | `LLMProxy`；`ProxyLLM`（`get_base_url` 拼 `/rollout/{rid}/attempt/{aid}`）；`RolloutAttemptMiddleware`（注入 `x-rollout-id`/`x-attempt-id`/`x-sequence-id`）；`LightningSpanExporter`（子树缓冲批量导出） |
| OTLP 传输 | `utils/otlp.py` | span 的 protobuf 批量上传协议 | `LightningStoreOTLPExporter`、`handle_otlp_export`、`spans_from_proto`（`POST /v1/traces`） |
| Server/Client | `store/client_server.py` | 同一接口的跨机 HTTP 化 | `LightningStoreServer`（`otlp_traces=True`，`/v1/agl/*` + `/v1/traces`）、`LightningStoreClient`（`_request_json` 重试 + 按 event loop 缓存 session）、`_call_store_method`（同进程直调 / 跨进程 HTTP） |
| Trainer | `trainer/` | 总装：接线所有组件 | `Trainer.fit()` |
| ExecutionStrategy | `execution/` | 部署形态：线程 or 进程/机器 | `SharedMemoryExecutionStrategy` / `ClientServerExecutionStrategy` |
| verl 集成 | `verl/` + `algorithm/verl/` | RL 训练后端 | `AgentLightningTrainer(RayPPOTrainer)`、`AgentModeDaemon`、`VERL(Algorithm)` |

> 下面 5 行属于**第四部分（共卡 / verl 0.8.0 engine 架构 + 项目 TQ 接入仓）**，与上表的 v0.3.0 组件不属于同一份代码：

| 组件（v0.8.0 / TQ 仓） | 代码位置 | 角色 | 关键接口 / 类 |
|---|---|---|---|
| RolloutMode | `verl/workers/rollout/replica.py:54` | 推理与训练的三种共存方式 | `HYBRID`（同进程共卡，切权重）/ `COLOCATED`（同卡分进程，不切权重）/ `STANDALONE`（分卡） |
| Training engine | `verl/workers/engine/base.py:112` + `verl/workers/engine/fsdp/transformer_impl.py` | 训练引擎：onload/offload + 前向反向 | `BaseEngine.train_batch`、`engine.to(device, model, optimizer, grad)`、`BaseEngineCtx._context_switch`、`FSDPEngine.optimizer_step` / `get_per_tensor_param` |
| CheckpointEngine | `verl/checkpoint_engine/base.py:96` | 训练→推理的权重同步抽象（6 个后端） | `CheckpointEngineManager`、`sleep_replicas` / `wake_up_replicas` / `release_kv_cache_replicas` / `update_weights`、`split_weight_chunks` |
| vLLM sleep/wake | `verl/workers/rollout/vllm_rollout/vllm_async_server.py:604` | 推理侧显存交接 | `wake_up(tags)` / `sleep()` / `_sleep_hybrid()`；vLLM 侧落到 `CuMemAllocator.sleep(offload_tags)` |
| TQ bridge / ReplayBuffer | `verl/utils/transferqueue_utils.py:298` + `verl/trainer/main_ppo_sync.py:194` | TQ 原生训练循环（元数据/数据分离） | `tqbridge`、`_meta_to_realdata` / `_update_meta_with_output`、`ReplayBuffer.sample`、`KVBatchMeta` |
| TQ 版 AGL Trainer | `agent-lightning/agentlightning/verl/trainer.py:165` | 继承 `main_ppo_sync.PPOTrainer` 并重写训练循环 | `AgentLightningTrainer._train_step`（九段式 + 分阶段计时）/ `fit` |

### 通信方式速查（哪些是 HTTP、哪些是实例直接调用）

| 通信对 | shared-memory 模式 | client-server 模式（**三机分离：我们的部署**） | 本质 |
|---|---|---|---|
| Algorithm → Store | 实例直接调用（同进程） | **HTTP**（algo 机经 `LightningStoreClient` 连 store 机） | store 独立机器时必走 HTTP |
| Runner → Store | 实例直接调用（`LightningStoreThreaded`） | **HTTP**（agent 机经 `LightningStoreClient` → store 机 `LightningStoreServer` `/v1/agl/*`） | 跨进程/跨机器时唯一通道 |
| Runner → Agent | **实例直接调用**（`agent.training_rollout()`） | **实例直接调用**（不变） | Agent 与 Runner 永远同进程 |
| Agent → LLMProxy | **HTTP** | **HTTP** | proxy 是 FastAPI 独立服务 |
| LLMProxy → vLLM/模型 | **HTTP** | **HTTP** | 转发到后端 chat 端点 |
| Algorithm → Adapter | 实例直接调用 | 实例直接调用 | 纯本地转换 |
| 任意 → Store OTLP | — | **HTTP**（`/v1/traces`） | OpenTelemetry 兼容 span 上报 |
| Runner ↔ Tracer | 实例直接调用 | 实例直接调用 | Tracer 是 Runner 成员 |

> **面试一句话总结**：Agent Lightning 的通信边界很清晰——**只有三类通信永远是 HTTP**：① 跨进程/跨机器的 Store 访问（**我们的三机分离部署下，algo 机和 agent 机都经 `LightningStoreClient` 走 HTTP 连 store 机的 `LightningStoreServer`**）；② agent 的 LLM 调用（Agent → LLMProxy → vLLM）；③ OTLP span 上报；**其余全部是实例直接调用**（Runner→Agent、Algorithm→Adapter、同进程内的 Store 访问）——这也是"Runner Bundle 与 Algorithm Bundle 零直接通信"设计的具体体现。三机分离 = store 单独一台跑 `LightningStoreServer`（`agl store --port 4747`），algo 机（role="algorithm"）和 agent 机（role="runner"，`server_host=store机IP`、`server_port=4747`）都作为 HTTP 客户端连它。

