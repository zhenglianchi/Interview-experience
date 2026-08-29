# Agent Lightning 完全指南

> **Microsoft 的 agent 强化学习训推框架：Algorithm × Runner × TrajStore（LightningStore）三件套 + VERL 集成。**

Agent Lightning（`agentlightning`，简称 agl）是微软研究院开源的"**用强化学习训练任意 agent**"的框架，核心卖点是**对 agent 几乎零代码改动**（任何 agent 框架甚至纯 Python 都能接入）。它的架构围绕三个核心组件组织：**Algorithm（算法，系统的"大脑"）、Runner（执行器，系统的"工人"）、LightningStore（轨迹存储，系统的"数据库+消息队列"）**——三者构成一个持续循环：Algorithm 派发任务 → Runner 执行 agent 并流式回传轨迹（spans）→ Algorithm 从 store 取回轨迹、转成训练样本、更新模型。**我们实际采用三机分离部署：store、algo、agent（runner）分别部署在三台不同的机器，全部通过 store 的 HTTP 地址连接**——本指南全文都以此部署形态为语境（三机分离下，algo 机与 agent 机对 store 的一切访问都是 HTTP；只有 Runner 进程内部的 Agent 调用和算法内部的 Adapter 转换是实例直接调用）。本指南先逐个讲清这三个部分各自的职责与实现，再讲它们如何组成整体架构，最后重点讲与 **verl**（Volcengine 的开源 RL 训练框架）的集成关系。

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

**存储的内容**（五类数据 + 一个队列）：`rollouts`（任务单元，含输入/元数据/生命周期状态 queuing→preparing→running→succeeded/failed/requeuing/cancelled）、`attempts`（每次执行尝试，一个 rollout 可多次 attempt，对应失败重试；状态 preparing→running→succeeded/failed，watchdog 可判 timeout/unresponsive）、`spans`（轨迹事件，按 (rollout_id, attempt_id, sequence_id) 单调序号排序）、`resources`（版本化资源，如 prompt 模板 / LLM 端点）、`workers`（runner 心跳元数据），以及一个 `rollout_queue`（FIFO 任务队列）。**Attempt 与 Rollout 的关系**：rollout 是"外部视图"，attempt 是"内部执行视图"——Runner 实际执行的是 attempt，rollout 状态是"最新 attempt 状态 + 排队/取消控制"的聚合。

**分层实现**（`store/` 目录，官方 classDiagram）：

1. **Collections 层**（`store/collection/`）：底层存储原语——`Collection`（带主键的索引存储：query/get/insert/update/upsert/delete）、`Queue`（FIFO：enqueue/dequeue/peek/size）、`KeyValue`（键值：get/set/inc/chmax/pop）。每个 backend 实现一套。**关键：`atomic()` 上下文管理器提供事务/锁语义**（InMemory 用排序锁防死锁，Mongo 用数据库原子性）。
2. **Store 层**（`store/collection_based.py`）：`CollectionBasedLightningStore` 在 collections 之上实现完整 LightningStore API，包括**业务逻辑：状态流转、watchdog 健康检查、重试策略**。
3. **包装层**：`LightningStoreThreaded`（加 mutex 线程安全，**同进程线程间直接用实例调用**）、`LightningStoreServer` / `LightningStoreClient`（**跨进程/跨机器的 HTTP 化**——Server 是 FastAPI 服务暴露 `/v1/agl/*` store API 和 `/v1/traces` OTLP 端点，Client 实现同一套 LightningStore 接口但内部把每个方法变成 aiohttp/httpx 的 HTTP 请求，带重试和按 event-loop 的 session 池）。**关键点：所有调用方（Algorithm/Runner）只面向 `LightningStore` 接口编程，同进程时是直接方法调用、跨进程时是 HTTP，对调用方透明**。

**开箱即用的实现**：`InMemoryLightningStore`（默认，零依赖，适合开发/CI/测试，锁模式可配 asyncio/thread）、`MongoLightningStore`（生产持久化、多进程安全、支持 `partition_id` 多 trainer 隔离）。通过 `capabilities` 属性（thread_safe / async_safe / zero_copy / otlp_traces）声明各自能力。

**spans 的分布式有序性**（关键设计）：分布式下不同进程产生的 span 时间戳可能乱序，store 强制每个 span 在写入前先 `get_next_span_sequence_id(rollout_id, attempt_id)` 领取单调递增的序号，保证 (rollout_id, attempt_id) 内的轨迹可稳定排序/合并。OTEL span 经 `Span.from_opentelemetry()` 归一化后存储，且 store 暴露标准 OTLP `/v1/traces` 端点，任何 OpenTelemetry 兼容的 SDK/collector 都能直接灌 span。

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

## 附：组件速查表

| 组件 | 代码位置（v0.3.0） | 角色 | 关键接口 / 类 |
|---|---|---|---|
| Algorithm | `algorithm/` | 大脑：派任务、学数据、更新资源 | `Algorithm.run()`；`VERL` / `APO` / `Baseline` |
| Runner | `runner/` | 工人：领任务、跑 agent、写轨迹 | `Runner`；`LitAgentRunner` |
| LightningStore | `store/` | 数据库+队列：解耦两侧 | `enqueue/dequeue_rollout`、`add_otel_span`、`query_spans`、`update_attempt` |
| Tracer | `tracer/` | 轨迹采集：插桩自动捕获 spans | `Tracer`；`AgentOpsTracer` / `OtelTracer` |
| Adapter | `adapter/` | 轨迹→训练样本 | `TraceAdapter`；`TracerTraceToTriplet` |
| LLMProxy | `llm_proxy.py` | agent↔模型的桥 + 插桩 + 换模型 | `LLMProxy`；`ProxyLLM`（main_llm 资源） |
| Trainer | `trainer/` | 总装：接线所有组件 | `Trainer.fit()` |
| ExecutionStrategy | `execution/` | 部署形态：线程 or 进程/机器 | `SharedMemoryExecutionStrategy` / `ClientServerExecutionStrategy` |
| verl 集成 | `verl/` + `algorithm/verl/` | RL 训练后端 | `AgentLightningTrainer(RayPPOTrainer)`、`AgentModeDaemon`、`VERL(Algorithm)` |

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

