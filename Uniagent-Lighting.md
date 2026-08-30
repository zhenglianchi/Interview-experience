# Uni-Agent-Lighting（本项目改造）完全指南

> **基于官方 uni-agent 构建的分离式强化学习平台：uni_agent_ext 三套自研 runner、腾讯云沙箱、多实例本地 WSL 跑 agent、双机全异步训练。**

Uni-Agent-Lighting 是**我们自己的改造仓**（`uniagent-lighting`），基于官方 uni-agent（commit `b139419`）做了两件事：**侵入式修改官方源码**（`patches/` 目录，23 项上游 bug 修复）+ **自研扩展**（`uni_agent_ext/` 三套 runner、腾讯云沙箱、`scripts/` 平台化脚本），并把它构建成**分离式强化学习平台**：agent 跑在任意位置（本地 WSL），云端只提供 Gateway（模型 + token-truth 轨迹）、沙箱（执行）、训练（verl GRPO）。

> **官方 uni-agent 组件本身**（Gateway 具体如何处理轨迹、session/chain 机制、Reward 分配、官方 Agent/Sandbox/Framework/TQ 抽象）见 `Uniagent.md`——本文只讲我们在官方框架之上**自己构建的部分**。对应简历"分离式 Agentic RL 后训练框架"项目。

---

# 一、分离式强化学习平台

## 1. 分离式强化学习整体架构：我们构建了什么

### 1. 现有问题：官方 uni-agent 缺什么，我们补什么

官方 uni-agent 提供了"Gateway 物化轨迹 + runner 驱动 agent + verl 消费"的框架，但用它做真实的**分离式 RL 训练**还缺四块（这四块正是本项目的工作）：

1. **可用的沙箱后端**：官方内置 docker/local/modal 等，但我们的训练跑在 UCloud GPU 机 + 腾讯云环境，需要**腾讯云 Agent Runtime（E2B 兼容）**后端——新增 `tencent_agent_runtime` provider（`uni_agent_ext/sandbox/tencent_agent_runtime.py`）；
2. **三套生产级 runner**：白盒 mini-swe-agent（harness 在训练机/本地 WSL）、黑盒 Claude Code（MCP 工具转发）、外部/平台化 agent（`external_agent_runner`）——都在 `uni_agent_ext/agents/`；
3. **分离式平台化形态**：agent 真正跑在**本地 WSL**（用户侧），云端只提供 Gateway / 沙箱 / 训练——任务下发、隧道、done 标记、云侧 reward 的整套机制（`scripts/platform_local_agent.py` / `platform_local_claude.py` / `run_grpo_platform_test_ucloud.sh`）；
4. **性能与稳定性**：双机全异步（separate_async）+ MooncakeStore 数据平面 + EAGLE-3 投机解码 + 23 项上游 bug 修复（`patches/`）。

### 2. 方法论：分离式架构是怎么分层的

平台化架构分六层（本项目 `docs/architecture.md`）：

```text
┌─────────── Agent 层（任意位置，重点：本地 WSL）───────────┐
│ 白盒 mini-swe-agent / 黑盒 Claude Code + MCP / 外部自定义 agent │
└────────────────┬─────────────────────────┘
                 │ base_url → 云端 Gateway；工具调用转发到沙箱
┌────────────────▼─────────────────────────┐
│ 接入层：Gateway（OpenAI/Anthropic 适配 + 会话路由 + 轨迹物化）│
│ 执行层：腾讯云沙箱（E2B，代码执行/测试/reward）          │
│ 数据层：TransferQueue（MooncakeStore，RDMA）           │
│ 训练层：verl GRPO（LoRA + FSDP2 + CPU offload）        │
│ 服务层：vLLM（合并权重推理）                          │
└───────────────────────────────────────────┘
```

**完整数据流闭环**（一条样本从任务到梯度，重点看 runner 如何接入）：

```text
① verl 训练循环（main_ppo，GRPO+LoRA）
    │ 取一批 prompt（DataProto）
    ▼
② AgentFrameworkRolloutAdapter（uni-agent framework，官方组件，见 Uniagent.md）
    │ 按 runner 注册表（agent_runners.*）调度
    ▼
③ 本项目 runner（uni_agent_ext/agents/）
    ├─ mini_swe_agent_runner（白盒：训练机/沙箱内跑 harness）
    ├─ claude_code_runner（黑盒：沙箱内 claude + MCP 工具转发）
    └─ external_agent_runner（分离式：建沙箱+写 task.json+等 done+云侧 reward）
    │ 模型调用指向 Gateway；工具执行落到腾讯沙箱
    ▼
④ Gateway session（官方组件，见 Uniagent.md）
    │ 物化 token 级轨迹（on-policy）
    ▼
⑤ Framework 转换 + TransferQueue（MooncakeStore）
    │ _trajectory_to_tq_field_and_tag → async_kv_batch_put
    ▼
⑥ verl trainer 消费（kv_batch_get）
    │ GRPO 更新（LoRA + FSDP2 + CPU offload）
    │ checkpoint → LoRA merge → vLLM 服务
    └──▶ 下一轮 agent 用新模型（权重随训练轮次更新）
```

**两种运行形态**（本项目明确区分）：
- **平台化形态（主线，对外）**：agent 在用户侧（**本地 WSL 多实例**，详见第 3 点），经 SSH 隧道接入云端 Gateway；沙箱只执行；训练/推理/轨迹全在云端；
- **内部训练形态（真实训练脚本）**：runner 部署在训练机侧以加速迭代（`run_grpo_humanevalfix_ucloud.sh` 等），但 Gateway 轨迹物化、TQ 数据平面、verl 训练与平台化形态完全一致，on-policy 本质不变（模型调用同样走云端 Gateway）——对外成果统一计为平台化训练产出。

**两条推理链路（重要区分，面试易混）**：**采样推理** = agent 调模型端点（正式期 = 云端单独启动的 OpenAI 兼容 vLLM server 或 Gateway）；**训练 rollout** = verl 内置 vLLM 引擎（双机形态下为 node2 独立引擎），权重随训练轮次更新——两者不要混为一谈。

### 3. 具体数值样例：双机全异步正式训练

以本项目实际跑通的 **双机全异步正式训练**（HumanEvalFix，161 条全样本）为样例：

```text
环境：node1 = 1×4090 48G（trainer）；node2 = 2×4090 24G（rollout 引擎）
配置：batch 32 / 并发 64 / LoRA rank=32 / GRPO / EAGLE-3 投机 / MooncakeStore

第 1 步（采样）：node2 上 4 个 agent runner 并发执行 64 个任务，
        agent 通过 Gateway（云端）调 vLLM（Qwen3-8B + EAGLE-3 投机，
        吞吐 282 tok/s）；每次生成后工具调用转发到腾讯沙箱执行。
第 2 步（物化）：Gateway session 把每个任务的 3~6 轮交互物化成
        1 条 Trajectory（平均 prompt 1273 / response 967 tokens），
        reward 由沙箱 pytest 真实判定（通过=1.0）。
第 3 步（传输）：framework 转 TQ 字段 → MooncakeStore（RDMA），
        32 条一批写入；node1 与 node2 采样训练完全重叠
        （node2 采第 N+1 批时 node1 训第 N 批）。
第 4 步（训练）：node1 verl GRPO 一步（48.1s，含 advantage/ref logprob/KL），
        LoRA 权重更新；ratio_mean 稳定 1.0 无漂移。
第 5 步（评估）：25 步后 merge LoRA → vLLM serve → 161 条全量评估
        （n=1 / temp 0.8 / 并发 16-24）→ 通过率 83.23%（134/161，
        较基座 76.4% +6.83pp）。
```

**对比单机同步基线**：单步 79.4s → 双机全异步 48.1s（-39%）；EAGLE-3 投机把生成吞吐从 199 提到 282 tok/s（+41.7%）且训练质量无损（83.23% vs baseline 83.2%）。

> **面试一句话总结**：uniagent-lighting 在官方 uni-agent 之上构建了分离式 RL 平台——六层架构（Agent 任意位置 → Gateway 轨迹物化 → 腾讯沙箱执行 → TQ(Mooncake) → verl GRPO → vLLM 服务），补上了腾讯沙箱后端、三套生产 runner、本地 WSL 平台化形态、双机全异步 + 投机解码四块；双机全异步（每步 -39%）+ Mooncake + EAGLE-3 的组合实现 HumanEvalFix 83.23%（+6.83pp）。

---

## 2. uni_agent_ext：我们自研的三套 runner 与腾讯沙箱

### 1. 现有问题：为什么需要自研 runner 而不是直接用官方的

官方 runner（`uni_agent/agents/` 内置）满足基础演示，但真实训练有三类场景官方覆盖不全：**① 白盒 humaneval_fix 自定义任务**——官方 mini-swe-agent 走 swebench-single 数据集硬编码，我们新增 humaneval_fix 口径（任务文件注入 + Python API 直连绕开数据集）；**② 黑盒 Claude Code 的工具转发**——官方在沙箱内直连 claude，我们既要支持沙箱内、也要支持本地 WSL 经 MCP 转发；**③ 分离式平台化**——agent 在用户侧（本地 WSL），训练侧需要一个"建沙箱 + 暴露任务 + 等完成 + 云侧 reward"的 runner，官方没有。因此我们在 `uni_agent_ext/agents/` 实现了三套 runner。

### 2. 方法论：三套 runner 与腾讯沙箱是怎么实现的

**`mini_swe_agent_runner.py`（白盒，训练机侧）**：核心是 `extract_task`（从 raw_prompt/tools_kwargs 解析 issue、FAIL_TO_PASS、instance_id）+ `create_task_sandbox`（建腾讯沙箱）+ `build_mini_swe_config`（生成 mini-swe config：`api_base=Gateway URL`、`attach_instance_id=沙箱实例`、`max_turns=60`）+ `run_mini_swe_agent_api`（humaneval_fix 用 Python API 直连）+ `evaluate_reward`（沙箱内写隐藏测试、批量跑 FAIL_TO_PASS + 抽样 PASS_TO_PASS，score=通过数/总数）。**humaneval_fix 关键设计**：任务文件（solution.py）由 `tools_kwargs.env.files` 注入沙箱 `/testbed`，隐藏测试只在 reward 阶段写入（无测试泄露）。

**`claude_code_runner.py`（黑盒，沙箱内/本地）**：核心是 `build_claude_task`（issue + FAIL_TO_PASS 测试列表，无 test_patch 泄露）+ `build_claude_command`（沙箱内 `claude -p` 命令：`ANTHROPIC_BASE_URL=Gateway`、`CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` 与 Gateway 截断对齐、`IS_SANDBOX=1`、`--permission-mode bypassPermissions`）。黑盒的工具调用经 MCP 转发到沙箱执行（本地 WSL 场景见第 3 点）。

**`external_agent_runner.py`（分离式/平台化，训练机侧）**：这是"本地 WSL 跑 agent"的训练侧配套（详见第 3 点），职责：建沙箱 → 注入任务文件 → 写 `<session_id>.task.json` → 轮询 `<session_id>.done` 标记 → 云侧 `evaluate_reward` → `POST reward_info` → 清理。

**腾讯云沙箱（`uni_agent_ext/sandbox/tencent_agent_runtime.py`）**：实现官方 `Sandbox` 接口的 `TencentAgentRuntimeSandbox`——通过 `E2B_DOMAIN=ap-guangzhou.tencentags.com` + `E2B_API_KEY` 接入腾讯云 Agent Runtime（E2B 兼容端点），SDK 用 `e2b-code-interpreter`（2.9.0）；支持 SWE-bench 托管镜像（`StartSandboxInstance` + `CustomConfiguration.Image="swebench/sweb.eval.x86_64.<org>_<repo>-<pr>:latest"`，实例内 /testbed 是题目仓库、swerex server 8000 端口、envd 49983），白盒通过 `tencent_e2b` 环境类直接执行，黑盒经 MCP 转发层落地。

**核心代码**（`mini_swe_agent_runner.py` 的主流程——"建沙箱 → 注入任务 → 跑 agent → 评估 reward → 上报"的代码落点）：

```python
async def mini_swe_agent_runner(*, raw_prompt, session: SessionHandle,
                                sample_index, tools_kwargs=None, max_turns=60, ...):
    task = extract_task(raw_prompt, tools_kwargs)      # ① 解析任务
    image = (tools_kwargs.get("env") or {}).get("image")
    gateway_url = session.base_url                      # ② Gateway 会话地址

    sandbox = await asyncio.to_thread(create_task_sandbox, image=image, gateway_url=gateway_url)
    try:
        await sandbox.start()
        instance_id = sandbox.instance_id
        # ③ 任务文件注入（humaneval_fix）：/testbed git 仓库 + solution.py，
        #    只注入任务文件，隐藏测试由 evaluate_reward 完成后写入（无测试泄露）
        env_files = (tools_kwargs.get("env") or {}).get("files") or {}
        for rel_path, content in env_files.items():
            await sandbox.write_file(f"/testbed/{rel_path}", content)
        await sandbox.exec_shell("cd /testbed && git add -A", timeout=60)
        ...
        rc, log_tail = await run_mini_swe_agent_api(task=task, config_path=config_path, ...)  # ④ 跑 agent

        score, details = await evaluate_reward(sandbox, task)   # ⑤ 沙箱内判 reward
        reward_info = {"reward": score, "reward_score": score,   # ⑥ 上报（framework 消费 "reward" 键）
                       "agent_exit_code": rc, **details}
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(session.reward_info_url, json={"reward_info": reward_info})
    finally:
        await sandbox.stop()                              # ⑦ 清理
```

**这段代码关键在哪**：整个 runner 就是"官方五步模板"的实现——`extract_task` 解析 issue/FAIL_TO_PASS、`create_task_sandbox` 建腾讯沙箱、**任务文件注入只写 solution.py 不写测试**（无测试泄露的核心）、`run_mini_swe_agent_api` 用 Python API 直连（绕开 swebench-single 数据集硬编码）、`evaluate_reward` 沙箱内跑 pytest、`POST session.reward_info_url` 上报（**注意 `reward` 键是 framework `_score_from_reward_info` 消费的，缺了会导致 rm_scores=0**）。

**reward 评估的核心**（`evaluate_reward`，真实 SWE-bench 口径）：

```python
async def evaluate_reward(sandbox, task, *, timeout=300, include_p2p=False, ...):
    env = SandboxEnvForReward(sandbox)
    fail_to_pass = task["fail_to_pass"]          # FAIL_TO_PASS 测试列表
    pass_to_pass = task["pass_to_pass"] or []
    # 无测试泄露：隐藏测试只在 reward 阶段写入
    for rel_path, content in (task.get("hidden_files") or {}).items():
        await env.write_file(f"/testbed/{rel_path}", content)
    passed_map = await _run_tests_batch(env, fail_to_pass, timeout=timeout)  # pytest -v 批量跑
    f2p_passed = sum(1 for t in fail_to_pass if passed_map.get(t))
    score = f2p_passed / len(fail_to_pass)       # 通过数/总数，0.0~1.0 连续
    return score, {"resolved": f2p_passed == len(fail_to_pass), ...}
```

**这段代码关键在哪**：`evaluate_reward` 是 reward 的真实来源——**隐藏测试（hidden_files）只在评估阶段写入沙箱**（训练期 agent 看不到测试，无测试泄露），`score = 通过数/总数`（连续分数给 GRPO 提供梯度），`resolved` 表示 FAIL_TO_PASS 全过。

### 3. 具体数值样例

以白盒 humaneval_fix runner 处理一个任务为例，逐步演算：

```text
第 1 步：extract_task：从 raw_prompt 解析 instance_id / issue；
        FAIL_TO_PASS = ['test_solution.py::test_all']（容错解析 list/JSON/换行）。
第 2 步：create_task_sandbox(image=python:3.12) → 腾讯沙箱实例
        （SandboxConfig(provider=tencent_agent_runtime, ...) → build_sandbox）。
第 3 步：任务文件注入：/testbed 建 git 仓库 + 写 solution.py（env_files），
        git add -A（让 agent 提交时 git diff 能拿到 patch）。
第 4 步：build_mini_swe_config：api_base=Gateway、attach_instance_id=沙箱、
        max_turns=60 → run_mini_swe_agent_api（Python API：get_sb_environment
        + get_agent 跑 humaneval_fix）。
        期间每次模型调用经 Gateway 物化轨迹；工具执行落沙箱。
第 5 步：evaluate_reward：写隐藏测试 test_solution.py → pytest -v 批量跑
        → score = 通过数/总数（0.0~1.0 连续）；resolved = FAIL_TO_PASS 全过。
第 6 步：POST reward_info = {"reward": score, "reward_score": score,
        "agent_exit_code": rc, "resolved": ..., "per_test": [...]}
        → framework._score_from_reward_info 消费 "reward" 键 → rm_scores。
```

> **面试一句话总结**：uni_agent_ext 的三套 runner 覆盖三类真实训练场景——白盒 humaneval_fix（任务文件注入 + Python API 直连 + 无测试泄露）、黑盒 Claude Code（MCP 工具转发 + 沙箱内/本地双形态）、外部/平台化（建沙箱 + task.json + done 标记 + 云侧 reward）；腾讯沙箱按官方 Sandbox 接口实现 E2B 兼容后端，支持 SWE-bench 托管镜像。

---

## 3. 分离式核心：多实例本地 WSL 跑 agent（平台化形态）

> 这是本项目**区别于"把 agent 塞进训练机"的核心亮点**（简历中"分离式 Agentic RL 后训练框架"）：agent 真正跑在**用户侧（本地 WSL）**，云端只提供 Gateway（模型+轨迹）、沙箱（执行）、训练。本节详细讲任务如何下发、agent 如何完成、与沙箱如何交互、黑盒白盒差异。

### 1. 现有问题：为什么必须让 agent 跑在本地 WSL

平台化形态的终态（用户 2026-08-12 明确拍板，AGENTS.md 记录）：**agent 跑在用户侧/本地（或任意位置），模型调用指向云端 Gateway，沙箱只负责执行**。为什么不能"图省事"把 agent 直接放训练机？有三个原因：

1. **on-policy 的语义**：on-policy 只要求 token-truth 轨迹由 Gateway 云侧记录（logprob 云侧生成、外部不可伪造），**并不要求 agent 在云端**——agent 在哪都行，这本身就是分离式的意义；
2. **真实部署形态**：真实用户场景中 agent 就是跑在用户自己机器上的（IDE 插件、CLI 工具），如果训练框架只能驱动"训练机上的 agent"，就失去了"训练任意位置 agent"的能力；
3. **解耦与扩缩**：agent 位置与训练解耦后，用户侧可以开任意多个 agent 实例（多实例并行），云端训练与沙箱执行各自独立扩缩，互不阻塞。

难点也随之而来：**本地 WSL 如何与云端训练机通信？** 训练机公网通常只开 22 端口（SSH），Gateway 的自定义端口（8001）不直接暴露——所以必须用 **SSH 隧道（paramiko direct-tcpip）** 打通"本地随机端口 → 训练机内网 Gateway"。这是本节的核心工程点。

### 2. 方法论：任务如何下发给本地 WSL、如何完成、如何与沙箱交互

整个流程是**训练侧（云端）与本地侧（WSL）通过文件系统 + SSH 隧道协作**的异步闭环。先讲训练侧 runner（`external_agent_runner.py`），再讲本地侧脚本（`platform_local_agent.py` / `platform_local_claude.py`），最后讲黑盒白盒差异。

**训练侧：`external_agent_runner`（云端，framework 驱动），逐步操作：**

**第 1 步（建沙箱 + 注入任务）**：`create_task_sandbox(image, gateway_url)` 建腾讯云 E2B 沙箱；humaneval_fix 任务把 `tools_kwargs.env.files`（solution.py 等任务文件，**不含测试**）写入沙箱 `/testbed`，`git init` 让文件被跟踪（agent 提交时 `git diff` 能拿到 patch）。

**第 2 步（写任务文件）**：`_write_task_file()` 把任务信息落盘为 `<PLATFORM_TEST_DIR>/<session_id>.task.json`，内容含：`session_id`、`base_url`（Gateway session 地址，形如 `http://10.60.56.10:8001/sessions/<id>/v1`）、`instance_id`（沙箱实例 id）、`image`、`raw_prompt`、`tools_kwargs`、`task`（issue / FAIL_TO_PASS 等）、`done_marker`（done 标记路径）。**这个文件是"任务下发给本地 WSL"的载体**——本地脚本经 SSH/SFTP 轮询读取它。

**第 3 步（等 done 标记）**：`_wait_for_done(session_id, run_timeout)` 每 5 秒轮询 `<session_id>.done` 文件是否存在——该标记由**本地 agent 完成后经 SSH 创建**（`touch`）。超时（默认 7200s）则放弃。

**第 4 步（云侧 reward）**：检测到 done → `evaluate_reward_msa(sandbox, task)` 在沙箱内**写入隐藏测试**（`hidden_files`，如 `test_solution.py::test_all`，无测试泄露）→ 批量跑 FAIL_TO_PASS（+ 可选抽样 PASS_TO_PASS）→ `score = 通过数/总数`（0.0~1.0 连续）→ `POST session.reward_info_url` 上报 `{"reward": score, "reward_score": score, "resolved": ..., "per_test": [...]}`。

**第 5 步（清理）**：删除 task.json 和 done 标记、停止沙箱。

**核心代码**（`external_agent_runner.py`——"任务下发"与"等完成"的代码落点）：

```python
# ① 任务下发：把任务信息写进 task.json（本地 WSL 经 SSH/SFTP 轮询读取）
def _write_task_file(*, session, sandbox, raw_prompt, tools_kwargs, task) -> Path:
    payload = {
        "session_id": session.session_id,
        "base_url": session.base_url,               # Gateway session 地址
        "instance_id": getattr(sandbox, "instance_id", ""),  # 训练侧建的沙箱
        "image": (tools_kwargs.get("env") or {}).get("image", ""),
        "raw_prompt": raw_prompt,
        "tools_kwargs": tools_kwargs,
        "task": task,
        "done_marker": str(PLATFORM_TEST_DIR / f"{session.session_id}.done"),
        "created_at": time.time(),
    }
    task_file = PLATFORM_TEST_DIR / f"{session.session_id}.task.json"
    task_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return task_file

# ② 等完成：轮询 done 标记（本地 agent 完成后经 SSH touch 创建）
async def _wait_for_done(session_id: str, run_timeout: float) -> bool:
    done_file = PLATFORM_TEST_DIR / f"{session_id}.done"
    deadline = time.time() + run_timeout
    while time.time() < deadline:
        if done_file.exists():
            return True
        await asyncio.sleep(POLL_INTERVAL)          # 默认 5 秒
    return False    # 超时（默认 7200s）
```

**这段代码关键在哪**：`_write_task_file` 生成的 task.json 是**"任务下发给本地 WSL"的载体**——包含 base_url（Gateway 地址）、instance_id（沙箱）、done_marker（完成标记路径），本地脚本 `wait_for_task` 轮询它；`_wait_for_done` 每 5 秒轮询 done 标记，超时放弃——整个"训练侧 ↔ 本地侧"的协作就是"写文件 + 轮询文件"，完全解耦。

**本地侧：`platform_local_agent.py`（WSL，白盒 mini-swe-agent），逐步操作：**

**第 1 步（连训练机 + 原子认领任务）**：paramiko SSH 连训练机（凭据在 `work/ucloud.env`），SFTP 轮询 `<PLATFORM_TEST_DIR>` 目录并**原子认领**一个 `*.task.json`（`claim_task`：把 `task.json` rename 成 `task.json.claimed.<worker-id>`，**谁 rename 成功谁领到**，多 worker 并发互斥，rename 失败（已被别人领）就试下一个）。

**第 2 步（建隧道）**：`TunnelForwarder` 用 paramiko **direct-tcpip** 把本地随机端口转发到训练机内网 `10.60.56.10:8001`（Gateway）——本地 `127.0.0.1:<port>` → 训练机 Gateway。之后 base_url 改写为 `http://127.0.0.1:<local_port><session_path>`。

**核心代码**（`platform_local_agent.py` 的 `TunnelForwarder`——SSH 隧道打通公网只开 22 端口的训练机）：

```python
class TunnelForwarder:
    """paramiko direct-tcpip 端口转发：本地 127.0.0.1:<port> → 远端 host:port。"""
    def __init__(self, transport: paramiko.Transport, remote_host: str, remote_port: int):
        self._transport = transport
        self._remote = (remote_host, remote_port)
        self._local = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._local.bind(("127.0.0.1", 0))      # 本地随机端口
        self._local.listen(16)
        self.port = self._local.getsockname()[1]

    def _forward(self, conn: socket.socket):
        channel = self._transport.open_channel("direct-tcpip", self._remote, conn.getpeername())
        # 双向泵数据：本地连接 ⇄ SSH 隧道 ⇄ 训练机内网 Gateway
        ...
```

**这段代码关键在哪**：`TunnelForwarder` 用 paramiko 的 `open_channel("direct-tcpip", ...)` 实现**端口转发**——训练机公网只开 22 端口（SSH），Gateway 的 8001 端口不暴露，隧道把"本地随机端口 → 训练机内网 Gateway"打通，agent 的模型调用经 `http://127.0.0.1:<port>` 打到云端 Gateway（轨迹云侧物化）。这是"分离式"的网络基础。

**第 3 步（构建配置 + 跑 agent）**：`build_config()` 用 `build_mini_swe_config` 生成 mini-swe config：`api_base=隧道后的 Gateway URL`、`attach_instance_id=沙箱实例 id`（**agent 直接 attach 训练侧已建的沙箱**）、`model=Qwen3-8B`、`max_turns=60`；然后 `run_local_agent()` 用 mini-swe-agent **Python API 模式**（`get_sb_environment` + `get_agent`）跑 humaneval_fix 任务——agent 的每次模型调用经隧道打到云端 Gateway（轨迹云侧物化），工具执行落在 attach 的沙箱里。

**第 4 步（创建 done 标记）**：agent 完成后经 SSH `touch <done_marker>`，通知训练侧可以评估 reward。

**本地侧：`platform_local_claude.py`（WSL，黑盒 Claude Code）**——与白盒的差异（黑盒重点）：

**第 1 步**：同样连训练机等 task.json、建隧道（`base_url` 去 `/v1` 给 Anthropic 客户端）。
**第 2 步（MCP 工具转发）**：写 MCP config：`mcpServers.sandbox` 指向 `sandbox_mcp_server.py`（手写 stdio JSON-RPC MCP server），env 带 `E2B_SANDBOX_ID=训练侧建的沙箱`、`E2B_API_KEY`——**Claude Code 的 Bash/Read/Write/Edit/Glob 等内置工具被 `--disallowedTools` 禁用，改走 MCP 转发到云端沙箱执行**。
**第 3 步（跑 claude）**：`claude --bare -p <task_text>` + `--model Qwen3-8B` + `--max-turns 60` + `--permission-mode bypassPermissions` + `--mcp-config`，`ANTHROPIC_BASE_URL` 指向隧道后的 Gateway session——Claude Code 在**本地 WSL** 编排，模型调用走云端，工具执行走 MCP→沙箱。
**第 4 步**：完成后 touch done 标记。

**多 worker 并发认领（2026-08 升级，替换早期"扫描取第一个"弱认领）**：早期 P0 版 `wait_for_task` 只是"倒序排列表取第一个 task.json"，**不做原子互斥**——多个本地 worker 同时轮询会拿到同一个任务（重复执行）。升级为**原子 rename 认领**（`scripts/platform/platform_local_agent.py`，与仓库 `docs/任务剖析-外部形态.md` §7.5 一致）：

```python
# platform_local_agent.py —— wait_for_task：多实例并发安全的原子认领
def wait_for_task(sftp, remote_dir, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _recover_stale_claims(sftp, remote_dir, timeout)   # 先回收过期认领（7.5.4）
        files = sorted(sftp.listdir(remote_dir), reverse=True)
        tasks = [f for f in files if f.endswith(".task.json")]
        for name in tasks:
            claimed = f"{name}.claimed"
            try:
                sftp.rename(f"{remote_dir}/{name}", f"{remote_dir}/{claimed}")  # ★ 原子认领
            except OSError:
                continue          # 已被其他实例抢走 → 试下一个
            with sftp.open(f"{remote_dir}/{claimed}") as fh:
                payload = json.loads(fh.read().decode())
            return claimed, payload   # 认领成功：文件状态 <id>.task.json → <id>.task.json.claimed
        time.sleep(5)
    raise TimeoutError(...)
```

- **互斥保证**：SFTP `rename` 在同一文件系统内是原子的——N 个实例同时 poll，**只有 rename 成功的那一个拿到该任务**，其余 `OSError → continue` 去找下一个文件，互不冲突——**一个任务永远只被一个实例认领**；
- **过期回收（`_recover_stale_claims`）**：认领实例正常处理完会删 claimed；若**崩溃残留**，其他实例发现 claimed 文件 **mtime 超过 `--timeout`** 视为失效，把它 `rename` 回 `.task.json` 重新认领——与内部形态"session 失败 → abort → framework 统计重试"的失败重试语义对齐（回收在**本地侧**，不是训练侧）；
- **`--max-tasks` 循环处理**：每个实例一个进程，`--max-tasks N` 控制单实例连续处理 N 个任务（`while processed < args.max_tasks: wait_for_task → process_one_task(隧道→agent→touch done) → sftp.remove(claimed)`），默认 1——多实例 = 多个进程各自 `--max-tasks`，无需 worker-id（rename 原子性本身互斥）；
- **done 幂等 + 清理**：重复执行同一任务时 reward 只记一次（训练侧收到 done 就评估，先到先得）；本地实例处理完 `sftp.remove(claimed)`，训练侧 finally 同时清理 `task_file` 与 `task_file.claimed` 双路径 + `<session>.done`；
- 每个 task.json 的 session_id 唯一，本地脚本用 `session_id[-8:]` 区分配置/轨迹文件，互不覆盖；训练侧 `max_concurrent_sessions` 控制并发上限（并发 N 个 session → 服务器 platform_test/ 同时出现 N 个 task.json，本地 N 个实例竞争认领）。

**训练侧任务生命周期：done 触发什么 / 何时写 TQ / 何时删文件**（回答"训练侧是不是监测所有 task.json 变 done？有一个 done 就写 TQ 吗？"）：

- **不是集中式监测**：训练侧没有"看所有 task.json 的监测器"。framework（`_run_prompt_sessions_to_tq`）用 `asyncio.gather` 并发驱动**每个 session 一个独立 runner 实例**（`external_agent_runner`），每个 runner **只轮询自己的 `<session_id>.done`**（`_wait_for_done`，5s 间隔）——per-session 文件天然隔离，多 worker 认领后各自对应的 session 各自等 done；
- **done 出现 ≠ 写 TQ**：done 触发的第一步是**云侧 reward 评估**（沙箱内写隐藏测试 → pytest → `POST reward_info`）；reward 完成后 runner 返回，framework 才 `finalize_session`（Gateway 云侧物化轨迹）→ `_score_from_reward_info`（用 runner 上报的 reward 标注）→ **`_write_session_trajectories_to_tq` 把轨迹写 TQ**（`kv_batch_put`，NestedTensor，每轨迹一个 key）；
- **"所有任务完成"的聚合点在 TQ tag，不在文件系统**：同一 prompt 的 `rollout.n` 个 session 都写完轨迹后，framework `kv_put(uid, tag={status: "finished"})`；训练侧 `ReplayBuffer.sample(global_steps)` 轮询 TQ，等该 step 的**全部 uid 都 finished** 才返回（失败 session 写 `status: "failure"` 同样算完成）——所以"什么时候开始训练"由 TQ 的 finished tag 决定；
- **删除时机（每 session 自己删）**：runner 的 `finally` 在 reward 上报后、返回前执行——**双路径清理** `task_file` 与 `task_file.claimed`（本地实例认领会把 task.json rename 成 `.claimed`，两种名字都删）+ `<session>.done` 删除 + 停沙箱；本地实例侧处理完也 `sftp.remove(claimed)`——**不存在集中清理，也没有遗留文件的累积**（崩溃残留的 claimed 由本地侧 `_recover_stale_claims` 按 mtime 回收）。

```text
单 session 完整时序（训练侧 framework ↔ 本地 worker 协作）：
framework._run_session:
  ① gateway_manager.create_session(session_id)          # Gateway 会话
  ② runner(external_agent_runner)：
       建腾讯沙箱 → 写 <session>.task.json
       → [本地实例 wait_for_task 原子认领(.claimed) → 跑 agent → SSH touch <session>.done
          → 本地 sftp.remove(claimed)]
       → _wait_for_done 轮询到自己的 done ✓
       → 云侧 reward（沙箱 pytest）→ POST reward_info
       → finally：删 task.json + task.json.claimed + <session>.done + 停沙箱
  ③ finalize_session → Gateway 云侧物化轨迹
  ④ _score_from_reward_info（用 runner 的 reward 标注）
  ⑤ 返回轨迹 → _run_prompt_sessions_to_tq 写 TQ（kv_batch_put）+ uid finished tag
  ⑥ ReplayBuffer 等全部 uid finished → sample → GRPO 训练
```

> **面试一句话**：训练侧是"每 session 一个 runner、各等各的 done"（不是集中监测）；done 只触发 reward 评估，轨迹在 reward 完成后由 framework 物化并写 TQ，聚合点靠 TQ 的 uid finished tag（ReplayBuffer 等全部完成才训练）；task 文件由每个 runner 的 finally 自删（含 claimed 后缀），超时会先回收再清理——文件系统只是"任务下发与完成"的信令，真正的数据与同步都在 TQ。

**黑盒 vs 白盒的完整对比**（面试高频）：

| 维度 | 白盒 mini-swe-agent | 黑盒 Claude Code |
|---|---|---|
| agent 位置 | 本地 WSL（Python API 驱动） | 本地 WSL（`claude --bare` CLI） |
| 模型接入 | `api_base` 指向隧道后的 Gateway（OpenAI 协议） | `ANTHROPIC_BASE_URL` 指向 Gateway（Anthropic 协议） |
| 工具执行 | attach 沙箱实例（`tencent_e2b` 环境类） | **MCP 转发层**（sandbox_mcp_server，stdio JSON-RPC）→ 云端沙箱 |
| 轨迹来源 | Gateway 云侧物化 | Gateway 云侧物化（Anthropic 适配） |
| 关键难点 | 配置生成（api_base/attach 沙箱） | 黑盒无远程执行抽象 → 手写 MCP 工具转发 + 禁用内置工具 |
| reward | 沙箱 pytest 判定 | 沙箱 pytest 判定 |

**MCP 工具转发完整链路（黑盒，真实指令样例）**——"为什么黑盒保留了 Bash/Edit 还需要 MCP"的答案：

- **前提**：官方 uni-agent 的 claude 在沙箱内、内置 Bash/Edit 就地 fork 执行，不需要 MCP；平台化把 claude 挪到本地后，**内置工具必须禁用**（否则在本地执行、改的是本地文件）——`platform_local_claude.py` 的 `--disallowedTools Bash Edit Read Write Glob Grep Agent Task WebFetch WebSearch`，同时 `--mcp-config` 挂上我们的 MCP server；
- **MCP server**（`scripts/sandbox_mcp_server.py`）：手写 **stdio JSON-RPC 2.0**（每行一个消息，绕开 mcp 2.0 拆包 FastMCP 的坑），通过 `tools/list` 声明 Bash/Read/Write/Edit/Glob 五个同名工具，每个 handler 内部 `Sandbox.connect(E2B_SANDBOX_ID)` 连云端沙箱执行；
- **真实指令样例**：claude 决定把 `/testbed/solution.py` 的 `return sorted(nums)` 改成 `nums.sort(); return nums`：

```json
// claude → MCP server（stdio 一行 JSON-RPC）
{"jsonrpc":"2.0","id":5,"method":"tools/call",
 "params":{"name":"Edit",
           "arguments":{"file_path":"/testbed/solution.py",
                        "old_string":"return sorted(nums)",
                        "new_string":"nums.sort(); return nums"}}}
```

```python
# sandbox_mcp_server.py → 转发到云端沙箱（真实实现）
async def _tool_edit(args: dict) -> str:
    sbx = await asyncio.to_thread(_get_sandbox)     # Sandbox.connect(sandbox_id=E2B_SANDBOX_ID)
    data = await asyncio.to_thread(sbx.files.read, args["file_path"])    # 沙箱读
    if count == 0: raise RuntimeError("old_string not found ...")
    if count > 1: raise RuntimeError("old_string appears N times ...")   # 唯一匹配校验
    await asyncio.to_thread(sbx.files.write, args["file_path"],
        text.replace(old, new, 1).encode())          # ← 真正改文件在云端沙箱
    return "edited /testbed/solution.py (1 replacement)"
```

- **完整链路**：claude 决策 → `tools/call(Edit, {...})`（JSON-RPC over stdio）→ `_HANDLERS["Edit"]` → `_tool_edit()` → `Sandbox.connect` → `sbx.files.read/write`（E2B SDK，云端沙箱落地）→ JSON-RPC 响应回 claude → 继续决策（下一步 `Bash` 跑 pytest 同样走 `_tool_bash` → `sbx.commands.run`）；
- **一句话**：官方管"agent 进沙箱、工具就地执行"（不需要 MCP），我们管"agent 出沙箱、工具远程转发"（MCP 就是那把"工具执行位置从本地重定向到沙箱"的钥匙）——这也是简历"针对黑盒无远程执行抽象、手写 stdio JSON-RPC MCP 工具转发层"的由来。

**为什么白盒能直接 attach、黑盒却要 disallowedTools + MCP（机制对比，面试必问）**：

- **本质：白盒的工具是"我们的"（代码可控，白名单语义），黑盒的工具是"它的"（闭源闭环，只能黑名单做减法）**；
- **白盒 attach** = 在训练机本地跑的 mini-swe-agent harness 通过 E2B `Sandbox.connect(sandbox_id=attach_instance_id)` 连到"runner 已建好的沙箱实例"（`tencent_e2b.py` 的 `_create()`），agent 循环的 `execute_actions` 里每个 action 走 `env.execute(action)` → `sandbox.commands.run` 在沙箱远端执行——工具集是 mini-swe-agent 的环境接口方法（`Environment.execute`），**执行位置由我们代码直接指定**（换环境类即可），所以"想 attach 就 attach"，不需要禁用/转发任何东西；
- **黑盒 disallowedTools**：claude 二进制自带完整工具闭环（own loop + own tools + own subagents），我们无法改它内部，只能从外部黑名单。`--disallowedTools` 具体禁掉两类并各有原因：
  - **`Agent` / `Task`（子代理 / 任务分解）**：会并行启动子进程/子 agent——① **官方 uni-agent 没有子轨迹的数据结构**（`Trajectory` 扁平、Gateway session 内是线性 `ChainState`、verl agent_loop 无 sub-agent，见 `Uniagent.md`），子代理调用即使走主 session 的 Gateway 也无法作为独立轨迹表达，只能混入主 session 事件，**破坏 token-truth 轨迹的 tool-call/tool-result 对应结构 → 训练数据污染**；② 子进程不可控（资源爆炸、文件乱写）；③ RL 要的是"单 agent 决策轨迹"；
  - **`WebFetch` / `WebSearch`（网络）**：SWE-bench 任务不需要联网；禁用保证**可复现 + 无信息泄露**（不能搜题/查 patch），轨迹纯净；
  - 保留 `Bash/Edit/Read/Write/Glob/Grep` 的"同名 MCP 版本"（转发到沙箱）——真正在沙箱里改 `/testbed`、跑 pytest 的工具；
- **对照结论**：attach 是"我们的工具连到沙箱"（白名单，代码说了算），disallowedTools + MCP 是"它的工具我们禁掉危险的、用同名 MCP 工具接管"（黑名单 + 协议拦截）——同一个"agent 在外、工具在沙箱"的目标，白盒靠换 Environment 实现、黑盒靠 MCP 转发，这是黑盒平台化额外多一层工程的根本原因。

### 3. 具体数值样例（真实 reward=1 轨迹逐字段拆解）

以本项目真实跑通的一条**黑盒 reward=1 轨迹**（`blackbox_full_20260812/.../session-sample-0-rollout-0-3d3cf465...`，humanevalfix-Python-115 水桶问题）为样例，完整拆解"一次成功的分离式 RL 样本"：

**轨迹元数据（framework.log / trajectory.json 真实值）**：

```text
session: sample-0-rollout-0-3d3cf465...  runner=claude_code  global_steps=1
turns = 7            # Claude Code 决策循环 7 轮（含工具调用）
prompt_tokens = 1663  # 题目 + 上下文
response_tokens = 983 # 7 轮累积的模型输出（含续轮插入上下文，mask 区分）
model_tokens = 82     # 纯模型生成的 token（其余是插入的上下文/工具结果）
logprobs = yes        # on-policy：Gateway 云侧 vLLM 生成
reward_score = 1.0    # resolved=True
per_test = ['test_solution.py::test_all: PASS']   # 沙箱 pytest 真实通过
elapsed ≈ 25.4s       # agent 从启动到完成
```

**时间线（task.log 真实值）**：

```text
01:05:31  session start（framework 创建 Gateway session）
01:05:35  腾讯沙箱创建（code-interpreter-v1 模板，id=v6p2sspdr...）
01:05:38  沙箱内装好 claude-code 2.1.153
01:05:41  Gateway 隧道打通：127.0.0.1:8000 → 117.50.199.93:8001
          （沙箱内 ssh -L，走训练机 22 端口转发 Gateway 端口）
01:05:41~01:06:06  Claude Code 在沙箱内执行：7 轮决策（读题目 → 写 solution.py
          → 改 math.floor 为 math.ceil → 提交）
01:06:06  claude-code finished rc=0（stdout："The solution now uses math.ceil
          instead of math.floor to correctly calculate the number of bucket
          lowerings required..."）
01:06:06  extract_task：FAIL_TO_PASS = ['test_solution.py::test_all']
01:06:07  evaluate_reward：写入隐藏测试 → pytest → PASS → score=1.0 resolved=True
01:06:07  reward 上报 reward_info → 轨迹 finalize → 物化 1 条 Trajectory
01:06:07  沙箱销毁
```

**这条轨迹进入训练后的样子**（Framework 转换，见 `Uniagent.md`）：`rm_scores = zeros(983,); rm_scores[-1] = 1.0`（reward 打在响应最后一位）→ TQ → verl GRPO：这条样本的序列分数 = 1.0，参与组内 advantage 计算——**一个成功样本的完整闭环：本地黑盒 agent 决策 → 云端 Gateway 物化轨迹 → 沙箱执行判 reward=1 → 进入 GRPO 训练**。

**对比同一 step 的失败样本**（同目录 step_1 其他 rollout）：reward=0 的轨迹 `per_test = ['test_solution.py::test_all: FAIL']`、`resolved=False`，同样物化进训练——GRPO 正是靠"组内正负样本的相对 advantage"来学习，reward=1 与 reward=0 的样本同批参与，形成策略梯度。

> **面试一句话总结**：分离式 RL 的核心是"agent 在本地 WSL 跑、云端只提供 Gateway/沙箱/训练"：训练侧 `external_agent_runner` 建沙箱 + 写 task.json（任务下发载体）+ 等 done 标记 + 云侧 reward；本地侧 paramiko direct-tcpip 隧道打通 Gateway（公网只开 22 端口）+ 白盒 Python API 直连 / 黑盒 MCP 工具转发执行；多实例并行（每实例唯一 session_id）；真实 reward=1 轨迹（7 turns / 983 response tokens / 25.4s）展示完整闭环——on-policy 由 Gateway 云侧 logprob 保证，与 agent 物理位置无关。

---

## 4. 特性设计亮点（对应简历项目一）

> 本节内容与简历「(一) 分离式 Agentic RL 后训练框架」的职责表述一一对应，面试时按这些亮点组织回答——每个亮点都要能讲出"设计动机 → 实现方式 → 量化收益"。

### 亮点 1：分离式训练链路（核心架构亮点）

**简历表述**：设计并实现分离式训练链路——Agent 经 SSH 隧道与 OpenAI 兼容端点接入云端 Gateway，token 级轨迹异步入库，奖励与训练在云端完成，沙箱仅负责执行。

**展开讲**（对应本文第 3 点的完整机制）：核心设计是"训练 / 推理 / 环境 / 奖励解耦"——agent 跑在任意位置（本地 WSL），通过 SSH 隧道（paramiko direct-tcpip）把模型调用指向云端 Gateway；Gateway 在服务模型的同时物化 token 级轨迹（prompt/response/logprob/mask/reward）；轨迹经 TransferQueue 异步入库；reward 由沙箱真实执行（pytest FAIL_TO_PASS）判定；训练在云端 verl 完成。**on-policy 的保证**：轨迹 logprob 由 Gateway 云侧生成，与 agent 物理位置无关——这是"任意位置部署 agent"能成立的理论基础。

### 亮点 2：EAGLE-3 投机解码

**简历表述**：集成 EAGLE-3 投机解码——生成吞吐由 199 提升至 282 tok/s（+41.7%），单 token 延迟降低 39.5%；开启投机后的全样本训练评测 83.23%，与 baseline 83.2% 持平，加速未引入训练质量损失。

**展开讲**：Agent RL 的生成瓶颈是"长 prompt 多轮"（SWE-bench/HumanEvalFix 每步 prompt+历史很长），decode 每步串行生成是吞吐瓶颈。EAGLE-3 用草稿模型一次草拟多个 token、大模型一次验证（详见 Speculative-Decoding.md），把串行 decode 变成"一次验证多位置"。**关键收益不是吞吐本身，而是"加速不损质量"**——做了严格 A/B：同配置 spec on/off 对照，25 步全样本训练后评测 83.23% vs baseline 83.2%（持平），证明投机解码的吞吐收益 +41.7% 没有引入训练质量损失。排障点：vLLM 0.11.1 在 EAGLE-3 下 `logprobs=0` 返回空 logprobs（vllm#30059），GRPO 的 old_log_prob 全丢无法训练，补丁改为 `logprobs=1` 修复。

### 亮点 3：双机分离式全异步 RL 训练（TQ MoonCake 存储后端）

**简历表述**：搭建双机 Ray 集群并实现分离式全异步 RL 训练（TQ MoonCake 存储后端）——平均单步耗时由同步方案的 79.4s 降至 48.1s（约 -39%）。

**展开讲**：双机架构 = node1（trainer，1×4090 48G）+ node2（独立 rollout 引擎，2×4090 24G），`separate_async` 模式下**采样与训练完全重叠**（node2 采第 N+1 批时 node1 在训第 N 批）。TQ（TransferQueue）作为两机之间的数据平面，**MoonCake 存储后端**支持 RDMA 高速传输。对照实验结论：`colocate_async`（trainer 与引擎同卡）在 vLLM 0.11.1 上不稳定（CUDA illegal memory），`separate_async` 稳定且收益显著（-39%）；MoonCake（TCP 传输下）与 SimpleStorage 训练效果一致、0 崩溃。排障点：`num_turns` 13B 写读类型不一致、空响应轨迹 0 字节 slice 等 TQ 链路 bug 已源码级修复。

### 亮点 4：黑盒（Claude Code）训练路径 + 手写 MCP 工具转发层

**简历表述**：接入黑盒（Claude Code）训练路径——针对黑盒无远程执行抽象的难题，手写 stdio JSON-RPC MCP 工具转发层，将 Bash/Read/Write/Edit 调用转发至云端沙箱执行；全样本 25 步训练通过率 80.75%（较基座 +4.35pp），行为分析显示黑盒初始能力更强但平均轮数翻倍、吞吐低约 35%。

**展开讲**：黑盒（Claude Code）闭源、无"远程执行抽象"，它的 Bash/Read/Write/Edit 工具调用发生在本地，必须被转发到云端沙箱执行。**核心工程**：手写 `sandbox_mcp_server.py`（stdio JSON-RPC MCP server，绕开 FastMCP 的拆包问题），把工具调用转发到云端腾讯沙箱；Claude Code 的模型调用走 Anthropic 协议经 Gateway 适配到云端 vLLM。**行为量化对比**（白盒 vs 黑盒）：

| 指标 | 白盒 mini-swe-agent | 黑盒 Claude Code |
|---|---|---|
| step1 起点通过率 | 0.29 | **0.57**（初始更强） |
| 平均轮数 | 11.4（训练后降 8-10） | **16-28（翻倍，无下降）** |
| 响应长度 | 1665 tokens | 3158 tokens |
| 吞吐 | 199 tok/s | 130 tok/s（**低约 35%**） |
| 最终评测 | 83.2% | 80.75%（+4.35pp） |

黑盒初始能力强但训练效率低（轮数/时长约 2 倍、吞吐低 35%），归因于长上下文 prefill 重复 + prefix cache 命中不足——这既是双 harness 平台化的价值（任意 agent 都能训练），也是黑盒路径的客观局限。

### 亮点 5：GRPO/LoRA 训练链路 + 稳定性排障

**简历表述**：基于 verl + vLLM 构建 GRPO/LoRA 训练链路（对应项目简介"基于 UniAgent 修改 GRPO/LoRA 训练链路"）。

**展开讲**：训练链路 = Qwen3-8B + LoRA（rank=32）+ FSDP2 + CPU offload + 梯度检查点 + fused kernels，打通数据构建（`make_humanevalfix_data.py`，161 条无测试泄露）→ 采样 → 训练 → 评测（`eval_humanevalfix.py`，n=1 / temp 0.8 / 并发 24）完整闭环。**稳定性排障**（体现源码级能力）：verl#7014（`lora.merge=True` 时 rollout 同步的是基座权重不带 adapter，梯度更新从未生效——修复后 LoRA 微调质量才真正生效）、vllm#30059（EAGLE-3 logprobs 空）、Gateway 工具解析 5.3% 错误率导致 38% 会话受影响（JSON repair + 合成重试修复为 0 会话失败）、权重同步 IPC 对 CPU 大张量崩溃等，共沉淀 23 项补丁（`patches/` 幂等应用）。

### 亮点 6：LoRA 引擎热插（训练-推理切换秒级）

**简历表述**：训练链路中的工程优化（对应"轨迹异步入库 + 云端统一训练"的效率基础）。

**展开讲**：LoRA + vLLM 引擎常驻（adapter 热插）——每步权重同步仅 2.5s（~100MB adapter）vs 全参 15GB refit，把"训练完 → 推理服务用新权重"的切换从分钟级降到秒级。这是 Agent RL"训练-推理频繁交替"场景的关键优化（每步训练后 agent 都要用新模型采样，如果每次全量 refit 15GB，训练根本跑不动）。

> **面试一句话总结**：本项目六大特性亮点——①分离式链路（agent 任意位置 + on-policy 云侧保证）；②EAGLE-3 投机解码（吞吐 +41.7% 且质量无损，A/B 验证）；③双机全异步 + TQ MoonCake（单步 -39%，采样训练完全重叠）；④黑盒 Claude Code + 手写 MCP 转发（80.75%，行为量化对比）；⑤GRPO/LoRA 全链路 + 23 项源码排障；⑥LoRA 引擎热插（2.5s 权重同步）——每点都能讲"动机 → 实现 → 量化收益"。

---

# 附：本项目自研组件速查表

| 组件 | 代码位置 | 角色 | 关键类 / 方法 |
|---|---|---|---|
| 白盒 runner | `uni_agent_ext/agents/mini_swe_agent_runner.py` | humaneval_fix 白盒训练 | `extract_task`、`create_task_sandbox`、`build_mini_swe_config`、`run_mini_swe_agent_api`、`evaluate_reward` |
| 黑盒 runner | `uni_agent_ext/agents/claude_code_runner.py` | Claude Code 黑盒训练 | `build_claude_task`、`build_claude_command` |
| 平台化 runner | `uni_agent_ext/agents/external_agent_runner.py` | 分离式训练侧：建沙箱+下发任务+等 done+云侧 reward | `_write_task_file`、`_wait_for_done`、`evaluate_reward_msa` |
| 腾讯沙箱 | `uni_agent_ext/sandbox/tencent_agent_runtime.py` | E2B 兼容沙箱后端 | `TencentAgentRuntimeSandbox`（官方 Sandbox 接口实现） |
| MCP 转发 | `scripts/sandbox_mcp_server.py` | 黑盒工具转发（stdio JSON-RPC） | Bash/Read/Write/Edit/Glob → 云端沙箱 |
| 本地 WSL 白盒 | `scripts/platform_local_agent.py` | 分离式本地侧：隧道+读任务+跑 agent+done | `TunnelForwarder`（paramiko direct-tcpip）、`wait_for_task`、`run_local_agent` |
| 本地 WSL 黑盒 | `scripts/platform_local_claude.py` | 分离式本地侧黑盒 | `build_mcp_config`、`build_claude_command`（`--disallowedTools` + MCP） |
| 平台化训练脚本 | `scripts/run_grpo_platform_test_ucloud.sh` | 加载已有权重 + external_agent_runner + save_freq=-1 | `main_ppo` + `agent_runners.external_agent` 配置 |

> **关联文档**：官方 uni-agent 组件（Gateway 轨迹处理、Reward 分配、Agent/Sandbox/Framework/TQ 抽象）详见 `Uniagent.md`。
