# SGLang 完全指南（RadixAttention 前缀树缓存 + 分层引擎）

> 对应简历"熟悉 vLLM、verl、uni-agent 等框架"知识体系中的推理服务侧对照：本文讲 **SGLang**（sglang-project/sglang，本地克隆 main 分支 commit `a328c19`）的核心机制。一句话知识框架：
> **SGLang 是"前缀树（radix tree）驱动的推理引擎"**——用 RadixAttention 在 token 粒度复用 KV cache，用 cache-aware 调度让复用最大化，用分层引擎（TokenizerManager → Scheduler → TpWorker → ModelRunner）支撑高吞吐在线服务；相比 vLLM 的 block 级前缀缓存，SGLang 把"前缀复用"做到了更细、更自动。
>
> 与本目录其他文档的关系：RadixAttention 的底层（分页 KV、attention backend、CUDA Graph）见 `VLLM.md`；投机解码（EAGLE 等）见 `Speculative-Decoding.md`；并行维度（TP/EP/DP）见 `Parallel.md`；机间传输（Mooncake KV 传输）见 `Mooncake.md`。

---

# 一、核心技术点

## 1. RadixAttention：前缀树 KV 缓存（SGLang 的灵魂）

### 1. 现有问题

- **推理服务中大量请求共享前缀**（system prompt、few-shot 示例、多轮对话的历史、检索增强的上下文），这些前缀的 KV cache 理论上只需算一次、被所有请求复用。但通用 KV cache 管理（vLLM 的 PagedAttention + 哈希前缀缓存）有几个痛点：
  1. **共享粒度粗**：vLLM 的 Automatic Prefix Caching 以 **block（页）为单位**做哈希匹配，共享只发生在"整块对齐"的前缀上——两个请求差几个 token，就可能一个都共享不了；
  2. **前缀动态增长**：多轮对话里每轮追加新 token，前缀不断变长；静态的哈希块很难高效表达"共享前缀 + 各自后缀"的树形结构；
  3. **命中后算力浪费**：前缀一旦共享，命中的部分不需要重算注意力，但如果缓存查找本身很贵（哈希全表扫描、块对齐截断），收益会被吃掉。
- 量化后果：典型 LLM 服务（ChatGPT 风格多轮、Agent 工具调用、RAG）中 **30%~70% 的输入 token 是重复前缀**；共享不了 = 这些 token 每请求重算一遍 prefill，既拖慢 TTFT 又烧算力。

### 2. 方法论

SGLang 用**基数树（radix tree / prefix tree）**管理 KV cache：树的每个节点存**一段连续 token 序列**（如 `"北京邮电大学的人工智能专业"`），边就是字符/token 的共享前缀；**公共前缀只有一个节点、被多个请求共享**；请求结束/增长时节点动态**分裂（split）与合并（merge）**。核心实现在 `python/sglang/srt/mem_cache/radix_cache.py`。

**① 键（RadixKey）**：一次查找/插入的身份 = token 序列 + 可选的隔离标签：

```python
# python/sglang/srt/mem_cache/radix_cache.py
class RadixKey:
    def __init__(self, token_ids, extra_key=None, is_bigram=False, cache_salt=None, page_size=1):
        ...
    def page_aligned(self, page_size: int) -> RadixKey:
        """truncate to multiple of page_size for alignment"""
    def maybe_to_bigram_view(self, is_eagle: bool, value=None):
        """EAGLE 投机解码用 bigram 视图（两个 token 拼成一个 key 单元）"""
    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """返回与 other 共享的最长前缀长度"""
```

`extra_key` 用于隔离（不同 LoRA adapter、不同采样盐不能共享缓存），`cache_salt` 也是隔离维度，`is_bigram` 是 EAGLE 投机解码时的 key 变形。

**② 查找（match_prefix）**：从根节点沿树走，返回**最长匹配前缀的 KV 索引**：

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """Find the longest cached prefix of ``key`` in the radix tree."""
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)
    if self.disable or len(key) == 0:
        return self._empty_match_result
    key = key.page_aligned(self.page_size)
    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = self._empty_match_result.device_indices
    return MatchResult(device_indices=value, last_device_node=last_node, ...)
```

`device_indices` 是被复用 KV 块的设备索引列表；命中后该请求只需对**未命中的后缀**做 extend（增量 prefill）。查找可能**分裂节点**：若匹配停在一个已存节点的中间，先把该节点沿匹配边界劈开，再返回——这是 radix tree 支持"任意长度前缀共享"的关键（`_split_node`）。

**③ 插入与生命周期**：请求结束（`cache_finished_req`）把完整 token 序列及其 KV 索引插入树；运行中的请求（`cache_unfinished_req`）可以边解码边把已生成部分挂进树（chunked 模式）：

```python
def cache_finished_req(self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int):
    """Cache request when it finishes."""
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
    kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, : len(token_ids)]
    radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle,
                         cache_salt=req.cache_salt).page_aligned(self.page_size)
    values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)
    if is_insert:
        result = self.insert(InsertParams(key=radix_key, value=values, priority=...))
```

**④ 淘汰（eviction）**：显存有限，树要淘汰冷节点。`server_args` 里 `--radix-eviction-policy` 可选 `lru / lfu / slru / priority`，淘汰单位是**叶子节点**（`_delete_leaf`），且被正在运行请求引用的节点会被 `lock_ref` 保护不被淘汰（`inc_lock_ref` / `dec_lock_ref`）。

**这段代码关键在哪**：① `RadixKey.match` 逐 token 比较、`page_aligned` 可关闭对齐（默认 page_size=1 → **token 级共享**，vLLM 的 block 对齐被取消）；② `match_prefix` 返回的是"最长前缀的 KV 索引"，命中即零计算；③ `cache_finished_req` 把"用完即弃"的 KV 变成"可复用资产"；④ 叶子淘汰 + lock_ref 保护保证在线请求的缓存不被误删。

### 3. 具体数值样例

设两个请求（忽略 BOS）：
- 请求 A：`北京邮电大学的人工智能专业怎么样`（20 token）
- 请求 B：`北京邮电大学的人工智能专业怎么找工作`（22 token）

两请求共享前 **17 个 token**（`北京邮电大学的人工智能专业怎么`）。radix 树形态：

```text
root
 └─ "北京邮电大学的人工智能专业怎么"（17 token，共享节点，KV 复用）
     ├─ "样"（A 的叶子）
     └─ "找工作"（B 的叶子）
```

- B 到达时 `match_prefix` 沿树走到共享节点，`device_indices` 返回这 17 个 token 的 KV 索引；
- B 只需对后缀 `找工作`（5 token）做 **extend 增量 prefill**，其余 17 token **零计算、零显存分配**；
- 对比 block 对齐（假设 16 token/block）：17 token 前缀需要 2 个 block，第二个 block 只有 1 个 token 对齐 → vLLM 式对齐下 B 只能命中 16 token（第 1 块），sglang 命中 17 token——**每多命中 1 token，就少算 1 token 的注意力**；多轮对话每轮追加 token 时树节点继续沿路径生长，下一轮直接命中。
- 显存侧：假设单 token KV 约 0.5 KB（8B 模型、GQA、bf16），1000 个请求共享 1000 token 前缀 → 省下约 1000×1000×0.5 KB ≈ 500 MB 的重复 KV 计算与存储。

> 面试一句话总结：**RadixAttention 用 radix tree 管理 KV cache：公共前缀只存一份、按 token 粒度共享，请求到达时 match_prefix 拿回最长匹配的 KV 索引、只对未命中后缀做增量 prefill，节点按需分裂/合并、LRU 类策略淘汰叶子——把"前缀复用"从 vLLM 的 block 级对齐提升到 token 级自动复用，是 SGLang 相对 vLLM 最核心的差异化。**

### 4. 演进与当前状态

- **chunked prefix cache**：新版支持"分块前缀缓存"布局（`CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`），部分 attention backend 可读该布局，降低大前缀的锁开销；
- **多级缓存**：`hiradix_cache.py` / `hicache_storage.py` 引入主机内存（host）分层缓存，GPU 放不下的前缀可放 host，命中再搬回（配合 `--radix-eviction-policy` 与 protect_host）；
- **EAGLE bigram**：`is_bigram` 视图让 radix tree 直接服务 EAGLE 投机解码的 bigram 草稿缓存（见 `Speculative-Decoding.md`）；
- 该机制在 SGLang 中始终是核心路径，未见被移除的迹象。

---

## 2. Cache-aware 调度：LPM / DFS-weight，让缓存命中率最大化

### 1. 现有问题

- 有缓存 ≠ 命中率高：**先调度谁、后调度谁，直接决定前缀复用率**。若一批请求都有 50% 前缀重叠，但调度器按到达顺序（FCFS）乱序处理，缓存会被不相关请求互相顶掉，命中率骤降；
- 更深一层：radix 树里"先扩展哪个子树"也影响长期命中——**同一子树下的请求互相共享，跨子树请求互相驱逐**；
- 多轮对话/Agent 场景请求的 max_new_tokens 差异巨大（有的回 1 个字，有的生成 2000 token），调度顺序还影响**队头阻塞**（长输出请求堵住短输出请求）。

### 2. 方法论

SGLang 把调度策略分成两类（`python/sglang/srt/managers/schedule_policy.py`），**每步动态选择**（`_determine_active_policy`）：

```python
class CacheAwarePolicy(Enum):
    LPM = "lpm"            # longest prefix match：最长前缀优先
    DFS_WEIGHT = "dfs-weight"  # 按子树请求数的深度优先加权

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"          # 先来先服务
    LOF = "lof"            # longest output first：长输出优先
    RANDOM = "random"
```

**LPM（最长前缀匹配优先）**：等待队列按"已匹配前缀长度"降序排——匹配最长的请求先被调度，让共享前缀尽快物化、复用最大化：

```python
@staticmethod
def _sort_by_longest_prefix(waiting_queue, temporary_deprioritized):
    """Sorts the waiting queue based on the longest prefix match."""
    waiting_queue.sort(key=lambda r: (
        -r.num_matched_prefix_tokens
        if r.rid not in temporary_deprioritized else float("inf")
    ))
```

**DFS-weight（深度优先加权）**：从根节点递归，**请求数多的子树优先整体调度**——把树切分（cache 分叉）推迟到最后一刻，最大化批内共享：

```python
@staticmethod
def _sort_by_dfs_weight(waiting_queue, tree_cache):
    """Sorts the waiting queue based on a depth-first search weighting."""
    last_node_to_reqs = defaultdict(list)
    for req in waiting_queue:
        last_node = tree_cache.resolve_node_handle(req.last_node)
        last_node_to_reqs[last_node].append(req)
    node_to_weight = defaultdict(int)
    for node in last_node_to_reqs:
        node_to_weight[node] = len(last_node_to_reqs[node])
    SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)
    waiting_queue.clear()
    SchedulePolicy._get_dfs_priority(tree_cache.root_node, node_to_weight,
                                     last_node_to_reqs, waiting_queue)
```

**优先级调度**：`enable_priority_scheduling` 时先按 `req.priority` 排（`_sort_by_priority_and_fcfs`），业务可以把 VIP 请求/在线交互请求插到批量任务前面；同时有 `_should_defer_prefill` / `prefill_delayer` 控制 prefill 与 decode 的穿插（见第 3 点）。

**这段代码关键在哪**：LPM 是"**逐请求贪心**"（谁命中多谁先），DFS-weight 是"**全局树视角**"（哪个子树共享潜力大先喂饱哪个）——两者都直接操作 radix 树结构（`last_node`、`resolve_node_handle`），说明调度器与缓存是**耦合设计**的，而不是像 vLLM 那样"先调度、再查缓存"。

### 3. 具体数值样例

设等待队列 3 个请求，`match_prefix` 后各自命中数：A=120、B=80、C=0（无共享前缀）：

- **LPM**：调度顺序 A → B → C。A、B 的 120/80 token 前缀都已在缓存（或先被 A 物化、B 复用），prefill 计算量 = (A 后缀 + B 后缀 + C 全长)；若先跑 C，C 的长 prefill 会挤压显存，可能把 A、B 要用的前缀**淘汰掉**，A、B 全部重算；
- **DFS-weight 数值**：树根下两个子树 S1（含 5 个请求，共享 200 token 前缀）与 S2（含 1 个请求，无共享）。`node_to_weight[S1]=5, [S2]=1`，`_calc_weight` 向上累加，根节点 DFS 先走进 S1——5 个请求共享的 200 token 前缀只物化一次，后面 4 个请求全部命中；
- **对比**：FCFS 下若 5 个 S1 请求被 1 个 S2 请求插队打散，前缀物化 5 次；LPM/DFS-weight 下物化 1 次——**5 倍的前缀 prefill 计算差**。

> 面试一句话总结：**SGLang 的调度是 cache-aware 的：LPM 让"命中前缀最长的请求"先跑，DFS-weight 让"共享潜力最大的子树"整体先跑，把 radix 树的复用潜力榨干；配合 priority 优先级调度和 prefill/decode 穿插，兼顾命中率、公平性与延迟。**

---

## 3. 连续批处理与 chunked prefill：细粒度准入控制

### 1. 现有问题

- **prefill 与 decode 的天平**：prefill 是计算密集（一次算整段），decode 是访存密集（一次算一个 token）。若一个大 prefill（如 4096 token）独占 GPU，正在 decode 的请求全部卡住（TTFT 与 TPOT 互相伤害）；若 prefill 太多，decode 吞吐饿死；
- **长 prefill 的显存峰值**：一个 32K token 的 prefill 即使共享前缀后，剩余后缀的一次性计算也产生巨大中间激活，容易 OOM；
- 连续批处理（continuous batching）本身 vLLM 也有，但 SGLang 的**准入控制（admission control）**做得更细：不是"整请求准入"，而是"**按 chunk 准入**"。

### 2. 方法论

核心在 `python/sglang/srt/managers/scheduler.py` 的 `event_loop` 与 `PrefillAdder`。

**① 主循环**（`run_event_loop` → `event_loop_normal`）：

```python
# python/sglang/srt/managers/scheduler.py（结构）
def run_event_loop(self):
    while True:
        if self.overlap:
            self.event_loop_overlap()      # 与 CUDA graph 捕获/解码重叠
        else:
            self.event_loop_normal()       # 常规：取请求 → 组 batch → forward
        if self.is_time_to_exit():
            break

def event_loop_normal(self):
    # 1. 从队列收新请求（process_input_requests，做 radix match）
    # 2. PrefillAdder 决定本轮放多少 prefill 进 batch（按 token 预算）
    # 3. 组 ScheduleBatch → forward_batch_generation → 解码 → 回写输出
    # 4. 每步都检查是否有请求完成，完成后 cache_finished_req 挂进 radix 树
```

**② chunked prefill（准入粒度=chunk）**：`--chunked-prefill-size` 把大 prefill 切成若干 token chunk，每个 chunk 与 decode 交错执行：

```python
# python/sglang/srt/server_args.py（关键参数）
chunked_prefill_size: Optional[int] = Field(
    default=4096,
    description="The maximum number of tokens in a chunk for the chunked prefill. "
                "Setting this to -1 means disabling chunked prefill.")
```

- `PrefillAdder._admitted_extend_lens` / `_check_prefill_tile_budget`：按**每步 token 预算**（如 8192 token/步）决定本轮准入哪些请求的哪些 chunk，预算用完即停，多余请求留到下步；
- `_should_defer_prefill`：当 decode 批已经很大（或新 prefill 会挤压 decode 显存）时**推迟 prefill**，优先保 decode 的 TPOT——这是"prefill 给 decode 让路"的显式策略；
- `schedule_batch.py` 的 `ScheduleBatch` 是单步执行单元：一个 batch 里同时有正在 decode 的旧请求和刚准入的 prefill chunk，一次 forward 混合执行。

**③ 请求生命周期**：`process_input_requests` 对新请求先做 radix `match_prefix`（记录命中长度），再决定是"纯 decode 起步"（全命中）还是"extend 起步"（部分命中）；`Req` 的状态机（`req.state`）贯穿 WAIT → RUN → DONE。

**这段代码关键在哪**：准入不是"请求级"而是"**token 预算级**"——每步 forward 的 token 数被预算卡住，prefill chunk 与 decode 在同一 batch 混合，长 prefill 不会一次性占满 GPU；`_should_defer_prefill` 是显式的"decode 优先"策略，保证在线请求的 TPOT 稳定。

### 3. 具体数值样例

- 场景：运行批 8 个请求正在 decode（每步 8 token），新到 2 个 prefill 请求，各 2048 token 后缀；`chunked_prefill_size=512`、每步 token 预算 1024；
- **无 chunked prefill**：2×2048 = 4096 token 一次性 prefill，decode 批被完全阻塞数步，TPOT 恶化；
- **chunked**：每个 prefill 请求切成 4 个 512-token chunk；第 1 步准入 prefill chunk A1（512 token）+ 继续 8 个 decode（8 token）→ 每步混合执行约 520 token；4 步后两个 prefill 全部完成，decode 全程只被轻微穿插影响；
- **准入预算**：若预算 1024 token/步，则每步最多放 2 个 512 chunk（或 1 个 1024 chunk）；`PrefillAdder` 按 `_admitted_extend_lens` 算好后逐 chunk 加入 `ScheduleBatch`，超预算的请求留在等待队列；
- 效果：大 prefill 的**峰值激活内存**从 4096 token 规模降到 512 token 规模，TTFT 与 TPOT 的权衡由 chunk 大小一个旋钮控制。

> 面试一句话总结：**SGLang 的连续批处理把"准入控制"细化到 token 预算级：chunked prefill 把大 prefill 切成小块与 decode 交错执行，PrefillAdder 按每步 token 预算准入、_should_defer_prefill 在 decode 拥挤时给 prefill 让路——用一个小旋钮（chunked_prefill_size）同时压住显存峰值和 TTFT/TPOT 权衡。**

---

## 4. 结构化输出（constrained decoding）：xgrammar / outlines / llguidance

### 1. 现有问题

- Agent / 函数调用场景要求模型输出**严格合法的 JSON / 代码 / 特定格式**。普通采样 + 事后解析，失败率可能 5%~20%（长 JSON 极易漏括号、多逗号），重试一次就是一次完整生成成本；
- 更糟的是：**格式错误会连锁**——下游 JSON parser 崩、工具调用失败、Agent 循环断裂；
- 朴素"采样后校验重试"在 RL 训练里还会污染 reward 信号（格式错 = 奖励 0，模型学不到"格式本身"的梯度）。

### 2. 方法论

SGLang 原生内置**文法约束解码**：用 grammar（CFG/正则/JSON schema）在**每一步采样时掩码掉不合法 token**，保证输出 100% 合法。三个可切换后端（`python/sglang/srt/constrained/`）：

```text
python/sglang/srt/constrained/
├── base_grammar_backend.py     # 抽象基类：GrammarBackend
├── grammar_manager.py          # 管理请求↔grammar 状态
├── xgrammar_backend.py         # xgrammar 后端（默认，字节级 GBNF）
├── outlines_backend.py         # outlines 后端（FSM/正则）
├── llguidance_backend.py       # llguidance 后端（微软，增量文法）
├── outlines_jump_forward.py    # outlines 的 jump-forward（跳过已知前缀）
└── reasoner_grammar_backend.py # 推理器专用文法
```

核心机制（以 xgrammar 为例）：

1. **编译期**：JSON schema / regex 编译成 **GBNF 文法**（Grammar-BNF），再转成**确定有限自动机（DFA）**——每个状态 = "下一个合法 token 集合"；
2. **解码期**：每步采样前，根据当前文法状态 + 已生成 token，得到**合法 token 掩码**（logits 处理器把非法 token 的 logit 置 -inf），采样只在合法集内进行；
3. **状态推进**：选中的 token 喂回 DFA 推进状态；支持 **jump-forward**——当文法前缀完全确定时（如 JSON 的 `"key":` 字面量），直接跳过采样、一次性生成固定 token；
4. 请求级隔离：`grammar_manager` 按请求维护文法状态机，多请求并发互不干扰。

### 3. 具体数值样例

- 要求输出 `{"name": str, "age": int}` 的 JSON：
  - 采样第 1 个 token 时，DFA 只允许 `{` 或空白；`{` 被选后，状态推进到"必须输出 `"name"` 字面量"；
  - 后续每步：合法 token 集合被约束（如 `"`、`:`、`,`, `}`、数字、字符串字符），**非法 token（如字母进 age 字段）被掩码为 -inf，采样器不可能选出**；
  - jump-forward：`"name":` 这段字面量完全确定，后端一次生成整段，不走逐 token 采样；
- 效果：JSON 合法率从"采样后解析的 ~85%~95%"到 **100%**；生成 token 数可能反而减少（跳过了格式 token 的逐位采样）；
- 在 RL 训练（verl + vLLM/sglang）里，结构化输出让 reward 只反映"任务成败"而非"格式对不对"，是 Agentic RL 的标配。

> 面试一句话总结：**结构化输出 = 用 grammar（JSON schema/正则）编译成 DFA，解码每一步用"合法 token 掩码"约束采样，保证输出 100% 合法，配合 jump-forward 跳过确定性前缀——比"采样后解析重试"既快又稳，是 Agent/工具调用场景的标配能力，SGLang 原生支持 xgrammar/outlines/llguidance 三后端。**

---

## 5. 投机解码支持（EAGLE / n-gram）：指向专项文档

### 1. 现有问题

- decode 是访存密集的**串行瓶颈**：每步只出一个 token，但要走完整模型前向（读全部权重）；
- 投机解码用"小草稿模型先猜 k 个 token → 大模型一次验证"把串行步数摊薄（详见 `Speculative-Decoding.md`）。

### 2. 方法论（SGLang 的实现要点）

SGLang 原生支持多种投机方案（`python/sglang/srt/speculative/`），且与 RadixAttention 深度集成：

```text
python/sglang/srt/speculative/
├── eagle_worker_v2.py / eagle_utils.py / eagle_info.py   # EAGLE-1/2/3 草稿 worker
├── multi_layer_eagle_worker_v2.py                        # 多层 EAGLE
├── ngram_worker.py / ngram_info.py                       # n-gram 投机（无需草稿模型）
├── adaptive_runtime_state.py / adaptive_spec_params.py   # 自适应投机参数（按批调整草稿长度）
├── decoupled_spec_io.py                                  # 解耦的草稿/验证 IO
├── ragged_verify.py                                      # 树形草稿的批量验证
└── spec_info.py / spec_registry.py                       # 投机信息注册
```

- **EAGLE**：草稿模块输入目标模型**倒数第二层 hidden state**，自回归预测下一个 token 的 feature，再用共享 LM head 出 token（比独立小模型 draft 质量高，见 `Speculative-Decoding.md`）；SGLang 的 `eagle_utils.py` 用**树形草稿（tree draft）+ build_tree_kernel 一次前向验证**；
- **n-gram 投机**：`ngram_worker` 用上下文里的 n-gram 统计当草稿（零额外模型），适合代码/重复文本；
- **自适应投机**：`adaptive_spec_params` 按当前 batch 的接受率动态调整 `num_speculative_tokens`，接受率高就多猜；
- 与 radix tree 的联动：`RadixKey.is_bigram` 让 EAGLE 的 bigram 草稿缓存直接复用前缀树（`maybe_to_bigram_view`）。

### 3. 具体数值样例

- 以 EAGLE 草稿长度 k=3 为例：草稿网络逐 token 猜 3 个 token（成本约主模型 1 次前向的 10%~20%），主模型一次前向并行验证 3 个位置，若全部接受 → **一步出 3 个 token**，decode 步数降到 1/3；
- 接受率 α=0.7 时，实际加速 ≈ $1/((1-\alpha) + \alpha \cdot c)$ 量级（c 为草稿/主模型单步成本比，细节见 `Speculative-Decoding.md` 的完整演算）；
- SGLang 里开 EAGLE 只需 `--speculative-algorithm EAGLE --speculative-draft-model <path>` 两个参数。

> 面试一句话总结：**SGLang 原生支持 EAGLE（树形草稿 + bigram 前缀树缓存）、n-gram（零模型）、自适应投机长度，用"草稿多猜 + 主模型一次验证"摊薄 decode 串行步数——与 RadixAttention 的 bigram 视图集成是它的特色，完整原理见 Speculative-Decoding.md。**

---

## 6. 分布式并行：TP / EP 的通信封装

### 1. 现有问题

- 单卡放不下大模型（权重 + KV + 激活），必须张量并行（TP）把一层切成多卡；MoE 模型还要专家并行（EP）把专家分布到多卡；
- TP/EP 的每次前向都伴随 **all-reduce / all-gather** 通信，通信量直接决定扩展效率（通信占比高 → 加速比低）。

### 2. 方法论

SGLang 的通信封装在 `python/sglang/srt/distributed/`，一组**组感知（group-aware）的通信算子**，每个算子绑定到特定并行组（TP 组 / EP 组 / attention TP 组）：

```python
# python/sglang/srt/distributed/communication_op.py
from .parallel_state import (get_attn_tp_group, get_moe_ep_group, get_moe_tp_group, get_tp_group)

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)

def tensor_model_parallel_fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps):
    """Fused TP all-reduce + RMSNorm（把通信与归一化融合，少一次 kernel 启动）"""
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)

def attention_tensor_model_parallel_all_reduce(input_):
    return get_attn_tp_group().all_reduce(input_)      # 注意力并行组

def moe_tensor_model_parallel_all_reduce(input_):
    return get_moe_tp_group().all_reduce(input_)       # MoE 的 TP 组
def moe_expert_parallel_all_reduce(input_):
    return get_moe_ep_group().all_reduce(input_)       # MoE 的 EP 组
```

- `parallel_state.py` 维护多个 **GroupCoordinator**：`tp_group`（张量并行）、`ep_group`（专家并行）、`attn_tp_group`（注意力 TP，可独立于权重 TP）、`moe_tp_group`（MoE 权重切分的 TP）；
- 底层通信走 `torch.distributed`（NCCL 等），`GroupCoordinator` 封装了 NCCL 组初始化、P2P 传输选择（NVLink/PCIe/IB）、以及 fused kernel 的调度（如 fused allreduce + RMSNorm）；
- 多个并行维度可叠加：TP 内通信走机内 NVLink，EP 跨机走 IB/RoCE——组划分决定通信走哪条物理链路（详见 `Parallel.md` / `Communication.md`）。

### 3. 具体数值样例

- 8B 模型 TP=2：每层 MLP 的 all-reduce 通信量 ≈ 2 × 隐藏维 × batch token 数 × 2 bytes（bf16）；假设 hidden=4096、batch 256 token → 单次 all-reduce ≈ 2×4096×256×2 ≈ 4 MB，跨 NVLink（900 GB/s 单向）耗时 ≈ 4.4 µs，跨 PCIe 5.0（64 GB/s）≈ 63 µs——**通信链路选择差一个数量级**；
- MoE 模型（如 DeepSeek 系）EP=8：专家激活的 all-to-all 通信量与"每 token 路由到多少专家"正相关，`moe_expert_parallel_all_reduce` 只归约本组专家的输出，避免整模型 all-reduce；
- 实际：`--tp-size 2 --dp-size 4` 组合时，TP 组在机内、DP 副本在机间，`GroupCoordinator` 让每类通信自动落在正确的物理链路上。

> 面试一句话总结：**SGLang 的分布式通信按"并行组"封装：TP 组 / EP 组 / attention TP 组各自持有 GroupCoordinator，all-reduce/all-gather 自动绑定到对应物理链路（机内 NVLink / 机间 IB-RoCE），并支持 fused allreduce+RMSNorm 等融合算子——通信组划分 = 物理拓扑的软件映射，是并行效率的关键。**

---

## 7. CUDA Graph 捕获与模型执行（与 vLLM 的差异点）

### 1. 现有问题

- decode 每步的 kernel 启动开销（launch overhead）在短序列下占比极高：一个 1 token 的 forward 可能只算 1ms，而 kernel 启动累计 0.1~0.3ms；
- 捕获成 CUDA Graph 后，整段 forward 一次 launch，把启动开销摊没。

### 2. 方法论（SGLang 的实现）

- `model_executor/model_runner.py` 的 `ModelRunner` 负责 CUDA Graph 捕获：按不同的 **batch size（最大 token 数）** 捕获多组 graph（`cuda_graph_config.py`、`cuda_graph_buffer_registry.py` 管理图缓存）；
- 与 vLLM 的区别：SGLang 的捕获是**按 forward mode 区分**（`ForwardMode`：prefill / extend / decode），extend（增量 prefill）也有独立 graph；graph 里 KV cache 用**固定地址池**（`memory_pool.py`），捕获后地址不变才能重放；
- `event_loop_overlap` 支持**捕获与执行重叠**：一个 batch 在 GPU 上跑时，CPU 侧预取/准备下一个 batch（`batch_overlap` 目录），进一步隐藏调度开销；
- attention backend 选择：`init_all_attention_backends` 支持 FlashAttention / FlashInfer / Triton / 昇腾 等，RadixAttention 层通过 `get_attn_backend()` 分发（见第 1 点的 `radix_attention.py` forward）。

### 3. 具体数值样例

- 捕获 8 个不同 batch size 的 decode graph（如 1/2/4/8/16/32/64/128 token），运行时按实际 batch 就近选择（不足则 padding 到最近的图尺寸）；
- decode 单步 kernel 启动开销从 ~0.2ms 降到 ~0.02ms，纯 decode 吞吐提升 10%~30%（batch 越大越接近饱和）；
- 代价：graph 捕获占显存（每个图一份中间 buffer），`--cuda-graph-max-bs` 控制上限，超出走 eager 回退。

> 面试一句话总结：**SGLang 用 CUDA Graph 把 decode/extend 的整段前向一次 launch，按 batch size 分级捕获、捕获与执行重叠，把 kernel 启动开销摊没（10%~30% 吞吐收益）；与 vLLM 的主要差异是 extend（增量 prefill）也有独立 graph、且与 RadixAttention 的固定地址 KV 池配合重放。**

---

# 二、整体架构：把技术点串成一条服务流水线

## 8. 引擎分层：TokenizerManager → Scheduler → TpWorker → ModelRunner

### 1. 现有问题

单进程服务把"HTTP 解析、tokenize、调度、模型前向、detokenize"全塞在一起，会互相拖慢：tokenizer（CPU、可能多线程）阻塞 GPU 前向，慢请求堵住快请求。需要一个**异步分层的流水线**。

### 2. 方法论

SGLang 的引擎是**多进程 + IPC** 的流水线（`python/sglang/srt/managers/` + `entrypoints/`）：

```text
HTTP/入口（http_server → engine）
   │
   ▼
TokenizerManager（tokenize + 输入规范化，CPU 异步）      ← managers/tokenizer_manager.py
   │  ZeroMQ / socket IPC（PortArgs）
   ▼
DataParallelController（dp>1 时按负载路由）               ← managers/data_parallel_controller.py
   │
   ▼
Scheduler（每 dp 一个：radix match + 组 batch + 准入）     ← managers/scheduler.py
   │
   ▼
TpWorker（forward_batch 执行，TP 组内 NCCL 通信）          ← managers/tp_worker.py
   │
   ▼
ModelRunner（CUDA Graph + attention backend + KV 池）      ← model_executor/model_runner.py
   │
   ▼
（反向：采样 → detokenize → 返回）
```

**① TokenizerManager**（前端大脑）：异步 `_tokenize_texts` / `_tokenize_one_request`，把请求转成 token ids + 输入格式检测（`_detect_input_format`），再 `_dispatch_to_scheduler` 发给对应 scheduler；它还管权重更新广播（`init_weight_update`，RL 训练每轮更新权重就是走这里）、LoRA 管理。

**② Scheduler**（每 DP 副本一个）：第 3 点的 event_loop 在这里跑，持有 `RadixCache`、`ReqToTokenPool`、`TokenToKVPoolAllocator`；`process_input_requests` 做 radix match 后进 `ScheduleBatch`。

**③ TpWorker**（执行者）：`forward_batch_generation(forward_batch)` 把 `ScheduleBatch` 转成 GPU 上的 `ForwardBatch`，驱动 `ModelRunner` 前向；`update_weights_from_disk` / `init_weights_update_group` 支持 RL 训练的热更新权重（verl 每轮训完推权重）。

**④ 进程间通信（IPC）**：`init_ipc_channels(port_args)` 用 **ZeroMQ（或 socket）** 在进程间传请求对象（`managers/communicator.py` 封装），而不是序列化成 HTTP——省掉重复的 HTTP 开销；HTTP 只在最外层（`http_server.py`）出现一次。

**这段代码关键在哪**：分层把"**CPU 密集（tokenize/detokenize）**"与"**GPU 密集（前向）**"异步隔离，调度器在中间做缓冲；IPC 直连让层间传输零 HTTP 开销；权重热更新通道（TpWorker.update_weights_from_disk）是它作为 **RL 训练推理引擎**（verl 集成）的关键能力。

### 3. 具体数值样例

- 部署 `--tp-size 2 --dp-size 2`：2 个 scheduler（各持一份 radix cache），每个 scheduler 下 2 个 TpWorker 组成一个 TP 组；
- 请求到达：TokenizerManager tokenize（CPU，~0.1ms/请求）→ DP 控制器按负载把请求发给 scheduler-0 或 scheduler-1（`LoadBalanceMethod` 支持按估算 token 数加权）→ scheduler radix match（命中 800 token）→ 组 batch → TpWorker 前向（TP 组内 all-reduce 走 NVLink）；
- 权重更新（RL）：verl 每轮训练完，`update_weights_from_disk` 让每个 TpWorker 重载权重，**不需要重启服务**——推理引擎常驻，训练与推理解耦（对应简历里"LoRA 引擎常驻 / 权重热更新"）。

> 面试一句话总结：**SGLang 引擎 = TokenizerManager（CPU 异步 tokenize）→ DataParallelController（按负载路由）→ Scheduler（radix 缓存 + 组批 + 准入）→ TpWorker（forward + 权重热更新）→ ModelRunner（CUDA Graph + attention backend）的多进程流水线，层间 ZeroMQ IPC 直连，权重热更新让它能当 RL 训练的常驻推理引擎。**

---

## 9. 数据并行（DP）与负载均衡

### 1. 现有问题

- 单 scheduler 的批处理吞吐受限于单进程调度/单 GPU 组；`--dp-size N` 可以跑 N 个独立 scheduler（各配自己的 TP 组），但**请求怎么分**决定负载是否均衡：按请求数均分会被"一个 32K 长 prefill + 一堆 1 token 请求"打破；
- 不同请求的 **radix 命中率不同**（命中多的算力小），简单轮询会让 cache 局部性变差。

### 2. 方法论

`managers/data_parallel_controller.py` 实现负载感知分发：

```python
class LoadBalanceMethod(Enum):
    """请求分发的负载均衡方法"""

class DPBudget:
    def __init__(self, dp_size: int):
        ...
    def update_budget(self, loads):
        """根据各 scheduler 当前负载更新预算"""
    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        """按方法选目标 scheduler，返回其 index"""

class DataParallelController:
    def __init__(self, ...):
        ...
    def dispatch(self, req: Req, ...):
        """给请求选择 dp 目标；支持按 estimated_tokens 加权、cache 亲和路由"""
```

- 每个 scheduler 独立持有 radix cache → **DP 之间缓存不共享**（各自为政），所以 DP 路由还考虑 **cache 亲和性**（同类请求尽量进同一 scheduler，提高局部命中）；
- `update_budget` 周期性从各 scheduler 收集负载，动态调整分发权重；`add_elastic_workers` / `update_active_ranks` 支持**弹性扩缩容**（训练时加机器）。

### 3. 具体数值样例

- `--dp-size 4`：4 个 scheduler；请求 A（命中 90%）与请求 B（命中 0%）若轮询分发，可能把 A 分到 cache 里没有其前缀的 scheduler → A 变成全 prefill；
- 带 cache 亲和的路由：按 `req.extra_key`（如会话 ID）哈希到固定 scheduler → 同一会话的多轮请求永远进同一 scheduler，多轮前缀命中率从 ~30% 提升到 ~90%；
- 负载感知：两个 scheduler 分别有 2000/100 token 在跑，新请求按 `estimated_tokens` 加权发给负载低的那个，避免队头阻塞。

> 面试一句话总结：**SGLang 的数据并行 = 多个独立 scheduler（各自持有 radix cache），DataParallelController 按"估算 token 数 + cache 亲和性"动态分发请求，避免长 prefill 打爆单点、并让同一会话的请求固定路由到同一 scheduler 以维持前缀命中率。**

---

## 10. SGLang vs vLLM：面试对比速查

| 维度 | SGLang（RadixAttention 系） | vLLM（PagedAttention 系） |
|---|---|---|
| KV 缓存管理 | **radix tree，token 级共享**，自动分裂/合并 | **block（页）表，block 级共享**，哈希精确匹配 |
| 前缀缓存 | RadixAttention 原生、自动、cache-aware 调度 | APC（Automatic Prefix Caching）以 block 为单位，调度基本 FCFS |
| 调度策略 | LPM / DFS-weight / priority（cache-aware） | FCFS + 简单优先级；v0.25 后 MRv2 调度更细 |
| chunked prefill | 内置，准入按 token 预算，prefill/decode 穿插 | 内置 chunked prefill（chunk 级） |
| 结构化输出 | **原生三后端**（xgrammar/outlines/llguidance） | 需外挂（outlines/xgrammar 插件） |
| 投机解码 | 原生 EAGLE/n-gram/自适应 | 原生 EAGLE/Medusa/MLP/grammar 等（见 Speculative-Decoding.md） |
| 引擎分层 | TokenizerManager→Scheduler→TpWorker→ModelRunner，ZeroMQ IPC | EngineCore→Scheduler→Worker→ModelRunner（v0.25 MRv2） |
| 权重热更新 | TpWorker.update_weights_from_disk（RL 训练友好） | 支持（LLM.update_weights），v1 引擎下也成熟 |
| 语言 | Python 引擎 + Rust 扩展（rust/ 目录） | Python + C++ 核心 |

> 面试一句话总结：**SGLang 与 vLLM 的最大分水岭是 KV 缓存：vLLM 是"block 级 PagedAttention + 哈希前缀缓存"，SGLang 是"token 级 radix tree + cache-aware 调度"；前者工程稳健、生态最广，后者前缀复用更细、结构化输出与投机解码原生内置——选型看场景：重前缀复用/Agent/结构化输出选 SGLang，重生态兼容/成熟度选 vLLM。**

---

# 附：组件速查表

| 组件 | 文件（sglang 仓库内） | 作用 |
|---|---|---|
| RadixAttention 层 | `python/sglang/srt/layers/radix_attention.py` | 注意力层实现，按 forward mode 分发到后端 |
| RadixCache（前缀树） | `python/sglang/srt/mem_cache/radix_cache.py` | token 级前缀树 KV 缓存：match/insert/evict |
| 调度策略 | `python/sglang/srt/managers/schedule_policy.py` | LPM/DFS-weight/FCFS/priority 排序 |
| 调度器 | `python/sglang/srt/managers/scheduler.py` | event_loop、准入、chunked prefill |
| 批结构 | `python/sglang/srt/managers/schedule_batch.py` | ScheduleBatch / ForwardBatch |
| 前端 | `python/sglang/srt/managers/tokenizer_manager.py` | 异步 tokenize、输入规范化、分发 |
| DP 控制 | `python/sglang/srt/managers/data_parallel_controller.py` | 负载感知分发、弹性扩缩容 |
| 执行 worker | `python/sglang/srt/managers/tp_worker.py` | forward 执行、权重热更新 |
| 模型执行 | `python/sglang/srt/model_executor/model_runner.py` | CUDA Graph、KV 池、attention 后端 |
| 分布式通信 | `python/sglang/srt/distributed/communication_op.py` | TP/EP 组感知 all-reduce/all-gather |
| 结构化输出 | `python/sglang/srt/constrained/` | xgrammar/outlines/llguidance 三后端 |
| 投机解码 | `python/sglang/srt/speculative/` | EAGLE/n-gram/自适应（详见 Speculative-Decoding.md） |
| 关键启动参数 | `python/sglang/srt/server_args.py` | `--tp-size --dp-size --chunked-prefill-size --schedule-policy --radix-eviction-policy --speculative-algorithm` |

**关键参数速查**：`--schedule-policy lpm|dfs-weight|fcfs|random`（默认 lpm 或 dfs-weight，取决于负载）、`--chunked-prefill-size 4096`（-1 关闭）、`--radix-eviction-policy lru|lfu|slru|priority`、`--radix-page-size 1`（1 = token 级共享）、`--speculative-algorithm EAGLE|NGRAM`、`--speculative-draft-model <path>`、`--speculative-num-steps 3`。
