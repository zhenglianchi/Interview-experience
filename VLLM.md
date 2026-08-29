vLLM 应该建立一个完整的 **“请求 → 调度 → KV Cache → Attention → 执行 → 通信 → 性能优化”** 的知识框架。

---

# 一、第一优先级：vLLM 最核心的技术点

## 1. PagedAttention
### 1. 现有问题：为什么要提出 PagedAttention

PagedAttention 的核心出发点是：**LLM 推理中的 KV Cache 是动态增长的，而传统的连续显存分配方式会造成严重的显存浪费和碎片问题，进而限制系统能够同时服务的请求数量。**

在 Transformer 的 autoregressive decoding 中，每生成一个新 token，都需要保存这个 token 在每一层的 Key 和 Value，因此一个请求的 KV Cache 会随着序列长度不断增长。假设一个请求一开始不知道最终会生成多少 token，如果系统按照传统方式分配连续显存，就有两个选择：要么提前按照最大可能长度预分配一大块连续 KV Cache，这会导致大量内部碎片；要么在 KV Cache 增长时不断申请更大的连续显存并搬迁已有数据，这又会产生较高的分配和复制成本，并导致外部碎片。更严重的是，在实际 serving 场景中，Continuous Batching 会让不同请求不断进入和退出系统，每个请求的长度都不同，因此 GPU 显存中会出现大量大小不一、生命周期不同的 KV Cache 空洞。

例如，有三个请求分别需要 20、100 和 50 个 token，但为了支持最长可能生成长度，传统系统可能为每个请求预留 128 token 的连续空间，那么前两个短请求就浪费了大量空间；而如果完全按需增长，随着请求不断结束和新请求进入，显存可能变成类似“这里空 16 token、那里空 32 token、这里空 8 token”的碎片化状态。此时即使 GPU 总剩余显存很多，也可能无法找到一块足够大的连续空间来容纳新的长请求。这实际上和操作系统的内存管理问题非常类似。

因此，vLLM 的一个关键观察是：**对于 Attention 计算来说，一个 sequence 的 KV Cache 并不一定必须物理连续。真正需要连续的是逻辑上的 token 顺序，而不是 GPU 物理显存地址。** 既然如此，就可以像操作系统虚拟内存一样，把 KV Cache 切成固定大小的 block，让 sequence 在逻辑上连续，但在物理显存中可以分散存储。这就是 PagedAttention 的基本思想。

---

### 2. 方法论：PagedAttention 是怎么实现的

PagedAttention 将每个 sequence 的 KV Cache 划分为固定大小的 **KV blocks**。假设 block size 是 $B$ 个 token，那么长度为 $L$ 的 sequence 不再需要申请一整块大小为 $L$ 的连续显存，而只需要：

$$
\left\lceil \frac{L}{B} \right\rceil
$$

个固定大小的 block。这些 block 可以来自 GPU 显存中的任意位置。

例如，一个 sequence 逻辑上需要的 KV Cache 是：

```text
Token:  [0 1 2 3 | 4 5 6 7 | 8 9 10 11]
Logical:    Block 0       Block 1        Block 2
```

在传统方式中，它们可能必须在 GPU 显存中：

```text
Physical Memory:
[连续的一大片 KV Cache 空间]
```

而在 PagedAttention 中，逻辑 block 可以映射到任意物理 block：

```text
Logical Block 0 → Physical Block 7
Logical Block 1 → Physical Block 2
Logical Block 2 → Physical Block 19
```

因此，每个 sequence 会维护一个 **Block Table**：

$$
\text{BlockTable} = [7, 2, 19]
$$

Attention kernel 在访问 token 的 KV Cache 时，首先根据 token position 计算它属于哪个 logical block：

$$
\text{logical\_block\_id}
=
\left\lfloor
\frac{\text{token\_position}}{B}
\right\rfloor
$$

然后通过 Block Table 找到对应的 physical block，再计算该 token 在 block 内部的 offset：

$$
\text{block\_offset}
=
\text{token\_position}
\bmod B
$$

也就是说，访问过程是：

$$
token\ position
\rightarrow
logical\ block
\rightarrow
BlockTable
\rightarrow
physical\ block
\rightarrow
block\ offset
$$

Attention 本身仍然是在逻辑 token 顺序上进行计算。对于当前 token 的 Query：

$$
Q_t
$$

仍然需要计算：

$$
Attention(Q_t,K_{1:t},V_{1:t})
$$

区别只是 $K_{1:t}$ 和 $V_{1:t}$ 在物理显存中不是连续的一整块，而是由多个 blocks 组成。PagedAttention kernel 会根据 Block Table 依次访问这些 physical blocks，因此**逻辑 sequence 是连续的，但物理存储可以是不连续的**。

这种设计带来三个核心收益。第一，KV Cache 可以按 block 增量分配，一个 sequence 生成新 token 时只有跨越新的 block 边界才需要申请新 block；第二，所有 KV blocks 大小相同，显存 allocator 可以像操作系统的页分配器一样统一管理，大幅降低外部碎片；第三，block 可以被多个 sequence 共享，这为 prefix caching 和 Copy-on-Write 提供了基础。例如两个请求有相同的 system prompt，它们可以直接指向同一批 physical KV blocks，而不需要复制。

---

### 3. 具体数值样例

假设我们有一个 GPU KV Cache block pool，一共有 10 个 physical blocks，每个 block 可以存储 **4 个 token** 的 KV Cache：

```text
Physical Blocks:

Block 0
Block 1
Block 2
Block 3
Block 4
Block 5
Block 6
Block 7
Block 8
Block 9
```

现在来了 Request A，它的 prompt 有 6 个 token：

```text
A:
[a0 a1 a2 a3 a4 a5]
```

因为每个 block 存 4 个 token，所以 Request A 需要：

$$
\lceil 6/4 \rceil = 2
$$

个 logical blocks。假设 allocator 分配给它：

```text
Logical Block 0 → Physical Block 3
Logical Block 1 → Physical Block 7
```

于是 A 的 Block Table 是：

```text
A Block Table = [3, 7]
```

物理存储实际上是：

```text
Physical Block 3:
[a0 a1 a2 a3]

Physical Block 7:
[a4 a5 _ _]
```

虽然 token `a0-a5` 在逻辑上是连续的，但在物理 GPU memory 中，Block 3 和 Block 7 中间可能隔着大量其他数据。

此时 A 继续生成两个 token：

```text
[a0 a1 a2 a3 a4 a5 a6 a7]
```

由于 Block 7 还剩两个位置，因此：

```text
Physical Block 7:
[a4 a5 a6 a7]
```

此时不需要申请新的 block。A 再生成 token `a8` 时，当前长度变成 9：

```text
[a0 a1 a2 a3 a4 a5 a6 a7 a8]
```

现在需要：

$$
\lceil 9/4 \rceil = 3
$$

个 block，因此 allocator 只需要从 free block pool 中拿一个，例如 Physical Block 1：

```text
Logical Block 0 → Physical Block 3
Logical Block 1 → Physical Block 7
Logical Block 2 → Physical Block 1
```

新的 Block Table：

```text
A Block Table = [3, 7, 1]
```

现在假设另一个 Request B 到来，它的前 6 个 token 与 A 完全相同：

```text
B:
[a0 a1 a2 a3 a4 a5 b6]
```

如果使用 Prefix Caching，B 不需要重新计算：

```text
[a0 a1 a2 a3 a4 a5]
```

对应的 KV Cache。理论上它可以共享 A 已经计算好的 block。但是注意，B 的第一个 block：

```text
[a0 a1 a2 a3]
```

是完整 block，因此可以直接共享：

```text
B Logical Block 0 → Physical Block 3
```

而第二个 block 如果共享策略要求完整 block，则需要根据当前实现策略处理未满 block。对于一个已经完整的 prefix block，共享非常直接；之后 B 继续生成不同 token 时，再申请自己的 block 或通过 Copy-on-Write 避免修改共享数据。

最后，如果 A 完成：

```text
A finished
```

系统并不需要释放一块连续的大显存，只需要遍历：

```text
A Block Table = [3, 7, 1]
```

并释放其中不再被其他 request 引用的 physical blocks。例如 Block 3 仍被 B 共享，那么它的引用计数不会归零，不能释放；而 Block 7 和 Block 1 如果没有其他引用，则直接返回 free block pool：

```text
Free Block Pool:
[0, 2, 4, 5, 6, 7, 8, 9, 1]
```

新的请求可以立即复用这些 block。

**面试时，你可以把 PagedAttention 的核心总结成一句话：**

> PagedAttention 并没有改变 Attention 的数学定义，它改变的是 KV Cache 的物理组织方式：将逻辑连续的 KV Cache 切分为固定大小的 blocks，并通过 Block Table 将 logical blocks 映射到任意 physical blocks，从而实现按需分配、降低显存碎片、提高 KV Cache 利用率，并为 continuous batching 和 prefix sharing 提供基础。

### 4. 核心代码（源码）

**Block Table 的代码实现在 `vllm/v1/worker/block_table.py` 的 `BlockTable`**——它就是"logical block → physical block 映射"这个核心数据结构的代码形态：

```python
class BlockTable:
    """映射一个请求的 KV blocks：逻辑 block → 物理 block（PagedAttention 的核心）。"""
    def __init__(self, block_size: int, block_ids: list[int] | None = None):
        self._block_size = block_size
        self._block_ids: list[int] = list(block_ids or [])   # 每个逻辑位置对应的物理 block

    def append_row(self, block_id: int, row_idx: int) -> None:
        """decode 增长：跨 block 边界时追加一个物理 block（按需分配）。"""
        ...

    def compute_slot_mapping(self, ...):
        """把 token 位置映射到 (physical_block, offset) 的 slot —— 
        供 attention kernel 按 BlockTable 间接寻址。"""
        ...

    def map_to_kernel_blocks(self, ...):
        """把 BlockTable 转成 kernel 需要的块指针（分页寻址）。"""
        ...
```

**这段代码关键在哪**：`BlockTable` 就是 PagedAttention 的代码落点——`_block_ids` 列表就是 `BlockTable = [7, 2, 19]` 的真实形态（每个元素是物理 block id）；`append_row` 对应"decode 跨 block 边界时按需分配"；`compute_slot_mapping` / `map_to_kernel_blocks` 对应"token position → BlockTable → physical block → offset"的寻址链路，attention kernel（FlashAttention/FlashInfer 的 paged 实现）靠它间接访问 KV。**v0.25.0 移除的是旧版 monolithic kernel，`BlockTable` 这个分页数据结构在 MRv2 里继续存在**——这就是"分页内存模型保留"的代码证据。

### 5. 新版 vLLM 特性（v0.25.x 演进）

**注意一个重要的版本事实：vLLM 在 v0.25.0（不是 v0.25.1）的 release notes 里正式宣布 "PagedAttention has been removed"（PR #47361），但这句"移除"需要精确理解，否则面试会踩坑。** 实际上，PagedAttention 这个词长期以来被用来指代三件不同的事：第一是**一个具体的 kernel**——2023 年那版读取 Block Table、对非连续 KV blocks 计算 attention 的 monolithic CUDA 实现；第二是**一个抽象**——通过间接层把 KV cache 暴露给 attention 操作（逻辑位置 → 物理 block 的映射）；第三是**一个内存模型**——把 KV cache 管理成固定大小的 block、按需分配、可分散在显存任意位置（源自 SOSP 2023 论文，借鉴操作系统虚拟内存）。v0.25.0 移除的是**第一件：legacy kernel 实现**——因为当时 FlashAttention 系和 FlashInfer 的 paged kernels 已经是默认路径，旧的 monolithic 代码路径没人再走，于是被删除（release notes 原文："The legacy attention implementation is deleted now that V1/MRv2 backends are the standard path"）。

**关键结论是：分页 KV 内存模型（第三件）被完整保留**，只是实现从 2023 年的自研 kernel 迁移到了现代 paged attention 后端（FlashAttention 系 / FlashInfer 的 paged kernels，配合 v0.25.0 成为默认的 Model Runner V2 架构）。所以面试时说"vLLM 用分页管理 KV cache（PagedAttention 思想）"依然完全正确；但如果说"vLLM 在跑 2023 年的 PagedAttention kernel"就不准确了。另一个值得记的数字：PagedAttention 论文（SOSP 2023）声称的相对收益是**相比 FasterTransformer、Orca 等 prior serving systems 提升 2~4 倍吞吐**（在相同延迟下），而 launch 博客常引用的"up to 24x"是对 naive HuggingFace Transformers serving 这个极弱基线的对比，两者量纲不同——面试引用时用 2~4x 更严谨。这体现了一个通用规律：**核心思想（分页内存模型）长期留存，具体 kernel 实现随硬件与后端演进不断更替**。

---

## 2. Continuous Batching / Continuous Scheduling

### 1. 现有问题：为什么要提出 Continuous Batching

Continuous Batching 要解决的核心问题是：**在服务端推理中，静态批处理（Static Batching）会导致 GPU 利用率极低和大量不必要的等待。** 在没有 Continuous Batching 之前，主流做法是 request-level batching（也叫 naive batching / static batching）：系统把一段时间内到达的请求攒成一个 batch，整个 batch 一起做 prefill、一起逐 token 做 decode，直到 batch 里所有请求都生成完毕才整体释放。这个模式的致命缺陷在于，不同请求的生成长度差异巨大：有的请求只需要生成 20 个 token，有的要生成 500 个 token，但静态 batching 要求**整个 batch 一起前进、一起结束**，所以短请求生成的 20 个 token 早就完成了，却要白白等着长请求把剩下的 480 个 token 生成完才能释放自己占用的 GPU 资源。

这种"木桶效应"造成的浪费是双重的。第一重是算力浪费：batch 中大量 slot 已经空闲（请求已完成）但无法被释放，GPU 实际上只在为少数几个长请求工作，有效吞吐远低于 batch size 对应的理论值。第二重是延迟浪费：新到达的请求必须等到整个 batch 全部结束后才能进入下一批，即使 GPU 明明有空闲 slot 也无法立即服务它，造成请求排队时间大幅拉长。举例来说，如果一个 batch 里有 8 个请求，其中 7 个只需生成 50 个 token，1 个需要生成 500 个 token，那么在 500 步 decode 中，前 50 步 GPU 在为 8 个请求工作，后 450 步实际上只有 1 个请求还在生成——GPU 的有效利用率大约只有 (8×50 + 1×450) / (8×500) ≈ 21%，剩下的 79% 都被白白浪费了。

### 2. 方法论：Continuous Batching 是怎么实现的

Continuous Batching 的核心思想是**把调度粒度从"请求级别"细化到"迭代级别"（iteration-level scheduling）**：不再要求一个 batch 同时开始、同时结束，而是让 batch 变成一个动态变化的集合——在**每一个 decode 迭代（即每生成一个 token 的 step）结束时都重新做一次调度决策**。系统维护三个队列作为核心数据结构：**running**（当前正在 GPU 上执行、每步生成一个 token 的序列集合）、**waiting**（已到达但还没开始 prefill 的请求队列）、**swapped**（被抢占换出到 CPU 显存、等待换回的序列集合）。

每一轮调度循环按以下步骤操作：

1. **扫描 running 队列**：逐个检查每个序列是否已生成完毕（生成了 EOS token 或达到 max_tokens）。若完成，将该序列从 running 中移除，并遍历它的 Block Table 逐个释放 KV block（见第 3 点的引用计数逻辑），空出的 slot 立即可以被复用；
2. **检查 KV Cache 水位**：计算当前空闲的 KV block 数量，判断能否容纳新的 prefill 请求；
3. **补入新请求**：若 waiting 队列非空且空闲 block 足够，从队首取出请求，分配 KV block、执行 prefill，将序列加入 running；
4. **处理资源不足**：若空闲 block 不足以支持新请求，触发抢占（见第 9 点），把 running 中部分序列 swap 到 CPU 或标记 recompute，腾出空间；
5. **执行本 step 的 GPU 计算**：对当前 running 集合做一次批量 forward，每个序列各生成一个 token，然后回到第 1 步进入下一轮循环。

也就是说，batch 的组成是"每步都在变"的：每生成一个 token 之后，完成的请求退出、等待的请求进入。需要强调的是，Continuous Batching 能高效工作的前提是 PagedAttention：因为 batch 里的请求长度各不相同、动态进出，每个请求的 KV Cache 大小都在变化，只有基于 block 的分页式显存管理才能让"随时移除旧请求、插入新请求"变成低成本的 block 表增删操作（分配/释放的粒度是固定大小的 block），而不会产生大块连续显存的分配释放和碎片问题。此外，现代实现还会结合 **Chunked Prefill**（见第 7 点）把长 prompt 的 prefill 切碎，插在 decode step 之间执行，进一步避免"一个大 prefill 卡住整个 batch"的问题。这个"每步一调度"的机制在 vLLM 中由 Scheduler 组件实现，调度决策（schedule）与 GPU 执行（execute）分离，形成"决策-执行"循环。

### 3. 具体数值样例

假设 GPU 的 batch size 上限是 4（即最多同时 4 个序列），KV Cache 足够充裕（暂不考虑显存瓶颈），某一时刻有 4 个请求 A、B、C、D 同时进入，它们需要生成的 token 数分别是 10、100、50、80。先看静态 batching：整个 batch 一起前进，必须等最长的 B（100 token）生成完毕才整体释放，总共需要 100 个 decode step。在这 100 步中，A 在第 10 步就完成了，之后 90 步它占用的 slot 完全空转；C 在第 50 步完成，之后 50 步空转；D 在第 80 步完成，之后 20 步空转。GPU 的总"有效工作量"是 10+100+50+80 = 240 个 token-step，而 GPU 实际付出的算力是 4 slots × 100 步 = 400 个 slot-step，利用率只有 240/400 = 60%。而且这期间如果有新请求 E 到达，E 必须干等 100 步直到这一批结束才能开始，白白增加了排队延迟。

再看 Continuous Batching，逐 step 演算 batch 的组成变化（用 [x] 表示 running 中的序列）：

```text
Step 1~10:   running = [A, B, C, D]        # 4 个一起 decode
Step 11:     A 生成满 10 token → 完成，移出 running，释放 A 的 KV block
             waiting 里有新请求 E（需生成 30 token）→ 立即 prefill 补入
             running = [B, C, D, E]        # batch 保持满 4 个
Step 11~50:  running = [B, C, D, E]
Step 51:     C 生成满 50 token → 完成移出；F（需 40 token）prefill 补入
             running = [B, D, E, F]
Step 51~80:  running = [B, D, E, F]
Step 81:     D 生成满 80 token → 完成移出；G（需 30 token）prefill 补入
             running = [B, E, F, G]
Step 81~100: running = [B, E, F, G]        # 注意此时 E(30) 已在 step 40 完成
```

需要修正上面的推演：E 只需要 30 token，它在第 11 步加入，到第 40 步（11+30−1）就完成了，所以第 41 步 E 也会被移出并补入新请求。我们把请求的到达时序设计得更清楚一些：假设 E 在第 11 步到达（需 30 token）、F 在第 41 步到达（需 40 token）、G 在第 81 步到达（需 30 token）。那么完整的 running 变化是：

```text
Step 1~10:   running = [A, B, C, D]
Step 11:     A 完成 → E 补入 → running = [B, C, D, E]
Step 41:     E 完成 → F 补入 → running = [B, C, D, F]
Step 51:     C 完成 → 此时无新请求 → running = [B, D, F]   # 3 个，暂不满
Step 81:     D 完成 → G 补入 → running = [B, F, G]
Step 91:     F 完成 → running = [B, G]
Step 100:    B 完成 → running = [G]
```

对比结果：在同样的 100 步时间内，静态 batching 只服务了 4 个请求（A、B、C、D），而 Continuous Batching 服务了 7 个请求（A、B、C、D、E、F、G），吞吐提升约 75%；更重要的是，每个请求从到达开始算的排队延迟大幅下降——E 在第 11 步到达后**立即**开始生成，而不是像静态 batching 那样干等 100 步。即使某几步 batch 不满（如第 51~81 步只有 3 个），整体利用率也远高于静态方案，且可以随时有请求到达随时插入，这正是现代推理引擎（vLLM、TensorRT-LLM、SGLang 等）都把 Continuous Batching 作为最基本的调度策略的原因。

> **面试一句话总结**：Continuous Batching 把批处理调度粒度从"整批同生共死"细化为"每个 decode 迭代动态调度"，通过 running / waiting / swapped 三队列每步一调度，请求完成立即移出、新请求立即补入，让 GPU 的 slot 始终保持满载，显著提升吞吐并降低排队延迟，其高效运行依赖 PagedAttention 的分页显存管理。

### 4. 核心代码（源码）

**当前 vLLM（main 分支，MRv2 架构）的 Continuous Batching 实现在 `vllm/v1/core/sched/scheduler.py` 的 `Scheduler.schedule()`**。新版 Scheduler 的一个关键设计是**没有显式的 "prefill 阶段 / decode 阶段" 之分**——每个请求只有 `num_computed_tokens`（已算的 token 数）和 `num_tokens_with_spec`（应算的 token 数），调度器每步给请求分配 token 让前者追上后者，从而统一覆盖 chunked prefill、prefix caching、投机解码：

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
    self.current_step += 1
    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and num_tokens_with_spec.
    # num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids)
    #                        + len(spec_token_ids).
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its
    # num_tokens_with_spec. This is general enough to cover
    # chunked prefills, prefix caching, speculative decoding,
    # and the "jump decoding" optimization in the future.

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []
    token_budget = self.max_num_scheduled_tokens      # 每步 token 预算
    input_budget = self.scheduler_config.max_num_batched_tokens

    # First, schedule the RUNNING requests.   ← 先保 running（decode 优先）
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        ...
        num_new_tokens = (request.num_tokens_with_spec
                          + request.num_output_placeholders
                          - request.num_computed_tokens)   # 还需算多少 token
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
        # 记录 num_scheduled_tokens / req_to_new_blocks，交给 worker 执行
        ...
    # 然后处理 waiting（新请求 prefill / chunked prefill）……
```

**这段代码关键在哪**：`schedule()` 是"每步一调度"的代码落点——①`token_budget`（每步总 token 预算）和 `input_budget`（`max_num_batched_tokens`）是全局硬约束；②`num_new_tokens = num_tokens_with_spec - num_computed_tokens` 统一了 prefill 和 decode 的调度（不需要区分阶段）；③先遍历 `self.running`（正在 decode 的请求优先），再处理 waiting——这正是"先保 running 再保新请求"的优先级体现；④`long_prefill_token_threshold` 把长 prefill 切成 chunk（Chunked Prefill 的代码入口）。**这就是 Continuous Batching 在 vLLM 里的真实实现**——不是概念，而是一个每次被调用的 `schedule()` 方法。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Continuous Batching 在 v0.25.x 中依然是 vLLM 调度的绝对核心路径，没有根本性变更——但它的**承载架构变了**：v0.25.0 起 Model Runner V2（MRv2）成为 dense 模型的默认执行路径，调度循环（Scheduler 的 running / waiting / swapped 三队列、每步一调度）的逻辑保持不变，但每轮的"执行计划"由 MRv2 以更模块化、GPU-native、async-first 的方式消费（详见第 13 点的新版特性）。另一个演进方向是**调度策略的插件化**：vLLM 较新版本把抢占策略、队列策略抽象成独立的 Policy 对象（如按优先级、按最长等待时间等），让 Continuous Batching 的"每步决策"可以针对不同业务定制，而不是写死一种行为。面试时可以提一句："Continuous Batching 的迭代级调度思想没有变，变的是它在新版 MRv2 架构下的实现与可定制策略。"

---

## 3. KV Cache Management

### 1. 现有问题：为什么要做 KV Cache Management

KV Cache 是 LLM 推理中**占用显存最大的动态资源**，管理不好会直接限制系统能服务的并发请求数，甚至导致 OOM。在 Transformer 的 autoregressive 解码中，每个请求每生成一个 token，都要把该 token 在每一层、每一个注意力头对应的 Key 和 Value 张量保存下来，供后续 token 做注意力计算使用。这个缓存的大小非常可观：以一个 7B 参数的模型为例（假设 32 层、32 个 KV 头、每头维度 128、FP16 存储），每个 token 的 KV Cache 大小可以精确算出来：32 层 × 32 头 × 128 维 × 2 字节 × 2（K 和 V）= 512 KB。也就是说**一个 4096 token 的序列光 KV Cache 就要占 2 GB 显存**，而模型权重本身（FP16）也只有 14 GB 左右；如果同时服务几十个并发请求、每个请求又有很长的上下文，KV Cache 的总量会轻松超过模型权重，成为显存消耗的绝对大头。

KV Cache 管理的难点在于它的**动态性和碎片化**。一方面，每个请求的 KV Cache 随着生成而不断增长，而且增长量事先未知——系统不知道这个请求最终会生成多少 token，也就没法"一次分配到位"；另一方面，在 Continuous Batching 下，请求随时完成、随时释放，不同请求长度差异巨大，如果按传统方式为每个请求分配连续的一大块显存，那么随着请求不断进出，显存中会出现大量大小不一的空洞，最终即使总空闲显存充足，也可能找不到一块足够大的连续空间容纳新请求——这正是 PagedAttention 要解决的碎片化问题（见第 1 点）。此外，KV Cache 还涉及**共享**（多个请求共享同一个前缀，如相同的 system prompt）和**换入换出**（显存不够时把部分请求的 KV Cache 挪到 CPU 内存）等高级能力，这些都需要一个专门的管理器来统筹。

### 2. 方法论：vLLM 的 KV Cache Manager 是怎么实现的

vLLM 的 KV Cache Management 建立在 PagedAttention 的 block 机制之上，由 **KV Cache Manager** 统一管理。系统在启动时根据 `gpu_memory_utilization` 配置预留出一块显存作为 **KV Cache pool**，把它切分成固定大小的 physical blocks（默认 block size 通常为 16 个 token），之后所有请求的 KV Cache 都从这块 pool 中按 block 粒度分配和释放。核心数据结构有三个：**BlockTable**（每个 sequence 一张，记录它的 logical block 依次映射到哪些 physical block）、**free list**（空闲 physical block 链表）、**refcount 表**（每个 physical block 当前被多少个 sequence 引用）。

分配和释放的具体操作流程如下。**分配**：当一个请求做 prefill 时，按它的 prompt 长度计算需要多少个 block（⌈L/16⌉），从 free list 头部依次取出相应数量的 physical block，写入该 sequence 的 BlockTable，并把每个 block 的 refcount 置 1；decode 过程中每跨过一个 block 边界（每生成满 16 个新 token），再向 free list 取一个新 block 追加到 BlockTable 末尾。**释放**：当一个 sequence 完成或被 abort 时，遍历它的 BlockTable，对每个 physical block 的 refcount 减 1；只有当 refcount 归零时才把该 block 归还 free list——这意味着被其他 sequence 共享的 block 不会被提前释放。**共享**：当两个 sequence 有相同前缀时（Prefix Caching，见第 4 点），后到的 sequence 的 BlockTable 前若干项直接指向与前者相同的 physical block，这些 block 的 refcount 累加；若之后需要修改某个共享 block 的内容（如 beam search 中两个分支生成不同 token），则触发 **Copy-on-Write（CoW）**：从 free list 拿一个新 block、把原 block 内容拷过去再修改，保证不破坏共享者。**换入换出（swap）**：当显存中的 KV Cache 达到水位上限（watermark）时，Scheduler 选择部分请求，把它们的 block 整体拷贝到 CPU 内存（swap out），腾出显存给新请求；等显存有空闲时再把它们换回来（swap in），代价是 CPU↔GPU 的 PCIe 拷贝开销，所以 swap 通常只作为 OOM 前的兜底手段。

这套设计的关键收益是：分配/释放的粒度是固定大小的 block，操作是 O(1) 级别的链表增删，不会产生大块显存拷贝和碎片整理；refcount 机制让前缀共享成为可能，多个请求共用一份 KV 显存；watermark 水位 + swap 机制让系统在显存紧张时优雅降级而不是直接 OOM。vLLM 的新版本还引入 BlockManager V2（prefix caching aware），把"相同内容的 block 只存一份"从"按前缀共享"推广为"按内容哈希共享"，与第 4 点 Prefix Caching 直接配合。

### 3. 具体数值样例

假设一张 80 GB 显存的 A100 上部署一个 7B 模型，权重 FP16 占 14 GB，激活值和 CUDA context 等留 6 GB，`gpu_memory_utilization` 设为 0.9，那么 KV Cache pool 大约有 80 × 0.9 − 14 − 6 ≈ 52 GB。已知该模型每个 token 的 KV Cache 是 512 KB，block size 取 16，那么每个 block 占 16 × 512 KB = 8 MB，整个 KV Cache pool 可以容纳 52 GB / 8 MB = 6656 个 physical block，对应总容量约 106,496 个 token 的 KV Cache。

现在逐步演算三个请求的分配、共享、释放全过程。假设 free list 初始为 [0, 1, 2, ..., 6655]，refcount 全为 0。

**第一步：Request A 做 prefill**，prompt 100 token。需要 ⌈100/16⌉ = 7 个 block。allocator 从 free list 头部取 7 个：physical block 0~6，写入 A 的 BlockTable：

```text
A BlockTable = [0, 1, 2, 3, 4, 5, 6]
refcount: block 0~6 = 1
free list: [7, 8, 9, ...]
```

**第二步：Request B 做 prefill**，prompt 1000 token，需要 ⌈1000/16⌉ = 63 个 block。allocator 取 free list 头部的 7~69：

```text
B BlockTable = [7, 8, 9, ..., 69]
refcount: block 7~69 = 1
free list: [70, 71, 72, ...]
```

**第三步：Request C 做 prefill**，prompt 100 token，且**前 96 个 token 与 A 完全相同**（96 = 6 个完整 block 的内容相同，最后 4 个 token 不同）。C 到达时，Prefix Caching 发现它的前 6 个 block 与 A 的 block 0~5 内容一致，于是 C 的 BlockTable 前 6 项直接复用 A 的 physical block：

```text
C BlockTable = [0, 1, 2, 3, 4, 5, 70]   # 前 6 项共享 A，最后一项是新 block 70
refcount: block 0~5 = 2（A 和 C 共享）, block 6 = 1, block 70 = 1
free list: [71, 72, ...]
```

此时显存占用：A 7 + B 63 + C 7 = 77 个 block 的逻辑需求，实际只分配了 7 + 63 + 1 = 71 个物理 block，省下 6 个（48 MB），同时省去了重复计算这 96 个 token 的 prefill 算力。

**第四步：A 完成**。遍历 A 的 BlockTable [0..6]，逐个 refcount 减 1：block 0~5 从 2 变 1（C 还在引用，**不能**释放），block 6 从 1 变 0（归还 free list）：

```text
refcount: block 0~5 = 1（C 持有）, block 6 = 0 → 回到 free list
free list: [6, 71, 72, ...]
```

**第五步：C 继续生成**，从第 96 token 之后不断增长。当 C 的长度从 100 token 增长到 112 token 时，跨过第 2 个 block 边界，需要从 free list 取一个新 block 追加到 C 的 BlockTable：

```text
C BlockTable = [0, 1, 2, 3, 4, 5, 70, 6]   # 追加的 block 6 正好是刚释放的
refcount: block 6 = 1（C 重新持有）
free list: [71, 72, ...]
```

**第六步：若 C 需要修改共享 block 的内容**（比如 C 的前缀与 A 相同但之后要改写第 2 个 block 里的 token），触发 Copy-on-Write：allocator 从 free list 取 block 71，把 block 1 的内容拷过去，C 的 BlockTable 第 2 项改为 71，block 1 的 refcount 减回 1（只剩 A 的逻辑引用，但 A 已结束则归 0 释放）：

```text
C BlockTable = [0, 71, 2, 3, 4, 5, 70, 6]
refcount: block 0=1, block 1→0(释放), block 71=1, block 2~5=1, block 70=1, block 6=1
```

整个过程中，所有操作都是"从 free list 拿/还固定大小的 block + 改 refcount"，没有一次大块显存的分配或搬运，也没有碎片——这就是 KV Cache Manager 以 block 为粒度管理的核心价值。

> **面试一句话总结**：KV Cache Management 以 PagedAttention 的 block 为粒度，通过统一预留的显存池 + BlockTable + free list + 引用计数 + Copy-on-Write + 换入换出，实现对动态增长的 KV Cache 的高效分配、共享与回收，避免碎片化并最大化显存利用率。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 KV Cache 管理在 `vllm/v1/core/kv_cache_manager.py` 的 `KVCacheManager`**——它把"block 分配/释放/前缀命中/换出"全部封装成接口，Scheduler 每步调用它：

```python
class KVCacheManager:
    """Manages the KV cache blocks for requests."""

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        注意：computed blocks 必须是"完整 block"（full）。
        """
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0   # prefix caching 关闭

        # 全部 token 都命中缓存时，必须重算最后一个 token 拿 logits，
        # 所以 max_cache_hit_length = prompt_length - 1
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length   # 按 block 哈希找最长前缀
            )
        )
        return computed_blocks, num_new_computed_tokens, num_uncached

    def allocate_slots(self, request, num_new_tokens, ...) -> KVCacheBlocks | None:
        """为请求的新 token 分配 KV slots（block）。
        num_lookahead_tokens 用于投机解码（eagle 草稿的 KV 预留）。
        """
        ...

    def free(self, request: Request) -> None:
        """请求结束/被抢占时释放其 KV blocks。"""
        ...

    def evict_blocks(self, block_ids: set[int]) -> None:
        """prefix cache 淘汰：把不常命中的 block 逐出缓存。"""
        ...
```

**这段代码关键在哪**：`KVCacheManager` 接口完整对应"KV Cache Management"的四个职责——①`get_computed_blocks` 是 **Prefix Caching 的代码入口**（`find_longest_cache_hit` 按 block 哈希找最长共享前缀，返回"已算好的 blocks"）；②`allocate_slots` 是**按需分配**（`num_lookahead_tokens` 参数对应投机解码的 KV 预留）；③`free` 是**释放**；④`evict_blocks` 是**缓存淘汰**。新版把 block 管理和 prefix cache 整合进一个类，比旧版 `BlockManager` 更统一——但"block 粒度 + 哈希共享 + 按需分配"的思想完全一致。

### 5. 新版 vLLM 特性（v0.25.x 演进）

KV Cache Management 是"分页内存模型被保留、实现被更替"的最典型例子（见第 1 点的新版特性）：v0.25.0 移除的是 2023 年的 legacy PagedAttention kernel，而 **block 级分页 KV 分配、BlockTable 映射、引用计数、Copy-on-Write 这些内存管理机制全部保留**，只是 KV 的实际分配与 attention 执行迁移到了现代 paged attention 后端（FlashAttention 系 / FlashInfer 的 paged kernels）之上。v0.25.x 还带来两个与 KV Cache 直接相关的增强：一是 **FP8 KVCache** 支持（KV 张量以 FP8 存储，KV 显存占用减半，配合 FP8 权重进一步降低显存压力，属于"KV Cache 压缩"方向的一个落地）；二是 **Mamba hybrid 模型的 prefix caching**（混合架构中线性注意力部分也能复用前缀缓存）。面试时可以强调："KV 分页管理的核心机制在新版没有消失，只是底层 kernel 换成了更现代的 paged attention 后端，并叠加了 FP8 KV 量化这类压缩手段。"

---

## 4. Prefix Caching / Automatic Prefix Caching

### 1. 现有问题：为什么要提出 Prefix Caching

Prefix Caching 要解决的问题是：**大量请求共享相同的前缀（system prompt、few-shot 示例、多轮对话的历史），却在每次请求时都重复计算这些前缀的 KV Cache，造成大量算力浪费。** 在实际的 LLM 服务场景中，请求的 prompt 往往由两部分组成：一段固定不变的前缀 + 一段变化的用户输入。典型例子包括：对话式应用里每个请求都带着相同的 system prompt 和全部历史消息；RAG 应用里每个请求都带着相同的检索指令和文档库内容；评测和批处理任务里所有样本共享同一套任务描述和 few-shot 示例。这些前缀动辄几百到几千个 token，而它们在**每次请求之间内容完全不变**。

问题在于，标准的 prefill 阶段会为每个请求从零开始计算所有 prompt token 的 KV Cache。也就是说，如果 100 个请求共享一段 2000 token 的 system prompt，系统就要重复计算 100 × 2000 = 20 万 token 的 KV Cache——这些计算的结果逐位相同，纯属浪费。prefill 是计算密集型的（一次要并行处理整个前缀），在长前缀、高并发的场景下，重复计算前缀可能占据总计算量的 30%~50% 甚至更多，直接拖累吞吐和响应延迟。更隐蔽的问题是，即使系统"记得"上次算过这段前缀，不同请求的 prompt 在 token 级别上可能有细微差异（比如末尾多了一个空格或换行），导致前缀不完全一致，简单的"整串匹配"方案命中率很低。因此需要一种机制，能够在**任意长度的前缀**上做细粒度的复用——这就是 Automatic Prefix Caching（APC）。

### 2. 方法论：vLLM 的 Automatic Prefix Caching 是怎么实现的

vLLM 的 Automatic Prefix Caching 建立在 PagedAttention 的 block 机制之上，核心思想是：**以 block 为粒度缓存 KV Cache，凡是内容与历史完全相同的 block 直接复用，只有不同的部分才重新计算。** 具体实现分为"缓存写入"和"缓存匹配"两条路径。

**缓存写入路径**（一个请求做完 prefill 后）：假设 block size = 16。第 1 步，把该请求 prompt 中"已经写满 16 个 token"的每个 block，对其内的 token 序列计算一个哈希值（vLLM 用 xxhash 之类的高性能哈希，例如对 block 内 16 个 token id 拼接后哈希，得到 64 位哈希值）；第 2 步，以"哈希值 → physical block id"的形式写入一个全局哈希表（hash table），这个哈希表就是"内容 → 显存位置"的索引。注意：**最后一个不满 16 个 token 的 block 不写入缓存**——因为它内容未定，之后还会追加新 token，共享它会有风险。

**缓存匹配路径**（一个新请求到达做 prefill 时）：第 1 步，把新请求的 prompt 按 block size 切分，从第 1 个 block 开始计算哈希；第 2 步，在全局哈希表中查找：若命中，则让新请求的 BlockTable 的这一项直接指向缓存中的 physical block，并把该 block 的 refcount +1；第 3 步，继续对下一个 block 做同样操作，直到某个 block 的哈希**未命中**为止——这个未命中点之前的所有前缀全部复用，不需要计算；第 4 步，从未命中点开始，剩余的 prompt 才真正执行 prefill，算出的新 block 走"缓存写入路径"入表。

这套机制的关键设计有三点。第一是**"完整 block 才可共享"原则**：哈希表只收录完整 block，未满 block 不参与匹配也不参与缓存，保证共享的永远是"内容已定型"的数据；第二是**自动性**：用户不需要显式标记哪些 token 是前缀，系统按 block 内容哈希自动发现可复用的前缀，对任意共享前缀的请求组合都有效——这比某些框架要求用户手动传 prefix 参数通用得多；第三是**淘汰与安全**：哈希表容量受显存限制，用类似 LRU 的策略淘汰不常命中的缓存项，被淘汰的 block 若仍有 sequence 引用则不能释放（引用计数兜底，见第 3 点）；同时由于 64 位哈希碰撞概率极低，vLLM 默认信任"哈希相等即内容相等"，不做碰撞检测。此外，命中前缀后若某个序列要修改共享 block 的内容，走 Copy-on-Write 即可（见第 3 点），因此 APC 与 PagedAttention 的 block 共享、beam search 的 CoW 天然配合。

### 3. 具体数值样例

假设 block size = 16，一个对话服务有 3 个并发请求，它们的 prompt 都包含相同的 system prompt + 历史共 **96 个 token**（正好 6 个完整 block），加上各自不同的用户输入：Request A 输入 20 token，B 输入 40 token，C 输入 60 token。先看没有 Prefix Caching：3 个请求都要完整 prefill 自己全部 prompt——A 要算 96+20=116 token，B 要算 96+40=136 token，C 要算 96+60=156 token，**重复计算的前缀部分共 3 × 96 = 288 token**，占全部 prefill 计算量 (116+136+156=408 token) 的约 70%。

现在逐步演算开启 APC 后的过程。

**第一步：Request A 到达（缓存为空）**。A 完整 prefill 116 个 token：前 96 个 token 分成 6 个完整 block（block 内容分别记为 P0~P5，对应 token 0~15、16~31、…、80~95），各自计算哈希并写入全局哈希表；后 20 个 token 不满 16，作为第 7 个 block 不写缓存。

```text
hash table:  { H(P0)→block_A0, H(P1)→block_A1, ..., H(P5)→block_A5 }
A BlockTable = [block_A0, block_A1, ..., block_A5, block_A6(不满)]
refcount: block_A0~A5 = 1
```

**第二步：Request B 到达**。B 的 prompt = 96 token 相同前缀 + 40 token 输入，共 136 token，需要 ⌈136/16⌉ = 9 个 block。逐个哈希 B 的前 6 个 block：

```text
block0 哈希 = H(P0) → 命中 block_A0 → B.BlockTable[0] = block_A0, refcount 2
block1 哈希 = H(P1) → 命中 block_A1 → B.BlockTable[1] = block_A1, refcount 2
...（block2~5 同理全部命中）...
```

6 个前缀 block 全部命中，于是 **B 不需要为这 96 个 token 做任何 prefill 计算**，只 prefill 自己的 40 个输入 token（对应 3 个新 block：block_B6、block_B7、block_B8，最后一个不满 16 不写缓存）：

```text
B BlockTable = [block_A0, block_A1, ..., block_A5, block_B6, block_B7, block_B8]
refcount: block_A0~A5 = 2（A 和 B 共享）
B 的 prefill 计算量 = 40 token（而非 136）
```

**第三步：Request C 到达**。同理，C 的 96 token 前缀 6 个 block 全部命中缓存，只 prefill 60 个输入 token（4 个新 block）：

```text
C BlockTable = [block_A0, ..., block_A5, block_C6, block_C7, block_C8, block_C9]
refcount: block_A0~A5 = 3（A、B、C 共享）
C 的 prefill 计算量 = 60 token（而非 156）
```

**汇总**：3 个请求的总 prefill 计算量从 408 token 降到 116 + 40 + 60 = 216 token，**节省约 47%**；显存上，6 个共享 block 只存一份，省下 3×96 − 96 = 192 token 的 KV 空间（12 个 block，约 96 MB，按每 block 8 MB 计）。如果模型 prefill 吞吐是 10,000 token/s，3 个请求的总 prefill 时间从约 40.8 ms 降到 21.6 ms。而且随着共享相同前缀的请求越多（100 个、1000 个），节省比例越接近"前缀占比"——因为**只有第一个请求承担前缀的 prefill 成本**，后续请求全部命中缓存。这就是 APC 在对话、RAG 等"高前缀复用"场景下收益巨大的原因。

> **面试一句话总结**：Automatic Prefix Caching 以 block 为粒度对 KV Cache 做内容哈希缓存，新请求的 prompt 与缓存逐 block 匹配，命中的前缀直接复用物理 block、只计算差异部分，从而省掉大量重复 prefill 算力和显存，且完全自动无需人工标记前缀。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 Prefix Caching 核心在 `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/kv_cache_coordinator.py`**——`KVCacheManager.get_computed_blocks` 调 `coordinator.find_longest_cache_hit`，后者按请求的 `block_hashes`（逐 block 内容哈希）在缓存里找最长匹配前缀：

```python
# kv_cache_manager.py —— Scheduler 每步调它获取"已算好的前缀 blocks"
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
    """Get the computed (cached) blocks for the request.
    Note that the computed blocks must be full.（必须是完整 block）"""
    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0
    # 全部命中时需重算最后 token 拿 logits，故 max_cache_hit_length = len - 1
    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, num_new_computed_tokens, num_uncached = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length)
    )
    return computed_blocks, num_new_computed_tokens, num_uncached
```

```python
# kv_cache_coordinator.py —— 协调器把查找转发给 SingleTypeKVCacheManager
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
        block_hashes=block_hashes, max_length=max_cache_hit_length,
        kv_cache_group_ids=[0], block_pool=self.block_pool,
        kv_cache_spec=self.kv_cache_spec, drop_eagle_block=...,
        alignment_tokens=self.block_size, ...)   # 逐 block 哈希匹配最长前缀
    return hit_blocks, hit_length, 0
```

**这段代码关键在哪**：Prefix Caching 的完整链路是——①Scheduler 调 `get_computed_blocks`（注意"computed blocks must be full"，对应"只有完整 block 才能共享"）；②`find_longest_cache_hit` 按 `block_hashes`（逐 block 的内容哈希）找最长匹配前缀，命中部分直接复用、未命中部分（`num_new_computed_tokens` 之后）才 prefill；③`max_cache_hit_length = num_tokens - 1` 保证最后一个 token 总被重算（拿 logits）。这个"哈希匹配 + 完整 block + 末位重算"就是 APC 的实现核心。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Prefix Caching 在 v0.25.x 中依然是默认启用的核心优化，且覆盖面在扩大：v0.25.0 的 release notes 明确提到 **prefix caching for Mamba hybrid models**（#42406）——即混合架构中 Mamba（线性注意力）部分的 KV/状态也能参与前缀缓存，而不是只对 Transformer 部分的 KV 生效；同时 multimodal-prefix bidirectional attention（#46942）也在推进，说明前缀复用在多模态输入上也向"双向注意力前缀"扩展。另一个值得注意的关联演进是：Prefix Caching 依赖"block 内容哈希 + 引用计数共享"，这套机制在新版 MRv2 架构下仍然由 KV block 管理负责（见第 3 点的新版特性），所以"哈希匹配 → 复用物理 block → 只 prefill 差异部分"的心智模型完全不过时。面试时可以补充一句："Prefix Caching 的粒度是 block，命中率取决于前缀的 token 级一致性，这也是为什么 vLLM 用 xxhash 做内容哈希、并要求完整 block 才可共享。"

---

# 二、Prefill 和 Decode：推理 Infra 面试的绝对重点


## 5. Prefill Phase

### 1. 现有问题：为什么要区分 Prefill Phase

Prefill Phase 是 LLM 推理中处理**输入 prompt** 的阶段，它与 Decode 阶段有着截然不同的计算特性和性能瓶颈，理解两者的差异是优化推理引擎的前提。在 autoregressive 生成中，模型一次只能生成一个 token，但输入 prompt 往往有成百上千个 token，这些 token 之间没有依赖关系（在 causal attention 的约束下），可以一次性并行计算。如果系统用"逐 token 处理输入"的方式做 prefill，那么一个 1000 token 的 prompt 就要串行执行 1000 次前向，延迟完全不可接受。因此 prefill 阶段必须利用并行性，一次性把整个 prompt 的所有 token 的前向计算做完，得到每个位置对应的 KV Cache 和最后一个 token 的 logits。

如果不把 prefill 当作一个独立阶段来专门优化，会出现两类问题。第一类是**计算效率问题**：prefill 是典型的 compute-bound（计算密集）操作，输入 token 全部参与矩阵乘法，GPU 的算力利用率很高，但前提是 batch 要足够大、序列要足够长；如果只是"顺便"用 decode 的方式逐 token 算，算力利用率会暴跌到个位数百分比。第二类是**调度公平性问题**：在 Continuous Batching 下，一个大请求的 prefill 会一次性占用大量 GPU 算力（它要做一次超大矩阵乘法），如果它插进正在 decode 的 batch，会大幅拖慢其他正在逐 token 生成的请求——这就是"长 prefill 卡住短 decode"问题，直接催生了 Chunked Prefill（见第 7 点）。所以 prefill 必须被识别为独立阶段，并用专门的策略（并行、chunking、与 decode 交错）来处理。

### 2. 方法论：Prefill Phase 是怎么实现的

Prefill 的实现本质上是**对整个 prompt 做一次完整的前向传播**，逐层逐步进行。以一个 prompt 为 L 个 token 的请求为例，具体步骤如下。

**第 1 步（Embedding）**：把 L 个 token id 查表映射成 L 个向量，得到形状为 (L, hidden_dim) 的输入张量。**第 2 步（逐层前向）**：依次通过每一层 Transformer block（共 N 层），每层内做两件事：self-attention 和 MLP。self-attention 这一步是关键：对第 l 层，输入 (L, hidden_dim) 先投影出 Q、K、V（形状都是 (L, num_heads, head_dim)），然后计算 Attention(Q, K, V)——由于 query 和 key 长度都是 L，这是一个完整的 L×L 注意力矩阵（带 causal mask 的上三角部分不用算），同时**把每层的 K、V 写入 KV Cache**（第 3 点管理的 block 里），供后续 decode 复用。MLP 部分则是两个线性变换加激活。**第 3 步（取最后位置）**：经过所有层后，输出 (L, hidden_dim) 的 hidden state，但**只有最后一个位置（第 L 个 token）的 hidden state** 会被接上 LM Head（词表投影）得到形状为 (vocab_size,) 的 logits，作为第一个生成 token 的概率分布。中间位置（1~L−1）的 hidden state 只用于生成自己的 KV Cache，不参与输出。

在工程实现上，vLLM 的 prefill 走的是 **XFormers / FlashAttention / FlashInfer 等的高效 attention kernel**（见第 10、11 点）：prefill 时 query 长度和 key 长度相同（都是整个 prompt），attention 计算是一个完整矩阵，kernel 会利用并行性做分块（tiling）计算，并把 softmax 的在线计算（online softmax）与 KV 的写入结合起来，避免把中间矩阵写回显存。由于 prefill 计算量正比于序列长度的平方（attention 部分是 O(L²)），长 prompt 的 prefill 非常耗时，因此 vLLM 提供了 `max_num_batched_tokens` 和 `max_num_seqs` 等参数来控制单次 prefill 的规模：当多个请求同时到达时，调度器会把它们按 token 预算打包进同一个 batch 一起 prefill（**prefill batching**），以充分利用 GPU 算力；当一个请求的 prompt 过长时，则会被切分成多个 chunk 分批处理（Chunked Prefill，见第 7 点）。此外，prefill 和 decode 的 kernel 是分开优化的：prefill 用高并行度的计算 kernel，decode 用访存优化的 kernel，vLLM 会根据当前 batch 里是否存在 prefill 请求动态选择合适的 kernel 和 CUDA Graph。

### 3. 具体数值样例

假设一个模型的 prefill 吞吐是 10,000 token/s（单 batch），现在同时到达 3 个请求，prompt 长度分别为 500、1000、2500 token。先看"用逐 token decode 方式处理 prefill"有多慢：假设 decode 吞吐 200 token/s，那么 2500 token 的请求光"吃掉"输入就要 2500/200 = 12.5 秒，3 个请求合计约 20 秒，延迟完全不可接受。而按并行 prefill 处理：3 个请求合计 4000 token，打包成一批做 prefill（输入张量形状 (4000, hidden_dim)，一次前向），只需要 4000/10000 = 0.4 秒，延迟降低约 50 倍。

再看 prefill 计算量的具体结构。假设模型 hidden_dim = 4096、32 层、num_heads = 32、head_dim = 128，一个 1000 token 的请求做 prefill：每层的 Q/K/V 投影输入是 (1000, 4096)，输出是 (1000, 4096)（Q/K/V 各 (1000, 4096)），这部分 FLOPs ≈ 3 × 2 × 1000 × 4096 × 4096 ≈ 100 GFLOPs；attention 部分是 (1000, 32, 128) × (1000, 32, 128)ᵀ，即每头 1000×1000 的矩阵乘，32 头共 2 × 1000 × 1000 × 128 × 32 ≈ 8.2 GFLOPs；MLP 两个线性层又是 2 × 2 × 1000 × 4096 × 16384 ≈ 268 GFLOPs。单层合计约 376 GFLOPs，32 层共约 12 TFLOPs——这就是为什么 prefill 是"计算密集"：它要在一次前向里完成 12 TFLOPs 的计算，GPU 算力利用率可以打到很高。

再看长 prefill 对 decode 的影响：假设一个 batch 里已有 4 个请求正在 decode（每步约 5 ms），此时到达一个 2000 token 的大请求。如果不做 Chunked Prefill，调度器一次性 prefill 这 2000 token，假设 prefill 一个 2000 token 的请求需要 200 ms，那么这 200 ms 内整个 batch 的 decode 全部被阻塞——4 个正在生成的请求每个都被拖慢 40 个 token 步的时间。如果开启 Chunked Prefill，把这个大请求切成 4 个 500 token 的 chunk，每个 chunk prefill 约 50 ms，交错插在 decode step 之间，那么每个 decode step 的延迟从 5 ms 增加到约 55 ms（多了 50 ms prefill），但这是可预测的、均摊的，不会出现一次 200 ms 的"卡死"，整体吞吐和尾延迟反而更好。

> **面试一句话总结**：Prefill 是并行处理整个输入 prompt 的计算密集阶段，一次前向算出全部位置的 KV Cache 和首个 token 的 logits；它必须与 decode 分开优化，并通过打包批处理提升算力利用率、通过 Chunked Prefill 避免长输入卡住 decode。

### 4. 核心代码（源码）

**Prefill 在 vLLM 里不是独立阶段，而是"请求的 `num_computed_tokens` 追 `num_tokens_with_spec`"的调度结果**（v1 Scheduler 的设计，见第 2 点）——多个新请求的 prompt 会被**打包进同一个 batch 一起前向**，这是 prefill 并行性的代码体现：

```python
# vllm/v1/core/sched/scheduler.py —— schedule() 里处理 waiting（新请求 = prefill）
def _enqueue_waiting_request(self, request: Request) -> None: ...
# 调度时把多个 waiting 请求的 prompt 一起安排（共享 token_budget），
# 它们在一个 step 里被 model_executor 打包成 (num_prompts, max_prompt_len)
# 的 batch 做一次前向 —— 这就是"prefill batching"。

# vllm/v1/engine/core.py —— 执行（prefill 和 decode 走同一个 step）
scheduler_output = self.scheduler.schedule(...)          # 决定谁 prefill / 谁 decode
future = self.model_executor.execute_model(scheduler_output, non_block=True)
# gpu_model_runner 把 batch 组织成输入张量（按最长序列 padding + attention mask）
```

**这段代码关键在哪**：v1 里 **prefill 和 decode 没有独立代码路径**——Scheduler 按 token 预算决定"这个 step 里新请求 prefill 多少、running 请求 decode 多少"，`model_executor.execute_model` 统一执行。新请求的多个 prompt 在同一 step 打包 = prefill batching；`long_prefill_token_threshold` 切 chunk = Chunked Prefill。**理解这一点比背 prefill 公式更重要**：v1 的"无阶段之分"设计让 prefill/decode/chunked/投机共用一套调度。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Prefill 的计算本质（并行处理整个 prompt、写 KV、取最后位置 logits）在 v0.25.x 没有变化，变化主要在执行路径上：v0.25.0 起 MRv2 成为 dense 模型默认路径，prefill 的 kernel 分派和输入打包逻辑被重构进 MRv2 的模块化执行链中（配合 FlashAttention 系 / FlashInfer 的 paged prefill kernels）。另一个值得注意的演进是 **Transformers modeling backend 性能追平 native**：v0.25.0 的 release notes 提到基于 transformers 的建模后端现在和 native vLLM 一样快（#47187）——这意味着 prefill 的执行不只依赖 vLLM 自研的模型实现，社区模型库的 forward 也能跑出接近的性能，对"prefill 阶段谁在算"的理解更宽了。面试时可以提："prefill 的并行计算思想不变，新版主要是在 MRv2 下换 kernel、换打包路径，并让 transformers 后端也达到 native 性能。"

---

## 6. Decode Phase

### 1. 现有问题：为什么要单独研究 Decode Phase

Decode 阶段是 LLM 推理中**逐 token 生成输出**的阶段，它是整个推理过程中最耗时、最决定用户体验的部分，也是推理引擎优化的主战场。Decode 的本质是一个串行循环：每次只输入当前序列的**最后一个 token**，前向传播得到下一个 token 的概率分布，采样出一个 token 追加到序列末尾，然后重复，直到生成 EOS 或达到 max_tokens。一个 500 token 的输出就需要 500 次这样的串行前向，而每次前向中，真正"新计算"的只有一个 token 的 embedding 及其对应的注意力计算——绝大部分计算量（之前所有 token 的 hidden state 计算）已经被缓存起来了。

Decode 阶段的核心问题在于**它是 memory-bound（访存密集）而非 compute-bound**。每次前向只处理一个 token，矩阵乘法的计算量很小（一个 token 的 hidden dim 向量乘权重矩阵），但模型权重（动辄几十 GB）每次都要从 HBM 读一遍，Attention 部分还要读取全部历史 KV Cache。换句话说，Decode 阶段 GPU 的算力利用率很低（通常只有个位数到十几个百分点），瓶颈在于"把权重和 KV Cache 从显存搬到计算单元"的带宽，而不是计算本身。另一个问题是**batch 内的长度不齐**：Continuous Batching 下每个请求的序列长度各不相同，attention 计算需要各自屏蔽，同时每次生成后序列长度变化，导致无法像 prefill 那样用固定的形状高效计算。Decode 的延迟直接决定首 token 之后的"字间间隔"（inter-token latency / TPOT），是流式输出体验的关键指标，因此必须针对它的访存特性做专门优化。

### 2. 方法论：Decode Phase 是怎么实现的

Decode 的实现是**每步一个 token 的前向循环**，具体步骤如下。**第 1 步（输入当前 token）**：取序列当前最后一个 token 的 id（形状 (1,)），查 embedding 得到形状 (1, hidden_dim) 的向量。**第 2 步（逐层前向）**：依次通过每层 Transformer。在 self-attention 这一步：把当前 token 的输入投影出 query（形状 (1, num_heads, head_dim)），同时计算当前 token 的 K、V 并**追加写入该序列的 KV Cache**；然后拿 query 与 KV Cache 中**全部历史位置**的 K、V 做注意力计算——即 query 长度为 1、key/value 长度为整个已生成序列长度 N 的"1×N"注意力，得到 (1, hidden_dim) 的输出。MLP 部分照常。**第 3 步（采样）**：最后一层的输出接 LM Head 得到 (vocab_size,) 的 logits，经采样（见第 15 点）选出一个 token id，追加到序列末尾。**第 4 步（循环）**：回到第 1 步，用新生成的 token 继续，直到 EOS 或 max_tokens。每步之间序列的 KV Cache 增长一个 token（跨 block 边界时分配新 block，见第 3 点）。

这里有两个关键优化。第一是 **KV Cache 的作用**：由于历史 token 的 K/V 已经缓存，decode 每步只需要计算当前 token 的 K/V 并追加到缓存，避免了重复计算，这正是 PagedAttention + KV Cache Management 存在的意义。第二是 **decode 专用 kernel**：FlashAttention 等 kernel 在 decode 形态下会做针对性的访存优化——它不会一次性把所有 KV 读进寄存器（那会溢出），而是分块读取、用 online softmax 维护 running max 和 running sum，把访存量从"全部 KV"降到"分块流式读取"，并且只在最后把结果写回。在 vLLM 中，Decode 是**批处理**的：一个 batch 里的所有请求并行执行各自的"1×N"注意力，GPU 同时为几十上百个序列服务，权重只从 HBM 读一次、被整个 batch 摊薄。为了降低 per-step 开销，vLLM 大量使用 **CUDA Graph**（见第 12 点）来消除 kernel launch 的 CPU 开销：把整个 decode step 的一串 kernel 捕获成一个 graph，之后每次只需一次 launch 即可完整执行，把每步的 CPU 开销从几毫秒降到几十微秒。此外，decode 阶段的调度还涉及 **batch size 控制**：batch 越大，每步计算量越大但吞吐越高，单个请求的 TPOT 也会相应变长（因为共享算力），系统需要在吞吐和延迟之间权衡（vLLM 通过 `max_num_seqs` 限制并发展）。Decode 阶段也是投机解码（Speculative Decoding）作用的舞台：用一个小模型/草稿模型先草拟多个 token，再用大模型一次验证，把多次串行 decode 变成"一次验证多个位置"，从而突破单 token 串行的延迟下限（详见 Speculative-Decoding.md）。

### 3. 具体数值样例

假设一个 7B 模型（FP16，权重 14 GB）在 H100（HBM 带宽约 3.35 TB/s）上做 decode，单请求、无 batch。逐步算一下每一步的访存量：第 1 项，模型权重 14 GB——每层的前向都要把该层权重从 HBM 读到寄存器，32 层全部读完就是 14 GB；第 2 项，当前序列的 KV Cache——假设已生成 1000 token，每 token 512 KB，共约 500 MB。每步总访存量约 14.5 GB，按带宽 3.35 TB/s 计算，每步理论最短时间约 14.5 GB / 3350 GB/s ≈ 4.3 ms，即单请求 TPOT 约 4.3 ms，吞吐约 232 token/s。而如果只看计算量：每个 token 的前向 FLOPs 大约 2 × 7B × 2（权重乘激活）≈ 28 GFLOPs，H100 算力约 989 TFLOPs，理论计算时间仅 28/989000 ≈ 0.028 ms——可见计算只占 0.6%，**99% 以上的时间都花在搬运权重上**，这就是"访存密集"的量化体现。

再看 batch 的效果，逐步演算：如果 batch 变成 8 个请求，每步需要读取 8 份 KV Cache（约 4 GB）+ 同一份权重 14 GB（权重可以广播复用，8 个请求共享一次读取），总访存约 18 GB，每步约 18 GB / 3350 GB/s ≈ 5.4 ms——单步时间只增加了一点（从 4.3 ms 到 5.4 ms），但一步同时推进了 8 个请求，系统吞吐从 232 token/s 提升到约 8 / 5.4 ms ≈ 1480 token/s，翻了 6 倍多。这就是为什么服务端必须靠 Continuous Batching 把并发请求攒在一起：权重只读一次、被整个 batch 摊薄，访存瓶颈被 batch 平均分担，吞吐随 batch size 近似线性增长（直到 KV Cache 带宽或算力成为新瓶颈）。再对比 batch=32：KV 部分约 16 GB，权重 14 GB，总访存约 30 GB，每步约 9 ms，吞吐约 32/9ms ≈ 3556 token/s——可见只要 KV Cache 总量不超过显存，增大 batch 是提升 decode 吞吐最直接的手段。

> **面试一句话总结**：Decode 是逐 token 串行生成、依赖 KV Cache 的访存密集阶段，单步时间几乎全花在搬运权重和历史 KV 上，通过批处理摊薄权重读取、用 decode 专用 kernel 和 CUDA Graph 压低每步开销，是推理吞吐与流式延迟优化的核心。

### 4. 核心代码（源码）

**Decode 的"1×N 注意力"由 attention backend 的 decode kernel 执行**（vLLM 按形态分派到不同 kernel）——FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper` 就是 query 长度 1、KV 长度 N 的批量 decode kernel：

```python
# vllm/v1/attention/backends/flashinfer.py —— decode 走分页 KV 的 1×N kernel
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

# 一个 decode step：每个请求 query 长度=1，KV 长度=已生成长度（分页寻址）
# 返回 (num_reqs, num_kv_heads, head_dim) 的注意力输出
output = wrapper.forward(
    q,                    # (num_reqs, 1, num_heads, head_dim)  ← query 长度 1
    k, v,                 # 通过 block table 间接访问的分页 KV
    block_tables,         # BlockTable（logical → physical，见第 1 点）
    seq_lens,             # 各请求已生成长度
    ...)
# wrapper 内部：分块读 KV + online softmax，把每步访存压到"只读一遍 KV"
```

```python
# vllm/v1/attention/backends/flash_attn.py —— FlashAttention 的 decode 分派
class FlashAttentionImpl:
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, ...):
        # 根据 attn_metadata.is_prompt 区分 prefill / decode，分派不同 kernel
        if attn_metadata.is_prompt:    # prefill：完整矩阵
            output = vllm_flash_attn.varlen_fwd(...)
        else:                          # decode：1×N 分页
            output = vllm_flash_attn.varlen_fwd(..., is_prompt=False, ...)
```

**这段代码关键在哪**：Decode 的 kernel 形态就是"query 长度 1 + 分页 KV"——FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper` 和 FlashAttention 的 `varlen_fwd(is_prompt=False)` 都是 decode 专用路径（`is_prompt` 标志区分 prefill/decode 形态）；`block_tables` 参数就是 PagedAttention 的 BlockTable 传给 kernel 做间接寻址。**这就是"decode 是访存密集"的代码证据**：kernel 每步只处理 1 个新 query，但要把整个 KV 读一遍。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Decode 的访存密集特性（权重 + KV 搬运主导每步时间）在 v0.25.x 依然是根本约束，但有两个直接相关的演进。第一是 **dynamic speculative decoding 与 full CUDA graphs 兼容**（v0.25.0，PR #45953）：之前投机解码（见 Speculative-Decoding.md）的草稿长度动态变化，往往被迫退出 CUDA Graph 走慢路径；v0.25.0 让动态投机解码也能跑在完整的 CUDA Graph 上，直接压低 decode 每步的 CPU 开销——这是"decode 阶段继续压低 per-step 开销"方向的最新落地。第二是 **MRv2 的 GPU-native / async-first 执行**：decode step 的 kernel 编排在 MRv2 下更模块化，配合 FlashAttention 系 / FlashInfer 的 paged decode kernels 与 FP8 KVCache（见第 3 点），进一步减少每步的显存搬运量。面试时可以补充："decode 的优化主线始终是'访存密集'——批处理摊薄权重、paged kernel 流式读 KV、CUDA Graph 消除 launch 开销，v0.25 的 dynamic spec decode + full graphs 是这条主线的最新一步。"

---

## 7. Chunked Prefill

### 1. 现有问题：为什么要提出 Chunked Prefill

Chunked Prefill 要解决的核心矛盾是：**长 prompt 的 prefill 会一次性独占 GPU 算力，导致 Continuous Batching 下的 decode 请求集体被"卡死"。** 前面（第 2 点）讲过 Continuous Batching 让新请求随时 prefill、随时加入 decode batch，但这里有个隐患：如果某个请求的 prompt 特别长（比如 2000 token），一次性 prefill 它需要执行一个巨大的矩阵乘法，会占用 GPU 全部算力几十到几百毫秒。在这段时间里，batch 中其他正在逐 token 生成的请求（每步只要几毫秒）完全得不到执行——它们被迫等待这个大 prefill 完成才能继续 decode。从用户体验上看，就是所有正在流式输出的请求突然"卡住"很久，尾延迟（tail latency）急剧恶化。

这个问题的本质是 **prefill 和 decode 的计算特性冲突**：prefill 是计算密集的"大块头"，decode 是访存密集的"小碎步"，两者混在一个 batch 里时，如果 prefill 不做切分，就会周期性"霸占"整块 GPU。此外还有显存峰值问题：一次性 prefill 一个超长 prompt，会瞬间申请大量 KV Cache block 和中间激活，可能导致显存水位剧烈波动，甚至触发不必要的抢占。更麻烦的是，在请求长度长尾分布明显的真实场景中（比如 RAG 检索结果很长、代码补全上下文很大），长 prompt 请求占比不低，如果不对 prefill 做切分，系统要么牺牲 decode 的流畅性，要么给 prefill 设很低的并发上限——两种选择都严重损害整体吞吐。

### 2. 方法论：Chunked Prefill 是怎么实现的

Chunked Prefill 的核心思想是：**把一个长 prompt 的 prefill 计算切成多个 chunk，每次只 prefill 一部分 token，并在多个 decode step 之间交错执行。** 具体实现分三个层面。

**第一层：切分**。调度器把 prefill 请求的 prompt 按 chunk 大小切分。chunk 大小由 `max_num_batched_tokens` 等参数控制（例如每次最多算 512 个 token）：一个 2000 token 的 prompt 会被切成 4 个 500 token 的 chunk。**第二层：混合调度**。每一轮调度中，GPU 的"token 预算"被分成两部分：一部分分给 decode（继续推进 running 中已有的请求），一部分分给某个 prefill 请求的一个 chunk。两者在同一个 step 里混合执行——即 batch 里同时包含 decode 请求和"部分 prefill 的请求"。vLLM 的 Scheduler 在每轮调度时按固定顺序分配预算：先满足 running 队列中 decode 请求的预算，再把剩余预算分配给 waiting 队列中 prefill 请求的一个或多个 chunk；当预留给 prefill 的 token 数超过一个 chunk 时，就把多个请求的各一个 chunk 打包成一次 prefill 计算，进一步提升算力利用率。**第三层：chunk 内计算与状态延续**。每个 chunk 的前向仍然是"并行处理 chunk 内所有 token"的 prefill 计算，但 attention 部分比较特殊：当前 chunk 的 query 长度 = chunk 大小（如 512），而 key/value 长度 = 该请求**到目前为止已处理的所有 token**（包括之前 chunk 的 token 和本 chunk 的 token）——也就是说，每个 chunk 都要对"全部历史 KV"做注意力，只是 query 只覆盖本 chunk。FlashAttention 等 kernel 天然支持这种"query 长度 < KV 长度"的不规则形态。每个 chunk 算完，把该 chunk 的 KV 写入 KV Cache，保存中间状态，等下一轮调度再处理下一个 chunk，直到整个 prompt 处理完，请求转为纯 decode。

这里有两个重要的权衡。第一，**切分不是免费的**：每个 chunk 都要重新计算 attention 的 softmax 归一化统计（对已处理 KV 做在线 softmax），且 chunk 之间有 kernel 启动的额外开销，所以 chunk 大小需要权衡——太小则 kernel 开销占比过高、效率低，太大则又回到"长 prefill 卡 decode"的问题。第二，**TTFT 会略有增加**：一个请求的 TTFT 不再是"一次性 prefill 完"，而是"最后一个 chunk 处理完"，中间可能插入了几轮别人的 decode，所以长请求的 TTFT 会从"一次 prefill 时间"变成"多个 chunk 时间之和 + 交错等待"，但因为 chunk 计算量小、交错紧密，整体 TTFT 增加通常很有限（在下一节用数字说明）。

### 3. 具体数值样例

假设 GPU 单次最多并行处理 2048 个 token（`max_num_batched_tokens = 2048`），chunk 大小取 512 token。当前 batch 有 4 个 decode 请求（每步约 5 ms），此时到达一个 prompt 为 4000 token 的大请求。先看不做 Chunked Prefill 的情况：调度器一次性 prefill 这 4000 token，假设 prefill 吞吐 10,000 token/s，需要 4000/10000 = 400 ms——这 400 ms 内 4 个 decode 请求全部停摆，每个都损失约 80 个 token 步的进度，用户体验就是明显的"打字机卡顿"。

开启 Chunked Prefill 后，逐轮演算（忽略切换开销）：4000 token 切成 8 个 chunk（7 个 512 + 1 个 416，最后一个不满也算一个 chunk）。每一轮调度：

```text
第 1 轮：decode 4 个请求（5 ms）+ prefill chunk1（512 token，512/10000 ≈ 51 ms）≈ 56 ms
第 2 轮：decode 4 个请求（5 ms）+ prefill chunk2（512 token ≈ 51 ms）≈ 56 ms
...（以此类推，每轮 56 ms）
第 8 轮：decode 4 个请求（5 ms）+ prefill chunk8（416 token ≈ 42 ms）≈ 47 ms
```

8 轮总共约 56×7 + 47 ≈ 439 ms，这个请求的 prefill 才完成。对比一次性 prefill 的 400 ms，总时间略长（多了约 39 ms 的 chunk 间开销和交错等待），但关键区别在于：**4 个 decode 请求在这 439 ms 内持续以"每 56 ms 一步"的稳定节奏前进**，而不是被冻结 400 ms 再恢复。decode 请求的 TPOT 从 5 ms 变成 56 ms 是"每轮多等 51 ms prefill"，这是可预测的均摊延迟；而一次性 prefill 的 400 ms 停顿是"完全无响应"。在批处理场景下收益更明显：假设同时有 4 个长 prompt 请求到达，每轮可以把 4 个请求的各一个 chunk 拼进同一轮 prefill（4 × 512 = 2048 token，正好打满 token 预算），GPU 算力利用率从"周期性尖峰 + 大片空闲"变成"持续满载"，整体吞吐可以提升 20%~50%，同时每个长请求的 TTFT 只增加约一个 chunk 的时间（56 ms），几乎无感。

> **面试一句话总结**：Chunked Prefill 把长 prompt 的 prefill 切成小块、交错插在 decode step 之间执行，让 prefill 与 decode 共享 GPU 算力而非互相阻塞，既消除了长请求"卡死"整个 batch 的尾延迟问题，又把 GPU 算力利用率拉平到持续满载。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 Chunked Prefill 由 Scheduler 的 token 预算 + `long_prefill_token_threshold` 实现**——长 prompt 的 prefill 不会一次算完，而是被 `num_new_tokens` 的预算切分成多个 chunk（第 2 点已展示主循环，这里看配置与切分逻辑）：

```python
# vllm/config/scheduler.py —— chunked prefill 的核心配置
DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048   # 每步最大 token 数（chunk 预算）
DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128
max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
# 每步最多并行处理多少 token：decode 请求 + prefill chunk 共享这个预算

# vllm/v1/core/sched/scheduler.py —— 切 chunk 的调度逻辑（schedule() 内）
num_new_tokens = (request.num_tokens_with_spec
                  + request.num_output_placeholders
                  - request.num_computed_tokens)   # 该请求还需算的 token 数
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
    # 长 prefill：一次只算 threshold 个 token → 切 chunk
num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
# 同时受"每步 token 预算"和"batch token 预算"约束 → 与 decode 共享算力
```

**这段代码关键在哪**：Chunked Prefill 的实现就是"**token 预算约束下的分批**"——①`max_num_batched_tokens`（默认 2048）是每步的总 token 预算，decode 和 prefill chunk 都从这里面分；②`num_new_tokens` 计算"这个请求还需要算多少"，再被 `token_budget` / `input_budget` 截断——**一个 4000 token 的 prompt 在一次调度里最多分到预算内的 token 数（如 2048），剩下的留到下一步**，这就是"切 chunk"；③`long_prefill_token_threshold` 可以进一步把单请求的每步 token 数压小。v1 的"无 prefill/decode 阶段之分"让这个机制天然支持 chunked prefill。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Chunked Prefill 在 v0.25.x 中仍是处理"长 prompt 与 decode 混跑"的核心机制，未见根本性变更；它在新版中的角色更多是作为 **MRv2 执行链 + token 预算调度**的一部分被整合——每轮调度里"先满足 decode 预算、再把剩余 token 预算分配给 prefill chunk"的逻辑保持不变（见第 8 点），只是 chunk 的执行由 MRv2 更模块化地编排，并且可以与 FlashInfer 等 paged prefill kernel 的"query < KV"不规则形态更好地配合。一个相关的演进方向是：在 PD 分离（见第 16 点）或超大并发场景下，长 prompt 的处理越来越倾向于"专门的 prefill 资源 + 分块/分离"组合，Chunked Prefill 作为"同一批 GPU 内缓解 prefill/decode 冲突"的方案仍然不可替代。面试时可以提："Chunked Prefill 的核心——token 预算分配 + chunk 间状态延续——在新版没有变，它依然是混跑场景下拉平算力利用率的基础。"

---

# 三、Scheduler：vLLM 的核心系统组件

## 8. Scheduler Architecture

### 1. 现有问题：为什么 Scheduler 是 vLLM 的核心系统组件

Scheduler 是 vLLM 的"大脑"，所有请求的准入、排队、执行、抢占都由它统一决策，它的设计质量直接决定整个推理系统的吞吐、延迟和稳定性。推理引擎面对的是一个高度动态的环境：请求随时到达、长度差异巨大、KV Cache 资源有限且不断变化、prefill 与 decode 的计算特性迥异，还有抢占、共享前缀、流式输出等各种复杂情况。如果这些决策分散在各处，系统会陷入混乱——比如两个请求同时抢同一块显存、长 prefill 无节制地卡死 decode、显存不足时随机失败等。Scheduler 的作用就是**把所有资源分配和请求调度决策集中到一个组件里**，保证系统在任何时刻都处于"可执行、不冲突、资源不超限"的一致状态。

传统推理框架（以及简单的 LLM 服务实现）往往只有"来了就排队、凑够一批就跑"的粗糙调度，这在长尾请求分布和高并发场景下会导致严重的资源浪费和延迟抖动。vLLM 的 Scheduler 之所以被反复强调，是因为它把之前讲过的所有机制——Continuous Batching、Chunked Prefill、KV Cache 水位控制、Prefix Caching、抢占换出——全部收拢进一个统一的决策循环，每次调度都基于系统的全局状态做出一致的最优决策。可以说，理解了 Scheduler，就理解了 vLLM 如何把这么多机制组合成一个可工作的整体。

### 2. 方法论：vLLM 的 Scheduler 是怎么实现的

vLLM 的 Scheduler 采用**集中式、迭代式**的调度模型。核心数据结构是三个队列：**running**（正在执行的序列）、**waiting**（等待执行的序列）、**swapped**（被换出到 CPU 的序列），以及每步一次的调度循环 `schedule()`。每次引擎要执行一个 step（即一批 GPU 计算）之前，都会调用调度循环，按固定的优先级做决策。具体步骤分解如下：

**第 1 步：处理 running 队列的完成项。** 遍历 running，检查每个序列是否已生成完毕（到达 EOS 或 max_tokens）或已被用户 abort。完成的序列移出 running，并释放其 KV Cache block（引用计数归零的 block 回 free list，见第 3 点）。

**第 2 步：检查抢占条件。** 计算当前 KV Cache 的使用量。如果可用 block 数低于高水位阈值（watermark，例如总 block 的 10%），说明显存紧张，触发抢占：从 running 中挑选"受害者"序列（策略见第 9 点，如最老或最短），执行 swap（把 KV block 拷到 CPU，序列移入 swapped 队列）或标记 recompute，腾出显存。这一步可以循环执行多次，直到可用 block 高于水位。

**第 3 步：处理 swapped 队列。** 如果显存有余量（可用 block 高于水位），把 swapped 队列中优先级最高的序列换回（swap in：从 CPU 拷回 GPU block），加入 running。优先恢复被抢占的序列，是因为它们已经付出过 swap 成本，且用户还在等待。

**第 4 步：处理 waiting 队列。** 在 token 预算（`max_num_batched_tokens`）和序列数预算（`max_num_seqs`）允许的范围内，从 waiting 队首开始逐个取出新请求，分配 KV block、安排 prefill（支持 Chunked Prefill 时按 chunk 分配预算，见第 7 点），加入 running。如果显存不足，则停止，剩余的留在 waiting。

**第 5 步：产出执行计划并返回。** 调度循环输出本轮的执行计划（SchedulerOutput）：哪些序列 decode、哪些序列 prefill、哪些序列 swap in/out，以及对应的 KV block 分配信息。引擎把这个计划交给 Model Runner 实际执行，执行结果（新生成的 token、序列是否完成、KV 用量变化）再反馈给 Scheduler，进入下一轮循环。

这套设计的关键原则有三条。第一是**优先级顺序固定**：先保 running（已在进行中的请求优先，避免频繁抢占），再保 swapped（已经为它付出过 swap 成本的请求），最后才轮到新的 waiting 请求——这是"稳定性优先于公平性"。第二是**资源预算是全局的**：每次调度都以 KV Cache 空闲 block 数、token 预算、序列数预算为硬约束，任何决策都不会让资源超限。第三是**决策与执行分离**：Scheduler 只做决策不做计算，输出执行计划后由 Model Runner 执行，这让 Scheduler 可以纯逻辑地测试和推理，也方便横向扩展（如多 GPU worker 场景下每个 worker 执行同一份计划的不同分片）。

### 3. 具体数值样例

假设 KV Cache 池共有 100 个 block，高水位阈值是 90%（即可用 block 少于 10 个就触发抢占），`max_num_seqs = 8`，`max_num_batched_tokens = 2048`。某一时刻系统状态：running 队列有 4 个序列（A、B、C、D，各占 10/20/15/25 个 block，共 70 个），waiting 队列有 2 个新请求（E 需要 30 个 block、F 需要 10 个 block）。

**第一轮调度，逐步演算：**

```text
第 1 步：检查 running 完成项——假设 A 刚刚生成完毕，释放 10 个 block。
         running = [B, C, D]，可用 block = 100 − 60 = 40
第 2 步：水位检查：可用 40 > 10，不触发抢占。
第 3 步：swapped 队列为空，跳过。
第 4 步：处理 waiting：E 需要 30 个 block ≤ 可用 40，prefill E 加入 running
         （占 30 block，剩 10）；再试 F，需要 10 个 block ≤ 剩 10，prefill F
         加入 running（占 10 block，剩 0）。
         running = [B, C, D, E, F]
第 5 步：输出执行计划：B、C、D decode，E、F prefill。
```

这一轮结束后 KV 用满 100 个 block，GPU 满载执行。

**第二轮调度：** 又来了新请求 G（需要 20 个 block），但当前可用 block 为 0。

```text
第 1 步：running 无完成项。
第 2 步：水位检查：可用 0 < 10，触发抢占。按策略挑选受害者——
         选择"最不划算"的序列，比如序列最短的 C（占 15 block）：
         swap out C：把 C 的 15 个 block 拷到 CPU（假设耗时 5 ms），
         C 移入 swapped 队列，释放 15 个 block，可用变为 15。
第 3 步：可用 15 > 10，不 swap in。
第 4 步：处理 waiting：G 需要 20 个 block > 可用 15？——不够，G 继续等待；
         但若把预算再压缩一下，或者再抢占一个序列（如 D 的 25 block 中
         挑更小的），可以让 G 进入。为简化，假设继续抢占 B（20 block），
         可用变为 35，G prefill 加入（占 20，剩 15）。
         running = [D, E, F, G]，swapped = [C, B]
第 5 步：输出执行计划：D、E、F、G decode + swap out C、B。
```

注意这里体现了"稳定性优先"：为了一个新请求 G，系统抢占了两个老序列 B、C，而不是让 G 无限等待。虽然 B、C 付出 swap 成本（之后要换回），但整体上避免了新请求饿死。真实系统中抢占策略、水位值、受害者选择都会按部署场景调优（如 `preemption_mode` 选 swap 还是 recompute、`max_num_batched_tokens` 控制每步预算），Scheduler 的价值就在于把所有这些决策集中、有序、可预测地执行。

> **面试一句话总结**：vLLM 的 Scheduler 是一个集中式的每步调度循环，用 running / waiting / swapped 三队列 + 全局资源预算（KV block、token、序列数）+ 固定优先级（先 running 再 swapped 再 waiting），统一决策 decode / prefill / swap 操作，实现 Continuous Batching、Chunked Prefill 与抢占机制的协同。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 Scheduler 在 `vllm/v1/core/sched/scheduler.py`**——`Scheduler.schedule()` 是每步入口（已在第 2 点贴主循环），这里是它**调度 running / 抢占 / 收尾**的接口全貌：

```python
class Scheduler(SchedulerInterface):
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """每步调度：先 running（decode）→ 再 waiting（prefill）。
        没有显式 prefill/decode 阶段之分，统一按 token 预算分配。"""
        ...

    def add_request(self, request: Request) -> None:
        """新请求入 waiting 队列。"""

    def update_from_output(self, scheduler_output: SchedulerOutput) -> None:
        """每步执行后回填：根据 worker 结果更新请求状态 / 释放 KV。"""

    def _preempt_request(self, request: Request, timestamp: float, ...) -> None:
        """抢占请求并放回 waiting 队列（recompute 模式的核心）。"""
        assert request.status == RequestStatus.RUNNING
        self._free_request_blocks(request)          # 释放 KV blocks
        request.status = RequestStatus.PREEMPTED    # 状态置为被抢占
        request.num_computed_tokens = 0             # 已算 token 清零 → 之后重算
        request.num_preemptions += 1

    def _handle_stopped_request(self, request: Request) -> bool:
        """请求完成（EOS/max_tokens）→ 移出 running、释放资源。"""

    def get_kv_cache_usage(self) -> float:
        """当前 KV cache 水位（触发抢占的判定依据）。"""

    def has_finished_requests(self) -> bool: ...
    def get_num_unfinished_requests(self) -> int: ...
```

**这段代码关键在哪**：Scheduler 接口对应"决策层"的全部职责——①`schedule()` 每步产生执行计划（谁 decode / 谁 prefill）；②`update_from_output()` 执行后回填（决策-执行分离的"回"半环）；③`_preempt_request()` 是**抢占的代码实现**：释放 KV blocks + `num_computed_tokens = 0`（recompute 模式：清零后从 prompt 重算）+ 状态置 PREEMPTED（v1 的 swapped 语义合并进 waiting/PREEMPTED）；④`get_kv_cache_usage()` 是水位检查（触发抢占的依据）。这套接口把 Continuous Batching + Chunked Prefill + Preemption 全部收进一个类。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Scheduler 的"三队列 + 每步一调度 + 资源预算"核心结构在 v0.25.x 中保持稳定，它依然是 vLLM 决策层的中心；变化集中在两点。第一是 **执行侧的承接**：调度循环输出的执行计划（SchedulerOutput）在 v0.25.0 起由 MRv2 以更模块化的方式消费（决策与执行分离更彻底），但调度语义本身（先 running 再 swapped 再 waiting、token 预算分配）不变。第二是 **策略可配置化**：vLLM 较新版本把抢占策略、队列选择等抽象成可插拔的 Policy（v1 scheduler 的方向），让"每步决策"可以按业务定制（如优先保证长连接、按优先级抢人等）。面试时可以把 Scheduler 讲成"三队列 + 预算 + 策略"三层：队列是数据、预算是约束、策略是决策规则——这个心智模型在新版依然成立。

---

## 9. Preemption

### 1. 现有问题：为什么要引入 Preemption

Preemption（抢占）解决的是**显存资源不足时的处理策略**问题。在 Continuous Batching 场景下，KV Cache 是有限的（启动时预留的显存池就那么大），但请求是动态到达的：新请求需要 KV Cache block 来 prefill，正在生成的请求也在不断消耗新 block。当显存被占满、又有新请求必须服务时，系统面临一个两难：如果不做任何处理，新请求只能无限排队（可能饿死），或者直接报 OOM 错误让请求失败——这两种都是糟糕的用户体验。更隐蔽的问题是，即使没有新请求，一个正在生成的超长序列也可能在生成过程中耗尽剩余显存。因此系统必须有一种机制，在资源紧张时**主动牺牲一部分"进行中"的工作，腾出资源给更需要的工作**，这就是抢占。

抢占在操作系统里是一个经典概念（进程被高优先级任务打断、上下文切换、换页到磁盘），LLM 推理里的抢占与之非常相似，但有一个关键差异：LLM 请求是**有状态**的——它已经生成了若干 token，这些 token 的 KV Cache 是后续生成的前提。被抢占的请求不能简单丢弃（用户还在等它的输出），必须要么把它的状态保存起来以后恢复（swap），要么保留"重算成本"以便将来重新生成（recompute）。抢占策略设计得好不好，直接决定系统在过载情况下是"优雅降级"还是"整体崩溃"，也决定尾延迟和公平性。

### 2. 方法论：vLLM 的 Preemption 是怎么实现的

vLLM 的抢占由 Scheduler 触发（见第 8 点第 2 步），有**两种模式**：swap（换出）和 recompute（重算）。触发条件是 KV Cache 使用量超过高水位（watermark）阈值——Scheduler 在每轮调度时检查，如果可用 block 数低于预留的"安全水位"，就从 running 队列中选择受害者序列，把它们移出当前批次。下面分别讲两种模式的完整操作流程。

**Swap 模式（换出到 CPU）**，逐步操作：**第 1 步**，Scheduler 从 running 中选择受害者序列（选择策略：通常优先选"换出代价最小"的——序列最短的 KV 数据量最小、拷贝最快，或最老的序列进度最接近完成判断）；**第 2 步**，把该序列的全部 KV Cache block 从 GPU 显存整体拷贝到 CPU 内存（由 CPU 侧的 block 管理器分配空间），记录"该序列的 block 现在在 CPU 的哪个位置"；**第 3 步**，序列移入 swapped 队列，其 GPU block 全部释放回 free list，供其他请求使用；**第 4 步**，将来显存有富余时（可用 block 高于水位），Scheduler 把该序列从 swapped 队列取出，把 CPU 上的 block 拷回 GPU（swap in），序列回到 running 队列，从**它之前停下的位置继续生成**——注意它已经生成的 token 和 KV Cache 都在，无需重算任何内容。Swap 的优点是恢复后**零重算**，缺点是 swap out/in 各需要一次 PCIe 拷贝（CPU-GPU 之间通常只有几十 GB/s），带宽成本高，且 CPU 内存本身也是有限资源。

**Recompute 模式（丢弃重算）**，逐步操作：**第 1 步**，同样选择受害者序列；**第 2 步**，直接释放该序列的全部 GPU KV block（**不拷贝到 CPU**，因此省去拷贝带宽、不占 CPU 内存），序列移入 swapped 队列但只保留它的 token 列表（prompt + 已生成 token 的文本）；**第 3 步**，将来显存富余时恢复：从该序列的 prompt 开始**重新做 prefill**，一直算到它之前生成的最后一个 token，重新建立完整 KV Cache，然后继续生成。Recompute 的优点是**不占用 CPU 内存、无拷贝开销**，缺点是恢复时要把已生成的全部 token 重新计算一遍，浪费算力。

vLLM 默认根据配置选择模式（`preemption_mode`，可设为 swap / recompute / 混合），受害者选择策略也可配置。一个重要的工程细节是：**抢占是"整序列"粒度的，不是"部分 block"粒度的**——要么整个序列被换出，要么不动，因为把一个序列的部分 KV 留在 GPU 上没有任何收益（它无法继续生成）。抢占发生后，GPU 的显存压力立即缓解，新请求得以 prefill，系统整体吞吐不会因个别长请求而崩溃。在 PD 分离架构（见第 16 点）中，由于 prefill 与 decode 在物理上分离，显存压力大幅缓解，抢占的需求也随之降低，这是 PD 分离的额外收益之一。

### 3. 具体数值样例

假设 KV Cache 池共 80 个 block，高水位阈值为 90%（可用 block 低于 8 个即触发抢占），每 block 8 MB。当前 running 有 4 个序列：A（10 block，已生成 300 token）、B（25 block，已生成 800 token）、C（20 block，已生成 500 token）、D（22 block，已生成 700 token），共占用 77 个 block，可用只剩 3 个。此时新请求 E 到达，需要 12 个 block 才能 prefill。

**逐步演算 Swap 模式：**

```text
第 1 步：调度器发现可用 block（3）< 水位（8），触发抢占。
        选择受害者：按"先抢最短序列"策略，选 A（10 block 最小）。
第 2 步：把 A 的 10 个 block（10 × 8 MB = 80 MB）经 PCIe 拷贝到 CPU，
        假设 PCIe 拷贝带宽 40 GB/s，耗时 80 MB / 40 GB/s ≈ 2 ms。
第 3 步：A 移入 swapped 队列，GPU 释放 10 个 block，可用变为 13。
        随后 E 获得 12 个 block prefill 进入 running，可用剩 1。
第 4 步：之后某请求（如 B）完成释放 25 个 block，可用变为 26 > 水位 8，
        调度器把 A swap in 回 GPU（再花约 2 ms 拷贝），A 回到 running
        继续生成。对用户来说，A 的输出只是"停顿了一会儿"，内容连续，
        没有丢失也没有重算。
```

**对比 Recompute 模式：**

```text
第 1 步：同样抢占 A，但不拷贝到 CPU。
第 2 步：直接释放 A 的 10 个 block（省去 2 ms 拷贝，也不占 CPU 内存），
        A 被记录为"需从 prompt 重算"，token 列表保留在内存。
第 3 步：显存富余时恢复：A 从 prompt 开始重新 prefill 到第 300 个 token，
        假设 prefill 吞吐 10,000 token/s，300 token 需要 30 ms；同时
        重新消耗约 ⌈300/16⌉ = 19 个 block 的显存（300 token 的 KV）。
第 4 步：A 的 KV Cache 重建完成，从第 300 个 token 之后继续生成。
```

对比结论：**Swap 恢复快（2 ms）但耗 CPU 内存和拷贝带宽；Recompute 省资源（不占 CPU 内存、无拷贝）但恢复慢（30 ms）且浪费算力（300 token 重新 prefill）。** 在真实系统中，如果 CPU 内存充足通常倾向 swap，否则用 recompute；也可以两者结合（vLLM 的策略扩展点）——比如对 KV 量大的长序列用 recompute（省 CPU 内存），对 KV 量小的短序列用 swap（拷贝快）。还有一个实际权衡：swap 需要预留 CPU 内存，而 recompute 需要在恢复时"抢"算力（可能再次挤占其他请求），两种方案的选择本质是"带宽/内存成本"与"算力成本"之间的取舍。

> **面试一句话总结**：Preemption 是显存过载时的"优雅降级"机制，Scheduler 在高水位触发下把部分序列整体换出（swap 到 CPU，恢复快但耗带宽和 CPU 内存）或丢弃重算（recompute，省资源但恢复慢），从而保证新请求不被饿死、系统不 OOM。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 Preemption 实现在 `vllm/v1/core/sched/scheduler.py` 的 `_preempt_request` + `_free_request_blocks`**。v1 架构下默认走 **recompute**（`num_computed_tokens = 0` 后从 prompt 重算），不再像旧版那样 swap 到 CPU 显存：

```python
def _preempt_request(self, request: Request, timestamp: float, ...) -> None:
    """Preempt a request and put it back to the waiting queue.
    NOTE: The request should be popped from the running queue outside of this method.
    """
    assert request.status == RequestStatus.RUNNING, "Only running requests can be preempted"
    self._free_request_blocks(request)      # ① 释放它的 KV blocks
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED    # ② 状态：running → PREEMPTED
    request.num_computed_tokens = 0             # ③ 已算 token 清零 → recompute
    if request.spec_token_ids:
        request.spec_token_ids = []             # ④ 清掉投机草稿 token
    request.num_preemptions += 1                # ⑤ 抢占计数（可配策略）

def _free_request_blocks(self, request: Request):
    """释放请求的 KV blocks 回缓存（被抢占/完成/取消时调用）。"""
    blocks = self.kv_cache_manager.pop_blocks_for_free(request.request_id)
    if blocks:
        self._drain_deferred_frees()            # 延迟释放队列排空
```

**这段代码关键在哪**：`_preempt_request` 就是 recompute 抢占的完整流程——①释放 KV blocks（显存立即回收）；②状态置 `PREEMPTED`（v1 里被抢占请求放回 waiting，没有独立的 swapped 队列）；③**`num_computed_tokens = 0` 是 recompute 的本质**（下次调度从 prompt 重算全部 KV）；④清投机草稿；⑤`num_preemptions` 计数。对比旧版（swap 到 CPU + 换回），v1 默认 recompute 更简单（省 CPU 内存、无拷贝），代价是恢复时重算——这正好呼应"swap vs recompute"的取舍。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Preemption 的两种模式（swap / recompute）在 v0.25.x 中仍是标准机制，未见根本性变更，但它的**触发频率在新架构下趋于下降**：随着 FP8 KVCache（KV 显存占用减半，见第 3 点）和更精细的 KV block 管理落地，显存紧张的概率降低，抢占作为"兜底手段"的角色更明确。另一个演进是抢占策略的可配置化（与第 8 点一致）：新版把"选谁当受害者、用 swap 还是 recompute"抽象成策略对象，可以按部署场景调优（例如对 KV 量大的长序列倾向 recompute 省 CPU 内存、对短序列倾向 swap 快恢复）。在 PD 分离架构（见第 16 点）中，decode 节点显存压力更可控，抢占需求进一步减少。面试时可以提："抢占的 swap/recompute 二选一逻辑没变，变的是显存更宽裕（FP8 KV）后它更少触发，以及受害者选择可以按策略定制。"

---

# 四、Attention Kernel 相关

## 10. FlashAttention

### 1. 现有问题：为什么要提出 FlashAttention

FlashAttention 要解决的核心问题是：**标准 Attention 实现的内存复杂度是 O(N²)（N 为序列长度），导致长序列场景下显存爆炸、计算效率极低。** 标准的 attention 计算是：先算出完整的 attention score 矩阵 S = QKᵀ（形状 N×N），对它做 softmax 得到 P，再乘 V 得到输出 O。这个 N×N 的中间矩阵需要被完整写入显存（HBM），对于长序列来说开销巨大：N=4096 时 S 是 4096×4096 的矩阵，N=32768 时仅一个头的 score 矩阵就有 32768×32768×2 字节 ≈ 2 GB，多层叠加直接爆显存。更糟糕的是，GPU 的显存带宽远低于计算速度（HBM 带宽是算力的瓶颈），把大矩阵写进 HBM 再读出来做 softmax，绝大多数时间都花在数据搬运上，而非真正的计算。

除了显存问题，还有**融合度问题**。标准的实现把 attention 拆成 QKᵀ → softmax → PV 三个独立的 kernel，每个 kernel 都要读一遍 Q/K/V（甚至中间矩阵），产生多次显存往返；softmax 本身还有数值稳定性问题——标准实现要先算每行的 max 再做减法（防止 exp 溢出），这需要先完整算一遍 S 才知道 max，进一步强制了"必须物化完整矩阵"的流程。在推理场景中，decode 阶段的 attention 是 query 长度为 1 的形态（见第 6 点），每次都把所有 KV 读出来，如果不能用高效 kernel 做流式处理，decode 每步的访存量会非常大。总之，attention 是整个 Transformer 里计算量最大、最值得优化的部分，FlashAttention 正是针对"O(N²) 显存 + 多次 kernel 往返 + 带宽瓶颈"这三个问题提出的。

### 2. 方法论：FlashAttention 是怎么实现的

FlashAttention 的核心思想是**分块（tiling）+ 在线 softmax（online softmax）+ kernel 融合**，把 attention 的显存复杂度从 O(N²) 降到 O(N)（时间仍是 O(N²) 计算），同时大幅减少 HBM 访问。核心数据结构是三个寄存器/SRAM 中的累积量：**running max（每行当前的全局最大值 m）**、**running sum（每行 exp 的累积和 l，即 softmax 分母）**、**输出累积 O**。整个算法按以下步骤逐步操作（以单头、单 query 行、KV 长度 N、块大小 B 为例）：

**第 1 步（分块）**：把 Q 按行分块（每块 B 行 query）、K/V 按列分块（每块 B 列 key/value），块大小 B 由 GPU 片上 SRAM 容量决定（例如 64×64 或 128×128），保证每一对 Q 块和 K/V 块都能整体放进 SRAM，中间结果不落 HBM。

**第 2 步（逐块计算，滚动更新）**：对每个 K/V 块（设当前是第 j 块，包含 key K_j 和 value V_j），取出对应的 Q 块 Q_i，在 SRAM 内做以下操作：

- **(a) 计算分数块**：$S_{ij} = Q_i K_j^T$，得到 (B, B) 的分数矩阵；
- **(b) 更新 running max**：取当前块内每行的最大值 $m_{ij} = \text{rowmax}(S_{ij})$，与之前累积的 m 合并：$m^{\text{new}} = \max(m, m_{ij})$；
- **(c) 修正之前块的贡献**：由于全局最大值变大了，之前累积的输出和分母都要按比例缩小：$O \leftarrow O \times e^{m - m^{\text{new}}}$，$l \leftarrow l \times e^{m - m^{\text{new}}}$（若最大值没变，修正因子为 1，无操作）；
- **(d) 计算当前块的概率和贡献**：$P_{ij} = e^{S_{ij} - m^{\text{new}}}$，然后 $l \leftarrow l + \text{rowsum}(P_{ij})$，$O \leftarrow O + P_{ij} V_j$；
- **(e) 继续下一个块**，直到所有 K/V 块处理完。

**第 3 步（归一化输出）**：全部块处理完后，每个位置的输出 = $O / l$。此时 m、l、O 都已经是**全局正确**的（不依赖任何预扫描），softmax 数值稳定（减去了全局最大 m）。

这个过程的关键在于：**running max 和 running sum 是滚动维护的，每个块只被读一次**，Q/K/V 各从 HBM 读一遍、O 写一遍，中间矩阵 S 和 P 从不落 HBM，全部在 SRAM 内完成——这就是"Flash"的含义：把访存量降到理论下界（读一遍输入、写一遍输出），让计算速度（而非带宽）成为瓶颈。kernel 融合体现在：QKᵀ、softmax 统计、PV 累加原本是三个独立 kernel（各自读写一遍 HBM），FlashAttention 把它们融合进一个 kernel 里，消除了中间矩阵的所有 HBM 往返。

FlashAttention 还有两个重要变体/扩展。第一是 **FlashAttention-2**：改进了并行策略（在序列维度上并行分配线程块，减少 shared memory 同步和负载不均），使不同序列长度下 GPU 利用率更高，实测比第一代快约 2 倍。第二是 **分块因果掩码**：对 causal attention，每个输出 token 只能看到自己及之前的 token，FlashAttention 通过"上三角块直接跳过或掩码"实现，跳过的块不参与计算，长序列下可以省近一半的注意力计算量。FlashAttention 本身是底层 kernel 库，vLLM 并不直接调用它，而是通过 XFormers、FlashInfer 等"attention backend"间接使用（见第 11 点）——它已经成为现代推理/训练引擎 attention 计算的性能基石。

### 3. 具体数值样例

用一个可手算的小例子演示 online softmax 的滚动更新过程。假设单头、head_dim = 1（简化）、块大小 B = 4，某 query 行与 8 个 key 的分数（即 S 的一行）为：

```text
scores = [2, -1, 3, 0, 1, 5, -0.5, 1.5]
```

**标准 softmax 的结果（作为正确性基准）**：整行最大值 m = 5。计算 e^(s_i − 5) 再求和：e^−3+e^−6+e^−2+e^−5+e^−4+e^0+e^−5.5+e^−3.5 ≈ 0.0498+0.0025+0.1353+0.0067+0.0183+1+0.0041+0.0302 = **1.2469**。最终概率 = e^(s_i−5)/1.2469。

**FlashAttention 的逐步演算（分 2 块：块 1 = [2, -1, 3, 0]，块 2 = [1, 5, -0.5, 1.5]）：**

```text
初始化：m = -∞，l = 0，O = 0

处理块 1（scores [2, -1, 3, 0]）：
  (a) 块内最大值 m_1 = 3
  (b) 更新 running max：m_new = max(-∞, 3) = 3
  (c) 修正：修正因子 e^(-∞ - 3) → 0（首次无历史，O、l 保持 0）
  (d) e^(s - 3) = [e^-1, e^-4, e^0, e^-3] = [0.3679, 0.0183, 1, 0.0498]
      l = 0 + 0.3679+0.0183+1+0.0498 = 1.4360
      O 累积（乘 V 的部分，这里略去具体 V 值，只记录 l 和 m 的变化）

处理块 2（scores [1, 5, -0.5, 1.5]）：
  (a) 块内最大值 m_2 = 5
  (b) 更新 running max：m_new = max(3, 5) = 5   ← 全局最大值变大了！
  (c) 修正之前块的贡献：修正因子 e^(3 - 5) = e^-2 ≈ 0.1353
      l 修正为 1.4360 × 0.1353 ≈ 0.1943
      O 也整体乘以 0.1353（对应 V 的加权累加同样缩放）
  (d) e^(s - 5) = [e^-4, e^0, e^-5.5, e^-3.5] = [0.0183, 1, 0.0041, 0.0302]
      l = 0.1943 + 0.0183+1+0.0041+0.0302 = 0.1943 + 1.0526 = 1.2469

最终：l = 1.2469，与标准 softmax 的分母完全一致 ✓
```

可以看到：块 1 处理时并不知道块 2 有更大的分数 5，但通过"running max 从 3 更新到 5 时，把之前累积的 l 和 O 乘以修正因子 e^(3−5)"，最终结果与标准 softmax 逐位一致——**这就是 online softmax 的核心正确性保证**。整个过程 S 和 P 矩阵从未写入 HBM，只流过 SRAM。

再看显存收益的量化对比：假设序列长度 N = 4096，单头、单样本，FP16。标准实现：score 矩阵 S = QKᵀ 是 4096×4096，FP16 下占 4096×4096×2 ≈ 32 MB 显存；若 N = 32768，S 占 2 GB，而且这还只是**一个注意力头**——一个 32 头的模型就要 64 GB，直接爆掉单卡显存。时间上，标准实现要写 S 到 HBM（32 MB）、读 S 做 softmax（32 MB）、写 P（32 MB）、读 P 和 V 算输出，光中间矩阵的 HBM 往返就有约 100+ MB 的读写，而 FlashAttention 只读 Q/K/V（3×4096×128×2 ≈ 3 MB）写 O（1 MB），中间矩阵零落盘，HBM 访问量减少一个数量级以上。decode 场景同理：标准实现每步要把整个 K、V 读出来做 1×4096 的 attention，FlashAttention 的 decode kernel 分块流式读取 KV、配合 online softmax，仍只需把 KV 读一遍（信息量下界），省去了中间 score 矩阵的写读，per-step 时间可以再省 10%~20%。

> **面试一句话总结**：FlashAttention 通过分块计算 + 在线 softmax + 单 kernel 融合，把 attention 的显存复杂度从 O(N²) 降到 O(N)，中间矩阵永不落 HBM，HBM 访问量降一个数量级，让计算而非带宽成为瓶颈，是长序列训练与推理的性能基石。

### 4. 核心代码（源码）

**FlashAttention 本身是外部 kernel 库（`vllm_flash_attn` / flash-attn），vLLM 通过 `vllm/v1/attention/backends/flash_attn.py` 的 `FlashAttentionBackend` 接入它**——这个 backend 的能力声明（`supports_*`）决定了 FlashAttention 在什么条件下被选中：

```python
class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flash-attn"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        # FlashAttentionImpl.forward() 内部调 vllm_flash_attn 的
        # varlen_fwd / prefill / decode kernel（分块 + online softmax 在 C++/CUDA 里）
        return FlashAttentionImpl

    @staticmethod
    def supports_head_size(cls, head_size: int) -> bool: ...
    @staticmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool: ...
        # 如 FP8 KV cache 需要 FA backend 支持
    @staticmethod
    def supports_sliding_window(cls) -> bool: ...
    @staticmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...
```

**这段代码关键在哪**：FlashAttention 在 vLLM 里的角色是"一个 attention backend"——`FlashAttentionImpl.forward()` 是入口，真正做分块 + online softmax 的 kernel 在 `vllm_flash_attn`（C++/CUDA 实现，`csrc/` 下）；`supports_kv_cache_dtype` 等能力声明让选择器决定"这个请求的形态能不能用 FA"。**面试时要分两层讲**：算法层（分块 + online softmax，第 2 节已详述）和接入层（vLLM 通过 backend 接口调用它，第 11 点）。FA-1 → FA-2 → FA-3 的演进在 kernel 层，backend 接口不变。

### 5. 新版 vLLM 特性（v0.25.x 演进）

FlashAttention 是"思想长期留存、实现持续更替"的另一个典型。FlashAttention 本身已经迭代到第三代：**FlashAttention-3** 针对 Hopper（H100）架构优化，利用 TMA（Tensor Memory Accelerator）和异步流水线进一步压榨 HBM 带宽利用率，比 FA-2 再快约 1.5~2 倍；而 v0.25.0 移除 legacy PagedAttention 后，**FlashAttention 系的 paged kernels 与 FlashInfer 的 paged kernels 一起成为 KV attention 的标准执行路径**（见第 1 点）。这意味着面试时讲 FlashAttention 要能衔接两层：**算法层**（分块 + online softmax 的思想，永不过时）和**实现层**（FA-1 → FA-2 → FA-3 的硬件适配演进，以及它如何作为 attention backend 被 vLLM 调用）。如果面试官问"FlashAttention 之后还有什么"，可以提 FP8 量化、与 KV 分页寻址的融合、以及针对新硬件（Blackwell 等）的 kernel 重构方向。

---

## 11. FlashInfer / Attention Backend

### 1. 现有问题：为什么要抽象 Attention Backend

Attention Backend（注意力后端）要解决的问题是：**vLLM 等推理引擎需要支持多种硬件、多种 attention 实现，并且要能根据运行时的实际形态（prefill / decode / 长序列 / 分页 KV / 不同 GPU 架构）自动选择最高效的 kernel——这个"选 kernel"的复杂性必须被抽象出来管理。** 直接原因有几点。第一，attention 的实现选择很多：FlashAttention、FlashInfer、XFormers 的 memory-efficient attention、cuDNN 的 attention、Triton 手写 kernel、以及各厂商硬件上的专用实现（如昇腾、AMD、寒武纪），各有适用场景和性能差异，引擎不可能只绑死一种。第二，不同阶段的最优实现不同：prefill 需要处理完整矩阵的高并行 kernel，decode 需要 query=1 的访存优化 kernel，两者形态差异巨大，甚至同一种后端内部也要分派。第三，PagedAttention 的存在让情况更复杂：KV 是分页存储的（物理上不连续），传统 kernel 假设 KV 连续，而 PagedAttention 要求 kernel 能通过 Block Table 间接寻址——不同后端对"分页 KV"的支持程度不同，引擎必须知道哪个后端支持分页、支持到什么程度。

如果不做这个抽象，引擎代码里就会到处是 `if backend == flash_attn: ... elif backend == flashinfer: ...` 的硬编码分支，新增一个后端要改一堆代码，而且很容易选错 kernel 导致性能暴跌甚至结果错误。vLLM 的答案是：定义一个统一的 **Attention Backend 接口**，所有 kernel 实现都实现同一套接口（forward 方法 + 能力声明），引擎只面向接口编程；运行时根据**模型配置 + 硬件 + 请求形态**通过一个选择器（AttentionSelector）自动挑选并缓存最合适的后端实例。

### 2. 方法论：vLLM 的 Attention Backend 是怎么实现的

vLLM 的 attention 后端体系由三层组成。**第一层是统一的抽象接口**：vLLM 定义 `AttentionBackend` 基类，要求每个后端实现 `forward()`（执行 attention 计算），并声明一组能力属性作为"自描述"——`supports_prefill`（是否支持 prefill 形态）、`supports_decode`（是否支持 decode 形态）、`supports_paged_attention`（是否支持分页 KV，即能否通过 Block Table 间接寻址）、`supports_cuda_graph`（是否可与 CUDA Graph 配合）、`is_available()`（当前硬件/环境是否可用）。这组能力声明是选择器的核心依据：**引擎只面向接口编程，从不直接依赖某个具体 kernel 库**。

**第二层是多个后端实现**：目前 vLLM 主要支持 FLASH_ATTN（基于 FlashAttention 的 PagedAttention kernel）、FLASH_INFER（基于 FlashInfer）、XFORMERS、ROCM_FLASH（AMD 版 FlashAttention）、TRITON 等；每个后端内部再根据 attention 的具体形态（prefill 的 MHA/GQA/MQA、decode 的分页与否）分派到不同的底层 kernel。以 FlashInfer 为例，它的核心机制是 **JIT（Just-In-Time）编译**：把 attention 的模板参数（head 数、head_dim、块大小、是否 GQA、是否因果掩码、KV 布局）在运行时编译成**特化 kernel**，为每种请求形态生成最优代码，而不是用"一个通用 kernel 打天下"；它的 PagedAttention 实现直接支持 Block Table 间接寻址、KV 缓存分页、跨序列共享，和 vLLM 的 KV Cache Manager 无缝对接，还支持 ragged tensor（长度不齐的 batch）。vLLM 中，prefill 和 decode 会被分派到 FlashInfer 的不同 kernel 模板（如 `BatchPrefillWithPagedKVCache`、`BatchDecodeWithPagedKVCache`）。

**第三层是运行时选择器**：`AttentionSelector` 在模型初始化时按以下步骤挑选后端——**第 1 步**，枚举所有已注册后端的 `is_available()`，过滤掉当前环境不可用的（如 ROCM_FLASH 在 NVIDIA 卡上不可用）；**第 2 步**，检查模型配置约束（如某些模型需要特定的 mask 支持、某些后端的头数限制），进一步过滤；**第 3 步**，若用户设置了 `VLLM_ATTENTION_BACKEND` 环境变量则直接强制使用指定后端（跳过前两步的自动选择，用于调试/对比）；**第 4 步**，从剩余候选中按优先级表（如 FLASH_INFER 或 FLASH_ATTN 优先）选出唯一后端，**缓存**这个选择结果——之后所有请求都复用该后端实例，不再重复判断。这套抽象的价值在于：当出现新 GPU 架构或新 kernel 库时，只需要**新增一个实现 `AttentionBackend` 接口的后端类**，引擎主体一行不改。

### 3. 具体数值样例

假设在 A100 上部署一个带 GQA（grouped-query attention，比如 32 个 Q head、8 个 KV head、head_dim=128）的模型，同时有 prefill 和 decode 两类请求混跑。逐步演算选择器和 kernel 分派过程：

**第一步（选择器）**：模型初始化时，`AttentionSelector` 检查各后端——XFORMERS 可用但优先级低于 FLASH_INFER；ROCM_FLASH 的 `is_available()` 在 NVIDIA 上返回 False 被过滤；FLASH_INFER 的 `is_available()` 为 True 且 `supports_paged_attention=True`、`supports_cuda_graph=True`，最终选中 FLASH_INFER 并缓存。

**第二步（kernel 分派）**：运行时，假设 batch 里有 8 个 prefill 请求（各 512 token）+ 32 个 decode 请求。prefill 请求走到 FlashInfer 的 `BatchPrefillWithPagedKVCache` kernel（模板参数：head_dim=128、块大小 64、GQA ratio 4、因果掩码、分页 KV 布局），JIT 编译出特化 kernel；decode 请求走到 `BatchDecodeWithPagedKVCache` 的 1×N kernel（query 长度 1、KV 长度 = 各请求已生成长度，分页寻址）。

**第三步（性能对比）**：FlashInfer 的 JIT 特化 kernel 让 prefill 部分达到接近 FlashAttention-2 的算力利用率（约 60%~80% TFLOPs 效率），decode 部分通过分块读取 KV + 在线 softmax 把每步访存压到"只读一遍 KV"。如果不用 FlashInfer 而用通用 naive kernel：prefill 需要把 8×512×512 的 score 矩阵（约 4 MB/头）落盘再读，decode 的 1×N attention 每次把全部 KV 完整读入寄存器（溢出时反复换出），实测同 batch 下吞吐可能低 30%~50%。用更直观的数字：假设 FlashInfer decode kernel 下 32 个 decode 请求一步耗时 12 ms（访存受限），naive 实现因为中间矩阵落盘和低效读 KV，一步耗时约 20 ms——同样的 32 个请求，每步多花 8 ms，一个 500 token 的生成就要多花 4 秒。而这一切对上层透明：引擎只调 `backend.forward()`，kernel 的选择、特化都由 FlashInfer 在运行时完成，这就是 Attention Backend 抽象的价值——**把"用什么 kernel、怎么选"的复杂度封装起来，让引擎专注于调度和推理逻辑**。

> **面试一句话总结**：Attention Backend 是 vLLM 对 attention kernel 的统一抽象——各后端（FlashAttention / FlashInfer / XFormers 等）实现同一接口并声明能力，运行时选择器按硬件和请求形态自动挑选最优实现；其中 FlashInfer 通过 JIT 特化和统一分页 attention，覆盖 prefill / decode 全形态，是 vLLM 高性能推理的关键后端之一。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 Attention Backend 抽象在 `vllm/v1/attention/backend.py`**——`AttentionBackend` 用 ABC 定义统一接口，**核心是"能力自描述"**（选择器靠这些方法判断哪个后端可用）：

```python
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @staticmethod
    def get_name() -> str:
        """返回后端名（如 "flash-attn" / "flashinfer"），供选择器匹配。"""

    @staticmethod
    def get_impl_cls() -> type["AttentionImplBase"]:
        """返回该后端的 kernel 实现类（forward 入口）。"""

    @staticmethod
    def get_builder_cls():
        """返回 metadata 构建器（把请求组织成 kernel 需要的输入）。"""

    # ↓↓↓ 能力自描述：选择器据此判断"这个后端能不能用" ↓↓↓
    @staticmethod
    def supports_head_size(head_size: int) -> bool: ...      # 支持的头维度
    @staticmethod
    def supports_dtype(dtype: torch.dtype) -> bool: ...      # 支持的精度
    @staticmethod
    def supports_kv_cache_dtype(kv_cache_dtype) -> bool: ... # 支持的 KV 精度（FP8?）
    @staticmethod
    def supports_block_size(block_size) -> bool: ...         # 支持的 block size
    @staticmethod
    def supports_sink() -> bool: ...                         # 是否支持 sink cache
    @staticmethod
    def supports_mm_prefix() -> bool: ...                    # 多模态前缀
```

**这段代码关键在哪**：`AttentionBackend` 的**能力自描述设计**是 backend 抽象的核心——每个后端（`vllm/v1/attention/backends/flash_attn.py` / `flashinfer.py` / `triton_attn.py` 等）实现 `get_name()` + 一堆 `supports_*()`，`AttentionSelector`（`vllm/v1/attention/selector.py`）在初始化时枚举所有后端、按 `supports_head_size`/`supports_dtype`/`supports_block_size` 等过滤出可用的，再按优先级选唯一后端并缓存。**新增一个后端 = 实现 `AttentionBackend` 接口 + 在 registry 注册**（`backends/registry.py`），引擎主体不改——这就是 v0.25.0 敢于移除 legacy PagedAttention 的架构前提。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Attention Backend 抽象在 v0.25.x 中不仅没被削弱，反而**地位更高了**：v0.25.0 移除 legacy PagedAttention 的底气，正是"FlashAttention 系 / FlashInfer 的 paged kernels 已经是标准路径"——也就是说，attention 执行现在完全由这些后端承担，backends 抽象成为 KV attention 的唯一入口。FlashInfer 侧也在持续演进：JIT 特化覆盖的形态更广（prefill / decode / 长序列 / GQA / 分页），并深度配合 CUDA Graph 与新架构（FP8 KV 等）。另一个值得提的点是 **DSpark / DFlash**：vLLM 较新版本把一些经过验证的社区 kernel 主线程化（mainlined），进一步丰富 backend 选择。面试时可以强调："backends 抽象 + 选择器 + JIT 特化"这套设计让 vLLM 能在新硬件/新 kernel 出现时只加一个后端类——这也是 v0.25 敢于删掉旧 kernel 的架构前提。

---

## 12. CUDA Graph

### 1. 现有问题：为什么要用 CUDA Graph

CUDA Graph 要解决的问题是：**GPU kernel 启动（launch）的 CPU 开销在短小的 kernel 上会严重拖慢推理速度。** 现代深度学习推理的一个 decode step 由几十个 kernel 组成（每个 Transformer 层里有多个矩阵乘、attention、归一化、激活函数 kernel），而每个 kernel 的 launch 都要经过 CPU 侧的一系列操作：检查参数、分配/获取 stream、把 kernel 参数打包、向 GPU 驱动提交命令，这串过程大约要花费几微秒到几十微秒的 CPU 时间。对 prefill 这种大 kernel 来说，launch 开销占比可以忽略；但 decode 的每个 kernel 本身只运行几十微秒（甚至更短），launch 开销就变得非常可观——一个 30 层模型、每层 5 个 kernel，就是 150 次 launch，仅 CPU 侧的 launch 开销就可能占每步总时间的一半以上，GPU 大部分时间在"等 CPU 喂命令"，利用率被严重拉低。

更麻烦的是，decode step 的 kernel 序列是**高度固定**的：batch 里的请求数量不变时，每一步执行的 kernel 种类、顺序、形状几乎完全相同（只是数据变了）。如果每次 step 都重新走一遍"CPU 逐 kernel 提交"的流程，就是在重复做同样的工作。CUDA Graph 正是针对这个模式提出的：把一串 kernel 的启动顺序**预先捕获**成一个 graph（有向无环图，记录 kernel 及其参数、依赖关系），之后每次执行只需一次 `graphLaunch` 调用，GPU 自己按图依次调度所有 kernel，CPU 的 launch 开销从"每 kernel 一次"变成"整图一次"，可以下降一到两个数量级。在 vLLM 这类推理引擎中，CUDA Graph 是让 decode 达到接近硬件极限吞吐的关键技术之一。

### 2. 方法论：CUDA Graph 在 vLLM 里是怎么实现的

CUDA Graph 的使用分两个阶段：**捕获（capture）** 和 **回放（replay）**。**捕获阶段**，CPU 把目标 kernel 序列在"图模式"下执行一遍——CUDA 在捕获期间不真正执行计算，而是记录每个 kernel 的启动参数、网格维度、共享内存大小和依赖关系，最终产出一个 CUDA Graph 对象。**回放阶段**，每次只需调用一次 `cudaGraphLaunch`，GPU 硬件调度器按照图中记录的依赖关系自动依次启动所有 kernel，CPU 不再逐个介入。

vLLM 里实现 CUDA Graph 有一个核心难点：**kernel 的形状必须在捕获时固定**，因为捕获时记录的 kernel 网格大小、共享内存等参数在回放时不能改变——而实际请求的 batch size 是动态变化的（Continuous Batching 下每步的请求数都不同）。vLLM 的解决方案是**按 batch size 分桶捕获多张 graph**，具体步骤如下：

**第 1 步（定义档位）**：系统预先定义一组 batch size 档位，例如 1, 2, 4, 8, 16, 32, 64, 128……直到 `max_num_seqs`。档位覆盖从 1 到最大并发数的所有 2 的幂。

**第 2 步（逐档捕获）**：对每个档位分别捕获一张 CUDA Graph：用该档位大小（如 8）、用假数据/占位数据跑一遍完整的 decode step 前向，在捕获模式下记录所有 kernel。捕获得到的 graph 的"形状"就是 8 个序列。

**第 3 步（运行时填充对齐）**：调度器把当前 batch 的请求**填充（pad）到最近的档位**——比如实际有 7 个请求，就填充到 8 的档位执行，多出的第 8 个 slot 用掩码（mask）屏蔽掉，不产生输出、不更新状态。这样无论 batch 怎么变化（如 3 个请求 → 4 档、5 个请求 → 8 档），都能命中预先捕获好的 graph，无需重新捕获。

**第 4 步（显存地址固定）**：graph 回放要求 kernel 使用的显存地址与捕获时一致，所以 vLLM 会为 graph 执行预留固定的显存缓冲（graph memory pool），保证地址稳定。

CUDA Graph 在 vLLM 中有几个配套细节。第一，**不是所有 kernel 都适合进 graph**：CUDA Graph 捕获期间不能有动态内存分配、不能有 CPU 同步（如 `cudaMemcpy` 的同步等待、`cudaStreamSynchronize`），所以 vLLM 把"必须 CPU 介入"的操作（采样、KV 管理、logits 后处理等）放在 graph 之外，graph 只包住纯计算部分。第二，**与 attention 后端配合**：不同 attention 后端（FlashInfer 等）要声明自己支持 CUDA Graph（`supports_cuda_graph`），vLLM 才能把该后端的 kernel 纳入 graph。第三，**显存成本**：每张 graph 要预留固定缓冲，batch 档位越多、预留的显存越大（vLLM 有 `gpu_memory_utilization` 和 graph 相关参数来平衡），所以档位不能无限制细分。开启 CUDA Graph 后，vLLM 的 decode per-step CPU 开销从毫秒级降到几十微秒级，单步延迟和吞吐都显著改善。

### 3. 具体数值样例

假设一个 30 层模型，decode step 有约 150 个 kernel（30 层 × 每层约 5 个 kernel）。先算不开 CUDA Graph 的开销：每个 kernel launch 平均 CPU 开销 5 µs（含参数打包、驱动提交），150 个 kernel 就是 750 µs 的 launch 开销；假设 GPU 实际计算每步 3 ms，那么每步总时间约 3.75 ms，其中 20% 是纯 CPU launch 等待。如果 kernel 更小（比如小模型或 batch 小），GPU 计算只有 1 ms，launch 开销 0.75 ms 就占了 43%——GPU 近一半时间空转等命令。

开启 CUDA Graph 后，逐步演算：**第 1 步**，启动时捕获"batch=8 档位"的 graph（用 8 个占位序列跑一遍完整前向，捕获 150 个 kernel）；**第 2 步**，运行时来了 7 个请求，填充到 8 档位执行；**第 3 步**，每步只做一次 `graphLaunch`，CPU 开销约 10~20 µs（一次提交整个图）。同样 150 个 kernel 的 step，CPU 开销从 750 µs 降到约 15 µs，每步总时间从 3.75 ms 降到约 3.02 ms，吞吐提升约 24%；如果 GPU 计算本身更快（1 ms），则从 1.75 ms 降到 1.02 ms，提升约 71%——**GPU 计算占比越小，CUDA Graph 收益越大**，这正是访存密集的 decode 阶段受益最大的原因。再看显存成本：假设每个档位的 graph 预留约 200 MB 缓冲（含输入/输出/KV 指针等），档位从 1 到 128 共 8 张图，合计约 1.6 GB——这是可以接受的代价，换取的是每步 700+ µs 的 CPU 开销节省。vLLM 实际生产中，开启 CUDA Graph 是默认行为，配合 Continuous Batching 的动态 batch 分桶，让系统在请求数变化时依然保持低 launch 开销的高吞吐。

> **面试一句话总结**：CUDA Graph 把 decode step 中固定的 kernel 序列预先捕获成图、之后一次 launch 整体回放，把 CPU 侧 per-kernel 启动开销降低一到两个数量级；vLLM 通过按 batch size 分桶捕获多张图 + 填充对齐，让动态 batch 也能命中预捕获图，是 decode 高吞吐的关键工程优化。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的 CUDA Graph 实现在 `vllm/compilation/cuda_graph.py` 的 `CUDAGraphWrapper` + `vllm/v1/cudagraph_dispatcher.py`**——`CUDAGraphWrapper` 用装饰器把"该进 graph 的 forward 调用"包起来，按 batch size 分桶捕获/回放：

```python
class CUDAGraphEntry:
    """一张捕获好的 CUDA Graph（对应一个 batch size 档位）。"""
    graph: torch.cuda.CUDAGraph
    input_buffers: dict[str, torch.Tensor]   # 固定输入缓冲（地址稳定）
    output_buffers: dict[str, torch.Tensor]  # 固定输出缓冲

class CUDAGraphWrapper:
    """把被装饰的函数（如 model forward）按 batch size 分桶捕获/回放。"""
    def __init__(self, graph_mode: CUDAGraphMode, ...):
        self.graphs: dict[int, CUDAGraphEntry] = {}   # batch_size → 图

    def __call__(self, *args, **kwargs):
        """第一次调用（捕获模式）→ 记录 kernel；
        之后（回放模式）→ 填充到最近档位 + cudaGraphLaunch 一次回放。"""
        batch_size = ...   # 从输入推断当前 batch size
        entry = self.graphs.get(batch_size)
        if entry is None:
            # 该档位未捕获：进入 graph_capture 上下文捕获新图
            with graph_capture() as graph_capture_ctx:
                output = self.forward_func(*args, **kwargs)   # 捕获模式跑一次
            entry = CUDAGraphEntry(graph=..., input_buffers=..., ...)
            self.graphs[batch_size] = entry
            return output
        else:
            # 命中已捕获图：输入拷进固定缓冲 → 一次 launch → 输出拷出
            entry.input_buffers.copy_(args)
            entry.graph.replay()          # ← 关键：一次 cudaGraphLaunch 回放全部 kernel
            return entry.output_buffers
```

**这段代码关键在哪**：`CUDAGraphWrapper.__call__` 体现了 CUDA Graph 的全部机制——①**按 batch_size 分桶**（`self.graphs: dict[int, CUDAGraphEntry]`，每个档位一张图）；②**捕获**（第一次遇到某 batch size 时 `graph_capture()` 上下文里跑一次 forward，记录 kernel）；③**回放**（命中时输入拷进固定缓冲 → `graph.replay()` 一次 launch → 输出拷出）；④固定 input/output buffers 保证地址稳定。v1 的 `cudagraph_dispatcher.py` 负责决定"这步能不能走 graph、走哪张图"（如投机解码、PP 等场景的图选择）。

### 5. 新版 vLLM 特性（v0.25.x 演进）

CUDA Graph 在 v0.25.x 中的地位进一步提升，最直接的证据是 **dynamic speculative decoding 兼容 full CUDA graphs**（v0.25.0，PR #45953）：之前投机解码因为草稿长度动态变化，往往只能走"部分 graph + 动态路径"的组合，v0.25.0 打通了动态投机与完整 CUDA Graph 的配合，说明 CUDA Graph 的覆盖面在扩大而不是收缩。另一个方向是 **MRv2 的 CUDA-graph-native 设计**：MRv2 从设计上就面向 CUDA Graph 优化（graph 捕获、分桶、显存池固定地址等作为执行链的一等公民），而不是像 V1 那样把 graph 当"事后优化"叠加。面试时可以讲两层：CUDA Graph 的**机制**（捕获/回放/分桶/填充，不变）和**演进**（从 decode 核心到投机解码全覆盖、在 MRv2 中成为原生执行方式）。

---

# 六、模型执行架构

## 13. vLLM Engine Architecture

### 1. 现有问题：为什么需要理解 vLLM 的整体架构

理解 vLLM 的整体架构，是为了回答面试中一个高频问题："一个请求从发起到返回，在 vLLM 内部走了哪些环节？"以及"vLLM 为什么能做得这么快？"。如果没有整体架构的认知，前面学的 PagedAttention、Scheduler、KV Cache Manager、Attention Backend、CUDA Graph 等知识点就是散落的碎片——面试官问任何一个"组件之间如何协作"的问题都会答不上来。vLLM 之所以是"工程上最成功的推理引擎之一"，正是因为它的分层架构把上面这些机制组织成了一个职责清晰、可扩展的整体：每一层只做一件事、只依赖下一层，新增硬件/模型/调度策略时不需要推翻重来。

如果不分层、把所有逻辑揉在一起，会出现典型的"意大利面条"问题：调度决策和 kernel 执行耦合、资源管理散落在各处、无法单元测试、也无法针对不同硬件插拔组件。vLLM 的分层设计（Client → API Server → Engine → Scheduler → KV Cache Manager → Model Runner → GPU Worker → Attention Backend → GPU）让每一层边界清晰：上层决定"做什么"，下层决定"怎么做"，中间通过数据结构（Request、Sequence、SchedulerOutput）传递信息。这也是 vLLM 能从单卡 demo 演进到支持多卡张量并行、多机流水线并行、PD 分离等复杂部署形态的基础。

### 2. 方法论：vLLM 的 Engine 架构是怎么分层的

vLLM 的整体架构从上到下大致是这样一个调用链（不同版本组件名可能有变化，但逻辑分层稳定）：

```text
                    Client
                      ↓
                  API Server
                      ↓
                   Engine
                      ↓
                 Scheduler
                      ↓
              KV Cache Manager
                      ↓
                 Model Runner
                      ↓
                 GPU Worker
                      ↓
              Attention Backend
                      ↓
                    GPU
```

**第一层 API Server**（如 OpenAI 兼容的 `/v1/chat/completions` 端点）：接收 HTTP 请求，做协议解析、鉴权、参数校验，把请求转换成内部的 `Request` 对象，通过 `Engine.add_request()` 提交给 Engine；同时以异步方式把生成结果流式返回给客户端。**第二层 Engine**（LLMEngine / AsyncLLMEngine）：是整个系统的"中央处理器"，它维护一个**事件循环**，每轮循环按以下步骤操作：**第 1 步**，调用 Scheduler 的 `schedule()` 得到本轮要执行的序列和操作（decode/prefill/swap）；**第 2 步**，把执行计划交给 Model Runner 在 GPU 上执行；**第 3 步**，处理执行结果（采样出的 token、序列是否完成）；**第 4 步**，更新序列状态、回收资源、把输出推给客户端；然后回到第 1 步进入下一轮。Engine 还负责初始化模型、KV Cache 池、各组件，并对外提供同步（LLMEngine）和异步（AsyncLLMEngine，配合 asyncio 支持高并发）两套接口。**第三层 Scheduler**：如第 8 点所述，做请求排队、资源分配、抢占决策，输出本轮执行计划（哪些序列 decode、哪些 prefill、哪些 swap）。**第四层 KV Cache Manager**：管理显存池的 block 分配/释放/引用计数/换入换出，为 Scheduler 提供"还有多少可用 block"的实时视图，并执行 Scheduler 决定的 swap 操作。

**第五层 Model Runner**：把 Scheduler 的执行计划翻译成实际的 GPU 计算——它持有模型权重，负责把 batch 里的序列组织成 GPU 的输入张量（如按最大序列长度 padding、构造 attention mask、准备 KV cache 的 block 指针），调用模型的 forward 得到 logits。**第六层 GPU Worker**：在多卡/多机部署下，Worker 是"单卡执行单元"，负责本卡上的模型分片（张量并行时每卡持有部分权重）和计算，Worker 之间通过 NCCL 等通信库同步（如 all-reduce）。**第七层 Attention Backend**：如第 11 点所述，提供具体的 attention kernel（FlashAttention / FlashInfer 等），Model Runner 通过它执行 attention 计算。最后才是 GPU 硬件。值得强调的是，vLLM 还有一层**异步解耦**：Engine 的"调度-执行"循环和 API Server 的"收请求-发结果"是异步的（AsyncEngine 用后台事件循环），请求的 token 级输出通过异步队列流式回传，这是高并发流式输出的关键。

### 3. 具体数值样例

用一条真实请求走一遍完整的生命周期，逐步演算每个环节。假设客户端 POST 一个 prompt（100 token）到 API Server，`max_tokens=200`，模型 prefill 吞吐 10,000 token/s、decode 每步 5 ms（含 CUDA Graph 回放）、block size = 16。

```text
t = 0 ms      客户端 POST 请求 → API Server 解析，创建 Request，
              调用 engine.add_request()，请求进入 Scheduler 的 waiting 队列。
              此时 Engine 的事件循环正在跑（每轮 ~5 ms）。

第 1 轮循环    Scheduler 发现 waiting 里有新请求，KV Cache 有足够 block
              （需要 ⌈100/16⌉ = 7 个 block），安排 prefill；
              Model Runner 把 100 个 token 打包执行 prefill kernel
              （100 / 10000 = 10 ms），KV 写入 7 个 block，
              采样出第一个输出 token。
t ≈ 15 ms     TTFT（首 token 延迟）≈ 15 ms，第一个 token 经异步队列
              推给 API Server，流式返回客户端。

第 2~200 轮   该请求转入 running，每轮 Scheduler 让它 decode 一步，
              每步 5 ms；KV 每跨过 16 个 token 边界增长一个 block
              （第 112 个 token 时申请第 8 个 block，以此类推）；
              每轮生成的 token 通过异步队列推给客户端。
t ≈ 1015 ms   200 步 decode 完成（200 × 5 ms = 1000 ms），序列 FINISHED，
              KV block 释放，API Server 返回流结束标记。
```

单请求总耗时 ≈ 1.015 秒。现在叠加并发：如果同时有 20 个请求，Engine 每轮调度会把这 20 个请求一起 decode（batch=20），每步 GPU 时间从 5 ms 涨到约 12 ms（权重读取被摊薄、KV 读取增加），但 20 个请求**同时**推进，系统吞吐从 200 token/s 提升到约 20 / 12 ms ≈ 1667 token/s，而每个请求的 TPOT 从 5 ms 变为 12 ms（略有牺牲，但吞吐提升 8 倍）。这整个过程里，API Server 只负责收请求、发结果（异步），Engine 只负责调度与执行（事件循环），两层通过 `add_request()` 和输出队列解耦——这就是"一个 Engine 循环调度、一层层组件协作"的整体效果。

> **面试一句话总结**：vLLM 按"Client → API Server → Engine（事件循环）→ Scheduler → KV Cache Manager → Model Runner → GPU Worker → Attention Backend → GPU"分层，上层决策、下层执行、异步解耦，把请求调度、显存管理、kernel 执行组织成一条清晰的流水线，这是它高性能与可扩展性的架构基础。

### 4. 核心代码（源码）

**当前 vLLM（main 分支，MRv2 架构）的 Engine 核心在 `vllm/v1/engine/core.py` 的 `EngineCore.step()`**——它就是"调度-执行-回填"的每步事件循环：

```python
class EngineCore:
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output."""
        if not self.scheduler.has_requests():
            return {}, False

        # ① 调度：Scheduler 产生本轮执行计划
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        # ② 执行：异步把计划交给 Model Executor（GPU）
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with self.capture_iteration_details(scheduler_output), \
             self.log_error_detail(scheduler_output):
            model_output = future.result()          # 等 GPU 执行完成
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # ③ 回填：根据模型输出更新请求状态、释放 KV、产出输出
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**这段代码关键在哪**：`EngineCore.step()` 就是"Engine 事件循环"的代码落点——①`scheduler.schedule()` 决策（决策层）；②`model_executor.execute_model(scheduler_output, non_block=True)` 异步执行（**non_block=True 是 async-first 的关键**：调度下一步时 GPU 还在算上一步）；③`scheduler.update_from_output()` 回填（决策-执行分离的闭环）。`LLMEngine`（`v1/engine/llm_engine.py`）对外提供 `add_request` / `step` / `abort_request`，API Server 调它；`EngineCore` 跑在独立进程/线程（`core_client.py` 经 IPC 连）——这就是"分层 + 异步解耦"的实现。

### 5. 新版 vLLM 特性（v0.25.x 演进）

这是 v0.25.0 变化最大的部分：**Model Runner V2（MRv2）成为 dense 模型的默认执行路径**。MRv2 是对 V1 model runner 的地基式重建——模块化（把 V1 那个动辄 6000+ 行的巨型 runner 拆成职责单一的模块，最大的文件降到 1300 行左右）、GPU-native（执行计划直接在 GPU 侧组织，减少 CPU 侧的重复编排）、async-first（执行与调度进一步异步化，配合 CUDA Graph 原生支持）。它对上承接 Scheduler 的输出、对下驱动 Attention Backend / GPU Worker，本质上是把前面所有机制（Continuous Batching 的执行、Chunked Prefill 的编排、CUDA Graph 的回放、attention 后端的调用）收进一个更干净的执行框架。**Transformers modeling backend 也借此追平 native 性能**（v0.25.0）。面试时如果被问"vLLM 最近的架构变化"，MRv2 是必答项：记住"V1 → MRv2 = 巨型 runner → 模块化执行框架 + GPU-native + async-first + CUDA-graph-native"，并说明它是 v0.25.0 移除 legacy PagedAttention 的架构前提（因为 attention 已经全部由现代后端承担）。

---

## 14. Request / Sequence 管理

### 1. 现有问题：为什么需要 Request / Sequence 的数据结构管理

Request / Sequence 管理要解决的问题是：**推理引擎必须在内部用一套精确的数据结构，把"一个用户请求"从进入到完成的整个生命周期状态都管起来**——包括它的输入、已生成的 token、KV Cache 占用、采样参数、状态流转（等待→执行→完成）、以及流式输出的进度。这个看似"内务"的工作其实是推理引擎正确运行的基石：Scheduler 要根据序列状态决定调度；KV Cache Manager 要知道每个序列占了哪些 block；采样器要知道每个序列的采样参数；输出模块要知道每个序列已经输出到哪个 token（流式返回的游标）。如果这些状态散落、不一致，任何一步（比如请求完成时该释放哪些 block、被抢占后该恢复什么）都会出错。

一个典型的复杂场景是：同一时刻有几百个请求在跑，每个请求经历了"等待 → prefill → 多步 decode → 可能被抢占换出 → 恢复 → 完成"的生命周期；其中还有 beam search（一个请求分裂成多个序列）、并行采样（一个请求多个输出）、前缀共享（多个请求共享 KV block）等特殊情况。没有一套统一的对象模型，这些场景根本无法可靠实现。vLLM 通过 `Request`（用户视角的请求）和 `Sequence`（引擎视角的序列）两层对象来建模：一个请求可以对应一个或多个序列，一个序列是引擎实际调度、计算、管理显存的最小单元。

### 2. 方法论：vLLM 的 Request / Sequence 是怎么管理的

vLLM 里有两个核心对象。**`Request`**：面向 API 层的完整请求对象，包含原始 prompt、采样参数（temperature、top_p、max_tokens 等）、请求元数据，以及一个或多个输出（`RequestOutput`，包含最终/流式结果）。**`Sequence`**：面向引擎层的最小调度单元，一个 `Request` 在引擎内部被转换成一个或多个 `Sequence`（并行采样 `n>1` 时一个请求对应 n 个 sequence，beam search 时同理）。每个 `Sequence` 维护自己的状态机，包含四个状态：**WAITING（等待调度）→ RUNNING（执行中）→ FINISHED（完成，含 stopped / aborted 等终态）**，以及被抢占时的 SWAPPED 状态。Sequence 还持有它的完整上下文：token 列表（prompt + 已生成部分）、`sequence_id`、KV Cache 的 Block Table（或 BlockManager 中对应的分配记录）、采样状态、输出游标（`output_len`，流式返回用）。

生命周期管理的逐步操作由 Engine + Scheduler 协作完成，具体流程如下：

**第 1 步（创建）**：请求进来时 `add_request()` 创建 Request，并按参数转换为一个或多个 Sequence（`n=2` 时创建 2 个），每个 Sequence 初始状态为 WAITING，放进 Scheduler 的 waiting 队列。

**第 2 步（调度）**：每轮调度循环（见第 8 点），WAITING 的 Sequence 被取出做 prefill，分配 KV block，状态变为 RUNNING，加入 running 队列。

**第 3 步（每步推进）**：RUNNING 的 Sequence 每轮 decode 一步，生成一个新 token 追加到 token 列表；KV Cache 增长（跨 block 边界时申请新 block）；输出游标 +1；新 token 通过事件队列推给 API Server。

**第 4 步（终止检查）**：每步执行后检查终止条件——遇到 EOS、达到 max_tokens、被用户 abort、被抢占。满足则状态流转：正常完成 → FINISHED（stopped），被 abort → FINISHED（aborted），被抢占 → SWAPPED（swap 模式：KV block 拷到 CPU，token 列表保留在内存）。

**第 5 步（回收）**：FINISHED 的 Sequence 移出 running，其 KV Cache block 按引用计数释放（见第 3 点）；输出通过 `RequestOutput` 按序返回。SWAPPED 的 Sequence 之后在显存富余时恢复为 RUNNING（swap in 或 recompute，见第 9 点）。

**流式输出**的管理是一个细节：引擎每步生成新 token 后，通过事件队列把增量（新 token、新 logprob 等）推给 API Server，API Server 把 `delta` 返回给客户端，客户端收到的是"逐 token 流"，而不是等全部生成完——这要求 Request/Sequence 的"已输出游标"被精确维护，保证每步恰好返回新生成的部分、不重不漏。此外，抢占时 Sequence 整体进入 swapped 状态并保留其完整 token 列表（recompute 模式恢复时用），这再次说明 Sequence 状态管理是整个系统的"单一事实来源"。

### 3. 具体数值样例

假设一个请求 prompt 100 token、`max_tokens=200`、`n=2`（并行采样两个输出），逐步演算两个 Sequence 的完整生命周期。API Server 解析后创建一个 `Request`，Engine 把它转换成 **2 个 Sequence**（seq-0 和 seq-1），都从 WAITING 开始。

```text
第 1 轮调度：两个 Sequence 一起 prefill，各占 ⌈100/16⌉ = 7 个 block，
            共 14 个 block，状态 WAITING → RUNNING，
            各自采样出第一个 token（可能不同）。

第 2~80 轮： 每轮 decode 一步，各自维护自己的 token 列表和 Block Table。
            seq-0 与 seq-1 长度都在增长：每满 16 个 token 各申请一个新 block
            （如第 112 个 token 时申请第 8 个 block）。

第 80 轮：   显存紧张，Scheduler 决定抢占 seq-0（比如它较短、swap 成本低）。
            seq-0 状态 RUNNING → SWAPPED：它的 block 拷到 CPU，token 列表保留；
            seq-1 继续 RUNNING，不受影响。对客户端来说 seq-0 的输出"暂停"。

第 120 轮：  某其他请求完成释放显存，Scheduler 把 seq-0 swap in 回 GPU，
            seq-0 状态 SWAPPED → RUNNING，从暂停处继续 decode（内容连续）。

第 150 轮：  seq-0 生成到 150 token（占 ⌈250/16⌉ = 16 块）时遇到 EOS，
            状态 RUNNING → FINISHED(stopped)，block 释放，输出完整返回。
            seq-1 继续生成。

第 200 轮：  seq-1 生成满 200 token（max_tokens），状态 RUNNING → FINISHED(stopped)，
            block 释放，输出完整返回。
```

整个过程中，Request 始终只有一个（用户只发了一个请求），而引擎内部管理着 2 个生命周期独立、状态各异的 Sequence——seq-0 经历了 WAITING → RUNNING → SWAPPED → RUNNING → FINISHED 的完整状态链，seq-1 则是 WAITING → RUNNING → FINISHED。这套"请求-序列"双层模型正是支撑并行采样、抢占恢复、流式输出等复杂特性的数据结构基础。

> **面试一句话总结**：vLLM 用 Request（API 层）和 Sequence（引擎层）两层对象建模，Sequence 是调度、显存、采样的最小单元，通过 WAITING/RUNNING/SWAPPED/FINISHED 状态机 + 精确的输出游标，支撑并行采样、抢占恢复与 token 级流式返回。

### 4. 核心代码（源码）

**当前 vLLM（main 分支，v1 架构）用单个 `Request` 对象（`vllm/v1/request.py`）统一建模**（v1 不再有独立的 Sequence 类，Request 内部直接管理 token 状态），它的**状态机**是调度的依据：

```python
class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()                                  # 等待调度
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()                   # 等远端 KV（PD 分离）
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()                                  # 执行中
    PREEMPTED = enum.auto()                                # 被抢占（可恢复）
    # Note: anything after PREEMPTED will be considered
    # as a finished status.                                 # ← PREEMPTED 之后都是终态
    FINISHED_STOPPED = enum.auto()                         # 正常结束（EOS）
    FINISHED_LENGTH_CAPPED = enum.auto()                   # 达到 max_tokens
    FINISHED_ABORTED = enum.auto()                         # 被 abort
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()                           # 出错
    FINISHED_REPETITION = enum.auto()                      # 重复检测终止
```

```python
# v1/request.py —— Request 的核心字段（v1 把 Sequence 的状态合并进 Request）
class Request:
    def __init__(self, request_id, prompt_token_ids, sampling_params, ...):
        self.status = RequestStatus.WAITING
        self.num_computed_tokens = 0      # 已算的 token 数（prefill/decode 进度）
        self.num_tokens_with_spec = ...   # 应算的 token 数（含投机草稿）
        self.num_output_placeholders = 0  # 投机解码的输出占位
        self.block_hashes: list[BlockHash] = ...   # 逐 block 哈希（prefix caching 用）
        self.output_token_ids: list[int] = ...     # 已生成 token
        ...
```

**这段代码关键在哪**：v1 的 `RequestStatus` 枚举比旧版更细——**`PREEMPTED` 之后的一切都是终态**（注释明确），Scheduler 判断"能不能继续调度"就看状态是否在 PREEMPTED 之前；`num_computed_tokens` / `num_tokens_with_spec` 是 v1 "无 prefill/decode 阶段之分"调度的核心字段（第 2 点）；`block_hashes` 是 prefix caching 的依据（第 4 点）。**对比旧版（v0.25 前）**：旧版是 Request + Sequence 两层（n>1 时一个 Request 多个 Sequence），v1 合并为单个 Request——这是 MRv2 架构的简化之一。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Request / Sequence 的对象模型与状态机在 v0.25.x 中保持稳定——WAITING / RUNNING / SWAPPED / FINISHED 四态流转依然是调度、显存、流式输出的"单一事实来源"。变化主要在**承载方式**：MRv2 的 async-first 设计让 Request 的提交与 Sequence 的执行进一步异步化（例如请求可以在前一个 step 还没结束时就被预处理、Sequence 状态更新与 GPU 执行解耦更深），但"一个 Request 对应一个或多个 Sequence、Sequence 是调度与显存的最小单元"这个心智模型完全不变。此外，随着 dynamic speculative decoding、多模态等新能力落地，Sequence 上需要维护的状态字段在增加（如投机草稿状态、模态 token 位置），但状态机的骨架没变。面试时可以把这层讲成"vLLM 的状态管理锚点"：所有机制（抢占、流式、并行采样）最终都落回到 Sequence 状态机上的状态迁移。

---

# 七、Sampling

## 15. Sampling Pipeline

### 1. 现有问题：为什么要单独看 Sampling Pipeline

Sampling Pipeline 要解决的问题是：**模型输出的 logits 只是"每个 token 的原始分数"，要变成"实际生成的文本"，中间还有一整套采样决策逻辑——这套逻辑的正确性和性能直接决定生成质量与吞吐，值得单独理解。** 一个常见的误解是"模型输出什么 token 就是什么 token"，实际上 LLM 生成是一个概率采样过程：模型最后输出的是一个词表大小的 logits 向量（比如 128k 维），要经过温度缩放、top-k/top-p 截断、重复惩罚、各种特殊 token 处理（EOS 停止、思维链标记强制等），最后按概率分布采样出一个 token id，再映射回文本。这些步骤看起来简单，但在高并发推理引擎里有大量工程细节：logits 处理是在 GPU 上还是 CPU 上做？如何批量处理几百个序列的采样？采样的随机性如何保证可复现？beam search 怎么和采样共存？

如果 Sampling Pipeline 做得不好，会出现两类问题。第一类是**质量问题**：温度、top-p、重复惩罚的参数处理错误会直接导致生成退化（重复、空泛、逻辑断裂）；对特殊 token 的处理错误会导致模型输出 EOS 后还继续生成、或输出非法格式。第二类是**性能问题**：logits 是词表大小的向量，每个 decode step 都要对它做后处理（mask、缩放、softmax 或 Gumbel 采样），如果这块逻辑在 CPU 上串行做、或与 GPU 计算不同步，会成为 decode 每步的隐藏瓶颈；更麻烦的是，logits 从 GPU 拷回 CPU 的拷贝本身就有延迟，必须在"拷贝哪些、何时拷贝"上做精细设计。

### 2. 方法论：vLLM 的 Sampling Pipeline 是怎么实现的

vLLM 的采样流程按"GPU 上处理 + CPU 上决策"的分工来组织，逐步操作如下。每个 decode step，模型 forward 输出的 logits（形状 [batch, vocab_size]）**不会整块拷回 CPU**（那太慢了），而是在 **GPU 上先完成大部分后处理**，由 vLLM 的 `LogitsProcessor` 按序列的采样参数逐项修改 logits：

- **第 1 步（temperature 缩放）**：logits 除以温度值 t（t=1 不变，t<1 更尖锐、t>1 更平滑）：$\text{logits}'_i = \text{logits}_i / t$；
- **第 2 步（重复惩罚）**：对序列中已出现的 token 的 logits 施加惩罚（如把已出现 token 的 logits 乘以一个小于 1 的系数或加上负值），降低重复生成的倾向；
- **第 3 步（top-k 截断）**：只保留 logits 最大的 k 个 token（如 k=50），其余全部置为 −∞，让它们概率为零；
- **第 4 步（top-p 截断）**：按降序累积 softmax 概率，当累积概率超过 p（如 0.9）时截断，把后续 token 置为 −∞；
- **第 5 步（采样）**：处理完的 logits 才进入采样，vLLM 默认用 **Gumbel-max 采样**或基于 CUDA 随机数生成器（Philox）的采样 kernel，在 GPU 上一次为 batch 里所有序列采样出下一个 token id，并把采样的 token id 和对应的 logprob 拷回 CPU。

**拷回 CPU 后**，CPU 侧做**最终决策与序列状态更新**：检查采样出的 token 是不是 EOS / 停止 token（是则标记序列 FINISHED，停止后续生成）、更新序列的 token 列表和输出游标、把新 token 推给输出队列。

这样设计的好处是：GPU 负责批量、并行的 logits 处理（计算密集部分不上 CPU），CPU 只做轻量的决策（停止判断、状态更新），logits 的 GPU→CPU 拷贝被压缩到"只拷采样结果（每个序列一个 token id + logprob）"而不是"拷整个 vocab 向量"，把每步的通信量从 [batch × vocab] 降到 [batch × 少量标量]。此外 vLLM 还支持 **seed 控制**（每个序列一个种子，保证可复现采样）、**beam search**（在 GPU 上做 beam 的 logits 处理，CPU 上做 beam 选择）、以及 streaming 场景下 logprob 的增量返回。采样参数（temperature/top_p/max_tokens 等）在每个 Sequence 上独立维护，同一个 batch 里的不同请求可以用不同的采样参数并行处理。

### 3. 具体数值样例

用一个可手算的小例子演示 logits 后处理的完整流程。假设词表只有 6 个 token（A~F），某个序列在某个 decode step 输出的 logits 为：

```text
logits = [2.0, 1.0, 3.0, 0.5, -1.0, 1.5]   (对应 A B C D E F)
```

**逐步演算（temperature=0.8、top_k=3、top_p=0.9）：**

```text
第 1 步（temperature=0.8）：除以 0.8（即 ×1.25）：
    [2.5, 1.25, 3.75, 0.625, -1.25, 1.875]

第 2 步（假设 A 已出现在历史中，重复惩罚系数 0.9）：A 的 logits ×0.9：
    [2.25, 1.25, 3.75, 0.625, -1.25, 1.875]

第 3 步（top_k=3）：取最大的 3 个：C(3.75)、A(2.25)、F(1.875)，其余置 −∞：
    [-∞, -∞, 3.75, -∞, -∞, 1.875]   （注意 A 虽被惩罚仍进前 3）

第 4 步（top_p=0.9）：对 [C=3.75, F=1.875] 算 softmax：
    e^3.75 = 42.52, e^1.875 = 6.52 → 概率 C = 42.52/49.04 = 0.867,
    F = 6.52/49.04 = 0.133；累积概率 0.867 < 0.9 且下一个没有
    （只剩 2 个），因此两个都保留。

第 5 步（采样）：按 [0.867, 0.133] 分布采样，大概率采到 C（87%），
    小概率采到 F（13%）。假设采到 C，把 token id 和 logprob 拷回 CPU。
```

再看性能层面的数值对比。假设 batch 里有 32 个序列，词表大小 128k。模型 forward 每步输出 logits 张量：32 × 128k × 2 字节 ≈ 8 MB。如果 naive 实现把整个 logits 拷回 CPU 做采样，每步要拷贝 8 MB（PCIe 上约 0.5~1 ms），而且 CPU 还要逐序列对 128k 维向量做 top-k/top-p（128k × 32 ≈ 400 万次操作），每步额外花费几毫秒——在 decode 每步只有几毫秒的预算里，这会造成吞吐大幅下降。vLLM 的做法：GPU 上先做 top-k（k=50）+ temperature + top-p，把每个序列的候选从 128k 压缩到几十个 token，然后采样 kernel 直接产出 32 个 token id 和 32 个 logprob——拷回 CPU 的数据量只有 32 个 id + 32 个 float ≈ 几百字节，拷贝开销几乎为零；CPU 只需做 32 次停止判断和状态更新（微秒级）。假设开启 CUDA Graph 后每步 GPU 计算 3 ms，采样环节优化前后：naive 每步 3 + 2 ≈ 5 ms（吞吐 200 token/s），vLLM 每步 3 + 0.05 ≈ 3.05 ms（吞吐约 328 token/s），吞吐提升约 60%——这就是把采样管线按"GPU 批量处理 + CPU 轻决策"分工带来的收益。

> **面试一句话总结**：vLLM 的 Sampling Pipeline 把 logits 后处理（temperature / top-k / top-p / 惩罚）和采样放在 GPU 上批量完成，只把每个序列的 token id 和 logprob 拷回 CPU 做停止判断与状态更新，把每步的通信和 CPU 开销压到接近零，是 decode 高吞吐的重要一环。

### 4. 核心代码（源码）

**当前 vLLM（main 分支）的采样在 `vllm/v1/sample/sampler.py` 的 `Sampler`**——它就是个 nn.Module，`sample()` 里依次调各种 logits 处理（全部 GPU 上批量做）：

```python
class Sampler(nn.Module):
    @staticmethod
    def apply_temperature(logits, temp, all_random):
        # 用 in-place 除法避免新建 tensor；greedy 请求 temp<ε 时置 1（防除零）
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))     # logits / temperature

    @staticmethod
    def greedy_sample(logits):
        return logits.argmax(dim=-1).view(-1)         # greedy：直接取 argmax

    def apply_logits_processors(self, logits, sampling_metadata, ...): ...
        # 自定义 logits processor（格式约束等）
    def apply_penalties(self, logits, sampling_metadata, ...): ...
        # 重复惩罚（presence/frequency penalty）

    def sample(self, logits, sampling_metadata, ...):
        """Sample logits based on sampling metadata.
        The various logits processing functions called in this method
        may update the logits tensor in-place."""
        if sampling_metadata.all_greedy:
            greedy_sampled = self.greedy_sample(logits)
            return greedy_sampled, processed_logprobs   # 全 greedy 直接短路
        ...
        # 否则：apply_temperature → top-k/top-p → 随机采样（Gumbel/Philox）
        sampled = ...  # GPU 上一次为 batch 所有序列采样出 token id
        return sampled, logprobs
```

**这段代码关键在哪**：`Sampler` 的每个方法对应方法论里的一个步骤——`apply_temperature`（logits/temp，**in-place 避免新 tensor**）、`greedy_sample`（argmax 短路，全 greedy 请求不用走随机采样）、`apply_logits_processors` / `apply_penalties`（自定义 processor + 重复惩罚）；`sample()` 是主入口，**全部在 GPU 上批量处理**，只把最终的 token id + logprob 拷回 CPU（对应"GPU 批量处理 + CPU 轻决策"）。`logits.div_(temp)` 的 in-place 细节说明 vLLM 对每步性能的极致优化。

### 5. 新版 vLLM 特性（v0.25.x 演进）

Sampling Pipeline 的"GPU 批量后处理 + CPU 轻决策"分工在 v0.25.x 中保持稳定，但有两个相关演进。第一是**与投机解码的深度耦合**：dynamic speculative decoding（v0.25.0 与 full CUDA graphs 兼容）要求采样环节处理"草稿验证 + 修正采样"（rejection sampling / typical acceptance）的逻辑，采样管线因此承担了比"单 token 采样"更复杂的批量验证职责，但"logits 在 GPU 处理、token id 拷回 CPU 决策"的骨架不变。第二是 **MRv2 下的采样执行**：logits 后处理（temperature / top-k / top-p / 惩罚）在 MRv2 中作为执行链的一环更模块化地编排，可以更好地与 CUDA Graph 和 attention 后端配合。面试时如果被问"采样管线怎么演进"，可以答："单 token 采样到投机验证采样的扩展，以及 GPU/CPU 分工在 MRv2 中的模块化重组——骨架不变，职责变重。"

---


# 九、Disaggregated Prefill / Decode

## 16. Prefill-Decode Disaggregation（PD Disaggregation）

### 1. 现有问题：为什么要提出 PD 分离

PD Disaggregation 要解决的核心问题是：**prefill 和 decode 的计算特性完全不同，把它们混在同一批 GPU 上，会导致资源利用率互相拖累、吞吐和延迟都达不到最优。** 回顾第 5、6 点：prefill 是计算密集（compute-bound）的——它一次并行处理整个 prompt，GPU 算力利用率高，但需要"爆发式"的大计算量；decode 是访存密集（memory-bound）的——它逐 token 生成，算力利用率低，但需要持续的、低延迟的算力供给。当两者混跑时（即使有 Chunked Prefill 缓解），prefill 的"大块计算"会周期性挤占 decode 的算力，导致 decode 的 TPOT（字间间隔）抖动、尾延迟升高；反过来，decode 的低算力利用率又"浪费"了 prefill 需要的算力资源——**两类请求对资源的需求形状完全不同，放在一起就是互相迁就、两头不讨好**。

更深层的问题是：在混合部署下，用户感知的两个关键指标——**TTFT（首 token 延迟，由 prefill 决定）** 和 **TPOT（token 间隔，由 decode 决定）**——是此消彼长的。prefill 请求多了，decode 变慢；decode 请求多了，prefill 排队变长。而且当 batch 里出现一个超长 prompt 的 prefill 时，所有正在 decode 的请求都要被拖慢（即使 Chunked Prefill 也只是"均摊"而非"消除"这种干扰）。另一个现实问题是**显存和算力的错配**：prefill 阶段对显存带宽要求低（它是计算密集）、对算力要求高；decode 阶段恰恰相反（算力需求低、显存带宽需求高）。把两者放在同一批卡上，无论怎么调参，总有一部分资源处于"闲置但被占用"的状态。PD 分离的思路就是：**用不同物理机/不同 GPU 分别跑 prefill 和 decode，各自按自己的特性做极致优化，互不干扰。**

### 2. 方法论：PD Disaggregation 是怎么实现的

PD Disaggregation 的架构是把推理服务拆成两类独立部署的节点：**Prefill 节点（P 节点）** 和 **Decode 节点（D 节点）**。P 节点配高算力、KV Cache 不长期驻留（prefill 完就传走）；D 节点按"尽量多驻留 KV Cache、高带宽"配置，专心地低延迟 decode。两个节点之间通过高速网络（RDMA，如 InfiniBand / RoCE，或自定义的 KV 传输协议）搬运 KV Cache。一个请求从进入到完成的完整流程逐步操作如下：

**第 1 步（路由到 P 节点）**：请求到达后，由控制器/负载均衡器决定送哪个 P 节点（可按 P 节点的负载、是否命中 Prefix Caching 前缀等策略路由）。

**第 2 步（P 节点 prefill）**：P 节点对 prompt 做完整 prefill，算出全部位置的 KV Cache；期间可以复用 Prefix Caching（相同前缀只算一次）、Chunked Prefill。

**第 3 步（KV 传输）**：prefill 完成后，把该请求的 KV Cache 张量从 P 节点传输到某个 D 节点。传输方式可以是 GPU 直传（GPUDirect RDMA）或先拷到 CPU 再经网络；传输量等于该请求的 KV Cache 大小。为降低传输延迟，业界有"KV Cache 压缩"（张量量化、稀疏化）和"分块传输"（边生成边传，不必等整个 prefill 完成）等优化。

**第 4 步（D 节点 decode）**：D 节点接收 KV Cache 后接管后续 decode，逐 token 生成直到完成；D 节点只做 decode，可以用更大的 batch 提升吞吐。

**第 5 步（完成与回收）**：请求生成完毕，D 节点释放 KV Cache；整个过程中客户端只看到一个完整的生成过程，P/D 节点的切换对客户端透明。

工程实现上有几个关键难点。第一是 **KV Cache 的跨节点传输**：传输延迟和带宽是系统设计的核心权衡点（传输量 = KV Cache 大小，长 prompt 请求的 KV 可达几百 MB）。第二是 **请求路由与协调**：需要一个控制器决定请求去哪个 P 节点、P 完成后通知哪个 D 节点接管，以及 KV Cache 的位置元数据如何传递。第三是 **资源配比与调度**：P 节点和 D 节点的数量配比要按流量特征调整（prompt 长、生成短的场景 P 节点要多；prompt 短、生成长的场景 D 节点要多），且节点间 KV 传输带宽要足够，否则传输会成为新瓶颈。第四是 **与现有机制的配合**：P 节点复用 Prefix Caching / Chunked Prefill；D 节点专注 decode 提升吞吐。业界已有多种实现形态（如 Mooncake 的 KVCache 分离架构、vLLM 的 disaggregated serving 实验形态、SGLang 的 RadixAttention + 分离部署等），它们都共享"prefill 与 decode 物理分离 + KV 跨节点流动"的核心思想。

### 3. 具体数值样例

假设一个在线聊天服务，平均 prompt 500 token、平均生成 300 token，峰值并发 100 个请求，目标是 TTFT < 300 ms、TPOT < 20 ms。先看混合部署：一台 8 卡 A100（80 GB）机器上混跑，prefill 和 decode 共享算力。当 100 个请求同时涌入时，所有请求都要先做 prefill——prefill 阶段 GPU 算力被占满，decode 中的请求被拖慢，实测 TPOT 可能抖到 30~50 ms，同时尾部请求的 TTFT 可能超过 1 秒。

改为 PD 分离后，逐步演算一个请求的完整时间线：配置 4 张卡跑 P 节点（高算力）、4 张卡跑 D 节点（大 KV 驻留 + 高带宽）。

```text
t = 0 ms      请求到达，控制器路由到某个 P 节点
t = 0~50 ms   P 节点 prefill：500 token / 10,000 token/s = 50 ms（TTFT 起点）
t = 50~55 ms  KV 传输：500 token × 512 KB = 250 MB，经 RDMA
              （假设 50 GB/s 实际带宽）传输约 5 ms
t = 55 ms     请求在 D 节点接管，开始逐 token decode（每步约 15 ms）
t = 55~455 ms 300 个 token 生成完成（300 × 15 ms = 4.5 s 若串行，但实际
              100 个请求共享 D 节点算力，batch=100 时每步约 15 ms）
```

对比结果：混合部署 TTFT 波动大（尾部可能 > 1 秒）、TPOT 不稳（30~50 ms 抖动）；PD 分离后 TTFT 约 50 ms（达标，远低于 300 ms 目标）、TPOT 稳定 15 ms（达标）。显存账也很好算：100 个请求的 KV Cache 共约 100 × 250 MB = 25 GB，4 卡 D 节点（每卡 80 GB，KV 部分按 60 GB 可用计）共 240 GB，足够驻留全部 KV。且 P/D 节点可以独立扩缩容——prefill 压力大就加 P 节点（比如突发大量长 prompt 请求），并发生成多就加 D 节点，资源利用率和可扩展性都显著提升。代价是多了一套节点间 KV 传输的网络和调度系统，这也是 PD 分离在超大并发、长上下文场景（如 Mooncake 面向的大模型推理集群）才更值得投入的原因。

> **面试一句话总结**：PD Disaggregation 把计算密集的 prefill 和访存密集的 decode 分离到不同 GPU 节点，通过 RDMA 把 KV Cache 从 P 节点传给 D 节点，让 TTFT 和 TPOT 各自独立优化、互不干扰，支持按流量独立扩缩容，代价是引入跨节点 KV 传输与协调调度。

### 4. 核心代码（源码）

**vLLM 的 PD 分离通过 `vllm/distributed/kv_transfer/` 的 KV connector 实现**——prefill 节点把算好的 KV 传给 decode 节点，`KVConnectorBase_V1` 是统一接口（Mooncake / FlexKV / NIXL 等 connector 都实现它）：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py
class KVConnectorRole(enum.Enum):
    """connector 的角色：决定是推 KV 还是拉 KV。"""
    PREFILL = 0      # prefill 节点：算完 KV → push 给 decode
    DECODE = 1       # decode 节点：从 prefill pull KV → 续生成

class KVConnectorBase_V1(ABC):
    def role(self) -> KVConnectorRole: ...
        # 本节点是 PREFILL 还是 DECODE（决定 KV 流向）

    def bind_connector_metadata(self, connector_metadata) -> None: ...
    def clear_connector_metadata(self) -> None: ...
    def request_finished_all_groups(self, request) -> None: ...
    # 子类（mooncake_connector / flexkv_connector / nixl）实现：
    #   push_kv(blocks)  /  pull_kv(block_hashes)   ← 跨节点传输 KV
```

```python
# 一个 PD 分离请求的流程（代码层面对应）：
# PREFILL 节点：prefill 完 → 把 KV blocks 通过 connector push 到共享存储/远端
# DECODE 节点：新请求 → 按 block_hashes 从 connector pull 已算好的 KV
#   （Scheduler 的 get_computed_blocks 命中远端 KV → num_computed_tokens 直接
#     跳到命中长度，decode 节点不用重算 prompt）
```

**这段代码关键在哪**：PD 分离的代码基础是 `KVConnectorRole`（PREFILL/DECODE 两种角色）——P 节点 `push_kv`、D 节点 `pull_kv`，传输走 Mooncake（RDMA）/FlexKV/NIXL 等后端（connector 目录下每个子类一种传输实现）；decode 节点通过 `get_computed_blocks` 命中远端 KV 后**跳过 prompt 的 prefill**（`num_computed_tokens` 直接前进）。**Mooncake（`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`）就是 PD 分离 + KV 传输的一个具体实现**，与本项目简历的 Mooncake 直接相关。

### 5. 新版 vLLM 特性（v0.25.x 演进）

PD Disaggregation 是 vLLM 持续投入的方向，v0.25.x 的进展与它有两个交叉点。第一是 **KV Cache 压缩降低传输成本**：FP8 KVCache（见第 3 点）让 KV 显存占用减半，也直接让 P→D 节点间要传输的 KV 数据量减半——这是对"KV 传输是 PD 分离最大代价"这一痛点的直接缓解；配合 KV 量化和分块传输，长 prompt 场景的跨节点 KV 流动成本显著下降。第二是 **MRv2 的模块化让分离更顺**：P 节点和 D 节点跑同一套 MRv2 执行框架、只是配置不同的执行形态（P 侧重 prefill 路径、D 侧重 decode 路径），比 V1 时代"两套执行逻辑"更容易维护。vLLM 的 disaggregated serving 仍属于实验/演进形态，但方向明确：配合 Mooncake 这类 KV 分离生态（见 Mooncake.md），PD 分离在超大并发、长上下文场景是确定性的演进方向。面试时可以讲："PD 分离的架构思想没变，变的是 KV 传输成本被 FP8 压缩等手段持续压低，以及 MRv2 让 P/D 节点共享同一执行框架。"

---
