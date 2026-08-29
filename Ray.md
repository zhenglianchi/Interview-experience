# Ray 分布式框架完全指南（组件 + 架构 + 底层传输协议全解）

> 对应简历"熟悉 Ray 分布式框架"与实习"基于 Ray 的 AgenticRL 训练作业（Kubernetes CRD 资源）管控面"知识体系。一句话知识框架：
> **Ray 是一个"进程级"的通用分布式计算框架：控制面（GCS + Raylet）管调度与状态，数据面（Plasma 共享内存 + gRPC 分块传输）管对象流动；跨节点走 gRPC over TCP、本地走 Unix socket + FlatBuffers、GPU 集合通信走用户态 NCCL（原生不支持 RDMA）**——verl/AgenticRL 训练就是跑在它上面的 actor 资源组。
>
> 素材来源：本地克隆 `C:\Users\HW\Desktop\简历投递\ray`（ray-project/ray master，HEAD `5cb244bc`），全部事实源码实证。与 `Parallel.md`（并行维度）、`Communication.md`（GPU/NPU 通信协议）、`Mooncake.md`（KV 传输，对照"对象传输"）配套阅读。

---

# 一、核心组件（先分点）

## 1. GCS（Global Control Store）：全局控制面

### 1. 现有问题

- 分布式系统需要一个"大家都信"的状态中心：谁在集群里（node 表）、哪些 job/actor/worker 存在（job/actor/worker 表）、资源分布、placement group 状态；
- 早期 Ray 用 **Redis** 当 GCS，Redis 单点 + 语义不匹配（GCS 需要的是"表 + 订阅"而非 KV），还要额外运维一个 Redis；
- 控制面必须高可用：leader 挂了要能安全切换，状态要能恢复。

### 2. 方法论

GCS 是一个**独立进程**（`gcs_server`），跑在 head 节点，对外是一个 **gRPC 服务**（服务名 `GcsServer`，端口默认 6379——向后兼容 Redis 时代）。它聚合了 job/node/actor/placement group/worker 等**元数据表**，并提供 Internal KV、资源广播、pubsub 订阅、任务事件上报。**现代 Ray 默认不依赖 Redis**：

```cpp
// src/ray/gcs/gcs_server.cc —— 存储后端三选一
GcsServer::StorageType GcsServer::GetStorageType() const {
  if (RayConfig::instance().gcs_storage() == kInMemoryStorage) {
    if (!config_.redis_address.empty()) {
      return StorageType::REDIS_PERSIST;   // 显式给了 redis → 用 Redis 持久化
    }
    return StorageType::IN_MEMORY;         // 默认：纯内存
  }
  if (RayConfig::instance().gcs_storage() == kRedisStorage) { return StorageType::REDIS_PERSIST; }
  if (RayConfig::instance().gcs_storage() == kRocksDbStorage) {
    return StorageType::ROCKSDB_PERSIST;   // 仅 Linux，需 gcs_storage_path
  }
  ...
}
```

**HA 机制**：① Kubernetes 部署下用 **K8s Lease 做 leader 选举**（`leader_elector.cc`：选举线程 + watchdog 线程，`renew_deadline < lease_duration` 保证安全交接）；② 纯内存模式下 GCS 重启后从各 raylet/worker **重新拉取状态**（`gcs_init_data.cc` 加载 6 张表）；③ 配 Redis/RocksDB 则状态持久化可恢复；④ 心跳健康检查 + `NotifyGCSRestart` RPC 通知全体。

### 3. 具体数值样例

- 一张 node 表记录所有存活节点（心跳 5s 超时判死）；actor 表记录每个 actor 的地址与状态；
- GCS 端口 6379：`--redis_port=6379`（`services.py`）与 `gcs_server_port` 默认一致；
- 现代 Ray 中**任务表和对象表已不在 GCS**（任务走 raylet lease 调度、对象位置由 owner 持有）——GCS 的 TaskManager 只做**任务事件（observability）**，避免控制面成为瓶颈。

> 面试一句话总结：**GCS 是 Ray 的全局控制面：一个 gRPC 服务聚合 job/node/actor/PG 元数据表，存储后端默认纯内存（可切 Redis/RocksDB），HA 靠 K8s lease leader 选举 + 重启重拉状态——关键设计是"任务/对象热路径不进 GCS"，控制面只存元数据不挡数据流。**

---

## 2. Raylet：节点大脑（NodeManager + ObjectManager + Plasma）

### 1. 现有问题

- 每个节点需要一个"本地管家"：管理 worker 进程、执行任务调度、协调对象传输、管理对象存储；
- 如果所有调度决策都走全局中心，控制面会成为瓶颈（Ray 的哲学是**本地优先 + 全局协调**）。

### 2. 方法论

**每节点一个 raylet 进程**，内部是三个组件合一：

```text
Raylet 进程
├── NodeManager    —— 调度与租约：RequestWorkerLease / PushTask / 资源管理
├── ObjectManager  —— 对象传输：Pull/Push gRPC，分块读写（memory/disk/spill）
└── Plasma Store   —— 共享内存对象存储（内嵌在 raylet 进程内，mmap + Unix socket）
```

raylet 进程入口（`raylet/main.cc`）里 Plasma 是内嵌的 `ObjectStoreRunner`，溢出回调直接连到 `LocalObjectManager`：

```cpp
// src/ray/raylet/main.cc —— plasma 内嵌 raylet
auto object_store_runner = std::make_unique<ray::ObjectStoreRunner>(
    object_manager_config,
    /*spill_objects_callback=*/[&]() {
      main_service.post([&]() { local_object_manager->SpillObjectUptoMaxThroughput(); },
                        "NodeManager.SpillObjects");
      return local_object_manager->IsSpillingInProgress();
    },
    /*object_store_full_callback=*/[&]() { node_manager->SetShouldGlobalGC(); },
    /*add_object_callback=*/..., /*delete_object_callback=*/...);
```

raylet 的关键命令行参数（`raylet/main.cc`）暴露了它的全部"接口"：

```cpp
DEFINE_string(raylet_socket_name, "", "The socket name of raylet.");      // 本地 IPC
DEFINE_string(store_socket_name, "", "The socket name of object store."); // plasma socket
DEFINE_int32(object_manager_port, -1, "The port of object manager.");     // gRPC 传输
DEFINE_int32(node_manager_port, -1, "The port of node manager.");         // gRPC 调度
DEFINE_string(gcs_address, "", "The address of the GCS server.");
DEFINE_int32(min_worker_port, 0, "The lowest port that workers' gRPC servers will bind on.");
DEFINE_int32(max_worker_port, 0, "The highest port that workers' gRPC servers will bind on.");
```

### 3. 具体数值样例

- 8 节点集群：8 个 raylet，每个管理本机 worker 进程池（`worker_pool.cc` 按需 fork/exec）；
- 端口：`node_manager_port` 默认 0（自动分配空闲端口）；worker gRPC 端口在 `min_worker_port`~`max_worker_port`（典型 10000~19999）段内由 raylet 分配；
- 一个 task 从提交到执行：worker → 本地 raylet（RequestWorkerLease）→ raylet 调度决策 → PushTask 给目标 worker → 执行 → 写 plasma。

> 面试一句话总结：**Raylet 是每节点的"大脑+仓库"：NodeManager 管调度租约、ObjectManager 管对象传输、Plasma 内嵌管共享内存存储，三者一个进程；对外接口 = 两个 Unix socket（本地 IPC + plasma）+ 三个 gRPC 端口（node manager / object manager / worker 段）——一个进程承载了节点侧的全部职责。**

---

## 3. Plasma Object Store：共享内存对象存储

### 1. 现有问题

- 分布式计算里"传对象"最怕**序列化 + 网络 + 反序列化**三连：大对象（数据集、中间结果）走网络慢，且每传一次拷贝一份；
- 同节点的多个 worker 要共享数据（如训练数据 batch），如果走"进程间消息"，每份数据要复制 N 次；
- 需要一个"**进程间零拷贝共享**"的对象存储：写一次，多个进程 mmap 读。

### 2. 方法论

Plasma 是**共享内存对象存储**：`memfd` + `mmap(MAP_SHARED)` 创建共享内存段，通过 **Unix domain socket**（`--store_socket_name`）对外服务，任意本机进程把该段 mmap 进自己的地址空间即可**零拷贝读写**：

```cpp
// src/ray/object_manager/plasma/shared_memory.cc —— mmap 共享内存
pointer_ = mmap(NULL, length_, PROT_READ | PROT_WRITE, MAP_SHARED, fd.first, 0);
...
madvise(pointer_, length_, MADV_DONTDUMP);   // Linux 下排除 coredump
```

`ray.put` 的写入链路（`core_worker.cc`）：

```cpp
Status CoreWorker::Put(const RayObject &object, ...) {
  *object_id = ObjectID::FromIndex(worker_context_->GetCurrentInternalTaskId(),
                                   worker_context_->GetNextPutIndex());
  reference_counter_->AddOwnedObject(*object_id, contained_object_ids, rpc_address_, ...);
  auto status = Put(object, contained_object_ids, *object_id, /*pin_object=*/true);
  ...
}
// PutInLocalPlasmaStore 中：写完后 pin 住，防止被 evict
local_raylet_rpc_client_->PinObjectIDs(rpc_address_, {object_id}, ...);
```

- **plasma 客户端协议**：`client.cc` 的 `Create/Seal/Get/Release`（创建 buffer → memcpy 写入 → Seal 发布 → 读者 Get/Release），分配器是 dlmalloc（`allocator.h`），淘汰策略 `eviction_policy.cc`；
- **FD 传递**：`fling.cc` 用 `sendmsg/SCM_RIGHTS` 把共享内存 fd 在进程间传递——**这是"零拷贝"的机制核心**：不是把数据复制过去，而是把"同一块内存的句柄"传过去；
- 对象生命周期：owner-based 引用计数（第 9 点），`Put` 后 `PinObjectIDs` 防止被 evict。

### 3. 具体数值样例

- `ray.put(np.zeros((1024,1024), dtype=np.float32))` ≈ 4 MB：写入 plasma 共享内存段（memcpy 一次），同节点 N 个 worker `ray.get` 各自 mmap 同一段——**写 1 次、读 N 次零拷贝**；
- 对比走 gRPC：4 MB 序列化 + 网络 + 反序列化，单次传输开销是共享内存的 10~100 倍；
- 容量：plasma 受 `object_store_memory`（默认节点内存的 30%）限制，满时触发 spill（第 8 点）。

> 面试一句话总结：**Plasma 是 Ray 的共享内存对象存储：memfd+mmap 建共享段、Unix socket 服务、sendmsg/SCM_RIGHTS 传 fd 实现零拷贝（写一次、同机任意进程 mmap 读），配合 dlmalloc 分配器与 eviction 策略——同节点对象交换不走网络、不反复序列化，是 Ray 数据面的性能根基。**

---

## 4. CoreWorker：Driver / Worker / Actor 的统一运行时

### 1. 现有问题

- Driver、普通 task worker、actor 是三种"角色"，但底层能力相同：提交任务、持有对象引用、执行任务；
- 若每种角色一套实现，代码爆炸且语义不一致。

### 2. 方法论

**所有"会跑 Ray 代码的进程"都是一个 CoreWorker**（`src/ray/core_worker/core_worker.cc`），只是 `WorkerType` 不同（DRIVER / WORKER / SPILL_WORKER 等）：

- **Driver**：`ray.init()` 的用户进程，本身也是 CoreWorker，负责提交任务、持有 ObjectRef；
- **Worker**：raylet 通过 `python_worker_command` fork/exec 出的临时进程（`worker_pool.cc`），执行完普通 task 即退出；
- **Actor**：**长期存活**的 Worker，同一进程串行执行多个 actor task（有状态）。

Python → C++ 的边界是 Cython（`python/ray/_raylet.pyx`）：

```python
# python/ray/_raylet.pyx —— task 提交进入 C++ 核心
with nogil:
    return_refs = CCoreWorkerProcess.GetCoreWorker().SubmitTask(
        ray_function, args_vector, task_options,
        max_retries, retry_exceptions, c_scheduling_strategy,
        debugger_breakpoint, serialized_retry_exception_allowlist,
        call_site, current_c_task_id,
    )
```

C++ 侧分派（`core_worker.cc`）：

```cpp
// SubmitTask 内部：普通 task 与 actor task 走不同提交器
normal_task_submitter_->SubmitTask(std::move(task_spec));   // 普通 task
actor_task_submitter_->SubmitTask(spec);                    // actor task
// 执行侧（raylet PushTask 到达后）
Status CoreWorker::ExecuteTask(...)   // 执行并写回 plasma / 内存 store
```

### 3. 具体数值样例

- 一个 `@ray.remote` 函数的调用：Python `RemoteFunction.__call__` → `core_worker.submit_task`（Cython）→ `CoreWorker::SubmitTask`（C++）→ `RequestWorkerLease`（gRPC 到本地 raylet）→ 目标 worker `PushTask` → `ExecuteTask` → 返回值（<100KB 内联在 task reply 的 owner 内存 store；≥100KB 写 plasma）——**一次调用跨 Python/Cython/C++ 三层，协议从本地 IPC 到 gRPC**；
- actor 与普通 task 的区别：actor 的 `ExecuteTask` 复用同一进程状态，普通 task 每次可能起新 worker。

> 面试一句话总结：**Driver/Worker/Actor 全是同一个 CoreWorker 运行时（只是 WorkerType 不同）：Python 侧经 Cython 进入 C++ 核心，SubmitTask 后普通任务走 normal_task_submitter、actor 任务走 actor_task_submitter；返回值按 100KB 阈值分流（小对象内联、大对象写 plasma）——统一运行时让三种角色共享同一套调度/存储/引用计数语义。**

---

## 5. 调度：本地优先的租约制 + Actor/PG 调度

### 1. 现有问题

- 任务调度最怕"全局中心化决策"（吞吐瓶颈）与"盲目随机"（数据局部性差——任务的数据在本地，任务却跑到远端）；
- 大模型训练要**确定性资源**：actor 常驻、placement group 把资源绑在一起。

### 2. 方法论

**① 任务调度 = 租约制（lease）**：core worker 向本地 raylet 发 `RequestWorkerLease`，raylet 的 `ClusterLeaseManager` 周期 `ScheduleAndGrantLeases()`，用 **Hybrid 调度策略**选节点——**默认本地优先**：

```cpp
// src/ray/raylet/scheduling/policy/hybrid_scheduling_policy.cc
scheduling::NodeID preferred_node_id = local_node_id_;   // 默认偏好本节点
...
bool prioritize_preferred_node = !force_spillback && preferred_node_is_available;
return GetBestNode(available_nodes, num_candidate_nodes,
                   prioritize_preferred_node
                       ? std::optional<scheduling::NodeID>(preferred_node_id)
                       : std::optional<scheduling::NodeID>(),
                   ComputeNodeScore(preferred_node_id, spread_threshold));
```

配套配置（`ray_config_def.h`）：`scheduler_spread_threshold=0.5`（低于该利用率算 0 分）、`scheduler_top_k_fraction=0.2`（在 top-k 节点里随机，避免热点）。

**② Actor 调度**：由 GCS 的 `GcsActorManager` / `GcsActorScheduler` 决策（调 raylet 的 `PrepareBundleResources` / `CommitBundleResources` / `RequestWorkerLease`）——actor 是全局资源，必须由控制面协调放置；普通 actor 也走 lease。

**③ Placement Group（PG）**：`GcsPlacementGroupScheduler` 把 bundle 按策略（`spread` / `strict_pack` / `PACK` 等）分配到节点，raylet 侧 `PlacementGroupResourceManager` 做**资源预留**（prepare）与提交（commit）——训练时把"1 个 driver + 8 个 trainer + 2 个 PS"绑在同一组节点上。

### 3. 具体数值样例

- 训练任务在 node1 上已有数据（如 dataset shard），`ray.remote` 的 task 默认**本地优先**落在 node1——避免把数据搬走；node1 资源不足才 spillback 到 node2；
- top-k 随机：候选 10 节点里按 `top_k_fraction=0.2` 随机取 2 个再打分，避免所有任务涌向"最高分"节点造成热点；
- PG 示例：`placement_group([{"GPU": 8}, {"GPU": 8}], strategy="PACK")` 把两个 bundle 打包到同一节点（8 卡机内互联），训练通信走 NVLink 而非跨机。

> 面试一句话总结：**Ray 的任务调度是"本地优先的租约制"：worker 向本地 raylet 租资源，raylet 用 Hybrid 策略默认把任务留在本节点（保住数据局部性）、资源不足才 spillback，top-k 随机防热点；actor 与 placement group 由 GCS 控制面决策、raylet 预留资源——普通任务去中心化快路径，全局资源集中协调。**

---

# 二、数据面：对象传输协议（底层传输协议整理全）

## 6. 同节点对象传输：plasma mmap 零拷贝

### 1. 现有问题

同节点的对象共享如果走 IPC 消息或网络回环，每份数据复制多次，浪费带宽和 CPU。

### 2. 方法论

`ray.get` 命中本地 plasma 时，worker 直接把 plasma 的 **mmap buffer 映射进自己地址空间**（`MemoryObjectReader` + `ObjectBufferPool`），**零拷贝**：

- 写：`ray.put` → plasma `Create`（拿到共享段 buffer）→ memcpy → `Seal`；
- 读：`ray.get` → 按 ObjectID 查 plasma → `Get` 返回 mmap 指针 → 直接读（无需反序列化到新 buffer，除非是跨语言/需要 owned buffer）；
- 关键点：**同节点根本没有网络参与**，传输 = 内存映射 + 引用计数。

### 3. 具体数值样例

- driver `ray.put` 一个 1 GB 的 numpy 数组 → 4 个 worker 同时 `ray.get`：写 1 次 memcpy（1 GB），读 4 次全零拷贝；若走 gRPC 回环，4 次传输 = 4 GB 网络 + 4 次序列化；
- 实际传输路径（`ray.get` 完整链路见第 13 点）：内存 store（小对象）→ plasma（大对象）→ 远端拉取（缺对象时）。

> 面试一句话总结：**同节点对象传输 = plasma mmap 零拷贝：put 写共享段一次，get 时多个进程 mmap 同一段直接读——写一次读 N 次零拷贝，是 Ray 数据面"本地快"的根本；跨语言/需独占 buffer 时才发生拷贝。**

---

## 7. 跨节点对象传输：gRPC over TCP 分块 Push

### 1. 现有问题

- 对象在远端时 `ray.get` 要跨节点拉取：谁负责传？怎么传？大对象（GB 级）一次 RPC 会超 gRPC 上限；
- 传输必须**按需拉取**（pull-based），而不是 put 时广播（push-based）——否则没人 get 的对象白传。

### 2. 方法论

**拉取驱动的两级协议**：

```text
ray.get 缺对象：
① core worker 向 owner 查对象位置（GetObjectLocationsOwner RPC + 订阅
   WORKER_OBJECT_LOCATIONS_CHANNEL）
② 本地 IPC AsyncGetObjectsRequest（FlatBuffers）→ 本地 raylet
③ raylet 的 LeaseDependencyManager 发 gRPC Pull 给远端 object manager
④ 远端按 5MB 分块（object_manager_default_chunk_size）用 gRPC Push 逐 chunk 推回
   （unary RPC，非 streaming）
⑤ 数据从远端 plasma mmap 零拷贝读出（MemoryObjectReader），落盘对象从文件读
   （SpilledObjectReader），写入本地 plasma → 本地 get 命中
```

关键配置（`ray_config_def.h`）：

```cpp
/// Default chunk size for multi-chunk transfers to use in the object manager.
RAY_CONFIG(uint64_t, object_manager_default_chunk_size, 5 * 1024 * 1024)  // 5MB 分块
/// Maximum size of a task return value that may be returned inline in the task reply.
RAY_CONFIG(int64_t, max_direct_call_object_size, 100 * 1024)             // 100KB 小对象内联
/// The max gRPC message size (the gRPC internal default is 4MB).
RAY_CONFIG(size_t, max_grpc_message_size, 512 * 1024 * 1024)             // Ray 将 gRPC 上限调到 512MB
```

### 3. 具体数值样例

- 拉 1 GB 对象跨节点：按 5 MB 分块 → 204 个 chunk，逐块 gRPC Push；每 chunk 一个 unary RPC（非 streaming），gRPC 消息上限被 Ray 调到 512 MB 但单 chunk 远小于此；
- 小对象（<100 KB）根本不进 plasma：直接内联在 task reply 里，存在 owner 的内存 store——省一次等离子体往返；
- 传输瓶颈：跨节点带宽（如 10 Gbps 网卡 ≈ 1.25 GB/s），1 GB 对象理论 ~0.8 s（不含 RPC 开销）；
- **RDMA：Ray 原生不支持**（全 src/ray 无 RDMA 实现，历史 plasma-over-RDMA 实验已移除）——跨节点对象传输就是 TCP + gRPC。

> 面试一句话总结：**跨节点对象传输是"拉取驱动"的：ray.get 缺对象 → 向 owner 查位置 → 本地 raylet 发 Pull 给远端 → 远端按 5MB 分块用 gRPC Push 逐 chunk 推回（unary RPC）→ 写本地 plasma；100KB 以下小对象内联在 task reply 不进 plasma；Ray 原生不支持 RDMA，跨节点就是 TCP+gRPC。**

---

## 8. 对象溢出（Spill to Disk / S3）：内存不够时怎么办

### 1. 现有问题

plasma 容量有限（默认节点内存 30%），对象放不下/放满时必须**腾地方**，且不能丢（对象可能仍被引用）。

### 2. 方法论

**自动溢出**：raylet 的 `LocalObjectManager` 监听 plasma 满（`spill_objects_callback`，见第 2 点代码）→ `SpillObjectUptoMaxThroughput()` → 通过 `CoreWorkerService.SpillObjects` RPC 让对象 **owner** 把对象写入外部存储（`object_spilling_config`：filesystem / S3 等），IO 由专门的 **spill worker 进程**执行：

```cpp
/* Configuration parameters for object spilling. */
RAY_CONFIG(std::string, object_spilling_config, "")
RAY_CONFIG(std::string, object_spilling_directory, "")
RAY_CONFIG(bool, automatic_object_spilling_enabled, true)   // 默认开启
```

- 溢出对象的位置记录在 owner 的对象目录里，后续 `ray.get` 会走 `SpilledObjectReader` 从文件/S3 读回；
- 本地文件系统溢出是默认；S3/外部存储通过 `object_spilling_config` JSON 配置（如 `{"type": "s3", "params": {...}}`）。

### 3. 具体数值样例

- 节点内存 64 GB、`object_store_memory` 默认 30% ≈ 19 GB：plasma 满后按 LRU 溢出到 `/tmp/ray/spill`；
- 溢出 4 GB 对象到本地 NVMe（2 GB/s）≈ 2 s；从 S3 读回受网络带宽限制；
- `automatic_object_spilling_enabled=true` 时无需用户干预，raylet 自动 spill + 读回，对应用透明（对象仍在，只是位置从内存变成磁盘/S3）。

> 面试一句话总结：**Plasma 放满时 raylet 触发自动 spill：让对象 owner 把对象写到本地文件系统或 S3（专用 spill worker 执行 IO），位置记录在对象目录、读取时走 SpilledObjectReader 拉回——对应用透明，用磁盘换内存容量。**

---

## 9. 分布式引用计数（Distributed Ref Counting）：对象何时释放

### 1. 现有问题

- 分布式系统里"谁还持有这个对象"无法靠单机 GC 判断：对象被传给了别的 worker、被别处引用，何时能安全删除？
- 若用集中式引用表（像 GCS 存所有引用），引用更新会变成热点且延迟高。

### 2. 方法论

**owner-based 去中心化引用计数**：每个对象有唯一 **owner**（创建它的 core worker）。owner 维护 `ReferenceCounter`（`object_id_refs_` 表：本地引用、borrower 引用、task 依赖引用、lineage 引用）：

- 对象作为 task 参数传给别的 worker → 对方 `AddBorrowedObject`；
- owner 通过 **pubsub 频道**（`WORKER_REF_REMOVED_CHANNEL` / `WORKER_OBJECT_LOCATIONS_CHANNEL`）收到"引用移除/位置更新"通知；
- 所有引用释放 → owner 触发删除回调 → 对象从 plasma/raylet 释放。

```proto
// src/ray/protobuf/pubsub.proto —— 引用计数协议的核心频道
enum ChannelType {
  WORKER_REF_REMOVED_CHANNEL = 0;      // 引用移除通知（引用计数协议核心）
  WORKER_OBJECT_LOCATIONS_CHANNEL = 1; // 对象位置更新
  GCS_ACTOR_CHANNEL = 2;
  GCS_JOB_CHANNEL = 3;
  ...
}
```

### 3. 具体数值样例

- driver put 对象 O → owner=driver，本地引用 +1；把 O 作为参数提交 task 到 worker W → W `AddBorrowedObject`，owner 记录 borrower=W；
- W 执行完释放 O 的引用 → pubsub 发 `WORKER_REF_REMOVED_CHANNEL` 通知 owner → owner 本地引用清零 → 删除 O（plasma 释放 + 通知 raylet）；
- owner 崩溃：`object_recovery_manager.cc` 用 **lineage 重建**（对象可由其创建 task 重算）——引用计数与对象目录完全去中心化，不依赖 GCS 对象表。

> 面试一句话总结：**Ray 的对象释放靠 owner-based 去中心化引用计数：每个对象的 owner 维护本地/borrower/依赖/lineage 四类引用，borrower 通过 pubsub 频道上报引用移除，引用清零即删除；owner 崩溃时用 lineage 重建——没有全局引用表，热点与延迟都压在 owner 本地。**

---

# 三、控制面协议

## 10. 本地 IPC：Worker ↔ Raylet 的 Unix socket + FlatBuffers

### 1. 现有问题

worker 与本地 raylet 的交互（注册、申请租约、请求拉对象）非常频繁，走 gRPC（HTTP/2 序列化开销）太贵；需要一个**低延迟的本地通道**。

### 2. 方法论

**Unix domain socket + FlatBuffers**（零拷贝反序列化，`src/ray/flatbuffers/node_manager.fbs`）：

```
enum MessageType:int {
  RegisterClientRequest,      // worker/driver 向 raylet 注册
  AnnounceWorkerPort,         // worker 上报自己的 gRPC 端口
  AsyncGetObjectsRequest,     // 请求 raylet 拉取远端对象（ray.get 的核心）
  CancelGetRequest,
  NotifyWorkerBlocked,        // 阻塞等待对象，raylet 释放其 CPU 资源
  NotifyWorkerUnblocked,
  WaitRequest, WaitReply,
  FreeObjectsInObjectStoreRequest,
  SubscribePlasmaReady,
}
```

- **RegisterClientRequest**：worker 启动后向本地 raylet 注册（拿 worker ID、上报资源）；
- **AsyncGetObjectsRequest**：`ray.get` 缺对象时经此请求本地 raylet 发起远端 Pull（第 7 点）；
- **NotifyWorkerBlocked/Unblocked**：worker 阻塞等对象时通知 raylet **释放其 CPU 资源**（让别的 task 用）——这是 Ray 资源利用率高的机制之一；
- FlatBuffers 的好处：**零拷贝读取**（不反序列化就能访问字段），比 protobuf 更适合高频小消息。

### 3. 具体数值样例

- 一次 `ray.get` 本地缺对象：`AsyncGetObjectsRequest`（FlatBuffers，<1 KB）经 Unix socket 到 raylet，微秒级延迟；对比 gRPC over TCP（同机回环）毫秒级；
- worker 阻塞等对象时 `NotifyWorkerBlocked` → raylet 把该 worker 占的 CPU 资源释放给其他 task——8 卡机器上 IO 密集任务不空占 CPU。

> 面试一句话总结：**worker↔本地 raylet 的高频控制消息走 Unix domain socket + FlatBuffers（零拷贝反序列化）：注册、上报端口、请求拉对象、阻塞/解除阻塞通知都在这一层——比 gRPC 快一个数量级，且 NotifyWorkerBlocked 让阻塞 worker 的 CPU 资源能被复用。**

---

## 11. gRPC 服务与端口全景：控制面协议速查

### 1. 现有问题

Ray 进程众多（GCS/raylet/worker/object manager），需要一张清晰的"谁监听哪个端口、用什么协议"地图。

### 2. 方法论

三个 gRPC 服务端 + 两个本地 socket（`src/ray/protobuf/` 下的 proto 定义）：

| 通道 | 协议 | 端口/端点 | 关键 proto |
|---|---|---|---|
| Worker ↔ 本地 Raylet | Unix socket + FlatBuffers | `raylet_socket_name` | `flatbuffers/node_manager.fbs` |
| Worker ↔ Plasma | Unix socket + 自定义二进制 | `store_socket_name` | `plasma/protocol.h` |
| Worker ↔ Raylet（远程调度） | **gRPC over TCP** | `node_manager_port` | `node_manager.proto` |
| Worker ↔ Worker（owner/依赖） | **gRPC over TCP** | worker 端口（min/max 段） | `core_worker.proto` |
| Raylet ↔ ObjectManager（对象） | **gRPC over TCP**（Push 分块 5MB） | `object_manager_port` | `object_manager.proto` |
| 各进程 ↔ GCS | **gRPC over TCP** + pubsub | `gcs_server_port`（默认 6379） | `gcs_service.proto`、`pubsub.proto` |
| 节点状态同步 | **gRPC bidi streaming**（RaySync） | node_manager 端口 | `ray_syncer.proto` |
| GPU Collective | **NCCL（用户态，cupy）** | IB/RoCE（与 Ray 无关） | `python/ray/util/collective/` |
| RDMA | **不支持** | — | — |

- **RaySync**（`src/ray/ray_syncer/`，`ray_syncer_bidi_reactor.h`）：raylet 之间用 **gRPC 双向流**同步节点状态（资源、负载），替代旧的"全量上报 GCS"模式——控制面信息增量同步；
- 端口：GCS 6379（`DEFAULT_PORT`）、dashboard 8265、dashboard agent 52365、node/object manager 默认自动分配、worker 端口在 min/max 段内。

### 3. 具体数值样例

- 一次跨节点 `ray.get` 的完整协议路径：本地 IPC（AsyncGetObjectsRequest，FlatBuffers）→ raylet 间 gRPC（Pull/Push，object_manager.proto）→ owner 侧 gRPC（GetObjectLocationsOwner，core_worker.proto）→ 本地 IPC（plasma Get）；
- 控制面：worker 注册（本地 IPC）→ GCS 记录（gRPC）；心跳（gRPC）→ GCS node 表；PG 放置（gRPC 三连：Prepare/Commit/Remove）。

> 面试一句话总结：**Ray 的协议全景 = 两条本地通道（worker↔raylet 的 FlatBuffers、worker↔plasma 的二进制）+ 五个 gRPC 面（调度/对象/worker 间/GCS/RaySync 双向流）+ 用户态 NCCL；端口约定 GCS 6379、worker 10000~19999 段、node/object manager 自动分配——控制面信息走 gRPC 增量同步（RaySync），热路径走本地 IPC。**

---

## 12. Collective 通信：NCCL（用户态，与 Ray 传输栈解耦）

### 1. 现有问题

GPU 训练需要集合通信（allreduce 等），但 Ray 的对象传输（plasma/gRPC）是 CPU 内存语义，不适用于 GPU 显存数据；NCCL 需要**绕过 Ray 的传输栈**直接在 GPU worker 之间建组。

### 2. 方法论

**Ray 对 NCCL 的支持全在 Python 用户态**（raylet/GCS 无 C++ NCCL 集成）：

```python
# python/ray/util/collective/collective.py —— 后端注册
try:
    from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
    register_collective_backend("NCCL", NCCLGroup)
except ImportError:
    pass
try:
    from ray.util.collective.collective_group.torch_gloo_collective_group import TorchGLOOGroup
    register_collective_backend("GLOO", TorchGLOOGroup)
except ImportError:
    pass
```

- `ray.util.collective`：`NCCLGroup`（基于 cupy + NCCL）与 `TorchGLOOGroup`，提供 allreduce/broadcast/allgather/reduce_scatter 等；
- 加速 DAG 的 NCCL 通道：`python/ray/experimental/channel/nccl_group.py` 直接 `from cupy.cuda import nccl` 在 GPU worker 间建 NCCL 通信组，**绕过对象存储**；
- Ray Train Torch：`TorchConfig.backend` 默认 GPU 用 nccl、CPU 用 gloo，在 worker 内调 `torch.distributed.init_process_group`——NCCL 组由 torch 自己管理。

```python
# python/ray/train/torch/config.py
class TorchConfig(BackendConfig):
    backend: Optional[str] = None
    # If set to None, nccl will be used if GPUs are requested, else gloo
```

### 3. 具体数值样例

- 8 GPU worker 做 allreduce：`ray.util.collective.allreduce(tensor, group_name=...)` → cupy 起 NCCL 组 → 8 卡间走 NVLink（机内）或 IB/RoCE（跨机）——**数据不经过 plasma**；
- Ray Train 跑 torch DDP：`TorchConfig(backend="nccl")` → 每个 worker `init_process_group("nccl", ...)`，通信由 torch.distributed 接管；
- 关键：NCCL 的物理链路（NVLink/IB/RoCE）由 NVIDIA 驱动决定，**Ray 不参与**（详见 `Communication.md`）。

> 面试一句话总结：**Ray 的 GPU 集合通信在 Python 用户态：ray.util.collective（cupy+NCCL/GLOO）与 Ray Train 的 TorchConfig（默认 GPU=nccl）直接在 GPU worker 间建 NCCL 组、绕过对象存储——Ray 只管"把 worker 放到有 GPU 的节点"，NCCL 通信本身（NVLink/IB/RoCE）由 NVIDIA 栈接管。**

---

# 四、整体架构：把组件串成一条链路

## 13. 全链路串讲：ray.put / ray.get / ray.remote 一次完整调用

### 1. 现有问题

前面是组件视角，面试还要能"一条调用串到底"。

### 2. 方法论（完整链路）

```text
ray.remote 装饰 → RemoteFunction（python/ray/remote_function.py）
  │
  ▼ __call__
core_worker.submit_task（Cython _raylet.pyx）
  │
  ▼
CCoreWorkerProcess.GetCoreWorker().SubmitTask（core_worker.cc L2056）
  ├─ 普通 task → normal_task_submitter_（task 不写 GCS 表！）
  └─ actor task → actor_task_submitter_
  │
  ▼ gRPC RequestWorkerLease（本地 raylet）
ClusterLeaseManager.ScheduleAndGrantLeases（Hybrid 本地优先）
  │
  ▼ gRPC PushTask（目标 worker 的 CoreWorkerService）
HandlePushTask → CoreWorker::ExecuteTask（core_worker.cc L2926）
  │
  ├─ 返回值 < 100KB → 内联在 task reply（owner 内存 store）
  └─ 返回值 ≥ 100KB → 写 plasma（PutInLocalPlasmaStore + PinObjectIDs）
  │
  ▼ ray.get（core_worker.cc Get L1429）
先查内存 store → 查 plasma（mmap 零拷贝）
  └─ 缺对象 → owner 查位置（GetObjectLocationsOwner）
              → 本地 IPC AsyncGetObjectsRequest → raylet
              → gRPC Pull → 远端 5MB 分块 Push → 本地 plasma → mmap 读回
```

**关键点**：
1. **普通 task 不写 GCS 表**——任务调度是"worker → 本地 raylet"的去中心化快路径，GCS 只做元数据/可观测；
2. **返回值按 100KB 分流**：小对象内联（省 plasma 往返），大对象进 plasma（零拷贝共享）；
3. **ray.get 三级查找**：内存 store → 本地 plasma → 远端拉取，逐级"便宜到贵"；
4. **owner 死亡**：`object_recovery_manager.cc` 用 lineage 重建对象。

### 3. 具体数值样例

- driver 提交 1000 个 task，每个返回 256 KB 数组：返回值写 plasma（256 KB × 1000 = 250 MB 共享内存），各 task 的 `ray.get` 全零拷贝；
- 若返回 50 KB：内联在 task reply，不碰 plasma，省一次本地 IPC；
- 远端依赖：task 参数引用 node2 上的对象 → `ray.get` 触发跨节点 Pull（5 MB 分块），完成后本地 plasma 缓存——**同对象二次 get 零拷贝命中**。

> 面试一句话总结：**一次 ray.remote 调用 = Python 装饰 → Cython 提交 → C++ SubmitTask → 本地 raylet 租约（本地优先）→ PushTask 执行 → 返回值按 100KB 分流（内联/plasma）；ray.get 三级查找（内存 store → plasma → 远端拉取）逐级变贵；普通任务不写 GCS 表，控制面不挡数据流——这就是 Ray"快"的架构本质。**

---

## 14. Ray 作为 RL 训练底座：verl 怎么跑在 Ray 上

### 1. 现有问题

verl（本项目训练框架）的 PPO/GRPO 需要：多个训练 worker（数据并行）、rollout 引擎（vLLM）、参考模型、reward 模型——它们怎么被编排？对应实习"管控面：基于 Ray 的 AgenticRL 训练作业（Kubernetes CRD）"。

### 2. 方法论

**verl 源码里没有 Ray 的代码级集成**（搜 `verl` 只有文档命中）——verl 是跑在 Ray 集群之上的第三方框架，Ray 提供的底座是 **Ray Train 的 WorkerGroup = 一组资源受限的 Actor**：

```python
# python/ray/train/_internal/worker_group.py —— 训练 worker = 带资源的 actor
self._remote_cls = ray.remote(
    num_cpus=self.num_cpus_per_worker,
    num_gpus=self.num_gpus_per_worker,
    memory=self.memory_per_worker,
    resources=_resources_per_worker,
)(self._base_cls)
self.start()
```

- **Actor 资源申请**：每个训练 worker（actor/rollout/ref/reward）通过 `ray.remote(num_gpus=...)` 申请 GPU，Ray 保证放置与隔离；
- **状态传递**：权重/checkpoint 通过共享文件系统或对象存储（plasma）传递，actor 常驻免重建；
- **KubeRay / RayCluster**：verl 官方文档展示在 Kubernetes 上以 RayCluster CRD 跑 PPO 训练——对应实习"AgenticRL 训练作业（Kubernetes CRD 资源）"：KubeRay Operator 管理 RayCluster 生命周期，head/worker 组各为 Pod；
- 实习亮点"并行创建 RayCluster Head 和 TrajStore 两条网络访问链路"：作业运行态给 head（提交训练/看日志）和 TrajStore（看轨迹数据）分别暴露公网 URL，就是基于 RayCluster 的 Service/Ingress 网络暴露。

### 3. 具体数值样例

- verl GRPO 作业：`trainer.ray` worker 组（如 4×GPU）、`rollout.ray` vLLM 引擎组（dp×tp 个引擎）、`ref.ray` 参考模型组、`reward.ray` 奖励组——各自是 `ray.remote(num_gpus=...)` 的 actor 组；
- 每轮训练：trainer 更新权重 → 写共享 checkpoint 目录 → rollout 引擎 `update_weights_from_disk` 重载 → 下一轮采集（对应 SGLang.md 第 8 点的"权重热更新"）；
- 双机部署：node1/node2 各起 raylet，GCS 在 head（node1），`num_gpus=2` 的 actor 落在 node2——Ray 自动处理跨机调度与对象传输。

> 面试一句话总结：**verl 不是 Ray 的代码级集成，而是跑在 Ray 之上的训练框架：用 Ray Train 的 WorkerGroup（=一组 ray.remote(num_gpus=...) 的常驻 actor）编排 trainer/rollout/ref/reward，权重经共享文件系统传递；KubeRay 的 RayCluster CRD 让训练作业成为 Kubernetes 资源——这就是实习"AgenticRL 训练作业（CRD）+ 双网络链路暴露"的底座。**

---

# 附：传输协议速查表（面试可直接背）

| 通道 | 协议 | 端口/端点 | 关键文件 |
|---|---|---|---|
| Worker ↔ 本地 Raylet | Unix socket + FlatBuffers | `raylet_socket_name` | `src/ray/flatbuffers/node_manager.fbs` |
| Worker ↔ Plasma | Unix socket + 自定义二进制 | `store_socket_name` | `plasma/protocol.h`、`client.cc` |
| Worker ↔ Raylet（远程调度） | gRPC over TCP | `node_manager_port` | `protobuf/node_manager.proto` |
| Worker ↔ Worker（owner/依赖） | gRPC over TCP | worker 端口（min/max 段） | `protobuf/core_worker.proto` |
| Raylet ↔ ObjectManager（对象） | gRPC over TCP（Push 分块 5MB，unary） | `object_manager_port` | `protobuf/object_manager.proto` |
| 各进程 ↔ GCS | gRPC over TCP + pubsub | `gcs_server_port`（默认 6379） | `protobuf/gcs_service.proto`、`pubsub.proto` |
| Raylet ↔ Raylet（状态同步） | gRPC bidi streaming（RaySync） | node_manager 端口 | `src/ray/ray_syncer/` |
| GPU Collective | NCCL（用户态，cupy） | NVLink/IB/RoCE | `python/ray/util/collective/` |
| RDMA | **不支持**（无源码实现） | — | — |
| 对象溢出 | owner 侧 spill worker → filesystem/S3 | — | `local_object_manager.cc` |

**核心数字**：GCS 端口 6379；`object_manager_default_chunk_size=5MB`；`max_direct_call_object_size=100KB`（小对象内联）；`max_grpc_message_size=512MB`；`scheduler_spread_threshold=0.5`、`scheduler_top_k_fraction=0.2`；plasma 默认节点内存 30%；worker 端口 10000~19999。

**面试金句**："Ray 的控制面（GCS+Raylet）走 gRPC、本地热路径走 Unix socket+FlatBuffers、同节点对象走 plasma mmap 零拷贝、跨节点走 gRPC 5MB 分块拉取、GPU 集合通信走用户态 NCCL——原生不支持 RDMA；verl 这类训练框架就是跑在它上面的资源受限 actor 组（Ray Train WorkerGroup / RayCluster CRD）。"
