# Kubernetes 核心概念完全指南（八股 + 怎么用）

> **一句话知识框架**：K8s 的本质是**"声明式期望状态 + 控制器不断把实际状态收敛到期望状态"**。所以学 K8s 只要抓住三件事：**① 我用什么资源（CRD/Spec）声明期望；② 谁在 reconcile 它（controller/operator）；③ 它在哪个层面落地（调度到哪个 node、用哪个网络插件、挂哪个存储）**。本文按"工作负载 → 网络 → 存储配置 → 元数据 → 扩展 → 架构串讲"组织，每个概念都回答"它解决什么问题、字段怎么写、怎么用"。
>
> **素材来源**：本地 Kubernetes 源码仓（**v1.34.11**，见 `kubernetes/CHANGELOG/CHANGELOG-1.34.md`）——`staging/src/k8s.io/api/core/v1/types.go`、`pkg/kubelet/kubelet.go`、`pkg/scheduler/schedule_one.go`、`staging/src/k8s.io/apiserver/pkg/server/config.go`、`pkg/controller/deployment/`。配套阅读：`K8S-KubeRay.md`（KubeRay 实操）、`Ray.md`、`Colocate-vs-Disaggregate.md`。

---

## 一、工作负载

## 1. Pod：K8s 的最小调度单位

### 1. 现有问题

**为什么不直接调度容器，而要多一层 Pod？** 因为真实业务经常需要"几个进程必须一起跑、共享网络和存储"。典型例子：

- **主容器 + 日志采集 sidecar**：两者要看到同一个日志目录；
- **主容器 + agent**：agent 要访问主容器监听的端口；
- **初始化任务**：跑完建表脚本才启动主服务。

如果直接调度容器，就得自己解决"它们必须落到同一台机器、共享同一个网络命名空间、共享同一个卷"——这正是 Pod 抽象掉的复杂度。

### 2. 方法论

**Pod = 一个或多个容器 + 共享的命名空间 + 共享的卷**。官方定义（`staging/src/k8s.io/api/core/v1/types.go:4358`）：

```go
// PodSpec is a description of a pod.
type PodSpec struct {
	// List of volumes that can be mounted by containers belonging to the pod.
	Volumes []Volume `json:"volumes,omitempty" ...`
	// List of initialization containers belonging to the pod.
	// Init containers are executed in order prior to containers being started. If any
	// init container fails, the pod is considered to have failed...
	InitContainers []Container `json:"initContainers,omitempty" ...`
	// List of containers belonging to the pod.
	// Containers cannot currently be added or removed.
	// There must be at least one container in a Pod.
	Containers []Container `json:"containers" ...`
	...
}
```

**Pod 里"共享什么、独立什么"**（这是最常考的一题）：

| 资源 | 同一个 Pod 内的容器 | 依据 |
|---|---|---|
| **Network namespace** | ✅ 共享 —— 同一个 IP、同一个端口空间（所以同 Pod 内容器**不能用相同端口**，互相访问用 `localhost`） | Pod 的 sandbox（pause）容器持有 netns |
| **UTS namespace** | ✅ 共享 —— 同一个 hostname | 同上 |
| **IPC namespace** | ✅ 共享（默认） | 同上 |
| **Volume** | ✅ 共享 —— Pod 级资源，各容器挂载到自己的挂载点 | `spec.volumes` 是 Pod 级 |
| **PID namespace** | ❌ 默认独立（`shareProcessNamespace: true` 可共享） | 各容器有自己的 PID 1 |
| **Mount namespace / 文件系统** | ❌ 独立 —— 各容器镜像自己的 rootfs | 容器级 |
| **CPU/内存 cgroup** | ❌ 独立计量，但**调度时按 Pod 求和** | `resources` 在容器级 |

**Pod 的完整组成**（回答"一个 Pod 有哪些部分"）：

```
Pod
├─ metadata（name/namespace/labels/annotations/ownerReferences）
├─ spec
│   ├─ initContainers[]     顺序执行、全部成功才启动主容器（每个都能有自己的镜像/命令/卷）
│   ├─ containers[]         ★ 至少一个
│   │   ├─ image / command / args
│   │   ├─ ports[]          （仅声明，供 Service/探针引用）
│   │   ├─ env / envFrom    （引用 ConfigMap/Secret）
│   │   ├─ resources        requests / limits
│   │   ├─ volumeMounts[]   把 spec.volumes 挂进容器
│   │   ├─ livenessProbe / readinessProbe / startupProbe
│   │   └─ securityContext / lifecycle（postStart/preStop）
│   ├─ volumes[]            ★ Pod 级：emptyDir / hostPath / configMap / secret / PVC / projected
│   ├─ restartPolicy        Always / OnFailure / Never
│   ├─ serviceAccountName
│   ├─ nodeSelector / affinity / tolerations / topologySpreadConstraints
│   ├─ dnsPolicy / hostNetwork / hostPID
│   └─ terminationGracePeriodSeconds
└─ status
    ├─ phase                Pending / Running / Succeeded / Failed / Unknown
    ├─ conditions[]         PodScheduled / Initialized / ContainersReady / Ready
    ├─ containerStatuses[]  ready / restartCount / state（waiting/running/terminated）
    └─ podIP / hostIP / startTime
```

**三种探针的区别（高频）**：

| 探针 | 失败后果 | 用途 |
|---|---|---|
| **startupProbe** | 一直重启容器 | 慢启动应用（JVM/大模型加载），成功后其余探针才开始 |
| **livenessProbe** | **重启容器** | 死锁/假死检测（进程还在但不可用） |
| **readinessProbe** | **从 Service Endpoints 摘除**（不重启） | 暂时不可用（预热中、依赖未就绪） |

**四种多容器模式**（面试加分）：

| 模式 | 干什么 | 例子 |
|---|---|---|
| **Init container** | 主容器启动**前**跑一次，顺序执行 | 等待数据库就绪、生成配置、clone 代码 |
| **Sidecar** | 与主容器**并行**、增强它 | 日志采集（fluent-bit）、服务网格代理（Envoy） |
| **Ambassador** | 代理主容器的**出向**流量 | 连接外部数据库的代理 |
| **Adapter** | 转换主容器的**出向输出** | 把自定义指标转成 Prometheus 格式 |

### 3. 具体数值样例

**一个典型的训练 Pod**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: train-worker-0
  labels: {app: trainer, role: worker}
spec:
  initContainers:
  - name: wait-for-data          # ① 等数据集挂载就绪
    image: busybox
    command: ["sh", "-c", "until [ -f /data/.ready ]; do sleep 5; done"]
    volumeMounts: [{name: dataset, mountPath: /data}]
  containers:
  - name: trainer                # ② 主容器
    image: my-trainer:v1
    resources:
      requests: {cpu: "96", memory: "1Ti", nvidia.com/gpu: "8"}
      limits:   {cpu: "96", memory: "1Ti", nvidia.com/gpu: "8"}
    volumeMounts:
    - {name: dshm,    mountPath: /dev/shm}      # NCCL 共享内存
    - {name: ckpt,    mountPath: /ckpt}         # checkpoint
    - {name: dataset, mountPath: /data, readOnly: true}
    envFrom: [{configMapRef: {name: train-config}}]
  - name: log-shipper            # ③ sidecar：采集日志
    image: fluent-bit:latest
    volumeMounts: [{name: varlog, mountPath: /var/log}]
  volumes:
  - {name: dshm,    emptyDir: {medium: Memory, sizeLimit: 64Gi}}
  - {name: ckpt,    persistentVolumeClaim: {claimName: ckpt-pvc}}
  - {name: dataset, persistentVolumeClaim: {claimName: dataset-pvc, readOnly: true}}
  - {name: varlog,  emptyDir: {}}
  restartPolicy: Never            # 训练任务：失败不自动重启，靠上层 Job 重试
  serviceAccountName: trainer-sa
  nodeSelector: {node-pool: a100}
  tolerations:
  - {key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule"}
```

**资源账**（回答"这个 Pod 要多少资源"）：

- 调度器看的是 **Pod 内所有容器 requests 之和**：主容器 96 CPU + sidecar 0.1 CPU ≈ **96.1 CPU**；
- **init container 的 requests 不累加**——取"所有 init container 的最大值"与"所有普通容器之和"的**较大者**（`types.go:4373-4376` 注释原文说明）；
- 所以这台机器就是"一个 Pod 占满整机"（requests = limits = 整机规格），这也是 KubeRay 样例推荐的形态。

> 面试一句话总结：**Pod 是 K8s 的最小调度单位，本质是"一组共享 Network/UTS/IPC namespace 和卷的容器"；同 Pod 内容器共享 IP 和端口空间（互访用 localhost，不能撞端口），但 PID/mount namespace 和 cgroup 独立；完整组成 = metadata + spec（initContainers/containers/volumes/restartPolicy/affinity/SA…）+ status（phase/conditions/containerStatuses）；三种探针分工是 startup 管启动、liveness 管重启、readiness 管摘流量；多容器四种模式是 init/sidecar/ambassador/adapter。**

---

## 2. 工作负载控制器：为什么不能只裸跑 Pod

### 1. 现有问题

**裸 Pod 有三个致命问题**：① 挂了没人拉起；② 想扩到 3 副本要手工复制；③ 滚动升级、回滚全要自己写。所以生产上从不直接创建 Pod，而是创建**控制器**，由控制器去管理 Pod。

### 2. 方法论

| 控制器 | 它解决的问题 | 管理的 Pod 有什么特点 |
|---|---|---|
| **ReplicaSet** | 保证副本数 = N（自愈 + 扩缩） | 无状态、可互换、名字随机 |
| **Deployment** | 在 RS 之上做**滚动升级/回滚/暂停**（每次改模板生成新 RS） | 无状态；**生产最常用** |
| **StatefulSet** | 需要**稳定网络标识 + 稳定存储** | Pod 名有序（`-0/-1/-2`）、各自 PVC、按序启停 |
| **DaemonSet** | 每个（或指定）节点跑**恰好一个** | 节点级 agent（device plugin、CNI、日志采集） |
| **Job** | 跑完就结束的**批处理任务**（保证成功 N 次） | `completions`/`parallelism`/`backoffLimit` |
| **CronJob** | 按 cron 周期创建 Job | 定时任务 |

**Deployment 的滚动升级机制**（面试常问"滚动更新怎么做的"）：Deployment 模板变了 → controller 建**新 ReplicaSet** → 逐步把新 RS 的 replicas 加、旧 RS 减，由两个参数控制：

```yaml
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # 最多超出期望副本数 2 个（先建后删）
      maxUnavailable: 0    # 升级期间不允许不可用（保证容量）
  revisionHistoryLimit: 5  # 保留几个旧 RS 供回滚
```

- `maxSurge=2, maxUnavailable=0` → **先扩后缩，零中断**（滚动过程中副本数在 10~12 之间）；
- 回滚：`kubectl rollout undo deploy/x --to-revision=3`（就是切回旧 RS）；
- 关键实现：Deployment controller 从旧 RS 中"**认领**"（adopt）符合新模板的 Pod，并给 RS 打 `pod-template-hash` label 来区分版本。

### 3. 具体数值样例

**训练任务该用哪个？**

| 场景 | 用什么 | 原因 |
|---|---|---|
| 反复提交的一次性训练 | **Job**（或 RayJob / 自定义 CRD） | 有明确的"完成"语义、可设重试次数 |
| 常驻推理服务 | **Deployment + Service**（或 RayService） | 需要自愈、滚动升级、多副本 |
| 每个节点都要跑的 GPU 监控 | **DaemonSet** | 节点级 agent |
| 分布式训练的每个 worker 有稳定编号 | **StatefulSet**（或交给 Ray/CRD） | 需要 `-0/-1/…` 稳定标识 |
| 每天定时评测 | **CronJob** | 周期调度 |

**一个数值样例（为什么训练不用 Deployment）**：8 个 worker 的分布式训练如果用 Deployment（`replicas: 8`），滚动升级时会**先建新 Pod 再删旧 Pod**（maxSurge=2）——新 Pod 拿不到 GPU（被旧 Pod 占着）→ Pending → 整个升级卡住；而训练的正确语义是"**整组一起换**"（要么全换、要么不动）。**这就是训练任务要用 Job/CRD + gang scheduling 而不是 Deployment 的原因。**

> 面试一句话总结：**裸 Pod 不自愈、不能扩缩、没有升级策略，所以生产用控制器：Deployment（无状态 + 滚动升级/回滚，靠切换 ReplicaSet 实现）> ReplicaSet > Pod；StatefulSet 给稳定网络标识与独立存储；DaemonSet 每节点一个；Job/CronJob 管批处理；训练任务不能用 Deployment——滚动升级的"先扩后缩"会因 GPU 被旧 Pod 占用而卡死，训练需要"整组一起换"的语义。**

---

## 3. 调度：Pod 怎么落到某个节点

### 1. 现有问题

一个 8 卡 Pod 和一堆 1 卡 Pod 抢同一批机器时，怎么保证大 Pod 能被满足？怎么避免所有 Pod 挤在一个节点？怎么让 GPU 节点只跑 GPU 任务？这些都是**调度**要解决的问题。

### 2. 方法论

**调度的输入是 Pod 的 `requests`，不是 `limits`**（这点最容易错）。核心流程（`pkg/scheduler/schedule_one.go:568` `schedulePod`）：

```
① 过滤（Filter）：把所有不满足硬约束的节点剔掉
   ├─ 资源够不够（requests 求和 ≤ 节点可分配）
   ├─ nodeSelector / nodeAffinity（硬亲和）
   ├─ 污点容忍（Taints/Tolerations）
   ├─ 端口冲突、卷拓扑（PV 的 nodeAffinity）
   └─ 拓扑分布约束（topologySpreadConstraints）
② 打分（Score）：给存活节点打分（NodeResourcesFit / ImageLocality / InterPodAffinity / …）
③ 选中最高分节点 → 绑定（Bind）→ 写回 Pod.spec.nodeName
④ 若无节点可行 → 触发抢占（Preemption）：踢掉低优先级 Pod 腾位
```

**调度框架的扩展点**（`pkg/scheduler/framework/`，顺序执行）：

```
PreEnqueue → QueueSort → PreFilter → Filter → PostFilter(抢占) 
→ PreScore → Score → Reserve → Permit → PreBind → Bind → PostBind
```

**三大类约束的写法**：

```yaml
# ① 节点选择：硬约束
nodeSelector: {disktype: ssd}                 # 最简单，等价于 nodeAffinity 的 In
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:      # 硬：不满足就不调度
      nodeSelectorTerms:
      - matchExpressions:
        - {key: node-pool, operator: In, values: [a100]}
    preferredDuringSchedulingIgnoredDuringExecution:     # 软：打分时加分
      - weight: 100
        preference:
          matchExpressions:
          - {key: zone, operator: In, values: [zone-a]}

# ② Pod 之间：反亲和（一台机器只放一个同组 Pod）——大规模训练必配
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels: {ray.io/group: gpu-workers}
      topologyKey: kubernetes.io/hostname

# ③ 节点排斥：污点 + 容忍
# 给节点打污点（运维做）：kubectl taint nodes gpu-1 dedicated=gpu:NoSchedule
tolerations:
- {key: dedicated, operator: Equal, value: gpu, effect: NoSchedule}
```

**QoS 三级**（由 requests/limits 的组合决定，影响 OOM 时的驱逐顺序）：

| QoS | 条件 | 被驱逐优先级 |
|---|---|---|
| **Guaranteed** | 每个容器都设了 limits，且 **requests == limits** | 最后被驱逐 |
| **Burstable** | 至少一个容器设了 requests，但不满足 Guaranteed | 中间 |
| **BestEffort** | 所有容器都没设 requests/limits | **最先被驱逐** |

**训练任务必须用 Guaranteed**（requests = limits）——既避免被驱逐，也避免"limits 远大于 requests 导致超卖后被 OOMKill"。

**`requests` 与 `limits` 的本质区别**：

- **requests**：调度依据 + cgroup 的 `cpu.shares`（相对权重）；
- **limits**：cgroup 的硬上限。**CPU 超了会被限流（throttle）**，**内存超了会被 OOMKill**。

### 3. 具体数值样例

**场景**：3 台机器，每台 8 GPU / 96 CPU / 1Ti 内存。要调度一个 8-worker 的训练任务，每 worker 要 8 GPU。

```
节点剩余：node-2: 8GPU 空闲 | node-3: 7GPU 空闲 | node-4: 8GPU 空闲

没有 gang scheduling：
  worker-0 → node-2 ✓
  worker-1 → node-4 ✓
  worker-2 → node-3 ✗（只要 7 GPU）→ Pending
  ...
  结果：2 个 worker 起来了在空转等，1 个永远 Pending → 整个任务死锁

有 gang scheduling（Volcano/KAI/coscheduling）：
  调度器发现"这个 PodGroup 需要 3×8=24 GPU，当前只有 23" 
  → 一个都不调度，全部 Pending，等资源够了整体起来
```

**这个例子就是"为什么多机训练必须 gang scheduling"**——**默认调度器是"逐个 Pod"贪心的，它不关心 Pod 之间的整体性**。

**QoS 的数值含义**：

- 若 Pod 内存 `requests: 512Mi / limits: 2Gi` → Burstable。容器实际用到 1.8Gi 没问题，但节点内存紧张时**先被驱逐**；
- 若 `requests: 2Gi / limits: 2Gi` → Guaranteed，实际超过 2Gi → **直接 OOMKill 容器**（不会影响别的 Pod）。

> 面试一句话总结：**调度看 requests（不是 limits），流程是 Filter（资源/亲和/污点/拓扑）→ Score → Bind，失败触发抢占；三大约束是 nodeSelector/nodeAffinity（节点）、podAntiAffinity（Pod 之间，一台机一个训练 Pod）、taints/tolerations（节点排斥）；QoS 由 requests/limits 组合决定（Guaranteed = 都设且相等，最后被驱逐），训练任务必须 Guaranteed；多机训练必须 gang scheduling——默认调度器逐个 Pod 贪心，会让 7/8 个 worker 起来空转、任务死锁。**

---

## 二、网络

## 4. Service 与 kube-proxy：Pod IP 会变，怎么稳定访问

### 1. 现有问题

Pod 是"用完即弃"的：重建后 **IP 会变**、副本数会变。如果其他服务直接连 Pod IP，一次滚动升级就全断了。所以需要一个**稳定的虚拟地址 + 负载均衡**。

### 2. 方法论

**Service = 一组 Pod 的稳定访问入口**。它靠 **label selector 找到 Pod**，维护一个 **Endpoints（或 EndpointSlice）列表**，并分配一个**虚拟 IP（ClusterIP）**。

**四种 Service 类型**：

| 类型 | 暴露范围 | 用法 |
|---|---|---|
| **ClusterIP**（默认） | 仅集群内 | 服务间调用（`http://agl-store:4747`） |
| **NodePort** | 每个节点的一个高端口（30000-32767） | 临时对外暴露，`<nodeIP>:<nodePort>` |
| **LoadBalancer** | 云厂商 LB（外部 IP） | 生产对外，**云上用这个** |
| **ExternalName** | DNS CNAME | 把外部服务映射成集群内名字 |

**`ClusterIP` 是虚拟的——它不属于任何网卡**。真正干活的是每个节点上的 **kube-proxy**（或 eBPF 实现），它把"访问 ClusterIP:Port"的流量改写成"访问某个 Pod IP:Port"。三种实现：

| 模式 | 机制 | 特点 |
|---|---|---|
| **iptables**（默认，老） | 写一堆 iptables 规则做 DNAT | 规则数随 Service 数**线性增长**，几千个 Service 时更新慢 |
| **IPVS** | 内核 LVS 的哈希表 | **O(1) 查找**，大规模集群更优；支持 rr/lc/sh/sed 等调度算法 |
| **nftables**（新，1.31+） | 用 nft 替代 iptables | 替代 iptables 的演进方向 |
| **eBPF**（Cilium 等） | 在 socket/TC 层直接改地址，**绕过 kube-proxy** | 性能最好，还能省掉一层 NAT |

**Headless Service**（`clusterIP: None`）：不做负载均衡、不分配 VIP，DNS 直接返回**所有 Pod IP**。**StatefulSet 和分布式训练常用**——因为客户端需要知道"所有 peer 的地址"（而不是随机连一个）。

**`port` / `targetPort` / `nodePort` 三者关系**（高频混淆点）：

```
Client ──► Service.port (4747) ──► Pod.targetPort (4747)
                      │
           nodePort (32474) ← 仅 NodePort/LoadBalancer 类型才有
```

### 3. 具体数值样例

```yaml
apiVersion: v1
kind: Service
metadata:
  name: agl-store
spec:
  type: ClusterIP                    # 集群内
  selector: {app: agl-store}         # ★ 靠 label 找 Pod
  ports:
  - name: http
    port: 4747                       # Service 对外暴露的端口
    targetPort: 4747                 # 转发到 Pod 的哪个端口（可写名字）
    protocol: TCP
```

**验证它到底连到哪些 Pod**：

```bash
kubectl get endpoints agl-store          # 老 API：Endpoints
kubectl get endpointslice -l kubernetes.io/service-name=agl-store   # 新 API：EndpointSlice
# NAME        ENDPOINTS                              AGE
# agl-store   10.244.1.5:4747,10.244.2.7:4747        5m
```

**排查"Service 连不上"的三步**：

1. `kubectl get endpoints <svc>` —— **空的说明 selector 没匹配到 Pod**（最常错：label 写错），此时不是网络问题；
2. `kubectl get pods -l <selector>` —— 确认 Pod 是否 Ready（**未 Ready 的 Pod 不会进 Endpoints**）；
3. 在 Pod 里 `curl <service>.<namespace>.svc.cluster.local:<port>` —— 排除 DNS。

> 面试一句话总结：**Service 解决"Pod IP 会变"的问题：用 selector 找 Pod、维护 Endpoints、给出稳定的 ClusterIP（虚拟 IP，不属于任何网卡），由每个节点的 kube-proxy 做 DNAT；四种类型 ClusterIP/NodePort/LoadBalancer/ExternalName，Headless（clusterIP: None）直接返回所有 Pod IP 供 StatefulSet/分布式训练发现 peer；kube-proxy 有 iptables（规则数线性增长）/IPVS（哈希表 O(1)）/nftables/eBPF 四种实现；排查第一步永远是 `kubectl get endpoints`——空的说明 selector 或 readiness 有问题，不是网络问题。**

---

## 5. Ingress 与 Gateway API：七层入口

### 1. 现有问题

用 `NodePort`/`LoadBalancer` 暴露服务有两个问题：① 每个服务一个 LB → **一个公网 IP 一个服务，很贵**；② 它们工作在**四层**，做不了"按域名/路径路由、TLS 终止"。

### 2. 方法论

**Ingress = 七层（HTTP/HTTPS）路由规则的声明**，由 **Ingress Controller**（Nginx / Traefik / Envoy / ALB 等）真正实现。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agl-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx            # ★ 指定由哪个 Controller 处理
  tls:
  - hosts: [store.example.com]
    secretName: store-tls            # 证书放 Secret
  rules:
  - host: store.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service: {name: agl-store, port: {number: 4747}}
```

**Ingress / Ingress Controller / Gateway API 的关系**：

| | 是什么 | 谁来干活 |
|---|---|---|
| **Ingress 资源** | 只是一份路由规则的**声明** | ——（纯数据） |
| **Ingress Controller** | 一个 Pod（Nginx/Envoy…），**watch Ingress 并重载自己的配置** | 真正的代理 |
| **Gateway API** | Ingress 的**继任者**：拆成 `GatewayClass` / `Gateway` / `HTTPRoute`，把"基础设施管理员"和"应用开发者"的职责分开、支持更多协议（gRPC/TCP/TLS） | 同上的 Controller（如 Istio/Envoy Gateway） |

**关键认知**：**装完 K8s 是没有 Ingress Controller 的**——只有 Ingress 资源没有任何作用（很多人踩这个坑）。

### 3. 具体数值样例

**为什么大公司喜欢 Gateway API**：Ingress 的注解是**控制器私有**的（`nginx.ingress.kubernetes.io/...` 换到 Traefik 就不生效），而 Gateway API 是**标准化 CRD**，可移植。

**与 KubeRay 的关联**：KubeRay 样例里有一整套 Ingress 样例（`ingress-rayclient-tls.yaml`、`ray-cluster.separate-ingress.yaml`、`ray-cluster-alb-ingress.yaml`、`ray-cluster-agc-gatewayapi.yaml`）——**它们做的正是"把 Ray 的 Dashboard/Client 端口用 Ingress 暴露出去"**，只不过从 `port-forward` 换成了正式的七层入口。

> 面试一句话总结：**Ingress 只是七层路由的声明，必须配 Ingress Controller（Nginx/Traefik/Envoy）才生效（装完 K8s 默认没有）；它解决"每个服务一个 LB 太贵 + 四层做不了按域名/路径路由和 TLS 终止"的问题；Gateway API（GatewayClass/Gateway/HTTPRoute）是它的继任者，标准化、职责分离、支持更多协议；Ingress 的注解是控制器私有的，换控制器不生效——这是 Gateway API 出现的直接动因。**

---

## 6. NetworkPolicy 与 CNI：Pod 网络是怎么实现的

### 1. 现有问题

K8s 默认网络模型是"**所有 Pod 可以互相访问**"（flat network）。但生产环境必须能限制"只有 runner 能访问 store"、"数据库不允许任何外部访问"。

### 2. 方法论

**CNI（Container Network Interface）负责"把 Pod 接上网"**：

| 步骤 | 做什么 |
|---|---|
| 1 | kubelet 创建 Pod sandbox（pause 容器） |
| 2 | kubelet 调 CNI 插件的 `ADD` |
| 3 | CNI 创建 veth pair：一端在 Pod 的 netns（视为 eth0），一端在宿主机的网桥/overlay |
| 4 | 从 IPAM 分配 Pod IP、写路由 |
| 5 | 跨节点通信由 CNI 自己解决（VXLAN/IPIP overlay、BGP、或云厂商的 ENI 方案） |

**主流 CNI**：Calico（BGP/策略强）、Cilium（eBPF，性能与可观测性最好）、Flannel（简单 overlay）、云厂商 CNI（AWS VPC CNI / 阿里 Terway）。

**K8s 网络的四条基本要求**（面试必背）：

1. 所有 Pod 之间**无需 NAT** 即可互通；
2. 所有 Node 与 Pod 之间**无需 NAT** 即可互通；
3. Pod 看到的自己的 IP = 别人看到的它的 IP；
4. （因此）**Pod IP 在全集群唯一**。

**NetworkPolicy = Pod 级的防火墙**，靠 label 选 Pod、声明进出方向的允许规则：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: store-allow-runner}
spec:
  podSelector: {matchLabels: {app: agl-store}}     # ★ 作用在哪些 Pod 上
  policyTypes: [Ingress]
  ingress:
  - from:
    - podSelector: {matchLabels: {app: agl-runner}}  # 只允许 runner
    ports: [{protocol: TCP, port: 4747}]
```

**关键语义**：**一旦某个 Pod 被任何 NetworkPolicy 选中，它就变成"默认拒绝"，只放行显式允许的流量**（没被选中的 Pod 仍然全通）。**这条是网络策略最容易理解错的地方。**

`ray-cluster.network-policy-deny-all.yaml` 就是"先默认拒绝一切，再按需放行"的范例。

### 3. 具体数值样例

**训练集群为什么需要 NetworkPolicy**：Ray 的 **Client 端口 10001 和 Dashboard 8265 默认没有任何鉴权**——只要能用网络访问到，就能提交任意任务、执行任意代码。所以：

```
默认状态：集群内任何 Pod → 可以访问 ray-head:10001 → 任意代码执行
加 NetworkPolicy 后：
  允许：runner Pod → ray-head:10001
  允许：monitoring Pod → ray-head:8265
  拒绝：其余一切
```

**注意 NetworkPolicy 需要 CNI 支持**——Flannel 默认**不支持**，Calico/Cilium 支持。写了策略但 CNI 不支持 = 策略静默失效（很危险）。

> 面试一句话总结：**CNI 负责把 Pod 接上网（veth pair + IPAM + overlay/BGP），K8s 网络四条要求是"Pod 之间、Node 与 Pod 之间都无需 NAT 互通，Pod IP 全集群唯一"；NetworkPolicy 是 Pod 级防火墙，语义是"被选中即默认拒绝、只放行显式允许"——所以一条 deny-all 就能锁死一个 namespace；注意它需要 CNI 支持（Flannel 不支持），否则策略静默失效；训练集群特别需要它，因为 Ray 的 10001/8265 默认无鉴权。**

---

## 三、配置与存储

## 7. ConfigMap 与 Secret：配置和镜像分离

### 1. 现有问题

把配置（数据库地址、超参）和密码写进镜像，会导致"改一个配置就要重新打镜像"、"密码泄露在镜像层里"。

### 2. 方法论

```yaml
apiVersion: v1
kind: ConfigMap
metadata: {name: train-config}
data:
  batch_size: "32"                    # 值必须是字符串
  train.yaml: |                       # 也可以塞整个文件
    lr: 1e-5
    epochs: 5
---
apiVersion: v1
kind: Secret
metadata: {name: db-secret}
type: Opaque
data:
  password: cGFzc3dvcmQ=              # ★ base64（不是加密！只是编码）
stringData:                           # 这个字段可以直接写明文，API 会帮你转 base64
  username: admin
```

**四种注入方式**：

```yaml
# ① 环境变量（单个 key）
env:
- name: BATCH_SIZE
  valueFrom: {configMapKeyRef: {name: train-config, key: batch_size}}
- name: DB_PASSWORD
  valueFrom: {secretKeyRef: {name: db-secret, key: password}}

# ② 全部 key 变环境变量
envFrom:
- configMapRef: {name: train-config}
- secretRef: {name: db-secret}

# ③ 挂成文件（卷）—— 推荐，配置改了能自动更新（约 1 分钟内）
volumeMounts: [{name: cfg, mountPath: /etc/train}]
volumes:
- name: cfg
  configMap: {name: train-config}

# ④ 只挂部分 key + 改文件名
volumes:
- name: cfg
  configMap:
    name: train-config
    items: [{key: train.yaml, path: config.yaml}]
```

**三个关键区别**：

| | ConfigMap | Secret |
|---|---|---|
| 内容 | 明文 | base64 编码（**默认不加密**） |
| 存储 | etcd 明文 | etcd 明文，除非开了 **encryption at rest** |
| 挂载时 | 普通文件 | **tmpfs（内存）**，不落盘 |

**`Secret` 的真相（面试常被追问）**：**base64 只是编码，不是加密**。任何能 `kubectl get secret -o yaml` 的人都能解出来。真正的保护来自 **RBAC**（限制谁能读 Secret）和 **etcd 加密**（`EncryptionConfiguration`）。

**第三个坑：环境变量方式的 Secret 不会更新**——env 在容器启动时注入一次，Secret 后续更新感知不到；而**卷挂载方式会更新**（kubelet 定期同步）。所以"需要热更新的配置用卷、只在启动读一次的用 env"。

### 3. 具体数值样例

**一个"配置改了就自动生效"的完整写法**：

```yaml
containers:
- name: trainer
  image: my-trainer:v1
  volumeMounts:
  - {name: cfg, mountPath: /etc/train, readOnly: true}
  command: ["sh", "-c", "python train.py --config /etc/train/train.yaml"]
volumes:
- name: cfg
  configMap:
    name: train-config
    items: [{key: train.yaml, path: train.yaml}]
```

改配置：`kubectl edit cm train-config` → **kubelet 在 ~1 分钟内把新内容写进挂载点**（因为 kubelet 用 symlink 原子切换 `..data` 目录）。但**进程不会自动重载**——应用要自己 watch 文件（如 SIGHUP）。

> 面试一句话总结：**ConfigMap/Secret 把配置和镜像分离；Secret 的 base64 只是编码不是加密，真正的保护靠 RBAC + etcd encryption at rest，且挂载时用 tmpfs 不落盘；两种注入方式差异很关键——env 方式启动注入一次、Secret 更新感知不到，卷挂载方式 kubelet 会同步更新（约 1 分钟）但不触发进程重载，所以"热更新配置"要挂卷 + 应用自己 watch 文件。**

---

## 8. 存储：Volume / PV / PVC / StorageClass / CSI

### 1. 现有问题

容器文件系统是**临时的**（容器一重建就没了），而且 Pod 内多容器需要共享文件。同时，存储不该让应用开发者关心"底层是 NFS 还是云盘"。

### 2. 方法论

**三层抽象**（这是 K8s 存储的核心设计）：

```
应用（Pod）──声明──► PVC（我要 100Gi、能读写）
                       │ 动态供给（或静态绑定到已有 PV）
                       ▼
                    PV（实际的一块存储：云盘 / NFS / Ceph）
                       │ 由谁创建？
                       ▼
              StorageClass（"用哪种存储的模板"+ Provisioner）
                       │ 真正干活的是
                       ▼
                    CSI 驱动（云盘/NFS 的插件）
```

| 概念 | 谁定义 | 说明 |
|---|---|---|
| **Volume** | Pod spec | Pod 级的目录（`emptyDir`/`hostPath`/`configMap`/`secret`/`PVC`…），生命周期跟着 Pod |
| **PV** | 管理员或动态创建 | 集群级的存储资源（PersistentVolume） |
| **PVC** | 应用开发者 | 存储的**申请单**（容量 + 访问模式），Pod 只引用 PVC |
| **StorageClass** | 管理员 | "存储模板"：指定 provisioner、参数、回收策略 |
| **CSI** | 插件 | 容器存储接口，让任意存储系统接入 K8s |

**访问模式（accessModes）**：

| 模式 | 缩写 | 含义 |
|---|---|---|
| ReadWriteOnce | RWO | 单节点读写（云盘典型） |
| ReadOnlyMany | ROX | 多节点只读 |
| ReadWriteMany | RWX | **多节点读写**（NFS/CephFS/对象存储 CSI） |
| ReadWriteOncePod | RWOP | 单 Pod 读写（1.29+ GA） |

**训练场景的关键点**：多机训练要**共享数据集** → 必须是 **RWX**（NFS、CephFS、JuiceFS、或对象存储 CSI），云盘（RWO）做不到多机同时挂。

**回收策略（reclaimPolicy）**：

- `Delete`（默认，动态供给）：删 PVC → 连带删底层存储（**小心：checkpoint 会没**）；
- `Retain`：删 PVC 后 PV 保留（需要手工清理），**生产上存 checkpoint 应该用这个**。

### 3. 具体数值样例

```yaml
# ① StorageClass（管理员建一次）
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata: {name: fast-ssd}
provisioner: ebs.csi.aws.com            # CSI 驱动
parameters: {type: gp3, iops: "16000", throughput: "1000"}
reclaimPolicy: Retain                   # ★ checkpoint 用 Retain
volumeBindingMode: WaitForFirstConsumer # ★ 延迟绑定：等 Pod 调度后再建盘（避免跨 AZ）
allowVolumeExpansion: true              # 允许在线扩容
---
# ② PVC（开发者写）
apiVersion: v1
kind: PVC
metadata: {name: ckpt-pvc}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: fast-ssd
  resources: {requests: {storage: 500Gi}}
---
# ③ Pod 用
spec:
  containers:
  - volumeMounts: [{name: ckpt, mountPath: /ckpt}]
  volumes:
  - name: ckpt
    persistentVolumeClaim: {claimName: ckpt-pvc}
```

**`volumeBindingMode` 的两个值很关键**：

- `Immediate`：PVC 一创建就建盘 → 可能建在 **Pod 调度不到的那个可用区**（跨 AZ 挂不上）；
- `WaitForFirstConsumer`：**等 Pod 调度后才建盘**，保证盘和 Pod 同 AZ。**生产默认应该用这个。**

**存储的四种形态在训练集群里的分工**：

| 需求 | 用什么 |
|---|---|
| `/dev/shm`（NCCL、多进程） | `emptyDir{medium: Memory}` —— **不是存储，是内存** |
| 日志/临时数据 | `emptyDir`（跟 Pod 生命周期） |
| checkpoint（单机、要保留） | PVC（RWO + `Retain`） |
| 数据集/模型（多机共享） | RWX（NFS/CephFS）或**对象存储 CSI** |
| 超大只读数据集 | 对象存储 CSI + 缓存（或 JuiceFS / Alluxio） |

> 面试一句话总结：**K8s 存储的三层抽象是"StorageClass（模板）→ PV（实际存储）→ PVC（申请单）"，Pod 只引用 PVC；访问模式里 RWO（云盘）实现不了多机共享，多机训练的数据集必须 RWX（NFS/CephFS/对象存储 CSI）；回收策略 Delete 会连存储一起删，存 checkpoint 要 Retain；`volumeBindingMode: WaitForFirstConsumer` 让盘等 Pod 调度后再建，避免跨可用区挂不上——这两个字段是生产必配。**

---

## 四、元数据与组织

## 9. Namespace / Label / Annotation

### 1. 现有问题

一个集群里几十个团队、几百个应用，怎么隔离、怎么分类、怎么附加工具元数据？

### 2. 方法论

| 机制 | 干什么 | 是否被 selector 使用 |
|---|---|---|
| **Namespace** | **逻辑隔离**：名字作用域 + 资源配额（ResourceQuota）+ 默认网络/安全策略 | 否（是一个作用域） |
| **Label** | **可被 selector 查询的键值对** | ✅ **是**（Service/Deployment/NetworkPolicy 全靠它） |
| **Annotation** | 附加**任意非标识性元数据**（工具配置、版本信息、`kubectl.kubernetes.io/last-applied-configuration`） | ❌ 否 |

**关键区别：Label 是"身份"（会被查询），Annotation 是"备注"（不会被查询）。**

```yaml
metadata:
  name: agl-store
  namespace: ml-platform            # 作用域
  labels:                           # ★ 会被 selector 匹配
    app: agl-store
    tier: backend
    version: v1
  annotations:                       # ★ 不会被 selector 匹配
    description: "轨迹存储服务"
    prometheus.io/scrape: "true"     # 工具读它来发现指标端点
    prometheus.io/port: "9090"
```

**Namespace 不隔离什么（重要）**：

- ❌ **不隔离网络**（默认跨 namespace 可通，要隔离得用 NetworkPolicy）；
- ❌ 不隔离 CPU/内存（要 ResourceQuota / LimitRange）；
- ✅ 隔离**名字**、✅ 是 RBAC 的作用域、✅ 是 ResourceQuota 的作用域。

### 3. 具体数值样例

**Label selector 的两种写法**：

```yaml
# ① 等值（Equality-based）
selector: {app: agl-store, tier: backend}
# ② 集合（Set-based）
selector:
  matchExpressions:
  - {key: tier, operator: In, values: [backend, frontend]}
  - {key: env, operator: NotIn, values: [dev]}
  - {key: gpu, operator: Exists}
```

**Namespace + ResourceQuota 的数值例子**：

```yaml
apiVersion: v1
kind: ResourceQuota
metadata: {name: ml-quota, namespace: ml-platform}
spec:
  hard:
    requests.cpu: "200"
    requests.memory: "2Ti"
    requests.nvidia.com/gpu: "64"        # ★ GPU 也能量化配额
    persistentvolumeclaims: "20"
```

**意义**：团队 A 最多用 64 张卡——**这是"多租户集群"的基础**。注意：**配额限制的是 requests 之和**，所以"requests 写小、limits 写大"会绕过配额但可能 OOM。

> 面试一句话总结：**Namespace 是逻辑隔离（名字作用域 + RBAC/ResourceQuota 的作用域），但它不隔离网络也不隔离资源；Label 是可查询的"身份"（Service/Deployment/NetworkPolicy 都靠 selector 找它），Annotation 是不被查询的"备注"（工具配置如 prometheus.io/scrape 放这里）；selector 有等值（app=x）和集合（In/NotIn/Exists）两种写法；ResourceQuota 能对 namespace 限 GPU 数量，是多租户集群的基础。**

---

## 10. OwnerReference 与 Finalizer：级联删除与删除钩子

### 1. 现有问题

删一个 Deployment 时，它下面几十个 Pod 怎么自动删掉？反过来，如果某个资源删除前必须先做清理（比如释放云上负载均衡器），怎么保证？

### 2. 方法论

**OwnerReference（谁拥有我）—— 级联删除的机制**：

```yaml
# Pod 的 metadata 里会自动带上
ownerReferences:
- apiVersion: apps/v1
  kind: ReplicaSet
  name: web-abc123
  uid: 5f6e...
  controller: true
  blockOwnerDeletion: true
```

- 每个对象可以声明多个 owner；带 `controller: true` 的那个是"管理控制器"；
- **删除 owner 时，**垃圾回收器（garbage collector）按 `propagationPolicy` 处理 children：
  - `Background`（默认）：owner 立即删除，children 后台异步清理；
  - `Foreground`：先删 children，owner 处于 `deletionTimestamp` 状态直到 children 清完；
  - `Orphan`：只删 owner，children 变孤儿（保留）。

**Finalizer（删除前必须完成的钩子）**：

```yaml
metadata:
  finalizers:
  - my-operator.example.com/cleanup-lb
```

**机制**：`kubectl delete` 时，如果对象还有 finalizer，API Server **不会真的删**，只是设置 `metadata.deletionTimestamp` —— 对象进入"terminating"状态；**控制器必须自己完成清理工作，然后把这个 finalizer 从列表里移除**，API Server 才会真正删除。

**这两个机制配合起来，就是"有序清理"的实现方式**（实习里"作业删除时先释放公网 LB 再删集群"用的就是这个）。

### 3. 具体数值样例

**一个典型的 Finalizer 使用流程**（Operator 侧）：

```python
# reconcile 里
if obj.metadata.deletion_timestamp is not None:
    if "cleanup-lb" in obj.metadata.finalizers:
        # ① 先做清理：删 Ingress / LoadBalancer / 释放公网 IP
        delete_ingress(obj)
        wait_until_gone(ingress_name)
        # ② 清理完，移除 finalizer
        obj.metadata.finalizers.remove("cleanup-lb")
        update(obj)
    return   # 不再做别的

# 正常路径：确保 finalizer 已加上
if "cleanup-lb" not in obj.metadata.finalizers:
    obj.metadata.finalizers.append("cleanup-lb")
    update(obj)
# ... 继续建 RayCluster / TrajStore ...
```

**常见坑**：**finalizer 加了但控制器忘了移除 → 对象永远卡在 Terminating**，`kubectl delete` 也没用。这时候只能手工 `kubectl patch ... -p '{"metadata":{"finalizers":null}}'`（危险操作，等于跳过清理）。

**级联删除的数值样例**：

```
删除 Deployment（10 副本）
  → 删 ReplicaSet（1 个）
    → 删 Pod（10 个）              ← ownerReference 链自动级联
  → OwnerReference 决定了"删谁连带删谁"

如果某个 CR 有 finalizer 且控制器没清理：
  kubectl delete rayjob x
  → rayjob 卡在 Terminating（deletionTimestamp 有值但对象还在）
  → kubectl get rayjob x 仍然可见
```

> 面试一句话总结：**OwnerReference 实现级联删除（删 Deployment → 连带 RS → 连带 Pod），传播策略有 Background（默认，异步）/Foreground（先删子再删父）/Orphan（保留子）；Finalizer 是"删除前的钩子"——带 finalizer 的对象被 delete 时只设置 deletionTimestamp 进入 Terminating，必须由控制器完成清理并自己移除 finalizer 才会真删；两者配合是"有序清理"的标准实现（如先释放公网 LB 再删集群），最常见的坑是控制器忘记移除 finalizer 导致对象永远卡在 Terminating。**

---

## 11. ServiceAccount 与 RBAC：Pod 用什么身份访问 API

### 1. 现有问题

Pod 里的程序如果要去调 K8s API（比如 Operator、或者读 Secret），**它用什么身份？能做什么？** 如果所有 Pod 都用集群管理员身份，一个被攻破的 Pod 就能控制整个集群。

### 2. 方法论

**四层模型**：

```
ServiceAccount（身份）→ Role/ClusterRole（权限集合）→ RoleBinding/ClusterRoleBinding（绑定）→ 校验
```

| 概念 | 作用域 | 说明 |
|---|---|---|
| **ServiceAccount** | namespace | Pod 的身份；token 自动挂到 `/var/run/secrets/kubernetes.io/serviceaccount/` |
| **Role** | namespace | 一组权限规则（apiGroups/resources/verbs） |
| **ClusterRole** | 集群 | 同上，但作用全集群（也可用于非资源型权限如 `/healthz`） |
| **RoleBinding** | namespace | 把 Role（或 ClusterRole）授予主体（User/Group/SA） |
| **ClusterRoleBinding** | 集群 | 同上，但作用全集群 |

```yaml
# ① 建 SA
apiVersion: v1
kind: ServiceAccount
metadata: {name: trainer-sa, namespace: ml-platform}
---
# ② 建 Role（只允许在 ml-platform 里读写 pods/rayclusters）
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: {name: trainer-role, namespace: ml-platform}
rules:
- apiGroups: [""]                       # core API group 写空字符串
  resources: ["pods", "pods/log", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["ray.io"]
  resources: ["rayclusters", "rayjobs"]
  verbs: ["get", "list", "watch", "create", "delete"]
---
# ③ 绑定
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: {name: trainer-binding, namespace: ml-platform}
subjects:
- {kind: ServiceAccount, name: trainer-sa, namespace: ml-platform}
roleRef: {kind: Role, name: trainer-role, apiGroup: rbac.authorization.k8s.io}
---
# ④ Pod 用它
spec:
  serviceAccountName: trainer-sa
```

**verbs 与 resources**：`get/list/watch/create/update/patch/delete/deletecollection`；`resources` 里还能用子资源（`pods/log`、`pods/exec`、`deployments/scale`）。

**最小权限的实践**（面试加分）：

- **一个工作负载一个 SA**，不要共用 `default`；
- **能用 Role 就不用 ClusterRole**（namespace 级更小）；
- **不要把 `cluster-admin` 绑给业务**（运维常见坑）；
- **关闭自动挂载 token**（不需要访问 API 的 Pod）：`automountServiceAccountToken: false`。

### 3. 具体数值样例

**Operator 需要的权限（对照 KubeRay）**：

```yaml
# KubeRay Operator 的 ClusterRole 大致长这样（简化）
rules:
- apiGroups: ["ray.io"]
  resources: ["rayclusters", "rayjobs", "rayservices", "raycronjobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets", "events"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/status", "pods/exec"]
  verbs: ["get", "patch", "create"]
# 还需要 finalizers 权限来更新自己的 CR
- apiGroups: ["ray.io"]
  resources: ["rayclusters/finalizers"]
  verbs: ["update"]
```

**为什么 Operator 需要这么大的权限**：因为它要**代表用户创建 Pod/Service**——这是 Operator 模式的固有代价。所以生产上会限制"**谁能创建 CR**"（而不是限制 Operator），因为"能创建一个 RayCluster"≈"能让 Operator 帮你建任意 Pod"。

> 面试一句话总结：**Pod 访问 API 的身份是 ServiceAccount，权限模型是 SA → Role/ClusterRole（权限集合）→ RoleBinding/ClusterRoleBinding（绑定），Role 是 namespace 级、ClusterRole 是集群级；最小权限实践是一工作负载一 SA、能用 Role 不用 ClusterRole、不给业务绑 cluster-admin、不需要 API 访问的 Pod 关掉 `automountServiceAccountToken`；Operator 天然需要大权限（它要代表用户建 Pod/Service），所以真正的安全边界应该卡在"谁能创建 CR"上。**

---

## 五、扩展机制

## 12. CRD / CR / Operator / reconcile loop

### 1. 现有问题

内置资源（Pod/Deployment）表达不了领域概念。比如"一个 Agentic RL 训练作业"包含：RayCluster + 轨迹存储 + runner + HPA + 两条公网链路 + 状态机 + 续训。用一堆 YAML 拼是**没法管理生命周期**的。

### 2. 方法论

**四件套**：

| 概念 | 是什么 |
|---|---|
| **CRD（CustomResourceDefinition）** | 告诉 K8s "有一种新资源叫 RayCluster，它的 schema 是这样" |
| **CR（Custom Resource）** | CRD 的一个实例（`kind: RayCluster` 的那份 YAML） |
| **Controller** | 一个进程，watch CR，把实际状态收敛到期望状态 |
| **Operator** | **Controller + 领域知识**（不只是同步状态，还管生命周期、升级、备份…） |

**reconcile loop 的骨架**（K8s 控制器的通用模式）：

```go
for {
    // ① 从队列取一个"需要处理的对象"（key = namespace/name）
    key, quit := queue.Get()
    // ② 读期望状态（spec）
    obj := client.Get(key)
    // ③ 读实际状态（观测到的子资源 / 外部系统）
    actual := observe(obj)
    // ④ 计算差异并收敛（幂等！）
    if obj.spec.replicas != actual.replicas {
        client.Patch(obj, ...)   // 改 spec 或改子资源
    }
    // ⑤ 写回 status
    client.UpdateStatus(obj, {ready: ..., phase: ...})
    // ⑥ 处理删除（finalizer）
    if obj.deletionTimestamp != nil { cleanup(); removeFinalizer() }
    queue.Forget(key)            // 成功才 forget；失败会重入队（指数退避）
}
```

**reconcile 的三个铁律**（面试必答）：

1. **幂等**：同一个对象 reconcile 一百次，结果必须一样（因为 watch 会重复投递、失败会重试）；
2. **不要假设顺序**：控制器是并发的，A 和 B 的 reconcile 可能交错；
3. **不要"记住了上次的状态"**：状态应该从**集群实际状态**读出来，而不是控制器内存里的变量（否则重启就丢）。

**`spec` 与 `status` 的分工**：

- `spec` = **期望状态**（用户写，控制器读）；
- `status` = **实际状态**（控制器写，用户读）。
- **禁止**控制器回写 `spec`（会和自己打架）——这就是 K8s 把两者分开的原因。

### 3. 具体数值样例

**`RayCluster` 的 spec/status**：

```yaml
spec:                            # 用户声明
  rayVersion: "2.52.0"
  headGroupSpec: {...}
  workerGroupSpecs:
  - {groupName: gpu, replicas: 8, minReplicas: 0, maxReplicas: 10}
status:                          # Operator 回报
  state: ready
  observedGeneration: 3          # ★ 控制器处理到第几代 spec（乐观并发）
  desiredWorkerReplicas: 8
  availableWorkerReplicas: 8
  head:
    podName: raycluster-gpu-head-xyz
    serviceName: raycluster-gpu-head-svc
    serviceIP: 10.96.12.34
  endpoints:
    dashboard: 8265
    client: "10001"
  conditions: [...]
```

**`observedGeneration` 是关键字段**：它表示"控制器已经把第 N 代的 spec 处理完了"。如果 `status.observedGeneration < metadata.generation`，说明**还在收敛中**——这是判断"我的改动生效了没"的标准方法。

**自定义 CRD 的完整例子**（实习的 AgenticRLTrainJob）：

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata: {name: agenticrltrainjobs.ml.example.com}
spec:
  group: ml.example.com
  names: {kind: AgenticRLTrainJob, plural: agenticrltrainjobs, shortNames: [arljob]}
  scope: Namespaced
  versions:
  - name: v1alpha1
    served: true
    storage: true
    schema:                            # ★ schema 校验：写错了字段直接拒绝
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              algorithm: {type: string}
              model: {type: string}
              rayClusterSpec: {type: object, x-kubernetes-preserve-unknown-fields: true}
              publicTrajStore: {type: boolean, default: true}
              steps: {type: integer, minimum: 1}
            required: [algorithm, model]
          status:
            type: object
            properties:
              phase: {type: string, enum: [Pending, Rollout, Training, Evaluating, Succeeded, Failed]}
              currentStep: {type: integer}
              trajStoreURL: {type: string}       # ★ 公网 URL 写回这里
              rayClusterName: {type: string}
    subresources:
      status: {}                     # ★ 开启 status 子资源（才能单独 update status）
    additionalPrinterColumns:        # kubectl get 时显示的列
    - {name: Phase, type: string, jsonPath: .status.phase}
    - {name: Step, type: integer, jsonPath: .status.currentStep}
```

**为什么训练作业值得做成 CRD**（面试高频）：

| 裸 Job/Deployment 表达不了 | CRD 能表达 |
|---|---|
| 状态机（Pending→Rollout→Training→Evaluating） | `status.phase` + `conditions` |
| 多个子资源联动（RayCluster + TrajStore + runner + HPA） | reconcile 统一创建与回收 |
| 续训（从 checkpoint 恢复） | `status.currentStep` + spec 里的 resume 字段 |
| 公网链路生命周期 | `status.trajStoreURL` + Finalizer 释放 |
| 声明式接口（用户只写期望） | `spec` 描述期望，Operator 收敛 |

> 面试一句话总结：**CRD 定义新资源类型、CR 是它的实例、Controller watch 它并收敛状态、Operator = Controller + 领域知识；reconcile 三条铁律是"幂等、不假设顺序、状态从集群读而不是记在内存"；spec 是期望（用户写）、status 是实际（控制器写），控制器绝不回写 spec；`status.observedGeneration` 是判断"改动生效了没"的标准字段；训练作业值得做成 CRD，因为它要表达状态机、多子资源联动、续训和公网链路生命周期——这些裸 Job/Deployment 都表达不了。**

---

## 13. Admission Webhook：在落库前拦截请求

### 1. 现有问题

有些人想"改 API 对象"——比如"所有 Pod 必须带 `owner` label"、"禁止用 latest tag 的镜像"、"给 Pod 自动注入 sidecar"。但这些规则**不能写在客户端**（客户端可以绕过），也不适合写在控制器里（控制器看到的是已经存进 etcd 的对象）。

### 2. 方法论

**API Server 的请求链路**（这是理解 Webhook 位置的关键）：

```
kubectl apply
   │
   ▼
① 认证（Authentication）：你是谁？  ← 证书 / Bearer Token / OIDC
   ▼
② 授权（Authorization）：你能做这个吗？ ← RBAC / Webhook / ABAC
   ▼
③ 准入（Admission）★ 在这里可以改对象、可以拒绝
   ├─ Mutating Admission：可以修改对象（如注入 sidecar、加默认值）
   │    ├─ 内置 MutatingAdmissionWebhook
   │    └─ 你自己注册的 MutatingWebhookConfiguration
   ▼
   ├─ 对象校验 + 默认值填充
   ▼
   ├─ Validating Admission：只能通过/拒绝，不能改
   │    ├─ 内置（如 ResourceQuota、LimitRanger、PodSecurity）
   │    └─ 你自己注册的 ValidatingWebhookConfiguration
   ▼
④ 存储：写 etcd
   ▼
⑤ Watch 通知：各控制器/调度器/let 收到变更
```

**顺序要点**：**Mutating 先于 Validating**（因为要先改完才能校验最终形态）；**Webhook 可以拒绝请求**（返回 `allowed: false` + 原因）。

**两类 Webhook 的差别**：

| | Mutating | Validating |
|---|---|---|
| 能否修改对象 | ✅ 能（返回 JSONPatch） | ❌ 不能 |
| 典型用途 | 注入 sidecar、填默认值、加 label | 合规校验、配额、安全策略 |
| 失败后果 | 可配 `failurePolicy: Ignore/Fail` | 同上 |

**`failurePolicy` 很关键**：`Fail`（默认）= webhook 挂了 → **整个集群的 Pod 都创建不了**（生产事故常见原因）；`Ignore` = webhook 挂了就放行。

### 3. 具体数值样例

**一个真实场景：自动给所有 Pod 注入 sidecar**（服务网格就是这么做的）

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata: {name: sidecar-injector}
webhooks:
- name: inject.example.com
  clientConfig:
    service: {name: injector-svc, namespace: injector, path: /mutate}
    caBundle: <base64 CA>
  rules:
  - operations: ["CREATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  namespaceSelector:              # ★ 只在带这个 label 的 namespace 生效
    matchLabels: {sidecar-injection: enabled}
  failurePolicy: Ignore            # ★ 注入器挂了不该阻塞业务
  sideEffects: None
  admissionReviewVersions: ["v1"]
```

**校验型 Webhook 的例子**（禁止 latest tag）：

```
请求：创建 Pod，image: nginx:latest
  → ValidatingWebhook 收到 AdmissionReview
  → 你的 webhook 检查 image tag
  → 返回：
     {
       "apiVersion": "admission.k8s.io/v1",
       "kind": "AdmissionReview",
       "response": {
         "uid": "<原样回传>",
         "allowed": false,
         "status": {"code": 403, "message": "image tag 'latest' is not allowed; pin a version"}
       }
     }
  → API Server 拒绝请求，kubectl 报错
```

**面试常问的"为什么准入控制比控制器更合适"**：

- 控制器看到的是**已经存进 etcd 的对象**——"事后补救"，而且期间对象已经对外可见；
- Webhook 在**落库前**拦截——可以**直接拒绝**（控制器没法拒绝，只能再改回去）；
- 对**平台一致性**很重要：所有团队创建的 Pod 都统一被注入/被校验，不依赖客户端自觉。

> 面试一句话总结：**API Server 的请求链路是"认证（你是谁）→ 授权（RBAC，你能做什么）→ 准入（可以改/可以拒）→ 写 etcd → watch 通知"；准入分 Mutating（能改对象，如注入 sidecar、填默认值）和 Validating（只能通过/拒绝，如配额、安全策略），且 Mutating 一定在 Validating 之前；`failurePolicy: Fail`（默认）意味着 webhook 挂了整个集群创建不了 Pod（生产事故高发点），注入类 webhook 一般设 `Ignore`；Webhook 相比控制器的优势是"落库前拦截、可以直接拒绝"，而控制器只能事后补救。**

---

## 六、架构串讲

## 14. 控制平面与数据平面

### 1. 现有问题

前面讲了各种资源怎么写，但它们**由谁保管、由谁执行**？分成两块：**控制平面（做决定）和数据平面（干活）**。

### 2. 方法论

```
┌─────────────────── Control Plane（决策）───────────────────┐
│  kube-apiserver   唯一入口：认证/授权/准入/校验 → 写 etcd      │
│  etcd             唯一真相源：存所有对象（唯一有状态组件）      │
│  kube-scheduler   决定 Pod 去哪个 node                       │
│  kube-controller-manager  跑所有内置控制器（Deployment/Node/  │
│                           Job/Endpoint… 的 reconcile）        │
│  cloud-controller-manager 云厂商对接（LB/路由/节点生命周期）    │
└────────────────────────────┬────────────────────────────────┘
                             │ watch / list（唯一通道）
┌────────────────────────────▼────────────────────────────────┐
│                     Data Plane（执行，每节点）                 │
│  kubelet          node agent：watch 分给自己的 Pod → 调 CRI 起  │
│                   容器、调 CNI 配网络、调 CSI 挂盘、跑探针      │
│  kube-proxy       实现 Service（iptables/IPVS）               │
│  container runtime（containerd/CRI-O）真正跑容器               │
│  CNI / CSI 插件    网络与存储                                  │
└─────────────────────────────────────────────────────────────┘
```

**关键认知**：

1. **所有组件都只和 apiserver 通信，互相之间不直接通信**（这是 K8s 的"星型"架构）——所以 apiserver 是唯一入口也是唯一瓶颈点；
2. **etcd 是唯一有状态的组件**（其他组件都能重启恢复）；
3. **kubelet 不是"被 apiserver 调用"的，而是主动 watch**（apiserver 不知道有哪些节点在跑）——这是"声明式"的体现；
4. **调度器只负责"决定"**，真正拉起容器的是 **kubelet**。

### 3. 具体数值样例

**一个 3 节点集群的进程分布**：

| 组件 | 跑在哪 |
|---|---|
| apiserver / etcd / scheduler / controller-manager | control-plane 节点（生产上通常 3 副本做 HA，etcd 用 Raft） |
| kubelet / kube-proxy / containerd / CNI | **每个节点都有**（包括 control-plane） |
| Ingress Controller / CoreDNS / CNI DaemonSet | 以 Pod 形式跑在 worker 上 |

**核心组件挂了的后果**（面试常问"高可用怎么做的"）：

| 挂掉 | 后果 |
|---|---|
| **etcd** | 整个集群"失忆"——所有 API 请求失败、无法新建任何资源；**已运行的容器还在跑**（kubelet 有本地缓存） |
| **apiserver** | 同样无法读写 API；已运行容器继续跑；kubelet 会重试连接，连接恢复后自动同步 |
| **scheduler** | 新 Pod 全部 Pending；已运行的不受影响 |
| **controller-manager** | 自愈/扩缩/滚动升级暂停；已运行的不受影响 |
| **kubelet** | **该节点的容器失去管理**（不重启、不更新、探针失效），节点最终被标记 NotReady |
| **kube-proxy** | 该节点的 Service 转发失效（新连接失败），Pod 直接互访不受影响 |

**这张表很有面试价值**：它说明 **K8s 的"降级行为"设计得很好**——控制面挂了，数据面还能继续跑（只是不能变更）；这也解释了为什么"滚动重启 apiserver"在生产上是安全的。

> 面试一句话总结：**控制平面 = apiserver（唯一入口，认证/授权/准入）+ etcd（唯一有状态、唯一真相源）+ scheduler（决定 Pod 去哪）+ controller-manager（跑所有内置控制器）；数据平面 = kubelet（每节点的 agent，watch 自己的 Pod 并调 CRI/CNI/CSI）+ kube-proxy + 容器运行时 + CNI/CSI 插件；关键认知是"所有组件只和 apiserver 通信、互相不直连"以及"kubelet 是主动 watch 而不是被调用"；高可用上看降级行为——etcd/apiserver 挂了无法变更但已运行容器继续跑，scheduler/controller-manager 挂只影响新建与自愈。**

---

## 15. 一次 Pod 创建的完整流程（串讲 1~14）

### 1. 现有问题

把前面所有概念串成一条线：`kubectl apply -f pod.yaml` 之后，到底发生了什么？这是面试收尾最常问的"串讲题"。

### 2. 方法论

```bash
kubectl apply -f pod.yaml
```

```
① kubectl 把 YAML 转成 HTTP 请求 → POST /api/v1/namespaces/default/pods
   （kubectl 读 ~/.kube/config 拿 apiserver 地址 + 客户端证书）

② apiserver 处理链（staging/src/k8s.io/apiserver/pkg/server/config.go）
   WithAuthentication (config.go:1075)   ← 客户端证书验证身份
   WithAudit          (config.go:1064)   ← 审计日志
   WithMaxInFlightLimit (config.go:1051) ← 限流，防止雪崩
   WithAuthorization  (config.go:1040)   ← RBAC：这个用户能 create pods 吗？
   → Mutating Admission（默认值填充、sidecar 注入）
   → 校验 + Validating Admission
   → 写 etcd（存的是 spec，status 待填）

③ scheduler watch 到"未绑定的 Pod"
   ① Filter（资源/亲和/污点/拓扑）
   ② Score
   ③ Bind：写 Pod.spec.nodeName = node-2

④ node-2 的 kubelet watch 到"分给自己的 Pod"
   ① 创建 sandbox（pause 容器，持有 netns）
   ② 调 CNI ADD → 分配 Pod IP、接上网络
   ③ 调 CSI → 挂卷
   ④ 调 CRI → 拉镜像、起业务容器
   ⑤ 起探针（startup/liveness/readiness）
   ⑥ 回写 Pod status（phase=Running、containerStatuses、podIP）

⑤ 其他控制器继续收敛
   - EndpointSlice controller：Pod Ready 后把它加进 Service 的 Endpoints
   - Deployment/ReplicaSet controller：确认副本数符合期望
   
⑥ kube-proxy watch 到 Endpoints 变化 → 更新本节点 iptables/IPVS 规则
   → 此时访问 Service ClusterIP 才能真的路由到这个 Pod
```

**这条链路里每一环都能展开成一道面试题**：

| 环节 | 追问 |
|---|---|
| ② 准入 | Mutating 和 Validating 谁先？被拒绝会怎样？ |
| ③ 调度 | requests 不够会怎样？抢占怎么触发？ |
| ④ kubelet | 为什么需要 pause 容器？CNI 什么时候被调？ |
| ⑤ Endpoints | Pod Ready 之前能不能被 Service 路由？（**不能**） |
| ⑥ kube-proxy | ClusterIP 是虚拟 IP，谁在做 NAT？ |

### 3. 具体数值样例

**一个"卡住"的排查实例**（把链路反过来用）：

```
现象：kubectl apply 后 Pod 一直是 Pending

按链路倒推：
① apiserver 成功了吗？     → kubectl get pod 能看到对象 = 成功
② 有 PodScheduled 条件吗？  → kubectl describe pod
    - 如果是 "0/3 nodes are available: 3 Insufficient nvidia.com/gpu"
      → 资源不够（调度阶段失败）
    - 如果是 "node(s) had untolerated taint"
      → 污点没容忍
    - 如果 PodScheduled=True 但容器没起
      → 问题在 kubelet（拉镜像失败 / CNI 失败 / 卷挂不上）
③ 看 events：kubectl describe pod 最后一段的 Events 就是 kubelet 报的错
   - ImagePullBackOff → 镜像名/tag/仓库凭据
   - FailedMount → PVC 没绑定 / CSI 报错
   - FailedCreatePodSandBox → CNI 问题
```

**这就是"为什么理解完整链路很重要"**：**Pod 卡住时的错误信息，直接对应链路的某一个环节**——Pending + Insufficient GPU 是调度器，ImagePullBackOff 是 kubelet，FailedMount 是 CSI，Endpoints 为空是 EndpointSlice controller。

> 面试一句话总结：**`kubectl apply` 之后的完整链路是：kubectl → apiserver（认证→审计→限流→授权→Mutating 准入→校验→Validating 准入→写 etcd）→ scheduler（Filter→Score→Bind 写 nodeName）→ 目标节点 kubelet（建 sandbox→CNI 配网→CSI 挂盘→CRI 起容器→跑探针→回写 status）→ EndpointSlice controller 把 Ready 的 Pod 加进 Service Endpoints → kube-proxy 更新转发规则；排障就是反过来用这条链路——Pending + Insufficient 是调度、ImagePullBackOff 是 kubelet、FailedMount 是 CSI、Endpoints 为空是 EndpointSlice。**

---

## 附：速查表与高频问答

### 资源速查

| 我想… | 用什么 |
|---|---|
| 跑一个长期服务、要自愈和滚动升级 | **Deployment** + Service |
| 跑一个跑完就结束的任务 | **Job**（定时用 CronJob） |
| 每个节点都要一个 | **DaemonSet** |
| 需要稳定名字和独立存储 | **StatefulSet** + Headless Service |
| 给 Pod 稳定访问入口 | **Service**（集群内 ClusterIP / 对外 LoadBalancer） |
| 七层路由 + TLS | **Ingress** + Ingress Controller（或 Gateway API） |
| 限制 Pod 之间互访 | **NetworkPolicy**（需要 CNI 支持） |
| 注入配置 | **ConfigMap**（非敏感）/ **Secret**（敏感） |
| 持久化数据 | **PVC** + StorageClass + CSI |
| 隔离团队 | **Namespace** + ResourceQuota + RBAC |
| 声明一个领域概念 | **CRD** + Controller（Operator） |
| 拦截/改写 API 请求 | **Admission Webhook** |

### 高频问答

**Q1：一个 Pod 由哪些部分组成？**
metadata（name/namespace/labels/annotations/ownerReferences）+ spec（initContainers / containers / volumes / restartPolicy / affinity / tolerations / serviceAccountName / dnsPolicy …）+ status（phase / conditions / containerStatuses / podIP）。容器本身有 image/command/args/env/ports/resources/volumeMounts/探针/securityContext。

**Q2：Pod 内多个容器共享什么？**
共享 **Network / UTS / IPC** namespace 和 **卷**（所以同 Pod 内容器用 localhost 互访、不能撞端口）；**PID 和 mount namespace、cgroup 独立**（`shareProcessNamespace: true` 才共享 PID）。

**Q3：liveness 和 readiness 的区别？**
liveness 失败 → **重启容器**；readiness 失败 → **从 Service Endpoints 摘除**（不重启）；startup 失败 → 一直重启，成功后另外两个才开始。

**Q4：Deployment 的滚动升级怎么做的？**
每次改 Pod 模板就建一个**新 ReplicaSet**，controller 按 `maxSurge`/`maxUnavailable` 把新 RS 扩容、旧 RS 缩容；回滚就是切回旧 RS（所以有 `revisionHistoryLimit`）。

**Q5：调度看 requests 还是 limits？**
看 **requests**。limits 是运行时 cgroup 上限（CPU 超了 throttle、内存超了 OOMKill）。

**Q6：ClusterIP 是什么？谁在实现它？**
一个**虚拟 IP，不属于任何网卡**。每个节点的 **kube-proxy**（iptables/IPVS/nftables）把访问 ClusterIP 的流量 DNAT 到具体 Pod IP；用 eBPF（Cilium）可以绕过 kube-proxy。

**Q7：Service 连不上怎么排查？**
先 `kubectl get endpoints <svc>`——**空的说明 selector 没匹配到 Pod 或 Pod 未 Ready**（不是网络问题）；再看 Pod 是否 Ready；最后在 Pod 里 curl Service DNS。

**Q8：Secret 是加密的吗？**
**base64 只是编码，不是加密**。保护靠 RBAC（谁能读）+ etcd encryption at rest；挂载时用 tmpfs 不落盘。

**Q9：PVC 的 `volumeBindingMode` 两个值区别？**
`Immediate` 一创建就建盘（可能跨 AZ 挂不上）；**`WaitForFirstConsumer` 等 Pod 调度后再建**，保证同 AZ——生产推荐。

**Q10：OwnerReference 和 Finalizer 分别解决什么？**
OwnerReference 实现**级联删除**（删 Deployment 连带删 RS/Pod）；Finalizer 实现**删除前钩子**（对象进 Terminating，控制器清理完并移除 finalizer 才真删）。

**Q11：CRD 的 `spec` 和 `status` 为什么分开？**
`spec` 是用户写的期望状态、`status` 是控制器写的实际状态。分开是为了**避免控制器回写自己的输入**（会和自己打架），也是 K8s 声明式模型的基础。

**Q12：`status.observedGeneration` 有什么用？**
表示"控制器已经把第 N 代 spec 处理完了"。`observedGeneration < metadata.generation` 说明**还在收敛中**——判断改动生效的标准方法。

**Q13：Mutating 和 Validating Webhook 哪个先？**
**Mutating 先**（要先改完才能校验最终形态）。`failurePolicy: Fail` 时 webhook 挂了会导致整个集群创建不了 Pod。

**Q14：控制面挂了会怎样？**
etcd/apiserver 挂 → **无法变更但已运行容器继续跑**；scheduler/controller-manager 挂 → 只影响新建与自愈；kubelet 挂 → 该节点容器失去管理。

**Q15：为什么多机训练要 gang scheduling？**
默认调度器**逐个 Pod 贪心**：8 个 worker 要 8 卡、只剩 7 台机器时，它会让 7 个起来、1 个 Pending——**已起的 7 个占着 GPU 空转等**，任务死锁。Volcano/KAI/YuniKorn/coscheduling 保证"要么全起、要么全等"。

**Q16：为什么训练任务不用 Deployment？**
Deployment 的滚动升级是"**先建新 Pod 再删旧 Pod**"（`maxSurge`），新 Pod 拿不到 GPU（被旧 Pod 占着）会 Pending，升级卡死。训练需要"整组一起换"的语义 → 用 Job / RayJob / 自定义 CRD。
