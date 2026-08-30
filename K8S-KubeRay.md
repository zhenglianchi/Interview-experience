# Kubernetes + KubeRay 完全指南（从核心概念到自定义 CRD 训练作业）

> 对应实习亮点"管控面：基于 Ray 分布式框架的 AgenticRL 训练作业（Kubernetes CRD 资源）"与"并行创建 RayCluster Head 和 TrajStore 两条网络访问链路"。一句话知识框架：
> **Kubernetes = 声明式容器编排系统（把"我要什么"写成 YAML，控制器持续让现实向期望收敛）；KubeRay = 把 Ray 集群变成 K8S 原生资源（RayCluster/RayJob 等 CRD + Operator）；AgenticRL 训练作业 = 自定义 CRD + Operator 把 agent-lighting 的 store/algo/agent 三件套按分离式架构编排成 K8S 工作负载**。
>
> 素材来源：本地克隆 `C:\Users\HW\Desktop\简历投递\kubernetes`（HEAD `e72c2715`）、`C:\Users\HW\Desktop\简历投递\kuberay`（HEAD `ffec815`）、`agent-lightning-official`（v0.3.0）。配套阅读：`Ray.md`（Ray 本身）、`Agent-Lighting.md`（三件套与三机分离）、`Communication.md`（网络层）。

---

# 一、Kubernetes 全部核心概念（先分点，逐个讲透）

## 1. 工作负载（Workload）：Pod / ReplicaSet / Deployment / StatefulSet / DaemonSet / Job

### 1. 现有问题

- 一个应用跑起来需要什么？容器（进程隔离）→ 但容器会崩、会退出、需要多副本、需要更新；
- 需要一层"**声明期望、自动收敛**"的抽象，让用户不关心"怎么拉起、怎么重启、怎么滚动"。

### 2. 方法论（概念逐层递进）

**① Pod——最小调度单元**：
- 一个 Pod = 一组**共享网络命名空间与存储卷**的容器（通常是 1 主容器 + 可选 sidecar）；
- 特点：Pod 内所有容器共享 **同一个 IP / localhost / 端口空间**、共享挂载的 Volume；Pod 是**临时**的（可被杀死、重建，IP 会变）；
- 生命周期：`Pending → Running → Succeeded/Failed`，异常时 `CrashLoopBackOff`；
- 关键字段：`spec.containers[].image/ports/resources`、`spec.restartPolicy`（Always/OnFailure/Never）、`spec.nodeSelector`、`spec.containers[].readinessProbe/livenessProbe/startupProbe`（探针：HTTP/TCP/Exec）。

**② ReplicaSet（RS）——维持副本数**：
- `spec.replicas` 期望副本数 + `spec.selector` 选 Pod；RS 的 controller 发现实际数 < 期望数就**创建 Pod**，多了就**删除**；
- 只保证"数量"，不保证"更新"。

**③ Deployment——RS 之上加滚动更新**：
- Deployment 管理 RS，RS 管理 Pod，形成三层：**Deployment → ReplicaSet → Pod**；
- 滚动更新：`strategy.rollingUpdate.maxUnavailable/maxSurge` 控制新旧 RS 的流量切换；`revisionHistoryLimit` 保留历史版本用于回滚（`kubectl rollout undo`）；
- 这是**无状态应用**的标准载体。

**④ StatefulSet——有状态应用（稳定身份 + 稳定存储）**：
- 每个 Pod 有**稳定网络标识**：`<sts-name>-0, -1, -2...`，配合 **Headless Service** 用固定 DNS 名访问；
- **稳定存储**：`volumeClaimTemplates` 为每个副本生成独立的 PVC（Pod 重建后挂同一块盘）；
- **有序部署/删除**（`podManagementPolicy: OrderedReady`）；
- 适用：数据库、ZooKeeper、**需要固定身份的分布式组件**（如 Ray head？不，head 用 Deployment 即可）。

**⑤ DaemonSet——每节点一个**：
- 每个 Node 上恰好跑一个副本（新节点加入自动起、删除自动清）；
- 适用：日志采集（fluentd）、监控（node-exporter）、CNI 插件、**GPU 设备插件**。

**⑥ Job / CronJob——一次性/定时任务**：
- Job：跑完即结束的批任务，`completions`（要成功几个）、`parallelism`（并发几个）、`backoffLimit`（失败重试上限）、`ttlSecondsAfterFinished`（完成多久后清理）；
- CronJob：按 cron 表达式定时创建 Job（`schedule`、`concurrencyPolicy`、`startingDeadlineSeconds`）；
- **训练任务（如 AgenticRL 单次训练、评估）天然适合 Job 语义**——但长时、需资源编排的训练通常升级为自定义 CRD（第 21 点）。

### 3. 具体数值样例

- Deployment `replicas: 3`、`maxSurge: 1, maxUnavailable: 0`：滚动更新时先起 1 个新 Pod（总数 4），就绪后删 1 个旧 Pod（回到 3），逐个替换——**零停机**；
- Job `completions: 4, parallelism: 2, backoffLimit: 3`：2 个并行跑，4 个成功即完成，单个失败自动重试最多 3 次；
- StatefulSet 3 副本：Pod 名 `redis-0/1/2`，headless service 下 `redis-0.redis.default.svc` 固定可达，重建后仍叫 redis-0 且挂同一 PVC。

> 面试一句话总结：**K8S 工作负载按"无状态/有状态/系统组件/批任务"分四类：Deployment（无状态，滚动更新）、StatefulSet（稳定身份+稳定存储）、DaemonSet（每节点一个）、Job/CronJob（一次性/定时）——底层都是"声明副本数，controller 收敛"的模式。**

---

## 2. 服务与网络：Service / Ingress / NetworkPolicy / kube-proxy / CNI

### 1. 现有问题

- Pod 的 IP 是**临时**的（重建就变），客户端怎么稳定访问？多副本怎么负载均衡？集群外怎么访问？
- 集群内网络谁打通？东西向安全怎么隔离？

### 2. 方法论

**① Service——稳定的服务入口（4 层）**：
- **ClusterIP**（默认）：分配一个集群内虚拟 IP（VIP），`kube-proxy` 把发往 VIP 的流量按 **iptables / ipvs** 规则转发到后端 Pod（EndpointSlice 维护 Pod IP 列表）——集群内负载均衡；
- **NodePort**：在每台节点上开一个端口（30000-32767），`nodeIP:nodePort` 可从集群外访问（生产少用）；
- **LoadBalancer**：让云厂商（或 MetalLB）分配公网/云负载均衡器，流量 → NodePort → Pod；
- **Headless**（`clusterIP: None`）：不分配 VIP，直接给每个 Pod 一条 DNS 记录——StatefulSet 稳定身份靠它；
- **ExternalName**：把 Service 映射到外部 DNS 名。

**② Ingress——7 层入口**：
- 按 **host/path** 做 HTTP/HTTPS 路由（nginx-ingress / ingress-nginx / ALB），支持 TLS 终止、重写、限流；
- IngressClass 选择具体 controller 实现。

**③ 集群内 DNS（CoreDNS）**：`<service>.<namespace>.svc.cluster.local` 解析——Pod 间用服务名互访，KubeRay 的 head FQDN 就是 `raycluster-head-svc.default.svc...`。

**④ NetworkPolicy——东西向防火墙**：
- 按 Label Selector 定义"谁可以访问谁"（ingress/egress 规则 + 端口），由 CNI（Calico/Cilium）实现；
- KubeRay 1.6+ 支持为 RayCluster 自动生成 head/worker 的 NetworkPolicy（`raycluster_types.go` 里 Head/Worker EgressRules）。

**⑤ CNI（Container Network Interface）**：容器网络的插件化标准（Calico/BGP、Cilium/eBPF、flannel/VXLAN）——负责给每个 Pod 分配 IP、打通跨节点网络、实现 NetworkPolicy。

### 3. 具体数值样例

- `myapp` Deployment 3 副本 → Service `myapp`（ClusterIP 10.96.0.5）→ 客户端访问 `http://myapp:8080`，kube-proxy 按 ipvs 轮询转发到 3 个 Pod IP；
- 从公网访问：`Ingress → Service(ClusterIP) → Pod`，Ingress controller 监听 80/443，按域名路由；
- KubeRay 场景：`raycluster-head-svc`（ClusterIP）+ 若需外部访问加 NodePort/Ingress——**实习"TrajStore 公网 URL 暴露"就是给 TrajStore Service 配 LoadBalancer/Ingress**。

> 面试一句话总结：**K8S 网络四层：CNI 给 Pod 发 IP 打通集群、kube-proxy 实现 Service 的 4 层负载均衡（iptables/ipvs）、Ingress 做 7 层 HTTP 路由、NetworkPolicy 做东西向隔离——Pod IP 会变，Service 名不变，应用只认服务名。**

---

## 3. 配置与存储：ConfigMap / Secret / Volume / PV / PVC / StorageClass

### 1. 现有问题

- 配置（环境变量、参数）与镜像分离——改配置不该重新构建镜像；
- 密钥（密码、token、证书）不能明文进镜像/YAML；
- 容器是临时的，数据要持久化（训练 checkpoint、轨迹数据）。

### 2. 方法论

**① ConfigMap**：键值配置，注入方式：环境变量 / 文件挂载（volume）/ 命令行参数；改 ConfigMap 后需滚动重启 Pod 生效。
**② Secret**：与 ConfigMap 同机制但**敏感数据**（etcd 里 base64 编码存储，注意"只是编码不是加密"）；类型：`Opaque`、`kubernetes.io/dockerconfigjson`（拉镜像凭证）、`kubernetes.io/tls`、`service-account-token`。
**③ Volume（容器内挂载点）**：
- `emptyDir`：Pod 生命周期内的临时目录（同 Pod 多容器共享）；
- `hostPath`：挂宿主机目录（DaemonSet 常用，如日志）；
- **PVC（PersistentVolumeClaim）**：声明式"我要 N GB 存储"，由 PV（实际存储，如云盘/NFS/本地盘）供给。
**④ PV / PVC / StorageClass**：
- PV = 集群级存储资源（静态创建或动态供给）；PVC = 用户申请；**StorageClass** = 动态供给模板（`provisioner: ebs.csi.aws.com`、`nfs.csi.k8s.io`...），PVC 引用 StorageClass 自动创建 PV；
- **CSI（Container Storage Interface）**：存储插件化标准（云盘/NFS/Ceph 都是 CSI driver）。
**⑤ 训练场景**：checkpoint 目录、数据集、轨迹库（TrajStore 的磁盘）都用 PVC 持久化，Pod 重建数据不丢。

### 3. 具体数值样例

- 训练作业声明 `volumeClaimTemplates`（StatefulSet）或 Deployment 挂 PVC：`requests.storage: 500Gi` + `storageClassName: nfs` → 自动创建 500Gi NFS PV 并挂载到 `/data`；
- Secret 注入：`envFrom: secretRef` 或挂载成文件（`/etc/credentials`）——**华为云 TrajStore 密钥、E2B token 都走 Secret**；
- 实习场景：TrajStore 的轨迹数据落 PVC，Pod 重启轨迹还在；Checkpoint 权重落 PVC 供续训。

> 面试一句话总结：**配置用 ConfigMap、密钥用 Secret（etcd 里 base64）、持久化用 PV/PVC/StorageClass（CSI 动态供给）——三者都是"声明 + 注入"，让镜像与运行环境解耦；训练数据/checkpoint/轨迹库都挂 PVC 保证 Pod 重建不丢。**

---

## 4. 元数据与组织：Namespace / Label / Annotation / Finalizer / OwnerReference

### 1. 现有问题

集群里资源成千上万，怎么**分组、筛选、标记、清理**？删除父资源时子资源怎么跟着删？

### 2. 方法论

- **Namespace**：逻辑隔离单元（`default/kube-system/kube-node-lease`）；RBAC、ResourceQuota、NetworkPolicy 都可按 namespace 隔离——**多租户/多环境（dev/prod）的边界**；
- **Label / Selector**：键值标签 + 选择器（`equality: app=api` / `set-based: env in (dev,prod)`）；Service/RS/NetworkPolicy 都靠 selector 找目标——**K8S 的组织方式是"标签"，不是"层级"**；
- **Annotation**：非标识性元数据（版本、负责人、工具配置），不被 selector 使用；
- **Finalizer**：删除保护——资源标记删除（`deletionTimestamp`）后，**先等 Finalizer 列表清空**才真正删除；自定义 Operator 用它做"删除前清理"（如删 RayCluster 前先删底层 Pod/云资源）；
- **OwnerReference**：声明父子关系（Deployment 拥有 RS，RS 拥有 Pod）→ **级联删除**（删 Deployment 自动删 RS 和 Pod）+ 垃圾回收（GC controller）；
- UID：每个对象唯一标识，OwnerReference 用 UID 而非名字。

### 3. 具体数值样例

- `kubectl get pods -l app=raycluster-sample,role=worker`：按标签筛 worker Pod；
- 删 Deployment：其 RS 与 Pod 因 OwnerReference 级联删除；若 Pod 有 Finalizer，删除会被"挂起"直到 Finalizer 被清；
- KubeRay Operator 给 RayCluster 相关 Pod 打标签（`ray.io/cluster`、`ray.io/node-type: head|worker`），Service/NetworkPolicy 据此选择。

> 面试一句话总结：**Namespace 分租户、Label 组织资源（selector 是 K8S 的"关联方式"）、Annotation 存元数据、Finalizer 做删除前清理、OwnerReference 实现级联删除与 GC——理解这套元数据机制是理解"声明式系统如何自组织"的关键。**

---

## 5. 调度与资源：requests/limits / QoS / HPA / 亲和性 / 污点

### 1. 现有问题

- 多容器抢资源怎么办？调度器怎么选节点？GPU 怎么分配？负载高了怎么自动扩容？

### 2. 方法论

**① 资源请求与限制（requests/limits）**：
- `requests`：调度依据（保证值）；`limits`：上限（CPU 可压缩、内存不可压缩，超限 OOMKill）；
- **QoS 三类**：Guaranteed（requests==limits）、Burstable（requests<limits）、BestEffort（都不设）——驱逐顺序 BestEffort 先被赶；
- **GPU**：`nvidia.com/gpu: 1` 是**扩展资源**（Extended Resource），只能设 limits 不能设 requests（K8S 1.27+ 支持细粒度），由**设备插件**（GPU device plugin）上报节点可用 GPU。

**② HPA（HorizontalPodAutoscaler）**：按 CPU/内存/自定义指标（Prometheus）调整 Deployment 副本数：`minReplicas/maxReplicas` + `metrics` + `behavior`（扩容/缩容策略，缩容有冷却窗口）。

**③ 节点选择与亲和**：
- `nodeSelector`：简单键值匹配；
- **nodeAffinity**：`requiredDuringScheduling`（硬）/`preferredDuringScheduling`（软）；
- **podAffinity/antiAffinity**：与某类 Pod 同节点/异节点（训练：trainer 与 PS 同机、GPU Pod 与 GPU Pod 分散）；
- **Taint/Toleration**：节点打污点（`dedicated=gpu:NoSchedule`），只有带对应 toleration 的 Pod 能调度上去——**GPU 节点池隔离**的标配；
- **PriorityClass**：抢占（高优先级 Pod 挤掉低优先级）。

### 3. 具体数值样例

- 训练 Pod：`resources: {requests: {cpu: 8, memory: 32Gi, nvidia.com/gpu: 1}, limits: {...}}` → 调度器选有 1 张空闲 GPU 且满足 CPU/内存的节点；
- GPU 节点打污点 `nvidia.com/gpu=true:NoSchedule` + 训练 Pod 带 toleration → **普通 Pod 不会误占 GPU 节点**；
- HPA：`min: 1, max: 16`，指标 `pods/runner_queue_depth`（自定义）→ 队列深了自动扩 runner 副本（agent-lighting runner 扩缩容场景）。

> 面试一句话总结：**调度 = 资源（requests 是准入线、limits 是上限）+ 亲和/污点（控制落点）+ QoS（驱逐顺序）；GPU 是扩展资源由设备插件上报；HPA 按指标自动扩缩容——训练场景用"GPU 污点节点池 + Pod 亲和（trainer 同机）+ HPA（runner 弹性）"组合。**

---

## 6. 身份与安全：ServiceAccount / RBAC / SecurityContext / PodSecurity

### 1. 现有问题

- 集群内进程（如 Operator、训练作业）访问 API 要"身份"；谁能读/写什么资源要"授权"；容器要以最小权限跑。

### 2. 方法论

- **ServiceAccount（SA）**：Pod 内进程的身份（默认 `default`）；SA 绑定**挂载的 token**（`automountServiceAccountToken`），kubelet 挂到 `/var/run/secrets/kubernetes.io/serviceaccount/`；
- **RBAC 四件套**：`Role`（namespace 内权限）/ `ClusterRole`（集群级）/ `RoleBinding` / `ClusterRoleBinding`；权限 = `apiGroups + resources + verbs`（get/list/watch/create/update/patch/delete）；
- **SecurityContext**：容器/Pod 级安全设置（`runAsUser`、`privileged`、`capabilities`、`readOnlyRootFilesystem`）；
- **PodSecurity（PSA）**：准入层强制 Pod 安全级别（privileged/baseline/restricted）；
- **网络与密钥安全**：NetworkPolicy + Secret 最小暴露。

### 3. 具体数值样例

- KubeRay 的 autoscaler 需要读/写 RayCluster 状态 → Operator 自动创建 `ray-autoscaler-<cluster>` 的 SA + Role + RoleBinding（`reconcileAutoscalerServiceAccount/Role/RoleBinding` 源码可见）；
- 自定义训练 Operator：ClusterRole 授予 `rayclusters/get/list/watch/update` + `pods/...` + `jobs/...` 等 verbs，Controller 以该 SA 运行；
- Agent 沙箱 token（E2B）、云密钥全部走 Secret 注入，容器内 `runAsNonRoot: true` 降权。

> 面试一句话总结：**SA 是身份、RBAC 是授权（Role/ClusterRole + Binding）、SecurityContext/PSA 是容器降权、NetworkPolicy+Secret 是最小暴露——Operator 必须自备一套 SA/RBAC，KubeRay autoscaler 就是现成例子。**

---

## 7. 扩展机制：CRD / CR / Operator / Admission Webhook（本文的灵魂）

### 1. 现有问题

- K8S 只内置了 Deployment/Service 等"通用资源"；**训练作业、Ray 集群这类领域资源没有原生表达**——怎么办？
- K8S 的设计答案：**API 可扩展**——用户自己定义资源类型（CRD），自己写控制器（Operator）实现"声明式收敛"。

### 2. 方法论

**① CRD（CustomResourceDefinition）——定义一种新资源类型**：
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: rayclusters.ray.io
spec:
  group: ray.io                 # API 组（资源分组）
  names:
    kind: RayCluster            # 资源 kind
    plural: rayclusters         # 复数名（kubectl 用）
    shortNames: [raycluster]    # 简写
    singular: raycluster
  scope: Namespaced             # 集群级 or 命名空间级
  versions:
    - name: v1
      served: true
      storage: true
      schema:                   # OpenAPI v3 结构校验
        openAPIV3Schema:
          type: object
          properties:
            spec: { type: object, properties: {...} }
            status: { type: object, properties: {...} }
      subresources:
        status: {}              # 允许 kubectl status 子资源（状态与规格分离）
```

**② CR（CustomResource）——资源实例**：就是 `kind: RayCluster` 的 YAML（第 14 点）。
**③ Operator = CRD + Controller**：Controller 用 **reconcile loop** 监听 CR 变化，不断把"现实状态"向"CR 声明的期望状态"收敛（第 11 点详讲）；KubeRay、Prometheus Operator、cert-manager 都是 Operator。
**④ Admission Webhook**：资源写入 etcd 前的拦截器——`MutatingWebhookConfiguration`（改对象，如注入 sidecar/默认值）、`ValidatingWebhookConfiguration`（校验拒绝）；KubeRay 1.6+ 的 RayCluster webhook 会校验 rayStartParams 合法性。
**⑤ controller-runtime**（Kubebuilder/Operator SDK 的底层）：封装 informer + workqueue + reconcile 的 Go 框架。

### 3. 具体数值样例

- 用户写 20 行 `RayCluster` CR → KubeRay Operator 的 reconcile 自动创建 head Deployment、worker StatefulSet/Deployment、head Service、autoscaler RBAC——**用户只声明"要一个 1 head + 4 worker 的 Ray 集群"，其余全自动**；
- 自定义 `AgenticRLTrainJob` CR（第 22 点）→ 你的 Operator 自动拉起 store/algo/runner 工作负载、跟踪状态、清理资源；
- Webhook 示例：MutatingWebhook 给训练 Pod 自动注入 GPU 环境变量与 volume，用户 YAML 不用写这些样板。

> 面试一句话总结：**CRD 让用户自定义"资源类型"（group/kind/schema/status），Operator（CRD+Controller）用 reconcile 循环让现实向声明收敛，Webhook 在写入前拦截校验/注入——K8S 的扩展能力就是"把领域知识（Ray、训练、数据库）编码成资源语义"，这是本文后面 KubeRay 与自定义训练作业的地基。**

---

# 二、Kubernetes 架构与串讲（把概念串成框架）

## 8. 控制平面：kube-apiserver / etcd / kube-scheduler / kube-controller-manager

### 1. 现有问题

"声明式系统"的骨架是什么？谁来存状态、谁来接 API、谁来做决策、谁去执行？

### 2. 方法论（四个组件各司其职）

**① kube-apiserver——一切流量的唯一入口（REST API）**：
- 所有 kubectl/controller/scheduler/kubelet 的读写都走它；**只有 apiserver 能读写 etcd**；
- 请求链路：**认证（Authentication）→ 授权（Authorization/RBAC）→ 准入（Admission：Mutating→Validating）→ 校验 → 写 etcd → 返回 + 广播 watch 事件**；
- 内部结构（源码 `cmd/kube-apiserver/app/`）：`server.go` 用 Cobra 起命令，`config.go` 组装配置，`aggregator.go` 把**聚合 API**（metrics.k8s.io 等）与扩展 API（CRD）挂到同一网关；
- **watch 机制**：客户端（controller/kubelet）通过长连接 watch 资源变化，实现"事件驱动"而非轮询。

**② etcd——一致性的状态存储**：
- 分布式 KV 存储，**Raft 共识**（多数派写入），存储集群全部对象（含 Secret 的 base64）；
- 为什么 apiserver 是唯一写入口：保证 schema 校验/版本转换/准入都经过同一道闸。

**③ kube-scheduler——把 Pod 放到哪个节点**：
- watch 未调度的 Pod（`spec.nodeName` 为空）→ **过滤（Filtering，硬约束）→ 打分（Scoring，软偏好）** → 写 Binding；
- 硬约束：资源满足、节点亲和、污点容忍、端口不冲突；打分：资源余量、拓扑分布、亲和权重。

**④ kube-controller-manager——内置控制器集合**：
- 每个控制器是一个 **reconcile loop**（第 11 点）：Deployment controller（维护 RS）、RS controller（维护 Pod 数）、Job controller、EndpointSlice controller、GC controller（级联删除）、Namespace controller...（源码 `cmd/kube-controller-manager/app/` 的 `apps.go/batch.go/autoscaling.go` 分别注册各组控制器）；
- **cloud-controller-manager**：云厂商对接（LoadBalancer、Node 生命周期）。

### 3. 具体数值样例

- `kubectl create -f deploy.yaml` 的完整控制面路径：kubectl → apiserver（认证→RBAC→准入→写 etcd）→ Deployment controller watch 到新对象 → 创建 RS → RS controller 创建 Pod 对象 → scheduler watch 到未调度 Pod → 选节点写 Binding → 目标节点 kubelet 执行（第 10 点）；
- 30 个节点集群：3 副本 apiserver + 3 副本 etcd（raft）+ 1 主 scheduler/controller-manager（可多副本 leader 选举）；
- 一次 watch：controller 通过 `watch` 长连接收到 Pod 增删改事件，入 workqueue，触发 reconcile——**不是轮询，是事件驱动**。

> 面试一句话总结：**控制平面四件套：apiserver 是唯一 API 入口（认证→授权→准入→etcd→watch）、etcd 用 Raft 存全部状态、scheduler 过滤+打分选节点、controller-manager 跑一堆内置 reconcile 循环——"声明式"的骨架就是"写进去 + watch + 收敛"。**

---

## 9. 数据平面：kubelet / kube-proxy / 容器运行时 / CNI / CSI

### 1. 现有问题

控制面只做"决策"，真正把容器跑起来、把网络打通、把盘挂上的是节点侧组件。

### 2. 方法论

- **kubelet**：每个节点上的"代理"，核心职责：watch 分配到本节点的 Pod → 通过 **CRI**（Container Runtime Interface，gRPC）指挥容器运行时创建/启停容器 → 汇报状态（Pod 状态、资源用量、探针结果）→ 执行探针（liveness/readiness/startup）→ 挂载 volume（CSI）；
- **容器运行时**：containerd / CRI-O（实现 CRI），内部用 runc 起容器、sandbox（pause 容器）持有网络命名空间；
- **kube-proxy**：实现 Service 的转发规则（iptables/ipvs 模式），watch Service/EndpointSlice 更新规则；
- **CNI 插件**：容器创建时被调用（`ADD/DEL`），分配 Pod IP、配置网络（Calico/Cilium/flannel）；
- **CSI 驱动**：kubelet 挂载卷时调用 CSI 插件（云盘/NFS）。

### 3. 具体数值样例

- Pod 分配到 node1 后：kubelet watch 到 → 调 CRI `RunPodSandbox`（pause 容器起网络）→ CRI 调 CNI 给 sandbox 分配 IP → 调 CRI `CreateContainer` 起业务容器 → 挂 CSI 卷 → 容器启动 → 就绪探针通过 → 加入 Service 端点；
- GPU 节点：设备插件（DaemonSet）上报 `nvidia.com/gpu: 8` → scheduler 据此分配 → kubelet 注入 GPU 环境变量。

> 面试一句话总结：**数据平面 = kubelet（执行者：CRI 起容器、CSI 挂卷、探针探活、汇报状态）+ kube-proxy（Service 转发）+ CNI（Pod 网络）+ 容器运行时（containerd/CRI-O）——控制面做决策，节点侧做执行，全靠 watch + gRPC 接口解耦。**

---

## 10. 一次 Pod 从创建到运行的完整流程（串讲 1~9）

### 1. 现有问题

把前面所有概念串成一条链路，面试必考。

### 2. 方法论（全链路 8 步）

```text
① 用户：kubectl apply -f deploy.yaml
② kube-apiserver：认证（你是谁）→ 授权（RBAC 允许吗）→ 准入（webhook 改/拒）
   → schema 校验 → 写入 etcd → 返回成功 → 向 watch 客户端广播事件
③ Deployment controller（controller-manager 内）：watch 到新 Deployment
   → reconcile：创建 ReplicaSet（期望副本 3）
④ ReplicaSet controller：watch 到 RS → reconcile：创建 3 个 Pod 对象（spec.nodeName 为空）
⑤ kube-scheduler：watch 到未调度 Pod → 过滤（资源/亲和/污点）→ 打分 → 选 node2
   → 写 Pod 的 spec.nodeName=node2（Binding）
⑥ node2 的 kubelet：watch 到"本节点的 Pod" → 调 CRI 创建 sandbox（pause）
   → CNI 分配 Pod IP → 调 CRI 创建业务容器 → CSI 挂卷 → 容器启动
⑦ kubelet：运行探针（startup→readiness）→ 就绪后上报 Pod 状态 Running
⑧ EndpointSlice controller：把 Pod IP 加入 Service 端点 → kube-proxy 更新转发规则
   → 客户端访问 Service 名即可负载均衡到新 Pod
```

**关键点**：
- **全程事件驱动**（watch + workqueue + reconcile），不是轮询；
- **每一层只负责一件事**（scheduler 只选节点、kubelet 只执行、controller 只收敛）；
- 任何一步失败：controller 重试（指数退避），Pod 重启（restartPolicy），最终"现实=期望"。

### 3. 具体数值样例

- 3 副本 Deployment 更新镜像：Deployment 滚动更新 → 新 RS 起 1 个新 Pod（maxSurge=1）→ 新 Pod 就绪 → 旧 RS 删 1 个 → 循环 3 次 → 全部替换，Service 全程可用；
- 节点宕机：kubelet 心跳超时 → node 标记 NotReady → 控制器把该节点 Pod 标记失败 → 在其他节点重建（受 PodDisruptionBudget 约束）。

> 面试一句话总结：**一次 Pod 创建 = 用户写声明 → apiserver（认证授权准入+etcd）→ 控制器逐层收敛（Deployment→RS→Pod）→ scheduler 选节点 → kubelet 经 CRI/CNI/CSI 真正拉起 → 探针就绪 → Service 端点生效——全链路事件驱动、职责单一、失败自愈，这就是"声明式编排"的完整循环。**

---

## 11. Controller 模式：reconcile loop（理解 Operator 的钥匙）

### 1. 现有问题

为什么叫"控制器"？"声明式"到底怎么实现"自动收敛"？

### 2. 方法论

**reconcile 循环**（所有 controller/operator 的通用骨架）：

```text
┌────────────────────────────────────────────────────┐
│ 1. LIST/WATCH 目标资源（如 RayCluster CR）+ 依赖资源 │
│    → informer 维护本地缓存（list 一次 + watch 增量）  │
│ 2. 事件入 workqueue（去重、限速、重试）              │
│ 3. Reconcile(key) 被调用：                          │
│    a. GET 当前 CR（期望状态）                       │
│    b. LIST 现实资源（现有 Deployment/Pod/Service）   │
│    c. 对比期望 vs 现实 → 生成 diff 动作列表          │
│    d. 执行动作（create/update/delete/status 更新）   │
│    e. 返回 (requeueAfter 或 error)                  │
│ 4. 出错 → 指数退避重试；成功 → 等待下次事件          │
└────────────────────────────────────────────────────┘
```

- **informer 模式**：`ListAndWatch` + **本地缓存**（读本地缓存、不直接打 apiserver）——这是控制器性能的关键；
- **workqueue**：事件去重合并（同一对象的多个事件合成一个 reconcile）；
- **Status 子资源**：reconcile 把"当前进展"写回 CR 的 `.status`（用户 `kubectl get raycluster -o yaml` 能看到），**spec 是期望、status 是现实**；
- **事件（Events）**：reconcile 记录 `kubectl describe` 可见的事件（"Created head deployment"）。

KubeRay 的 `RayClusterReconciler` 就是标准实现（`ray-operator/controllers/ray/raycluster_controller.go`）：

```go
func (r *RayClusterReconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
    ...
    return r.rayClusterReconcile(ctx, instance)
}

func (r *RayClusterReconciler) rayClusterReconcile(ctx context.Context, instance *rayv1.RayCluster) (ctrl.Result, error) {
    // 依次收敛子资源：autoscaler 的 SA/Role/RoleBinding、head Service、head/worker Pods...
    r.reconcileAutoscalerServiceAccount, r.reconcileAutoscalerRole, r.reconcileAutoscalerRoleBinding,
    r.reconcilePods, ...
}
```

### 3. 具体数值样例

- 用户把 RayCluster 的 `workerGroupSpecs[0].replicas` 从 2 改成 4：Operator 的 informer watch 到更新 → workqueue 入队 → reconcile 对比"期望 4 vs 现实 2" → 创建 2 个 worker Pod → 更新 status（`readyWorkerReplicas: 4`）；
- 删 RayCluster：deletionTimestamp + Finalizer → reconcile 先删子资源（Pod/Service）→ 清 Finalizer → 真正删除；
- **面试点：reconcile 必须"幂等"**（重放同一事件结果一致），且**一个对象一个请求**（workqueue 保证）。

> 面试一句话总结：**Controller 模式 = informer（list+watch+本地缓存）+ workqueue（去重限速）+ Reconcile（对比期望 spec 与现实资源、执行 diff 动作、写回 status）——幂等、事件驱动、失败退避重试；KubeRay 和任何自定义 Operator 都是这个骨架，理解它就能读懂一切 Operator。**

---

## 12. kube-apiserver 内部：认证→授权→准入→存储→watch（源码视角）

### 1. 现有问题

apiserver 是"唯一入口"，它内部到底怎么处理一个请求？看源码怎么对应。

### 2. 方法论（源码对应）

- 入口：`cmd/kube-apiserver/app/server.go`（Cobra）→ `config.go`（构建配置）→ `CreateServerChain` 把三个 API 服务器串成一条链：**aggregator（聚合 API）→ apiextensions-apiserver（CRD）→ kube-apiserver（核心资源）**；
- 请求处理管线（kube-apiserver 内）：`Authentication`（X.509/Bearer Token/Webhook）→ `Authorization`（RBAC/ABAC/Webhook）→ `Admission`（Mutating 先改、Validating 后拒）→ 存储层（`pkg/registry` 里每个资源的 REST 实现）→ etcd（`pkg/registry/.../storage.go` 用 `storage.Interface` 抽象）；
- **watch 实现**：apiserver 维护 etcd watch → 转成 REST watch 流，客户端（informer）长连接接收；
- 为什么 CRD 也走同一管线：apiextensions-apiserver 把 CRD 的 schema 编译成通用存储，复用同一套认证/授权/准入/watch 机制。

### 3. 具体数值样例

- 一次 `kubectl get rayclusters`：kubectl → apiserver（tls 认证 → RBAC 查 `ray.io/rayclusters` 的 get 权限 → 无 webhook）→ apiextensions-apiserver 从 etcd 读 CR → 返回；
- 一次 `kubectl apply -f rayjob.yaml`：认证 → RBAC → MutatingWebhook（KubeRay 默认注入）→ ValidatingWebhook（校验字段）→ 写 etcd → watch 广播 → RayJob controller 收到事件开始工作。

> 面试一句话总结：**apiserver = 统一网关：认证→授权→准入（Mutating/Validating）→schema 校验→etcd 存储→watch 广播，CRD 复用同一管线（apiextensions-apiserver 挂链）——所以 CR 与内置资源有完全一致的 API 体验。**

---

# 三、KubeRay：把 Ray 变成 Kubernetes 原生资源

## 13. KubeRay 架构：Operator + 四类 CRD

### 1. 现有问题

- Ray 集群（head + workers）直接裸机/容器起，生命周期、故障恢复、扩缩容、网络都要手动管；
- 能不能"像声明 Deployment 一样声明一个 Ray 集群"？——这就是 KubeRay（ray-project/kuberay）。

### 2. 方法论

**KubeRay = Ray 的 Kubernetes Operator**（本地克隆 `kuberay`，HEAD `ffec815`）：

```text
┌─────────────────────────────────────────────────────────┐
│ KubeRay Operator（ray-operator，controller-runtime 实现） │
│   CRD（apis/ray/v1）：                                    │
│   ├── RayCluster    —— 声明一个 Ray 集群（head+workers）   │
│   ├── RayJob        —— 跑一个 Ray 训练/推理任务           │
│   ├── RayService    —— Ray Serve 应用的高可用部署         │
│   └── RayCronJob    —— 定时 RayJob                       │
└─────────────────────────────────────────────────────────┘
        │ reconcile
        ▼
  创建/管理：head Deployment + head Service、worker Deployment/StatefulSet、
  autoscaler（SA/Role/RoleBinding）、NetworkPolicy、Ingress/Route...
```

- **控制器**（`ray-operator/controllers/ray/`）：`raycluster_controller.go`、`rayjob_controller.go`、`rayservice_controller.go`、`raycronjob_controller.go`、`networkpolicy_controller.go`；
- **部署方式**：Helm chart（`helm-chart/kuberay-operator`）或 Kustomize（`config/`）；Operator 本身跑在 `kuberay-system` namespace；
- **版本约定**：`ray.io` API 组，当前主版本 `v1`（旧 `v1alpha1` 已废弃迁移）。

### 3. 具体数值样例

- 装好 KubeRay 后 `kubectl apply -f raycluster.yaml`（1 head + 4 worker）→ Operator 在 ~1 分钟内拉起 head Pod（自动执行 `ray start --head`）与 worker Pod（`ray start --address=head:6379`），自动建 head Service（`<name>-head-svc`）；
- 用户 `kubectl get rayclusters` 看集群状态（`state: ready`、`desiredWorkerReplicas/readyWorkerReplicas`）；
- 扩展：`ray-operator` 的 `podpool/`（Pod 池预热）、`kubectl-plugin/`（`kubectl ray` 插件）、`dashboard/`（KubeRay dashboard）。

> 面试一句话总结：**KubeRay 是 Ray 的 Operator：用 RayCluster/RayJob/RayService/RayCronJob 四类 CRD 声明式描述 Ray 集群与任务，Operator 的 reconcile 自动创建 head/worker 的 K8S 工作负载、Service、autoscaler RBAC、NetworkPolicy——用户从此"声明集群"而非"手动搭集群"。**

---

## 14. RayCluster CRD 详解（head / worker group）

### 1. 现有问题

声明一个 Ray 集群要表达什么？head 和 worker 各是什么？GPU 怎么分配？自动扩缩容怎么开？

### 2. 方法论（`ray-operator/apis/ray/v1/raycluster_types.go`）

**RayClusterSpec 核心字段**：

```go
type RayClusterSpec struct {
    HeadGroupSpec   HeadGroupSpec     `json:"headGroupSpec"`    // head 组
    WorkerGroupSpecs []WorkerGroupSpec `json:"workerGroupSpecs"` // worker 组（可多个）
    EnableInTreeAutoscaling *bool      `json:"enableInTreeAutoscaling,omitempty"` // 内置 autoscaler
    // NetworkPolicy 配置、head Service annotations、升级策略（RayClusterUpgradeStrategy）...
}

type HeadGroupSpec struct {
    Template corev1.PodTemplateSpec `json:"template"`        // 完整 Pod 模板（镜像/资源/命令）
    RayStartParams map[string]string `json:"rayStartParams"` // ray start 参数（node-manager-port、object-store-memory...）
    ServiceType corev1.ServiceType  `json:"serviceType,omitempty"` // head Service 类型
}

type WorkerGroupSpec struct {
    GroupName  string             `json:"groupName"`          // 组名（-l ray.io/group=...）
    Replicas   *int32             `json:"replicas,omitempty"`
    MinReplicas *int32            `json:"minReplicas"`        // autoscaler 下界
    MaxReplicas *int32            `json:"maxReplicas"`        // autoscaler 上界
    Template   corev1.PodTemplateSpec `json:"template"`       // Pod 模板
    RayStartParams map[string]string `json:"rayStartParams"`  // ray start 参数（address 自动指向 head）
}
```

**要点**：
- **head**：Ray 集群的 GCS/调度器（`ray start --head`），KubeRay 用 **Deployment** 承载（head 挂了重建）；head 的 Service（`<cluster>-head-svc`，默认 ClusterIP，可配 NodePort/LoadBalancer）是所有 worker 与外部访问的入口；
- **worker**：一个 RayCluster 可有多个 worker group（不同规格：CPU 组 / GPU 组），每组是 Deployment 或 StatefulSet（`headless` worker 需要稳定身份时用 StatefulSet）；worker 的 `rayStartParams.address` 由 Operator 自动填 head FQDN；
- **GPU 分配**：在 worker group 的 Pod 模板 `resources.limits."nvidia.com/gpu": 1` 声明，Ray 的 `num_gpus` 与 K8S GPU 配额对应；
- **In-tree autoscaling**：`enableInTreeAutoscaling: true` + min/maxReplicas → Operator 自动部署 Ray autoscaler（复用第 6 点的 SA/Role/RoleBinding），按 Ray 的调度需求扩缩 worker；
- **网络**：head Service FQDN（`<cluster>-head-svc.<ns>.svc`）+ NetworkPolicy（head/worker 各自的 ingress/egress 规则）。

### 3. 具体数值样例

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: raycluster-sample
spec:
  headGroupSpec:
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:latest
            resources: {requests: {cpu: 4, memory: 16Gi}, limits: {cpu: 4, memory: 16Gi}}
    rayStartParams:
      dashboard-host: "0.0.0.0"
  workerGroupSpecs:
    - groupName: cpu-group
      replicas: 2
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray:latest
              resources: {requests: {cpu: 8, memory: 32Gi}}
      rayStartParams: {}
    - groupName: gpu-group
      replicas: 1
      minReplicas: 1
      maxReplicas: 4
      enableInTreeAutoscaling: true
      template:
        spec:
          containers:
            - name: ray-worker-gpu
              image: rayproject/ray:latest
              resources: {limits: {"nvidia.com/gpu": 1}}
```

- 效果：Operator 创建 `raycluster-sample-head-xxx`（Deployment 1 副本）、`cpu-group`（Deployment 2 副本）、`gpu-group`（Deployment 1 副本，可自动扩到 4），head Service `raycluster-sample-head-svc`；
- 删除 RayCluster：Finalizer 清理 → head/worker Pod 与 Service 全部级联删除。

> 面试一句话总结：**RayCluster = 1 个 head（GCS/调度，Deployment 承载 + head Service 入口）+ N 个 worker group（按规格分组、各自 replicas、min/max 支持内置 autoscaler、GPU 在 Pod 模板里申请）；Operator 把"声明"翻译成 head/worker 的 K8S 工作负载 + Service + RBAC + NetworkPolicy——这就是"Ray 集群即代码"。**

---

## 15. RayJob 与 RayService：任务与 Serve 应用

### 1. 现有问题

光有集群不够——训练任务怎么提交？Serve 应用怎么声明式部署、滚动更新？

### 2. 方法论

**① RayJob**（`rayjob_types.go`）——跑一个 Ray 任务（训练/评估）：
- 核心字段：`rayClusterSpec`（任务专用的 RayCluster 模板）+ `entrypoint`（要执行的 `ray job submit` 命令）+ `submitter` 配置 + `shutdownAfterJobFinishes`（任务结束是否自动删集群）+ `ttlSecondsAfterFinished`（保留多久）；
- **两种提交模式**：`K8sJobMode`（submitter 是独立的 K8s Job，跑 `ray job submit`；默认）/ `SidecarMode`（submitter 与 head 同 Pod 的 sidecar）；
- 生命周期：创建集群 → 提交任务 → 跟踪状态（`jobStatus: RUNNING/SUCCEEDED/FAILED`）→ 可选自动清理集群——**"任务即资源"**。

**② RayService**（`rayservice_types.go`）——Ray Serve 应用：
- 核心字段：`rayClusterConfig`（RayCluster 模板）+ `serveConfigV2`（Serve deployment graph 的 YAML）+ 滚动更新策略（`serveDeploymentGraphSpec` 的 autoscaling）；
- 能力：部署新版本集群 → 验证 Serve 健康 → 切换流量 → 删除旧集群（**蓝绿/金丝雀式 Serve 发布**）。

**③ RayCronJob**：按 cron 定时创建 RayJob（如周期评估）。

### 3. 具体数值样例

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: rayjob-sample
spec:
  entrypoint: python /home/ray/samples/sample_code.py
  shutdownAfterJobFinishes: true      # 跑完自动删 RayCluster（省钱）
  ttlSecondsAfterFinished: 600        # 结果保留 10 分钟
  rayClusterSpec:
    headGroupSpec: {...}
    workerGroupSpecs: [...]
```

- 提交后：Operator 建临时 RayCluster → submitter（K8s Job）执行 `ray job submit --address http://<cluster>-head-svc:8265 -- entrypoint ...` → 任务完成 → 状态 `SUCCEEDED` → 集群被清理；
- **训练场景**：每次训练 = 一个 RayJob（自带独立集群），练完即删——但 AgenticRL 训练要"常驻引擎 + 多轮 rollout"，更适合自定义 CRD（第 21 点）。

> 面试一句话总结：**RayJob 把"训练/任务"声明成资源（集群模板 + entrypoint + 跑完自动清理），RayService 把 Serve 应用声明成资源（serveConfigV2 + 蓝绿切换），RayCronJob 定时触发——三者共享 RayCluster 作为底层集群模板。**

---

## 16. KubeRay 控制器实现（reconcile 骨架，源码）

### 1. 现有问题

KubeRay 的 Operator 代码长什么样？reconcile 里具体做什么？

### 2. 方法论（`ray-operator/controllers/ray/raycluster_controller.go`）

```go
// NewReconciler：用 controller-runtime 注册
func NewReconciler(mgr manager.Manager, options RayClusterReconcilerOptions) *RayClusterReconciler {...}

// Reconcile：标准入口，把请求转给内部实现
func (r *RayClusterReconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
    instance := &rayv1.RayCluster{}
    if err := r.Get(ctx, request.NamespacedName, instance); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)   // 不存在则忽略
    }
    return r.rayClusterReconcile(ctx, instance)
}

// rayClusterReconcile：串起所有子资源收敛步骤
func (r *RayClusterReconciler) rayClusterReconcile(ctx context.Context, instance *rayv1.RayCluster) (ctrl.Result, error) {
    // 1. 清理孤儿 Pod（deleteAllPods：按 owner/label 清理不属于本集群的 Pod）
    // 2. 收敛 autoscaler 相关资源：
    r.reconcileAutoscalerServiceAccount, r.reconcileAutoscalerRole, r.reconcileAutoscalerRoleBinding,
    // 3. 收敛 head Service（reconcileService）
    // 4. 收敛 head/worker 的 Pods（reconcilePods：建 Deployment/StatefulSet）
    // 5. 更新 status（headPodIP、readyWorkerReplicas、state...）
    ...
}
```

**关键点**：
- **幂等**：每次 reconcile 都是"全量对比期望 vs 现实"，重复执行结果一致；
- **owner 关系**：创建的 head/worker Deployment 的 `OwnerReference` 指向 RayCluster → 级联删除；
- **status 更新**：head Pod IP、worker 就绪数、集群 state（`pending/ready`）写回 CR 的 `.status`；
- **事件**：`r.Recorder.Event(instance, corev1.EventTypeNormal, "Created", "Created head deployment")` 让 `kubectl describe` 可见。

### 3. 具体数值样例

- 首次 apply RayCluster：reconcile 创建 head Deployment + head Service + worker Deployment → status 更新 `state: ready`；
- worker 手动被删：informer watch 到 Pod 删除 → 由于 Deployment 存在，其 controller 自动重建（K8S 自身收敛）——Operator 只管"声明了哪些 Deployment"，Pod 数量由 Deployment 管；
- autoscaler 把 worker 从 1 扩到 4：KubeRay 直接更新 worker group 的 Deployment replicas。

> 面试一句话总结：**KubeRay 控制器 = 标准 reconcile：Get CR → 清理孤儿 → 收敛 autoscaler RBAC/head Service/head-worker Pods → 更新 status；子资源挂 OwnerReference 实现级联删除，全部幂等——看懂了它就看懂了怎么写任何 Ray 相关的 Operator。**

---

# 四、分离式 agent-lighting 在 K8S 上的部署（重点）

## 17. agent-lighting 三件套回顾与通信矩阵

### 1. 现有问题

在 K8S 上编排 agent-lighting 之前，先固化"三件套各自是什么、谁和谁通信、走什么协议"。

### 2. 方法论（回顾 + 通信矩阵）

- **Algorithm（大脑）**：决定任务、学习、更新资源（模型/提示词）；VERL 集成 = `AgentLightningTrainer(RayPPOTrainer)` + `AgentModeDaemon`（v1 模式默认）；
- **Runner（工人）**：从 store 取任务（`dequeue_rollout`）、跑 agent、流式回传 span（`add_span/add_otel_span`）、更新 attempt 状态；Runner→Agent 永远是**进程内**（LitAgentRunner 持有 agent）；
- **LightningStore（数据库+队列）**：任务队列 + 资源 + span 存储，唯一"真相源"；接口 `enqueue_rollout/dequeue_rollout/add_span/get_latest_resources/wait_for_rollouts/query_spans/update_attempt`；
- **LLMProxy**：Agent 与模型之间的 HTTP 桥（OpenAI 兼容），算法可动态换后端（RL 时换成刚训好的 vLLM 端点）；
- **通信矩阵**（三机分离版，见 `Agent-Lighting.md`）：

| 链路 | 协议 | 说明 |
|---|---|---|
| Runner ↔ Store | **HTTP**（`LightningStoreClient` → `LightningStoreServer`，`/v1/agl/*`，端口 4747） | 取任务/回传 span |
| Algorithm ↔ Store | **HTTP**（server 内嵌 client 或直连） | 入队/查 span/更新资源 |
| Runner → Agent | **进程内调用**（同进程） | 无网络 |
| Agent → LLMProxy | **HTTP**（OpenAI 兼容端点） | 推理调用 |
| LLMProxy → vLLM | **HTTP**（OpenAI 兼容 `/v1/chat/completions`） | 推理后端 |
| Algorithm → vLLM | **HTTP**（注册为 proxy 后端 / 直连） | RL 训练引擎 |

**三机分离形态**（对应"store 机 / algo 机 / agent 机"）：
- store 机：`agl store --host 0.0.0.0 --port 4747`（`agentlightning/cli/store.py`：backend=memory/mongo、asyncio/mp 启动）；
- algo 机：算法进程（含 Trainer/VERL/vLLM/LLMProxy），`server_host` 指向 store 机；
- agent 机：runner 进程（`role="runner"`），`server_host=<store 机 IP> server_port=4747` 连 store。

### 3. 具体数值样例

- 一批 48 个任务：Algorithm `enqueue_rollout` ×48 → store 队列；12 个 runner 进程 `dequeue_rollout` 抢任务并行执行 → span 流式回写 → Algorithm `query_spans` 拉取 → `TracerTraceToTriplet` 转 `(prompt, response, reward)` → VERL 更新权重 → vLLM 引擎热更新 → 下一轮；
- 全部跨机通信只有两类：**① 各组件↔store 的 HTTP（4747）② Agent↔LLMProxy/vLLM 的 HTTP**——这就是"分离式"的通信面，K8S 编排只需暴露这两个网络面。

> 面试一句话总结：**agent-lighting 三件套的通信极简：Runner/Algorithm 与 LightningStore 之间走 HTTP（4747，唯一跨机数据面）、Agent 与 LLMProxy/vLLM 走 HTTP（OpenAI 兼容）、Runner→Agent 进程内——K8S 上要做的只是"把 store/algo/runner 各编排成一个工作负载 + 暴露这两个 HTTP 面"。**

---

## 18. K8S 编排方案总览：一个 RayCluster 内部分工 vs 独立工作负载

### 1. 现有问题

三件套 + vLLM + 训练（VERL 依赖 Ray）在 K8S 上怎么摆？两种思路各有什么取舍？

### 2. 方法论（两种方案对比）

**方案 A：一个 RayCluster 内部分角色（推荐，贴合 VERL）**
- VERL 的 `AgentLightningTrainer(RayPPOTrainer)` 本身就跑在 Ray 上，所以用**一个 RayCluster** 承载全部角色，靠 **Ray 的 worker group + 资源（num_gpus/num_cpus）** 区分职责：
  - **head Pod**：GCS + Ray 控制面 + 训练入口（AgentLightningTrainer/AgentModeDaemon）；
  - **algo 组**（worker group，GPU）：VERL 训练 worker + vLLM 引擎（`rollout.ray` 的 vLLMHttpServer actor）+ LLMProxy；
  - **runner 组**（worker group，CPU/GPU）：跑 `LitAgentRunner`（`role="runner"`），agent 进程内执行；
  - **store**：可以是 Ray 之外的独立 Deployment（更稳，或与 head 同 Pod sidecar）；
- **优点**：复用 Ray 的资源调度/容错/扩缩容；VERL 原生集成；**缺点**：所有角色共用一个集群，故障域耦合。

**方案 B：独立工作负载（服务化，最贴合"三机分离"语义）**
- **LightningStore** → 独立 `Deployment + Service`（`agl store`，backend=mongo 持久化）；
- **Algorithm/训练** → 独立 `Deployment`（或 RayJob/自定义 CRD）跑 `AgentLightningTrainer`，内含 vLLM + LLMProxy；
- **Runner** → 独立 `Deployment/StatefulSet` + HPA（按队列深度扩缩容），`server_host=<store-svc>`；
- **vLLM 引擎** → 独立 `Deployment + Service`（若 algo 与 vLLM 分离部署）；
- **优点**：各角色独立扩缩容/升级/故障域（真正的"训练、推理、环境、奖励解耦"）；**缺点**：要自己编排跨组件依赖与网络。

**选型建议**：小规模/单机调试用方案 A（一个 RayCluster）；生产/多租户/分离式平台用方案 B（各组件独立工作负载 + 自定义 CRD 编排，即实习的形态）。

### 3. 具体数值样例

- 方案 A：1 head + algo 组 2 GPU + runner 组 8 CPU，一个 `raycluster.yaml` 搞定；`ray status` 看到全部角色；
- 方案 B：`store` Deployment 1 副本 + Service；`algo` Deployment 1 副本（2 GPU）；`runner` Deployment 12 副本（HPA min 4 max 32）；`vllm` Deployment 1 副本（1 GPU）+ Service；
- 面试表述：**"分离式在 K8S 上的本质 = 每个角色一个工作负载 + 两个 HTTP 面（store 的 4747、模型端点）用 Service 打通，Runner 用 HPA 按队列深度弹性扩缩。"**

> 面试一句话总结：**两种编排：一个 RayCluster 内用 worker group 分工（贴合 VERL，耦合紧）或独立工作负载（store/algo/runner/vllm 各一个 Deployment+Service，真正解耦可独立扩缩）；生产分离式平台选后者——store 用 ClusterIP 内网 + 需要时 LoadBalancer/Ingress 公网暴露，runner 用 HPA 弹性。**

---

## 19. 分别起 store / algo / runner 的 K8S 编排（含 YAML）

### 1. 现有问题

把方案 B 落到具体的 K8S 资源上：每个角色需要什么工作负载、什么 Service、什么配置。

### 2. 方法论（三个角色的完整编排）

**① LightningStore（store 机 → Deployment + Service）**：

```yaml
# 01-store.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agl-store
  labels: {app: agl-store}
spec:
  replicas: 1
  selector: {matchLabels: {app: agl-store}}
  template:
    metadata: {labels: {app: agl-store}}
    spec:
      containers:
        - name: store
          image: agentlightning:0.3.0        # 镜像内含 agl CLI
          command: ["agl", "store"]
          args: ["--host", "0.0.0.0", "--port", "4747",
                 "--backend", "mongo", "--mongo-uri", "mongodb://mongo:27017/?replicaSet=rs0"]
          ports: [{containerPort: 4747}]
          resources: {requests: {cpu: "2", memory: 4Gi}, limits: {cpu: "4", memory: 8Gi}}
          readinessProbe: {httpGet: {path: /health, port: 4747}}
          volumeMounts: [{name: store-data, mountPath: /data}]   # span 持久化
      volumes:
        - name: store-data
          persistentVolumeClaim: {claimName: agl-store-pvc}
---
apiVersion: v1
kind: Service
metadata:
  name: agl-store
spec:
  selector: {app: agl-store}
  ports: [{port: 4747, targetPort: 4747}]      # 集群内 ClusterIP
  # 需要公网访问时改 type: LoadBalancer（对应实习"TrajStore 公网 URL 自动暴露"）
```

**② Algorithm（algo 机 → Deployment，含 VERL 训练 + vLLM + LLMProxy）**：

```yaml
# 02-algo.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agl-algo
  labels: {app: agl-algo}
spec:
  replicas: 1                                   # 算法侧单进程（复杂并行交给 VERL/DeepSpeed）
  selector: {matchLabels: {app: agl-algo}}
  template:
    metadata: {labels: {app: agl-algo}}
    spec:
      containers:
        - name: algo
          image: agentlightning:0.3.0
          command: ["agl", "train"]             # 或 python -m 训练入口
          env:
            - name: AGL_STORE_URL
              value: "http://agl-store:4747/v1/agl"   # ← store 的 Service 名（DNS 自动解析）
            - name: ROLE
              value: "algorithm"
            - name: VLLM_ENDPOINT
              value: "http://agl-vllm:8000/v1"       # vLLM 引擎 Service
          resources: {limits: {"nvidia.com/gpu": "2"}}   # VERL 训练 + vLLM 共享/分离
      serviceAccountName: agl-algo-sa           # RBAC（如需要操作 Ray/CRD）
---
apiVersion: v1
kind: Service
metadata: {name: agl-algo}                      # 可选：算法侧暴露的端点（如 LLMProxy）
spec:
  selector: {app: agl-algo}
  ports: [{port: 8000, targetPort: 8000}]       # LLMProxy / vLLM OpenAI 兼容端口
```

**③ Runner（agent 机 → Deployment + HPA）**：

```yaml
# 03-runner.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agl-runner
  labels: {app: agl-runner}
spec:
  replicas: 12
  selector: {matchLabels: {app: agl-runner}}
  template:
    metadata: {labels: {app: agl-runner}}
    spec:
      containers:
        - name: runner
          image: agentlightning:0.3.0
          command: ["agl", "run"]
          args: ["--role", "runner"]            # agent 机
          env:
            - name: AGL_STORE_URL
              value: "http://agl-store:4747/v1/agl"     # ← 指向 store Service（三机分离的 HTTP 面）
            - name: SERVER_HOST
              value: "agl-store"
            - name: SERVER_PORT
              value: "4747"
            - name: LLM_PROXY_URL
              value: "http://agl-algo:8000/v1"          # Agent → LLMProxy（HTTP）
            - name: SANDBOX_TOKEN                # 腾讯沙箱密钥
              valueFrom: {secretKeyRef: {name: sandbox-cred, key: token}}
          resources: {requests: {cpu: "4", memory: 8Gi}}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: agl-runner-hpa}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: agl-runner}
  minReplicas: 4
  maxReplicas: 32
  metrics:
    - type: Pods
      pods:
        metric: {name: agl_runner_queue_depth}   # 按 store 队列深度扩缩 runner（Prometheus adapter）
```

**④ 配套**：vLLM 引擎独立 Deployment + Service（`agl-vllm:8000`）；MongoDB（store 后端）StatefulSet + PVC；Secret（沙箱/云密钥）；NetworkPolicy（只放行 store 4747 与模型 8000）。

### 3. 具体数值样例

- 网络面：所有组件只用两个 Service——`agl-store:4747`（数据面）与 `agl-algo:8000` / `agl-vllm:8000`（推理面）；Runner 不需要对外暴露（纯消费者）；
- 扩缩容：队列积压 100 个任务 → HPA 把 runner 从 4 扩到 32 → 队列清空 → 缩回 4（`behavior.scaleDown.stabilizationWindowSeconds` 防抖）；
- 故障：runner Pod 崩溃 → Deployment 重建（新 Pod 从 store 重新 `dequeue_rollout`，attempt 语义保证不丢任务）；store 崩溃 → readiness 探针失败 → Service 摘除 → 重建后 mongo 数据仍在；
- 公网：`agl-store` Service 改 `type: LoadBalancer` → 云平台分配公网 IP → 用户 agent（任意位置）通过公网 URL 连 store——**这就是实习"TrajStore 公网 URL 自动暴露与生命周期管理"的 K8S 落点**。

> 面试一句话总结：**分离式三件套在 K8S 上 = store 一个 Deployment+Service（4747，mongo 持久化，可 LoadBalancer 公网暴露）、algo 一个 Deployment（VERL+vLLM+LLMProxy，GPU）、runner 一个 Deployment+HPA（按队列深度弹性）——所有跨机通信收敛到两个 HTTP 面（store 4747、模型 8000），用 Service 名解耦 IP，这就是"分离式架构的 K8S 化"。**

---

## 20. 与实习对照：RayCluster Head + TrajStore 双网络链路

### 1. 现有问题

实习亮点原话："在作业运行态时**并行创建 RayCluster Head 和 TrajStore 两条网络访问链路**，实现 TrajStore 公网 URL 的自动暴露与生命周期管理，使用户通过公网即可查看训练产生的轨迹数据"——这句话对应的 K8S 机制是什么？

### 2. 方法论（逐句拆解）

| 简历表述 | K8S 机制 |
|---|---|
| AgenticRL 训练作业（Kubernetes CRD 资源） | 训练作业 = 自定义 CRD（第 21 点），Operator 管理其生命周期 |
| RayCluster Head 链路 | RayCluster 的 head Service（`<cluster>-head-svc`）+ NodePort/LoadBalancer/Ingress → 用户访问 Ray dashboard（8265）/提交任务（8265 API）|
| TrajStore 链路 | 轨迹存储服务（agent-lighting LightningStore / TQ TrajStore）的 Service + LoadBalancer/Ingress → 公网 URL |
| 并行创建两条链路 | 作业创建时 Operator 同时 reconcile head Service 与 TrajStore Service（`createHeadSvc + createTrajStoreSvc` 并行），各自暴露独立端点 |
| 公网 URL 自动暴露 | Service type=LoadBalancer（云 LB）或 Ingress + TLS；自动分配公网 IP/域名 |
| 生命周期管理 | 作业删除时级联清理 Service/Ingress/LB；Finalizer 保证清理顺序；TTL/自动释放 |

**网络拓扑示意**：

```text
用户（任意位置）
  │ HTTPS
  ├──► Ingress/LB ──► RayCluster Head Service（8265 dashboard / 10001 任务提交）
  └──► Ingress/LB ──► TrajStore Service（4747 /v1/agl/*，公网查看轨迹）
                                        │
                              ┌─────────┴─────────┐
                              │ AgenticRL 训练作业 CR │
                              │   └─ RayCluster（head+workers）│
                              │   └─ TrajStore Deployment     │
                              └──────────────────────────┘
```

### 3. 具体数值样例

- 作业创建：Operator reconcile 顺序 = 建 head Service（ClusterIP，供 worker 内网）+ 建 TrajStore Service（`type: LoadBalancer`）→ 云平台分配公网 IP `1.2.3.4:4747` → 写入 CR status（`trajStoreUrl: http://1.2.3.4:4747`）→ 用户拿到 URL 即可看轨迹；
- 安全：Ingress 配 TLS + NetworkPolicy 只放行 4747/8265；公网只暴露 TrajStore 与 dashboard，训练内网（GPU 通信）不暴露；
- 生命周期：作业完成/删除 → Finalizer 依次删除 TrajStore Service（释放公网 IP）→ 删 RayCluster → 清 Finalizer——**公网 URL 与作业同生命周期**。

> 面试一句话总结：**实习的"双网络链路"= 作业 CR 的 Operator 并行创建两个 Service：RayCluster Head Service（dashboard/任务提交）与 TrajStore Service（轨迹数据），后者配 LoadBalancer/Ingress 自动分配公网 URL 并写回 CR status；作业删除时 Finalizer 级联清理、释放公网资源——这就是"公网访问轨迹数据"的完整 K8S 实现。**

---

# 五、自定义 CRD 训练作业（AgenticRL 作业的 Operator 化）

## 21. 为什么训练作业要用 CRD + Operator（而不是裸 Deployment/Job）

### 1. 现有问题

训练作业不就是"起几个 Pod 跑训练吗"？为什么值得做成 CRD？

### 2. 方法论（CRD/Operator 的优势，逐条对应训练痛点）

| 训练痛点 | 裸 Deployment/Job | 自定义 CRD + Operator |
|---|---|---|
| 生命周期状态机（Pending→Running→Succeeded/Failed） | 靠 Job 的 status（粗糙） | CR 的 `.status.phase` 自定义（含 checkpoint 轮次、评估结果） |
| 子资源多且联动（RayCluster + TrajStore + vLLM + 数据卷 + 网络链路） | 手动逐个 apply/删，易漏 | Operator reconcile 统一创建/更新/级联删除 |
| 长时运行 + 断点续训（训练 25 步 7 小时） | Job 重跑全部重来 | CR 记录 step/checkpoint 进度，重建续训 |
| 需要外部资源（GPU 配额、云盘、公网 LB、密钥） | 手动管理 | Operator 统一申请/回收 |
| 多实例/多租户（同时多个训练作业） | YAML 复制粘贴 | 每个 CR 一个隔离作业，配额/调度统一 |
| 训练与推理/环境解耦（AgenticRL 分离式） | 各组件分散难管 | 一个 CR 声明整条链（store/algo/runner/vllm） |
| 团队协作/审计 | 无版本概念 | CR 声明式 + git 化（GitOps），status/events 可审计 |

**本质**：训练作业是"**有状态、长时、多子资源、需要外部协调**"的领域资源——正是 Operator 模式的典型场景（KubeRay 的 RayJob 就是官方先例，但 AgenticRL 作业需要更多定制：TrajStore 链路、轨迹库、黑/白盒配置、评估阶段）。

### 3. 具体数值样例

- 裸 Job：训练到第 12/25 步时节点故障 → Job 重跑 → 25 步重来（数小时浪费）；
- CRD 作业：status 记录 `currentStep: 12, checkpoint: s3://.../step12` → Pod 重建后 Operator 从 checkpoint 续训（对应 verl `resume` 语义）；
- 3 个同时进行的训练作业 = 3 个 CR，Operator 统一管理各自的 RayCluster/TrajStore/GPU 配额。

> 面试一句话总结：**训练作业 = 有状态、长时、多子资源、需外部协调的领域资源，裸 Job 表达不了"状态机/续训/子资源联动/公网链路"；CRD 把训练作业变成一等资源（spec 期望 + status 现实），Operator 统一 reconcile——这就是实习"训练作业（Kubernetes CRD 资源）"的原因。**

---

## 22. 设计一个 AgenticRLTrainJob CRD（spec / status 设计）

### 1. 现有问题

如果我来设计训练作业 CRD，字段怎么定？参考 RayJob 但加上 AgenticRL 语义。

### 2. 方法论（完整字段设计）

```go
// 概念：apis/agenticrl/v1/agenticrltrainjob_types.go
type AgenticRLTrainJobSpec struct {
    // 训练算法与模型
    Algorithm   AlgorithmConfig   `json:"algorithm"`            // GRPO/PPO 超参、lr、rollout_n、batch...
    Model       ModelConfig       `json:"model"`                // base model 路径、LoRA rank、dtype
    Dataset     DatasetConfig     `json:"dataset"`              // train/val 数据源（S3/NFS/内置）

    // 分离式组件（对应第 19 点三件套）
    Store       StoreConfig       `json:"store"`                // backend(memory/mongo)、副本、持久化
    AlgorithmDeploy AlgorithmDeployConfig `json:"algorithmDeploy"` // GPU 数、vLLM 引擎配置
    Runner      RunnerConfig      `json:"runner"`               // 初始副本、min/max（HPA）、agent 类型（白盒/黑盒）
    Sandbox     SandboxConfig     `json:"sandbox"`              // 腾讯 E2B / 本地 WSL 沙箱配置

    // 训练运行参数
    Rounds      int               `json:"rounds"`               // 训练轮数（如 25）
    Resume      bool              `json:"resume,omitempty"`     // 断点续训
    StopAt      *metav1.Time      `json:"stopAt,omitempty"`     // 时间截止（对应 run_loop_opt 的 STOP_AT）
    EvalAfter   bool              `json:"evalAfter"`            // 训练完自动官方评估

    // 网络与安全
    ExposeTrajStore bool          `json:"exposeTrajStore"`      // 是否公网暴露轨迹（对应实习双链路）
    TrajStoreAuth   *SecretRef    `json:"trajStoreAuth,omitempty"` // 轨迹库访问鉴权
}

type AgenticRLTrainJobStatus struct {
    Phase        string   `json:"phase"`         // Pending|Creating|Rollout|Training|Evaluating|Succeeded|Failed
    CurrentStep  int      `json:"currentStep"`   // 已训练步数（续训依据）
    TotalSteps   int      `json:"totalSteps"`
    ReadyRunners int      `json:"readyRunners"`  // 就绪 runner 数
    HeadURL      string   `json:"headUrl,omitempty"`      // RayCluster Head 访问地址（链路 1）
    TrajStoreURL string   `json:"trajStoreUrl,omitempty"` // TrajStore 公网 URL（链路 2）
    Checkpoint   string   `json:"checkpoint,omitempty"`   // 最新权重路径
    Conditions   []metav1.Condition `json:"conditions,omitempty"` // Ready/Completed/Evicted...
    LastEval     *EvalResult `json:"lastEval,omitempty"`  // 最近一次评估（通过率 83.23% 等）
}
```

**设计要点**：
- **spec 全声明**（要什么），**status 全现实**（现在到哪了）——用户只看 status 就知道作业进展；
- 子资源列表（Operator 要管的）：RayCluster、TrajStore Deployment+Service(+LB)、vLLM Deployment、runner Deployment+HPA、PVC、Secret、NetworkPolicy、Ingress；
- 与 KubeRay 的关系：**RayCluster 部分直接内嵌 `rayClusterSpec`（复用 KubeRay CRD 做"集群"层），本 CRD 管"作业"层**——两层 Operator 协作（也可以用 `RayJob` 扩展而非自研）。

### 3. 具体数值样例

```yaml
apiVersion: agenticrl.example.io/v1
kind: AgenticRLTrainJob
metadata: {name: humanevalfix-spec-run1}
spec:
  algorithm: {type: GRPO, lr: 5e-6, rolloutN: 4, clipEpsilon: 0.2, klBeta: 0.01}
  model: {base: qwen/Qwen3-8B, loraRank: 32, speculative: {method: eagle3, draftModel: Qwen3-8B-speculator.eagle3}}
  dataset: {source: nfs://data/humanevalfix_train164.jsonl}
  store: {backend: mongo, replicas: 1}
  runner: {replicas: 12, min: 4, max: 32, agentType: whitebox, sandbox: tencent-e2b}
  rounds: 25
  resume: true
  stopAt: "2026-08-19T17:00:00Z"
  evalAfter: true
  exposeTrajStore: true
```

- 提交后 `kubectl get agenticrltrainjob humanevalfix-spec-run1 -o jsonpath='{.status}'` 看到 `phase: Training, currentStep: 12/25, trajStoreUrl: http://1.2.3.4:4747`；
- 训练完成 → `phase: Succeeded, lastEval: {passRate: 83.23, tasks: 161}`。

> 面试一句话总结：**训练作业 CRD = spec 声明算法/模型/数据/三件套配置/轮数/续训（期望）+ status 报告阶段/步数/就绪数/Head 与 TrajStore URL/评估结果（现实）；"作业层"CRD 可内嵌 KubeRay 的 rayClusterSpec 做"集群层"——两层 Operator 协作，把 AgenticRL 训练全链路变成声明式资源。**

---

## 23. 写一个最小 Operator（controller-runtime reconcile 骨架）

### 1. 现有问题

CRD 定义了，谁来干活？——写 Operator。最小实现长什么样？

### 2. 方法论（可运行的骨架，参照 KubeRay RayClusterReconciler）

```go
// controllers/agenticrltrainjob_controller.go（骨架）
package controllers

import (
    "context"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    agenticrlv1 "example.io/api/agenticrl/v1"
    rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

type AgenticRLTrainJobReconciler struct {
    client.Client
    Scheme *runtime.Scheme
}

// Reconcile：核心入口（KubeRay 同款骨架）
func (r *AgenticRLTrainJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    job := &agenticrlv1.AgenticRLTrainJob{}
    if err := r.Get(ctx, req.NamespacedName, job); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)   // 已删则忽略
    }

    // 1. 阶段状态机
    switch job.Status.Phase {
    case "", "Pending":
        return r.createSubResources(ctx, job)             // 建 RayCluster + TrajStore + runner...
    case "Rollout", "Training":
        return r.tickTraining(ctx, job)                   // 轮次推进/续训/检查 stopAt
    case "Evaluating":
        return r.checkEval(ctx, job)
    case "Succeeded", "Failed":
        return ctrl.Result{}, nil                         // 终态：静默
    }

    // 2. 子资源收敛（幂等：期望 vs 现实）
    //    - reconcileRayCluster（若不存在则创建，并设 OwnerReference）
    //    - reconcileTrajStore（Deployment + Service + 可选 LoadBalancer/Ingress）
    //    - reconcileRunnerHPA（min/max/指标）
    //    - reconcileExpose（head svc + trajstore svc，写 status.HeadURL/TrajStoreURL）

    // 3. 更新 status（写回 phase/currentStep/readyRunners/...）
    return ctrl.Result{RequeueAfter: 30 * time.Second}, nil // 定期 tick（配合事件驱动）
}
```

**配套**：
- **SetupWithManager**：`ctrl.NewControllerManagedBy(mgr).For(&agenticrlv1.AgenticRLTrainJob{}).Owns(&rayv1.RayCluster{}).Owns(&appsv1.Deployment{}).Complete(r)`——自动 watch 子资源变化触发 reconcile（`Owns` 实现"子资源变了父资源也收敛"）；
- **Finalizer**：`AddFinalizer`（删除时先清 TrajStore LB/Ingress、删 RayCluster、释放公网 IP）再移除 Finalizer；
- **RBAC**（controller 的 ClusterRole）：`rayclusters`、`deployments`、`services`、`horizontalpodautoscalers`、`ingresses`、`persistentvolumeclaims`、`secrets` 的 get/list/watch/create/update/patch/delete；
- **生成工具**：`kubebuilder` / `operator-sdk` 脚手架（`kubebuilder create api --group agenticrl --version v1 --kind AgenticRLTrainJob`），自动生成 CRD 清单与 RBAC。

### 3. 具体数值样例

- 首次 reconcile：建 RayCluster（KubeRay Operator 再收敛 head/worker）→ 建 TrajStore Deployment + Service → 建 runner Deployment + HPA → 更新 status `phase: Creating`；
- `stopAt` 到达：reconcile 检测时间 → 停止 rollout、进入 Evaluating → 评估完成写 `lastEval` → `phase: Succeeded`；
- 手动删 CR：Finalizer 清理（释放公网 LB）→ 移除 Finalizer → 级联删除所有子资源。

> 面试一句话总结：**最小训练 Operator = 标准 reconcile：按 phase 状态机推进（Pending→Rollout→Training→Evaluating→终态），幂等收敛子资源（RayCluster/TrajStore/runner/HPA），Owns 注册子资源触发，Finalizer 保证删除清理（先释放公网链路再删集群）——Kubebuilder 脚手架 + controller-runtime，与 KubeRay 同构。**

---

## 24. CRD 作业生命周期：从提交到完成的完整时序

### 1. 现有问题

一个 AgenticRL 训练作业从提交到完成的完整时序，把前面所有内容串起来。

### 2. 方法论（完整时序）

```text
① 用户 kubectl apply -f humanevalfix-spec-run1.yaml（AgenticRLTrainJob CR）
② apiserver：认证 → RBAC → ValidatingWebhook（校验字段）→ 写 etcd → watch 广播
③ AgenticRLTrainJob Operator：watch 到新 CR → reconcile
   ├─ phase: Pending → 建子资源（幂等）：
   │    ├─ RayCluster CR（KubeRay Operator 再收敛 head/worker Pod）
   │    ├─ TrajStore Deployment + Service（exposeTrajStore=true → LoadBalancer/Ingress）
   │    ├─ runner Deployment + HPA
   │    └─ PVC / Secret / NetworkPolicy
   └─ status: phase=Creating, headUrl/trajStoreUrl 填充
④ KubeRay Operator：watch RayCluster → 建 head/worker → head Service → 集群 ready
   （实习：并行创建 Head 与 TrajStore 两条链路，公网 URL 写回 status）
⑤ 训练循环（AgentLightningTrainer 在 Ray head 上）：
   rollout（runner 从 store 取任务）→ span 入库 → VERL 更新权重 → vLLM 热更新
   → status.currentStep 每步 +1（Operator 定期 tick 同步）
⑥ stopAt / rounds 达到 → phase=Evaluating → 官方评估 → status.lastEval
⑦ phase=Succeeded → 可选保留 TrajStore 数据（TTL）→ 删除作业 → Finalizer 清理
```

**关键点**：
- **两层 Operator 协作**：作业 Operator（管作业层）owns RayCluster CR（管集群层），各自 reconcile 各自职责；
- **状态机幂等**：任何一步失败，reconcile 重试（指数退避），从 status 恢复；
- **可观测**：`kubectl get agenticrltrainjob -w` 实时看 phase/step；Events 记录关键动作；
- **与 verl 训练脚本对应**：CR 的 `rounds/stopAt/resume` 对应 `run_loop_opt.sh` 的 `ROUNDS/STOP_AT/START_ROUND`；`speculative` 配置对应 `speculative_config` JSON。

### 3. 具体数值样例

- 作业"humanevalfix-spec-run1"：25 轮 GRPO + EAGLE-3 + 双机，7:11:40 完成——CR 上看到 `currentStep: 25/25, lastEval.passRate: 83.23%`；
- 中途节点故障：runner Pod 重建（Deployment 收敛），训练从 checkpoint 续（resume=true），CR 状态不丢；
- 删除作业：Finalizer 先释放 TrajStore 公网 IP → 删 RayCluster（KubeRay Finalizer 再清 head/worker）→ 清 CR——**两级 Finalizer 有序清理**。

> 面试一句话总结：**训练作业生命周期 = 提交 CR → 作业 Operator 建子资源（RayCluster/TrajStore/runner/HPA + 网络链路）→ KubeRay Operator 收敛集群 → 训练循环推进（status.currentStep 实时同步）→ 评估 → 终态；两层 Operator + 幂等状态机 + Finalizer 有序清理，让"7 小时的训练作业"变成可声明、可观测、可续训、可审计的 K8S 资源。**

---

# 六、面试问答与速查

## 25. 高频追问速答

**Q1：Pod 的生命周期与探针？**
Pending→Running→Succeeded/Failed；CrashLoopBackOff（反复崩溃）；探针：liveness（存活，失败重启）、readiness（就绪，失败摘除 Service）、startup（启动保护，慢启动容器）。

**Q2：etcd 怎么保证一致性？**
Raft 共识：多数派（2N+1 节点容忍 N 故障）写入；线性一致性读（可选）；apiserver 是唯一写入口，天然串行化。

**Q3：Deployment 滚动更新原理？**
新旧两个 RS：maxSurge（额外起的）、maxUnavailable（允许不可用的），逐个替换；`kubectl rollout undo` 回滚到历史 revision。

**Q4：Service 的负载均衡怎么实现的？**
kube-proxy 两种模式：iptables（随机 DNAT，规则多时性能差）、ipvs（内核态 LB，支持轮询等算法）；EndpointSlice 维护后端 Pod 列表。

**Q5：Operator 和 Helm 的区别？**
Helm 是"打包/渲染模板"（安装时一次性）；Operator 是"运行时持续收敛"（CR 变化驱动 reconcile）；两者互补（Helm 装 Operator，Operator 管应用）。

**Q6：为什么训练用 KubeRay 而不是原生 K8S Job？**
训练需要 Ray 的分布式运行时（GCS/调度/actor 资源组/对象存储），K8S Job 只给"进程组"；KubeRay 把两者桥接：K8S 管生命周期/网络/GPU，Ray 管分布式执行/容错/扩缩容。

**Q7：CRD 的 status 子资源有什么用？**
spec 与 status 分离：用户只写 spec（期望），Operator 写 status（现实）；避免用户误改 status、支持条件等待（watch status 变化）。

**Q8：Finalizer 卡住怎么办？**
`kubectl patch <res> -p '{"metadata":{"finalizers":[]}}' --type=merge` 强制移除（危险，生产慎用）；正常应等 Operator 清理完成。

**Q9：GPU 在 K8S 里怎么分配？**
设备插件（DaemonSet）上报扩展资源 `nvidia.com/gpu`；Pod `limits` 声明；scheduler 按节点可用数分配；多卡用 `nvidia.com/gpu: 8` + 显存/算力调度（MIG/时间片可选）。

**Q10：分离式训练中"环境/推理/训练解耦"在 K8S 怎么体现？**
三个独立工作负载（runner=环境+agent、vllm=推理、algo=训练），各自 Deployment 独立扩缩容/升级/故障域，只通过 Service 通信——解耦是架构属性，K8S 把它变成部署属性。

## 26. 面试一句话总结（背诵版）

- **K8S 本质**："声明式 + 控制器收敛"——用户写"要什么"，apiserver 存期望，控制器让现实向期望收敛；
- **架构**：控制面（apiserver/etcd/scheduler/controller-manager）+ 数据面（kubelet/kube-proxy/CRI/CNI/CSI）；
- **工作负载**：Deployment（无状态）/StatefulSet（有状态）/DaemonSet（系统）/Job（批任务）；
- **网络**：CNI 发 IP、kube-proxy 做 Service、Ingress 管 7 层、NetworkPolicy 管隔离；
- **扩展**：CRD 定义资源、Operator（reconcile）实现领域逻辑、Webhook 拦截注入；
- **KubeRay**：RayCluster/RayJob/RayService 四 CRD + Operator，声明式管 Ray 集群；
- **分离式 agent-lighting 在 K8S**：store/algo/runner 各一个工作负载，两个 HTTP 面（4747/8000）用 Service 打通，runner HPA 弹性，TrajStore 可 LoadBalancer 公网暴露；
- **训练作业 CRD**：spec 期望 + status 现实，两层 Operator（作业层 + KubeRay 集群层）协作，Finalizer 有序清理，断点续训。

---

# 附：速查表

## K8S 资源速查

| 资源 | kind | 一句话 |
|---|---|---|
| 最小调度单元 | Pod | 共享网络/存储的容器组，临时 |
| 无状态应用 | Deployment→RS→Pod | 滚动更新/回滚 |
| 有状态应用 | StatefulSet | 稳定身份/稳定存储/有序部署 |
| 系统组件 | DaemonSet | 每节点一个 |
| 批任务 | Job/CronJob | 跑完即结束/定时 |
| 服务入口 | Service/Ingress | 4 层 LB / 7 层路由 |
| 配置/密钥 | ConfigMap/Secret | 键值注入 |
| 存储 | PV/PVC/StorageClass | 持久化 + 动态供给 |
| 身份/授权 | SA/Role/ClusterRole/Binding | RBAC |
| 弹性 | HPA | 按指标扩缩容 |
| 自定义资源 | CRD/CR | 定义/实例 |
| 领域逻辑 | Operator | CRD+reconcile |
| 拦截 | Mutating/ValidatingWebhook | 改/拒请求 |

## KubeRay 速查

| CRD | 用途 | 关键字段 |
|---|---|---|
| RayCluster | 声明 Ray 集群 | headGroupSpec / workerGroupSpecs / enableInTreeAutoscaling |
| RayJob | 跑任务 | rayClusterSpec + entrypoint + shutdownAfterJobFinishes |
| RayService | Serve 应用 | rayClusterConfig + serveConfigV2 |
| RayCronJob | 定时任务 | schedule + rayJobSpec |

## 简历亮点 ↔ 本文章节映射

| 实习亮点 | 对应章节 |
|---|---|
| AgenticRL 训练作业（Kubernetes CRD 资源） | 第 7、21~24 点（CRD/Operator/作业设计） |
| 基于 Ray 分布式框架的管控面 | 第 13~16 点（KubeRay）+ `Ray.md` |
| 并行创建 RayCluster Head 和 TrajStore 两条网络访问链路 | 第 20 点（双链路） |
| TrajStore 公网 URL 自动暴露与生命周期管理 | 第 19~20 点（Service/LB/Ingress/Finalizer） |
| 分离式 RL（agent 任意位置部署、仅填 TrajStore 地址） | 第 17~19 点（三件套 K8S 化） |
| 昇腾 NPU 单/双机全链路 | `Communication.md`（HCCL/HCCS/RoCE）+ `Parallel.md` |
