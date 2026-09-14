# KubeRay 实操手册：从三台服务器到大规模训练集群

> **这篇怎么用**：不讲概念定义，按"**准备机器 → 装 Operator → 起集群 → 跑训练 → 起服务 → 排障**"的顺序走一遍，每步都是能直接粘贴的命令和真实 YAML。YAML 全部来自本地 KubeRay 仓（`kuberay/ray-operator/config/samples/`），字段逐条解释。
>
> 前置阅读：`Ray.md`（GCS/raylet/Plasma 的机制）、`K8S-KubeRay.md` 里已删掉的纯概念部分可参考 K8s 官方文档。本文只保留"要动手的部分"。
>
> **素材来源**：`kuberay` 全仓（`docs/deploy/installation.md`、`ray-operator/config/samples/*.yaml`、`helm-chart/`）；`uniagent-lighting/docs/deployment.md`；`agent-lightning` v0.3.1 源码（`agentlightning/cli/store.py`、`agentlightning/store/client_server.py`）。

---

## 0. 先看全景：我们要搭出什么

```
                    ┌─────────────────────────────────────────┐
                    │  K8s Control Plane (node-1)             │
                    │  apiserver / etcd / scheduler / cm      │
                    │  + KubeRay Operator（watch CRD 并建 Pod）│
                    └───────────────┬─────────────────────────┘
                                    │ 调度
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ node-2 (GPU)  │          │ node-3 (GPU)  │          │ node-N (GPU)  │
│ Ray Head Pod  │          │ Ray Worker Pod│          │ Ray Worker Pod│
│ GCS 6379      │◄─ Ray ──►│ :10001        │◄────────►│ :10001        │
│ Dash 8265     │  gRPC    │               │          │               │
│ Client 10001  │          │ GPU × 8       │          │ GPU × 8       │
└───────────────┘          └───────────────┘          └───────────────┘
```

**四个角色**（后面每一步都在填这四块）：

| 角色 | 是什么 | 谁创建 |
|---|---|---|
| **KubeRay Operator** | 一个 Deployment，watch `RayCluster`/`RayJob`/`RayService` CRD，按 spec 建/删 Pod | 你手动装（第 2 节） |
| **RayCluster** | 一组 Pod = 1 个 head + N 个 worker group | Operator 按你的 YAML 建（第 3 节） |
| **RayJob** | 一次性任务：建集群 → 跑 entrypoint → 销毁（或保留） | Operator（第 5 节） |
| **RayService** | 常驻在线服务：Ray Serve + 零停机升级 | Operator（第 5 节） |

---

## 1. 环境准备：三台服务器

### 1.1 集群拓扑（本文的假设）

| 节点 | 角色 | 规格 | 装什么 |
|---|---|---|---|
| **node-1** | control-plane + Operator | 8C/32G，无 GPU | kubelet、containerd、KubeRay Operator |
| **node-2** | GPU worker + **Ray head 落点** | 8×A100 80G | kubelet、containerd、nvidia 驱动 + toolkit |
| **node-3** | GPU worker | 8×A100 80G | 同上 |

**为什么 Ray head 落在 GPU 节点**：head 也要跑 driver 和部分 actor，而且 `ray-cluster.verl.yaml` 这类样例就是把 **4 张 GPU 直接给 head**（后面第 6 节会看到）。当然也可以把 head 放 CPU 节点、用 `num-cpus: "0"` 禁止业务调度上去（autoscaler 样例就是这么做的，见第 7.1 节）。

### 1.2 前置条件清单

```bash
# ① K8s 版本：官方要求至少 1.23
#   kuberay/docs/deploy/installation.md:3
#   "Make sure your Kubernetes cluster and Kubectl are both at version at least 1.23."
kubectl version --short

# ② 关闭 swap（kubelet 硬要求）
sudo swapoff -a
sudo sed -i '/ swap / s/^/#/' /etc/fstab

# ③ 内核模块与转发参数
sudo modprobe br_netfilter
cat <<'EOF' | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
cat <<'EOF' | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sudo sysctl --system

# ④ 容器运行时（containerd）+ cgroup driver 与 kubelet 对齐
sudo containerd config default | sudo tee /etc/containerd/config.toml
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl restart containerd

# ⑤ GPU 节点：驱动 + container toolkit + device plugin
nvidia-smi                      # 先确认驱动正常
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
# device plugin 用 DaemonSet 部署（让 K8s 认识 nvidia.com/gpu 资源）
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml

# ⑥ 验证 GPU 已成为可调度资源（关键一步）
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
# 期望：GPU 节点各显示 8，control-plane 显示 <none>
```

**常见坑**：`nvidia.com/gpu` 显示 `<none>` 说明 device plugin 没起来（`kubectl -n kube-system logs <plugin-pod>`）；显示 `0` 通常是 containerd 的 `SystemdCgroup` 或 toolkit 配置没生效。

### 1.3 没有 GPU 机器？用 kind 单机验证（可选）

只验证 CRD/Operator 逻辑时用 kind 就够了（`ray-cluster.complete.yaml` 的注释明确说它的资源配置就是为 "resource-constrained local Kubernetes testing environments such as **KinD and minikube**" 设计的，见该文件 54-55 行）：

```bash
kind create cluster --name kuberay-test
kubectl cluster-info --context kind-kuberay-test
```

---

## 2. 安装 KubeRay Operator

> ⚠️ **版本纠正（很重要）**：本地 `kuberay` 仓的 `helm-chart/*/Chart.yaml` 和 `docs/deploy/installation.md` 里写的都是 **`1.1.0`**，但**代码里的 feature gate 已经到 `v1.7`**，上游实际已发布 **v1.7.0**。**安装请用 `--version 1.7.0` 或 `?ref=v1.7.0`，不要照抄本地的 1.1.0。**
>
> 另外，官方最新推荐把 Operator 装进**专职的 `ray-system` namespace**（原文理由：*"isolate the operator's service account from workload pods"*），而不是默认 namespace。下面命令里的 `-n kuberay-system` 请按你实际安装的 namespace 调整。

**三种方式，任选一种**（命令来自 `kuberay/docs/deploy/installation.md`）。

### 2.1 Helm（官方推荐）

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# 一条命令装 CRDs + Operator
helm install kuberay-operator kuberay/kuberay-operator

# 需要自定义时先看默认值
helm show values kuberay/kuberay-operator > my-values.yaml
helm install kuberay-operator kuberay/kuberay-operator -f my-values.yaml
```

本地 helm chart 位置：`kuberay/helm-chart/kuberay-operator/`（含 `values.yaml`、`templates/`、`README.md`）；另外还有 `kuberay-apiserver`（REST API）和 `ray-cluster`（**用 Helm 直接起一个 RayCluster**，见第 3.3 节）。

### 2.2 Kustomize / kubectl（适合锁版本或离线）

```bash
# 稳定版：指定 tag
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.1.0&timeout=90s"

# 或者用本地仓（离线/改源码时）
kubectl create -k kuberay/ray-operator/config/default

# 夜间版
export KUBERAY_VERSION=master
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=${KUBERAY_VERSION}&timeout=90s"
```

### 2.3 从源码构建部署（要改 Operator 时）

```bash
cd kuberay/ray-operator
make deploy IMG=quay.io/kuberay/operator:nightly   # 生成 image 并部署
```

### 2.4 验证安装

```bash
# ① Operator Pod 就绪
kubectl get pods -n kuberay-system
# NAME                                READY   STATUS
# kuberay-operator-xxx                1/1     Running

# ② CRD 已注册（应该看到 4 个）
kubectl get crd | grep ray.io
# rayclusters.ray.io
# rayjobs.ray.io
# rayservices.ray.io
# raycronjobs.ray.io          ← 定时任务

# ③ Operator 日志（排障第一步）
kubectl -n kuberay-system logs deploy/kuberay-operator -f
```

**面试常问**：Operator 装完之后它在干什么？——它是一个 **controller-runtime 的 reconcile loop**：watch `RayCluster` 的增删改，按 spec 创建/删除 head Pod、worker Pod、Service、ConfigMap，并把实际状态写回 `status`。所以"改 YAML 就能改变集群"这件事的实现在 Operator 里，不在 K8s 本身。

---

## 3. 起第一个 RayCluster

### 3.1 完整 YAML 逐字段讲（`ray-cluster.complete.yaml`，120 行）

```yaml
apiVersion: ray.io/v1                 # KubeRay 的 API group（不是 apps/v1）
kind: RayCluster
metadata:
  name: raycluster-complete
spec:
  rayVersion: "2.52.0"                # ★ 必须与镜像里的 Ray 版本一致
  headGroupSpec:
    serviceType: ClusterIP            # head 的 Service 类型：ClusterIP / NodePort / LoadBalancer
    rayStartParams:
      dashboard-host: "0.0.0.0"       # ★ 不设 0.0.0.0，dashboard 在 Pod 外访问不到
    template:                         # 这里往下全是标准 PodTemplateSpec
      metadata:
        labels: {}
        # 注意：自定义 label 不要以 `raycluster` 开头，会和 Operator 的 label 冲突
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.52.0
          ports:
          - {containerPort: 6379,  name: gcs}        # GCS：Ray 的全局控制面
          - {containerPort: 8265,  name: dashboard}  # Dashboard UI
          - {containerPort: 10001, name: client}     # Ray Client / Job 入口
          volumeMounts:
          - {mountPath: /tmp/ray, name: ray-logs}
          resources:
            limits:   {cpu: "1", memory: "5Gi"}
            requests: {cpu: "1", memory: "2Gi"}
        volumes:
        - name: ray-logs
          emptyDir: {}
  workerGroupSpecs:
  - replicas: 1                       # 当前副本数
    minReplicas: 1                    # ★ 开 autoscaler 时的下界
    maxReplicas: 10                   # ★ 上界
    groupName: small-group            # ★ 组名：Pod 名 = <cluster>-worker-<group>-<hash>
    scaleStrategy:                    # 缩容时优先删哪些（可选）
      workersToDelete:
      - raycluster-complete-worker-small-group-bdtwh
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.52.0
          volumeMounts:
          - {mountPath: /tmp/ray, name: ray-logs}
          resources:
            limits:   {cpu: "1", memory: "1Gi"}
            requests: {cpu: "1", memory: "1Gi"}
        volumes:
        - name: ray-logs
          emptyDir: {}
```

**三个字段级要点**（都写在样例注释里）：

1. **`rayVersion` 与镜像 tag 必须一致**——head 和所有 worker 的 Ray 版本不同会连不上 GCS。
2. **`dashboard-host: "0.0.0.0"`**——默认只绑 localhost，不设置的话 `kubectl port-forward` 也连不上。
3. **不要用 `raycluster` 开头的自定义 label**——会和 Operator 的 label 冲突（注释原文）。

### 3.2 生产环境的资源怎么配（样例注释里的三条建议，很值得背）

`ray-cluster.complete.yaml` 的注释反复强调（第 41-46、93-98 行）：

> - *"It is better to use **a few large Ray pod** than many small ones."*
> - *"For production, it is ideal to **size each Ray pod to take up the entire Kubernetes node** on which it is scheduled."*
> - *"For production use-cases, we recommend specifying **integer CPU requests and limits**. We also recommend setting **requests equal to limits** for both CPU and memory."*
> - *"For production use-cases, we recommend allocating **at least 8Gb memory for each Ray container**."*

**这条"少而大"的建议背后是 Ray 的架构**：Ray 的调度是**节点级**的——一个 worker Pod 就是 Ray 眼里的一个 node。如果 Pod 切得太碎（比如 1 CPU/Pod），Ray 会看到几百个"小节点"，调度开销和对象传输的跨节点概率都会上升。**所以生产上一个 GPU 节点 = 一个 Ray Pod 最省事。**

### 3.3 部署与验证

```bash
# 部署
kubectl apply -f kuberay/ray-operator/config/samples/ray-cluster.complete.yaml

# ① Pod 起来了吗（head + worker）
kubectl get pods -l ray.io/cluster=raycluster-complete
# raycluster-complete-head-xxxxx              1/1  Running
# raycluster-complete-worker-small-group-yyyy 1/1  Running

# ② 看 RayCluster 的 status（Operator 写回来的实际状态）
kubectl get raycluster raycluster-complete -o yaml | sed -n '/^status:/,$p'
# 关注 state: ready、以及 head.serviceName、endpoints

# ③ 看 Operator 是否给 head 自动建了 Service
kubectl get svc -l ray.io/cluster=raycluster-complete

# ④ 进 head 容器看 Ray 自己的视角
kubectl exec -it $(kubectl get pod -l ray.io/node-type=head -o name | head -1) -- bash
ray status          # 节点数、资源总量（CPU/GPU/内存）、autoscaler 状态
ray list nodes      # Ray 眼里的 node（= 你的 Pod）
exit

# ⑤ Dashboard（本地浏览器）
kubectl port-forward svc/raycluster-complete-head-svc 8265:8265
# 打开 http://localhost:8265
```

**`ray status` 是排障第一现场**：它同时显示"K8s 期望的副本数"和"Ray 实际看到的资源"。如果两者不一致（比如 Pod Running 但 `ray status` 里没有该节点），基本是 `ray start` 失败或 GCS 连不上，去 `kubectl logs` 看。

---

## 4. 提交训练任务：RayJob 的四种 submissionMode + 复用已有集群

> **纠正一个常见说法**：RayJob 不是"三种模式"，而是 **4 种 `submissionMode`**（`apis/ray/v1/rayjob_types.go`）：**`K8sJobMode`（默认）/ `HTTPMode` / `InteractiveMode` / `SidecarMode`**；而"提交到已有集群"（`clusterSelector`）是**第 5 条路径**，不是 submissionMode。
>
> ⚠️ 另外注意：**`clusterSelector` 模式不支持 gang scheduling**（官方 v1.7 说明）——想用 gang 就得让 RayJob 自己建集群。

手工 `kubectl apply` 一个 RayCluster 只适合调试。**跑生产任务应该用 `RayJob`**——它把"建集群 → 跑 entrypoint → 回收"打包成一个资源。

### 4.1 模式一：K8s Job 模式（最常用，集群随任务生灭）

```yaml
# 基于 ray-job.sample.yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: rayjob-sample
spec:
  entrypoint: |                     # ★ 真正跑的命令，在 head 上执行
    python -c "
    import ray; ray.init()
    print(ray.get([f.remote() for f in [ray.remote(lambda: i) for i in range(4)]]))
    "
  # 三种前置行为（三选一）：
  #   - 不写 rayClusterSpec：KubeRay 自动建一个默认 RayCluster
  #   - 写 rayClusterSpec：用你自定义的集群（见下面）
  #   - 写 clusterSelector：复用已有集群（模式三）
  rayClusterSpec:                    # 自定义集群（字段与 RayCluster.spec 完全相同）
    rayVersion: "2.52.0"
    headGroupSpec:
      rayStartParams: {dashboard-host: "0.0.0.0"}
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.52.0
            resources:
              limits: {cpu: "2", memory: "4Gi"}
              requests: {cpu: "2", memory: "4Gi"}
    workerGroupSpecs:
    - replicas: 1
      minReplicas: 1
      maxReplicas: 5
      groupName: gpu-group
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.52.0
            resources:
              limits: {cpu: "4", memory: "8Gi", nvidia.com/gpu: "1"}   # ★ GPU 就这样申请
              requests: {cpu: "4", memory: "8Gi", nvidia.com/gpu: "1"}
  shutdownAfterJobFinishes: true     # ★ 任务结束就删集群（不设则集群保留，方便看现场）
  ttlSecondsAfterFinished: 600       # 结束 10 分钟后清理，便于捞日志
  # 失败重试与删除规则见 ray-job.deletion-rules.yaml
```

```bash
kubectl apply -f my-rayjob.yaml

# 看任务状态机
kubectl get rayjob rayjob-sample -o jsonpath='{.status.jobStatus}{"\n"}'
# PENDING → RUNNING → SUCCEEDED / FAILED

# 看 entrypoint 的 stdout
kubectl logs -l ray.io/cluster=... -c ray-head --tail=100
# 或者直接
kubectl get rayjob rayjob-sample -o jsonpath='{.status.jobDeploymentStatus}'
```

### 4.2 模式二：Interactive 模式（交互式调试用）

```yaml
# ray-job.interactive-mode.yaml 的思路
spec:
  submissionMode: Interactive      # ★ 不把 entrypoint 交给 K8s Job，而是提交给已有的 Ray Job API
  entrypoint: "python train.py"
  shutdownAfterJobFinishes: false  # 任务结束保留集群，可以继续 exec 进去玩
```

**`Interactive` 与默认（`K8sJob` 模式）的区别**：默认模式把 entrypoint 包成一个 K8s Job 来跑（生命周期由 K8s 管）；`Interactive` 模式把 entrypoint 提交给 Ray 自己的 Job Submission API（**集群里可以同时提交多个 job**）。调试训练脚本时用后者更方便。

### 4.3 模式三：复用已有 RayCluster（长期集群 + 多个任务）

```yaml
# ray-job.use-existing-raycluster.yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: my-job
spec:
  clusterSelector:                 # ★ 用 label 选已有集群，不新建
    ray.io/cluster: my-long-lived-cluster
  entrypoint: "python train.py"
  submissionMode: Interactive
```

**什么时候用**：集群启动慢（要拉镜像、装依赖），但任务频繁提交——保持一个常驻 `RayCluster`，用 `RayJob` 往上压任务。**代价**是任务之间会争抢资源，需要靠 Ray 的 `num_gpus`/自定义资源隔开。

### 4.4 顺带：RayCronJob（定时任务）

```yaml
# ray-cronjob.sample.yaml / ray-cronjob-timezone.sample.yaml
apiVersion: ray.io/v1
kind: RayCronJob
metadata: {name: nightly-eval}
spec:
  schedule: "0 2 * * *"
  timeZone: "Asia/Shanghai"
  jobTemplate: {spec: {...}}       # 内嵌一个 RayJob spec
```

**典型用途**：每天凌晨跑一次评测/数据生成——不需要人守着建集群。

---

## 5. 跑真实训练任务：verl on KubeRay

### 5.1 一个真实的 verl 样例（`ray-cluster.verl.yaml`，全文 29 行）

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: verl-cluster
spec:
  rayVersion: "2.43.0"
  headGroupSpec:
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-head
          image: hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0
          resources:
            limits:
              cpu: "48"
              memory: "192G"
              nvidia.com/gpu: "4"          # ★ 4 张 GPU 直接给 head
            requests:
              cpu: "36"
              memory: "144G"
              nvidia.com/gpu: "4"
          ports:
          - {containerPort: 6379,  name: gcs-server}
          - {containerPort: 8265,  name: dashboard}
          - {containerPort: 10001, name: client}
```

**三个值得注意的点**：

1. **`requests` 与 `limits` 不相等**（36/48 CPU、144G/192G）——这里有个**必须知道的确切语义**（KubeRay 官方 `config.md`）：**KubeRay 取的是容器的 `limits`；如果没设 limit，才回落到 CPU 的 request；CPU 会向上取整；而内存和 GPU 的 request 被完全忽略**。所以
   - 这份样例的实际含义是"**Ray 看到 48C/192G，但 K8s 只按 36C/144G 来调度**"——这是一种**超卖写法**，Ray 以为自己有更多资源；
   - **结论：内存和 GPU 的 `requests` 必须等于 `limits`**（否则写了也没用），CPU 可以留弹性但要清楚 Ray 按 limit 算。这跟 `complete.yaml` 里"requests = limits"的建议是一致的方向。
2. **镜像名就是环境清单**：`ngc-th2.6.0`（NGC PyTorch 2.6 + CUDA 12.6）、`vllm0.8.4`、`flashinfer0.2.2`、`cxx11abi0`——**训练镜像的 tag 必须把框架版本写全**，这是复现的前提。
3. **只有 head 没有 workerGroupSpecs**——单机 4 卡的场景，Ray 的 head 自己也是可调度节点。

### 5.2 规模放大：从 1 机 4 卡到多机多卡

KubeRay 样例里有一条极端的规模参考可直接引用：**`ray-cluster.tpu-v6e-256-multihost.yaml`**（256 主机多机 TPU），另外还有 `tpu-v4-multihost`、`tpu-v6e-16-multihost` 等。**多机 TPU/GPU 训练的关键是"一个 Pod 内的多卡 + 跨 Pod 的集合通信"**，需要：

```yaml
workerGroupSpecs:
- groupName: gpu-workers
  replicas: 8                    # 8 个 Pod × 8 卡 = 64 卡
  minReplicas: 8
  maxReplicas: 8                 # ★ 训练任务不设弹性（gang scheduling 要求整体起来）
  template:
    spec:
      # ① 一个 Pod 独占整台机器
      affinity:
        podAntiAffinity:         # 同一个 group 的 Pod 不要挤在一台机器
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels: {ray.io/group: gpu-workers}
            topologyKey: kubernetes.io/hostname
      # ② 共享内存：NCCL 走 shm，默认 64MB 远远不够
      volumes:
      - name: dshm
        emptyDir: {medium: Memory, sizeLimit: 64Gi}
      containers:
      - name: ray-worker
        image: <你的训练镜像>
        volumeMounts: [{mountPath: /dev/shm, name: dshm}]
        resources:
          limits: {nvidia.com/gpu: "8", cpu: "96", memory: "1Ti"}
          requests: {nvidia.com/gpu: "8", cpu: "96", memory: "1Ti"}
        env:
        - {name: NCCL_IB_DISABLE, value: "0"}            # 有 RDMA 就别关
        - {name: NCCL_SOCKET_IFNAME, value: "eth0"}      # 指定通信网卡（多网卡时必须）
        - {name: RAY_gcs_rpc_server_reconnect_timeout_s, value: "300"}
```

**五个必须配的东西**（缺一个都可能跑不起来）：

| 配置 | 为什么 |
|---|---|
| `emptyDir{medium: Memory}` 挂 `/dev/shm` | NCCL/多进程共享内存；默认 64MB 会导致 `Bus error` |
| `podAntiAffinity` + `hostname` | 一个 Pod 独占一台机（否则两个 8 卡 Pod 挤一台，调度不上） |
| `NCCL_SOCKET_IFNAME` | 多网卡机器上 NCCL 可能选错网卡，跨机带宽暴跌 |
| `memory` 给足（1Ti 级） | Ray object store 默认吃掉 30% 内存，训练进程还要用 |
| **训练任务 `maxReplicas = replicas`** | 训练不能"缩到一半"——gate scheduling 要求整组就绪；弹性只适合无状态的推理/数据任务 |

### 5.3 GPU 共享与 gang scheduling（大规模训练的核心难题）

**问题**：8 个 worker Pod，每个要 8 张卡。如果集群只剩 7 台空闲机器，K8s 默认调度器会让 7 个 Pod 起来、1 个 Pending——**已经起来的 7 个 Pod 在空转等你，集群算力被死锁**。这就是需要 **gang scheduling（成组调度：要么全起，要么全等）** 的原因。

KubeRay 样例里已经备好了四种调度器集成：

| 样例文件 | 调度器 | 特点 |
|---|---|---|
| `ray-cluster.volcano-scheduler.yaml` / `...-queue.yaml` | **Volcano** | 国内最常用，支持 queue/quota/gang |
| `ray-cluster.kai-scheduler.yaml` / `ray-cluster.kai-gpu-sharing.yaml` | **KAI Scheduler** | 支持 GPU 共享（一张卡切给多个 Pod） |
| `ray-cluster.yunikorn-scheduler.yaml` | **YuniKorn** | Apache 项目，队列与公平调度 |
| `ray-cluster.scheduler-plugins.yaml` | **K8s scheduler-plugins**（coscheduling） | 官方插件方案，改动最小 |

用法（以 Volcano 为例）：

```yaml
spec:
  headGroupSpec:
    template:
      spec:
        schedulerName: volcano          # ★ 关键：指定调度器
        containers: [...]
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 8
    minReplicas: 8
    maxReplicas: 8
    template:
      spec:
        schedulerName: volcano
        # Volcano 的 gang 语义靠 PodGroup；KubeRay 会按 group 自动生成
```

**面试表述**："多机训练必须 gang scheduling——8 个 Pod 要么全起要么全等，否则先起的 Pod 会占着 GPU 死等，整个集群被一个任务锁死。KubeRay 支持 Volcano / KAI / YuniKorn / scheduler-plugins 四种；GPU 切分用 KAI 的 sharing 样例。"

---

## 6. 关键特性逐个讲

### 6.1 Autoscaler（弹性伸缩）

```yaml
# ray-cluster.autoscaler.yaml 的关键字段
spec:
  enableInTreeAutoscaling: true       # ★ 开了才会在 head Pod 里注入 autoscaler sidecar
  autoscalerOptions:
    upscalingMode: Default            # Conservative（限速，pending Pod 数 ≤ 集群规模）/ Default / Aggressive（同 Default）
    idleTimeoutSeconds: 60            # ★ 空闲多久缩掉一个 worker
    imagePullPolicy: IfNotPresent
    env:
    - {name: AUTOSCALER_UPDATE_INTERVAL_S, value: "5"}   # 默认 5s 检查一次
    resources: {limits: {cpu: 500m, memory: 512Mi}, requests: {cpu: 500m, memory: 512Mi}}
  headGroupSpec:
    rayStartParams:
      num-cpus: "0"                   # ★ head 不接业务任务（否则训练会跑到 head 上）
  workerGroupSpecs:
  - replicas: 0                       # ★ 可以从 0 开始（冷启动按需拉起）
    minReplicas: 0
    maxReplicas: 10
```

**机制**：autoscaler sidecar 跑在 head Pod 里，它看 Ray 层面的 resource demand（`ray status` 里的 "Demands"），再通过 K8s API 改 `RayCluster.spec.workerGroupSpecs[].replicas`。**所以扩容的本质是"改 CR"，再由 Operator 建 Pod**——两级 reconcile 协作。

**四个关键点**：

1. **`rayStartParams: {num-cpus: "0"}`** 是标配——不让 head 抢业务任务；
2. **`idleTimeoutSeconds`** 决定缩容激进度（默认 60s），太小会抖动；
3. **`replicas: 0` + `minReplicas: 0`** 支持"从零拉起"，省钱但要等 Pod 启动（拉镜像可能几分钟）；
4. **训练任务不要开 autoscaler**（弹性只适合无状态的推理/数据/rollout 任务）——理由同 gang scheduling。

### 6.2 GCS 高可用与容错（这块正在演进，样例里有明确的"废弃"标记）

样例目录里同时存在三份相关文件，正好展示演进：

| 样例 | 含义 |
|---|---|
| `ray-cluster.external-redis.yaml` | 老方案：head 连**外部 Redis** 做 GCS 容错（外部 Redis 不随集群销毁，重建集群可恢复） |
| `ray-cluster.persistent-redis.yaml` / `persistent-redis-sidecar.yaml` | 折中：Redis 用 PVC 持久化（保留状态但不占外部资源） |
| `ray-cluster.embedded-gcs-ft.yaml` | 新方向：**内嵌 GCS 容错**（不需要 Redis） |
| **`ray-cluster.deprecate-gcs-ft.yaml`** | **废弃声明**——这个文件名本身就是结论：基于 Redis 的 GCS FT 已不再推荐 |

**★ 核心机制：head 挂掉后 worker 还能不能跑？——能，靠一个环境变量。**

KubeRay 在开 FT 时**只给 worker 注入 `RAY_gcs_rpc_server_reconnect_timeout_s=600`（head 保持默认 60s）**。源码注释原文：

> *"By default, the value is 60s… Typically, the new GCS server will be available in 120 seconds, so we **set the timeout to 600s to avoid the worker nodes crashing**."*

**不开 FT 的后果**（官方文档原文）：*"the worker Pods are perceived as **'unknown workers'** by the new head Pod"* —— 然后被全部干掉。

**所以这道题的正确答案是**：

> **head 重启后，worker 有 600 秒的重连窗口**（比新 GCS 起来的 ~120 秒更宽），窗口内重连上就继续跑；**不开 FT 则 worker 会被新 head 当成 "unknown workers" 清掉**，整个集群等于重建。这也是为什么长跑训练任务要理解这个字段——**它决定了"head 崩一次"是"抖动"还是"全灭"**。

**新方案（v1.7 + Ray 2.57，alpha）**：`gcsFaultToleranceOptions.backend: rocksdb`（feature gate `GCSFaultToleranceEmbeddedStorage`），用 PVC `{cluster}-gcs-pvc` 挂 `/data/gcs`，单写者，可设 `Retain` 或自带 `claimName`——**这才是 `deprecate-gcs-ft.yaml` 这个文件名背后的演进方向**。

**面试表述**："GCS 容错经历了两代：老方案外挂 Redis 存 GCS 状态（现在已标记废弃），新方案是 `backend: rocksdb` 的内嵌存储。但真正值得记的是那个环境变量——**KubeRay 给 worker 注入 600 秒重连超时（head 只有 60 秒默认值）**，所以 head 重启时 worker 有足够窗口重连而不被杀；**不开 FT 的话，新 head 会把老 worker 当成 'unknown workers' 全部清掉**。"

### 6.3 监控与可观测

| 样例 | 干什么 |
|---|---|
| `ray-cluster.embed-grafana.yaml` | 在 head Pod 里**内嵌 Grafana**，开箱能看指标 |
| `ray-cluster.fluentbit.yaml` | 用 **Fluent Bit** sidecar 收集日志，推到外部（S3/ES 等） |
| `ray-cluster.py-spy.yaml` | 注入 **py-spy**，可对运行中的 worker 采样 Python 栈（查卡死/热点） |
| `install/prometheus/` | KubeRay 自带的 Prometheus 配置（ServiceMonitor/规则） |
| Ray Dashboard（8265） | 节点/actor/task/对象/日志，`kubectl port-forward` 后浏览器看 |

**`py-spy` 那个样例很实用**：分布式训练卡住时（比如 NCCL hang），能直接 `py-spy dump` 看所有 rank 卡在哪一行 —— 这是定位死锁的标准手段。

### 6.4 安全：认证 / TLS / 网络隔离

| 样例 | 作用 |
|---|---|
| `ray-cluster.auth.yaml` / `ray-cluster.auth-manual.yaml` / `ray-cluster.auth.secret-ref.yaml` | Ray 的 **token 认证**（凭据来自 Secret，或手动指定） |
| `ray-cluster.tls.yaml` / `ray-cluster.mtls.yaml` | **TLS / 双向 TLS**（证书从 Secret 挂载） |
| `ray-cluster.network-policy-deny-all.yaml` | 默认拒绝所有入站，再按需放行（最小权限） |
| `ray-cluster.kubernetes.auth.yaml` | 从 K8s ServiceAccount 取认证信息 |

**为什么训练集群也要管这个**：Ray 的 Client 端口（10001）和 Dashboard（8265）**默认没有任何鉴权**——一旦用 `LoadBalancer`/`Ingress` 暴露到公网，等于把集群的任意代码执行能力开放出去。所以生产上要么只走 ClusterIP、要么配 auth + TLS。

### 6.5 存储

| 样例 | 用途 |
|---|---|
| `ray-cluster.gke-bucket.yaml` | 挂 GCS bucket（GKE 的 CSI 驱动） |
| `ray-cluster.historyserver.yaml` | Ray History Server（把已结束集群的日志/状态持久化后回看） |
| `ray-cluster.complete.yaml` 的 `emptyDir: {}` | **注意**：`emptyDir` 随 Pod 销毁而丢——训练要存 checkpoint 必须换 PVC 或挂对象存储 |

**训练任务的存储四件套**：`emptyDir`（`/dev/shm`、`/tmp/ray`，临时）、**PVC**（checkpoint）、**对象存储 CSI**（数据集/模型，多机共享）、以及 `/tmp/ray` 的显式挂载（按 KubeRay 指引，"没有明确挂载时默认值在不同 K8s 发行版上可能不同"，且挂载后 head 重启仍保留日志）。

### 6.6 生态扩展：apiserver / kubectl-plugin / podpool

本地仓里有三个额外的子项目，面试时能提一句说明"知道 KubeRay 的全貌"：

| 组件 | 作用 |
|---|---|
| `apiserver/` | **REST/gRPC API**（`kuberay-apiserver` Helm chart），让不写 YAML 的系统（比如平台前端）也能创建集群 |
| `kubectl-plugin/` | `kubectl ray` 子命令（`kubectl ray job submit` / `kubectl ray session`），免写 YAML |
| `podpool/` | **预热 Pod 池**——提前把 Pod 拉起来待命，避免任务来了才拉镜像（对"冷启动几分钟"的直接解法） |
| `benchmark/` | 基准测试脚本 |
| `clients/` | Python/Go 客户端 |

**`podpool` 值得单独说**：agentic RL 的训练任务经常是"批一批地来"，冷启动（拉镜像+装依赖）可能占掉几分钟。podpool 的思路是**保持一批已初始化的 Pod 待命**，任务来了直接接管——这和第 8 节要讲的 Polar 的 `READY` buffer 是同一个设计思想。

### 6.7 其他值得知道的样例

| 样例 | 场景 |
|---|---|
| `ray-cluster.label-selector.yaml` | 用 label 把 worker 绑到特定节点 |
| `ray-cluster.resource-isolation.gke.yaml` | 资源隔离（GKE 特定） |
| `ray-cluster.custom-head-service.yaml` / `ray-cluster.separate-ingress.yaml` | 自定义 head Service / 分别配 Ingress |
| `ray-cluster.head-command.yaml` / `overwrite-command.yaml` | 覆盖容器启动命令（自定义初始化） |
| `ray-cluster.sandbox.yaml` / `agent-sandbox` | 沙箱化执行（agent 场景常用） |
| `ray-cluster.uv.yaml` | 用 `uv` 管理依赖 |
| `ray-job.sidecar-mode.yaml` / `light-weight-submitter.yaml` | 提交器不进集群（轻量提交）/ sidecar 模式 |
| `ray-job.kueue-toy-sample.yaml` | 与 **Kueue**（K8s 原生排队系统）集成 |
| `vllm/` 目录、`ray-service.llm-serve.yaml` / `deepseek.yaml` / `high-throughput-llm.yaml` | **vLLM 在线推理**（RayService） |

---

## 7. 用 K8s 起"分离式 RL 服务"（以 agent-lighting 为例）

前面讲的都是"起训练集群"。但分离式 RL 平台的形态是**多个常驻服务 + 一个训练作业**，需要的是 Deployment/Service 那一套。

### 7.1 ★ 先分清版本：v0.3.x 没有 K8s，v1.0.0 才有原生 K8s

必须纠正一个常见误解——**agent-lightning 的 K8s 支持是 v1.0.0 才有的，而且 v1.0.0 是一次彻底重构**：

| | **v0.3.x** | **v1.0.0**（tag `8f8b8f95`，2026-08-17） |
|---|---|---|
| K8s 支持 | **没有**（`kubectl`/`helm`/`raycluster`/`kuberay`/`ServiceAccount`/`StatefulSet` 全仓 NOT FOUND） | **原生支持** |
| 官方定位 | 3 进程模型（pip 安装） | "**Native Kubernetes support:** Run agents directly as Kubernetes Jobs **without relying on external sandbox services**" |
| 三组件 | `store` / `runner` / `algo` | **API Gateway** / **Rollout Controller** / **Customized Trainer** |
| CLI | `agl store` / `agl vllm` / `agl prometheus` | `agl-server` / `agl-controller`（v0.3 那几个子命令**已不存在**） |
| 端口 | store `4747` | API Gateway `8080`（官方示例脚本里用 `8181`） |
| 容器编排 | 只有 `docker/`（Dockerfile.dev + 5 个 compose） | `runner_type: k8s`，**每个 rollout 一个 K8s Job** |
| Helm | 无 | **无**（命令式提交 Jinja 渲染出的 Job） |
| 新增依赖 | — | `kr8s>=0.18.0` |

> ⚠️ **本地工作树都 checkout 在 v0.3.x**，但 `agent-lightning-official` 的 git object 里**有 v1.0.0 的 tag**（`git tag` 可见、工作树没切过去）。所以"本地看不到 K8s 文件"≠"官方没有 K8s 方案"。

**v1.0.0 的核心机制**（官方原文）：*"**Kubernetes mode:** creates **one Kubernetes Job for each rollout** from a user-provided template."*

即：**Rollout Controller 不再自己起进程跑 agent，而是把 agent 包成一个 K8s Job 提交出去**——这正是 `Colocate-vs-Disaggregate.md` 里讲的 **rollout-as-a-service** 思路（Polar 用 API 网关、AGL 用 K8s Job，本质都是"把 agent 执行挪出训练进程"）。

### 7.2 v1.0.0 的 minikube 实操（官方 `run_minikube.sh` 逐行）

```bash
# ① 起 Ray head（Trainer 侧要用 Ray）
ray start --head --dashboard-host=0.0.0.0

# ② 起 minikube（官方给了 64GB 内存 + 16 核）
minikube start --memory=65536 --cpus=16 --driver=docker

# ③ 把 agent 镜像 build 进 minikube 的镜像仓库（不 push registry）
minikube image build -t calc-x-agent:dev -f Dockerfile .

# ④ 起 API Gateway（对应 v0.3 的 store）
agl-server port=8181 key=dummy \
    default_proxy.model_name=Qwen/Qwen2.5-1.5B-Instruct &

# ⑤ 等健康
for _ in $(seq 1 60); do curl -sf "http://localhost:8181/healthz" && break; sleep 1; done

# ⑥ 起 Rollout Controller —— 关键就一行：runner_type=k8s
agl-controller \
    runner_type=k8s \
    agl_server.url="http://host.minikube.internal:8181" \
    agl_server.key=dummy \
    k8s_runner.ttl_after_finished=600 &

# ⑦ 跑 Trainer
python train_calc_agent.py --agl-base-url http://localhost:8181 --agl-key dummy --run-name minikube
```

**三个必须注意的坑**：

1. **`host.minikube.internal`** —— minikube 里的 Pod 访问宿主机上的 `agl-server` 要用这个特殊 DNS；**真实集群必须换成 Service 名或 Ingress**（这就是"K8s 起服务"的核心工作，也正是实习做的"公网 URL 自动暴露"）；
2. **`minikube image build`** —— 本地构建的镜像不 push registry，直接进 minikube 镜像存储；**真实集群要么 push registry、要么每个节点预拉**（这就是"Pod 冷启动慢"的根源，对应 KubeRay 的 `podpool` 解法）；
3. **`runner_type` 是唯一开关** —— 改成 `local` 就退回本地进程模式（v0.3 的形态）。官方把"agent 在哪跑"抽象成一个配置项，这是很干净的设计。

**安装（v1.0.0）**：`uv sync` → `bash scripts/setup_verl.sh 0.8.0 cu130`（支持 verl 0.7.1 + cu129 / 0.8.0 + cu130；两条路径都要**本地编译 flash-attn 2.8.3，视 CPU 核数 10–30 分钟**）；Quick Start 只需**单机 1×A100**。

### 7.3 v0.3.x 的做法（本地工作树的实际状态）

v0.3.x **没有 K8s 方案**，官方部署是 **3 进程模型**（`pip install agentlightning` 之后）：

```bash
# 终端 1：Store（唯一数据面）
agl store --host 0.0.0.0 --port 4747

# 终端 2：Runner（跑 agent，需要 OPENAI_API_KEY）
AGL_SERVER_HOST=<store-ip> AGL_SERVER_PORT=4747 AGL_CURRENT_ROLE=runner python -m <runner>

# 终端 3：Algorithm（训练侧）
AGL_SERVER_HOST=<store-ip> AGL_SERVER_PORT=4747 AGL_CURRENT_ROLE=algorithm python -m <algorithm>
```

它自带的容器化只有 `docker/`：`Dockerfile.dev` + 5 个 compose（store 4747 / Prometheus 4748 / Grafana 9091 / Mongo 8.2 副本集）。**所以想在 K8s 上跑 v0.3.x，就是"把 3 个进程各包成一个 Deployment"**——这也是实习里做的形态。

### 7.4 通信矩阵（v0.3.x，决定了要暴露几个网络面）

| 链路 | 协议 | 端点 |
|---|---|---|
| Runner ↔ LightningStore | **HTTP** | `:4747`（`/v1/agl/*`） |
| Algorithm ↔ LightningStore | **HTTP** | 同上（可内嵌 client） |
| Runner → Agent | **进程内调用** | 无网络 |
| Agent → LLMProxy | **HTTP**（OpenAI 兼容） | proxy 端口 |
| LLMProxy → vLLM | **HTTP**（OpenAI 兼容） | `:8000` 或 vLLM 实际端口 |

**结论**：**跨机通信只有两个 HTTP 面**——store 的 4747 和模型端点。K8s 编排要做的就是把这两个面用 Service 暴露出来，其余交给 Ray。

### 7.5 store 的 backend 选择（决定 K8s 形态）

v0.3.x 的 store CLI 在 `agentlightning/cli/store.py`，backend 由 `agentlightning/store/` 决定（`memory.py` / `mongo.py` / `sqlite.py` 是实现，`base.py` 是接口、`client_server.py` 是 C/S）。**backend 直接决定 K8s 上要不要给 store 配 PVC 或外部数据库**：

| backend | K8s 形态 | 适用 |
|---|---|---|
| `memory` | Deployment，无持久化 | 调试 |
| `sqlite` | Deployment + **PVC**（单副本，不能水平扩） | 小规模 |
| `mongo` | Deployment + 外部 MongoDB（或 StatefulSet） | 生产（可多副本） |

### 7.6 三种编排方案（v0.3.x）

**方案 A：一个 RayCluster 内部分角色**（贴合 verl）

```
head Pod        → GCS + 训练入口（AgentLightningTrainer / AgentModeDaemon）
algo worker 组  → verl 训练 worker + vLLM 引擎 + LLMProxy（GPU）
runner worker 组→ LitAgentRunner（CPU 为主）
store           → 独立 Deployment 或 head 的 sidecar
```

优点：复用 Ray 的调度/容错/扩缩容，verl 原生集成；缺点：**所有角色共用一个集群，故障域耦合**。

**方案 B：独立工作负载**（真正的分离式平台形态，也是实习采用的形态）

```yaml
# ① store：Deployment + Service（唯一的数据面）
apiVersion: apps/v1
kind: Deployment
metadata: {name: agl-store, labels: {app: agl-store}}
spec:
  replicas: 1
  selector: {matchLabels: {app: agl-store}}
  template:
    metadata: {labels: {app: agl-store}}
    spec:
      containers:
      - name: store
        image: <你的 agent-lightning 镜像>     # 镜像里要有 agl CLI
        command: ["agl", "store"]
        args: ["--host", "0.0.0.0", "--port", "4747",
               "--backend", "mongo", "--mongo-uri", "mongodb://mongo:27017/?replicaSet=rs0"]
        ports: [{containerPort: 4747}]
        resources: {requests: {cpu: "2", memory: 4Gi}, limits: {cpu: "4", memory: 8Gi}}
        readinessProbe: {httpGet: {path: /health, port: 4747}}
        volumeMounts: [{name: data, mountPath: /data}]
      volumes:
      - name: data
        persistentVolumeClaim: {claimName: agl-store-pvc}
---
apiVersion: v1
kind: Service
metadata: {name: agl-store}
spec:
  selector: {app: agl-store}
  ports: [{port: 4747, targetPort: 4747}]
  type: ClusterIP          # ★ 需要公网时改 LoadBalancer / 加 Ingress
```

```yaml
# ② runner：Deployment + HPA（按队列深度弹性）
apiVersion: apps/v1
kind: Deployment
metadata: {name: agl-runner}
spec:
  replicas: 4
  selector: {matchLabels: {app: agl-runner}}
  template:
    metadata: {labels: {app: agl-runner}}
    spec:
      containers:
      - name: runner
        image: <你的 agent 镜像>
        command: ["python", "-m", "your_runner"]
        env:
        - {name: AGL_SERVER_HOST, value: "agl-store"}     # ★ 用 Service 名，不用 IP
        - {name: AGL_SERVER_PORT, value: "4747"}
        resources: {requests: {cpu: "4", memory: 8Gi}}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: agl-runner}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: agl-runner}
  minReplicas: 4
  maxReplicas: 32
  metrics:
  - type: External        # 按 store 队列深度扩缩（需要 prometheus-adapter）
    external:
      metric: {name: agl_pending_rollouts}
      target: {type: AverageValue, averageValue: "8"}
```

```yaml
# ③ algo / 训练：RayJob 或自定义 CRD（因为训练要 Ray + GPU + gang scheduling）
apiVersion: ray.io/v1
kind: RayJob
metadata: {name: agl-train}
spec:
  entrypoint: "python train.py"
  submissionMode: Interactive
  clusterSelector: {ray.io/cluster: agl-training-cluster}   # 复用常驻训练集群
  shutdownAfterJobFinishes: false
```

**方案 B 的四个要点**：

1. **Service 名做 DNS**（`agl-store`），不用 IP——Pod 重建 IP 会变；
2. **`readinessProbe` 指向 store 的健康端点**，否则 runner 会在 store 没就绪时连上去失败重启；
3. **runner 用 HPA 按队列深度扩缩**（不是 CPU——agent 任务是 IO 密集的，CPU 利用率低但排队很长）；
4. **训练部分交给 RayJob / CRD**，不要用裸 Deployment（训练需要 Ray 集群 + gang scheduling + 生命周期管理）。

### 7.7 与实习的对照：RayCluster Head + TrajStore 双网络链路

实习做的"轨迹数据公网访问能力"就是把上面的方案 B 往前推了一步——**在作业运行态并行创建两条网络链路**：

```
┌──────────────── K8s 集群 ────────────────┐
│  ┌─────────────────┐  ┌────────────────┐ │
│  │ RayCluster Head │  │ TrajStore      │ │
│  │ Service         │  │ Service        │ │
│  │ (dashboard/提交) │  │ (轨迹数据)      │ │
│  └────────┬────────┘  └───────┬────────┘ │
│           │ ClusterIP         │ ClusterIP│
└───────────┼───────────────────┼──────────┘
            │                   │
      ┌─────▼─────┐      ┌──────▼──────┐
      │ Ingress / │      │ Ingress /   │
      │ LoadBalancer│    │ LoadBalancer│  ← 公网 URL 自动分配
      └───────────┘      └─────────────┘
            │                   │
        用户看训练面板        用户看轨迹数据
```

**Operator 要做的事**：作业 CR 被创建时，`reconcile` 里**并行**创建这两个 Service/Ingress，把分配到的公网 URL 写回 `status`；作业删除时用 **Finalizer** 先释放公网资源（Ingress/LB）再删集群，避免留下悬空的 LB 计费。

**面试表述**："轨迹数据的公网访问不是'开个端口'，而是**作业级生命周期管理**——Operator 在作业运行态并行建两条链路（RayCluster Head 给训练面板、TrajStore 给轨迹数据），把公网 URL 写回 CR status，删除时用 Finalizer 有序回收。这里的关键设计是**两条链路的生命周期必须和作业绑定**，否则作业删了 LB 还在计费。"

---

## 8. 大规模集群的真实案例（能直接引用的硬数字）

前面讲的是"怎么配"。这一节是**别人在真集群上跑出来的数字和踩过的坑**——面试里讲这些，比讲 YAML 更显深度。

### 8.1 五组必背的硬数字

| 公司 / 来源 | 规模与数字 | 值得注意的点 |
|---|---|---|
| **Uber**（第一方） | *"we run **several hundreds of Ray clusters at a time**"*；*"**1.5- to 4-times improvement in training speed**"* | 因为用 host networking，**不用 K8s Service 做 head 发现**，而是**自研 init container**；治理策略是"**准入通过但 25 分钟还没调度上就杀掉**"（防止资源被长期占住） |
| **Microsoft AI RELAY**（GCS 瓶颈的最强证据） | actor 创建 **42.6s @8k → 89.4s @32k**；P99 RPC **18.3s → 55.3s**；优化后 **32.7–34.5s（2.85×）/ ~3.6s（16.4×）** | **集群越大，GCS 越成为瓶颈**——这是"Ray 的 GCS 是单点"的量化证明 |
| **Anyscale 10k 节点** | PG ready **303×@10k**、**6.5×@2k** 启动 actor；**10k 节点 40,000 actor**；**GCS 主线程空转 61%（2.51）→ 38%（nightly）**；syncer **200s**；**发布锁占调度循环 17.4%** | "GCS 主线程空转 61%"是很有冲击力的数字——说明 GCS 大量时间在空转而非干活 |
| **NVIDIA Nemotron 3 Ultra** | 550B 总参 / 55B 激活、**>20T tokens**、**>3,000 GPU on Ray Core**；GB300 在**同一 NVLink 域内放置 → RL 迭代吞吐 +13%（零硬件改动）** | "纯调度优化 +13%"说明**放置策略（placement）本身就是性能变量** |
| **Capital One**（唯一 KubeRay + Ray Data + Ray Train + RayTune 全套数字） | 500M+ 记录 / 3.5TB / 512 timesteps；**单 epoch 32 小时且瓶颈在数据加载**；**TorchTrainer 64 GPU worker**；200 并发 HPO；**3–5 天 → ~4 小时（16×）**；**no S3 hand-off** | **瓶颈在数据加载不在 GPU** —— 这是大规模训练最常被忽略的一类瓶颈 |

### 8.2 其他可引用案例

| 来源 | 数字 / 要点 |
|---|---|
| **腾讯**（中文第一方，最详细的超大规模工程叙事） | 单个 Ray 集群 **>10,000 GPU**；联邦了上百个 K8s 集群；**Virtual Kubelet 在 >100 节点时失效** |
| **Spotify** | Ray **2.2.0**、每 Pod **15 CPU / 48Gi**、T4、**一个 worker 一个 GKE 节点**；**镜像拉取从分钟级降到秒级** |
| **Pinterest**（Ray Data 调优阶梯，很实用） | **880k → 4M examples/s**；**第一次上 Ray 只有 1.1M，比单机还差**；靠 **zstd 对象存储 patch 减少 >10× 传输**、**单缓冲把 unpickle 从 400ms 降到 <20ms** |
| **ByteDance**（抢占场景） | 唯一公开的 actor_pool 抢占修复：**`actor_pool` 里的 actor 设 `max_restarts=0`** |
| **Alpa** | **1024×A100 训练 175B**，**57.5% MFU / 179 TFLOPs/GPU** |

### 8.3 坑清单（全部带 issue 号，面试追问时能报出来）

| 坑 | issue / 现象 |
|---|---|
| **GCS 内存泄漏** | ~598KB/h 持续泄漏（#45338） |
| **head OOM** | 内存涨到 11.4GB 被 OOMKill，**而且加内存反而更快挂**（#64241） |
| **actor handle 解析风暴** | 15.8M 次解析（#65782） |
| **KubeRay 写死 200m CPU** | 卡在 worker 启动的关键路径上（#5138） |
| **object store 默认吃 30% 内存** | 且上限 200GB——大内存机器上这是浪费 |
| **driver RSS 线性泄漏** | 且不可调优（#66016） |
| **Volcano gang 在 suspend 时泄漏队列资源** | #4939 |
| **scheduler-plugins 的 PodGroup 只建不更新** | 导致**扩容失效**（#5205） |
| **autoscaler hang** | 导致 **180+ Pod 空转 14 小时**（#60566） |
| **head 永不缩容** | 设计如此（#4768）——head 是常驻的 |

**本地 Kuberay benchmark 的真实数字**（`kuberay/benchmark/perf-tests/*/results/junit.xml`，GKE + KubeRay v1.1.1）：

| 操作 | 耗时 |
|---|---|
| 10,000 个 RayCluster（= **40,000 Pod**）总耗时 | **6459s** |
| 创建 10,000 CR | 88.8s |
| 等全部 ready | 1515s |
| **镜像预热** | **1613s（占比最大）** |

**结论**：**规模化的瓶颈不在"创建 CR"（88.8s），而在"等 ready + 镜像预热"（3128s，占 48%）**——这直接解释了为什么 KubeRay 要做 **`podpool`（预热 Pod 池）**：把镜像和运行时准备好，任务来了直接接管。

### 8.4 ⚠️ 引用这些材料时**不能编**的东西（负面清单）

调研中明确确认的"不存在"：

1. **没有任何一家公司同时公布过「节点数 + GPU 型号卡数 + 网络 + 并行策略 + tokens/s + 版本」**——都是零零散散的；
2. **`tokens/s/GPU` 从未被任何组织或 Ray Summit 公布过**；
3. **没有任何组织公布过自己的 RayCluster YAML**；
4. **没有任何组织提过 `NCCL_IB_DISABLE` / `NCCL_SOCKET_IFNAME` / `NCCL_DEBUG` / `NCCL_ALGO` / `NCCL_IB_HCA`**——所以"大厂在 KubeRay 上怎么配 NCCL"没有公开答案（我前面写的 NCCL 建议是从"多网卡 + RDMA"的通用工程实践推的，**不是**引用的公开案例）；
5. `ray.io/blog` **404**；**不存在** KubeRay production guidance 页；
6. **OpenAI 的 7,500 节点 K8s 与 Ray 无关**（那是 MPI/SSH 方案）——别拿它当 KubeRay 案例；
7. **Ray 官方 K8s 文档里完全没有 sysctl / swap / kernel 参数**——我第 1 节写的那些是标准 K8s 节点前置要求（来自 K8s 侧一般实践），**不是** Ray 官方要求。

> 面试一句话总结：**大规模集群的硬数字是——Uber 同时跑几百个 Ray 集群、训练速度提升 1.5~4×；Microsoft AI RELAY 证明 GCS 是大集群瓶颈（actor 创建 42.6s@8k→89.4s@32k，P99 RPC 18.3s→55.3s）；Anyscale 10k 节点 40,000 actor、GCS 主线程空转 61%；NVIDIA 3,000+ GPU 上纯放置优化让 RL 吞吐 +13%；Capital One 16× 提速但瓶颈在数据加载。规模化真正的成本在"等 ready + 镜像预热"（本地 benchmark：10,000 集群 40,000 Pod 总 6459s，其中预热 1613s）——这才是 podpool 存在的理由。**

---

## 9. 大规模集群的坑与调优清单

| 类别 | 坑 | 解法 |
|---|---|---|
| **调度** | 8 个 Pod 起 7 个，剩下的 Pending，已起的空转 | **gang scheduling**（Volcano/KAI/YuniKorn/coscheduling） |
| **调度** | 一个 8 卡 Pod 调度不上（机器被碎片占用） | `podAntiAffinity` + `hostname` 让 Pod 独占机器；或整机预留 |
| **显存/资源** | Pod 起来了但 Ray 看不到 GPU | 检查 `nvidia.com/gpu` 是否被 device plugin 暴露、`limits` 是否写了 GPU |
| **通信** | 多机 NCCL 带宽远低于线速 | `NCCL_SOCKET_IFNAME` 指定网卡；确认 RDMA/IB 可用；`/dev/shm` 挂够 |
| **共享内存** | 训练进程 `Bus error` / 随机崩 | `/dev/shm` 默认 64MB，必须 `emptyDir{medium: Memory}` 放大到几十 GB |
| **内存** | Pod 被 OOMKilled | Ray object store 默认吃 30% 容器内存，训练还要用；把 memory 给足或调 `--object-store-memory` |
| **存储** | checkpoint 随 Pod 消失 | `emptyDir` 换成 PVC 或对象存储 CSI |
| **冷启动** | 拉镜像 + 装依赖几分钟 | **podpool 预热池**；或提前 `docker pull`（`uniagent-lighting` 实测："`docker run` 隐式拉镜像超 120 秒会被判超时，先 pull 再跑"） |
| **长跑** | 训练跑几小时，head 是单点 | 关注 GCS FT（新方案是 `embedded-gcs-ft`，Redis 方案已标记废弃）；训练侧靠 checkpoint 续训 |
| **弹性** | 训练任务被 autoscaler 缩容 | 训练任务 `maxReplicas = replicas`（不设弹性） |
| **安全** | Dashboard/Client 端口裸暴露 | 只走 ClusterIP，或配 `auth` + `tls/mtls` + NetworkPolicy |
| **可观测** | 卡住不知道卡在哪 | `ray-cluster.py-spy.yaml` 注入 py-spy；Dashboard + Prometheus + Grafana |

**一个很实用的排障顺序**（从外到内）：

```
① kubectl get pods            → Pod 起来了没？状态是什么（Pending/ImagePullBackOff/CrashLoop）？
② kubectl describe pod        → 调度失败原因（资源不足/亲和性冲突/污点）
③ kubectl logs <pod> -c ray-head  → Ray 启动日志（GCS 连不上、版本不匹配都在这）
④ kubectl exec ... -- ray status  → Ray 自己看到的资源与 demand
⑤ Dashboard / py-spy          → 进到任务内部看谁在卡
```

**环境变量的一个硬坑**（`uniagent-lighting` 实测记录，很有代表性）：

> **"Ray worker 的环境变量在 `ray start` 时固定，不继承训练脚本内的 `export`。凡 agent/沙箱/Gateway 运行需要的变量，必须在 `ray start` 之前 `export`，否则 Ray task 内拿不到。"**

排查现场是"`E2B_API_KEY` 未传入，24 个会话全部 `AuthenticationException`"——**类型症状是"任务全失败但代码逻辑没错"，根因是环境变量没进 Ray 运行时**。在 K8s 上对应的做法是：**环境变量必须写在 Pod spec 的 `env` 里**（因为我们没法控制 Operator 内部 `ray start` 的时机），而不是靠 entrypoint 里 `export`。

---

### 9.1 四个容易漏但很关键的点

**① Ray 的日志不写 stdout，写 `/tmp/ray/session_latest/logs`。**

所以 **`kubectl logs` 默认看不到 Ray 的业务日志**（只能看到 `ray start` 那几行）。官方 helm values 里甚至把这个路径拼错成 `session_latests`。**解法**：给 `/tmp/ray` 挂卷（否则 Pod 重建日志就没了）+ 用 FluentBit sidecar 收集（样例 `ray-cluster.fluentbit.yaml`）。

**② KubeRay 会自动注入 `ulimit`。**

官方文档原文：*"If you don't set the annotation, **KubeRay automatically injects the `ulimit` command into the container**"*（`ulimit -n 65536`）。所以"文件描述符不够导致大量连接失败"这个问题**默认已经被处理**；要改得用 annotation 覆盖。

**③ gang scheduling 的开关就是一个 label。**

```
ray.io/gang-scheduling-enabled: "true"
```

**Volcano 的 PodGroup 由 operator 自动创建**（不需要你手写），命名 `ray-<name>-pg`，**size = desiredReplicas（或 minReplicas）+ 1**——那个 **+1 是给 head 留的**。这个细节能体现"真的读过源码"。

**④ Autoscaler 的 RBAC 是 per-cluster 动态创建的（最小权限的好例子）。**

Operator 会为**每个 RayCluster 动态建一个同名 namespaced Role**，只给：`pods` 的 `get/list/watch/patch`、`pods/resize` 的 `patch`、`rayclusters` 的 `get/patch`，然后绑到 **head 的专用 ServiceAccount**。即：autoscaler 只能改"自己这个集群"的副本数，动不了别的集群——**这就是"每个集群一套凭证"的最小权限设计**。

**⑤ v1.7 的升级能力：只支持改 `replicas`。**

官方文档明确：*"only modifications to the **`replicas`** field in RayCluster/RayJob CR are supported"*。想改镜像/资源/命令，得**重建集群**。v1.7 为此新增了 `upgradeStrategy.type: Recreate`——它的实现很巧：**哈希时排除 `replicas` 和 `workersToDelete`**（这样单纯扩缩容不会触发重建），**且 KubeRay 自身版本变化时跳过重建**（避免升级 Operator 就把所有集群重建一遍）。

---

## 10. 速查表

### 端口

| 端口 | 用途 |
|---|---|
| **6379** | GCS（Ray 全局控制面） |
| **8265** | Ray Dashboard |
| **10001** | Ray Client / Job 入口 |
| **4747** | agent-lightning LightningStore 的 HTTP API |

### 常用命令

```bash
# 安装
helm install kuberay-operator kuberay/kuberay-operator
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.1.0&timeout=90s"

# 起集群 / 看状态
kubectl apply -f ray-cluster.complete.yaml
kubectl get raycluster,rayjob,rayservice
kubectl get pods -l ray.io/cluster=<name>
kubectl exec -it <head-pod> -- ray status

# Dashboard
kubectl port-forward svc/<cluster>-head-svc 8265:8265

# 提交任务
kubectl apply -f my-rayjob.yaml
kubectl get rayjob <name> -o jsonpath='{.status.jobStatus}'

# 排障
kubectl -n kuberay-system logs deploy/kuberay-operator -f
kubectl describe pod <pod>
kubectl logs <pod> -c ray-head
```

### 四种 CRD 的定位

| CRD | 用途 | 生命周期 |
|---|---|---|
| **RayCluster** | 一个 Ray 集群 | 手动建/删（或 CRD 删除） |
| **RayJob** | 一次性任务（建集群→跑→回收） | 任务结束按 `shutdownAfterJobFinishes`/`ttlSecondsAfterFinished` 处理 |
| **RayService** | 在线服务（Ray Serve + 零停机升级） | 常驻 |
| **RayCronJob** | 定时任务 | 按 cron 周期 |

### 本地样例文件速查（`kuberay/ray-operator/config/samples/`）

| 我想做的事 | 看哪个样例 |
|---|---|
| 最完整的 RayCluster 参考 | `ray-cluster.complete.yaml` |
| 自动扩缩容 | `ray-cluster.autoscaler.yaml` / `autoscaler-v2.yaml` |
| **跑 verl 训练** | `ray-cluster.verl.yaml` |
| 多机大规模（TPU） | `ray-cluster.tpu-v6e-256-multihost.yaml` |
| gang scheduling | `volcano-scheduler` / `kai-scheduler` / `yunikorn-scheduler` / `scheduler-plugins` |
| 提交任务（三种模式） | `ray-job.sample.yaml` / `interactive-mode` / `use-existing-raycluster` |
| 在线推理 | `ray-service.llm-serve.yaml` / `deepseek.yaml` |
| 监控 | `embed-grafana` / `fluentbit` / `py-spy` |
| 安全 | `auth` / `tls` / `mtls` / `network-policy-deny-all` |
| 容器沙箱（agent 场景） | `ray-cluster.sandbox.yaml` / `agent-sandbox/` |
| vLLM 示例 | `vllm/` 目录 |

---

## 附：高频追问速答

**Q1：KubeRay Operator 到底做了什么？**
一个 controller-runtime 的 reconcile loop：watch `RayCluster`/`RayJob`/`RayService`/`RayCronJob` 四类 CRD，按 spec 创建/删除 head Pod、worker Pod、Service、ConfigMap，把实际状态写回 `status`。所以"改 YAML 就能改集群"的能力在 Operator 里，不在 K8s 本身。

**Q2：为什么生产上建议"少而大的 Ray Pod"？**
Ray 的调度是**节点级**的，一个 Pod = Ray 眼里的一个 node。Pod 切太碎会让 Ray 看到几百个"小节点"，调度开销上升、对象跨节点传输概率上升。样例注释原话："It is better to use a few large Ray pod than many small ones."

**Q3：多机训练为什么必须 gang scheduling？**
8 个 Pod 要 8 张卡，如果只剩 7 台机器，默认调度器会让 7 个起来、1 个 Pending——已起的 7 个占着 GPU 死等，整个集群被这个任务锁死。gang scheduling（Volcano/KAI/YuniKorn/coscheduling）保证"要么全起、要么全等"。

**Q4：`/dev/shm` 为什么要挂？**
NCCL 和多进程共享内存都走 `/dev/shm`，容器默认只有 64MB，训练时会 `Bus error` 或随机崩。解法是 `emptyDir: {medium: Memory, sizeLimit: 64Gi}` 挂到 `/dev/shm`。

**Q5：RayJob 和直接 apply RayCluster 有什么区别？**
RayCluster 只管"建集群"；RayJob 把"建集群 → 跑 entrypoint → 按策略回收"打包，并跟踪 `jobStatus` 状态机（PENDING/RUNNING/SUCCEEDED/FAILED）。跑任务用 RayJob，调环境用 RayCluster。

**Q6：RayJob 的三种提交模式？**
① 默认（K8sJob 模式）：entrypoint 包成 K8s Job 跑，集群随任务生灭；② `Interactive`：把 entrypoint 提交给 Ray 自己的 Job Submission API，同一集群可并存多个 job；③ `clusterSelector`：复用已有集群，不新建。

**Q7：autoscaler 扩容的本质是什么？**
autoscaler sidecar（跑在 head Pod 里）读 Ray 层面的 resource demand，然后**改 `RayCluster.spec.workerGroupSpecs[].replicas`**，再由 Operator 建 Pod——两级 reconcile 协作。所以扩容是"改 CR"而不是"直接建 Pod"。

**Q8：为什么训练任务不该开 autoscaler？**
训练不能"缩到一半"（gang scheduling 要求整组就绪），弹性只适合无状态任务（推理、数据生成、rollout）。训练任务应设 `maxReplicas = replicas`。

**Q9：怎么定位"训练卡住"？**
顺序：`kubectl get pods` → `describe pod` → `logs -c ray-head` → `exec -- ray status` → Dashboard → **`ray-cluster.py-spy.yaml` 注入 py-spy dump 所有 rank 的栈**（定位 NCCL 死锁的标准手段）。

**Q10：环境变量为什么不生效？**
**Ray worker 的环境变量在 `ray start` 时固定，不继承训练脚本里的 `export`。** 必须写在 Pod spec 的 `env`（或 `rayStartParams` 之前的启动脚本）里。实测症状是"任务全失败但代码没错"（如 24 个会话全部 `AuthenticationException`，根因是 API key 没传进 Ray 运行时）。
