<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# K8S 资源管理与作业调度
Kubernetes 的作业调度与资源管理系统如同集群的"智能大脑"和"资源管家"：
- 调度器是决策中枢：通过智能算法将工作负载精准分配到最佳位置。
- 资源管理是保障系统：确保每个应用获得所需资源而不相互干扰。

## 资源管理
### 资源类型
在 K8S 中，CPU 和内存是最主要的两种资源类型，统称为计算资源。CPU 的单位是 Kubernetes CPU，内存单位是字节。不同于 API 资源，计算资源是可以被请求、分配和消耗的。

CPU 资源的限制和请求以 CPU 为单位，1CPU 等于 1 个物理 CPU 或者 1 个虚拟核，取决于运行在物理主机还是虚拟机上。当然也可以使用带小数的 CPU 请求，比如 0.1CPU，等价于 100 millicpu。

值得注意的是，CPU 资源的值是绝对数量，不是相对数量。即无论运行在单核、双核或更高配置的机器上，500mCPU 表示的是相同的算力。
memory 的请求以字节为单位，单位有 E、P、T、G、M、k。

CPU 资源被称为可压缩资源，当资源不足时，Pod 只会饥饿，不会退出。内存资源被称为不可压缩资源，当压缩资源不足时，Pod 会被 Kill 掉。

### 资源请求与限制
针对每个 Pod 都可以指定其资源限制与请求，示例如下：
> spec.containers[].resources.limits.cpu
> spec.containers[].resources.limits.memory
> spec.containers[].resources.requests.cpu
> spec.containers[].resources.requests.memory

在调度时按照 requests 的值进行计算，真正设置 Cgroups 限制的时候，kubelet 会按照 limits 的值来进行设置。Kubenetes 对 CPU 和内存的限额设计，参考了 goole 的 Borg 设计，即设置的资源边界不一定是调度系统必须严格遵守的，实际场景中，大多数作业使用到的资源其实远小于它所请求的资源限额。

Borg 在作业提交后，会主动减小它的资源限额配置，以便容纳更多的作业。当作业资源使用量增加到一定阈值时，会通过快速恢复，还原作业原始资源限额。

Kubenetes 中的 requests+limits 的思想是类似的，requests 可以设置一个相对较小的值，真正给容器设置的是一个相对较大的值。

### QoS 模型
QoS 中设计了三种类型：Guaranteed、Burstable 和 BestEffort。
当 requests=limits，这个 Pod 就属于 Guaranteed 类型，当 Pods 仅设置了 limits，没有设置 requests，这个 Pod 也属于 Guaranteed。当至少有一个 Container 设置了 requests，那么这个 Pod 属于 Burstable。如果一个 Pod 都没有设置，那么这个 Pod 被划分为 BestEffort。

为什么会设置这三种类别呢？
> 主要应用于宿主机资源紧张的时候，kubelet 对 Pod 进行资源回收。比如宿主机上不可压缩资源短缺时，就有可能触发 Eviction。

当 Eviction 发生的时候，就会根据 QoS 类别挑选一些 Pod 进行删除操作。首先就是删除 BestEffort 类别的 Pod，其实是 Burstable 类别、并且饥饿的资源使用量超出了 requests 的 Pod。最后才是 Guaranteed 类别的 Pod 资源。

## 作业调度
Kubernetes 调度系统的核心由两个独立协同的控制循环构成，共同实现高效的资源编排：

Kubernetes 调度系统的核心由两个独立协同的控制循环构成，共同实现高效的资源编排：
- “Informer 循环”：启动多个 Informer 来监听 API 资源（主要是 Pod 和 Node）状态的变化。一旦资源发生变化，Informer 会触发回调函数进行进一步处理。
- “Scheduling 循环”：从调度队列（PriorityQueue）中不断出队一个 Pod，并触发两个核心的调度阶段：Predicates 和 Priority。

K8S 中主要通过 Predicates 和 Priorities 两个调度策略发生作用，K8S 中默认策略可分为四种。

![抢占机制](images/04schedule.png)
### GeneralPredicates
该策略负责基础调度策略，比如 PodFitsResources 计算的是宿主机的 CPU 和内存资源是否足够，但是该调度器并没有适配 GPU 等硬件资源，统一用一种 Extended Resource、K-V 格式的扩展字段来描述。

比如申请 4 个 NVIDA 的 GPU，可以在 request 中写入以下值：alpha.kubernetes.io/nvidia-gpu=4.

在 PodFitsResources 中，调度器直接根据 key 进行匹配，根据 value 进行计算。

### Volume 过滤规则
这一组过滤规则负责 Volume 相关的调度策略，根据 Pod 请求的卷和已挂载的卷，检查 Pod 是否合适于某个 Node (例如 Pod 要挂载/data 到容器中，Node 上/data/已经被其它 Pod 挂载，那么此 Pod 则不适合此 Node)。

### 宿主机相关的过滤规则
该组规则主要判断 Pod 是否满足 Node 某些条件。比如 PodToleratesNodeTaints：检查 Pod 的容忍度是否能承受被打上污点的 Node。只有当 Pod 的 Toleration 字段与 Node 的 Taint 字段能够匹配的时候，这个 Pod 才能被调度到该节点上。NodeMemoryPressurePredicate 检查当前节点的内存是否充足，如果不充足，那么待调度 Pod 就不能被调度到该节点上。

### Pod 相关过滤规则
该组规则很大一部分和 GeneralPredicates 重合，其中比较特别的是 PodAffinity。用于实现 Pod 亲和性与反亲和性（Pod Affinity/Anti-Affinity）的核心预选（Predicate）策略之一。它通过分析 Pod 之间的拓扑关系，决定新 Pod 能否调度到某个节点。
- Pod 亲和性：将新 Pod 调度到与指定 Pod 所在节点相同或相近的节点。
- Pod 反亲和性：避免将新 Pod 调度到与指定 Pod 所在节点相同的节点。适用场景：高可用部署（如避免单点故障）。适用场景：需要紧密协作的服务（如数据库与缓存）。

规则配置示例如下所示：

```yaml
affinity:
  podAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:  # 硬性要求
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values: [web]
      topologyKey: kubernetes.io/hostname  # 拓扑域（如节点、机架）
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:  # 软性偏好
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values: [cache]
        topologyKey: topology.kubernetes.io/zone

```

### 调度器的优先级和抢占机制
该机制在 K8S 1.10 版本后进行支持，使用该机制，需要在 K8S 中定义 PriorityClass，示例如下：
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority  # 优先级类名称（必填）
value: 1000000         # 优先级数值（必填，越大优先级越高）
globalDefault: false   # 是否作为集群默认优先级类（可选，默认 false）
description: "用于关键业务 Pod 的高优先级"  # 描述信息（可选）
preemptionPolicy: PreemptLowerPriority  # 抢占策略（可选，默认 PreemptLowerPriority）
```
其中 Priority 的 value 是一个 32bit 的整数，值越大优先级越高。超出 10 亿的值分配给系统 Pod 使用，确保系统 Pod 不会被用户抢占。

调度器中维护着一个调度队列，高优先级的 Pod 优先出队。同时，当优先级高的 Pod 调度失败时，抢占机器开始发挥作用。调度器尝试从集群中删除低优先级的 Pod，从而使高优先级 Pod 可以被调度到该节点。

![抢占机制](images/04depre.png)

抢占细节如下：
- 抢占候选选择：调度器选择优先级最低且资源释放后能满足需求的 Pod。
优先抢占与高优先级 Pod 资源需求重叠度低的 Pod（如占用 CPU 但新 Pod 需要内存）。
- 优雅终止：被抢占的 Pod 会收到 SIGTERM 信号，进入 Terminating 状态。默认 30 秒后强制终止（可通过 Pod 的 terminationGracePeriodSeconds 调整）。
- 不可抢占的 Pod：优先级 ≥ 高优先级 Pod 的 Pod。
具有 preemptionPolicy: Never 的 Pod。系统关键 Pod（如 kube-system 命名空间下的 kube-dns）。

## 总结与思考

## 参考与引用

- https://www.thebyte.com.cn/container/kube-scheduler.html
