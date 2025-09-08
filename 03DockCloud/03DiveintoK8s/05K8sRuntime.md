# Kubernetes 运行时
前文中已经介绍完如何进行资源管理和调度，完成这些工作后，就需要将调度成功的Pod在宿主机中创建出来。
这个功能正是kubelet通过CRI和底层容器运行时进行交互。

为什么需要抽象出CRI这个概念？
> 首先是统一容器生命周期管理接口，避免Kubernetes与特定容器运行时强绑定。支持多种容器运行时（Docker、containered、CRI-O）。


## CRI核心组件
- kubelet：负责管理节点上的 Pod 和容器，通过 CRI 与容器运行时交互。
- CRI shim：实现CRI 接口的适配层（如 dockershim、containerd-shim），作为gRPC服务端负责将CRI请求的内容转为具体的容器运行时API并响应。
- Container Runtime：实际执行容器操作的引擎（如 containerd、CRI-O）。

![CRI架构](images/05cri.png)

## kubelet与CRI交互原理
整体架构如下图所示，当k8s创建一个Pod后，就会通过调度器选择一个具体的节点来运行。接着kubelet就会通过SyncLoop来执行具体的操作。

![CRI架构](images/05criinfra.png)

### SyncLoop是什么？
SyncLoop是kubelet的工作核心，主要负责Pod生命周期的核心控制循环，负责保障Pod实际状态与期望状态一致。其核心职责包括：
- 监控Pod状态：持续监听API Server的Pod配置变更（如创建、更新、删除）。
- 协调Pod状态：根据最新配置，触发Pod的创建、更新或删除操作。
- 处理运行时异常：检测容器崩溃、资源不足等问题，并尝试恢复。

其主要输入的Channel包括：
- Pod配置变更（来自API Server的Watch 机制）。
- 定时同步触发（默认每10秒一次，由 sync-frequency参数控制）。
- 容器运行时状态更新（如容器退出、镜像拉取失败）。
- 探针（Probe）结果（如 Readiness/Liveness检查失败）。


主要逻辑在syncPod函数中执行，负责单个Pod的状态同步：

(1) 状态对比
对比实际状态和期望状态，期望状态来自API Server或静态 Pod 配置文件。实际状态通过容器运行时（如 containerd）查询当前容器状态。

(2) 操作执行
- 创建 Pod：创建Pod 的沙箱环境（pause 容器）。按顺序启动 Init 容器和业务容器。挂载 Volume 并设置网络命名空间。
- 更新 Pod：对比新旧配置差异，仅更新变化的容器或资源（如镜像版本、环境变量）。
- 删除 Pod：发送 SIGTERM 信号终止容器。等待优雅终止超时后强制终止（SIGKILL）。清理 Volume 和网络资源。

(3) 状态回写
将 Pod 的当前状态（Status）和事件（Event）更新到 API Server，供控制器（如 Deployment Controller）使用。

### CRI规范
CRI中主要定义了两类接口：
- ImageService：定义拉取、查看和删除镜像等操作。
```
// ImageService defines the public APIs for managing images.
service ImageService {
    
    rpc ListImages(ListImagesRequest) returns (ListImagesResponse) {}
    
    rpc ImageStatus(ImageStatusRequest) returns (ImageStatusResponse) {}
   
    rpc PullImage(PullImageRequest) returns (PullImageResponse) {}
   
   
    rpc RemoveImage(RemoveImageRequest) returns (RemoveImageResponse) {}
   
    rpc ImageFsInfo(ImageFsInfoRequest) returns (ImageFsInfoResponse) {}
}
```

- RuntimeService：定义容器相关的操作。管理容器的生命周期，以及与容器交互的调用等操作。主要包括container和PodSandbox相关的管理接口，包括exec、attach等。

### CRI操作容器容器的生命周期
通过kubectl来运行一个Pod时，会执行以下动作：
- 首先调用RunPodSandbox接口来创建一个Pod容器，Pod容器是用来持有容器的相关资源的，比如说网络空间、PID空间、进程空间等资源。
- 接着调用CreatContainer接口，在Pod容器的空间创建业务容器。
- 然后再调用StartContainer 接口启动容器。
- 最后调用StopContainer与RemoveContainer接口删除容器。

![CRI架构](images/05cri_cicle.png)

### CRI streaming接口
流式接口不仅可以节省资源，还能保证连接的可靠性。当执行kubectl exec命令时，其执行过程如下：
- 第一步：kubectl 将 exec 请求封装为一个 HTTP 请求，该请求将被 API Server 解析。
- 第二步：API Server 将请求转发至目标 Pod 所在节点的 kubelet，建立与 kubelet 的 WebSocket 连接，用于实时数据传输。
- 第三步：kubelet 接收请求后，通过CRI与容器运行时交互以执行命令。

![CRI架构](images/05cri_streaming.png)

### 为何要设计CRI shim这一模块？
最后再来探讨一个问题，为什么要设计CRI shim模块。早期Kubernetes 最初直接依赖 Docker 的 API 管理容器，导致无法支持其他容器运行时（如 containerd、CRI-O）。还有一点原因就是用户的多样化需求，比如轻量级运行时等。

CRI shim有哪些优势呢？
- 解耦核心组件：kubelet 仅需实现 CRI 客户端逻辑，无需为每个运行时编写适配代码。
- 支持多运行时：支持Docker、containerd、CRI-O等。
- 简化维护

## 参考文档
https://www.cnblogs.com/zhangmingcheng/p/17495311.html
