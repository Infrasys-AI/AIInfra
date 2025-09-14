<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# K8S 容器编排与作业管理

!!!!!!!!!
1）注意 markdown 的格式，看我修改和提交的内容；2）看上去基本上是大模型生成的，尽可能用自己的语言和自己的理解去写内容，才有骨肉；3）成段落，不要用大模型的列表和总结展现形式，一定要自己深入核心技术，去挖掘自己以前不懂的东西。4）图很重要，图是理解的很重要一步，自己理解了，然后自己画图，这才是最重要最核心的内容。

## 一、容器编排

### 核心概念

- **Pod**  
  最小部署单元，包含 1 个或多个共享网络/存储的容器
- **Deployment**  
  管理无状态应用的 Pod 副本集（支持滚动更新/回滚）
- **Service**  
  提供负载均衡和服务发现（ClusterIP/NodePort/LoadBalancer）
- **Ingress**  
  管理外部 HTTP/HTTPS 流量路由

### 关键功能

| 功能 | 描述 | 配置示例 |
|------|------|----------|
| 自动调度 | 基于资源需求和节点亲和性 | `nodeSelector: {disk: ssd}` |
| 自愈能力 | 自动重启故障容器 | `restartPolicy: Always` |
| 滚动更新 | 零停机部署 | `strategy: {type: RollingUpdate}` |
| 存储编排 | 动态挂载持久卷 | `persistentVolumeClaim: my-pvc` |

### Deployment 配置示例

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deploy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
```

## 二、作业管理（Job Management）

### 1. Job（一次性任务）

运行直到**成功完成**（退出码为 0）的离散任务

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-processing
spec:
  completions: 6     # 需要完成的任务总数
  parallelism: 2     # 同时运行的 Pod 数量
  backoffLimit: 3    # 失败重试次数
  template:
    spec:
      containers:
      - name: processor
        image: data-tool:v3.2
        command: ["python", "/app/process.py"]
      restartPolicy: OnFailure  # 失败时自动重启
```

### 2. CronJob（定时执行任务）

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-report
spec:
  schedule: "0 3 * * *"  # 每天 3 点运行
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: report-generator
            image: report-tool:latest
          restartPolicy: OnFailure
```

### 关键参数

| 参数| 作用 | 示例值 |
|------|------|----------|
| backoffLimit | 失败重试次数 | 3 |
| activeDeadlineSeconds | 任务超时时间 | 3600 |
| successfulJobsHistoryLimit | 保留成功 Job 记录数 | 5 |
| failedJobsHistoryLimit | 保留失败 Job 记录数 | 2 |

### 三、容器编排 vs 作业管理对比

| 维度| 容器编排（Deployment） | 作业管理（Job/CronJob） |
|------|------|----------|
| 设计目标 | 长期运行服务 | 离散任务执行 |
| 生命周期 | 持续运行 | 运行到完成/超时 |
| 重启策略 | Always (默认) | OnFailure/Never |
| 扩缩容机制 | HPA 自动扩缩 | parallelism 手动控制并发 |
| 典型场景 | Web 服务/数据库 | 批处理/定时报表/数据迁移 |

### 四、最佳实践

下面是使用时的推荐的一些最佳配置：

- **资源限制：为 Job 设置 resources.requests/limits 避免资源竞争。**
- **超时控制：使用 activeDeadlineSeconds 防止任务卡死。**
- **存储分离：Job 中挂载临时卷（emptyDir）避免数据残留。**
- **监控：通过 Prometheus 监控 Job 执行状态和时长。**

## 总结与思考

## 参考与引用

!!!!!!!!!!!!!
