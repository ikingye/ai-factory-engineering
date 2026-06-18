# 第 20 章：AI Workload 的形态

## 本章回答的问题

- AI Factory 中常见 workload 有哪些形态？
- 在线推理、批量推理、训练、微调、评测、数据处理和 HPC-style job 的调度需求有什么差异？
- 为什么 AI workload 不能简单套用普通微服务的资源模型？

## 20.1 online inference

Online inference 是面向用户实时请求的在线推理。它关注 TTFT、TPOT、端到端延迟、错误率、可用性和成本。在线推理通常需要常驻服务、弹性扩缩容、灰度发布、模型路由、限流、熔断和 token 计量。

和普通 Web 服务相比，在线推理更受 GPU HBM、KV Cache、batching 和上下文长度影响。扩容也不是简单增加 Pod 数，因为每个副本可能需要完整加载模型权重，占用大量显存和启动时间。平台需要在低延迟和高吞吐之间取舍。

## 20.2 batch inference

Batch inference 是离线或准实时批量生成任务，例如批量摘要、内容审核、数据标注、离线评测或批量 embedding。它通常不要求单请求低延迟，但要求单位时间吞吐、任务完成时间和成本可控。

批量推理适合更激进的 batching、更高 GPU 利用率和队列式调度。它可以使用 Job、Ray、Argo Workflows 或专门的批处理框架运行。平台应把它和在线推理分开治理，避免离线任务抢占在线服务的显存、网络或存储带宽。

## 20.3 distributed training

Distributed training 是多 GPU、多节点协同训练模型的 workload。它通常需要所有 worker 同时启动，依赖 NCCL、RDMA、共享数据读取和 checkpoint 写入。训练任务持续时间长，失败成本高，对节点健康、网络稳定性和存储吞吐非常敏感。

```mermaid
flowchart LR
  Submit["训练提交"] --> Queue["队列 / 配额"] --> Gang["gang scheduling"] --> Pods["多 worker 启动"]
  Pods --> Data["数据读取"] --> NCCL["NCCL 初始化"] --> Loop["forward / backward"] --> Comm["梯度通信"] --> Ckpt["checkpoint"]
```

调度器必须避免半启动。若 8 个 worker 只启动 7 个，任务无法前进，却可能占住已分配 GPU。Gang scheduling、队列、配额、优先级和抢占策略是训练平台的基础能力。

## 20.4 fine-tuning

Fine-tuning 的规模通常小于预训练，但频率更高、租户更多、数据隔离要求更强。企业微调任务可能要求使用私有数据、专属镜像、指定基础模型和独立评测流程。

微调平台需要处理数据上传、权限、任务模板、LoRA/QLoRA 配置、产物管理和模型注册。它对调度的要求介于在线服务和大规模训练之间：既要支持队列和配额，也要提供更易用的产品化入口。

## 20.5 evaluation

Evaluation 包括离线 benchmark、回归评测、安全评测、人工评测和线上 A/B。它消耗的资源可能是推理 GPU、CPU、存储、标注系统和日志分析平台。评测 workload 常常需要可复现环境、固定模型版本、固定数据集和可追溯报告。

评测的特殊性在于它直接影响上线决策。平台需要把评测结果和模型版本、prompt 版本、推理引擎版本、数据集版本绑定，否则无法解释为什么一次上线后质量或延迟发生变化。

## 20.6 data processing

Data processing 包括数据清洗、去重、过滤、tokenization、embedding 构建、数据混合和样本生成。它可能主要消耗 CPU 和存储，也可能使用 GPU 做 embedding 或多模态处理。

数据处理经常成为训练链路的隐性瓶颈。GPU 等待数据时，表面上是训练效率低，根因可能是对象存储吞吐不足、数据格式不适合顺序读取、tokenization 没有预处理，或缓存策略不合理。

## 20.7 HPC-style job

HPC-style job 强调批式提交、队列、资源独占、拓扑、长时间运行和高性能通信。大规模训练与 HPC 有很多共同点，因此 Slurm 在 AI 集群中仍然重要。Kubernetes 生态也通过 Volcano、Kueue 等组件补足批式调度能力。

选择 Slurm 还是 Kubernetes，不应只看团队偏好。Slurm 在 HPC 作业和拓扑调度上成熟，Kubernetes 在服务化、控制器生态和云原生集成上强。很多组织会同时使用二者，或者在不同集群承担不同 workload。

## 20.8 AI workload 与普通微服务的差异

普通微服务通常可以独立副本扩缩容，失败后快速重启，资源以 CPU/内存为主，网络通信以请求响应为主。AI workload 则经常需要 GPU、HBM、RDMA、拓扑、长启动时间、模型权重加载、KV Cache、数据集、checkpoint 和多副本同步。

这导致调度系统必须表达更多语义：资源组、gang、队列、配额、优先级、拓扑、健康状态、镜像版本、驱动版本和存储路径。AI Factory 的调度层不是普通 Kubernetes Scheduler 的简单使用，而是围绕 AI workload 特征构建的资源编排与作业调度层。

## 小结

- Online inference 关注低延迟、稳定性、token 计量和成本。
- Batch inference 关注吞吐、完成时间和离线资源效率。
- Distributed training 需要 gang scheduling、NCCL、RDMA、checkpoint 和长时间稳定性。
- Fine-tuning、evaluation 和 data processing 有不同的租户、数据和复现需求。
- AI workload 比普通微服务更依赖 GPU、拓扑、显存、网络、存储和调度语义。

## 延伸阅读

- TODO: 官方文档
- TODO: 经典论文
- TODO: 工程案例
