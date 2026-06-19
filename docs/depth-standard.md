# 深度标准与覆盖矩阵

本书的目标是让读者系统掌握 AI Factory 的工程技术，而不是只获得术语概览。读者读完本书后，应该能理解一个 AI Factory 从应用入口到 GPU、网络、存储、机房和经济性之间的关键链路，并能在真实生产问题中定位责任边界、设计排障路径、制定验收标准。

## 深度标准

一章达到生产级深度，至少应满足以下标准：

- 机制说清：不仅说明“是什么”，还要说明组件在控制流、数据流、生命周期中的位置。
- 边界说清：说明它不负责什么，和相邻层如何交互，避免把平台、调度、IaaS、runtime 混在一起。
- 路径说清：至少给出一条请求、任务、设备、数据或故障穿过系统的完整路径。
- 证据说清：涉及产品能力、规范、接口、版本兼容时，优先引用官方文档、标准或公开论文。
- 工程说清：给出真实配置、命令、YAML、状态机、排障步骤或验收流程，而不是停留在概念解释。
- 故障说清：列出常见失败模式、症状、定位入口、责任团队和修复方向。
- 指标说清：说明观测指标的口径、用途、标签维度、阈值或基线来源。
- 取舍说清：说明多种方案适合哪些场景，以及选择错误会带来什么后果。
- 图表说清：图要表达架构、流程、状态或依赖关系，不能只做装饰。
- 跨章闭环：一个复杂主题可以分布在多个章节，但必须能在全书范围内覆盖核心知识点。

## 图表标准

图表应优先覆盖以下类型：

- 分层架构图：说明系统边界和责任归属。
- 生命周期图：说明对象从提交、调度、启动、运行、失败到回收的状态变化。
- 数据路径图：说明 token、checkpoint、模型权重、训练数据或监控事件如何流动。
- 控制路径图：说明 API、controller、scheduler、runtime、operator 如何协同。
- 故障树图：说明一个症状如何分解到可能根因。
- 验收流水线图：说明准入、基线、异常检测和资源池状态如何闭环。

## NVIDIA GPU Container 知识覆盖矩阵

用户提供的《Nvidia GPU Container 原理》不是要求整篇进入某一章，而是要求全书读完后覆盖并深化其核心知识点。当前规划如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| 容器不是 VM，依赖 namespace、cgroup、rootfs 和 OCI runtime | 第 21 章 | 解释容器隔离边界、OCI spec、runc 生命周期 |
| 手动挂载 `/dev/nvidia*`、driver library 与 `--privileged` 的区别 | 第 21 章 | 说明原理、风险、为什么生产不应依赖 privileged |
| `nvidia-docker`、`nvidia-docker2`、NVIDIA Container Toolkit 演进 | 第 21 章 | 解释演进原因、兼容性和当前推荐路径 |
| `nvidia-container-runtime` 修改 OCI spec | 第 21 章 | 讲清 runtime wrapper 位置和职责 |
| `nvidia-container-runtime-hook` 作为 OCI prestart hook | 第 21 章 | 讲清 hook 调用时机和失败表现 |
| `nvidia-container-cli` 与 `libnvidia-container` 注入设备和库 | 第 21 章 | 讲清 bind mount、环境变量、capabilities 和 debug 入口 |
| `NVIDIA_VISIBLE_DEVICES`、`NVIDIA_DRIVER_CAPABILITIES` | 第 21 章、第 22 章 | 解释语义、风险、与 Kubernetes request 的关系 |
| Docker、containerd、CRI-O 与 Kubernetes CRI | 第 21 章 | 说明 kubelet 到 CRI 到 OCI runtime 的链路 |
| Kubernetes device plugin 和 GPU 资源分配 | 第 22 章 | 区分“分配 GPU”和“容器内可访问 GPU” |
| GPU Operator 管理 driver、Toolkit、device plugin、DCGM | 第 19 章、第 22 章 | 说明管理边界、冲突和升级策略 |
| driver、CUDA、NCCL、OFED、Toolkit 兼容矩阵 | 第 19 章、第 29 章 | 说明主机/容器边界、版本矩阵和漂移控制 |
| GPU 容器准入与 smoke test | 第 29 章、第 38 章 | 说明节点入池前如何验证容器内 GPU、NCCL、RDMA |
| 容器内 GPU 与 RDMA/NIC 协同 | 第 22 章、第 32 章、第 38 章 | 说明 device、NIC、NUMA、RDMA、NCCL 的完整路径 |

## 全书循环更新策略

全书更新按“主题链路”推进，而不是按章节孤立推进。每轮选择一条关键链路，补齐机制、图、配置、故障、指标和验收。

优先链路：

- GPU 容器链路：driver、CUDA、NVIDIA Container Toolkit、CRI、Kubernetes、device plugin、准入。
- 推理请求链路：Gateway、认证、路由、prefill、decode、KV Cache、streaming、计量、成本。
- 训练任务链路：提交、队列、quota、gang、GPU 拓扑、NCCL、checkpoint、评测、注册。
- 网络通信链路：NVLink/NVSwitch、IB/RoCE、RDMA、NCCL、telemetry、故障树。
- 存储数据链路：dataset、object storage、parallel file system、local NVMe、cache、checkpoint。
- 可靠性链路：DCGM、Xid、ECC、NCCL hang、network error、SLO、incident、准入基线。
- 经济性链路：tokens/s、tokens/W、cost per token、revenue per token、GPU 利用率和毛利。

每轮完成后必须：

- 更新对应章节正文。
- 补充或修正 Mermaid 图。
- 补齐延伸阅读来源。
- 运行 `mkdocs build --strict`。
- 检查是否产生死链、占位、乱码或语义重复。
- 更新 `docs/codex-handoff.md`。
