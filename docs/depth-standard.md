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

## 推理请求链路覆盖矩阵

推理请求链路的目标是让读者能从一个用户 Chat 请求一路追到 Gateway、模型服务、推理引擎、GPU/HBM、streaming、计量和经济模型。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| Chat API、message、prompt、context 的边界 | 第 1 章 | 说明应用输入如何变成模型 prompt 和 tokenized context |
| input/output token 与 prefill/decode 的资源差异 | 第 1 章、第 15 章、第 41 章 | 说明 token 对 TTFT、TPOT、KV Cache 和成本的影响 |
| streaming 生命周期、取消和部分输出计量 | 第 1 章、第 6 章、第 14 章 | 说明 generated token、delivered token、close reason 和 drain |
| Gateway admission chain | 第 6 章 | 说明 identity、capability、budget、policy、route、commit 的顺序 |
| 可路由健康与 endpoint picker 思想 | 第 6 章、第 14 章 | 说明 queue、KV pressure、TTFT/TPOT 和 endpoint 状态如何影响路由 |
| append-only metering event | 第 6 章、第 41 章 | 说明 request_admitted、first_token、usage_delta、request_closed 的账实关系 |
| OpenTelemetry / GenAI telemetry 语义 | 第 8 章 | 说明通用语义与 AI Factory 自定义标签如何组合 |
| 模型服务 endpoint/replica 生命周期 | 第 14 章 | 说明 warming、ready、canary、stable、draining、rollback 状态 |
| serving release 组合版本 | 第 14 章 | 绑定 weights、tokenizer、chat template、runtime、engine config 和 rollback |
| 推理引擎 admission model | 第 15 章 | 说明 input tokens、max output、KV block、deadline 和 queue class |
| continuous batching 与 KV Cache 状态流 | 第 15 章 | 说明 waiting、prefill、active decode、release、usage 的阶段 |
| benchmark matrix | 第 15 章 | 覆盖短输入短输出、长输入短输出、短输入长输出和生产混合负载 |
| Token Factory ledger | 第 41 章 | 说明 token ledger、resource ledger、阶段成本和毛利约束指标 |

## 训练任务链路覆盖矩阵

训练任务链路的目标是让读者能从一次训练提交一路追到队列、quota、gang、拓扑放置、NCCL、checkpoint、评测、模型注册和训练 ROI。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `TrainingJob` 生产对象 | 第 10 章 | 说明数据、模型、运行时、并行、调度和恢复语义 |
| 训练状态机 | 第 10 章 | 覆盖 submitted、admitted、preflight、rendezvous、running、checkpointing、recovering |
| checkpoint manifest | 第 10 章 | 说明 sharded checkpoint、optimizer/scheduler/rng/data loader state 和恢复校验 |
| 并行策略与 rank mapping | 第 17 章 | 说明 rank 到 GPU/NIC/rack/rail/group 的映射和一致性校验 |
| 并行模板 scorecard | 第 17 章 | 说明扩展效率、通信、显存、稳定性和可恢复性基线 |
| communication diagnostic bundle | 第 18 章 | 说明异常 op、rank、节点、NIC、端口、NCCL 环境和 telemetry 的证据包 |
| 通信基线库 | 第 18 章、第 38 章 | 说明 NCCL baseline 与真实训练退化对比 |
| job admission event / pending reason | 第 23 章 | 说明 quota、gang、topology、image、data、checkpoint 和 node baseline 检查 |
| checkpoint-aware preemption | 第 23 章 | 说明训练抢占点、通知、保存状态和浪费 GPU 小时 |
| Slurm 平台同步事件 | 第 24 章 | 说明 Slurm job/step/accounting 与 experiment、checkpoint、registry 的统一事件 |
| TrainingJob smoke test | 第 38 章 | 说明 gang、rank mapping、NCCL rendezvous、first effective step 和 checkpoint manifest 验收 |
| training ROI ledger | 第 41 章 | 说明 allocated/effective/wasted GPU hours、checkpoint、评测、上线收益和成本变化 |

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
