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
| CDI / NRI 与 legacy hook 模式 | 第 19 章、第 21 章、第 22 章 | 说明 Container Device Interface、Node Resource Interface、RuntimeClass、device list strategy 和验收差异 |
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
| `engine_admission_health` | 第 6、14、15、37、39 章 | 说明 Gateway 和 Model Serving 如何基于 queue、KV pressure、active sequence、deadline miss 做可路由健康判断 |
| `kv_block_ledger` | 第 1、14、15、37、39、41 章 | 说明 KV block 分配、释放、碎片、租户归属、prefix cache 和取消泄漏如何进入容量与成本 |
| `engine_canary_record` | 第 14、15、37、39 章 | 说明 engine/runtime 变更如何同时验证协议、质量、延迟、KV、token drift 和成本 |
| `speculative_decoding_report` | 第 15、41 章 | 说明 draft/target 模型、接受率、验证开销、质量漂移和真实 cost/token 收益 |
| `pd_disaggregation_contract` | 第 14、15、37、39 章 | 说明 prefill/decode 分离的 KV 传输、容量比例、失败语义、观测和回滚边界 |
| `inference_runtime_diagnostic_bundle` | 第 37、39 章 | 说明 TTFT/TPOT/streaming gap 事故如何自动冻结 Gateway、Serving、Runtime、KV、canary 和 PD 证据 |
| `inference_runtime_cost_ledger` | 第 41 章 | 说明 speculative decoding、PD 分离、KV block、取消浪费和质量成本如何共同决定成功回答成本 |
| Token Factory ledger | 第 41 章 | 说明 token ledger、resource ledger、阶段成本和毛利约束指标 |

## 训练任务链路覆盖矩阵

训练任务链路的目标是让读者能从一次训练提交一路追到队列、quota、gang、拓扑放置、NCCL、checkpoint、评测、模型注册和训练 ROI。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `TrainingJob` 生产对象 | 第 10 章 | 说明数据、模型、运行时、并行、调度和恢复语义 |
| 训练状态机 | 第 10 章 | 覆盖 submitted、admitted、preflight、rendezvous、running、checkpointing、recovering |
| checkpoint manifest | 第 10 章 | 说明 sharded checkpoint、optimizer/scheduler/rng/data loader state 和恢复校验 |
| 并行策略与 rank mapping | 第 17 章 | 说明 rank 到 GPU/NIC/rack/rail/group 的映射和一致性校验 |
| `placement_commit_record` | 第 17、23、37、39、41 章 | 说明并行放置意图、实际放置、降级原因、rank mapping 和性能影响 |
| 并行模板 scorecard | 第 17 章 | 说明扩展效率、通信、显存、稳定性和可恢复性基线 |
| communication diagnostic bundle | 第 18 章 | 说明异常 op、rank、节点、NIC、端口、NCCL 环境和 telemetry 的证据包 |
| 通信基线库 | 第 18 章、第 38 章 | 说明 NCCL baseline 与真实训练退化对比 |
| job admission event / pending reason | 第 23 章 | 说明 quota、gang、topology、image、data、checkpoint 和 node baseline 检查 |
| `training_lifecycle_event` | 第 23、37、41 章 | 说明 submitted、admitted、placement committed、gang、rendezvous、first effective step、checkpoint、recovering 的阶段事实 |
| `queue_fairness_ledger` | 第 23、41 章 | 说明 guaranteed、borrowed、lent、preempted、starved 和 effective GPU hours 如何支撑公平运营 |
| checkpoint-aware preemption / `preemption_record` | 第 23、41 章 | 说明训练抢占点、通知、保存状态、恢复结果和浪费 GPU 小时 |
| Slurm 平台同步事件 / `training_accounting_reconciliation` | 第 24、41 章 | 说明 Slurm job/step/accounting 与 experiment、checkpoint、registry、平台成本的统一事件和对账 |
| TrainingJob smoke test | 第 38 章 | 说明 gang、rank mapping、NCCL rendezvous、first effective step 和 checkpoint manifest 验收 |
| `training_lifecycle_telemetry_event` | 第 37 章 | 说明训练生命周期、拓扑、checkpoint 和 GPU 小时口径如何进入观测事实层 |
| `training_incident_record` | 第 39、41 章 | 说明训练事故如何回指 admission、placement、rank mapping、checkpoint、资源健康和 ROI 损失 |
| training ROI ledger | 第 41 章 | 说明 allocated/effective/wasted GPU hours、调度等待、放置降级、抢占、checkpoint、评测、上线收益和成本变化 |

## 网络通信链路覆盖矩阵

网络通信链路的目标是让读者能从训练或推理的性能症状一路追到 GPU 拓扑、NIC、rail、leaf/spine、RDMA/NCCL、telemetry、准入基线和调度动作。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `network_path_evidence` | 第 30 章 | 说明 job/request 到 placement、GPU、NIC、rail、switch port、baseline 和动作的证据链 |
| east-west / north-south 分流 | 第 30 章 | 说明训练通信、推理入口、存储路径和管理流量的边界 |
| scale-up topology contract | 第 31 章 | 说明 NVLink/NVSwitch 域、GPU-to-GPU bandwidth、拓扑碎片和资源等级 |
| GPU-to-NIC / NUMA 亲和 | 第 31 章、第 32 章 | 说明节点内拓扑如何影响 RDMA、NCCL 和调度 |
| `gpu_nic_affinity_report` | 第 31 章 | 说明 GPU、NUMA、NIC、rail、container device 和 NCCL interface 选择如何对齐 |
| `fabric_baseline` | 第 32 章、第 38 章 | 说明 fabric、rail、版本、测试项、失效条件和可调度能力 |
| 多 rail 放置与诊断 | 第 32 章 | 说明 rank、NIC、rail、leaf group 和 rail balance 的一致性 |
| `rail_balance_report` | 第 32 章、第 37 章 | 说明设计 rail、实际接口、端口利用、rank 流量和失衡动作 |
| `fabric_change_record` | 第 32 章、第 38 章 | 说明交换机、NIC、OFED、CNI、NCCL、PFC/ECN 和调度标签变更如何触发回归 |
| RDMA in container | 第 22 章、第 32 章、第 38 章 | 说明宿主机 RDMA 正常不等于容器内 RDMA 可用 |
| `network_diagnostic_bundle` | 第 32 章、第 39 章 | 说明 rank mapping、NCCL env、RDMA counters、switch ports、baseline 和 verdict |
| `congestion_event_record` | 第 30、37、39 章 | 说明 ECN/PFC、队列、水位、流量类别、job 影响和止血动作如何串联 |
| network telemetry 到业务影响 | 第 37 章 | 说明端口事件如何映射到 job、model、tenant、baseline drift 和 owner |
| fabric acceptance matrix | 第 38 章 | 说明 same rack、cross rack、cross rail、host/container/Kubernetes 的验收矩阵 |
| NCCL hang 网络故障树 | 第 39 章 | 说明 rank 退出、GPU/NVLink、RDMA/fabric、container/runtime、collective mismatch 的排查顺序 |
| `network_cost_ledger` | 第 41 章 | 说明网络退化、拥塞、错误放置和 checkpoint 叠加如何转化为 GPU idle 与 token 成本 |

## 存储数据链路覆盖矩阵

存储数据链路的目标是让读者能从 GPU idle、checkpoint 慢、模型冷启动或成本异常一路追到 dataset、checkpoint、model artifact、cache、manifest、storage backend、telemetry、准入基线和经济影响。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `dataset_manifest` | 第 10 章、第 20 章、第 33 章 | 说明数据来源、处理版本、shard、checksum、统计、权限和缓存策略 |
| `workload_storage_intent` | 第 20、33、37 章 | 说明训练、推理、评测和数据处理如何声明 dataset、checkpoint、artifact、cache 和观测需求 |
| `checkpoint_commit_record` | 第 10 章、第 33 章、第 41 章 | 说明分片写入、校验、manifest commit、latest valid、恢复候选和 GPU idle 成本 |
| `model_artifact_distribution` | 第 14 章、第 33 章 | 说明权重、tokenizer、template、digest、缓存预热和 readiness 的关系 |
| `cache_residency` | 第 14 章、第 33 章、第 41 章 | 说明本地 NVMe、rack cache、权重 cache 与调度、冷启动和成本归因的关系 |
| `data_path_evidence` / `storage_evidence` | 第 33 章、第 37 章、第 39 章、第 41 章 | 说明 workload 到 path、manifest、client、cache、backend、telemetry 和 impact 的证据链 |
| `storage_acceptance_matrix` / `storage_composite_regression_gate` | 第 38 章 | 说明 dataset read、checkpoint write/restore、model load、cache miss 以及 NCCL+checkpoint+artifact 并发门禁 |
| 存储故障树 | 第 39 章 | 说明 GPU idle、checkpoint slow、model load slow 如何定位到 dataset/checkpoint/artifact/cache，并要求 workload impact 证据 |
| `storage_cost_ledger` | 第 41 章 | 说明 dataset read、checkpoint、artifact retention、cache miss、local NVMe 保留和 storage-induced GPU idle 的成本归因 |

## 可靠性与运维链路覆盖矩阵

可靠性链路的目标是让读者能从 SLO 违约、训练失败、GPU degraded、变更回归或机房故障一路追到 health state、maintenance window、acceptance baseline、fault domain、incident、error budget、change safety 和经济损失。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `resource_health_record` | 第 28 章、第 37 章 | 说明 GPU/node/fabric/storage 健康信号如何转成资源池状态和调度动作 |
| `maintenance_window` | 第 28 章、第 40 章 | 说明计划维护、drain、复测、回滚、容量影响和用户沟通 |
| `change_safety_case` | 第 29 章、第 40 章 | 说明 driver、kernel、NCCL、OFED、runtime、模型服务变更如何绑定验证、灰度、停止条件和回滚 |
| `fault_domain_map` | 第 36 章、第 39 章 | 说明 rack、power、cooling、rail、fabric、storage、batch 与 workload 影响面 |
| `reliability_evidence` | 第 37 章、第 39 章 | 说明 SLO symptom 到 request/job、resource health、baseline、change、fault domain 和 action 的证据链 |
| `acceptance_baseline` invalidation | 第 38 章 | 说明变更如何使基线失效，并触发影响范围复测 |
| `incident_record` | 第 39 章、第 40 章 | 说明事故时间线、影响面、止血动作、根因证据、成本影响和行动项 |
| `slo_budget_ledger` | 第 40 章、第 41 章 | 说明 error budget、reliability cost、wasted GPU hours、赔付和毛利之间的关系 |

## 物理设施与能源链路覆盖矩阵

物理设施链路的目标是让读者能从 tokens/W、GPU 降频、机柜降额、液冷告警或扩容不达预期一路追到 GPU server、compute tray、power shelf、rack、cooling domain、capacity unit、acceptance baseline、调度能力和经济影响。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `gpu_server_profile` | 第 34 章、第 28 章 | 说明 CPU/GPU/HBM/PCIe/NVLink/NIC/NVMe/BMC/power/cooling 如何组成可调度服务器画像 |
| `gpu_capability_scorecard` | 第 35 章、第 41 章 | 说明芯片能力、runtime 验证、模型适配、能效和生产成熟度的评分口径 |
| `power_thermal_envelope` | 第 34 章、第 36 章、第 38 章 | 说明功耗、温度、液冷、降额、满载验收和调度限制 |
| `rack_capacity_unit` | 第 36 章、第 28 章、第 40 章 | 说明 rack/power/cooling/fabric/storage 如何构成可承诺产能单元 |
| `physical_acceptance_matrix` | 第 38 章 | 说明 power、cooling、cabling、BMC、full-load、thermal soak 和 workload 的准入矩阵 |
| `capacity_activation_record` | 第 36 章、第 40 章 | 说明 planned、installed、accepted、limited、allocatable 到 retired 的交付状态流转 |
| `energy_ledger` | 第 36 章、第 41 章 | 说明 GPU power、rack power、PUE、tokens/W、joules/token 和 power/cooling-induced waste 的经济归因 |

## 安全与多租户链路覆盖矩阵

安全与多租户链路的目标是让读者能从一次 API Key 泄露、越权模型调用、跨租户数据风险、GPU 共享隔离争议、runtime privilege 误配或账单归属异常，一路追到 tenant boundary、credential lifecycle、policy decision、runtime privilege、GPU isolation、resource pool、audit event 和成本隔离。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `tenant_boundary` | 第 5 章、第 6 章、第 27 章 | 说明租户在身份、项目、模型、数据、资源池、网络、存储、观测和账单中的边界 |
| `credential_lifecycle` / `api_key_audit_event` | 第 5 章、第 6 章、第 37 章 | 说明 API Key、service account、短期 token 的创建、轮换、禁用、异常检测和审计 |
| `policy_decision_record` | 第 6 章、第 40 章 | 说明 Gateway 策略如何记录 allow/deny/route/fallback、规则版本、输入事实和可回放决策 |
| `tenant_cost_isolation` | 第 7 章、第 41 章 | 说明共享池、专属池、免费额度、失败成本、reservation 和账单争议如何归属到租户 |
| `runtime_privilege_profile` | 第 21 章、第 22 章 | 说明 container privilege、hostPath、capability、seccomp、device 注入、RDMA device 和 debug 豁免 |
| `gpu_isolation_matrix` | 第 22 章、第 27 章 | 说明整卡、MIG、time-slicing、vGPU、PCIe passthrough 和容器共享的隔离边界 |
| `data_boundary_policy` | 第 5 章、第 6 章、第 37 章 | 说明 prompt、response、RAG 文档、模型权重、日志、trace、cache 和导出数据的边界 |
| `bmc_driver_access_policy` | 第 27 章、第 28 章 | 说明 BMC、driver、kernel、MIG 配置、节点维护和资源池状态变更的特权边界 |
| `security_audit_event` | 第 28 章、第 37 章、第 40 章 | 说明租户、操作者、策略、资源、时间线、证据和影响面如何进入不可变审计事件 |

## 模型评测与质量闭环覆盖矩阵

模型评测与质量闭环的目标是让读者能从一次模型上线、RAG 答案退化、Agent 工具失败、用户投诉、质量回归或灰度争议，一路追到 eval dataset、quality gate、online experiment、feedback event、regression case、serving release、routing policy、SRE decision 和 Token Factory 质量经济账本。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `eval_dataset_manifest` | 第 13 章、第 2 章、第 3 章 | 说明评测数据来源、任务切片、证据、权限、版本、盲测和污染控制 |
| `quality_gate_record` | 第 13 章、第 14 章、第 40 章 | 说明模型、prompt、RAG、工具、runtime、成本和安全指标如何组成上线门禁 |
| `online_experiment_record` | 第 6 章、第 13 章、第 14 章 | 说明 A/B、canary、流量切分、护栏指标、统计窗口、回滚和影响范围 |
| `quality_feedback_event` | 第 1 章、第 2 章、第 3 章、第 37 章 | 说明用户反馈、人工接管、引用错误、工具失败和投诉如何进入质量事实层 |
| `quality_regression_record` | 第 13 章、第 37 章、第 40 章 | 说明线上事故和评测失败如何沉淀为回归样本、owner、修复策略和复测状态 |
| `serving_quality_contract` | 第 14 章、第 15 章 | 说明 serving release 中 weights、tokenizer、template、engine、参数和质量门禁的绑定关系 |
| `routing_quality_scorecard` | 第 6 章、第 13 章 | 说明 Gateway 如何把质量、安全、成本、延迟和能力用于模型路由，而不是只看健康 |
| `quality_cost_ledger` | 第 41 章 | 说明低质量 token、人工接管、退款、重试、投诉和质量评测成本如何影响毛利 |

## 行业案例与建设方法链路覆盖矩阵

行业案例与建设方法链路的目标是让读者能从一个行业应用想法、商业模式、客户交付承诺或“买 GPU 建平台”的项目，一路追到 workload profile、business model profile、案例证据包、建设计划、架构决策记录、生产就绪评审、验收到上线流水线和 Token Factory/SRE 后果。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `workload_profile` | 第 4 章、第 44 章 | 说明应用交互、上下文、SLO、数据边界、工具副作用、成本口径和资源池如何变成平台输入 |
| `application_readiness_review` | 第 4 章、第 13 章、第 40 章 | 说明行业应用进入生产前如何检查权限、评测、灰度、观测、成本、人工接管和退出条件 |
| `business_model_profile` | 第 42 章、第 41 章 | 说明价值单位、计量事件、客户承诺、成本账本、SLO/SLA、支持模型和退出责任如何结构化 |
| `commercial_readiness_matrix` | 第 42 章、第 44 章 | 说明 MaaS、算力租赁、私有化、推理服务和 Agent 平台上线前必须具备的商业工程能力 |
| `case_study_evidence_pack` | 第 43 章 | 说明公开事实、推断、假设、证据可信度、能力缺口和可复用经验如何分层记录 |
| `ai_factory_maturity_assessment` | 第 43 章、第 44 章 | 说明不同 AI Factory 类型如何按目标层、生产层、运营层和经济层做成熟度诊断 |
| `ai_factory_build_plan` | 第 44 章 | 说明阶段、进入条件、退出条件、owner、证据、风险、停止条件和下一阶段投资如何绑定 |
| `architecture_decision_record` | 第 44 章 | 说明 GPU、网络、存储、调度、推理引擎和商业模式选择如何记录背景、备选方案、取舍、回滚和复审触发器 |
| `production_readiness_review` | 第 44 章、第 38 章、第 40 章 | 说明从资源准入、模型质量、SLO、计量、安全、runbook、成本到发布回滚的上线门禁 |
| `acceptance_to_launch_record` | 第 38 章、第 44 章 | 说明资源、模型、服务和应用如何从 accepted、staging、canary、production 到 rollback/scale 的证据流 |

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
