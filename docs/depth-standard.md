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
| `oci_runtime_injection_diff` | 第 21 章、第 38 章、第 39 章 | 说明 NVIDIA runtime、CDI 或 NRI 对 OCI spec 的实际改动如何被审计和排障 |
| `gpu_device_visibility_reconciliation` | 第 21 章、第 22 章、第 38 章、第 39 章 | 说明 kubelet/device plugin 分配结果、runtime 注入结果和容器内可见 GPU 如何对账 |
| `NVIDIA_VISIBLE_DEVICES`、`NVIDIA_DRIVER_CAPABILITIES` | 第 21 章、第 22 章 | 解释语义、风险、与 Kubernetes request 的关系 |
| Docker、containerd、CRI-O 与 Kubernetes CRI | 第 21 章 | 说明 kubelet 到 CRI 到 OCI runtime 的链路 |
| CDI / NRI 与 legacy hook 模式 | 第 19 章、第 21 章、第 22 章 | 说明 Container Device Interface、Node Resource Interface、RuntimeClass、device list strategy 和验收差异 |
| Kubernetes device plugin 和 GPU 资源分配 | 第 22 章 | 区分“分配 GPU”和“容器内可访问 GPU” |
| `gpu_resource_claim_contract` | 第 22、23、38、44 章 | 说明 extended resource 或 DRA 的 DeviceClass/ResourceClaim 如何与 GPU class、entitlement、runtime、可见性对账和计费口径绑定 |
| `resource_claim_admission_record` | 第 23、38、39、44 章 | 说明 ResourceClaim 或 GPU request 在 queue、quota、DeviceClass、MIG、拓扑、runtime baseline 和成本预算上的准入判断 |
| `resource_claim_acceptance_matrix` | 第 38、44 章 | 说明 extended resource 与 DRA 两种模式下 DeviceClass、ResourceClaim、ResourceSlice、device plugin、CDI、MIG、可见性和计量标签如何验收 |
| `resource_claim_fault_tree_execution` | 第 39、44 章 | 说明 claim pending、错误 GPU class、MIG 越界、拓扑等待和计量标签断链如何执行故障树 |
| `resource_claim_incident_cost_record` | 第 41、44 章 | 说明资源声明事故如何把 pending 容量缺口、错误 class、MIG 边界风险、拓扑等待、fallback 和 billing hold 写入经济账本 |
| `resource_claim_prr_drill` | 第 44 章 | 说明 extended resource 迁移到 DRA、DeviceClass 调整、MIG profile 和 GPU class 映射变更上线前如何演练 claim、分配、可见性、计量和成本 |
| GPU Operator 管理 driver、Toolkit、device plugin、DCGM | 第 19 章、第 22 章 | 说明管理边界、冲突和升级策略 |
| driver、CUDA、NCCL、OFED、Toolkit 兼容矩阵 | 第 19 章、第 29 章 | 说明主机/容器边界、版本矩阵和漂移控制 |
| GPU 容器准入与 smoke test | 第 29 章、第 38 章 | 说明节点入池前如何验证容器内 GPU、NCCL、RDMA |
| 容器内 GPU 与 RDMA/NIC 协同 | 第 22 章、第 32 章、第 38 章 | 说明 device、NIC、NUMA、RDMA、NCCL 的完整路径 |
| `gpu_nic_topology_evidence` | 第 22 章、第 38 章、第 39 章 | 说明多卡和 RDMA 任务中 GPU、NUMA、NIC、rail、NCCL interface 和 switch port 如何绑定成运行事实 |
| `container_runtime_change_record` | 第 21、22、38、40、41、44 章 | 说明 containerd/runc/Toolkit/CDI/NRI/device plugin strategy 变更如何触发复测、PRR 和成本归因 |
| `container_runtime_change_acceptance` | 第 38、44 章 | 说明 runtime 变更后如何验证 host/runtime/Kubernetes、CDI、RuntimeClass、可见性对账、DCGM 标签和 RDMA 拓扑 |
| `gpu_operator_upgrade_evidence` | 第 22、44 章 | 说明 GPU Operator 升级如何证明 Toolkit、device plugin、DCGM exporter、MIG/Runtime 策略和观测 schema 没有破坏生产路径 |
| `container_gpu_runtime_fault_tree_execution` | 第 39、44 章 | 说明 CreateContainerError、CDI 解析失败、可见性错配、MIG/整卡错配和容器 RDMA 问题如何执行故障树 |
| `container_runtime_incident_cost_record` | 第 41、44 章 | 说明容器 runtime 事故如何把失败扩容、可见性错配、拓扑错配、回滚复测和观测断链写入可靠性成本 |
| `container_runtime_prr_change_drill` | 第 44 章 | 说明 Toolkit/CDI/NRI/RuntimeClass/device plugin 变更上线前如何演练非 GPU Pod、单 GPU、MIG、多卡 RDMA、故障树、回滚和成本账本 |

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
| `serving_release_bundle` | 第 14、37、39、41、44 章 | 说明 serving release 如何绑定 provenance、distribution、quality contract、route contract、runtime gate、usage schema、rollback 和成本对象，成为最小可审计发布单元 |
| `serving_route_release_contract` | 第 6、14、37、39、44 章 | 说明 Gateway route、canary、fallback 和 rollback 如何引用具体 release bundle，并验证 capability、usage schema、数据边界和质量门禁兼容 |
| 推理引擎 admission model | 第 15 章 | 说明 input tokens、max output、KV block、deadline 和 queue class |
| continuous batching 与 KV Cache 状态流 | 第 15 章 | 说明 waiting、prefill、active decode、release、usage 的阶段 |
| `engine_request_state_ledger` | 第 15、39、41、44 章 | 说明单个推理请求从 admission、prefill、decode、streaming 到 usage close 的状态、KV 和计量对账 |
| benchmark matrix | 第 15 章 | 覆盖短输入短输出、长输入短输出、短输入长输出和生产混合负载 |
| `endpoint_admission_decision` | 第 6、37、39、44 章 | 说明 Gateway 如何按 request shape、SLO、budget、engine health 和 canary 状态做请求级接入、拒绝、shed、fallback 或路由，并可回放 |
| `engine_admission_health` | 第 6、14、15、37、39 章 | 说明 Gateway 和 Model Serving 如何基于 queue、KV pressure、active sequence、deadline miss 做可路由健康判断 |
| `kv_block_ledger` | 第 1、14、15、37、39、41 章 | 说明 KV block 分配、释放、碎片、租户归属、prefix cache 和取消泄漏如何进入容量与成本 |
| `kv_block_leak_forensic_record` | 第 1、14、37、39、41、44 章 | 说明请求取消、超时、断连、worker restart 或 PD 失败后，KV block 是否泄漏、由谁持有、影响多少 admission 和成本 |
| `engine_canary_record` | 第 14、15、37、39 章 | 说明 engine/runtime 变更如何同时验证协议、质量、延迟、KV、token drift 和成本 |
| `engine_canary_guardrail_action` | 第 14、15、37、39、41、44 章 | 说明 canary 护栏触发后如何冻结、降权、关闭 runtime feature、回滚并保留证据 |
| `speculative_decoding_report` | 第 15、41 章 | 说明 draft/target 模型、接受率、验证开销、质量漂移和真实 cost/token 收益 |
| `speculative_decoding_regression_record` | 第 15、37、39、41、44 章 | 说明 speculative decoding 上线后如何按 workload slice 记录格式、质量、长度、接受率和成本回归，并限制启用范围 |
| `pd_disaggregation_contract` | 第 14、15、37、39 章 | 说明 prefill/decode 分离的 KV 传输、容量比例、失败语义、观测和回滚边界 |
| `pd_transfer_evidence` | 第 14、37、39、41、44 章 | 说明 PD 分离中一次或一段 KV transfer 的时延、完整性、租户隔离、重试和瓶颈归因 |
| `inference_runtime_diagnostic_bundle` | 第 37、39 章 | 说明 TTFT/TPOT/streaming gap 事故如何自动冻结 Gateway、Serving、Runtime、KV、canary 和 PD 证据 |
| `inference_runtime_cost_ledger` | 第 41 章 | 说明 speculative decoding、PD 分离、KV block、取消浪费和质量成本如何共同决定成功回答成本 |
| `inference_runtime_fault_tree_execution` | 第 39、44 章 | 说明推理 runtime 故障树如何记录 admission、prefill、decode、KV、PD、canary、streaming 和 metering 分支判断 |
| `inference_runtime_incident_cost_record` | 第 41、44 章 | 说明推理 runtime 事故如何把未交付 token、取消浪费、KV 泄漏、PD retry、canary rollback 和账单修正写入经济账本 |
| `serving_release_evidence_bundle` | 第 37、39、41、44 章 | 说明发布、fallback、回滚、cache 或 usage schema 事故时如何冻结 release bundle、route contract、quality contract、cache、runtime、metering 和 rollback 证据 |
| `serving_release_fault_tree_execution` | 第 39、44 章 | 说明回滚未恢复、fallback 质量下降、发布后账单漂移或旧产物误用时如何拆分 bundle、route、cache、runtime、rollback 和 metering 分支 |
| `serving_release_cost_record` | 第 41、44 章 | 说明发布组合事故如何把低质量 token、不兼容 fallback、半回滚、cache rewarm、usage replay、billing hold 和客户 credit 写入经济账本 |
| `serving_release_prr_drill` | 第 44 章 | 说明高 SLA endpoint 上线前如何演练 release bundle、Gateway route/fallback、cache、rollback、usage schema、故障树和成本账本 |
| Token Factory ledger | 第 41 章 | 说明 token ledger、resource ledger、阶段成本和毛利约束指标 |

## 多模态与媒体链路覆盖矩阵

多模态链路的目标是让读者能从一次文件、图片、音频或视频请求，一路追到应用画像、媒体上传、对象存储、预处理、派生产物、模型服务、质量门禁、计量、成本、删除和 PRR 演练。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `multimodal_workload_profile` | 第 4、20、44 章 | 说明媒体类型、页数/帧数/音频时长、预处理链路、人工复核、保留策略和计量单位如何进入应用画像 |
| `media_processing_workload` | 第 20 章 | 说明 OCR、ASR、抽帧、layout、embedding 等阶段如何被调度、重试、计量和隔离 |
| `media_artifact_manifest` | 第 33、37、41、44 章 | 说明原始媒体、页面图、OCR/ASR、layout、embedding、权限、保留和派生产物如何形成可追溯事实源 |
| `media_processing_pipeline_record` | 第 33、37、41、44 章 | 说明上传、扫描、解码、渲染、OCR/ASR、layout、embedding 的阶段、runtime、耗时、失败和资源消耗 |
| `multimodal_serving_contract` | 第 14、37、44 章 | 说明多模态 serving 如何绑定 processor、encoder、tile/frame 策略、source region、media manifest、输出引用和 usage 口径 |
| `multimodal_quality_gate_execution` | 第 13、14、37、44 章 | 说明 OCR/ASR fidelity、表格抽取、视觉 grounding、source region citation、隐私脱敏和人工复核如何成为发布门禁 |
| `multimodal_evidence_bundle` | 第 37、44 章 | 说明多模态事故如何冻结 manifest、pipeline record、serving contract、quality gate、metering event 和 retention/delete 证据 |
| `multimodal_metering_event` | 第 41、44 章 | 说明 text token、OCR token、页、tile、音频秒、视频帧、预处理 CPU/GPU 秒、模型 GPU 秒和存储天数如何计量 |
| `multimodal_cost_ledger` | 第 41、44 章 | 说明上传扫描、OCR/ASR、layout/抽帧、encoder、LLM 推理、派生产物存储、失败重试和人工复核如何进入单位经济 |
| `multimodal_prr_drill` | 第 44 章 | 说明文件损坏、超限、OCR 低置信、表格回归、引用坐标错配、删除派生产物和部分失败计量如何在上线前演练 |

## 安全、身份与租户边界覆盖矩阵

安全多租户链路的目标是让读者能从一次请求或一次事故追到身份、租户边界、数据边界、策略决策、provider 外联、trace 脱敏、secret、审计、账单争议和安全成本。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `tenant_boundary` | 第 5、6、44 章 | 说明租户在身份、模型、数据、资源、日志和账单中的边界如何被 Gateway、RAG、Agent、存储、观测和计费共同消费 |
| `tenant_isolation_evidence` | 第 5、37、44 章 | 说明多租户隔离不是声明，而是 identity、目录、策略、资源池、缓存、trace、存储和计量的抽样证据 |
| `credential_lifecycle` / `api_key_audit_event` | 第 5、40、44 章 | 说明 API Key、服务账户和短期凭据如何创建、轮换、禁用、审计和进入事故取证 |
| `policy_decision_record` | 第 6、37、40、44 章 | 说明 Gateway 的 allow、deny、route、fallback、budget、data boundary 和 safety 决策如何回放 |
| `egress_provider_decision` | 第 6、37、41、44 章 | 说明请求是否允许发往第三方 provider、跨区域 endpoint 或私有 provider，以及 provider 合同和数据边界如何约束 fallback |
| `denial_of_wallet_admission_guard` | 第 6、37、41、44 章 | 说明合法 API 请求如何通过 credential risk、request shape、spend velocity 和 business intent 信号识别经济型攻击或误用 |
| `prompt_trace_redaction_record` | 第 8、37、44 章 | 说明 prompt、response、RAG chunk、tool arguments 在 trace、日志和导出中如何脱敏、引用化、设 TTL 和审计 |
| `secret_boundary_evidence` | 第 33、37、41、44 章 | 说明 KMS、provider credential、registry token、签名 key、STS token 和 break-glass token 如何被扫描、挂载、轮换和审计 |
| `security_evidence_bundle` | 第 37、40、41、44 章 | 说明 key 泄露、provider 越权、trace 泄露和安全成本事故如何冻结跨系统证据 |
| `security_policy_fault_tree_execution` | 第 39、44 章 | 说明安全与经济型事故如何按 credential、Gateway policy、provider egress、spend velocity、trace/export 分支执行故障树 |
| `denial_of_wallet_incident_record` | 第 39、40、41、44 章 | 说明 stolen key、长上下文攻击、Agent 循环、昂贵 provider 路由如何造成经济型事故并进入止血和账单处理 |
| `billing_dispute_replay` | 第 7、41、44 章 | 说明账单争议如何从 invoice 回放到 metering event、policy decision、served model、价格版本、hold 和修正 |
| `denial_of_wallet_billing_replay` | 第 7、41 章 | 说明合法凭据下的异常成本如何按客户 key 泄露、平台策略缺口、产品免费额度缺口或未知窗口拆分责任和账单动作 |
| `security_cost_ledger` / `abuse_cost_ledger` | 第 41 章 | 说明隔离、密钥、脱敏、审计、security incident、denial-of-wallet 和争议处理如何进入 secure cost/token |
| `security_prr_abuse_drill` | 第 44 章 | 说明公共入口、外部 provider、免费试用和 Agent 平台上线前如何演练 key 泄露、provider 外联阻断、billing hold 和滥用成本闭环 |

## 训练任务链路覆盖矩阵

训练任务链路的目标是让读者能从一次训练提交一路追到队列、quota、gang、拓扑放置、NCCL、checkpoint、评测、模型注册和训练 ROI。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `TrainingJob` 生产对象 | 第 10 章 | 说明数据、模型、运行时、并行、调度和恢复语义 |
| 训练状态机 | 第 10 章 | 覆盖 submitted、admitted、preflight、rendezvous、running、checkpointing、recovering |
| `launcher_contract` | 第 16 章、第 10 章、第 41 章 | 说明 torchrun/DeepSpeed/Megatron/Slurm/Ray 等 launcher 如何把 placement、rank、env、checkpoint/log path 和失败语义变成可审计契约 |
| `rendezvous_evidence` | 第 10 章、第 16 章、第 41 章 | 说明所有 rank 是否在同一 world size、endpoint、NCCL/env contract 和拓扑契约下完成 process group 初始化 |
| `first_effective_step_record` | 第 10 章、第 37 章、第 41 章 | 说明训练何时真正完成首个数据读取、forward/backward、collective、optimizer step 和指标上报，作为 effective GPU hours 起点 |
| checkpoint manifest | 第 10 章 | 说明 sharded checkpoint、optimizer/scheduler/rng/data loader state 和恢复校验 |
| `framework_runtime_matrix` | 第 16、23、38、44 章 | 说明框架、CUDA/NCCL/driver、launcher、checkpoint 和指标能力如何作为训练 runtime 准入矩阵 |
| `training_runtime_spec` | 第 16、37、39 章 | 说明单个训练任务实际引用的 runtime 矩阵、镜像、精度、分布式策略、checkpoint 和观测配置 |
| `parallelism_plan_record` | 第 17、23、37、39、41、44 章 | 说明并行维度为什么适合模型、batch、拓扑、checkpoint 和训练目标 |
| `rank_topology_contract` | 第 17、23、37、39、44 章 | 说明哪些 rank 拓扑约束是 hard constraint，哪些可降级，以及违反后的调度动作 |
| 并行策略与 rank mapping | 第 17 章 | 说明 rank 到 GPU/NIC/rack/rail/group 的映射和一致性校验 |
| `placement_commit_record` | 第 17、23、37、39、41 章 | 说明并行放置意图、实际放置、降级原因、rank mapping 和性能影响 |
| 并行模板 scorecard | 第 17 章 | 说明扩展效率、通信、显存、稳定性和可恢复性基线 |
| `nccl_env_contract` | 第 18、23、37、39、44 章 | 说明 NCCL 版本、接口、RDMA 设备、拓扑文件、timeout 和 debug 策略如何被平台约束 |
| `collective_trace_record` | 第 18、37、39、41、44 章 | 说明真实训练窗口中 collective op、rank group、耗时、等待关系和 critical path 暴露情况 |
| `communication_critical_path_record` | 第 18、37、39、41 章 | 说明哪些通信等待真正暴露在训练 step 关键路径上，并如何折算 GPU idle |
| `communication_regression_record` | 第 18、32、37、38、39、44 章 | 说明 NCCL、OFED、fabric、runtime 或调度标签变更后，代表性训练是否回归 |
| `checkpoint_overlap_evidence` | 第 18、37、38、39、41、44 章 | 说明 checkpoint 写入是否与 collective、存储和数据路径叠加并造成周期性 step spike |
| communication diagnostic bundle | 第 18 章 | 说明异常 op、rank、节点、NIC、端口、NCCL 环境和 telemetry 的证据包 |
| 通信基线库 | 第 18 章、第 38 章 | 说明 NCCL baseline 与真实训练退化对比 |
| job admission event / pending reason | 第 23 章 | 说明 quota、gang、topology、image、data、checkpoint 和 node baseline 检查 |
| `training_lifecycle_event` | 第 23、37、41 章 | 说明 submitted、admitted、placement committed、gang、rendezvous、first effective step、checkpoint、recovering 的阶段事实 |
| `queue_fairness_ledger` | 第 23、41 章 | 说明 guaranteed、borrowed、lent、preempted、starved 和 effective GPU hours 如何支撑公平运营 |
| checkpoint-aware preemption / `preemption_record` | 第 23、41 章 | 说明训练抢占点、通知、保存状态、恢复结果和浪费 GPU 小时 |
| Slurm 平台同步事件 / `training_accounting_reconciliation` | 第 24、41 章 | 说明 Slurm job/step/accounting 与 experiment、checkpoint、registry、平台成本的统一事件和对账 |
| TrainingJob smoke test | 第 38 章 | 说明 gang、rank mapping、NCCL rendezvous、first effective step 和 checkpoint manifest 验收 |
| `training_communication_acceptance_matrix` | 第 38、44 章 | 说明 framework/runtime、parallelism、rank topology、NCCL、fabric、collective trace 和 checkpoint overlap 的组合验收 |
| `training_lifecycle_telemetry_event` | 第 37 章 | 说明训练生命周期、拓扑、checkpoint 和 GPU 小时口径如何进入观测事实层 |
| `training_debug_bundle` | 第 37、39、44 章 | 说明训练事故如何冻结 runtime、并行、调度、通信、checkpoint、数据和成本影响证据 |
| `training_fault_tree_execution` | 第 39、44 章 | 说明训练事故故障树如何记录分支判断、证据引用、排除项、置信度、动作和证据缺口 |
| `training_incident_record` | 第 39、41 章 | 说明训练事故如何回指 admission、placement、rank mapping、checkpoint、资源健康和 ROI 损失 |
| `training_incident_cost_record` | 第 41、44 章 | 说明训练事故如何把 GPU 小时、checkpoint 回退、队列机会成本、工程响应和模型发布延迟写入 ROI 与 PRR |
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
| `fabric_change_regression_gate` | 第 32、38、44 章 | 说明 fabric 变更如何从配置变更推进到可恢复调度能力，覆盖配置、路径、workload、调度和回滚证据 |
| `fabric_change_acceptance_matrix` | 第 38、44 章 | 说明 host/container/Kubernetes RDMA、NCCL、rail、PFC/ECN/QoS、checkpoint+NCCL 和 scheduler state 如何组合验收 |
| RDMA in container | 第 22 章、第 32 章、第 38 章 | 说明宿主机 RDMA 正常不等于容器内 RDMA 可用 |
| `network_diagnostic_bundle` | 第 32 章、第 39 章 | 说明 rank mapping、NCCL env、RDMA counters、switch ports、baseline 和 verdict |
| `congestion_event_record` | 第 30、37、39 章 | 说明 ECN/PFC、队列、水位、流量类别、job 影响和止血动作如何串联 |
| network telemetry 到业务影响 | 第 37 章 | 说明端口事件如何映射到 job、model、tenant、baseline drift 和 owner |
| fabric acceptance matrix | 第 38 章 | 说明 same rack、cross rack、cross rail、host/container/Kubernetes 的验收矩阵 |
| NCCL hang 网络故障树 | 第 39 章 | 说明 rank 退出、GPU/NVLink、RDMA/fabric、container/runtime、collective mismatch 的排查顺序 |
| `congestion_fault_tree_execution` | 第 39、44 章 | 说明拥塞、rail 失衡、PFC/ECN/QoS、RDMA/NIC、NCCL runtime、checkpoint overlap、混部流量和观测缺口如何执行故障树 |
| `network_cost_ledger` | 第 41 章 | 说明网络退化、拥塞、错误放置和 checkpoint 叠加如何转化为 GPU idle 与 token 成本 |
| `network_incident_cost_record` | 第 41、44 章 | 说明一次 fabric 事故或变更回归如何把通信等待、重启、降级容量、checkpoint 回退、fallback 和复测成本写入经济账本 |
| `fabric_change_prr_drill` | 第 44 章 | 说明 fabric 变更上线前如何演练 baseline 失效、调度降级、拥塞故障树、回滚复测和成本账本 |

## 存储数据链路覆盖矩阵

存储数据链路的目标是让读者能从 GPU idle、checkpoint 慢、模型冷启动或成本异常一路追到 dataset、checkpoint、model artifact、cache、manifest、storage backend、telemetry、准入基线和经济影响。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `dataset_manifest` | 第 10 章、第 20 章、第 33 章 | 说明数据来源、处理版本、shard、checksum、统计、权限和缓存策略 |
| `dataset_lineage_record` | 第 10 章、第 33 章、第 38 章、第 44 章 | 说明训练数据从原始来源、清洗、去重、安全过滤、tokenization、sharding、删除请求到 manifest 的可追溯链路 |
| `workload_storage_intent` | 第 20、33、37 章 | 说明训练、推理、评测和数据处理如何声明 dataset、checkpoint、artifact、cache 和观测需求 |
| `checkpoint_commit_record` | 第 10 章、第 33 章、第 41 章 | 说明分片写入、校验、manifest commit、latest valid、恢复候选和 GPU idle 成本 |
| `checkpoint_restore_drill` | 第 10 章、第 38 章、第 39 章、第 41 章、第 44 章 | 说明 checkpoint 不是存在即可用，必须用真实镜像、并行配置、reader 版本和短窗口训练验证可恢复性 |
| `model_artifact_provenance` | 第 14 章、第 33 章、第 38 章、第 39 章、第 41 章、第 44 章 | 说明模型产物从 checkpoint、adapter、tokenizer、template、转换工具、评测门禁到签名发布的来源证明 |
| `model_artifact_distribution` | 第 14 章、第 33 章 | 说明权重、tokenizer、template、digest、缓存预热和 readiness 的关系 |
| `cache_residency` | 第 14 章、第 33 章、第 41 章 | 说明本地 NVMe、rack cache、权重 cache 与调度、冷启动和成本归因的关系 |
| `cache_invalidation_record` | 第 14 章、第 33 章、第 38 章、第 39 章、第 41 章、第 44 章 | 说明权重、tokenizer、template、RAG 索引和数据缓存撤销后如何阻止调度复用旧缓存，并如何重新预热 |
| `supply_chain_invalidation_evidence` | 第 33、41、44 章 | 说明 artifact、tokenizer、RAG index 或数据撤销后，registry、调度、autoscaler、running replica、local NVMe 和 rack cache 是否真正失效 |
| `storage_security_boundary` | 第 33 章、第 37 章、第 38 章、第 41 章、第 44 章 | 说明训练数据、checkpoint、模型权重、adapter、prompt log、trace 和导出路径的命名空间、权限、加密、审计和删除边界 |
| `data_path_evidence` / `storage_evidence` | 第 33 章、第 37 章、第 39 章、第 41 章 | 说明 workload 到 path、manifest、client、cache、backend、telemetry 和 impact 的证据链 |
| `storage_acceptance_matrix` / `storage_composite_regression_gate` | 第 38 章 | 说明 dataset read、checkpoint write/restore、model load、cache miss 以及 NCCL+checkpoint+artifact 并发门禁 |
| `supply_chain_acceptance_matrix` | 第 38 章、第 44 章 | 说明 dataset lineage、checkpoint restore、artifact provenance、cache invalidation 和 storage security boundary 如何成为生产资源和模型上线门禁 |
| `supply_chain_incident_cost_record` | 第 41、44 章 | 说明供应链撤销、旧缓存误用、tokenizer 口径修正、RAG index 重建、cache rewarm、账单冻结和客户 credit 如何进入经济账本 |
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
| `reliability_evidence` / `reliability_evidence_bundle` | 第 37 章、第 39 章 | 说明 SLO symptom 到 request/job、resource health、baseline、change、fault domain、action 和现场冻结的证据链 |
| `acceptance_baseline` invalidation / `baseline_invalidation_record` | 第 28、29、38、40、44 章 | 说明变更和维护如何使基线失效，如何降级资源池，如何触发影响范围复测，并如何进入 SRE/PRR 门禁 |
| `incident_record` | 第 39 章、第 40 章 | 说明事故时间线、影响面、止血动作、根因证据、成本影响和行动项 |
| `slo_budget_ledger` / `reliability_cost_ledger` | 第 40 章、第 41 章 | 说明 error budget、reliability cost、wasted GPU hours、赔付、容量延迟和毛利之间的关系 |

## 物理设施与能源链路覆盖矩阵

物理设施链路的目标是让读者能从 tokens/W、GPU 降频、机柜降额、液冷告警或扩容不达预期一路追到 GPU server、compute tray、power shelf、rack、cooling domain、capacity unit、acceptance baseline、调度能力和经济影响。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `gpu_server_profile` | 第 34 章、第 28 章 | 说明 CPU/GPU/HBM/PCIe/NVLink/NIC/NVMe/BMC/power/cooling 如何组成可调度服务器画像 |
| `heterogeneous_gpu_pool_profile` | 第 28 章、第 35 章、第 44 章 | 说明多代 GPU、HBM、互联、runtime baseline、准入状态、entitlement 和 workload tier 如何形成可调度异构资源池画像 |
| `gpu_capability_scorecard` | 第 35 章、第 41 章 | 说明芯片能力、runtime 验证、模型适配、能效和生产成熟度的评分口径 |
| `gpu_generation_readiness_gate` | 第 35 章、第 44 章 | 说明新 GPU 或新系统形态进入生产前需要硬件、runtime、质量、能效、回滚和热验证门禁 |
| `model_hardware_fit_record` | 第 35 章、第 38 章、第 44 章 | 说明模型 artifact、精度、context、engine、SLO、HBM、互联和 runtime 如何匹配或拒绝某个 GPU class |
| `gpu_generation_route_decision` | 第 35 章、第 41 章、第 44 章 | 说明 Gateway/Serving 如何按模型硬件匹配、entitlement、健康、质量、SLO、成本和 fallback 选择 GPU class，并保留可回放决策 |
| `heterogeneous_pool_acceptance_matrix` | 第 38 章、第 44 章 | 说明 GPU class × workload slice 的准入矩阵如何输出 pass、limited、canary 或 block，并写回资源池和路由策略 |
| `heterogeneous_gpu_cost_scorecard` | 第 41 章、第 44 章 | 说明不同 GPU class 在同一 workload slice 上的 tokens/s、tokens/W、cost/token、cost per successful answer、质量、SLO 和回滚成本如何比较 |
| `heterogeneous_gpu_prr_drill` | 第 44 章 | 说明异构 GPU 或新代际上线前如何演练 canary 失败、fallback、标签降级、路由回放、质量门禁和成本账本更新 |
| `power_thermal_envelope` | 第 34 章、第 36 章、第 38 章 | 说明功耗、温度、液冷、降额、满载验收和调度限制 |
| `capacity_derating_record` | 第 34 章、第 36 章、第 38 章、第 40 章、第 41 章、第 44 章 | 说明 power/cooling/thermal 风险如何把 allocatable capacity 临时降为 limited，并触发复测和成本归因 |
| `cooling_degradation_record` | 第 36 章、第 40 章、第 41 章、第 44 章 | 说明 cooling domain 退化、液冷/风冷信号、GPU 降频、workload 影响和恢复复测 |
| `rack_capacity_unit` | 第 36 章、第 28 章、第 40 章 | 说明 rack/power/cooling/fabric/storage 如何构成可承诺产能单元 |
| `physical_acceptance_matrix` | 第 38 章 | 说明 power、cooling、cabling、BMC、full-load、thermal soak 和 workload 的准入矩阵 |
| `capacity_activation_record` | 第 36 章、第 40 章、第 41 章 | 说明 planned、installed、bootstrapped、accepted、allocatable、workload-fit、limited 到 retired 的交付状态流转和投产延迟成本 |
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
| `rag_agent_admission_context` | 第 6 章、第 5 章 | 说明 Gateway 如何把入口身份、用户代理、数据边界、检索范围、工具范围和任务预算传给 RAG/Agent 下游 |
| `retrieval_permission_decision` | 第 2 章、第 5 章、第 6 章、第 37 章 | 说明 RAG 检索时哪些文档/chunk 在当前租户、用户、ACL、数据边界和缓存策略下被允许或拒绝 |
| `tool_side_effect_policy` | 第 3 章、第 5 章、第 40 章、第 44 章 | 说明 Agent 工具是否只读、是否有副作用、是否幂等、是否需要确认、是否允许重试和如何回滚 |
| `agent_tool_execution_record` | 第 3 章、第 37 章、第 40 章、第 41 章 | 说明一次工具调用的模型意图、策略决策、参数校验、执行环境、副作用、输出摘要、成本和回滚引用 |
| `tool_security_incident_record` | 第 40 章、第 37 章 | 说明 RAG 越权、Agent 工具越权、敏感数据暴露或高风险工具异常如何冻结证据、止血、评估影响和更新门禁 |
| `bmc_driver_access_policy` | 第 27 章、第 28 章 | 说明 BMC、driver、kernel、MIG 配置、节点维护和资源池状态变更的特权边界 |
| `security_audit_event` | 第 28 章、第 37 章、第 40 章 | 说明租户、操作者、策略、资源、时间线、证据和影响面如何进入不可变审计事件 |

## 模型评测与质量闭环覆盖矩阵

模型评测与质量闭环的目标是让读者能从一次模型上线、RAG 答案退化、Agent 工具失败、用户投诉、质量回归或灰度争议，一路追到 eval dataset、quality gate、online experiment、feedback event、regression case、serving release、routing policy、SRE decision 和 Token Factory 质量经济账本。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `eval_dataset_manifest` | 第 13 章、第 2 章、第 3 章 | 说明评测数据来源、任务切片、证据、权限、版本、盲测和污染控制 |
| `eval_slice_contract` | 第 13 章、第 37 章、第 40 章、第 44 章 | 说明每个业务任务切片为什么存在、最低覆盖要求、硬门禁、owner、业务指标和失败后阻断的发布或路由动作 |
| `eval_dataset_lineage_record` | 第 13 章、第 44 章 | 说明评测数据集版本实际如何生成、脱敏、标注、去重、切片、排除训练污染，并解释分数变化来自模型还是评测口径 |
| `golden_set_governance_record` | 第 13 章、第 37 章、第 40 章、第 44 章 | 说明 golden regression、blind holdout、访问控制、训练排除、overlap scan、样本过期和 judge drift 如何治理，防止门禁样本污染 |
| `eval_contamination_invalidation_record` | 第 13 章、第 37 章、第 40 章、第 41 章、第 44 章 | 说明评测集、golden set 或 holdout 被污染、过期或与训练数据重叠时，如何失效门禁证据、重跑 gate 并阻断高价值发布 |
| `judge_drift_calibration_record` | 第 13 章、第 37 章、第 40 章、第 41 章、第 44 章 | 说明 judge model、rubric 或 parser 升级后如何用人工 anchor 和历史输出校准漂移、重标阈值并决定 gate 是否重跑 |
| `quality_gate_record` | 第 13 章、第 14 章、第 40 章 | 说明模型、prompt、RAG、工具、runtime、成本和安全指标如何组成上线门禁 |
| `quality_gate_execution` | 第 13 章、第 6 章、第 14 章、第 40 章、第 44 章 | 说明某次门禁执行的输入、环境、数据 lineage、judge/rubric、结果、豁免和发布动作，避免只保存静态 pass/fail |
| `online_experiment_record` | 第 6 章、第 13 章、第 14 章 | 说明 A/B、canary、流量切分、护栏指标、统计窗口、回滚和影响范围 |
| `online_experiment_guardrail` | 第 13 章、第 37 章、第 40 章、第 44 章 | 说明线上实验的随机化单元、会话粘性、排除范围、停止规则、自动冻结、证据包和回滚要求 |
| `quality_feedback_event` | 第 1 章、第 2 章、第 3 章、第 37 章 | 说明用户反馈、人工接管、引用错误、工具失败和投诉如何进入质量事实层 |
| `quality_feedback_intake_pipeline` | 第 13 章、第 37 章、第 40 章、第 41 章、第 44 章 | 说明线上反馈如何经过关联、脱敏、triage、replay、人工评审和回归样本治理，避免噪声和不可复现样本污染门禁 |
| `human_feedback_evidence` | 第 13 章、第 37 章、第 40 章、第 41 章、第 44 章 | 说明用户点踩、CRM、人工接管、专家评审和标注如何绑定 trace、task slice、rubric、experiment、regression 和质量成本 |
| `rag_context_snapshot` | 第 2 章、第 13 章、第 37 章 | 说明 RAG 最终进入 prompt 的证据、引用、token 预算、截断原因、冲突处理和无答案策略如何被冻结 |
| `rag_quality_regression_record` | 第 2 章、第 13 章、第 37 章 | 说明 RAG 线上反馈如何绑定权限决策、context 快照、索引版本、失败层级、owner 和复测门禁 |
| `quality_regression_record` | 第 13 章、第 37 章、第 40 章 | 说明线上事故和评测失败如何沉淀为回归样本、owner、修复策略和复测状态 |
| `agent_budget_ledger` | 第 3 章、第 6 章、第 37 章、第 41 章 | 说明 Agent run 的模型调用、工具调用、token、沙箱、外部 API、人工接管和预算控制动作如何计量 |
| `rag_agent_evidence_bundle` | 第 37 章、第 40 章、第 41 章 | 说明 RAG/Agent 事故时如何冻结权限、上下文、工具执行、预算、安全审计和成本证据 |
| `serving_quality_contract` | 第 14 章、第 15 章 | 说明 serving release 中 weights、tokenizer、template、engine、参数和质量门禁的绑定关系 |
| `serving_release_bundle` | 第 14 章、第 37 章、第 39 章、第 41 章、第 44 章 | 说明模型发布组合如何把产物、route、fallback、runtime、usage、cache、rollback 和成本证据打成最小生产单元 |
| `serving_route_release_contract` | 第 6 章、第 14 章、第 39 章、第 44 章 | 说明 Gateway 的 primary、canary、fallback 和 rollback 目标必须引用兼容 release bundle，而不是只引用 endpoint 名称 |
| `serving_rollback_record` | 第 14 章、第 40 章、第 44 章 | 说明一次质量或 runtime 回滚到底回滚了权重、tokenizer、template、engine、Gateway 路由还是配置，并保留事故证据 |
| `serving_rollback_drill` | 第 14 章、第 37 章、第 40 章、第 41 章、第 44 章 | 说明高 SLA endpoint 是否预先演练过权重、tokenizer、template、runtime、Gateway route、cache、drain、计量和质量探针的完整回滚 |
| `serving_release_evidence_bundle` | 第 37 章、第 39 章、第 41 章 | 说明发布事故时如何冻结 release bundle、route contract、cache、runtime、metering、rollback 和质量证据 |
| `serving_release_fault_tree_execution` | 第 39 章、第 44 章 | 说明发布后质量/协议/计费/回滚异常如何按 bundle、route、cache、runtime、rollback 和 billing 分支执行故障树 |
| `serving_release_cost_record` | 第 41 章、第 44 章 | 说明发布事故如何把 fallback 质量损失、半回滚延长事故、cache rewarm、usage replay、billing hold 和客户 credit 计入毛利 |
| `serving_release_prr_drill` | 第 44 章 | 说明 release bundle、Gateway route/fallback、rollback、cache、metering 和成本账本上线前如何演练 |
| `routing_quality_scorecard` | 第 6 章、第 13 章 | 说明 Gateway 如何把质量、安全、成本、延迟和能力用于模型路由，而不是只看健康 |
| `routing_quality_decision_record` | 第 6 章、第 37 章、第 41 章 | 说明某个请求或流量切片为什么选择、拒绝或 fallback 到某个模型，并能回放质量、SLO、成本、能力和数据边界依据 |
| `quality_evidence_bundle` | 第 37 章、第 40 章、第 41 章 | 说明质量事故时如何冻结反馈、人工评审、路由、serving contract、gate execution、slice contract、golden governance、实验护栏、回滚和成本证据 |
| `quality_cost_ledger` | 第 41 章 | 说明低质量 token、人工接管、退款、重试、投诉、评测成本、切片维护、golden set 治理、人工反馈、实验伤害、路由决策和回滚演练如何影响毛利 |
| `rag_agent_cost_attribution` | 第 41 章 | 说明 RAG embedding/search/rerank/context 与 Agent model/tool/sandbox/external API 成本如何归因到每成功答案或任务 |

## 行业案例与建设方法链路覆盖矩阵

行业案例与建设方法链路的目标是让读者能从一个行业应用想法、商业模式、客户交付承诺或“买 GPU 建平台”的项目，一路追到 workload profile、business model profile、案例证据包、建设计划、架构决策记录、生产就绪评审、验收到上线流水线和 Token Factory/SRE 后果。当前覆盖如下：

| 知识点 | 主要章节 | 覆盖要求 |
| --- | --- | --- |
| `workload_profile` | 第 4 章、第 44 章 | 说明应用交互、上下文、SLO、数据边界、工具副作用、成本口径和资源池如何变成平台输入 |
| `application_readiness_review` | 第 4 章、第 13 章、第 40 章 | 说明行业应用进入生产前如何检查权限、评测、灰度、观测、成本、人工接管和退出条件 |
| `customer_onboarding_evidence` | 第 4 章、第 5 章、第 44 章 | 说明客户或关键租户上线前如何证明 tenant、项目、API Key、模型访问、预算、SLA、支持入口和数据边界已经准备好 |
| `business_model_profile` | 第 42 章、第 41 章 | 说明价值单位、计量事件、客户承诺、成本账本、SLO/SLA、支持模型和退出责任如何结构化 |
| `commercial_readiness_matrix` | 第 42 章、第 44 章 | 说明 MaaS、算力租赁、私有化、推理服务和 Agent 平台上线前必须具备的商业工程能力 |
| `sla_credit_model` / `sla_operation_record` / `sla_credit_replay` | 第 5 章、第 7 章、第 40 章、第 41 章、第 44 章 | 说明 SLA 如何从 SLI/SLO/SLA 边界、排除项、incident 证据、credit 计算和 invoice 动作形成可回放赔付链路 |
| `private_deployment_acceptance_record` | 第 42 章、第 44 章 | 说明私有化交付如何验收离线包、版本矩阵、GPU runtime、RAG ACL、计量导出、升级回滚、诊断包和责任矩阵 |
| `release_train_record` | 第 29 章、第 40 章、第 44 章 | 说明 OS、driver、OFED、container runtime、Toolkit、device plugin、base image 和验收脚本如何作为发布列车进入灰度、回滚和基线失效 |
| `lts_support_policy` | 第 29 章、第 42 章、第 44 章 | 说明长期支持 baseline、EOL、backport、升级路径和客户通知如何约束版本分叉 |
| `support_ticket_taxonomy` | 第 40 章、第 42 章、第 44 章 | 说明 incident、request、problem、change、billing dispute 和 security case 如何决定 owner、时钟、证据和升级路径 |
| `diagnostic_bundle_sla` | 第 37 章、第 40 章、第 42 章、第 44 章 | 说明客户支持场景下诊断包的采集时限、脱敏、导出边界、客户同意和留存审计 |
| `offline_release_bundle_manifest` | 第 29 章、第 33 章、第 42 章、第 44 章 | 说明私有化或受限出网环境中 release train 如何被封装成可签名、可导入、可回滚、可验收的离线交付包 |
| `offline_import_record` | 第 33 章、第 37 章、第 42 章、第 44 章 | 说明客户现场实际导入了哪些镜像、模型 artifact、RAG index、chart、配置和 migration，并如何与离线包 digest 对账 |
| `offline_upgrade_rehearsal` | 第 33 章、第 42 章、第 44 章 | 说明私有化或离线环境升级前如何演练镜像导入、artifact/cache、数据迁移、runtime smoke、回滚和诊断导出 |
| `private_delivery_diagnostic_export` | 第 37 章、第 42 章、第 44 章 | 说明不远程登录、不导出生产数据时，私有化事故如何导出脱敏的运行 digest、导入记录、runtime、cache 和 migration 证据 |
| `field_patch_governance` | 第 40 章、第 42 章、第 44 章 | 说明现场紧急补丁的适用条件、签名 delta、客户审批、过期时间、合回 release train 和支持成本归因 |
| `field_patch_execution_record` | 第 40 章、第 41 章、第 44 章 | 说明现场补丁实际应用到哪些镜像、chart、配置、cache 和服务，如何验证、回滚、过期并合回 release train |
| `private_delivery_lifecycle_contract` | 第 42 章、第 44 章 | 说明私有化客户从验收、支持、升级、补丁、EOL 到退出的长期产品责任 |
| `private_delivery_incident_cost_record` | 第 41 章、第 44 章 | 说明私有化升级失败、现场补丁回归或客户环境漂移如何产生支持、离线包重打、迁移回滚、cache 预热、credit 和 P&L 成本 |
| `private_delivery_prr_upgrade_drill` | 第 44 章 | 说明私有化上线前如何演练离线包导入、digest 错配拒绝、artifact 回滚、RAG ACL 迁移、现场补丁和脱敏诊断导出 |
| `commercial_pnl_ledger` | 第 41 章、第 42 章、第 44 章 | 说明 revenue、折扣、free quota、SLA credit、support cost、private delivery cost、reserved capacity 和各类成本账本如何形成商业 P&L |
| `case_study_evidence_pack` | 第 43 章 | 说明公开事实、推断、假设、证据可信度、能力缺口、不可复用条件和可复用经验如何分层记录 |
| `ai_factory_maturity_assessment` | 第 43 章、第 44 章 | 说明不同 AI Factory 类型如何按目标层、生产层、运营层和经济层做成熟度诊断 |
| `ai_factory_build_plan` | 第 44 章 | 说明阶段、进入条件、退出条件、owner、证据、风险、停止条件和下一阶段投资如何绑定 |
| `architecture_decision_record` | 第 44 章 | 说明 GPU、网络、存储、调度、推理引擎和商业模式选择如何记录背景、备选方案、取舍、回滚和复审触发器 |
| `launch_risk_register` | 第 44 章 | 说明商业化上线和关键客户上线前如何记录剩余风险、owner、证据缺口、缓解措施、停止条件和关闭条件 |
| `production_readiness_review` | 第 44 章、第 38 章、第 40 章、第 41 章 | 说明从资源准入、基线有效性、模型质量、SLO、计量、安全、runbook、成本到发布回滚的上线门禁 |
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
