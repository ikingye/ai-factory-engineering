# 第 44 章：从 0 到 1 建设 AI Factory

## 本章回答的问题

- 从零开始建设 AI Factory 应该按什么顺序推进？
- 需求、模型、容量、GPU、网络、存储、调度、推理平台、运维、验收和上线节奏如何互相约束？
- 如何避免“先买 GPU，再补平台”的高成本弯路？

## 一个真实场景

一个企业计划建设 AI Factory，预算已经批准，采购团队准备先下单 GPU。平台团队追问几个问题：第一批要服务哪些应用，是 Chat、RAG、Agent、批量推理、微调还是预训练？目标模型多大，上下文多长，是否需要多模型路由？SLO 是什么，首 token 要多快，失败是否影响客户？数据在哪里，是否能进入云，是否需要私有化隔离？这些问题没有答案时，GPU 型号和数量都无法可靠决定。

基础设施团队又提出另一组问题：机房电力和制冷能支持多少 rack，是否需要液冷，网络是 InfiniBand 还是 RoCE，存储要支撑 checkpoint 还是模型权重加载，准入测试怎么跑，维修回池怎么验收？业务团队则关心上线时间、成本、客户可见功能和安全合规。每个团队的问题都合理，但如果没有统一路线图，建设会变成并行猜测。

最常见的弯路，是先买 GPU，再补网络、存储、调度、推理平台、计量、观测和运维。这样做初期看起来推进快，但后续会发现：GPU 到货后机房承载不足，网络拓扑不适合训练，存储无法支撑 checkpoint，调度系统不了解拓扑，推理服务没有 token 计量，故障时没有基线。硬件越贵，返工越痛。

从 0 到 1 建设 AI Factory 的关键，不是先堆硬件，而是把业务目标、模型路径和基础设施约束连成一张可执行路线图。路线图要能回答：第一阶段服务谁，交付什么能力，验收什么指标，保留哪些扩展边界，哪些能力暂时不做。没有边界的建设会无限膨胀，没有验收的建设无法进入生产。

这个场景也说明，AI Factory 建设不是单纯采购、平台开发或模型部署，而是跨业务、模型、平台、基础设施、SRE 和财务的系统工程。成功的 0 到 1，不是一次性做完所有能力，而是在正确顺序下交付最小可生产系统。

## 核心概念

AI Factory 建设至少包含 11 个决策面：需求分析、模型和业务目标、容量规划、GPU 选型、网络选型、存储选型、调度平台选型、推理平台选型、运维体系、验收标准和上线节奏。这些决策互相约束，不能孤立完成。模型大小影响 GPU 和显存，训练规模影响网络，推理 SLO 影响 batching，数据位置影响存储，商业模式影响计量和租户。

从 0 到 1 的目标不是“一步到位建成最大平台”，而是交付 Minimum Viable AI Factory，也就是最小可生产系统。它至少应具备一个清晰应用入口、一个可服务的模型路径、一个可调度资源池、一个可验收基础设施基线、一个可观测运行路径、一个故障处理流程和一个成本口径。缺少这些能力，系统只能 demo，不能生产。

建设顺序应从需求和模型出发，而不是从 GPU 出发。需求定义服务对象和 SLO，模型定义计算和显存，容量定义资源规模，网络存储定义数据和通信路径，调度定义资源如何被使用，推理和训练平台定义能力如何被消费，运维和验收定义系统如何长期可信。顺序错了，后续会用大量人工弥补架构缺口。

还要区分第一阶段和长期目标。第一阶段可以只支持一个模型、一个内部应用和一个资源池，但租户、计量、观测、验收、版本和升级边界应从一开始就设计。早期可以实现简单，不能没有边界。没有边界的“快速上线”，通常会在第二阶段变成重构。

最后，AI Factory 建设要有退出和纠偏机制。不是所有模型路线、商业模式或硬件选择都会成功。路线图应包含阶段性评审：业务价值是否成立，成本是否可控，SLO 是否达标，扩容是否继续，哪些能力暂停或下线。工程系统需要能学习，而不是把第一版选择锁死。

## 系统架构

从 0 到 1 的架构应按“业务目标到生产能力”的链路设计。业务需求定义应用和用户，模型路径定义使用外部 API、开源模型、微调还是自研模型，容量规划把请求和训练目标转成 GPU、网络、存储和电力，平台能力把模型暴露给应用，调度能力把 workload 放到资源池，运维和验收保证系统可信。

架构设计的第一原则是让每层有明确产物。Application 层产物是应用清单和用户体验目标；Platform 层产物是 API、Gateway、租户、计量和观测；Model 层产物是模型目录、评测和服务策略；Runtime 层产物是推理引擎、训练框架和版本矩阵；Orchestration 层产物是队列、配额、调度和拓扑策略；GPU IaaS、网络存储和物理层产物是可验收资源池。

第二原则是保留关键接口。即使第一阶段只用一个推理引擎，也要保留模型 registry 和 endpoint 语义；即使只有一个租户，也要保留租户标签；即使只部署单集群，也要保留资源池状态；即使暂不收费，也要保留 token 计量。接口边界决定后续扩展成本。

第三原则是从第一天建立基线。GPU burn-in、NCCL test、nvbandwidth、RDMA、storage benchmark、推理 benchmark、训练 smoke test 和 SLO baseline，都是后续故障诊断和扩容验收的参照。没有基线，生产问题会变成口水仗：是新节点慢，还是模型变了，还是网络退化，没人能证明。

第四原则是让架构可分阶段交付。第一阶段可以只开一个资源池、一个模型和一个内部应用，但它们必须走完整生产路径：准入、调度、服务、观测、计量、故障处理和复盘。路径完整比功能数量更重要，因为它证明系统可以生产。

第五原则是把成本和风险放进架构。每个能力都应回答成本来源、故障影响、owner 和验收方式。没有 owner 的组件会在事故中无人负责，没有成本口径的组件会在扩容时失控，没有验收方式的组件无法判断是否可上线。

```mermaid
flowchart TB
  Demand["业务需求 / 应用清单"] --> Model["模型路径 / 评测目标"]
  Demand --> SLO["SLO / 安全 / 合规"]
  Model --> Capacity["容量规划 / 成本模型"]
  SLO --> Platform["MaaS / Gateway / 租户 / 计量"]
  Capacity --> GPU["GPU / 服务器 / 资源池"]
  Capacity --> Network["网络 / RDMA / 拓扑"]
  Capacity --> Storage["对象存储 / PFS / NVMe"]
  GPU --> Orchestration["Kubernetes / Slurm / Queue / Quota"]
  Network --> Orchestration
  Storage --> Orchestration
  Orchestration --> Runtime["推理引擎 / 训练框架"]
  Runtime --> Serving["模型服务 / 训练任务"]
  Serving --> Ops["观测 / SRE / 成本"]
  Ops --> Acceptance["准入 / 验收 / 基线"]
  Acceptance --> Launch["分阶段上线"]
```

## 44.1 需求分析

需求分析要回答服务对象、应用类型、数据边界、SLO、预算、时间线、商业模式和组织能力。不要一开始就问“买多少 GPU”，先问“要生产什么能力”。如果目标是内部 Copilot，重点是数据权限、RAG、推理体验和内部成本；如果目标是大模型训练，重点是数据、网络、checkpoint 和评测；如果目标是 MaaS，重点是 API、计量、SLA 和毛利。

需求应转化为 workload 清单。典型 workload 包括 online inference、batch inference、RAG、Agent、embedding、fine-tuning、evaluation、distributed training、data processing 和 HPC-style job。每类 workload 的资源形态不同：online inference 关注 TTFT、TPOT 和峰值流量；distributed training 关注 gang scheduling、拓扑和 checkpoint；Agent 关注多轮调用、工具权限和任务级 trace。

需求分析还要确定优先级。第一阶段不应同时服务所有场景，除非团队和预算非常充足。更好的方式是选择 1-2 个高价值、边界清晰、可验收的场景作为首批生产目标。例如一个内部 RAG 应用加一个模型 API 服务，比同时建设预训练、MaaS、Agent 平台和私有化交付更可控。

数据边界必须前置。数据能否出域、是否包含敏感信息、是否需要脱敏、RAG 索引如何更新、日志能否保存、prompt 是否可用于诊断，都会影响架构。若数据边界后置，平台上线后可能发现观测、训练和评测都不能合法使用必要数据。

需求分析的输出应是一份可签字的建设输入：workload inventory、用户和租户、目标 SLO、数据和合规约束、成功指标、预算边界、第一阶段范围和明确不做事项。这份输入比采购清单更重要，因为它决定后续所有技术选择。

需求分析还应包含反例。明确哪些场景第一阶段不支持，例如不做基础模型预训练、不支持外部客户、不做跨地域容灾、不支持多模态，能减少范围失控。写清不做事项，是保护交付质量的工程动作。

## 44.2 模型和业务目标

模型路径决定 AI Factory 的技术复杂度。使用外部 API、部署开源模型、微调企业模型、自研基础模型、多模型路由和专属行业模型，对基础设施要求完全不同。外部 API 可以快速验证业务，但数据、安全和成本受限；开源模型需要推理平台；微调需要训练和模型管理；自研基础模型需要完整数据、训练、评测和大规模基础设施。

业务目标必须和模型指标连接。不能只写“模型效果好”，而要定义准确率、拒答策略、安全策略、响应时间、工单解决率、人工接管率、代码接受率、搜索成功率或客户留存等指标。模型评测若不能预测业务结果，训练和上线决策就会失去依据。

模型选择还决定容量和成本。参数量、上下文长度、精度、MoE 结构、多模态能力、reasoning 行为、KV Cache 大小和模型并行策略，都会影响 GPU、显存、网络和推理延迟。一个模型在单机 demo 中可用，不代表能在目标 SLO 下服务生产流量。

模型生命周期也要设计。模型从候选、评测、灰度、上线、回滚到退役，应有清晰流程。第一阶段可以简单，但必须有模型版本、评测记录、服务配置和回滚路径。否则一旦模型升级导致质量或延迟回归，平台无法快速恢复。

模型和业务目标的最终输出，是模型策略文档：候选模型、使用场景、评测口径、SLO 目标、推理成本估算、微调或训练计划、安全边界和上线流程。它是容量规划和平台选型的直接输入。

模型策略也要有淘汰机制。候选模型如果质量、延迟、成本或安全无法达标，应及时退出，而不是继续消耗平台资源。模型路线不是一次选择，而是随业务反馈、硬件能力和成本变化持续调整。

## 44.3 容量规划

容量规划把业务需求转成 GPU、网络、存储、电力和平台容量。推理容量可从请求量、输入输出 token、上下文长度、模型大小、SLO、batching、冗余和增长曲线推导；训练容量可从模型规模、数据量、训练周期、并行策略、checkpoint、失败重试和实验数量推导。容量规划是技术和经济之间的桥。

容量规划不能只给一个峰值数字。应给基线、峰值、增长、冗余、故障保留、实验预算、验收容量和扩容触发条件。在线推理需要应对流量峰值和可用性冗余，离线训练需要大块连续 GPU 和拓扑域，批量推理可以错峰，实验任务可以使用低优或可抢占资源。不同资源池不能简单相加。

还要区分库存容量、可用容量、可调度容量和可交付容量。采购合同里的 GPU、机房已上架的 GPU、通过准入的 GPU、满足拓扑的 GPU、可分配给某租户的 GPU，是不同数字。容量承诺必须使用可交付口径，否则业务会被虚假容量误导。

容量规划应进入成本模型。预计 tokens/s、GPU 小时、cost per token、tokens/W、训练成本、存储增长、网络水位和人力运维，都要有初始估算。估算不必完美，但要透明。后续运行数据应反哺规划，形成滚动预测。

容量规划的输出包括资源池设计、首批规模、扩容阶段、冗余策略、预算消耗、风险假设和监控指标。没有容量规划，采购只是下注；有容量规划，采购才是生产能力投资。

容量规划还要给出失败情景。若流量增长低于预期，资源如何转给批量推理或训练；若增长高于预期，扩容周期能否跟上；若某类 GPU 供应受限，是否有替代模型或降级策略。这些问题决定平台是否具备经营弹性。

## 44.4 GPU 选型

GPU 选型要匹配 workload，而不是追逐单一最高规格。预训练看显存、计算、互联、稳定性和能效；在线推理看显存、HBM、低精度、KV Cache、batching 和成本；微调看显存和调度弹性；embedding、小模型或批量处理不一定需要最高端 GPU。不同资源池可以使用不同 GPU 组合。

GPU 还要与服务器、网络、存储和机房一起选。高端 GPU 如果没有足够电力、制冷、NVLink/NVSwitch、RDMA 网络和存储吞吐，能力无法释放。服务器形态、PCIe 拓扑、NIC 数量、DPU、BMC、power shelf、液冷和维护方式，都影响生产可用性。

生态兼容也重要。driver、CUDA、NCCL、cuDNN、推理引擎、训练框架、容器镜像、GPU Operator 和监控工具，需要支持目标 GPU。新硬件可能性能强，但软件生态和团队经验不足，会增加上线风险。选型应考虑成熟度，而不是只看规格。

采购策略要分阶段。第一批可以用于验证模型、平台、网络和运维能力；第二批再根据真实 tokens/s、训练效率和毛利扩容。一次性买满，若模型路线或商业模式变化，会形成沉没成本。分批采购还能用第一批准入和运行数据优化后续规格。

GPU 选型输出应包含 GPU 型号组合、服务器形态、资源池划分、适用 workload、不适用 workload、版本矩阵、验收基线、功耗和散热要求、采购批次和升级路径。它不是一行型号，而是一组生产约束。

GPU 选型还要考虑故障和维护。高密度服务器维修复杂，液冷节点对设施能力要求高，新架构可能备件和经验不足。选择硬件时，要把维修时间、备件、供应商支持和运维技能纳入总成本。

还要定义替代策略。某型号 GPU 供应不足、价格变化或软件成熟度不达预期时，是否可以用另一类资源承接部分 workload，模型是否能降级，容量是否能分层交付。没有替代策略，选型会把平台锁死。

## 44.5 网络选型

网络选型要区分推理入口网络、训练通信网络、存储网络、管理网络和 BMC 网络。训练集群关注 scale-out 带宽、延迟、RDMA、rail、拓扑、拥塞控制和 collective communication；推理集群关注入口负载均衡、服务发现、streaming 稳定性、权重加载和多区域访问。不同网络服务不同目标。

InfiniBand 和 RoCE 都可用于高性能训练，但运维模型不同。InfiniBand 通常提供成熟 HPC 网络能力和生态，RoCE 依赖以太网配置、PFC/ECN、拥塞控制和更强网络工程纪律。选择时要评估团队经验、现有网络、成本、供应链、可观测性和故障处理能力，而不是只比较带宽。

网络拓扑必须与训练规模和调度策略一致。若目标是大规模分布式训练，需要关注 rack、rail、leaf-spine、oversubscription、fault domain 和 topology-aware scheduling；若主要做推理，可能更关注权重分发、服务入口和跨 AZ 容灾。网络不是后装配件，而是容量规划的一部分。

网络验收要前置。NCCL test、RDMA benchmark、端口计数、拥塞指标、packet loss、PFC/ECN、MTU、GID、容器内 RDMA 和 switch telemetry，都应进入准入。只验证 ping 和 iperf，不足以说明 AI workload 可用。生产故障中的很多“模型慢”，本质是网络基线不完整。

网络选型输出应包含拓扑图、带宽和收敛比、rail 设计、IP/IB 地址规划、CNI 策略、RDMA 配置、可观测性指标、验收基线和扩容路径。网络越早进入设计，GPU 产能越容易兑现。

网络还要设计故障域。一个 leaf、rail、rack 或链路故障会影响多少训练任务和推理副本，是否能通过调度避开，是否有降级路径，都应在拓扑设计阶段回答。故障域不清，事故影响面会不可预测。

网络方案还要给出运维模型。谁看交换机 telemetry，谁处理 RDMA 错误，谁维护配置基线，谁参与 NCCL hang 排障。高性能网络如果没有对应团队能力，会在事故中变成黑盒。

## 44.6 存储选型

存储选型要按数据生命周期分层。对象存储适合源数据、模型 artifact、日志和归档；并行文件系统适合热训练数据、checkpoint 和高吞吐读取；local NVMe 适合缓存、scratch、临时 shard 和模型权重热加载；数据库和向量库服务 RAG、metadata 和平台状态。单一存储无法高效覆盖所有场景。

训练场景要关注 data loader、metadata、checkpoint 写入、恢复速度和并发访问。许多训练效率问题不是 GPU 或网络，而是数据读取、元数据操作或 checkpoint 抖动。容量规划要同时看吞吐、IOPS、metadata、延迟、并发客户端和故障恢复，而不是只看总容量。

推理场景关注模型权重分发、冷启动、缓存命中、版本一致性和扩容速度。大模型权重加载慢，会直接影响 autoscaling 和故障恢复；多模型 serving 还会带来权重缓存和存储热点。RAG 场景则要考虑文档更新、embedding、索引构建、向量库一致性和权限过滤。

存储也有成本治理。checkpoint 保留策略、数据生命周期、热冷分层、对象存储请求成本、重复数据、过期模型和日志保留，都可能成为长期成本。没有生命周期策略，存储会从技术支撑变成成本黑洞。

存储选型输出应包含数据分类、容量增长、吞吐和 IOPS 目标、metadata 要求、checkpoint 策略、模型 artifact 管理、缓存策略、备份恢复、权限、加密、成本模型和验收 benchmark。存储不是“买够容量”，而是保障数据路径稳定。

存储还要定义数据责任。训练数据、模型权重、checkpoint、RAG 索引、日志和计费事件的 owner、保留期、删除流程和恢复目标不同。没有数据责任，存储会逐渐变成无人敢删、无人能恢复的堆积系统。

存储方案还应包含迁移路径。数据规模增长、模型权重增多或 checkpoint 策略变化后，如何扩容、分层、归档和清理，应在早期设计中保留接口。否则第一版存储很快会成为平台瓶颈。

## 44.7 调度平台选型

调度平台选型要把 Kubernetes、Slurm、Volcano、Kueue、Ray、Kubeflow 和 Argo 的边界说清楚。Kubernetes 适合服务化、容器生态、平台扩展和在线推理；Slurm 适合 HPC 风格训练、成熟队列和批作业管理；Volcano/Kueue 补充 Kubernetes 的 queue、quota、gang scheduling；Ray 适合分布式 Python、数据处理和 Agent/AI 应用执行。

不要把调度简单归为 PaaS 或 IaaS。它位于资源编排与作业调度层，负责把 workload、GPU、拓扑、配额、队列、优先级和租户策略连接起来。调度层设计不清，GPU IaaS 交付了资源，MaaS 和训练平台却无法稳定使用。

选型要从 workload 组合出发。在线推理需要 deployment、autoscaling、service discovery 和滚动发布；大训练需要 gang scheduling、拓扑感知、checkpoint 和长任务稳定性；微调和评测需要队列、公平共享和镜像管理；数据处理需要弹性和任务编排。单一调度器不一定覆盖所有场景。

调度还要考虑组织经验。已有 Kubernetes 团队可以在 Kueue/Volcano 上增强训练调度；已有 HPC 团队可以用 Slurm 承载预训练，再与模型 registry 和推理平台集成。选型不是技术名词比赛，而是看团队能否长期运维、排障和升级。

调度平台输出应包含 workload 映射、队列和配额模型、资源池边界、GPU 和 RDMA 设备管理、拓扑感知策略、抢占和恢复、租户隔离、观测指标和与 MaaS/训练平台的接口。调度层是 AI Factory 的交通系统，必须在早期设计清楚。

调度选型还要设计用户体验。用户需要知道任务为什么 pending、预计等待多久、是否可抢占、失败是否可恢复、配额由谁管理。没有这些解释能力，调度系统即使技术正确，也会被用户认为“不可靠”。

调度平台还要与成本系统连接。不同队列、优先级和资源池对应不同成本，用户提交任务时应能理解资源代价。这样调度不只是资源分配器，也是成本治理入口。

## 44.8 推理平台选型

推理平台要支持模型加载、endpoint、replica、batching、streaming、autoscaling、模型路由、canary、rollback、token 计量、观测和安全策略。推理引擎可以选择 vLLM、SGLang、TensorRT-LLM 或其它方案，关键是与模型、硬件、SLO、团队能力和商业模式匹配。

选型前要定义服务口径。是否提供 OpenAI-compatible API，是否支持内部 SDK，是否需要专属 endpoint，是否支持 batch inference、PD 分离、多模型 serving、长上下文、reasoning 模型、多模态和工具调用。服务口径决定网关、路由、计量、缓存和扩缩容能力。

推理平台的核心不是跑通一个模型，而是管理模型生命周期。模型注册、版本、评测、灰度、回滚、路由、弃用、兼容期和安全策略，都要进入平台。没有生命周期管理，模型升级会变成高风险手工操作，客户应用也无法稳定依赖。

验收应使用真实 prompt 分布，而不只是固定 benchmark。长上下文、短请求、高并发、流式输出、冷启动、错误处理、限流、重试、缓存命中和模型切换都要覆盖。固定 benchmark 可以做对比，但不能替代生产流量模拟。

推理平台输出应包括 API 形态、模型目录、引擎选择、部署拓扑、SLO、token 计量、billing 接口、观测指标、灰度回滚策略、容量模型和成本报表。推理平台是 Token Factory 的主生产线，必须从工程和经济两侧同时设计。

推理平台还要处理多租户公平性。高价值生产租户、内部测试、批量推理和低优实验不应共享同一限流和资源策略。路由、队列、配额和缓存都要理解租户等级，否则成本优化可能损害关键体验。

推理平台还应预留评测入口。模型上线前后的质量、安全、延迟和成本对比，需要与模型服务连接。没有评测入口，模型发布只能靠人工判断，无法形成可靠生命周期。

## 44.9 运维体系

运维体系包括可观测性、告警、oncall、incident、变更、升级、容量、成本、资产、准入、runbook 和故障演练。AI Factory 的运维对象比普通 Web 服务更多：GPU、驱动、CUDA、NCCL、RDMA、NVLink、checkpoint、模型质量、推理延迟、token 计量和机房状态都要进入视野。

从 0 到 1 阶段，至少要建立统一标签体系。tenant、model、job、pod、node、GPU、NIC、rack、resource pool 和 cost center 必须能关联。没有统一标签，观测、计量、故障诊断和成本分摊都会变成手工拼表。标签体系是运维自动化的地基。

核心 dashboard 应覆盖推理、训练、资源池、网络、存储和成本。推理看 TTFT、TPOT、tokens/s、错误率和 KV Cache；训练看队列、step time、NCCL、checkpoint 和失败原因；资源池看 GPU health、Xid、ECC、维修和准入；成本看 cost/token、GPU hour 和浪费。dashboard 要支持从结果反查原因。

变更和升级流程要尽早建立。driver、CUDA、NCCL、OFED、Kubernetes、GPU Operator、推理引擎、模型版本和网络配置都可能引发事故。即使第一阶段团队很小，也需要变更记录、灰度、停止条件和回滚方案。没有变更纪律，事故复盘会缺少最关键线索。

运维体系的输出应包括 oncall 规则、告警分级、incident 模板、变更模板、升级矩阵、runbook、诊断包、容量例会、成本例会和复盘机制。没有运维体系的 AI Factory 可以演示，不能长期生产。

运维体系还要演练。NCCL hang、GPU Xid、推理延迟尖刺、存储慢、网络丢包、模型回滚和计费异常，都应有演练场景。没有演练的 runbook，在事故中很可能不可用或权限不足。

运维体系还应包含资产状态。每张 GPU、每台服务器、每个 NIC、每个模型版本和每个资源池，都要有 owner、状态和历史。资产状态不清，会让容量、维修和故障诊断同时失真。

## 44.10 验收标准

验收标准要覆盖硬件、软件、网络、存储、调度、推理、训练、安全和成本。验收不是项目最后一天的形式，而是资源进入生产池的门禁。新节点入池、维修回池、驱动升级、网络调整、存储扩容和模型服务重大变更，都应触发相应范围的验收。

最低验收包括 GPU burn-in、nvbandwidth、HPL 或计算压力测试、NCCL test、RDMA/network benchmark、storage benchmark、镜像/驱动版本检查、推理 benchmark、训练 smoke test、故障演练和观测指标。不同 workload 还要补充专门测试，例如长上下文推理、checkpoint restore、容器内 RDMA 和多租户隔离。

验收必须形成 baseline。一次测试通过不够，测试结果要写入资源池，作为后续异常检测、升级回归、维修回池和故障诊断的比较对象。没有 baseline，生产中看到性能下降时无法判断是资源退化、模型变化、流量变化还是测试口径不同。

验收还要定义失败处理。硬失败直接阻止入池，例如 GPU Xid、NCCL hang、RDMA 不通；软偏离可以进入 limited 状态，例如性能低于同组但可跑低优任务。验收结果应改变资源状态，而不是只生成报告。资源状态再反馈调度和容量承诺。

验收标准的输出包括测试清单、阈值、拓扑覆盖、运行环境、数据保留、失败分级、回归策略和 owner。验收越标准，扩容越可预测；验收越随意，生产任务越会变成昂贵的测试工具。

验收还要和采购、交付、运维联动。供应商交付、机房上架、平台入池、业务上线和维修回池都应引用同一套基线。否则每个环节各测各的，最终没有一份证据能说明资源是否适合生产。

验收标准也要定期修订。生产事故暴露的新故障，应补入验收；长期无效且昂贵的测试，可以降低频率。验收不是一次性文档，而是生产经验的沉淀机制。

## 44.11 上线节奏

上线节奏应分阶段：设计验证、验收环境、小规模生产、核心业务灰度、多租户扩展、成本优化和规模化运营。不要第一天就把所有业务迁入新平台，也不要在没有计量、观测和回滚的情况下开放外部客户。AI Factory 的复杂度需要组织学习曲线。

每个阶段都应有进入条件和退出条件。设计验证阶段要求模型和 workload 明确；验收环境要求基础设施基线通过；小规模生产要求一个模型和一个应用稳定运行；核心业务灰度要求 SLO、oncall 和回滚可用；多租户扩展要求租户、配额、计量和账单可用；规模化运营要求容量、成本和 SRE 节奏稳定。

上线节奏还要保护用户体验。早期只开放少数内部用户，收集 prompt、延迟、错误和成本数据；再逐步扩展到关键业务；最后才对外或私有化复制。每一步都要有回滚和降级策略。没有回滚的上线，不是上线，是赌注。

组织也要随阶段演进。第一阶段可能由少数平台工程师和模型工程师推进；进入生产后，需要 SRE、网络、存储、安全、财务和客户支持加入。若组织能力跟不上技术复杂度，系统会依赖少数专家，无法稳定扩张。

上线节奏的输出是一份阶段计划：每阶段目标、范围、进入条件、退出条件、指标、风险、回滚方案、owner 和预算。分阶段不是拖慢，而是让系统在真实压力下逐步变强。

阶段计划还要包含停止条件。若 SLO 连续不达标、成本超出假设、故障无法诊断或关键业务价值未验证，应暂停扩容或延后对外开放。能停下来纠偏，是成熟建设节奏的一部分。

## 工程实现

工程实现可以从一份建设计划开始。计划不是愿景文档，而是可执行 backlog。第一部分是目标和边界：服务哪些 workload，不服务哪些 workload，第一阶段支持哪些模型和租户。第二部分是能力清单：平台、模型、调度、GPU、网络、存储、观测、验收和成本。第三部分是阶段里程碑和验收指标。

```yaml
ai_factory_plan:
  phase_0_design:
    outputs:
      - workload_inventory
      - model_strategy
      - target_slo
      - capacity_model
      - cost_model
  phase_1_foundation:
    outputs:
      - gpu_resource_pool
      - network_storage_baseline
      - acceptance_pipeline
      - observability_labels
  phase_2_platform:
    outputs:
      - maas_api
      - inference_serving
      - job_queue
      - tenant_quota
      - token_metering
  phase_3_production:
    outputs:
      - sre_runbook
      - billing_dashboard
      - change_management
      - cost_per_token_report
```

生产级建设计划应更接近 `ai_factory_build_plan`。它不仅列输出物，还要列每阶段进入条件、退出条件、owner、证据、风险、停止条件和下一阶段投资依据。这样计划才能被 SRE、平台、基础设施、财务和业务共同使用，而不是只被项目经理跟进进度。

```yaml
ai_factory_build_plan:
  id: build-ai-factory-2026-h1
  objective: support_internal_rag_and_maas_pilot
  scope:
    included_workloads:
      - online_inference
      - rag
      - evaluation
      - limited_fine_tuning
    excluded_workloads:
      - foundation_model_pretraining
      - external_sla_maas
      - multi_region_disaster_recovery
  phase_0_design:
    entry_criteria:
      - executive_sponsor_assigned
      - first_two_workload_profiles_drafted
    exit_criteria:
      - workload_profiles_approved
      - business_model_profile_for_pilot_approved
      - capacity_model_reviewed
      - data_boundary_policy_approved
    evidence:
      - workload_profile
      - business_model_profile
      - capacity_model
      - risk_register
    stop_conditions:
      - no_clear_value_unit
      - data_boundary_unresolved
  phase_1_resource_foundation:
    entry_criteria:
      - facility_power_and_cooling_confirmed
      - gpu_network_storage_decisions_recorded
    exit_criteria:
      - gpu_resource_pool_accepted
      - fabric_baseline_recorded
      - storage_acceptance_matrix_passed
      - observability_labels_available
    owner:
      - infra
      - network
      - storage
      - sre
    stop_conditions:
      - acceptance_failure_without_remediation
      - rack_capacity_unit_not_workload_fit
  phase_2_platform_path:
    entry_criteria:
      - accepted_resource_pool_available
      - model_strategy_approved
    exit_criteria:
      - maas_api_smoke_passed
      - token_metering_append_only
      - model_serving_canary_ready
      - job_queue_and_quota_ready
    owner:
      - ai_platform
      - model_serving
      - sre
    stop_conditions:
      - no_request_trace_to_cost_ledger
      - no_rollback_for_serving_release
  phase_3_production_pilot:
    entry_criteria:
      - production_readiness_review_passed
      - oncall_and_runbook_ready
    exit_criteria:
      - pilot_slo_met_for_review_window
      - cost_per_successful_task_within_budget
      - incident_actions_closed
      - next_scale_decision_recorded
    owner:
      - business_owner
      - platform_owner
      - sre_owner
    stop_conditions:
      - sev1_without_root_cause
      - quality_gate_regression
      - sustained_cost_overrun
```

这个计划有两个重要特点。第一，它把第 4 章的 `workload_profile` 和第 42 章的 `business_model_profile` 放到建设入口，而不是等平台做好后再补。第二，它允许每个阶段停止。AI Factory 建设中最危险的不是慢，而是在价值单位、数据边界或准入基线没成立时继续扩容。停止条件是治理能力，不是失败姿态。

关键技术选择还应写入 `architecture_decision_record`。ADR 不需要长，但必须能回答：为什么选这个，拒绝了什么，承担什么风险，什么时候复审，如何回滚。GPU、网络、存储、调度、推理引擎和商业模式都应有 ADR。没有 ADR，半年后团队只会记得“当时大家觉得这样好”，无法判断环境变化后是否该调整。

```yaml
architecture_decision_record:
  id: adr-007-inference-runtime-vllm-first
  status: accepted
  decision_area: inference_runtime
  context:
    workload_profiles:
      - wp-internal-rag-v2
      - wp-maas-chat-pilot-v1
    constraints:
      - openai_compatible_streaming_required
      - limited_runtime_engineering_team
      - need_fast_iteration_before_external_sla
  options:
    - name: vllm_first
      strengths:
        - fast_model_iteration
        - continuous_batching_support
        - broad_model_support
      risks:
        - engine_upgrade_may_change_token_behavior
        - advanced_optimization_requires_benchmark
    - name: tensorrt_llm_first
      strengths:
        - strong_optimized_serving_path
      risks:
        - higher_build_and_conversion_complexity_for_pilot
  decision: vllm_first_for_phase_2
  guardrails:
    - runtime_quality_gate_required_for_engine_upgrade
    - benchmark_matrix_required_before_scale
    - serving_quality_contract_binds_engine_version
  revisit_triggers:
    - cost_per_token_above_budget
    - model_family_requires_different_engine
    - external_sla_requires_lower_tail_latency
  rollback:
    - keep_previous_serving_release_warm
    - route_canary_tenants_back_to_stable_endpoint
```

ADR 的价值在于把工程取舍变成可复审证据。比如第一阶段为了速度选择 Kubernetes + vLLM，不代表未来不能引入 TensorRT-LLM 或 Slurm；选择 RoCE 不代表忽略拥塞控制和 telemetry；选择统一推理池不代表高价值客户永远不能专属容量。ADR 让“阶段性正确”不会变成“永久惯性”。

上线前应通过 `production_readiness_review`。它把资源、模型、平台、安全、SRE、成本和商业承诺放在同一张门禁里。生产就绪不是“服务能访问”，而是当它慢、错、贵、坏、被滥用或需要回滚时，团队有证据和动作。

```yaml
production_readiness_review:
  id: prr-maas-chat-pilot-2026-06
  release_scope:
    workload_profile: wp-maas-chat-pilot-v1
    business_model_profile: bmp-internal-maas-pilot-v1
    serving_release: sr-chat-model-a-v3
    resource_pool: gpu-inference-shared-prod
  gates:
    resource:
      acceptance_baseline: pass
      capacity_reserve: pilot_only
      fault_domain_review: pass
    model:
      quality_gate_record: pass
      safety_eval: pass
      rollback_model: sr-chat-model-a-v2
    platform:
      gateway_policy: pass
      token_metering: append_only_enabled
      request_trace: end_to_end
    security:
      tenant_boundary: pass
      data_boundary_policy: pass
      api_key_lifecycle: pass
    sre:
      dashboard: ready
      runbook: ready
      oncall: ready
      incident_template: ready
    economics:
      cost_model: reviewed
      budget_owner: assigned
      stop_condition: cost_per_successful_answer_over_budget
    commercial:
      customer_onboarding_evidence: ready_if_external_or_key_customer
      sla_credit_model: ready_if_sla_committed
      private_deployment_acceptance_record: ready_if_private_delivery
      commercial_pnl_ledger: initialized_if_commercial
      launch_risk_register: reviewed
  decision: approve_canary
  canary_plan:
    traffic: 5_percent_internal_tenants
    review_window: 7d
    rollback_conditions:
      - ttft_slo_breach
      - quality_regression_record_open
      - metering_gap_detected
```

`production_readiness_review` 应和第 38 章的验收基线、第 40 章的 SRE 流程、第 41 章的经济账本连接。它不是独立审批系统，而是把已有证据聚合成上线决策。若某项证据缺失，结论应该是 `block` 或 `conditional_approve`，并明确条件，而不是口头放行。

```mermaid
stateDiagram-v2
  [*] --> Draft: release scope defined
  Draft --> EvidenceCollecting: workload + business + resource scope
  EvidenceCollecting --> Blocked: required evidence missing
  Blocked --> EvidenceCollecting: evidence gap closed
  EvidenceCollecting --> ConditionalApprove: non-critical risk accepted with owner
  EvidenceCollecting --> ApproveCanary: all hard gates pass
  ConditionalApprove --> Canary: stop conditions machine-enforced
  ApproveCanary --> Canary
  Canary --> Scale: SLO + quality + cost + support stable
  Canary --> Rollback: guardrail breach
  Scale --> Review: drift / incident / cost overrun
  Review --> EvidenceCollecting: PRR re-opened
  Rollback --> EvidenceCollecting: rollback evidence + new gate
```

这张状态机把 PRR 从一次会议变成持续控制流程。`Blocked` 不是失败，而是证据缺口尚未关闭；`ConditionalApprove` 必须绑定 owner、停止条件和观察窗口；`Canary` 不是上线完成，而是继续收集 SLO、质量、成本和支持证据；一旦 drift、incident 或 cost overrun 出现，PRR 应重新打开。AI Factory 的生产就绪不是某一天获得批准，而是在运行中持续证明。

商业化上线还应维护 `launch_risk_register`。PRR 判断“当前证据是否足以放行”，risk register 则持续记录“哪些风险仍然存在、谁负责、何时停止、如何验证关闭”。它尤其适合处理不能简单二元判断的风险：客户支持成本尚无历史数据，SLA credit 口径刚建立，私有化客户环境存在特殊 DNS/证书限制，首批客户流量形态可能偏离 workload profile，或者商业折扣导致毛利缓冲较薄。没有风险登记，conditional approve 往往变成口头放行。

```yaml
launch_risk_register:
  id: lrr-maas-premium-enterprise-a-202606
  release_scope:
    production_readiness_review: prr-maas-chat-prod-2026-06
    customer_onboarding_evidence: coe-enterprise-a-support-rag-202606
    business_model_profile: bmp-enterprise-maas-standard-v2
  risks:
    - risk_id: lrr-001
      description: premium_sla_has_limited_real_incident_history
      owner: ai-sre
      evidence_gap: no_prior_sla_credit_replay_for_this_tier
      mitigation:
        - canary_first_traffic_window
        - enhanced_reliability_evidence_bundle_sampling
        - daily_slo_budget_review
      stop_condition: ttft_or_streaming_slo_breach_above_policy
      close_condition: two_review_windows_without_creditable_sla_event
    - risk_id: lrr-002
      description: private_rag_customer_environment_differs_from_reference
      owner: delivery-platform
      evidence_gap: dns_proxy_and_offline_registry_variance
      mitigation:
        - private_deployment_acceptance_record_required
        - upgrade_and_rollback_drill_before_scale
      stop_condition: diagnostic_bundle_export_failed
      close_condition: acceptance_and_upgrade_drill_passed
    - risk_id: lrr-003
      description: launch_margin_sensitive_to_support_cost
      owner: ai-business-ops
      evidence_gap: insufficient_support_cost_baseline
      mitigation:
        - initialize_commercial_pnl_ledger
        - tag_support_tickets_by_tenant_and_product
      stop_condition: support_cost_exceeds_margin_buffer
      close_condition: pnl_review_confirms_margin_after_support_cost
  review_cadence: twice_per_week_during_canary
  escalation:
    unresolved_high_risk: block_scale
    missing_owner: block_launch
```

这个对象把“风险可接受”变成可执行承诺。每个风险必须有 owner、证据缺口、缓解措施、停止条件和关闭条件；否则它不是风险管理，而是愿望列表。它还把商业风险和工程风险放在一起：SLA 经验不足会影响流量放量，私有化环境差异会影响交付节奏，支持成本缺口会影响毛利。AI Factory 上线的风险从来不只在技术栈里，也在客户承诺、交付边界和经济模型里。

成熟的 PRR 还应检查“证据是否仍然有效”。很多上线事故不是完全没有验收，而是验收基线已经被 driver、fabric、存储、模型 runtime 或维护动作失效；不是没有容量，而是 capacity activation 只到 installed，没有到 workload-fit；不是没有可观测性，而是缺少事故触发时能冻结的 `reliability_evidence_bundle`。因此 PRR 应把证据有效性作为一等门禁：

```yaml
production_readiness_review:
  id: prr-maas-chat-prod-2026-06
  evidence_validity:
    acceptance_baselines:
      status: valid
      invalidation_records_open: none
      required_scopes: [resource_pool, fabric, container_gpu_runtime, storage]
    capacity_activation:
      capacity_activation_record: dc-a-rack-12-2026-06
      workload_fit_capacity: sufficient_for_canary
      limiting_factors_acknowledged: true
      rack_capacity_unit: workload_fit
      physical_acceptance_matrix: pass
      open_capacity_derating_records: none_for_required_scope
      open_cooling_degradation_records: none_for_required_scope
      gpu_generation_readiness_gate: pass_if_new_gpu_generation
      heterogeneous_gpu_pool_profile: required_if_multiple_gpu_classes
      heterogeneous_pool_acceptance_matrix: pass_for_required_workload_slices
      model_hardware_fit_record: required_for_target_models
      gpu_generation_route_decision: dry_run_passed_if_route_across_gpu_classes
    change_safety:
      recent_high_risk_changes: reviewed
      canary_stop_conditions: machine_enforced
      rollback_drill: passed
      release_train_record: pass_if_platform_or_baseline_release
      lts_support_policy: pass_if_customer_or_long_lived_pool
      field_patch_governance: pass_if_out_of_band_patch_exists
    observability:
      reliability_evidence_bundle_trigger: configured
      inference_runtime_diagnostic_bundle: configured
      security_evidence_bundle_trigger: configured
      diagnostic_bundle_sla: configured_if_external_customer_or_private_delivery
      prompt_trace_redaction_record: pass
      token_metering_reconciliation: pass
    security_and_tenant_boundary:
      credential_lifecycle: pass
      api_key_audit_event_stream: configured
      tenant_isolation_evidence: pass
      policy_decision_record_replay: pass
      egress_provider_decision_replay: pass_if_external_or_cross_region_provider
      secret_boundary_evidence: pass_if_secret_or_provider_involved
      denial_of_wallet_runbook: ready
    inference_runtime:
      endpoint_admission_decision_replay: pass
      engine_admission_health_freshness: pass
      engine_request_state_ledger: configured
      kv_block_ledger_rollup: configured
      kv_block_leak_forensic_record_template: ready
      pd_transfer_evidence: pass_if_pd_enabled
      speculative_decoding_regression_guardrail: pass_if_speculative_enabled
      engine_canary_guardrail_action: machine_enforced
      inference_runtime_fault_tree_execution_dry_run: pass
      inference_runtime_incident_cost_record_pipeline: configured
    serving_release:
      serving_release_bundle: pass_for_target_endpoint
      serving_route_release_contract: pass
      release_bundle_integrity_check: pass
      fallback_release_compatibility: pass_if_fallback_enabled
      rollback_bundle_warm_capacity: pass_if_high_sla
      serving_release_evidence_bundle_trigger: configured
      serving_release_fault_tree_execution_dry_run: pass
      serving_release_cost_record_pipeline: configured
    container_gpu_runtime:
      gpu_resource_claim_contract: pass_if_gpu_or_dra_resource_used
      resource_claim_acceptance_matrix: pass_if_gpu_or_dra_resource_used
      resource_claim_admission_record_replay: pass_if_gpu_or_dra_resource_used
      container_gpu_runtime_acceptance_matrix: pass
      oci_runtime_injection_diff: sampled_and_clean
      gpu_device_visibility_reconciliation: pass
      gpu_nic_topology_evidence: pass_if_rdma_or_multigpu
      runtime_privilege_profile: enforced
      container_runtime_change_record: reviewed_if_recent
    training_runtime_and_communication:
      framework_runtime_matrix: pass_if_training_or_model_release_from_training
      parallelism_plan_record: reviewed_if_distributed_training
      rank_topology_contract: enforced_if_distributed_training
      placement_commit_record: present_if_training_job_executed
      nccl_env_contract: pass_if_nccl_or_rdma
      training_communication_acceptance_matrix: pass_for_large_training_pool
      collective_trace_record: configured_for_high_value_training
      communication_regression_record: pass_if_recent_runtime_or_fabric_change
      checkpoint_overlap_evidence: required_if_checkpoint_heavy_training
      training_debug_bundle_template: ready
      training_debug_bundle_trigger: configured_for_hang_loss_spike_checkpoint_slow
      training_fault_tree_execution_dry_run: pass
      training_incident_cost_record_pipeline: configured
      first_effective_step_record: required_for_training_roi
    quality:
      quality_gate_execution: qge-af-chat-20260620-001
      eval_slice_contract: esc-support-202606
      eval_dataset_lineage_record: edl-support-quality-20260620
      golden_set_governance_record: gsg-support-202606
      eval_contamination_invalidation_record: none_open
      judge_drift_calibration_record: pass_if_judge_or_rubric_changed
      quality_feedback_intake_pipeline: configured
      routing_quality_scorecard: rqs-20260619-support
      serving_rollback_record_template: ready
      serving_rollback_drill: pass_for_target_endpoint
      quality_evidence_bundle_trigger: configured
      online_experiment_guardrail: oeg-support-20260620
      human_feedback_evidence_pipeline: configured
      rag_agent_evidence_bundle_trigger: configured_if_applicable
      retrieval_permission_decision_replay: pass_if_rag
      rag_context_snapshot_replay: pass_if_rag
      tool_side_effect_policy: approved_if_agent
      agent_tool_execution_record_template: ready_if_agent
      agent_budget_ledger: initialized_if_agent
      multimodal_quality_gate_execution: pass_if_multimodal
      multimodal_serving_contract: pass_if_multimodal
      multimodal_evidence_bundle_trigger: configured_if_multimodal
    data_and_artifact_supply_chain:
      dataset_lineage_record: required_for_training_or_rag
      checkpoint_restore_drill: required_for_model_release_from_training
      model_artifact_provenance: required_for_serving_release
      cache_invalidation_record_replay: pass_for_release_and_rollback
      supply_chain_invalidation_evidence: pass_if_artifact_tokenizer_index_or_data_recalled
      supply_chain_incident_cost_record_pipeline: configured
      storage_security_boundary: valid_for_sensitive_data
      supply_chain_acceptance_matrix: pass_for_production_scope
      media_artifact_manifest: required_if_multimodal
      media_processing_pipeline_record: required_if_multimodal
      derived_media_delete_replay: pass_if_multimodal_sensitive_data
    sre_and_economics:
      slo_budget_ledger: initialized
      reliability_cost_ledger: initialized
      energy_ledger: initialized_if_power_or_cooling_relevant
      heterogeneous_gpu_cost_scorecard: required_if_route_across_gpu_classes
      resource_claim_incident_cost_record_pipeline: configured_if_gpu_or_dra_resource_used
      serving_release_cost_record_pipeline: configured_if_serving_release_or_fallback_used
      quality_cost_ledger: initialized
      multimodal_metering_event: configured_if_multimodal
      multimodal_cost_ledger: initialized_if_multimodal
      multimodal_prr_drill: pass_if_multimodal
      security_cost_ledger: initialized
      billing_dispute_replay: ready
      abuse_cost_ledger: initialized_if_public_or_untrusted_access
      commercial_pnl_ledger: initialized_if_external_customer_or_chargeback
      sla_credit_model: ready_if_sla_committed
      customer_onboarding_evidence: pass_if_external_customer_or_key_internal_tenant
      private_deployment_acceptance_record: pass_if_private_delivery
      private_delivery_lifecycle_contract: pass_if_private_delivery
      offline_release_bundle_manifest: pass_if_private_delivery_or_restricted_egress
      offline_import_record: pass_if_private_delivery_upgrade_or_install
      offline_upgrade_rehearsal: pass_if_private_delivery_or_restricted_egress
      private_delivery_incident_cost_record_pipeline: configured_if_private_delivery
      support_ticket_taxonomy: configured_if_external_customer
      launch_risk_register: reviewed_and_owned
      owner_for_error_budget_burn: assigned
  decision_logic:
    block_if:
      - open_baseline_invalidation_for_required_scope
      - no_workload_fit_capacity
      - rack_capacity_unit_not_workload_fit_for_required_scope
      - physical_acceptance_matrix_not_passed_for_target_pool
      - open_capacity_derating_record_for_required_capacity
      - open_cooling_degradation_record_for_required_capacity
      - new_gpu_generation_without_readiness_gate
      - heterogeneous_pool_without_pool_profile
      - heterogeneous_pool_without_acceptance_matrix_for_required_slice
      - model_without_hardware_fit_record
      - route_across_gpu_classes_without_replayable_decision
      - heterogeneous_route_without_cost_scorecard
      - no_rollback_path
      - platform_release_without_release_train_record
      - long_lived_customer_pool_without_lts_policy
      - out_of_band_patch_without_field_patch_governance
      - private_delivery_without_offline_upgrade_rehearsal
      - private_delivery_without_offline_release_bundle_manifest
      - private_delivery_import_without_digest_reconciliation
      - external_customer_without_support_ticket_taxonomy
      - external_customer_without_diagnostic_bundle_sla
      - no_metering_reconciliation
      - no_incident_owner_or_runbook
      - no_credential_lifecycle_or_api_key_audit
      - no_tenant_isolation_evidence_for_multitenant_scope
      - no_policy_decision_replay_for_gateway_security_policy
      - external_provider_without_egress_provider_decision
      - prompt_or_trace_logging_without_redaction_record
      - secret_or_provider_scope_without_secret_boundary_evidence
      - public_or_untrusted_access_without_denial_of_wallet_runbook
      - commercial_billing_without_dispute_replay
      - customer_launch_without_onboarding_evidence
      - committed_sla_without_credit_model
      - private_delivery_without_acceptance_record
      - private_delivery_without_incident_cost_pipeline
      - commercial_launch_without_pnl_ledger
      - open_launch_risk_without_owner_or_stop_condition
      - no_endpoint_admission_decision_replay
      - stale_or_missing_engine_admission_health
      - engine_request_state_ledger_missing
      - no_kv_block_ledger_for_target_endpoint
      - recent_kv_incident_without_leak_forensic_record
      - pd_enabled_without_transfer_evidence
      - speculative_enabled_without_regression_guardrail
      - recent_engine_canary_guardrail_failure_unresolved
      - inference_runtime_fault_tree_dry_run_failed
      - inference_runtime_incident_cost_record_pipeline_missing
      - serving_release_bundle_missing_or_invalid
      - serving_route_release_contract_missing
      - fallback_release_without_capability_usage_or_boundary_compatibility
      - high_sla_serving_without_warm_rollback_bundle
      - serving_release_evidence_bundle_trigger_missing
      - serving_release_fault_tree_dry_run_failed
      - serving_release_cost_record_pipeline_missing
      - gpu_resource_without_claim_contract
      - resource_claim_acceptance_matrix_not_passed
      - resource_claim_admission_replay_failed
      - gpu_or_dra_claim_without_cost_record_pipeline
      - no_container_gpu_runtime_acceptance_for_target_pool
      - gpu_device_visibility_reconciliation_failed
      - rdma_or_multigpu_without_gpu_nic_topology_evidence
      - recent_container_runtime_change_without_retest
      - distributed_training_without_framework_runtime_matrix
      - distributed_training_without_parallelism_plan_record
      - rank_topology_contract_not_enforced_for_large_training
      - nccl_env_contract_missing_for_rdma_training
      - recent_fabric_or_nccl_change_without_communication_regression_record
      - large_training_pool_without_training_communication_acceptance_matrix
      - checkpoint_heavy_training_without_checkpoint_overlap_evidence
      - training_pool_without_debug_bundle_trigger
      - training_fault_tree_dry_run_failed
      - training_incident_cost_record_pipeline_missing
      - no_valid_quality_gate_execution
      - no_eval_slice_contract_for_target_task
      - no_eval_dataset_lineage_for_required_task_slices
      - stale_or_contaminated_golden_set
      - open_eval_contamination_invalidation_record
      - judge_or_rubric_change_without_calibration
      - feedback_intake_pipeline_without_trace_replay
      - online_experiment_without_guardrail
      - high_sla_serving_without_recent_rollback_drill
      - human_feedback_not_linked_to_regression_or_gate
      - no_quality_rollback_or_freeze_path
      - rag_without_permission_or_context_replay
      - agent_without_tool_side_effect_policy
      - agent_without_budget_ledger
      - multimodal_without_media_artifact_manifest
      - multimodal_without_processing_pipeline_record
      - multimodal_without_quality_gate_or_serving_contract
      - multimodal_without_source_region_replay
      - multimodal_sensitive_data_without_derived_delete_replay
      - multimodal_without_metering_event_or_cost_ledger
      - multimodal_without_prr_drill
      - no_dataset_lineage_for_training_or_rag
      - checkpoint_without_restore_drill
      - artifact_without_provenance_or_signature
      - invalid_cache_not_blocked_from_scheduling
      - supply_chain_invalidation_without_propagation_evidence
      - supply_chain_incident_cost_record_pipeline_missing
      - missing_storage_security_boundary_for_sensitive_data
      - supply_chain_acceptance_matrix_not_passed
    conditional_approve_if:
      - limited_capacity_with_explicit_canary_scope
      - noncritical_observability_gap_with_due_date
```

这份门禁会迫使上线讨论从“服务能不能访问”转为“证据是否足以承受生产风险”。例如资源池有 GPU，但 `baseline_invalidation_record` 仍然 open，就只能批准单节点低风险 canary，不能批准 premium inference；模型质量门禁通过，但评测集没有 lineage、没有 `eval_slice_contract` 或没有覆盖目标 task slice，就不能进入高价值租户；golden set 被污染或过期，门禁分数就不能作为生产证据；线上实验没有 guardrail，就不能把真实客户当作无边界试验场；高 SLA endpoint 没有近期 `serving_rollback_drill`，就不能承诺快速回滚；训练产物没有 `dataset_lineage_record`、`checkpoint_restore_drill` 和 `model_artifact_provenance`，就不能证明模型来自被批准的数据、可恢复 checkpoint 和合格转换链路；缓存撤销不能回放，就不能保证旧 tokenizer、旧权重或旧 RAG 索引已离开生产路径；RAG 没有权限决策回放和 context 快照，就不能接入敏感知识库；Agent 没有工具副作用策略和预算账本，就不能自动执行有副作用动作；token 计量未对账，就不能进入商业化计费；容量激活记录显示 cooling_limited，就不能承诺持续满载训练。PRR 的价值在于把这些限制提前暴露，而不是等事故后再解释。

容量投产也应有专门的 PRR 演练：`capacity_activation_prr_drill`。它验证的不是某台服务器能不能跑 benchmark，而是一批资源从 planned 到 workload-fit 的整条链路是否可证明、可降级、可恢复、可入账。演练应覆盖三类失败：物理资源已安装但 thermal soak 或 PDU 冗余未通过；运行中 cooling degradation 导致 rack 降额；容量系统仍把 limited rack 计入销售或训练排期。

```yaml
capacity_activation_prr_drill:
  drill_id: cap-prr-drill-20260620-001
  production_readiness_review: prr-training-prod-2026-06
  scope:
    capacity_activation_record: dc-a-rack-12-2026-06
    rack_capacity_unit: dc-a-rack-12
    workload_slices:
      - premium_inference
      - large_distributed_training
      - checkpoint_heavy_training
  injected_or_simulated_failures:
    - pdu_redundancy_lost_before_launch
    - cooling_limited_during_soak
    - thermal_throttle_above_policy
    - workload_fit_capacity_below_commitment
    - scheduler_label_not_removed_after_derating
    - reservation_system_uses_installed_gpu_not_workload_fit_gpu
  required_outputs:
    physical_capacity_activation_matrix: generated
    facility_capacity_evidence_bundle: generated
    workload_fit_capacity_gate: pass_or_block_recorded
    capacity_derating_record: generated_if_derating_injected
    energy_ledger: updated
    capacity_activation_cost_record: generated
    capacity_commitment_guard: blocks_or_conditions_recorded
  pass_criteria:
    installed_capacity_not_treated_as_sellable_capacity: true
    large_training_blocked_when_thermal_soak_missing: true
    scheduler_labels_follow_derating_state: true
    reservation_commitment_uses_workload_fit_capacity: true
    recovery_requires_full_load_retest: true
    delayed_capacity_cost_recorded: true
```

```mermaid
flowchart LR
  Drill["capacity_activation_prr_drill"] --> Matrix["physical_capacity_activation_matrix"]
  Matrix --> Gate["workload_fit_capacity_gate"]
  Gate --> Guard["capacity_commitment_guard"]
  Matrix --> Bundle["facility_capacity_evidence_bundle"]
  Bundle --> Energy["energy_ledger"]
  Energy --> Cost["capacity_activation_cost_record"]
  Guard --> PRR["PRR capacity gate"]
  Cost --> PRR
```

这类演练能拦截“纸面产能”事故。GPU 已经在资产系统里，采购和设施都认为交付完成，但对业务真正重要的是可承载目标 workload 的产能：能不能持续满载、能不能跨 rack 通信、能不能 checkpoint、能不能在 SLO 下生产 token、能不能在故障或降额时自动限制承诺。PRR 不应接受 installed GPU 作为产能证据；它应要求 workload-fit gate、降额回放、能效账本和投产成本记录共同通过。

异构 GPU 或新代际资源池上线前，应增加 `heterogeneous_gpu_prr_drill`。它的目标不是证明新 GPU benchmark 漂亮，而是证明模型硬件匹配、资源池 entitlement、异构验收矩阵、路由决策、fallback、质量门禁和经济账本能在一个受控窗口内闭环。演练应至少覆盖三类失败：新 GPU canary 触发 runtime 或质量护栏；长上下文请求从成熟池迁移到高 HBM 池后成本或 TPOT 偏离；某类 GPU 因 thermal derating 或 baseline invalidation 被降级后，Gateway 和调度器是否停止把目标 workload 路由过去。

```yaml
heterogeneous_gpu_prr_drill:
  drill_id: hgpu-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    heterogeneous_gpu_pool_profile: gpu-prod-mixed-202606
    target_models:
      - code-assistant-large
    gpu_classes:
      - h100-full-card-prod
      - h200-long-context-prod
      - b200-canary
  injected_scenarios:
    - b200_runtime_error_rate_above_baseline
    - h200_long_context_tpot_regression
    - h100_pool_capacity_derating
    - model_precision_profile_not_validated_on_candidate_class
  required_evidence:
    heterogeneous_pool_acceptance_matrix: hpa-mixed-prod-202606
    model_hardware_fit_record: mhf-code-assistant-20260620
    gpu_generation_route_decision: replayed_for_each_scenario
    heterogeneous_gpu_cost_scorecard: hgc-code-assistant-20260620
    quality_gate_execution: pass_or_block_by_slice
    energy_ledger: collected_for_compared_classes
  pass_criteria:
    canary_route_disabled_when_guardrail_fails: true
    fallback_to_mature_pool_verified: true
    premium_tenant_not_routed_to_unapproved_class: true
    schedulable_labels_removed_when_matrix_invalidated: true
    cost_scorecard_updated_after_route_change: true
    production_readiness_review: updated_if_gap_found
```

```mermaid
flowchart LR
  Drill["heterogeneous_gpu_prr_drill"] --> Pool["heterogeneous_gpu_pool_profile"]
  Pool --> Matrix["heterogeneous_pool_acceptance_matrix"]
  Matrix --> Fit["model_hardware_fit_record"]
  Fit --> Route["gpu_generation_route_decision"]
  Route --> Serving["serving route / fallback"]
  Serving --> Quality["quality + SLO gate"]
  Serving --> Energy["energy_ledger"]
  Quality --> Cost["heterogeneous_gpu_cost_scorecard"]
  Energy --> Cost
  Cost --> PRR["PRR gate update"]
```

这类演练能避免多代 GPU 上线时最常见的三种错觉。第一种是“新 GPU 跑分高，所以可以接高价值流量”；真正需要证明的是目标模型、目标精度、目标 context、目标 runtime 和目标租户都被批准。第二种是“路由有 fallback，所以风险可控”；真正需要证明的是 fallback 后不会突破成熟池容量、不会错计费、不会让质量门禁失效。第三种是“异构池提高利用率”；真正需要证明的是它降低了 `cost_per_successful_answer`，而不是把低质量、重试、回滚和支持成本藏进平均 cost/token。

训练资源池的 PRR 还应做一次故障树演练。演练不需要真的破坏生产任务，但必须能用一条受控 smoke training job 触发 `training_debug_bundle`，执行 `training_fault_tree_execution`，生成 `training_incident_record` 和 `training_incident_cost_record` 的演练版本。若 bundle 缺少 rank mapping、NCCL env、checkpoint overlap 或资源健康记录，说明事故时无法定位；若故障树只能输出 unknown 且没有 evidence gaps，说明 runbook 没有表达不确定性；若成本记录无法写入 `training_roi_ledger`，说明技术事故和经济账本仍然割裂。

```yaml
training_prr_failure_drill:
  drill_id: train-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    resource_pool: h100-rdma-prod
    workload: smoke_distributed_training
    injected_or_simulated_symptoms:
      - controlled_rank_exit
      - checkpoint_slow_window
      - nccl_env_mismatch_dry_run
  required_outputs:
    training_debug_bundle: generated
    training_fault_tree_execution: generated
    training_incident_record: generated
    training_incident_cost_record: generated
    prr_gate_update: generated_if_gap_found
  pass_criteria:
    evidence_completeness: above_policy
    unsafe_actions_blocked: true
    rollback_or_checkpoint_recovery_decision_recorded: true
    cost_ledger_append_verified: true
    evidence_redaction_policy_applied: true
```

这个演练能把“训练平台可观测”变成可验收事实。很多团队在建设阶段能展示 dashboard，却无法在事故发生时拿到 rank、容器注入、RDMA、checkpoint 和成本的同一条证据链；也有团队能定位技术根因，却无法说明浪费了多少 GPU 小时、是否影响模型发布日期、是否应该暂停扩容。PRR 要求演练这些路径，是为了避免昂贵训练任务成为第一次真实集成测试。

在 runtime 事故演练之前，还应做 `serving_release_prr_drill`。它验证的不是某个引擎指标，而是一个 release bundle 是否能被 Gateway 正确路由、被 fallback 正确降级、被 rollback 完整恢复、被 cache 和 usage schema 正确对账。很多上线事故来自“单项都通过，组合不一致”：权重和 tokenizer 通过了，Gateway fallback 却指向不兼容 release；quality gate 通过了，usage schema 没进入计费；rollback drill 通过了镜像回退，但没有恢复 route、template 和 cache。PRR 必须让这些组合问题在受控窗口暴露。

```yaml
serving_release_prr_drill:
  drill_id: sr-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    endpoint: af-chat-large-prod
    candidate_bundle: srb-af-chat-large-20260619-r3
    baseline_bundle: srb-af-chat-large-20260612-r9
    gateway_contract: srrc-af-chat-large-20260620
    task_slices:
      - short_chat
      - rag_citation
      - tool_call_json
      - long_context_streaming
  injected_or_simulated_scenarios:
    - candidate_tokenizer_digest_mismatch
    - fallback_target_lacks_tool_call_schema
    - usage_schema_changed_without_metering_replay
    - cache_residency_contains_invalid_template
    - rollback_route_restored_but_engine_config_not_restored
  required_outputs:
    serving_release_evidence_bundle: generated
    serving_release_fault_tree_execution: generated_by_controlled_failure
    endpoint_admission_decision: replayed_for_primary_canary_fallback
    serving_rollback_record: generated_for_rollback_scenario
    serving_release_cost_record: generated_if_failure_injected
    billing_dispute_replay: opened_if_usage_schema_unclear
    prr_gate_update: generated_if_gap_found
  pass_criteria:
    incompatible_fallback_blocked_before_first_token: true
    full_bundle_rollback_restores_quality_protocol_and_usage: true
    invalid_cache_prevents_new_replica_or_route: true
    metering_hold_applied_when_usage_schema_unknown: true
    route_decisions_reference_release_contract: true
    cost_ledger_append_verified: true
```

```mermaid
flowchart LR
  Drill["serving_release_prr_drill"] --> Bundle["serving_release_bundle"]
  Drill --> Route["serving_route_release_contract"]
  Bundle --> Evidence["serving_release_evidence_bundle"]
  Route --> Evidence
  Evidence --> Fault["serving_release_fault_tree_execution"]
  Fault --> Rollback["serving_rollback_record"]
  Fault --> Cost["serving_release_cost_record"]
  Fault --> Billing["billing hold / replay"]
  Cost --> PRR["PRR serving release gate"]
```

这个演练让“可以回滚”和“可以正确回滚”分开。正确回滚至少要恢复 release bundle 中声明的 weights、tokenizer、template、runtime profile、Gateway route、cache 状态、usage schema 和 drain 语义；如果业务只做局部 route 回退，也必须证明剩余组合仍与质量契约一致。对高 SLA endpoint，缺少这类 drill 时，PRR 不应接受口头承诺，因为事故窗口内很难再临时确认每个组件是否等价。

在线推理资源池也需要 runtime 事故演练。演练应覆盖一条真实请求从 Gateway admission 到 engine request state、KV block、streaming close、metering close 和成本记录的闭环。常见演练包括客户端取消、长上下文 KV pressure、PD transfer timeout、engine canary guardrail breach 和 usage close drift。目标不是制造复杂事故，而是证明系统能在短窗口内冻结证据、执行故障树、止血并对账。

```yaml
inference_runtime_prr_failure_drill:
  drill_id: inf-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    endpoint: af-chat-large-prod
    engine_profile: vllm-prod-h100-v7
    workload_slices: [short_chat, long_context_qa, streaming_generation]
    injected_or_simulated_symptoms:
      - client_cancel_after_prefill
      - kv_block_release_delay
      - pd_transfer_timeout_if_enabled
      - engine_canary_guardrail_breach
  required_outputs:
    inference_runtime_diagnostic_bundle: generated
    engine_request_state_ledger: generated
    inference_runtime_fault_tree_execution: generated
    inference_runtime_incident_cost_record: generated
    metering_replay: generated
    prr_gate_update: generated_if_gap_found
  pass_criteria:
    generated_vs_delivered_tokens_reconciled: true
    kv_blocks_released_or_leak_recorded: true
    unsafe_full_endpoint_rollback_avoided_if_slice_action_sufficient: true
    billing_hold_or_correction_recorded_if_needed: true
    cost_ledger_append_verified: true
```

这类演练能拦截推理平台最贵的隐性问题：用户已经断开但引擎继续 decode，usage 事件与实际交付不一致，KV block 泄漏让 endpoint 看似 ready 却无法接新请求，PD 分离失败后 prefill 成本无人归属，canary 自动回滚但账本没有记录 prevention cost。若这些路径没有演练，runtime 优化越多，事故越难解释。

数据和模型产物供应链也要演练“撤销”。PRR 不能只检查 dataset lineage、checkpoint restore 和 artifact provenance 是否存在，还要验证某个对象被撤销后，旧 cache、旧 release、旧 RAG index 和调度状态是否同步失效。供应链撤销演练尤其适合在 tokenizer 修复、模型 artifact 召回、RAG 权限策略更新、数据删除请求和私有化离线升级前执行。

```yaml
supply_chain_prr_invalidation_drill:
  drill_id: sc-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    serving_release: af-chat-large-20260619-r3
    resource_pool: inference-premium-a
    cache_scopes: [local_nvme, rack_cache, registry_pointer]
    optional_rag_index: kb-index-20260618.3
  simulated_or_controlled_events:
    - tokenizer_digest_recall
    - model_artifact_signature_revoke
    - rag_index_permission_rebuild_if_applicable
  required_outputs:
    cache_invalidation_record: generated
    supply_chain_invalidation_evidence: generated
    storage_security_boundary_check: pass
    scheduler_blocks_invalid_cache_nodes: verified
    serving_trace_replay_uses_replacement_artifact: verified
    supply_chain_incident_cost_record: generated
  pass_criteria:
    no_new_replica_uses_invalid_cache: true
    running_replicas_drained_or_verified: true
    replacement_cache_ready_before_scale: true
    billing_hold_or_replay_recorded_if_tokenizer_changed: true
    prr_gate_update_generated_if_gap_found: true
```

这个演练的价值在于暴露“控制面成功、数据面失败”的风险。Registry 指针已经更新，但本地 NVMe 仍保留旧权重；RAG 索引权限已经修复，但 rack cache 仍可命中旧索引；tokenizer digest 已撤销，但某些 warmed replica 仍使用旧模板；调度器看到节点 cache ready，却不知道它是 invalid ready。AI Factory 的供应链安全不是只签名和登记来源，还要能撤销、阻断、预热替代版本并计算影响成本。

私有化交付还需要单独的 PRR 演练。原因是客户现场的失败模式与公有云不同：出网受限、registry 私有、证书和 KMS 由客户控制、远程登录受审批、诊断导出需要脱敏、现场维护窗口有限，且客户可能在供应方不知情的情况下修改配置或替换镜像。PRR 不能只检查 `private_deployment_acceptance_record` 是否存在，还要证明离线发布包能被导入、digest 能对账、升级能回滚、现场补丁受治理、诊断包能导出、成本能入账。

```yaml
private_delivery_prr_upgrade_drill:
  drill_id: pd-prr-upg-drill-20260620-001
  production_readiness_review: prr-enterprise-a-private-2026-06
  scope:
    lifecycle_contract: pdlc-enterprise-a-202606
    customer_rings: [customer_staging, customer_production]
    restricted_egress: true
    supported_workloads: [maas_chat, rag, finetuning_optional]
  simulated_or_controlled_events:
    - offline_bundle_import_to_empty_registry
    - digest_mismatch_rejected
    - model_artifact_upgrade_and_rollback
    - rag_index_acl_migration_if_applicable
    - emergency_field_patch_apply_and_expire
    - redacted_diagnostic_export_without_remote_login
  required_outputs:
    offline_release_bundle_manifest: generated
    offline_import_record: generated
    offline_upgrade_rehearsal: pass
    private_delivery_diagnostic_export: generated
    field_patch_execution_record: generated_if_patch_path_tested
    private_delivery_incident_cost_record: generated_for_failure_path
    production_readiness_review: updated_if_gap_found
  pass_criteria:
    imported_digests_match_bundle_manifest: true
    unsupported_manual_delta_detected: true
    rollback_restores_service_and_artifact_pointer: true
    cache_residency_matches_target_release: true
    diagnostic_export_contains_no_prompt_or_secret: true
    support_cost_and_credit_path_recorded: true
```

```mermaid
flowchart TB
  PRR["private delivery PRR"] --> Bundle["offline_release_bundle_manifest"]
  Bundle --> Import["offline_import_record\nregistry / artifact / chart / config"]
  Import --> Rehearsal["offline_upgrade_rehearsal\nsmoke / migration / rollback"]
  Rehearsal --> Patch["field_patch_execution_record\nif emergency path"]
  Rehearsal --> Export["private_delivery_diagnostic_export"]
  Rehearsal --> Cost["private_delivery_incident_cost_record"]
  Patch --> Cost
  Export --> Support["support_ticket_taxonomy\ncustomer communication"]
  Cost --> PNL["commercial_pnl_ledger"]
  Cost --> Gate["PRR gate update\nblock / conditional / approve"]
```

这类演练尤其适合在第一批私有化客户上线前执行。它会暴露很多“云内看不见”的问题：导入工具依赖公网包源，chart overlay 不可验证，模型权重签名在客户 KMS 下无法校验，RAG index 迁移没有 ACL snapshot，回滚只回滚服务不回滚 cache，诊断包导出包含 prompt 或 secret，现场补丁没有到期和合回机制。把这些问题留到客户生产事故中解决，代价通常远高于在 PRR 阶段修掉。

安全和滥用也需要 PRR 演练。公共 MaaS、外部客户、免费试用、第三方 provider 聚合、Agent 平台和高敏企业租户，都不能只检查认证和 TLS。PRR 必须证明：异常 key 能被发现和冻结，provider 外联能被策略证明或阻断，长上下文/长输出/Agent 循环能被预算和形态 guard 降级，异常 usage 能进入 billing hold，事故证据能进入 `security_evidence_bundle`，成本能进入 `abuse_cost_ledger`，并且策略缺口会回写 Gateway 和上线门禁。

```yaml
security_prr_abuse_drill:
  drill_id: sec-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    endpoint: maas-public-chat
    tenants: [public-trial, enterprise-a]
    providers: [third-party-x]
    workload_slices: [long_context_chat, agent_run, provider_fallback]
  simulated_or_controlled_events:
    - stolen_key_source_asn_change
    - free_quota_long_context_burst
    - external_provider_fallback_candidate_for_sensitive_prompt
    - agent_loop_with_repeated_tool_failures
  required_outputs:
    policy_decision_record: generated
    egress_provider_decision: generated_for_provider_candidate
    denial_of_wallet_admission_guard: triggered_or_proven_not_triggered
    security_evidence_bundle: generated
    security_policy_fault_tree_execution: generated
    denial_of_wallet_incident_record: generated_if_cost_attack_simulated
    billing_dispute_replay: opened_if_chargeability_unclear
    abuse_cost_ledger: appended
    prr_gate_update: generated_if_gap_found
  pass_criteria:
    external_provider_route_blocked_if_data_boundary_forbids: true
    suspicious_key_frozen_before_budget_exhaustion: true
    provider_fallback_disabled_for_untrusted_or_sensitive_scope: true
    billing_hold_marks_suspicious_usage: true
    no_sensitive_prompt_copied_into_security_bundle: true
    cost_ledger_append_verified: true
    guardrail_update_owner_assigned_if_gap_found: true
```

这类演练能拦截两种常见误判。第一种是“认证通过就是合法流量”：stolen key 使用的仍是合法 API Key，但来源、形态和成本速度已经异常。第二种是“provider fallback 提高可用性”：如果数据边界、合同、日志和训练使用策略不允许外联，可用性不能凌驾于边界之上。PRR 要求演练这些路径，是为了证明平台能在真实成本和真实风险出现前止血，而不是事后靠人工查账。

```mermaid
flowchart LR
  Drill["security_prr_abuse_drill"] --> Gateway["Gateway guard\nidentity / policy / budget"]
  Gateway --> PDR["policy_decision_record"]
  Gateway --> EPD["egress_provider_decision"]
  Gateway --> Guard["denial_of_wallet_admission_guard"]
  PDR --> Bundle["security_evidence_bundle"]
  EPD --> Bundle
  Guard --> Bundle
  Bundle --> Fault["security_policy_fault_tree_execution"]
  Fault --> DOW["denial_of_wallet_incident_record"]
  DOW --> Abuse["abuse_cost_ledger"]
  DOW --> Replay["billing_dispute_replay"]
  Abuse --> PRR["PRR gate update"]
  Replay --> PRR
```

安全 PRR 的通过标准应避免两个极端。一个极端是把所有 provider 外联都禁止，牺牲多模型聚合和成本优化；另一个极端是把所有外联都交给业务配置，平台无法证明边界。更合理的做法是把 provider、区域、数据等级、日志策略、训练使用策略、价格、fallback 目标和客户披露写成 `egress_provider_decision`，让允许和拒绝都能回放。denial-of-wallet 也类似：不是所有成本突增都要封禁，而是要根据身份、历史、审批和风险分层采取降级、人工确认、冻结、hold 或放行。

GPU 资源声明也需要 PRR 演练。任何从 extended resource 迁移到 DRA、调整 DeviceClass、修改 ResourceClaimTemplate、切换 MIG profile、改变 GPU class 映射、调整 queue quota 或 entitlement 的动作，都可能让用户意图、Kubernetes claim、device plugin 分配、CDI/runtime 注入、容器内可见性和计量标签之间断链。PRR 不能只看 Pod 是否能启动，而要证明资源声明能被正确准入、绑定、使用、计量和故障分类。

```yaml
resource_claim_prr_drill:
  drill_id: rc-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    node_pool: gpu-prod-mixed-202606
    claim_modes:
      - extended_resource
      - dra_if_enabled
    workload_slices:
      - premium_long_context_inference
      - mig_inference_if_applicable
      - multigpu_rdma_training
  injected_scenarios:
    - device_class_has_no_matching_resource_slice
    - resource_claim_bound_to_wrong_gpu_class
    - mig_claim_exposes_whole_gpu
    - entitlement_allows_quota_but_denies_resource_class
    - metering_resource_label_missing_after_claim
  required_outputs:
    gpu_resource_claim_contract: generated
    resource_claim_acceptance_matrix: pass
    resource_claim_admission_record: generated_for_each_scenario
    resource_claim_fault_tree_execution: generated_by_controlled_failure
    gpu_assignment_record: generated_if_claim_bound
    gpu_device_visibility_reconciliation: pass_if_pod_started
    resource_claim_incident_cost_record: generated_if_failure_injected
  pass_criteria:
    pending_reason_is_actionable: true
    wrong_class_claim_is_blocked_or_reconciled: true
    mig_boundary_violation_blocks_node_or_pool: true
    entitlement_and_quota_decisions_are_replayable: true
    billing_hold_applied_when_metering_label_missing: true
    fallback_to_container_runtime_fault_tree_when_assignment_exists: true
```

```mermaid
flowchart LR
  Drill["resource_claim_prr_drill"] --> Contract["gpu_resource_claim_contract"]
  Contract --> Accept["resource_claim_acceptance_matrix"]
  Accept --> Admission["resource_claim_admission_record"]
  Admission --> Fault["resource_claim_fault_tree_execution"]
  Fault --> Cost["resource_claim_incident_cost_record"]
  Admission --> Assign["gpu_assignment_record"]
  Assign --> Runtime["container runtime PRR\nif claim bound"]
  Cost --> PRR["PRR resource claim gate"]
```

这类演练能避免 DRA 或 GPU class 上线时的“控制面成功幻觉”：DeviceClass 创建成功，但没有匹配的生产资源；ResourceClaim 绑定成功，但绑定到了不符合模型硬件匹配的 GPU class；MIG claim 看似细粒度，容器内却暴露整卡；queue quota 有余额，但 entitlement 不允许使用该隔离等级；Pod 能启动，但 DCGM、metering 和账单无法从 claim 追到 GPU UUID。资源声明语义一旦错，后续 runtime、SLO 和账单都会被污染。

容器 GPU runtime 也应有 PRR 演练。任何涉及 containerd/CRI-O、runc、NVIDIA Container Toolkit、GPU Operator、device plugin strategy、RuntimeClass、CDI spec 生成、NRI 插件或 MIG 暴露策略的变更，都可能同时影响调度、启动、隔离、观测和成本。PRR 不能只检查 DaemonSet Ready，也不能只跑 `nvidia-smi`。它必须演练从变更记录、准入矩阵、OCI diff、可见性对账、故障树到成本账本的完整路径。

```yaml
container_runtime_prr_change_drill:
  drill_id: cr-prr-drill-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    node_pool: h100-inference-canary
    runtime_change: crc-gpu-runtime-20260620-001
    migration_path: legacy_hook_to_cdi
    workload_slices:
      - non_gpu_pod
      - single_gpu_inference
      - mig_inference_if_applicable
      - multigpu_rdma_training_if_applicable
  required_outputs:
    container_runtime_change_record: reviewed
    gpu_operator_upgrade_evidence: generated_if_operator_changed
    container_runtime_change_acceptance: pass
    oci_runtime_injection_diff: sampled_and_clean
    gpu_device_visibility_reconciliation: pass
    container_gpu_runtime_fault_tree_execution: generated_by_controlled_failure
    container_runtime_incident_cost_record: generated_if_failure_injected
    baseline_invalidation_record: generated_if_runtime_path_changed
  pass_criteria:
    non_gpu_pod_cannot_see_gpu: true
    assigned_gpu_uuid_matches_container_view: true
    no_unexpected_devices_mounts_or_capabilities: true
    cdi_or_runtimeclass_failure_is_classified: true
    dcgm_pod_gpu_uuid_mapping_preserved: true
    rollback_to_previous_runtime_path_rehearsed: true
    cost_ledger_append_verified_if_failure_injected: true
```

这类演练能拦截 GPU Kubernetes 平台最常见的“局部验证幻觉”：节点上 `nvidia-smi` 正常，但 Kubernetes Pod 没走同一个 runtime；单 GPU smoke 正常，但 MIG 或多卡 RDMA 路径失败；Pod 能看到 GPU，但看到的是未分配 GPU；Operator 组件 Ready，但 DCGM 标签变化导致账本断链。容器 runtime 是 token 生产线的底座，PRR 要证明它的变更不会破坏分配、注入、隔离、观测和经济归因。

```mermaid
flowchart LR
  Drill["container_runtime_prr_change_drill"] --> Change["container_runtime_change_record"]
  Change --> Accept["container_runtime_change_acceptance"]
  Accept --> Diff["oci_runtime_injection_diff"]
  Accept --> Recon["gpu_device_visibility_reconciliation"]
  Accept --> Fault["container_gpu_runtime_fault_tree_execution"]
  Fault --> Cost["container_runtime_incident_cost_record"]
  Diff --> Gate["PRR container runtime gate"]
  Recon --> Gate
  Cost --> Gate
  Gate -->|pass| Rollout["rollout beyond canary"]
  Gate -->|fail| Rollback["rollback / quarantine / retest"]
```

网络 fabric 变更也应进入 PRR 演练。很多 AI Factory 在建设阶段会把网络变更当作基础设施内部维护：交换机配置改完、端口 up、RDMA 小测试通过，就恢复生产。这个做法在普通服务集群里有时可以接受，在大训练和高性能推理场景里风险很高，因为真实影响要经过 rank placement、NCCL interface、rail balance、checkpoint overlap、调度标签和成本账本才会显现。PRR 必须证明一次 fabric 变更失败时，系统能自动降级资源、阻止高风险 workload、保留证据、执行故障树、生成成本记录并完成回滚复测。

```yaml
fabric_change_prr_drill:
  drill_id: fab-prr-drill-20260620-001
  production_readiness_review: prr-training-prod-2026-06
  scope:
    fabric: train-fabric-a
    change_type: roce_qos_profile_update
    resource_pools: [training-prod-h100]
    workload_slices:
      - 64gpu_same_rack_training
      - 256gpu_cross_rack_training
      - checkpoint_heavy_training
      - distributed_evaluation
  injected_or_simulated_failures:
    - pfc_ecn_profile_drift
    - rail_balance_ratio_below_policy
    - container_rdma_path_mismatch
    - nccl_selected_interface_changed
    - checkpoint_plus_nccl_overlap_regression
    - scheduler_restores_allocatable_before_gate_pass
  required_outputs:
    fabric_change_record: generated
    baseline_invalidation_record: generated
    fabric_change_regression_gate: pass_or_block_recorded
    fabric_change_acceptance_matrix: generated
    network_diagnostic_bundle: generated
    congestion_fault_tree_execution: generated
    network_incident_cost_record: generated_if_failure_injected
    communication_regression_record: pass_or_fail_recorded
    scheduler_state_transition: limited_then_restored_only_after_pass
  pass_criteria:
    large_training_blocked_while_baseline_invalidated: true
    container_and_kubernetes_paths_validated_not_only_host_rdma: true
    pfc_ecn_qos_drift_detected: true
    rail_imbalance_classified_with_owner: true
    rollback_rehearsed_and_same_topology_retested: true
    cost_ledger_append_verified_if_failure_injected: true
    prr_gate_blocks_when_evidence_missing: true
```

```mermaid
flowchart LR
  Drill["fabric_change_prr_drill"] --> Change["fabric_change_record"]
  Change --> Invalidate["baseline_invalidation_record"]
  Invalidate --> Gate["fabric_change_regression_gate"]
  Gate --> Matrix["fabric_change_acceptance_matrix"]
  Matrix --> Diag["network_diagnostic_bundle"]
  Diag --> Fault["congestion_fault_tree_execution"]
  Fault --> Cost["network_incident_cost_record"]
  Gate --> Scheduler["scheduler limited / restored"]
  Cost --> PRR["PRR fabric gate update"]
  Scheduler --> PRR
```

这类演练能拦截三种高损失事故。第一种是“回归跑了，但没管调度状态”：baseline 已失效，调度器仍把 512 卡任务放进受影响 fabric。第二种是“网络单项通过，但 workload 组合失败”：host RDMA 正常，Kubernetes 容器里的 NCCL 选择了错误 NIC，或 checkpoint 与 AllReduce 叠加后触发拥塞。第三种是“事故成本不可见”：训练只是慢了半小时，没有失败票据，结果 GPU idle、checkpoint 回滚和模型发布延迟都没有进入经济账本。PRR 通过标准必须覆盖证据、动作和成本，不能只覆盖网络连通性。

从验收到上线的流水线可以用下面的图表示：

```mermaid
flowchart LR
  Hardware["硬件 / 节点 / 网络 / 存储"] --> Acceptance["acceptance baseline\nGPU / NCCL / RDMA / storage"]
  Acceptance -->|pass| ResourcePool["accepted resource pool"]
  Acceptance -->|fail| Remediation["remediation\nrepair / config / vendor"]
  Remediation --> Acceptance
  ResourcePool --> ModelGate["model quality gate\nquality / safety / cost"]
  ModelGate -->|pass| ServingRelease["serving release\nweights + tokenizer + runtime"]
  ModelGate -->|fail| ModelFix["model / prompt / eval fix"]
  ModelFix --> ModelGate
  ServingRelease --> QualityEvidenceGate["quality evidence gate\nslice / golden / experiment guardrail / rollback drill"]
  QualityEvidenceGate -->|pass| SupplyChainGate["supply chain gate\ndataset / checkpoint / provenance / cache"]
  QualityEvidenceGate -->|fail| QualityFix["fix slice / golden / guardrail / rollback"]
  QualityFix --> QualityEvidenceGate
  SupplyChainGate -->|pass| RAGAgentGate["RAG / Agent gate\npermission / context / tools / budget"]
  SupplyChainGate -->|fail| SupplyFix["fix lineage / restore / provenance / cache"]
  SupplyFix --> SupplyChainGate
  RAGAgentGate -->|pass| PRR["production readiness review"]
  RAGAgentGate -->|fail| AppFix["fix retrieval / tool policy / budget"]
  AppFix --> RAGAgentGate
  PRR -->|approve_canary| Canary["canary\nlimited tenants / traffic"]
  PRR -->|block| FixGap["close evidence gaps"]
  FixGap --> PRR
  Canary -->|healthy| Production["production scale"]
  Canary -->|breach| Rollback["rollback / drain / incident"]
  Production --> Telemetry["telemetry + cost + feedback"]
  Telemetry -->|"drift / incident / cost overrun"| PRR
```

质量证据本身也要演练失效路径。PRR 不能只检查 `quality_gate_execution` 是否通过，还要验证当评测集被污染、judge 漂移、反馈 pipeline 丢失 trace 或线上实验护栏触发时，系统是否会阻断放量、重跑门禁、冻结实验并把成本写入账本。否则团队会在报告上看到“质量通过”，但这个通过结论可能已经失效。

```yaml
quality_evidence_prr_invalidation_drill:
  drill_id: qe-prr-invalid-20260620-001
  production_readiness_review: prr-maas-chat-prod-2026-06
  scope:
    application: support-chat
    task_slices: [rag_citation, tool_call_json]
    serving_release: af-chat-large-20260619-r3
  simulated_or_controlled_events:
    - golden_set_unauthorized_export
    - overlap_scan_detects_training_contamination
    - judge_model_upgrade_without_backtest
    - feedback_events_missing_prompt_context_snapshot
    - online_experiment_guardrail_hard_stop
  required_outputs:
    eval_contamination_invalidation_record: generated_if_contamination
    judge_drift_calibration_record: generated_if_judge_changed
    quality_feedback_intake_pipeline: evidence_health_report
    quality_evidence_validity: generated
    quality_cost_ledger: prevention_cost_appended
    production_readiness_review: gate_updated
  pass_criteria:
    invalid_gate_not_usable_for_high_value_release: true
    judge_threshold_rebaseline_or_gate_rerun_required: true
    feedback_without_replay_not_promoted_to_gate_case: true
    experiment_freeze_preserves_quality_evidence_bundle: true
    quality_evidence_validity_cost_recorded: true
```

```mermaid
flowchart TB
  Drill["quality_evidence_prr_invalidation_drill"] --> Contam["eval_contamination_invalidation_record"]
  Drill --> Judge["judge_drift_calibration_record"]
  Drill --> Intake["quality_feedback_intake_pipeline health"]
  Contam --> Validity["quality_evidence_validity"]
  Judge --> Validity
  Intake --> Validity
  Validity --> Gate{"gate evidence usable?"}
  Gate -->|yes| Ramp["continue canary / ramp"]
  Gate -->|no| Block["block high-value release\nrerun gate / replace data"]
  Block --> Cost["quality_cost_ledger\nprevention cost + avoided loss"]
  Cost --> PRR["PRR gate update"]
```

这类演练会暴露质量治理的底层问题：golden set 访问审计只是记录但不会阻断，judge 升级改变分数却没有阈值重标定，反馈事件没有 prompt context snapshot 导致无法复现，实验 hard stop 后没有保留质量证据包。高水平的 AI Factory 不只是有评测报告，而是知道评测报告什么时候不能再被相信。

多模态应用还需要单独的 PRR 演练。原因是它的风险横跨上传、对象存储、预处理、模型服务、质量、计量、隐私和删除：文件能上传不代表可处理，OCR 成功不代表表格正确，模型能回答不代表引用区域可回放，删除原文件不代表派生产物已删除，文本 token 计量正确也不代表媒体处理成本正确。PRR 必须证明这些链路在生产前已经演练过。

```yaml
multimodal_prr_drill:
  drill_id: mm-prr-drill-20260620-001
  production_readiness_review: prr-claims-mm-prod-2026-06
  scope:
    application: claims-document-review
    multimodal_workload_profile: mwp-claims-document-review-202606
    endpoint: claims-document-review-prod
    media_types: [scanned_pdf, image]
  simulated_or_controlled_events:
    - corrupt_file_rejected_before_processing
    - oversized_file_hits_admission_policy
    - ocr_low_confidence_region_requires_human_review
    - layout_table_column_regression_detected
    - source_region_citation_replay_failure
    - sensitive_media_redaction_check
    - delete_request_removes_original_and_derived_artifacts
    - metering_hold_for_partial_pipeline_failure
  required_outputs:
    media_artifact_manifest: generated
    media_processing_pipeline_record: generated
    multimodal_serving_contract: validated
    multimodal_quality_gate_execution: generated
    multimodal_evidence_bundle: generated_for_failure_path
    multimodal_metering_event: generated
    multimodal_cost_ledger: appended
    production_readiness_review: updated_if_gap_found
  pass_criteria:
    no_model_serving_without_valid_media_manifest: true
    source_region_replay_matches_answer_citations: true
    partial_pipeline_failure_does_not_double_bill: true
    derived_artifacts_follow_retention_and_delete_policy: true
    human_review_triggered_for_low_confidence_or_high_risk: true
    cost_ledger_append_verified: true
```

```mermaid
flowchart TB
  Drill["multimodal_prr_drill"] --> Profile["multimodal_workload_profile"]
  Profile --> Manifest["media_artifact_manifest"]
  Manifest --> Pipeline["media_processing_pipeline_record"]
  Pipeline --> Serving["multimodal_serving_contract"]
  Serving --> Gate["multimodal_quality_gate_execution"]
  Gate --> Bundle["multimodal_evidence_bundle"]
  Serving --> Meter["multimodal_metering_event"]
  Meter --> Cost["multimodal_cost_ledger"]
  Bundle --> PRR["PRR gate update"]
  Cost --> PRR
```

这类演练能拦截很多真实事故。上传网关允许了平台不支持的 TIFF 或加密 PDF，后端反复重试并计费；OCR 低置信度区域没有触发人工复核，理赔结论依据了错误字段；source region 坐标系在页面旋转后错位，用户点击引用看到错误位置；删除请求只删了原始文件，OCR 文本和 embedding 仍在索引里；媒体预处理失败但模型已生成部分回答，账单系统不知道应该 hold、退款还是按成功任务计费。多模态 PRR 的目的，是让这些问题在受控演练中暴露，而不是在客户材料上暴露。

最后，建设计划应落到时间节奏。下面的 30/60/90/180 天不是固定日历，而是用于提醒第一阶段应该产出什么证据。不同组织可以调整顺序，但不应跳过证据。

| 阶段 | 目标 | 必须产出 | 不应做的事 |
| --- | --- | --- | --- |
| 0-30 天 | 明确目标和边界 | `workload_profile`、`business_model_profile`、容量模型、数据边界、初始 ADR | 在没有 SLO 和验收口径时下大额硬件单 |
| 31-60 天 | 建立资源和基线 | GPU/resource pool、fabric baseline、storage baseline、观测标签、准入流水线 | 把未准入资源给生产任务使用 |
| 61-90 天 | 打通平台路径 | Gateway、模型服务、token 计量、评测门禁、灰度回滚、基础 runbook | 对外承诺 SLA 或大规模多租户 |
| 91-180 天 | 生产试点和规模化决策 | PRR、canary、成本账本、incident 复盘、扩容/暂停决策 | 只按 GPU 利用率决定继续扩容 |

这张表的核心是“证据先于规模”。AI Factory 可以从小规模开始，但每一步都要留下可以复用的对象：profile、ADR、baseline、release、PRR、ledger 和 incident record。后续扩容时，团队复用的是这些工程对象，而不只是复用一套部署脚本。

这些对象应进入日常工程流程，而不是停留在文档目录中。`ai_factory_build_plan` 进入项目例会和阶段审计，`architecture_decision_record` 进入技术评审，`production_readiness_review` 进入发布门禁，acceptance baseline 进入资源池状态，cost ledger 进入容量和商业决策。每个对象都要有 owner、版本和触发更新的条件，否则它们会很快过期。

最小治理应从第一天开始。即使平台还小，也要有资源命名、租户标签、模型版本、计量事件、变更记录和验收基线。治理早期看起来麻烦，但比后期补齐历史数据便宜得多。没有这些基础对象，后续再做多租户、账单、SRE 和商业化时，团队会发现关键证据从未被采集。

成本报表也要进入上线评审。新模型、新租户或新资源池上线时，应给出预估 cost/token、GPU hour、存储增长、支持成本和停止条件。没有成本评审，平台会在功能增长后才发现经济模型不成立；没有停止条件，团队会把成本异常解释成“增长的代价”，而不是及时纠偏。

## 常见故障

第一类故障是先买 GPU，再发现机房电力、制冷、网络或存储不匹配。GPU 到货后无法上架、无法满载或无法通过 NCCL 基线，预算已经消耗，调整成本很高。解决方向是在采购前完成容量、电力、散热、网络和存储联合评审。

第二类故障是只建设训练集群，没有模型服务和应用入口。训练任务能跑，但模型无法稳定上线，业务无法调用，ROI 无法闭环。解决方向是同时设计训练链路和推理链路，至少让一个模型能从评测进入服务，再回收线上反馈。

第三类故障是推理平台上线后才补 token 计量和账单。历史请求无法归因，免费额度和失败重试无法计入成本，定价只能凭经验。解决方向是第一天就保留 token 计量和租户标签，即使早期不真实收费，也要记录事实。

第四类故障是调度系统不了解拓扑。GPU 总量足够，但大训练启动不了；任务启动了，但跨 rack 或跨 rail 导致 NCCL 慢；推理副本被放到不合适节点，延迟长尾严重。解决方向是把 GPU、NIC、NUMA、NVLink、rack 和网络拓扑写入调度和资源池。

第五类故障是没有准入和 runbook。生产任务暴露坏节点，事故依赖少数专家，维修回池没有回归验证。解决方向是把准入测试、诊断包、故障树和回池验收纳入日常流程。没有这些能力，平台越大，事故成本越高。

第六类故障是阶段目标过大。第一阶段同时做 MaaS、私有化、预训练、Agent、行业云和算力租赁，结果每条线都不完整。解决方向是收缩首批 workload，先交付可生产闭环，再扩展模式。AI Factory 可以长期演进，不需要第一版包打天下。

## 性能指标

建设指标包括计划完成度、阶段交付周期、验收通过率、资源入池时间、上线范围、回滚次数、未完成风险和技术债关闭率。它们回答建设是否按可控节奏推进，而不是只看采购是否完成。采购完成只是资源到位，不代表产能上线。

推理指标包括 TTFT、TPOT、TPOP、E2E latency、tokens/s、错误率、限流率、streaming 中断、模型加载时间、KV Cache 水位、cost per token 和推理毛利。它们回答模型服务是否能稳定生产有价值 token，并支撑商业或内部效率目标。

训练指标包括作业成功率、排队时间、gang scheduling 等待、step time、NCCL 带宽、checkpoint 时长、恢复成功率、GPU 小时浪费、非用户原因失败和评测通过率。它们回答训练平台是否能把 GPU 时间转化为模型能力，而不是被等待和故障消耗。

基础设施指标包括 GPU 健康、Xid/ECC、NVLink、RDMA 错误、packet loss、storage throughput、network telemetry、准入基线偏离、节点维修回池时间、资源碎片和有效 GPU 小时。它们回答底层资源是否可信、可调度、可恢复。

经济和运营指标包括预算消耗、cost per token、tokens/W、资源利用率、租户成本、训练 ROI、SLO 达成率、error budget、incident 数量、MTTR 和复盘行动项完成率。AI Factory 建设最终要同时解释技术效果、用户体验和经济结果。

指标还要分阶段使用。设计阶段看假设完整性，验收阶段看基线，通过生产后看 SLO 和故障，规模化阶段看成本和毛利。不同阶段用同一套指标权重，会误导决策。早期不应过分追求利用率，规模化后也不能只看功能完成度。

所有指标都应能追溯到 owner。资源指标归基础设施，模型指标归模型团队，平台指标归平台团队，业务指标归业务 owner，SRE 负责把它们串成运行事实。没有 owner 的指标只是装饰。

## 设计取舍

第一个取舍是先快跑还是先打底。完全打底会拖慢业务验证，完全快跑会留下昂贵技术债。务实做法是第一阶段只服务少数高价值 workload，但租户、计量、观测、验收和版本边界必须保留。功能可以少，边界不能乱。

第二个取舍是自建还是采购。自建能更贴合业务和长期控制，采购或托管能缩短上线时间。判断标准包括团队能力、差异化需求、合规边界、长期成本和供应商锁定。核心生产能力越关键，越要理解底层；非核心能力可以借助成熟产品。

第三个取舍是训练优先还是推理优先。若业务目标是快速落地应用，推理、RAG、Agent 和 MaaS 可能优先；若目标是基础模型能力，数据、训练、评测和大规模网络必须优先。很多组织同时想要两者，但预算和团队无法支撑。优先级要由业务目标决定。

第四个取舍是统一资源池还是专用资源池。统一池提高利用率，专用池提高隔离和可预测性。生产推理、大训练、实验、验收和低优批量任务的资源特性不同，通常需要分层资源池。资源池不是越统一越好，而是要让 SLO、成本和调度语义一致。

最终，从 0 到 1 建设 AI Factory 的设计目标，是用最小可生产系统验证价值，同时不关闭未来扩展路径。好的第一版不是最完整的版本，而是边界清楚、指标可信、故障可诊断、成本可解释、能够继续演进的版本。

取舍还要保留复盘机制。第一版选错某些组件并不可怕，可怕的是没有数据证明哪里错、没有接口替换、没有预算纠偏。系统能持续学习，才配得上 Factory 这个词。

## 小结

- AI Factory 建设要从业务和模型目标反推基础设施。
- GPU、网络、存储、调度和推理平台必须协同选型。
- 准入验收、可观测性和 SRE 是生产系统的基础，不是后续补丁。
- 分阶段上线能降低风险，让组织能力和技术系统同步成熟。

## 延伸阅读

- [Google Cloud Architecture Framework: AI and ML perspective](https://cloud.google.com/architecture/framework/perspectives/ai-ml)
- [NVIDIA DCGM Diagnostics documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)；[NCCL tests repository](https://github.com/NVIDIA/nccl-tests)
- [vLLM documentation](https://docs.vllm.ai/)；[TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)
