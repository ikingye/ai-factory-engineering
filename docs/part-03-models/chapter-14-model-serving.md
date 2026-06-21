# 第 14 章：模型服务

## 14.1 导读

### 14.1.1 本章回答的问题

- model server、endpoint、replica、batching、autoscaling 和 canary 如何组成模型服务系统？
- 模型服务和 MaaS、AI Gateway、推理引擎、Kubernetes 的边界是什么？
- 如何设计可灰度、可回滚、可观测、可扩缩容的模型服务？


### 14.1.2 本章上下文

- 层级定位：本章属于 `Model 层`，重点讨论模型训练、后训练、微调、评测和模型服务化。
- 前置依赖：建议先理解 第 13 章：模型评测 中的核心对象和路径。
- 后续关联：本章内容会继续连接到 第 15 章：推理引擎，并在系统地图、深度标准和读者测试中被交叉引用。
- 读完能力：读完本章后，读者应能把《模型服务》中的概念映射到 AI Factory 的生产路径、工程对象、观测证据和设计取舍。


### 14.1.3 读者测试

- 机制题：读者能否解释 model server、endpoint、replica、batching 的核心机制，以及它们如何共同支撑《模型服务》？
- 边界题：读者能否区分 模型算法、模型产物、serving release、评测证据和基础设施容量 的责任边界，并说明哪些问题不能简单归因到本章组件？
- 路径题：读者能否从模型产物追到数据、训练、评测、serving release、路由和回滚，并指出本章对象在路径中的位置？
- 排障题：当《模型服务》相关生产症状出现时，读者能否列出第一层证据、下一跳证据、可能 owner 和止血动作？


### 14.1.4 一个真实场景

一个模型服务升级推理引擎后，单副本压测吞吐提升了，但生产上线后错误率上升。排查发现，新版本改变了 streaming chunk 的边界和 usage 返回时机，AI Gateway 的适配不完整；同时 autoscaler 仍然只看 CPU，没有看 GPU HBM、队列长度和 KV Cache 使用，高峰期扩容滞后；回滚时又发现模型权重、tokenizer 和 runtime 镜像没有绑定记录，只能人工确认上一稳定组合。吞吐提升是真的，生产事故也是真的。

这个场景说明，模型服务不是“启动一个推理进程”。它是模型权重、tokenizer、推理引擎、服务协议、资源分配、批处理、可观测性、扩缩容和发布流程的组合。任何一个边界不清楚，模型能力都可能无法稳定交付。AI Factory 中，模型服务是模型层进入生产系统的执行面，是 MaaS 和 GPU 基础设施之间的关键连接点。

模型服务还承担成本责任。Batching、replica 数、资源池、并发限制和 autoscaling 策略都会影响 tokens/s、TTFT、TPOT 和 cost per token。一个服务可以在单副本压测中表现很好，却因为冷启动、长上下文、流式连接和灰度策略在生产中不稳定。模型服务设计必须同时考虑质量、体验、容量、成本和发布安全。

它也是跨团队接口。模型团队交付 artifact 和能力说明，推理团队选择 runtime，平台团队管理部署和观测，SRE 负责稳定性，MaaS 团队承接 API 和租户体验。若模型服务缺少清晰契约，上游会把所有问题归咎于模型，下游会把所有问题归咎于基础设施。服务边界越清楚，协作成本越低。

本章讨论的模型服务，重点是生产系统，而不是本地 demo。能回答一次请求只是最低要求；能在多租户、灰度、故障、扩容和成本约束下持续回答，才是 AI Factory 需要的模型服务能力。


## 14.2 基础模型

### 14.2.1 核心概念

模型服务（model serving）把模型权重和推理引擎包装成可调用服务。它向上接受 MaaS 或 AI Gateway 的请求，向下使用 GPU、CPU、HBM、网络和存储。一个生产模型服务通常由 endpoint、replica、runtime、model artifact、tokenizer、batching、health check、autoscaling、observability、canary 和 rollback 组成。

Model server 是实际加载模型并执行推理的进程。Endpoint 是平台暴露的服务入口，可以对应一个模型版本、多个 replica 或一个路由组。Replica 是服务副本，提供并发和高可用。Batching 把多个请求组合执行，提高 GPU 利用率。Autoscaling 根据负载调整 replica。Canary 和 rollback 控制发布风险。

模型服务与其他层有明确边界。MaaS 面向租户、模型目录、计量和 API 产品；AI Gateway 执行认证、限流、路由和策略；模型服务负责具体推理执行；推理引擎负责 runtime 内部的 batching、KV Cache 和 kernel 调度；Kubernetes 或调度系统负责容器、GPU 和副本生命周期。边界清楚，问题才能定位。

模型服务的难点来自 LLM 的状态性。Streaming 请求持续持有连接和 KV Cache，长上下文占用显存，batch 结构随请求动态变化，模型加载时间长，副本下线需要排水。普通无状态微服务经验只能解决一部分问题。LLM serving 需要围绕 token、cache、GPU 和发布状态重新设计服务控制面。

这些概念也决定了故障责任。Endpoint 失败可能是路由或发布问题，replica 失败可能是资源或 runtime 问题，batching 问题可能表现为延迟，autoscaling 问题可能表现为排队，canary 问题可能表现为版本差异。模型服务指标必须保留这些对象维度，才能定位问题。

还要把 artifact 和 deployment 分开。一个模型 artifact 可以被多个 endpoint 使用，一个 endpoint 可以逐步切换 deployment，一个 deployment 又可以包含多个 replica。分层管理能支持灰度和回滚；混在一起则会让发布变成替换文件。


### 14.2.2 系统架构

模型服务架构通常包括控制面和数据面。控制面管理模型 artifact、endpoint、replica、配置、发布策略、扩缩容策略和观测规则；数据面处理实际请求，从 Gateway 接收请求，进入 endpoint，选择 replica，执行 tokenizer、prefill、decode、streaming 和 usage 上报。控制面决定服务形态，数据面决定请求体验。

一次请求进入模型服务后，通常先经过 endpoint 的协议和参数校验，再进入某个 replica 的队列。Replica 内部的 model server 调用推理引擎，执行 batching、prefill、decode 和 KV Cache 管理。响应以 streaming 或非 streaming 方式返回，同时上报 TTFT、TPOT、token、错误码、finish reason、队列时间和资源指标。Autoscaler 根据这些指标调整 replica。

发布链路也属于架构的一部分。新模型或新 runtime 先创建 candidate deployment，经过健康检查和预热后接入少量 canary 流量；观察质量、错误、延迟、token 和成本指标；若通过门禁，再逐步扩大流量；若失败，快速 rollback 到上一稳定版本。没有发布控制，模型服务每次升级都是直接修改生产状态。

架构还应让配置变更可审计。Batch 参数、max context、并发上限、LoRA 加载策略、超时和错误映射都可能影响线上体验。它们不应以临时环境变量散落在副本中，而应进入 deployment 配置并带版本。生产问题经常来自配置漂移，而不是代码错误。

数据面还要尽量少做控制决策。租户权限、模型目录和价格策略属于上层控制面；model server 应专注执行推理和上报事实。边界过厚会让服务进程难以替换，边界过薄又缺少运行时保护。关键是把策略和执行分层。

架构评审时，应逐项确认请求路径、指标路径、发布路径和回滚路径。能画出请求怎么进来还不够，还要能画出指标怎么反馈到扩缩容，配置怎么发布到副本，事故时流量怎么退回。模型服务是多个控制循环的组合。

![图：14.2.2 系统架构](../assets/diagrams/part-03-models-chapter-14-model-serving-01.svg)


## 14.3 关键技术

### 14.3.1 model server

Model server 是承载模型推理的服务进程。它负责加载权重、初始化 tokenizer、暴露协议接口、接收请求、调用推理引擎、管理 batch、处理 streaming、返回 usage，并上报日志、metrics 和 traces。常见实现可能基于 vLLM、SGLang、TensorRT-LLM、Triton 或自研服务。无论实现不同，生产职责类似：稳定执行模型推理。

Model server 的启动不是简单进程启动。大模型需要加载权重、初始化 CUDA context、分配 HBM、建立并行通信、加载 tokenizer 和 warmup kernel。任何一步失败，都可能表现为容器 ready 但模型不可用。因此健康检查不能只看 HTTP 端口，还要检查模型是否加载完成、推理是否可执行、GPU 是否可用、关键指标是否上报。

运行中，model server 必须处理请求取消、streaming 中断、超时、OOM、KV Cache 释放、优雅下线和错误映射。客户端取消后，如果服务端继续 decode，会浪费 GPU；副本下线时，如果直接 kill 进程，会中断长回答；OOM 后如果不隔离实例，可能持续影响流量。模型服务需要把这些边界条件作为核心逻辑。

Model server 还要提供稳定协议。OpenAI-compatible API、内部推理协议、streaming chunk、usage 返回、错误码和工具调用格式都可能被上游依赖。升级推理引擎时，即使模型输出质量不变，协议细节也可能变化。生产 model server 应通过兼容性测试和 contract test，避免 runtime 升级破坏 MaaS 和应用。

Model server 也要控制资源边界。它应拒绝超过上下文或输出预算的请求，限制并发，防止单个租户耗尽 KV Cache，并在异常时返回可解释错误。把所有保护都放在 Gateway 不够，因为只有 model server 最清楚实际 runtime 状态。入口治理和服务端保护需要配合。

此外，model server 应提供可诊断启动日志和运行状态。模型加载到哪一步、占用多少 HBM、初始化哪个并行组、使用哪个 tokenizer，都应能查询。否则冷启动和加载失败会很难排查。


### 14.3.2 endpoint

Endpoint 是对外暴露的模型服务入口。它可以代表一个模型版本、一个 deployment、一个资源池或一个路由组。上游 Gateway 通常不应直接感知每个 replica，而是调用 endpoint；endpoint 再根据负载和策略分配到后端副本。Endpoint 是模型服务的数据面入口，也是控制面管理对象。

Endpoint 不只是 Kubernetes Service。它应包含模型名、版本、协议、tokenizer、能力标签、资源池、SLA、发布状态、观测标签和权限边界。MaaS 模型目录中的一个模型，可能映射到多个 endpoint：标准池、premium 池、灰度版本、区域版本或专属租户版本。没有 endpoint 抽象，模型路由会和底层部署强耦合。

Endpoint 还承担兼容性和稳定性边界。它应声明支持哪些 API 参数、最大上下文、streaming、tool calling、response format 和模型版本。若后端 runtime 改变，endpoint 要么保持兼容，要么发布新版本。上游应用依赖的是 endpoint contract，而不是某个临时服务地址。契约不清，升级风险会传递给应用。

工程上，endpoint 应有生命周期：draft、warming、ready、canary、stable、draining、deprecated。每个状态对应不同流量行为。比如 warming 阶段可以预热但不接收生产流量，draining 阶段不接新请求但等待 streaming 结束。状态化管理让发布和回滚可控，而不是依赖人工操作。

Endpoint 还应绑定观测命名。所有请求、指标、日志和计量事件都应带 endpoint id 和版本。否则一个模型多个部署形态并存时，平台无法区分 premium 池、标准池和灰度池。Endpoint 是观测和成本归因的关键维度，不只是网络入口。

Endpoint 设计还影响用户承诺。稳定 endpoint 应尽量保持协议和行为兼容，实验 endpoint 可以允许更快变化但不承诺 SLA。把实验流量和生产流量放在同一 endpoint，会让应用无法管理风险。入口命名本身就是产品契约。


### 14.3.3 replica

Replica 是模型服务副本，多个 replica 提供并发能力、弹性和高可用。每个 replica 通常加载一份模型权重，可能占用一张或多张 GPU，也可能通过 tensor parallel 跨多卡运行。副本数量直接影响容量和成本：副本越多，可服务流量越高，但空闲成本也越高。

Replica 管理首先要处理冷启动。大模型加载权重和初始化 runtime 可能较慢，扩容后不能立即承接流量。平台需要 readiness gate、warmup 请求和预热容量。若 autoscaler 在流量峰值到来后才创建副本，实际可用时间可能已经晚了。对高 SLA 模型，保留热副本往往比极限节省成本更重要。

下线同样复杂。Streaming 请求可能持续很久，直接删除 pod 会中断用户输出；长 decode 请求占用 KV Cache，排水时间不可忽略。Replica 下线应先停止接收新请求，等待已有请求完成或到达超时，再释放资源。发布系统和 Kubernetes 生命周期钩子需要配合，否则滚动升级会制造用户可见错误。

Replica 还需要健康分级。进程存活、模型加载完成、推理成功、延迟正常、HBM 充足、错误率正常，是不同层次的健康。一个副本可能端口可用但队列严重堆积，或 GPU 出现错误但还未退出。负载均衡和 autoscaling 应使用更丰富的健康信号，而不是简单 ready/not ready。

Replica 也要考虑故障隔离。某个副本出现 OOM、Xid 或连续超时，应被摘除并保留诊断信息，而不是继续接收请求。对于多卡副本，任一 GPU 或通信链路异常都可能影响整个 replica。健康检查必须理解实际服务拓扑。

副本数量也不是越多越好。过多副本会增加空闲成本，降低每个副本的 batch 效率；过少副本会降低可用性和弹性。容量规划需要结合流量分布、冷启动时间、SLA 和 batching 效率选择副本策略。


### 14.3.4 batching

Batching 把多个请求合并执行，提高 GPU 利用率。传统静态 batching 适合离线任务，输入形状和执行时间较一致；LLM 在线推理更常见 continuous batching，因为请求到达时间、输入长度和输出长度不同。Continuous batching 可以在 decode 过程中动态加入和移除序列，提高吞吐，但调度逻辑更复杂。

Batching 是延迟和吞吐的核心取舍。更大的 batch 通常提高 tokens/s 和 GPU 利用率，但会增加排队和 TTFT；更小 batch 降低等待但可能浪费 GPU。Prefill 和 decode 对 batching 的要求也不同：prefill 受输入长度影响，decode 受 active sequence 和 KV Cache 影响。一个固定 batch 参数很难适配所有 workload。

Batching 还会影响公平性。长上下文请求可能占用大量 prefill，长输出请求可能长时间占用 decode slot，短请求可能被排队。多租户服务中，如果没有优先级和隔离，高价值低延迟请求可能被低价值长任务拖慢。模型服务需要结合租户、服务等级、请求长度和资源池策略管理 batch。

观测指标是调参基础。平台应记录 queue length、queue time、batch size、prefill batch、decode batch、active sequence、tokens/s、TTFT、TPOT、KV Cache 使用和拒绝请求。没有这些指标，batching 调参只能靠经验。Batching 的目标不是最大化单一吞吐，而是在目标 SLO 下最大化有效产能。

Batching 策略还应按 workload 分层。低延迟 Chat、代码补全、长文档总结和批量推理不应使用同一等待时间和 batch 上限。平台可以用不同 endpoint 或资源池承载不同策略。把所有流量混在一个 batch 队列中，是很多尾延迟问题的根源。

还要处理取消和超时。请求在队列中取消，应及时移除；decode 中取消，应释放 KV Cache；超时请求应记录已生成 token 和阶段。Batching 若不能正确处理生命周期，会出现隐性资源泄漏。


### 14.3.5 autoscaling

Autoscaling 根据负载自动调整 replica 数。普通微服务常看 CPU 或请求数，但 LLM 服务更适合看 queue length、TTFT、TPOT、tokens/s、GPU utilization、HBM、KV Cache 使用、active sequence 和错误率。CPU 低不代表 GPU 空闲，GPU utilization 高也不一定表示需要扩容；关键是用户体验和队列是否恶化。

扩容有冷启动问题。模型权重加载、镜像拉取、GPU 分配、通信初始化和 warmup 都需要时间。若 autoscaler 只根据当前队列扩容，可能在副本 ready 前已经出现 SLA 违约。高价值在线服务常需要预测扩容、预热副本、最小热容量和分时容量策略。Autoscaling 不只是反应式控制。

缩容也要谨慎。副本可能正在处理 streaming 请求，或者持有大量 KV Cache。直接缩容会中断输出或造成重试。缩容策略应先 drain，再等待完成或超时，并避免在短周期波动中频繁扩缩。Thrashing 会增加冷启动、缓存失效和延迟抖动。稳定性往往比极限省钱更重要。

Autoscaling 还要考虑模型和租户差异。Premium endpoint 可能保留更多热容量，批量推理 endpoint 可以更激进缩容，低频模型可以按需加载。一个全局 autoscaling 策略无法适配所有模型。平台应按 workload profile 和 SLA 定义扩缩容策略，并持续用线上指标校准。

还要避免扩缩容与发布同时造成扰动。模型升级期间副本本就不稳定，如果 autoscaler 同时频繁调整容量，问题归因会困难。发布系统可以临时冻结或限制 autoscaling，让 canary 指标更可信。控制循环之间需要协调。

Autoscaling 也要有上限和预算。无限扩容可能保护了延迟，却让成本失控；过低上限又会造成持续限流。扩缩容策略应同时读取 SLA 和预算约束。模型服务弹性是经济控制的一部分。


### 14.3.6 multi-model serving

Multi-model serving 指一个服务系统承载多个模型或多个模型变体。它可以提高资源利用率，尤其适合小模型、低频模型、embedding、rerank 或大量租户 adapter 场景。但它也会增加权重加载、缓存管理、隔离、路由和可观测性复杂度。多模型服务不是免费合并，而是用复杂度换利用率。

关键问题是模型常驻还是按需加载。常驻多个模型可以降低请求延迟，但占用 HBM 和内存；按需加载节省资源，但冷启动可能不可接受。平台需要根据流量频率、模型大小、SLA 和租户价值决定策略。高频高 SLA 模型通常独立部署，低频低 SLA 模型可以共享资源。

多模型服务还要处理隔离。一个模型的长请求、OOM 或异常不应影响其他模型；一个租户的 adapter 不应被错误加载给另一个租户；指标必须按模型和租户切分。若共享服务缺少隔离和观测，资源利用率提高会换来排障难度和风险增加。共享越多，边界越要清楚。

工程上，multi-model serving 应先从低风险场景开始。比如把多个小 embedding 模型合并，或为低频 LoRA 做按需加载；对于大语言模型主力在线服务，应谨慎评估。多模型服务的成熟度取决于 registry、路由、加载、缓存、监控和回滚是否都支持模型维度。

多模型服务还要考虑安全边界。不同模型可能对应不同租户、数据权限和合规要求，共享进程时日志、缓存和错误信息都要避免串租户。资源共享不能突破数据隔离。平台如果做不到强隔离，就应限制共享范围。

多模型服务的观测也更复杂。指标必须按 model、tenant、adapter、artifact version 和 load state 切分。否则一个冷门模型的加载失败可能被主力模型流量淹没。共享服务越复杂，标签体系越重要。


### 14.3.7 canary

Canary 是灰度发布策略，在扩大影响前用少量流量验证新模型、runtime 或配置。模型服务 canary 可以按租户、用户、流量比例、endpoint、模型版本或请求类型切分。它不仅用于权重升级，也用于 tokenizer、推理引擎、batch 参数、quantization 和服务镜像变更。任何可能影响输出或性能的变更都应灰度。

Canary 指标必须多维。错误率只是基础，还要看 TTFT、TPOT、P95/P99、streaming 中断、token 分布、质量代理指标、工具调用成功率、成本和用户投诉。一个版本可能 HTTP 错误率正常，但输出更长、格式更差或成本更高。模型服务 canary 必须同时观察系统指标和模型行为指标。

Canary 还需要版本可解释。Trace 中应记录模型版本、runtime 镜像、tokenizer、配置版本、路由规则和 canary 组。出现问题时，平台要能判断请求是否命中新版本，是否只影响某个租户或某个资源池。没有版本标签，灰度会失去排障价值。灰度不是流量随机切分，而是可审计发布。

发布流程应支持自动暂停和手动审批。低风险指标异常可以自动停止放量，高风险安全或质量问题应立即回滚，边界不清的指标进入人工判断。Canary 的目标是尽早发现问题并限制影响面，而不是形式上走几个比例。好的 canary 设计能让团队敢于更频繁地发布。

Canary 还应覆盖回放测试。在接入真实流量前，可以用历史请求样本或标准评测集打到新版本，检查协议、延迟、错误码和输出格式。离线回放不能替代真实灰度，但能提前发现明显兼容性问题，减少用户暴露。

灰度样本也要可解释。只把 5% 流量切过去还不够，要知道这 5% 来自哪些租户、哪些应用、哪些输入长度和哪些服务等级。否则灰度通过可能只是因为没有覆盖高风险流量。


### 14.3.8 rollback

Rollback 是把流量或服务恢复到上一稳定版本。模型服务 rollback 需要版本化权重、tokenizer、runtime 镜像、配置、prompt 模板、路由策略和资源池。只回滚镜像不回滚模型，可能无法恢复行为；只回滚权重不回滚 tokenizer，可能出现 token 口径和输出格式变化。模型服务的版本是组合版本。

生产系统应保留上一稳定版本的可用容量。若发布时立即删除旧副本，回滚仍要经历镜像拉取、权重加载和 warmup，恢复时间会变长。对于关键模型，可以在 canary 期间保留旧版本热容量；对于成本敏感模型，也至少要保留可快速恢复的 artifact 和配置。Rollback 能力需要提前设计。

Rollback 决策应由门禁指标触发。错误率、延迟、streaming 中断、质量指标、安全拦截、投诉或成本异常都可能触发回滚。不同指标的阈值和 owner 应预先定义。事故中临时讨论是否回滚，会浪费时间。回滚不是承认失败，而是生产系统保护用户的机制。

回滚后还要保留现场。问题版本的日志、trace、指标、请求样本、配置和副本状态应被归档，供后续复盘。若回滚过程中清理了所有证据，团队只能知道“新版本不好”，不知道为什么不好。模型服务发布体系应同时支持快速恢复和证据保全。

Rollback 也要定期演练。很多平台文档上支持回滚，实际事故中才发现旧 artifact 被清理、旧镜像无法拉取、旧配置不兼容或权限不足。回滚能力不经过演练就不能算生产能力。模型服务发布应把回滚演练作为验收项。

回滚不一定只回到旧权重。某些事故需要回滚 runtime，某些需要回滚路由，某些需要回滚 batch 配置。发布系统应支持细粒度回滚，同时能保证组合版本一致。粗糙回滚会制造二次事故。


## 14.4 工程落地

### 14.4.1 工程实现

模型服务发布单元应把模型 artifact、runtime、资源、扩缩容、发布策略和指标门禁写在同一配置中。这个配置应被部署系统、Gateway、模型目录、观测系统和回滚工具共同理解。若部署系统只知道镜像，模型目录只知道模型名，Gateway 只知道 endpoint，出现事故时很难还原完整版本。发布单元是生产模型的最小可审计对象。

模型服务还需要定义 endpoint 与 replica 的状态机。很多生产事故不是推理逻辑错，而是状态切换粗糙：模型还在 warming 就接流量，副本进入 terminating 却没有 drain，canary 失败后旧版本容量已经释放。状态机应成为发布系统、Gateway 和 autoscaler 的共同语言。

![图：14.4.1 工程实现](../assets/diagrams/part-03-models-chapter-14-model-serving-02.svg)

Gateway 只应把生产流量路由到 `Ready`、`Canary` 或 `Stable` 中被策略允许的 endpoint；autoscaler 在缩容前必须把副本切到 `Draining`；发布系统在 `Warming` 阶段应执行最小推理探针，而不是只看端口。这些状态要体现在指标和事件里，否则线上看到的只是 pod 名称变化，很难解释请求为什么失败。

示例配置如下：

```yaml
model_deployment:
  name: af-chat-large-v2
  model_artifact: registry/af-chat-large:v2
  runtime_image: inference-runtime:v1.8
  engine: vllm
  replicas: 4
  resources:
    gpu: 1
  rollout:
    strategy: canary
    steps: [5, 25, 50, 100]
  metrics:
    required: [ttft, tpot, error_rate, tokens_per_second]
```

更完整的 deployment 记录还应绑定 tokenizer、engine 配置、兼容性门禁和回滚对象。模型服务故障经常来自组合不一致：权重回滚了，tokenizer 没回滚；runtime 回滚了，KV Cache 参数没回滚；模型目录版本改了，Gateway route 还指向旧 endpoint。组合版本可以减少这类漂移。

```yaml
serving_release:
  release_id: af-chat-large-20260619-r3
  endpoint: af-chat-large-prod
  artifacts:
    weights: registry/models/af-chat-large@sha256:example
    tokenizer: registry/tokenizers/af-chat-large@sha256:example
    chat_template: templates/af-chat-large:v12
  distribution:
    artifact_manifest: registry/manifests/af-chat-large-20260619.json
    weight_cache_policy: prewarm_before_canary
    cache_key: weights_tokenizer_template_digest
  runtime:
    image: registry/inference/vllm@sha256:example
    engine_config: configs/vllm/af-chat-large-prod:v7
    cuda_compat_profile: cuda-driver-runtime-profile-a
  rollout_gates:
    contract_tests: required
    warmup_probe: required
    ttft_p95_regression: block_on_regression
    token_count_drift: block_on_unexplained_drift
  rollback_to: af-chat-large-20260612-r9
```

在高 SLA MaaS 中，`serving_release` 还应被打包成 `serving_release_bundle`。前者描述一次部署的核心组合，后者描述这个组合进入生产所需的所有可验证外部依赖：模型目录、Gateway 路由合同、质量契约、runtime gate、artifact 分发、cache 状态、计量 schema、rollback/drain 语义和成本对象。发布事故常常不是某个单项错误，而是组合边界断裂：模型目录指向新 release，Gateway fallback 仍指旧能力，计量 schema 没同步，cache 里有旧 tokenizer，rollback 只恢复权重没有恢复 template。Bundle 的目标是让这些关系在发布前就成为显式证据。

```yaml
serving_release_bundle:
  bundle_id: srb-af-chat-large-20260619-r3
  release_id: af-chat-large-20260619-r3
  model_registry:
    model_alias: af-chat-large
    registry_entry: model-registry/af-chat-large/20260619-r3
    model_artifact_provenance: map-af-chat-large-20260619-r3
  serving_contracts:
    serving_quality_contract: sqc-af-chat-20260619-r3
    serving_route_release_contract: srrc-af-chat-large-20260620
    multimodal_serving_contract: optional_if_multimodal
  artifacts:
    model_artifact_distribution: mad-af-chat-large-20260619-r3
    cache_residency_policy: prewarm_primary_and_rollback
    cache_invalidation_record: required_if_replacing_or_revoking
  runtime:
    runtime_quality_gate: rqg-vllm-prod-h100-v7
    engine_canary_record: engine-canary-20260620-001
    engine_admission_health_schema: eah-schema-v3
    usage_schema: usage-openai-compatible-v6
  rollout:
    canary_plan: 1_5_25_50_100
    stop_conditions:
      - protocol_contract_failure
      - token_count_drift_unexplained
      - quality_guardrail_breach
      - cost_per_successful_answer_regression
    drain_contract: dc-af-chat-large-v3
  rollback:
    rollback_to_bundle: srb-af-chat-large-20260612-r9
    serving_rollback_drill: srd-af-chat-20260620
    route_restore_verified: true
    warm_capacity_available: true
  economics:
    inference_runtime_cost_ledger: ircl-af-chat-large-202606
    quality_cost_ledger: qcost-support-202606
    serving_release_cost_record: required_on_incident_or_rollback
```

这份 bundle 让发布系统有了最小不可拆单元。上线审批不应只看 release id，而要检查 bundle 中每个引用是否 valid：provenance 是否来自批准训练，quality contract 是否绑定同一 tokenizer/template/runtime，route contract 是否只把兼容 release 作为 fallback，usage schema 是否被计量系统识别，rollback bundle 是否仍有热容量。任何一个引用缺失，都意味着发布不是“有风险”，而是“无法证明风险边界”。

![图：14.4.1 工程实现](../assets/diagrams/part-03-models-chapter-14-model-serving-03.svg)

Bundle 还可以减少“半回滚”。如果一次事故触发 rollback，系统应恢复 `rollback_to_bundle` 中声明的 route、weights、tokenizer、template、runtime profile、usage schema、cache 和 drain 语义，而不是让操作者手工挑组件。若业务只需要回滚 route 或关闭 canary，也应在 `serving_rollback_record` 中记录是局部回滚，并证明其余 bundle 仍与质量门禁一致。回滚粒度可以细，但证据必须仍按 bundle 对齐。

工程流程应包括预检、部署、预热、灰度、扩大、稳定和清理。预检查模型 artifact、tokenizer、runtime 兼容性和资源配额；部署后先做健康检查和 warmup；灰度期间观察指标；通过后逐步扩大；稳定后再清理旧版本。每个阶段都应有自动状态和人工可见性。发布不是一次 kubectl apply。

实现中还要标准化错误处理。模型加载失败、请求超限、OOM、后端超时、streaming 取消、tokenizer 错误和引擎异常，都应映射为可解释错误码，并进入 trace 和指标。上游 Gateway 和 MaaS 需要这些信息做 fallback、计量和用户提示。模型服务不能把所有异常都变成 500。

还要建立发布前检查清单。检查项包括 artifact 是否存在、checksum 是否匹配、tokenizer 是否兼容、runtime 是否支持、资源是否足够、指标是否上报、rollback 是否可用。清单自动化后，发布事故会明显减少。模型服务的工程质量来自这些细节。

服务配置还应进入代码评审或变更审批。模型服务的 YAML、路由、资源和门禁都可能影响生产，不应被视为低风险配置。配置即代码，配置变更也需要验证。

实现完成后，应通过回放测试、压测和故障注入验证。只验证正常请求成功，无法证明服务可生产。

验证结果应写入发布记录。

模型服务还应定义 artifact 分发协议。权重、tokenizer、chat template、engine config 和 LoRA/adapter 不能以“目录里有什么就加载什么”的方式分发，而应由 manifest 指定 digest、大小、权限、兼容 runtime、预热策略和回滚对象。副本进入 `Warming` 前先校验 manifest，再检查本地缓存或拉取 artifact；进入 `Ready` 前必须完成最小推理探针。这样才能把模型加载慢、缓存 miss、registry 异常和权重损坏区分开。

```yaml
model_artifact_distribution:
  release_id: af-chat-large-20260619-r3
  manifest_digest: sha256:example
  artifacts:
    weights:
      digest: sha256:weights
      size: recorded
    tokenizer:
      digest: sha256:tokenizer
    chat_template:
      digest: sha256:template
  cache:
    required_before_canary: true
    local_nvme_path: managed
    verify_on_read: true
  readiness:
    manifest_verified: required
    warmup_probe: required
    cache_hit_or_pull_time_recorded: required
```

Artifact 分发协议是推理冷启动治理的基础。没有它，autoscaler 只能看到副本还没 ready，却不知道是在拉镜像、拉权重、校验 digest、初始化 engine，还是被存储限流。

Artifact 分发之前还要有 `model_artifact_provenance`。它说明一个线上可加载产物从哪个 checkpoint、base model、LoRA/adapter、tokenizer、chat template、量化配置、转换工具、评测门禁和签名流程生成。Digest 只能证明文件没有被改，不能证明它应该被使用；provenance 才能证明产物来源、转换链路和发布资格。没有 provenance，团队看到一个权重 digest 只能知道“它是什么文件”，无法知道“它是否来自批准的训练、是否通过评测、是否允许服务这个租户”。

```yaml
model_artifact_provenance:
  provenance_id: map-af-chat-large-20260619-r3
  release_id: af-chat-large-20260619-r3
  source:
    base_checkpoint: ckpt-step-120000
    checkpoint_restore_drill: crd-20260620-ckpt-120000
    dataset_lineage_records: [dlr-corpus-v3-2-20260620]
    fine_tune_artifacts: []
  build:
    conversion_tool: hf-to-runtime-v8@sha256:example
    quantization_profile: none_or_recorded
    tokenizer_digest: sha256:tokenizer
    chat_template_digest: sha256:template
    engine_config_digest: sha256:engine-config
  governance:
    quality_gate_execution: qge-af-chat-20260620-001
    safety_gate: pass
    license_review: pass
    signed_by: model-release-owner
    signature: sigstore-or-internal-signature
  output_artifacts:
    weights_digest: sha256:weights
    manifest_digest: sha256:example
    registry_entry: model-registry/af-chat-large/20260619-r3
```

这个对象是 model registry、serving release 和 PRR 的共同证据。若某个线上回答质量退化，评测平台可以沿 provenance 找到训练数据、checkpoint、转换工具和模板；若某个产物被发现许可证或安全问题，平台可以找出所有依赖它的 serving release、cache 和租户；若转换工具升级导致 token drift，release gate 可以要求重新生成 provenance。模型产物供应链应像容器镜像供应链一样可追溯、可签名、可撤销。

Artifact 分发还需要状态机和 `cache_residency`。同一个 release 在不同节点上可能处于未下载、下载中、校验中、已缓存、已预热、可服务或失效状态。Autoscaler 和调度器如果不知道这些状态，就会把副本调度到完全冷的节点，扩容窗口被权重拉取和校验放大。模型服务控制面应把 artifact cache 当成资源状态，而不是让 Pod 启动脚本临时处理。

![图：14.4.1 工程实现](../assets/diagrams/part-03-models-chapter-14-model-serving-04.svg)

```yaml
cache_residency:
  release_id: af-chat-large-20260619-r3
  artifact_digest: sha256:weights
  scope:
    resource_pool: inference-prod-h100
    node: gpu-node-042
    rack: rack-11
  state:
    artifact: cached
    tokenizer: cached
    engine_warmup: passed
    ready_for_endpoint: af-chat-large-prod
  timings:
    pull_ms: measured
    verify_ms: measured
    warmup_ms: measured
  policy:
    protect_while_endpoint_active: true
    evict_after_release_unused: policy_defined
```

这份状态能直接改变扩容策略。若某个 rack 的 `cache_residency` 已经 ready，调度器可以优先放置新 replica；若 cache miss 正在高峰期集中发生，Gateway 应降低新流量导入速度或提前扩容；若 digest mismatch，发布系统应冻结 canary，而不是让副本反复重启。模型冷启动不是单一耗时指标，而是 artifact 分发、校验、缓存、warmup 和 readiness 的组合路径。

缓存还需要 `cache_invalidation_record`。模型权重、tokenizer、chat template、engine config、LoRA、RAG 索引和数据 shard 一旦版本、权限、签名或安全状态变化，旧缓存可能从性能资产变成风险。Invalidation record 说明为什么失效、影响哪些节点和资源池、是否需要停止调度到这些节点、是否要保留证据、何时重新预热。没有失效记录，发布回滚、许可证撤销、数据删除和 digest mismatch 都可能留下“热但不可信”的缓存。

```yaml
cache_invalidation_record:
  invalidation_id: cir-20260620-001
  target:
    artifact_digest: sha256:weights
    release_id: af-chat-large-20260619-r3
    cache_scopes:
      - resource_pool: inference-prod-h100
        racks: [rack-11, rack-12]
  trigger:
    reason: manifest_revoked_or_template_changed
    linked_records:
      model_artifact_provenance: map-af-chat-large-20260619-r3
      serving_rollback_record: optional
  action:
    mark_cache_state: invalid
    block_new_replica_on_invalid_cache: true
    preserve_for_forensics: true
    prewarm_replacement_release: af-chat-large-20260612-r9
  outcome:
    affected_nodes: measured
    eviction_completed: true_or_false
    replacement_cache_ready: measured
```

`cache_invalidation_record` 让缓存进入发布和安全控制面。若某个 tokenizer 修复了严重 tokenization bug，旧 tokenizer cache 必须失效；若某个模型 artifact 被撤销，所有节点的 cache residency 都要从 ready 变成 invalid；若回滚需要旧版本热容量，失效策略又不能把旧稳定版本提前清理。缓存不是简单 LRU，它承载发布安全和恢复时间目标。

多模态模型服务还需要 `multimodal_serving_contract`。文本模型的 serving release 主要绑定权重、tokenizer、chat template、runtime 和 sampling；多模态服务还要绑定 media processor、image/audio/video encoder、tile 或 frame 策略、最大页数/帧数、派生产物引用、source region 格式、同步/异步边界和计量口径。否则，预处理管线和模型服务会各自升级：OCR 输出字段改了，模型仍按旧 schema 解析；抽帧策略变了，评测 gate 没有重跑；客户端要求引用页码，服务只返回一段文字。

```yaml
multimodal_serving_contract:
  contract_id: mmsc-claims-doc-20260620-r2
  endpoint: claims-document-review-prod
  model_artifacts:
    weights: registry/models/claims-mm@sha256:weights
    tokenizer: registry/tokenizers/claims-mm@sha256:tokenizer
    vision_encoder: registry/models/vision-encoder@sha256:vision
    processor_config: registry/processors/claims-mm@sha256:processor
  media_inputs:
    supported_types: [scanned_pdf, image]
    requires_media_artifact_manifest: true
    max_pages_or_tiles: policy_defined
    accepted_derived_artifacts:
      - ocr_text
      - layout_blocks
      - page_images
      - visual_embeddings
  runtime:
    inference_engine: vllm_or_sglang_or_custom_mm_runtime
    batching_policy: separate_text_prefill_from_media_prefill_if_supported
    streaming: answer_stream_after_media_ready
    cancellation_semantics: stop_decode_and_preserve_media_manifest
  output_contract:
    citations:
      required_for_claims: true
      coordinate_system: page_pixel_coordinates
      fields: [page, region, artifact_ref, confidence]
    usage:
      emit_multimodal_metering_event: true
      include_text_tokens: true
      include_media_units: true
  gates:
    multimodal_quality_gate_execution: mm-qge-claims-20260620
    media_pipeline_compatibility: pass
    source_region_replay: pass
    redaction_policy: pass
```

这个 contract 把模型服务和第 33 章的媒体 manifest 连接起来。请求进入 endpoint 时，服务不应直接读一个任意文件路径，而应消费已准入的 `media_artifact_manifest`；模型输出中的引用不应是自由文本，而应引用 manifest 中的 page、frame、time range 或 region；usage 事件不应只写 output token，而应写页数、tile、OCR token、视觉 encoder cost 和派生产物存储。这样，SRE 能定位媒体链路阶段，评测能重放同一派生产物，计费能对账，删除请求也能找到所有引用。

![图：14.4.1 工程实现](../assets/diagrams/part-03-models-chapter-14-model-serving-05.svg)

多模态 serving 的发布风险通常来自 schema 漂移，而不只是权重本身。OCR service 从 `bbox` 改成 `polygon`，视觉 encoder 调整 tile size，视频 pipeline 改变 frame sampling，processor config 调整 resize 规则，都可能改变模型输入和答案引用。`multimodal_serving_contract` 应把这些变化视为 serving 变更，触发 compatibility test、质量 gate 和成本漂移检查。不能因为权重 digest 没变，就认为多模态服务行为没变。

模型服务还应提供 drain contract。Drain contract 至少说明：停止接收新请求的信号是什么，已有 streaming 请求最大等待多久，超时后如何关闭，已生成 token 如何计量，KV Cache 如何释放，副本退出前必须上报哪些事件。没有 drain contract，滚动升级、节点维护和缩容都会随机打断用户。

```yaml
drain_contract:
  stop_new_requests: true
  max_drain_seconds: 900
  active_stream_policy: wait_until_complete_or_deadline
  on_deadline:
    close_reason: server_drain_timeout
    emit_usage: true
    release_kv_cache: true
  required_events:
    - drain_started
    - active_requests_snapshot
    - request_closed
    - replica_stopped
```

模型服务还需要 `serving_quality_contract`。它把第 13 章的 `quality_gate_record` 与实际 serving release 绑定，声明本次发布的权重、tokenizer、chat template、runtime、engine config、sampling 默认值、协议行为和质量门禁是一组不可拆开的生产契约。没有这个契约，模型团队可能说质量评测通过，服务团队却上线了不同 tokenizer 或 runtime 参数；SRE 看到线上质量退化时，也无法判断发布是否仍等同于当时评测的候选。

```yaml
serving_quality_contract:
  contract_id: sqc-af-chat-20260619-r3
  release_id: af-chat-large-20260619-r3
  bound_quality_gate: qg-af-chat-20260619
  artifacts:
    weights_digest: sha256:weights
    tokenizer_digest: sha256:tokenizer
    chat_template_version: support-v18
    prompt_policy_version: prompt-policy-202606
  runtime:
    engine: vllm
    engine_config_digest: sha256:engine-config
    runtime_image_digest: sha256:runtime
    sampling_defaults:
      temperature: recorded
      top_p: recorded
  protocol_contract:
    streaming_chunk_schema: openai-compatible-vX
    usage_timing: final_chunk
    finish_reason_set: [stop, length, tool_calls, content_filter]
    error_mapping_version: serving-errors-v6
  drift_tests:
    token_count_drift: required
    output_format_drift: required
    tool_call_schema_drift: required
    quality_regression_suite: support-quality-202606
```

这个契约应在发布前自动验证。第一步是 artifact 校验：digest、大小、权限和缓存状态一致。第二步是协议校验：固定请求的 streaming chunk、usage、finish reason、错误码和 tool call JSON 与 contract 相符。第三步是 token drift：同一组输入在新组合下的 input token 和输出长度分布是否变化，变化是否有解释。第四步是质量回归：运行与 `quality_gate_record` 绑定的评测集，确认没有严重回退。任何一步失败，都应阻止扩大流量。

![图：14.4.1 工程实现](../assets/diagrams/part-03-models-chapter-14-model-serving-06.svg)

`serving_quality_contract` 还应进入线上 trace。每个请求至少要能查到 contract id、release id、模型版本和 runtime profile。这样线上反馈进入第 1 章的 `quality_feedback_event` 后，评测平台可以知道应使用哪组 artifact 回放。若 contract id 缺失，线上请求与离线评测之间就断了，质量闭环只能停在“可能是新版本问题”。

回滚也需要结构化记录。`serving_rollback_record` 描述一次回滚从触发到完成的全过程：触发信号是什么，回滚了权重、tokenizer、runtime、Gateway route 还是 batch 配置，哪些请求和租户受影响，旧版本容量是否足够，证据是否被保留，后续需要打开哪些 `quality_regression_record` 或 `runtime_quality_gate`。没有 record，团队只能知道“回滚了”，但不知道回滚是否恢复了质量、是否污染了实验、是否丢失了事故现场。

```yaml
serving_rollback_record:
  rollback_id: srr-20260620-001
  trigger:
    source: canary_guardrail
    linked_records:
      engine_canary_record: engine-canary-20260620-001
      quality_gate_execution: qge-af-chat-20260620-001
      online_experiment_record: exp-support-model-20260619
    reason: citation_failure_rate_increase
  scope:
    endpoint: af-chat-large-prod
    from_release: af-chat-large-20260619-r3
    to_release: af-chat-large-20260612-r9
    tenants: [enterprise-a]
    traffic_percent: measured
  components_reverted:
    weights: true
    tokenizer: true
    chat_template: true
    runtime_image: false
    engine_config: false
    gateway_route: true
  preservation:
    request_samples_frozen: true
    quality_telemetry_window: retained
    runtime_diagnostic_bundle: retained_if_applicable
  outcome:
    rollback_started_at: recorded
    rollback_completed_at: recorded
    slo_recovered: true_or_false
    quality_guardrail_recovered: true_or_false
  follow_up:
    open_quality_regression_records: [qrr-20260620-0012]
    block_release_until_gate_passes: true
```

这个记录能避免两类事故。第一类是“半回滚”：只回滚权重，没有回滚 tokenizer 或 chat template，线上行为仍然不同；第二类是“证据清空”：回滚删掉了新版本副本和日志，导致后续无法解释质量退化。生产回滚既要快，也要保留足够证据。对高价值模型，回滚记录应进入 PRR 和 SRE 复盘，证明同类问题在下一次 release gate 中被覆盖。

回滚能力还必须演练，形成 `serving_rollback_drill`。`serving_rollback_record` 证明某次事故中做了什么，drill 证明在没有事故时平台也能按预期回滚。很多系统文档写着“支持回滚”，实际只验证了 deployment 镜像回退；真正的模型服务回滚还要覆盖权重、tokenizer、chat template、engine config、Gateway route、experiment variant、cache、warming、drain、metering 和 evidence preservation。没有演练，事故时才会发现旧权重已从节点 cache 清理、旧 tokenizer 没有签名、Gateway route 无法快速切回，或回滚后 usage 口径变化导致账单争议。

```yaml
serving_rollback_drill:
  drill_id: srd-af-chat-20260620
  endpoint: af-chat-large-prod
  baseline_release: af-chat-large-20260612-r9
  candidate_release: af-chat-large-20260619-r3
  drill_scope:
    traffic: shadow_or_internal_canary
    tenants: [internal_eval]
    request_slices:
      - short_chat
      - rag_citation
      - tool_call_json
      - long_context_streaming
  components_verified:
    gateway_route_weight: pass
    endpoint_state_transition: pass
    weights_and_adapter_digest: pass
    tokenizer_digest: pass
    chat_template_version: pass
    runtime_image_and_engine_config: pass
    cache_invalidation_record: pass
    model_artifact_distribution: pass
    streaming_drain: pass
    usage_and_metering_schema: pass
    quality_evidence_bundle_preserved: pass
  timing:
    detect_to_freeze: measured
    freeze_to_route_restore: measured
    route_restore_to_quality_recovery_probe: measured
  recovery_probes:
    protocol_contract_test: pass
    token_drift_back_to_baseline: pass
    golden_slice_smoke: pass
    cost_metering_reconciliation: pass
  decision:
    rollback_capability_valid_until: next_major_runtime_or_artifact_change
    blockers: []
```

这个演练应在三类变化后自动失效：第一，artifact 供应链变化，例如权重、adapter、tokenizer、模板、签名和分发路径改变；第二，runtime 变化，例如推理引擎、container runtime、GPU driver、CDI/NRI、batch 策略或 PD 分离策略改变；第三，Gateway 和计量变化，例如路由策略、experiment 平台、usage schema、价格版本或计费 pipeline 改变。失效后，PRR 不应接受“理论可回滚”的说法，而要重新跑 drill。回滚是生产能力，不是文档承诺。

![图：14.4.1 工程实现](../assets/diagrams/part-03-models-chapter-14-model-serving-07.svg)

对在线大模型，回滚演练还要验证“回滚不会制造第二次事故”。例如长 streaming 请求在 drain 期间应完成或按协议终止，不能被新旧 release 同时计费；回滚到旧 tokenizer 后，正在执行的 RAG context 和 tool call JSON 不能被旧模板误解析；旧版本容量不足时，Gateway 应降低流量而不是把所有请求压回 baseline；证据保留应发生在删除 candidate 副本之前。一次合格的 drill 至少要证明回滚路径、容量路径、证据路径和计量路径同时成立。

模型服务还应向 Gateway 暴露 `engine_admission_health`。这是 endpoint 级健康摘要，生成方通常是 model server 或 serving control plane，消费方是 AI Gateway、autoscaler 和 SRE。它把内部 queue、active sequence、KV block、deadline miss、drain、canary freeze 等状态压缩成可路由信号，避免 Gateway 把流量送到端口正常但 runtime 已经不可承诺 SLO 的副本组。

```yaml
engine_admission_health:
  endpoint: af-chat-large-prod
  release_id: af-chat-large-20260619-r3
  serving_quality_contract: sqc-af-chat-20260619-r3
  engine_profile: vllm-prod-h100-v7
  state: routable
  capacity:
    short_context_slots: available
    long_context_slots: limited
    active_sequences: measured
    queue_deadline_miss_rate: measured
  kv_cache:
    block_pressure: medium
    allocation_failure_rate: zero
    prefix_cache_hit_rate: measured
  lifecycle:
    draining: false
    canary_frozen: false
    last_engine_restart: none_recent
  recommendation:
    gateway_action: route_short_context_allow_long_context_with_weight
    autoscaler_action: no_change
```

这个对象不能代替引擎内部 scheduler，也不应包含每个请求的敏感内容。它的职责是把模型服务“还能接什么流量”表达清楚。对发布系统来说，canary 期间如果 `queue_deadline_miss_rate` 或 `kv_cache.allocation_failure_rate` 上升，应冻结放量；对 Gateway 来说，如果 long context limited，就可以把长上下文请求路由到专用池；对 SRE 来说，健康摘要能解释为什么 endpoint 没有 5xx，但 TTFT 已经开始变差。

模型服务还应汇总请求级 `kv_block_ledger`。推理引擎负责具体 block 分配和释放，Serving 层负责把这些账本按 endpoint、release、tenant、request bucket 和 close reason 汇总成可运营信号。这样 autoscaler、Gateway 和成本系统看到的不是“显存使用率 92%”这种模糊状态，而是哪些请求形态占用了 KV、占用多久、是否产生有效 token、取消后是否及时释放。

```yaml
kv_block_ledger_rollup:
  endpoint: af-chat-large-prod
  serving_release: af-chat-large-20260619-r3
  engine_profile: vllm-prod-h100-v7
  window: 2026-06-20T10:00Z/2026-06-20T10:05Z
  dimensions:
    tenant: enterprise-a
    request_bucket: long_context_streaming
    close_reason: client_cancelled
  kv_usage:
    peak_blocks: measured
    block_seconds: calculated
    prefix_cache_reused_blocks: measured
    allocation_failures: measured
    unreleased_after_close_blocks: zero_or_measured
  serving_actions:
    gateway_hint: reduce_long_context_weight
    autoscaler_hint: scale_decode_or_separate_pool
    investigation_hint: check_cancel_release_path
```

这个 rollup 是 Serving 层和 Runtime 层的边界。Runtime 不应该知道商业租户策略，Serving 也不应该重新实现 KV allocator；两者通过账本关联。若某个 release 上线后 `block_seconds` 上升但 output token 没有增加，可能是 prompt 模板、prefix cache key、取消释放路径或 engine 变更造成浪费。若 allocation failures 只集中在长上下文租户，Gateway 应调整路由或上下文限制，而不是全局扩容。

当 rollup 显示取消、超时或 worker restart 后仍有 `unreleased_after_close_blocks`，Serving 层应自动生成 `kv_block_leak_forensic_record`，并把它挂到 incident、runtime canary 和成本账本。这个记录不是 runtime 内部 debug 文件，而是跨 Gateway、model server、engine 和计量系统的生命周期对账。它至少要证明三件事：请求关闭信号是否到达所有组件，KV allocator 是否释放了引用，泄漏是否影响后续 admission。

```yaml
kv_block_leak_forensic_record:
  forensic_id: kv-leak-20260620-001
  endpoint: af-chat-large-prod
  serving_release: af-chat-large-20260619-r3
  engine_profile: vllm-prod-h100-v7
  trigger:
    signal: unreleased_after_close_blocks
    detected_by: kv_block_ledger_rollup
    window: 2026-06-20T10:00Z/2026-06-20T10:05Z
  lifecycle_reconciliation:
    gateway_close_event: client_cancelled
    model_server_close_event: propagated
    engine_close_event: missing_or_delayed
    metering_close_event: request_closed
  allocator_evidence:
    orphaned_blocks: measured
    owning_sequence_ids: sampled
    prefix_cache_refs: sampled
    release_latency_ms: measured
  production_impact:
    admission_rejects_caused: calculated
    kv_block_seconds_wasted: calculated
    affected_tenants: [enterprise-a]
  action:
    immediate: freeze_long_context_admission_or_restart_worker
    permanent: fix_cancel_release_path_and_add_regression
```

这份取证记录能阻止“重启副本就算修复”的坏习惯。重启可以释放显存，但不能解释泄漏发生在哪个生命周期边界；如果不把 close event、allocator owner 和 admission impact 关联起来，同类问题会在下一次客户端取消风暴、网关超时调整或 PD transfer 故障中复发。Serving 层的职责是把 runtime 细节翻译成可运营事实。

引擎或 runtime 变更还需要 `engine_canary_record`。它比普通 canary 更细，因为 runtime 变更可能同时影响协议、token 口径、KV block、prefix cache、speculative decoding、错误码和成本。记录应绑定 serving release、runtime quality gate 和线上样本，避免“压测通过但线上格式/计费/质量漂移”的事故。

```yaml
engine_canary_record:
  canary_id: engine-canary-20260620-001
  endpoint: af-chat-large-prod
  from_engine_profile: vllm-prod-h100-v6
  to_engine_profile: vllm-prod-h100-v7
  traffic_slice:
    tenants: [internal_eval, enterprise-a-canary]
    percent: 5
    request_buckets: [short_chat, long_context_qa, tool_calling]
  gates:
    protocol_contract: pass
    token_count_drift: explained
    ttft_p95_regression: pass
    tpot_p95_regression: pass
    kv_allocation_failure_rate: pass
    output_format_regression: pass
    quality_feedback_guardrail: pass
    cost_per_successful_answer_delta: pass
  decision:
    state: continue_or_freeze_or_rollback
    rollback_target: vllm-prod-h100-v6
```

Canary 的动作也应被结构化保存为 `engine_canary_guardrail_action`。`engine_canary_record` 说明观测结果，guardrail action 说明系统实际做了什么：冻结放量、摘除 endpoint、降低长上下文权重、关闭 speculative decoding、回滚 engine profile，还是只延长观察窗口。没有动作记录，团队只能知道“guardrail 触发过”，不知道当时有没有止血、止血范围多大、是否影响 SLA 和成本。

```yaml
engine_canary_guardrail_action:
  action_id: ecga-20260620-001
  canary_id: engine-canary-20260620-001
  trigger:
    guardrail: kv_allocation_failure_rate
    observed_delta: regressed
    affected_slices: [long_context_qa]
  automated_action:
    freeze_traffic_expansion: true
    route_weight_change:
      from_candidate_percent: 5
      to_candidate_percent: 0
    gateway_policy_patch: ead_policy_patch_20260620_001
    runtime_feature_patch:
      speculative_decoding: disabled_if_related
  evidence_preserved:
    endpoint_admission_decisions: sampled
    kv_block_leak_forensic_record: kv-leak-20260620-001
    inference_runtime_diagnostic_bundle: inc-20260620-ttft-001
  outcome:
    slo_recovered: true_or_false
    cost_impact_recorded: inference_runtime_cost_ledger
```

这个动作对象把发布系统、Gateway 和 runtime 连接起来。它要求自动止血必须留下证据，不能只改一个配置开关；也要求成本系统知道 canary 预留容量和回滚带来的资源浪费。对高价值 endpoint，guardrail action 应进入 SRE 复盘：如果动作太慢，说明检测或自动化不足；如果动作过于激进，说明护栏阈值或切片定义需要修订。

如果服务采用 PD 分离，还需要 `pd_disaggregation_contract`。它声明 prefill pool、decode pool、KV 传输、失败语义、容量比例和回滚方式。PD 分离把一个 replica 内部阶段拆成跨资源池链路，如果没有合同，Gateway、autoscaler 和 SRE 很难判断 TTFT 变差是 prefill 池不足、KV 传输慢、decode 池拥塞，还是两边容量比例失衡。

```yaml
pd_disaggregation_contract:
  endpoint: af-chat-large-pd-prod
  mode: prefill_decode_disaggregated
  pools:
    prefill_pool: af-chat-prefill-h100
    decode_pool: af-chat-decode-h100
  state_transfer:
    kv_transfer_protocol: platform_defined
    max_kv_transfer_ms: policy_defined
    encryption_or_isolation: required_by_tenant_boundary
  admission:
    prefill_queue_budget_ms: policy_defined
    decode_queue_budget_ms: policy_defined
    capacity_ratio_prefill_to_decode: measured_and_configured
  failure_semantics:
    prefill_failed_before_first_token: retry_or_fallback_allowed
    kv_transfer_failed: close_with_typed_error
    decode_failed_after_streaming: no_fallback_emit_partial_usage
  rollback:
    fallback_to_monolithic_serving: required
```

PD 分离合同的价值在于把复杂性显式化。它允许平台先用影子流量和低风险租户验证 KV 传输、容量比例和失败语义，再决定是否让核心 Chat 流量进入。没有这个合同，PD 分离很容易只在 benchmark 中证明 prefill 更快，却在线上引入新的状态传输故障和账单争议。

合同之外，还需要请求级或窗口级 `pd_transfer_evidence`。合同描述“应该如何传”，evidence 描述“实际如何传”。它记录一次 prefill 到 decode 的状态转移，包括 KV 大小、传输时延、完整性校验、租户隔离、重试、失败阶段和对 TTFT/TPOT 的贡献。没有这个对象，PD 分离事故会退化成 prefill 团队、decode 团队和网络团队互相猜测。

```yaml
pd_transfer_evidence:
  transfer_id: pdx-20260620-001
  request_id: req-20260620-001
  endpoint: af-chat-large-pd-prod
  pd_disaggregation_contract: pd-contract-20260620
  pools:
    prefill_replica: prefill-7
    decode_replica: decode-3
  kv_state:
    context_tokens: 8192
    kv_blocks_transferred: measured
    checksum_or_integrity: pass
    tenant_isolation_context: enterprise-a
  timing:
    prefill_queue_ms: measured
    prefill_compute_ms: measured
    kv_transfer_ms: measured
    decode_admission_wait_ms: measured
  failure_semantics:
    retries: measured
    fallback_allowed: before_first_token_only
    partial_usage_emitted: true_or_false
  verdict:
    transfer_within_contract: true_or_false
    bottleneck: prefill_queue_or_transfer_or_decode_wait
```

PD 分离是否值得上线，不能只看平均 TTFT。若 `pd_transfer_evidence` 显示 transfer 时延长尾抵消了 prefill 池收益，或者 decode admission wait 长期高于 prefill compute 节省，系统应回到 monolithic endpoint 或重调两池比例。若 integrity 或租户隔离证据缺失，即使性能好也不能进入高敏租户。PD 分离把一次推理变成分布式状态机，因此证据粒度必须跟着升级。


### 14.4.2 常见故障

第一类故障是健康检查过浅。服务端口可用，但模型尚未加载、GPU 不可用、tokenizer 初始化失败或首个推理请求失败。Kubernetes 认为 pod ready，Gateway 开始导流，用户请求失败。健康检查应包含模型 readiness 和最小推理探针，而不只是 HTTP 端口。

第二类故障是 autoscaler 指标错误。只看 CPU 或请求数，无法反映 GPU、HBM、KV Cache 和队列压力；流量峰值时副本扩容滞后；缩容时中断 streaming。解决方向是使用 token、queue、latency、GPU 和 cache 指标，并配合预热和 drain。LLM autoscaling 需要专用信号。

第三类故障是发布版本不完整。模型权重、tokenizer、runtime、prompt 模板和配置没有绑定，回滚时无法恢复旧行为；canary 只标记镜像版本，trace 中看不到模型版本。生产模型版本必须是组合版本，并写入所有请求上下文。否则发布事故难以定位。

第四类故障是 batching 参数不适配 workload。为了追求吞吐增大 batch，导致 TTFT 变差；为了低延迟减小 batch，导致 GPU 利用率过低；长短请求混部造成 P99 抖动。排查需要同时看 batch size、queue time、prefill/decode、token 分布和租户维度。Batching 故障常被误判为模型慢。

第五类故障是升级破坏协议。Runtime 或 model server 升级后，streaming chunk、usage 字段、错误码或 tool calling 输出发生细微变化，上游解析失败。兼容性测试缺失时，这类问题常在灰度后才暴露。协议也是模型服务契约的一部分。

第六类故障是缺少排水。副本下线或节点维护时仍有 streaming 请求，用户看到中断。排水失败通常不是模型问题，而是服务生命周期管理问题。发布和维护流程都应验证 drain 行为。

还有一类故障是观测缺失。请求失败了但没有 endpoint、replica、model version 和 error stage，事故复盘只能猜测。模型服务必须先可观测，再谈自动化。

第七类故障是 admission 与 runtime 状态脱节。Gateway 认为 endpoint 健康，model server 实际 KV Cache 接近上限或 queue 已经超过 SLO，结果继续接流量造成雪崩。解决方向是让 model server 暴露 admission health，包括 queue p95、active sequence、KV pressure、draining state 和 engine restart，Gateway 按这些信号做可路由健康判断。

第八类故障是 contract test 缺失。新 runtime 版本在样例请求上能回答，但 usage 字段、finish reason、streaming chunk、错误码或 tool calling JSON 与旧版本不同，上游应用和计量系统被破坏。模型服务升级必须把协议契约作为测试对象，而不是只看模型输出文本。

第九类故障是 serving release 与质量门禁脱钩。候选模型在某个 tokenizer、模板、runtime 和 sampling 配置下通过评测，但上线时为了性能调整了 engine config 或默认参数，输出分布和 token 口径发生变化。解决方向是发布必须绑定 `serving_quality_contract`，任何 artifact 或 runtime 变化都要重新计算 token drift 和质量回归，而不是沿用旧 gate。

第十类故障是 canary 只看服务指标。新版本 TTFT、TPOT、错误率都正常，但引用正确率下降、工具 JSON 不稳定或输出更长导致成本上升。模型服务 canary 应消费第 13 章的质量指标和第 6 章的线上实验护栏。服务指标保护可用性，质量指标保护用户价值，二者缺一不可。

第十一类故障是回滚未经演练。事故中发现旧 release 的权重还在 registry，但节点 cache 已清理，旧 tokenizer 没有随 release 签名，Gateway route 只能回滚流量不能回滚 experiment sticky assignment，或者 usage schema 回退后账单 pipeline 不兼容。解决方向是维护 `serving_rollback_drill`，并让 artifact、runtime、Gateway、cache 和计量任一关键变化自动要求重跑。没有 drill 的回滚能力不能进入高 SLA 承诺。


### 14.4.3 性能指标

请求指标包括 QPS、并发、成功率、错误率、取消率、超时率、streaming duration 和 fallback 率。它们回答服务是否可用、谁受影响、错误在哪里发生。请求指标应按 endpoint、model version、replica、tenant 和 route 切分。全局平均值无法支撑发布回滚。

Token 指标包括 input tokens/s、output tokens/s、total tokens/s、tokens per request、TTFT、TPOT、finish reason 和 output length distribution。它们回答服务负载和用户体验。LLM 服务容量和成本都与 token 强相关，请求数只是辅助指标。没有 token 指标，autoscaling 和成本分析都会失真。

Runtime 指标包括 queue length、queue time、batch size、prefill time、decode time、KV Cache 使用、active sequence、cache allocation failure、engine restart 和 model load time。它们回答瓶颈在模型服务内部哪里。Runtime 指标是调参、扩容和排障的关键。

资源和发布指标包括 GPU utilization、HBM、功耗、节点健康、冷启动时间、canary 指标、回滚时间、drain 时间和版本错误率。它们回答基础设施和发布是否健康。模型服务是长期运行的生产系统，发布指标和运行指标同等重要。一次糟糕发布可能抵消所有性能优化收益。

指标还应支持容量和成本分析。按 endpoint 和版本统计 tokens/s、空闲副本、扩容次数、拒绝请求和 cost per token，才能判断服务策略是否经济。模型服务不是只追求稳定，也要持续提高资源效率。

还应关注指标基数。按租户、模型、endpoint 和 replica 切分很有价值，但过多动态标签会压垮监控系统。标签设计要稳定，临时调试字段应有采样和保留期限。

指标也要服务 SLO。没有目标阈值的指标只是曲线，有 SLO 才能触发扩容、告警和回滚。

每个关键 endpoint 都应有独立 SLO。

指标还应进入容量例会和发布复盘。若某个版本提升吞吐但增加错误，或降低延迟但提高成本，团队需要基于同一组指标做取舍。模型服务指标的价值在于驱动容量、发布和成本决策，而不是只展示运行状态。

建议把模型服务指标分成 admission、execution、lifecycle、release 四个面板。Admission 看入口是否还能接请求，execution 看运行时是否高效，lifecycle 看副本启动和排水是否可靠，release 看版本变化是否安全。

| 面板 | 指标 | 说明 |
| --- | --- | --- |
| Admission | queue length、queue p95、admission reject、KV pressure | 决定 Gateway 是否继续导流 |
| Execution | prefill tokens/s、decode tokens/s、TTFT、TPOT、batch size | 决定 serving 配置和引擎调优 |
| Lifecycle | model load time、warmup time、drain time、active streams on drain | 决定扩缩容和维护窗口 |
| Release | canary error delta、token drift、protocol test result、rollback time | 决定发布是否继续 |

这里的 token drift 很关键。模型、tokenizer、chat template 或工具 schema 变化后，同一组标准请求的 input token 和 output token 分布可能改变。若 token 数漂移没有解释，成本、延迟和账单都会受影响。发布门禁应把 token drift 当成一等信号。

服务质量契约指标包括 contract 覆盖率、发布请求中 contract id 缺失率、token drift 失败数、协议契约失败数、质量回归失败数、canary 期间质量反馈率、contract 与 gate 不一致次数和回滚后 contract 恢复时间。这些指标衡量模型服务是否真的按评测过的组合运行。一个平台如果经常出现“评测通过但线上不是同一组合”，说明模型服务治理还不合格。

回滚演练指标包括 drill 覆盖的 endpoint 比例、最近一次有效 drill 距离当前 release 的变更跨度、detect-to-freeze、freeze-to-route-restore、baseline capacity readiness、cache invalidation replay 通过率、usage reconciliation 通过率、drain 成功率、回滚后 golden slice smoke 通过率和 drill blocker 关闭时长。它们衡量的是“事故来临前是否真的能退回去”。回滚时间短但证据丢失、计量错乱或质量未恢复，都不能算合格回滚。


### 14.4.4 设计取舍

第一个取舍是独立部署与共享部署。独立部署隔离强、性能稳定、排障简单，但资源利用率低；共享部署或 multi-model serving 提高利用率，但增加加载、隔离和观测复杂度。高 SLA 大模型适合独立部署，低频小模型或 adapter 场景可以考虑共享。部署形态应由流量和风险决定。

第二个取舍是吞吐与延迟。更激进 batching 提高 GPU 利用率和降低 cost per token，但可能伤害 TTFT 和 P99；更保守 batching 改善交互体验，但成本更高。平台应按应用类型设置策略：代码补全和 Chat 重视低 TTFT，批量推理重视吞吐。没有统一最优参数。

第三个取舍是自动扩缩容与容量预留。Autoscaling 能降低空闲成本，但大模型冷启动慢，高峰期扩容滞后会影响 SLA；容量预留提高稳定性，但空闲成本高。Premium 服务通常需要热容量，best-effort 服务可以更依赖弹性。成本优化不能破坏服务等级承诺。

最后是发布速度与发布安全。频繁发布能快速迭代模型和 runtime，但每次变更都可能影响输出、协议和成本；严格门禁降低风险，但可能拖慢改进。可行路径是自动化 canary、指标门禁和快速 rollback。让发布安全，不是减少发布次数，而是降低每次发布的爆炸半径。

取舍还包括通用 serving 平台与专用优化。统一平台降低维护成本，专用部署能为大客户或关键模型优化极致性能。平台应先统一控制面和观测，再允许数据面按场景优化。这样既能复用治理能力，也能保留性能空间。

最终取舍应由 workload profile 决定。流量稳定、高价值、低延迟的模型，和低频、best-effort 的模型，不应使用同一种 serving 策略。

回滚还有一个取舍：保留旧版本容量与降低空闲成本。高价值 endpoint 保留 baseline 热容量和 cache，会增加成本，但能在质量或协议事故中快速止血；低价值或低频 endpoint 可以接受较慢回滚，但必须在 SLA 中说明。工程上不应把所有模型都做成最高等级，而应按租户价值、请求风险、输出副作用和业务承诺确定 rollback class。Rollback class 应进入 endpoint contract 和 PRR。


## 14.5 小结与延伸阅读

### 14.5.1 小结

- 模型服务是模型能力生产化的执行层，连接网关、推理引擎和 GPU。
- Batching、autoscaling、canary 和 rollback 是模型服务的关键机制。
- LLM autoscaling 应关注 token、队列、GPU 和 KV Cache，而不是只看 CPU。
- 模型发布必须版本化权重、tokenizer、runtime 和配置。
- `serving_quality_contract` 把模型服务发布与质量门禁绑定，确保线上运行的 artifact/runtime 组合就是被评测、可回放、可回滚的组合。
- `serving_rollback_drill` 证明回滚不是纸面能力，而是覆盖 artifact、runtime、Gateway、cache、drain、计量和质量探针的生产演练。


### 14.5.2 延伸阅读

- [vLLM documentation](https://docs.vllm.ai/)；[SGLang documentation](https://docs.sglang.ai/)；[TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [MLflow Model Registry documentation](https://mlflow.org/docs/latest/model-registry.html)
