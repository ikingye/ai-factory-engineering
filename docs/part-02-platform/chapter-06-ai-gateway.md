# 第 6 章：AI Gateway

## 本章回答的问题

- 为什么 AI Factory 需要专门的 AI Gateway，而不是普通 API Gateway？
- 认证鉴权、流量治理、模型路由、fallback、灰度和多模型聚合如何协同？
- Envoy AI Gateway 与 Gateway API Inference Extension 代表了什么方向？

## 一个真实场景

一个 MaaS 平台早期直接把应用流量转发到模型服务，普通 API Gateway 只负责 TLS、鉴权和 HTTP 路由。上线初期流量不大，看起来没有问题；租户增多后，故障开始连锁出现。某个租户突然提交大量长上下文请求，模型服务 prefill 队列堆积，其他短请求 TTFT 被拖慢；另一个租户开启 streaming 后连接保持很久，占满网关连接池；模型版本灰度没有按租户隔离，客服应用输出风格发生变化；后端模型超时后 fallback 到另一个模型，却因为 tool calling 格式不同导致应用解析失败。

这些问题说明普通 API Gateway 不理解模型流量的关键语义。普通网关可以按路径、header、QPS 和 upstream 健康转发请求，但 AI 流量还包含模型名、上下文长度、input/output token、streaming、工具调用、模型能力、租户预算、SLA 和成本。一个 200 状态码的请求可能生成了错误格式；一个 QPS 很低的租户可能消耗大量 TPM；一个 fallback 成功的 HTTP 响应可能让业务结果变差。

AI Gateway 的价值，是把模型流量治理前置到入口。它既保护后端推理资源，也保护应用免受模型版本、能力差异和后端故障的直接冲击。它不是“更懂 AI 的反向代理”这么简单，而是 AI 平台的数据面策略执行点：在每个请求进入模型服务前，完成身份、配额、能力、路由、风险、观测和计量的基本判断。

如果缺少这个入口治理，平台会在规模扩大后被迫把策略补到各处。应用自己判断模型能力，模型服务自己做限流，计费系统事后补 token，SRE 用日志猜测租户影响面，安全团队再去追工具权限。每个局部方案都可能暂时有效，但整体不可审计。AI Gateway 的目标，是让这些策略在同一入口按同一对象模型执行，并让结果能被追踪。

## 核心概念

AI Gateway 是面向模型 API、RAG 和 Agent 流量的网关层。它承担认证鉴权、租户识别、配额、限流、模型路由、fallback、灰度、参数校验、请求改写、响应处理、token 计量、日志、metrics、trace 注入和安全策略等能力。它位于应用和模型服务之间，是 MaaS 数据面的核心组件。没有 AI Gateway，MaaS 控制面中的模型目录、配额和服务等级很难落实到每个请求。

AI Gateway 与普通 API Gateway 的差异在于流量语义。普通 API Gateway 主要理解 HTTP 连接、路径、方法、证书、认证和 upstream；AI Gateway 还要理解模型能力、context window、token 预算、streaming 生命周期、tool calling、RAG/Agent 调用树和后端推理状态。它的限流不能只看请求数，路由不能只看服务地址，fallback 不能只看可用性，灰度不能只看 5xx。

AI Gateway 也不是推理引擎。它不负责具体 batching、KV Cache、CUDA kernel 或 NCCL 通信；这些属于模型服务和 AI Runtime。Gateway 的职责是入口治理：让请求以正确身份、正确预算、正确模型能力和正确观测标签进入后端。职责边界清楚，才能避免网关变成复杂单点，也避免治理逻辑散落到每个应用。

因此，AI Gateway 的成功标准不是“转发成功率高”，而是“策略执行一致且可解释”。一个请求被拒绝，应能说明是认证失败、配额超限、模型无权限、上下文超限还是安全策略拦截；一个请求被路由，应能说明命中的模型版本和资源池；一个请求被 fallback，应能说明原因和后果。可解释性是网关治理能力的一部分。

这也是它区别于普通网关的核心。

否则入口层只是在转发流量。

治理结果必须可验证。

验证失败时应阻断请求。

默认拒绝更安全。

## 系统架构

AI Gateway 通常分为控制面和数据面。控制面接收模型目录、租户、配额、路由、灰度、fallback、安全策略和服务等级配置；数据面处理每个请求，执行认证、限流、能力校验、路由选择、trace 注入、streaming 管理和响应计量。控制面决定规则，数据面执行规则。两者之间需要版本化配置和可回滚能力，否则策略变更本身会成为事故来源。

数据面处理请求时，顺序很重要。通常先识别调用者和租户，再检查 API Key、项目权限和模型访问权限；随后根据请求参数估算或校验上下文、输出预算和工具能力；再执行配额与限流；接着根据模型目录、服务等级、健康状态和灰度规则选择 upstream；响应返回时记录 token、延迟、错误码、fallback、取消和账单事件。每一步都应进入 trace。

架构上，AI Gateway 需要与模型服务共享健康和负载信息。后端是否能接收请求，不只取决于进程是否存活，还取决于队列长度、KV Cache、GPU/HBM 状态、错误率和模型版本。Gateway 不需要掌握所有 runtime 细节，但至少应获得可路由的健康信号。否则它只能做静态转发，无法承担 AI 平台治理职责。

配置发布也要纳入架构设计。路由规则、限流阈值、fallback 目标和灰度比例都可能影响全站流量，必须有版本、审批、灰度和回滚。网关策略错误的影响范围通常比单个模型服务更大，因为它位于入口。生产系统应把 Gateway 配置当作高风险配置管理，而不是临时修改的 YAML。

策略发布应有审计、预览和回滚。

并且要确认所有网关实例已同步。

实例一致性应自动检查。

检查失败应阻止继续放量。

```mermaid
flowchart LR
  Client["应用 / SDK"] --> GW["AI Gateway"]
  GW --> Auth["认证鉴权"]
  Auth --> Quota["租户限流 / 配额"]
  Quota --> Policy["策略\n灰度 / fallback / 安全"]
  Policy --> Router["模型路由"]
  Router --> M1["Model Endpoint A"]
  Router --> M2["Model Endpoint B"]
  Router --> M3["Private Endpoint"]
  GW --> Meter["token 计量"]
  GW --> Trace["trace / logs / metrics"]
  Meter --> Billing["Billing"]
```

## 6.1 为什么需要 AI Gateway

AI Gateway 首先解决统一接入问题。一个 AI Factory 中可能同时存在自研模型、开源模型、第三方 API、租户私有模型、实验模型和不同推理引擎。应用不应直接依赖这些后端的地址、认证、参数和错误格式。Gateway 对上提供稳定 API，对下适配不同模型服务，使模型迁移、供应商切换、版本升级和资源池调整不必每次都改应用。

第二个问题是统一治理。租户、项目、API Key、配额、限流、审计、计量和安全策略必须在入口处执行。若治理散落在应用和模型服务中，不同团队会形成不同口径：有的按 QPS 限流，有的按 token 限流，有的不记录 streaming 取消，有的不记录 fallback。AI Gateway 让平台可以在同一入口执行一致策略，并把策略结果写入 trace 和账单。

第三个问题是统一演进。模型升级、灰度、fallback、多模型聚合、安全策略和成本优化都需要在不破坏应用的情况下迭代。Gateway 是流量切分和回滚的关键点。没有 Gateway，模型服务变更会直接暴露给应用；有 Gateway，平台可以按租户、项目、用户、比例或请求特征逐步放量，并在发现质量或成本异常时回滚。AI Gateway 是模型平台规模化演进的缓冲层。

还有一个问题是统一证据。入口处最容易绑定 request id、tenant、project、model、route、quota 和 token 计量。若这些信息在网关处缺失，后续模型服务、计费和观测系统只能各自补字段，口径难以一致。AI Gateway 让每个请求从进入平台开始就带着正确标签，这对账单、SLO 和事故复盘都很关键。

标签越早绑定，后续证据链越完整。

入口标签也是计费标签的来源。

缺失标签会污染账单。

## 6.2 认证鉴权

认证回答“调用者是谁”，鉴权回答“调用者能做什么”。AI Gateway 应把 API Key、服务账户、用户身份、租户、项目、模型权限和服务等级绑定起来。对普通 Chat，请求只需确认调用者能否使用某个模型；对 RAG 和 Agent，还要把用户身份传递给检索、工具和业务 API，确保下游按同一权限边界执行。模型请求不应成为绕过企业权限系统的通道。

鉴权不能只在入口做一次。RAG 检索需要文档权限，Agent 工具调用需要业务权限，私有模型访问需要租户隔离，高风险操作可能需要二次确认。AI Gateway 可以注入租户、项目、用户、trace id 和策略上下文，让下游服务继续做细粒度校验。它也可以拦截明显不允许的模型访问和参数组合，减少无效请求进入后端。

安全工程上还要处理凭据生命周期。API Key 泄露、租户权限变化、项目迁移、人员离职、模型访问下线，都应能快速影响网关鉴权。Gateway 应支持 key 禁用、来源限制、异常调用告警和审计查询。认证鉴权不是一次性接入功能，而是 MaaS 和企业安全体系的持续同步点。任何身份漂移，最终都会变成成本和数据风险。

鉴权还要关注“代理身份”问题。很多企业应用用服务账户调用 MaaS，但实际用户不同；如果 Gateway 只识别服务账户，下游 RAG 和工具就无法按真实用户权限过滤。更稳妥的做法是同时传递应用身份和用户身份，并明确哪些动作使用应用权限，哪些动作使用用户权限。Agent 场景尤其需要这种区分，否则工具调用很容易扩大权限面。

权限上下文应随 trace 一起传递。

## 6.3 流量治理

AI 流量治理包括限流、熔断、超时、重试、并发控制、streaming 连接控制、token 预算和任务预算。LLM 请求不能只按 QPS 限制，因为资源压力更接近 input token、output token、上下文长度、并发序列和模型成本。Gateway 应支持 RPM、TPM、并发请求、最大上下文、最大输出 token、streaming 连接数、每租户预算和每项目预算。Agent 还需要每 run 模型调用数、工具调用数和总 token 限制。

超时和重试策略必须谨慎。普通 HTTP 请求失败后重试通常安全，但 LLM streaming 已经输出部分 token 后重试，可能产生重复内容；Agent 工具调用失败后重试，可能重复发送邮件、提交工单或执行写操作。Gateway 应区分连接失败、后端超时、模型拒绝、限流、安全拒绝、客户端取消和部分输出。不同错误类型对应不同重试和计费语义。

流量治理还要服务公平性和成本控制。高价值租户可以有更高优先级或专属资源池，实验流量可以更低优先级，批量任务可以排队或异步执行。若所有流量共享同一限流规则，短请求会被长请求拖慢，生产业务会被实验流量影响。AI Gateway 应把流量治理与服务等级、资源池和成本预算绑定，而不是只做全局速率限制。

治理策略还需要用户可见。被限流时，应用应知道是 RPM、TPM、并发、预算还是上下文超限，并获得可操作建议，例如稍后重试、缩短上下文、申请配额或转异步任务。模糊的 429 会让应用盲目重试，进一步放大流量。AI Gateway 的错误响应应是平台契约的一部分。

清晰错误能减少无效重试。

错误语义本身就是治理接口。

应用依赖它做退避。

## 6.4 模型路由

模型路由根据请求属性选择后端 endpoint。路由依据包括模型名、租户、项目、区域、SLA、成本、模型能力、灰度规则、健康状态、上下文长度、工具调用需求和数据驻留要求。一个高质量路由系统必须理解 capability：是否支持 tool calling、JSON 输出、多模态、最大上下文、embedding 或指定安全策略。能力不匹配的路由，比后端失败更隐蔽，因为它可能返回 HTTP 成功但业务结果错误。

路由还要考虑后端 runtime 状态。模型服务不是普通无状态服务，模型加载、GPU HBM、KV Cache、batching、队列长度和推理引擎状态都会影响可接收请求能力。Gateway 至少应获得后端健康、错误率、TTFT/TPOT、队列、资源池和版本状态。更高级的系统可以把长上下文请求、低延迟请求和批量请求路由到不同资源池，减少互相干扰。

路由决策必须可解释。Trace 中应记录匹配的模型目录版本、路由规则、灰度命中、fallback 状态、upstream 和服务等级。出现事故时，平台要能回答请求为什么进了某个模型、是否命中灰度、是否因为健康状态切换、是否违反能力约束。不可解释路由会让模型质量问题和账单争议都难以处理。路由是治理动作，不只是负载均衡算法。

路由策略还要处理冷启动和容量预留。某些模型 endpoint 虽然健康，但权重未加载或副本刚启动，首批请求会经历明显延迟；某些 Premium 租户需要保留容量，不能被普通流量耗尽。Gateway 可以通过预热状态、资源池标签和服务等级约束避免把请求送到不合适的后端。路由若只看当前错误率，就会忽略这些体验风险。

路由应同时考虑当前健康和未来承诺。

## 6.5 fallback

Fallback 是后端失败、超时或不可用时切换到备用模型、备用 endpoint 或备用 provider。它能提升可用性，但风险很高。备用模型的上下文长度、tokenizer、输出风格、工具调用格式、价格、安全策略和质量都可能不同。HTTP 层面 fallback 成功，不代表业务层面成功。客服、数据分析和 Agent 场景尤其不能随意切换到能力不同的模型。

Fallback 策略应按应用、模型能力和错误类型配置。客服系统可以 fallback 到同能力同版本的备用集群，但不应自动切到回答风格不同或没有知识库约束的模型；代码补全可以 fallback 到较小模型，但应标记质量等级；Agent 工具调用请求不能 fallback 到不支持 function calling 的模型。对已经部分输出的 streaming 请求，fallback 还要定义是否中断、重试或返回部分结果。

每次 fallback 都应进入 trace、指标和账单。平台需要知道 fallback 发生频率、原因、目标、延迟、成本和业务影响。若 fallback 过于频繁，说明主集群容量或健康有问题；若 fallback 后错误率下降但投诉上升，说明可用性提升牺牲了质量。Fallback 是可靠性手段，不是隐藏故障的手段。成熟平台会对 fallback 设置预算和告警。

Fallback 还要尊重幂等和输出一致性。对尚未开始生成的请求，切换后端相对简单；对已经 streaming 的请求，强行切换可能产生重复或断裂输出；对 Agent 内部调用，fallback 可能改变工具参数生成，影响后续步骤。平台应明确哪些阶段允许 fallback，哪些场景只能失败并让上层处理。否则 fallback 会把清晰失败变成隐蔽错误。

隐蔽错误比显式失败更难治理。

因此 fallback 不能默认开启。

它应是显式策略。

## 6.6 灰度发布

灰度发布让平台按租户、项目、用户、流量比例、模型版本、区域或请求特征逐步切换。AI 模型灰度不只看 5xx 和延迟，还要看输出质量、工具调用成功率、引用准确率、投诉、人工评测、业务指标和成本。一次模型升级可能没有技术错误，却改变回答风格、拒答率、JSON 稳定性或 token 消耗。普通服务灰度指标不足以覆盖 AI 风险。

灰度对象也不只模型权重。Prompt 模板、tokenizer、RAG 索引、rerank 模型、安全策略、推理引擎、sampling 参数和网关适配器都可能需要灰度。若多个对象同时变化，质量波动时很难定位原因。Gateway 应支持细粒度流量切分，并把灰度版本写入 trace，让评测和线上反馈可以按版本回放。

灰度必须配套快速回滚。回滚条件应提前定义，例如错误率、TTFT、输出格式失败、工具调用失败、人工质检下降或成本异常。回滚也要注意状态：streaming 请求是否继续，Agent run 是否使用旧版本完成，账单如何记录。灰度不是发布流程的装饰，而是 AI Factory 在模型不可完全预测时控制风险的必要机制。

灰度结果应和评测系统连接。线上灰度可以发现真实用户分布下的问题，离线评测可以解释具体能力变化。若两者割裂，团队只能看到“投诉变多”却不知道原因，或看到 benchmark 提升却不知道线上是否受益。AI Gateway 记录的灰度标签，是把线上流量、用户反馈和离线评测连接起来的关键字段。

没有灰度标签，就没有可信对比。

没有回滚条件，就没有安全灰度。

灰度必须可停止。

停止后还要能清理旧策略。

否则策略会残留。

## 6.7 多模型聚合

多模型聚合指一个平台同时接入自研模型、开源模型、第三方 API、租户私有模型和专用小模型。AI Gateway 对上提供统一 API，对下适配不同 provider 的认证、参数、错误码、streaming 格式、tool calling、计量口径和能力描述。聚合的价值是让应用通过一个入口使用多种模型能力，也让平台能按质量、成本和可用性灵活路由。

聚合的难点是标准化和差异表达。平台不能假装所有模型完全一样。不同模型的上下文、工具调用、JSON 稳定性、多模态能力、拒答策略和价格都不同。Gateway 可以做参数转换和错误映射，但应通过模型目录暴露 capability，避免应用误用。统一 API 负责降低接入成本，能力差异负责保护应用正确性。二者必须同时存在。

多模型聚合还带来数据和合规问题。请求是否可以发到第三方 provider，是否需要留在某个区域，是否允许使用租户私有模型，是否记录原始 prompt，是否用于训练，都需要策略控制。Gateway 是执行这些策略的关键入口。若聚合只关注“多接几个模型”，而不处理数据边界和计费边界，平台会在商业化和合规阶段遇到风险。

聚合还会影响故障责任。第三方 provider 超时、自研模型失败、私有模型资源不足，对用户来说都是 MaaS 平台不可用，但内部处理方式不同。Gateway 应把 provider、endpoint、错误类型和 fallback 结果写入 trace，帮助支持团队解释影响范围。多模型聚合不是把复杂性消除，而是把复杂性收敛到平台可管理的位置。

可管理的前提是差异可见。

差异不可见时，统一接口会误导应用。

目录要表达差异。

## 6.8 Envoy AI Gateway 与 Gateway API Inference Extension

Envoy AI Gateway 和 Gateway API Inference Extension 代表了一个趋势：AI 推理流量治理正在进入云原生网关和 Kubernetes Gateway API 生态。它们尝试在标准网关模型中表达模型路由、推理后端、策略扩展和流量治理，使 AI Gateway 不再完全依赖每家公司自研。标准化的意义在于降低集成成本，让模型平台、网关、Kubernetes 和推理服务之间有更清晰接口。

这类项目关注的问题，正是本章讨论的工程边界：如何把模型、endpoint、路由、策略和后端健康纳入网关；如何让 AI 流量治理与云原生基础设施共存；如何避免每个 MaaS 平台重复造一套入口层。它们不意味着所有业务逻辑都应塞进标准网关，而是提供一个可扩展底座，让团队把通用流量治理与业务策略分层实现。

生产落地仍要结合组织现状。很多企业已有 API Gateway、服务网格、身份系统、计费系统、模型目录和可观测平台；引入标准 AI Gateway 时，需要决定哪些能力使用社区组件，哪些能力保留在自研控制面。标准接口能减少重复建设，但不会自动解决模型质量、业务灰度、成本归因和安全策略。落地时要看接口能否承载自己的对象模型和运营流程。

因此评估这类项目时，不应只看功能列表，还要看扩展点、配置模型、可观测数据、与现有身份和计费系统的集成方式。标准组件适合承载通用流量治理，自研系统仍需要表达业务对象和商业规则。二者结合得好，平台既能减少底层重复建设，又能保留自己的 MaaS 治理能力。

标准化应服务治理，而不是替代治理。

业务策略仍要回到 MaaS 对象模型中。

## 工程实现

AI Gateway 工程实现应把路由、配额、fallback 和能力校验表达为版本化策略。策略既要能被控制面管理，也要能在数据面高效执行。一个请求进入 Gateway 后，应生成 trace id，解析租户和模型，校验权限和能力，检查 token 与并发预算，选择 upstream，并在响应结束时写入计量事件。对 streaming 请求，还要在首 token、每段输出、取消和结束时维护状态。

生产 Gateway 的核心不是“路由规则多”，而是 admission chain 清楚。每个请求都应按稳定顺序经过 identity、capability、budget、policy、route、commit 六个阶段。identity 绑定租户和项目；capability 确认模型是否支持请求参数；budget 检查 token、并发和费用上限；policy 执行安全、灰度和 fallback 约束；route 选择 endpoint；commit 写入开始事件和 trace。顺序固定后，拒绝原因才稳定，指标才能按阶段解释。

```mermaid
flowchart LR
  Req["HTTP Chat Request"] --> Id["1 identity\nkey / tenant / user"]
  Id --> Cap["2 capability\nmodel supports features"]
  Cap --> Budget["3 budget\ntoken / rpm / tpm / cost"]
  Budget --> Policy["4 policy\nsafety / canary / fallback"]
  Policy --> Route["5 route\nendpoint / pool / provider"]
  Route --> Commit["6 commit\ntrace / metering start"]
  Commit --> Upstream["Model Endpoint"]
  Id -->|deny| Err["typed error"]
  Cap -->|deny| Err
  Budget -->|deny| Err
  Policy -->|deny| Err
```

Gateway 还应维护“可路由健康”而不是普通 upstream 健康。模型 endpoint 的 HTTP 端口可用，只说明进程活着；能否接收请求还要看模型是否 loaded、queue 是否过长、KV Cache 是否接近上限、TTFT/TPOT 是否越界、是否处于 draining、是否命中 canary 冻结。Gateway 不需要执行 runtime 调度，但必须消费这些摘要信号，否则路由会把压力推给已经拥塞的副本。

可路由健康应来自 Model Serving 和推理引擎联合上报的 `engine_admission_health`。它不是细粒度 scheduler 状态，而是给 Gateway 的接入摘要：当前 endpoint 对某类请求是否还能在 SLO 内 admission，瓶颈是 queue、prefill、decode、KV block、engine canary 还是 drain。Gateway 根据它选择路由、fallback、拒绝或降级，而不是只看 Kubernetes readiness。

```yaml
engine_admission_health:
  endpoint: af-chat-large-prod
  engine_profile: vllm-prod-h100-v7
  state: routable_limited
  observed:
    queue_p95_ms: measured
    prefill_queue_depth: measured
    decode_active_sequences: measured
    kv_block_pressure: high
    kv_allocation_failures: zero
    queue_deadline_miss_rate: low
    engine_restarts_5m: 0
  admission_hints:
    short_context_interactive: allow
    long_context_interactive: shed_or_route_elsewhere
    batch_generation: defer
  guardrails:
    canary_frozen: false
    draining: false
    max_context_tokens_current: 16000
  freshness:
    observed_at: recent
    ttl_ms: policy_defined
```

这个对象让 Gateway 的路由从“后端活着”升级为“后端能否承诺这类请求”。如果 `kv_block_pressure` 高，Gateway 可以把长上下文流量路由到另一个 pool，同时继续让短请求进入当前 endpoint；如果 `canary_frozen` 为 true，Gateway 应停止扩大新 runtime 流量；如果健康信号过期，生产策略应保守处理。可路由健康的关键是摘要足够稳定，不能把每个 engine 内部指标都泄露到 Gateway，也不能只给一个模糊的 ready。

一个简化路由规则可以这样表达。实际系统还应补充服务等级、数据驻留、模型 capability、灰度版本和观测标签。

```yaml
route:
  match:
    model: af-chat-large
    tenant: enterprise-a
  policy:
    max_input_tokens: 32000
    max_output_tokens: 4096
    rate_limit:
      rpm: 600
      tpm: 2000000
    fallback:
      enabled: true
      target: af-chat-large-backup
      on_errors: [timeout, unavailable]
  upstreams:
    - endpoint: inference-pool-a
      weight: 90
    - endpoint: inference-pool-b
      weight: 10
```

生产网关还应把每次接入判断写成 `endpoint_admission_decision`。`engine_admission_health` 告诉 Gateway 某个 endpoint 当前能接什么形态的流量；`endpoint_admission_decision` 则记录某个请求为什么被允许、拒绝、延后、降级或路由到另一个 endpoint。前者是可路由健康摘要，后者是请求级可回放证据。没有后者，事故复盘只能知道“当时 pool 压力高”，不能证明某个租户请求是否被正确处理。

```yaml
endpoint_admission_decision:
  decision_id: ead-20260620-001
  request_id: req-20260620-001
  trace_id: trace-abc
  tenant: enterprise-a
  service_tier: premium_interactive
  request_shape:
    estimated_input_tokens: 8192
    max_output_tokens: 1024
    stream: true
    workload_slice: long_context_streaming
    deadline_ms: 30000
  candidates:
    - endpoint: af-chat-large-prod
      engine_admission_health: eah-af-chat-large-prod-1020
      decision: shed
      reason: kv_block_pressure_high_for_long_context
    - endpoint: af-chat-large-longctx
      engine_admission_health: eah-af-chat-large-longctx-1020
      decision: admit
      reason: long_context_slots_available
  policy_context:
    route_policy_version: 2026-06-20.3
    budget_policy_version: 2026-06-20.1
    canary_guardrail: no_block
    fallback_allowed_before_first_token: true
  result:
    action: route
    selected_endpoint: af-chat-large-longctx
    metering_start_event: request_admitted
```

这个对象的价值在三类场景最明显。第一，用户投诉首 token 慢时，可以回放当时是否错误地把长上下文请求送进短上下文池。第二，runtime canary 出现 guardrail breach 时，可以证明 Gateway 是否停止扩大新 engine profile 的流量。第三，账单或 SLA 争议时，可以说明请求是被正常 admission、被 shed、被 fallback，还是在尚未产生 token 前被拒绝。它让 AI Gateway 的策略从“配置存在”变成“决策可审计”。

更完整的策略应把 capability 和 fallback 写成显式约束，避免“可用性提升”变成“能力降级”。下面的例子中，fallback 目标必须满足同样的 streaming 和 tool calling 能力，并且只允许在尚未产生 token 前触发。

```yaml
gateway_policy:
  version: 2026-06-19.1
  match:
    tenant: enterprise-a
    model: af-chat-large
  required_capabilities:
    - streaming
    - tool_calling
    - json_schema_output
  admission:
    max_context_tokens: 32000
    max_output_tokens: 4096
    max_concurrent_streams: 200
    token_rate_limit:
      input_tpm: 2000000
      output_tpm: 800000
  routing:
    primary_pool: inference-premium-a
    health_inputs:
      - endpoint_ready
      - queue_p95_ms
      - kv_cache_pressure
      - rolling_ttft_p95_ms
  fallback:
    enabled: true
    allowed_before_first_token: true
    require_same_capabilities: true
    targets:
      - inference-premium-b
  observability:
    emit_policy_version: true
    emit_route_reason: true
    emit_quota_rule_id: true
```

上线前应测试关键路径：认证失败、权限拒绝、配额超限、长上下文拒绝、streaming 取消、fallback、灰度命中、后端超时、错误码映射和账单事件。AI Gateway 的正确性不是只看能否转发，而是看每种边界条件是否产生预期策略结果。入口层一旦出错，影响面往往覆盖所有模型和租户。

工程实现还应提供策略审计和回放能力。给定一个历史请求，平台应能用当时的策略版本解释它为什么被允许、拒绝、路由或 fallback。策略回放能帮助事故复盘，也能在发布新策略前做 dry-run，评估会影响哪些租户和模型。没有回放能力，网关策略变更只能靠线上试错。

对 RAG 和 Agent 请求，Gateway 还应生成 `rag_agent_admission_context`。它不是新的业务对象，而是把入口处已经确认的身份、权限、预算、数据边界、工具边界和观测要求传递给下游。RAG 服务用它生成 `retrieval_permission_decision`，Agent Orchestrator 用它选择 `tool_side_effect_policy` 和初始化 `agent_budget_ledger`，观测系统用它决定 trace 脱敏级别。没有这个上下文，下游服务只能各自查权限，最终一定会出现口径漂移。

```yaml
rag_agent_admission_context:
  context_id: raac-20260620-0001
  request_id: req-support-0001
  identity:
    tenant: enterprise-a
    project: support-copilot-prod
    application_principal: svc-support-copilot
    user_principal: user-42
    delegation_mode: require_user_context_for_retrieval_and_tools
  data_boundary:
    policy: dbp-20260610.3
    prompt_logging: redacted
    retrieval_text_logging: disabled
    third_party_provider_allowed: false
  rag:
    allowed_knowledge_bases: [support-kb]
    require_retrieval_permission_decision: true
    require_rag_context_snapshot: true
  agent:
    allowed_tool_scopes: [read_customer_ticket, search_docs, create_draft_reply]
    side_effect_policy_selector: support-agent-prod
    require_agent_budget_ledger: true
    high_risk_tool_requires_approval: true
  budgets:
    request_max_input_tokens: policy_defined
    run_max_total_tokens: policy_defined_if_agent
    run_max_tool_calls: policy_defined_if_agent
```

这个 admission context 应被写入 `policy_decision_record` 或与其关联。它能解释很多跨层事故：RAG 检索为什么没有看到某个文档，Agent 工具为什么被拒绝，为什么某个请求不能 fallback 到第三方 provider，为什么 trace 里没有明文 chunk，为什么 run 在预算未耗尽前就进入人工确认。Gateway 的职责不是执行检索或工具，而是把入口策略变成下游必须消费的事实。

`policy_decision_record` 是 Gateway 最关键的安全证据对象。它记录一次请求在 identity、capability、budget、data boundary、safety、route 和 commit 阶段的输入事实、命中规则、决策结果和策略版本。它不应包含完整 prompt 或敏感响应，而应包含足够解释策略的引用和摘要。这样既能保护数据边界，又能回答“为什么这个请求被允许、拒绝、限流、fallback 或路由到某个资源池”。

```mermaid
flowchart TB
  Req["request"] --> Identity["identity check"]
  Identity --> Boundary["tenant_boundary"]
  Boundary --> Credential["credential_lifecycle"]
  Credential --> DataPolicy["data_boundary_policy"]
  DataPolicy --> Budget["quota / budget"]
  Budget --> Capability["model capability"]
  Capability --> Route["route / fallback"]
  Route --> Decision["policy_decision_record"]
  Decision --> Audit["security_audit_event"]
  Decision --> Metering["metering event"]
  Decision --> Trace["trace context"]
```

一个简化记录如下。注意这里使用 `prompt_ref` 和 `policy_inputs_hash`，避免把原始敏感内容复制到审计系统；使用 `matched_rules` 和 `effective_policy_versions`，保证策略能回放；使用 `owner_stage`，让拒绝或异常能进入正确团队的队列。

```yaml
policy_decision_record:
  decision_id: pdr-20260619-0001
  trace_id: trace-abc
  request_id: req-123
  time: measured
  subject:
    tenant_id: enterprise-a
    project_id: customer-service-prod
    credential_id: key_6f2c_redacted
    user_context: present
  target:
    requested_model: af-chat-large
    operation: chat.completions.create
    requested_region: cn-north
  policy_inputs:
    prompt_ref: object://redacted-reference
    policy_inputs_hash: sha256:example
    input_token_estimate: measured_or_estimated
    requested_capabilities: [streaming, tool_calling]
  effective_policy_versions:
    tenant_boundary: tb-20260619.1
    credential_policy: cred-20260601.2
    gateway_policy: gw-20260619.4
    data_boundary_policy: dbp-20260610.3
  matched_rules:
    - allow_model_af_chat_large
    - deny_third_party_provider
    - require_same_region
  result:
    action: allow_and_route
    route_pool: inference-premium-a
    fallback_allowed: true
    owner_stage: policy
```

`policy_decision_record` 还应支持 dry-run。发布新策略前，平台可以拿历史请求做回放，计算新增拒绝、路由变化、fallback 变化、成本变化和受影响租户。成熟 Gateway 的策略发布流程不应是“上线后观察”，而应先回答“如果这条策略昨天生效，会影响哪些请求”。这对安全策略尤其重要，因为过宽会放大风险，过严会中断业务。

实现时还要关注性能路径。认证、配额和路由需要低延迟执行，复杂策略可以预编译或缓存；计量和日志可以异步写入，但不能丢失关键结束事件。尤其是 streaming 请求，开始、取消和结束都应可靠记录。工程实现要在入口延迟和治理完整性之间做明确设计，而不是把所有逻辑串行阻塞在请求路径上。

关键路径失败时，应优先安全拒绝，而不是绕过策略继续转发。

多模型聚合和第三方 provider 路由还需要 `egress_provider_decision`。它专门记录一次请求是否允许出站到外部 provider、跨区域 endpoint 或租户私有 provider。普通路由记录说明“选了谁”，egress decision 说明“为什么可以把数据发出去”。它应绑定 `data_boundary_policy`、tenant boundary、provider 合同、区域、日志策略、训练使用策略和计费口径。没有它，平台很难证明某个敏感请求没有被错误发往第三方。

```yaml
egress_provider_decision:
  decision_id: epd-20260620-001
  trace_id: trace-abc
  tenant_id: enterprise-a
  requested_model: af-chat-large
  candidate_provider: third-party-x
  data_classification:
    prompt: confidential
    attachments: none
    rag_context: internal_only
  policies:
    tenant_boundary: tb-20260619.1
    data_boundary_policy: dbp-20260610.3
    provider_contract: provider-x-cn-no-training
  checks:
    residency_allowed: false
    provider_training_disabled: verified
    prompt_logging_policy: redacted_only
    billing_disclosure_required: true
  result:
    action: deny_provider_route
    fallback: internal_model_only
    audit_event: security_audit_event_id
```

这个对象把“数据边界”从抽象原则变成路由门禁。它也让 fallback 更安全：当内部模型不可用时，Gateway 不能只按可用性切到外部 provider，而必须先证明该租户、该数据等级、该区域和该合同允许外联。对商业平台来说，它还支撑账单解释：用户是否被路由到第三方、第三方价格如何计入、是否使用了合规折扣或禁训合同，都应可追溯。

`egress_provider_decision` 的判断不能只发生在最终 route 阶段。外联风险在 admission chain 中会逐步累积：identity 阶段确认调用主体，data boundary 阶段确认数据等级和驻留要求，budget 阶段确认第三方 provider 成本上限，capability 阶段确认内部模型是否能满足需求，route 阶段才选择候选 provider。若只在 route 阶段做一次开关判断，就会出现“安全上允许但经济上不可接受”或“经济上便宜但合同上禁止”的路线。成熟 Gateway 应把外联决策拆成安全、合同、区域、日志、训练使用、计费和止损多个子结论，并把每个子结论写入同一条 decision。

外联决策还要能解释拒绝后的行为。一个 provider 被拒绝后，Gateway 可以选择内部模型、低风险小模型、异步批量队列、人工确认或直接失败；这些动作的用户体验和成本不同。比如高敏客服请求不能外发给第三方，但可以路由到内部模型；公开文档摘要可以外发，但需要低成本模型和 token 预算；跨区域请求可以失败并提示用户切换区域。`egress_provider_decision` 因此不只是审计日志，而是后续 fallback、计费、支持和客户沟通的事实来源。

```mermaid
flowchart TB
  Req["request"] --> Id["identity / tenant"]
  Id --> Classify["data classification\nprompt / file / RAG context"]
  Classify --> Contract["provider contract\nregion / no training / logging"]
  Contract --> Cost["provider cost guard\nbudget / price / quota"]
  Cost --> Internal{internal model usable?}
  Internal -->|yes| RouteInternal["route internal endpoint"]
  Internal -->|no| Egress{egress allowed?}
  Egress -->|yes| RouteProvider["route provider endpoint"]
  Egress -->|no| DenyOrDegrade["deny / degrade / async / human approval"]
  RouteInternal --> EPD["egress_provider_decision"]
  RouteProvider --> EPD
  DenyOrDegrade --> EPD
  EPD --> PDR["policy_decision_record"]
  EPD --> Billing["metering / provider cost"]
  EPD --> Audit["security_audit_event"]
```

公共 MaaS 或开放试用场景还必须把 denial-of-wallet 作为入口治理问题。Denial-of-wallet 指攻击者或误配置调用者通过合法 API 消耗大量昂贵资源，让平台、租户或受害 key 承担成本。它不一定表现为 5xx，也不一定让服务不可用：请求可能都是 2xx，只是每个请求都带超长 context、强制长输出、触发昂贵 provider、制造 Agent 循环或绕过缓存。若 Gateway 只看 QPS、可用性和鉴权，通过的流量仍可能在经济上构成事故。

Gateway 对这类风险的防线应在 admission chain 中前置。第一层是 credential risk：新 key、长期未用 key、来源 ASN 突变、地理位置突变、同一 key 多地域并发、被禁用 key 持续重试。第二层是 request shape risk：input token 极长、max output 异常、工具调用上限异常、provider 强制路由、reasoning effort 过高。第三层是 spend velocity：每分钟成本、每小时预算、免费额度燃烧速度、第三方 provider 成本和专属容量被挤占。第四层是 business intent：该项目历史是否有这类任务，是否处于压测窗口，是否有审批或实验记录。只有把这些信号合并，才能区分真实增长、压测、客户 key 泄露和主动攻击。

```yaml
denial_of_wallet_admission_guard:
  guard_id: dow-guard-maas-public-v1
  scope:
    tenants: public_and_untrusted
    endpoints: [chat_completions, responses, agent_runs]
  signals:
    credential_risk:
      new_key_high_spend_window: 10m
      source_asn_change: alert_or_step_up
      disabled_key_retry: block
    request_shape:
      input_token_p99_multiplier: policy_defined
      max_output_token_ceiling: policy_defined
      provider_forcing: require_egress_provider_decision
      agent_step_ceiling: policy_defined
    spend_velocity:
      cost_per_minute_budget: tenant_or_project_budget
      free_quota_burn_rate: alert_and_throttle
      provider_cost_burst: hold_and_step_up
      premium_capacity_displacement: block_if_untrusted
  actions:
    soft:
      - lower_max_output_tokens
      - require_async_batch_for_large_context
      - disable_external_provider_fallback
    hard:
      - revoke_or_freeze_credential
      - mark_usage_under_billing_hold
      - emit_security_evidence_bundle
      - open_denial_of_wallet_incident_record
```

这个 guard 不应简单替代租户预算。预算回答“最多能花多少”，denial-of-wallet guard 回答“这笔花费是否符合身份、历史、数据边界和业务意图”。一个大客户在合同内做批量生成可能合法，免费试用 key 在 5 分钟内触发同样 provider 成本则应进入 hold。相反，不能因为预算未超就继续消耗昂贵 provider，也不能因为短期成本高就误伤已审批的生产批任务。Gateway 的作用是把风险信号变成 typed decision，而不是把所有异常都归为限流。

当 guard 触发 hard action 时，Gateway 应同时写三类对象：`policy_decision_record` 说明为什么拒绝、降级或冻结；`security_evidence_bundle` 冻结 key、来源、请求形态、provider route、trace 脱敏和 metering 证据；`denial_of_wallet_incident_record` 进入事故与成本流程。这样安全团队可以处置凭据，计费团队可以对异常 usage 做 billing hold，SRE 可以判断是否挤占高价值容量，业务团队可以和客户确认是否真实流量。没有这三类对象，denial-of-wallet 会在安全、账单和 SRE 之间来回转交。

质量路由需要 `routing_quality_scorecard`。普通路由回答“哪个后端可用”，质量路由回答“这个请求在质量、安全、延迟、成本和业务价值约束下，应该由哪个模型或资源池服务”。同一个模型不必服务所有任务：小模型可以处理低风险分类，大模型服务高价值复杂推理；某个模型在代码场景强，但在客服场景不稳定；某个 provider 便宜但数据边界不满足企业租户。Gateway 应把这些差异显式写入 scorecard，而不是隐藏在 if/else 路由规则里。

```yaml
routing_quality_scorecard:
  scorecard_id: rqs-20260619-support
  scope:
    tenant: enterprise-a
    application: support-chat
    task_slices: [policy_lookup, troubleshooting, complaint]
  candidates:
    - model: af-chat-large-202606
      quality_score:
        offline_eval: pass
        citation_accuracy: high
        tool_call_success: high
        safety_gate: pass
      serving_score:
        ttft_p95: within_slo
        tpot_p95: within_slo
        availability: healthy
      economic_score:
        cost_per_successful_answer: acceptable
        fallback_cost: bounded
      constraints:
        data_boundary: first_party_only
        required_capabilities: [streaming, tool_calling]
    - model: af-chat-small-202606
      quality_score:
        offline_eval: pass_for_simple_cases
        citation_accuracy: medium
      routing_use: simple_low_risk_only
  decision_policy:
    primary: maximize_quality_under_slo_and_budget
    fallback: require_same_capabilities_and_safety_gate
```

Scorecard 的输入来自多处：第 13 章的 eval report、第 14 章的 serving release、第 15 章的 runtime qualification、第 37 章的线上质量 telemetry、第 41 章的质量成本账本。Gateway 不应在请求路径上实时计算复杂质量模型，而应消费已发布、版本化、可回放的 scorecard。请求进入时，Gateway 根据任务特征、租户、能力要求和当前健康状态选择候选模型，并把命中的 scorecard id 写入 `policy_decision_record`。这样，线上质量争议可以追溯到当时采用的质量证据。

```mermaid
flowchart LR
  Req["request features\n任务 / 租户 / 风险 / 能力"] --> Score["routing_quality_scorecard"]
  Eval["offline eval"] --> Score
  Online["online quality telemetry"] --> Score
  Cost["quality_cost_ledger"] --> Score
  Health["serving health / SLO"] --> Score
  Score --> Route["route model / pool / fallback"]
  Route --> PDR["policy_decision_record"]
```

在高价值场景中，还应生成 `routing_quality_decision_record`。它比普通 `policy_decision_record` 更关注质量和业务结果：请求被识别成哪个 task slice，哪些候选模型被排除，最终模型为什么在质量、SLO、成本、数据边界和 capability 约束下胜出，fallback 是否仍满足同一质量门槛。没有这个记录，线上出现“为什么这个客户被路由到小模型”的争议时，团队只能查路由代码和当时的配置快照。

```yaml
routing_quality_decision_record:
  decision_id: rqd-20260620-0001
  trace_id: trace-abc
  request_features:
    tenant: enterprise-a
    application: support-chat
    task_slice: policy_lookup
    risk_level: medium
    required_capabilities: [streaming, tool_calling, citation]
    data_boundary: first_party_only
  scorecard:
    scorecard_id: rqs-20260619-support
    quality_gate_execution: qge-af-chat-20260620-001
    quality_cost_ledger_window: qcost-20260620-1000
  candidates:
    - model: af-chat-large-202606
      decision: selected
      reason: best_quality_under_slo_and_budget
    - model: af-chat-small-202606
      decision: rejected
      reason: citation_accuracy_below_task_threshold
    - model: third_party-model-x
      decision: rejected
      reason: data_boundary_not_allowed
  fallback_policy:
    allowed: true
    require_same_capabilities: true
    require_quality_gate: true
```

这个记录不应包含原始 prompt，但应包含足够回放路由决策的引用。它能连接四类事后分析：质量投诉时查 task slice 和候选模型；账单争议时查为什么选了更贵模型；SLO 事故时查是否因健康状态临时降级；安全审计时查第三方 provider 为何被排除。质量路由一旦进入结构化记录，Gateway 就不只是流量入口，而是模型能力与业务价值之间的决策执行点。

AI Gateway 还要承载 `online_experiment_record`。A/B 和 canary 不是简单按比例分流，而是一个带假设、样本、随机化单元、护栏指标、统计窗口、停止条件和回滚动作的实验对象。模型上线实验尤其要防止污染：同一用户多轮会话不能在不同模型之间来回切换，Agent run 中途不能切模型，RAG 索引实验不能让同一个请求同时混用多个索引版本。实验记录必须能说明谁进入了实验、为什么进入、观察了哪些指标、何时停止。

```yaml
online_experiment_record:
  experiment_id: exp-support-model-20260619
  hypothesis: "new model improves citation correctness without increasing cost per successful answer"
  owner: maas-platform
  randomization_unit: user_id
  eligibility:
    tenants: [enterprise-a]
    task_slices: [policy_lookup]
    exclude: [high_risk_complaint, agent_runs]
  variants:
    control:
      model: af-chat-large-202605
      prompt_template: support-v17
    treatment:
      model: af-chat-large-202606
      prompt_template: support-v18
  guardrails:
    - ttft_p95_not_regress
    - safety_reject_rate_not_drop
    - complaint_rate_not_increase
    - cost_per_successful_answer_not_increase
  stop_conditions:
    rollback_on_severe_quality_feedback: true
    freeze_on_guardrail_breach: true
  sample_harvesting:
    feedback_to_eval_dataset: enabled_with_review
```

```mermaid
stateDiagram-v2
  [*] --> Designed
  Designed --> DryRun: policy replay
  DryRun --> Canary: eligibility and guardrails pass
  Canary --> Observe: collect quality / SLO / cost
  Observe --> Expand: guardrails pass
  Observe --> Freeze: weak signal or insufficient sample
  Observe --> Rollback: severe guardrail breach
  Expand --> RolledOut: decision approved
  Freeze --> Observe: extend window or adjust
  Rollback --> Harvest: collect regression samples
  RolledOut --> Harvest
  Harvest --> [*]
```

Gateway 的计量事件应采用 append-only 设计。请求开始、首 token、完成、取消和失败是不同事件，不应只在成功完成时写一条 usage。Append-only 事件能处理 streaming 中断、重试和补账，也便于审计。

```yaml
metering_events:
  - type: request_admitted
    trace_id: trace-abc
    policy_version: 2026-06-19.1
    requested_model: af-chat-large
    route_pool: inference-premium-a
  - type: first_token
    ttft_ms: measured
    served_model: af-chat-large-202606
  - type: usage_delta
    input_tokens: 2380
    output_tokens_generated: 128
    output_tokens_delivered: 128
  - type: request_closed
    close_reason: client_cancelled
    billable_policy: generated_or_delivered
```

## 常见故障

第一类故障是网关只按 QPS 限流，无法限制超长上下文、长输出和 Agent 内部调用，导致后端推理服务被少量请求拖垮。第二类故障是 fallback 到能力不同的模型，HTTP 请求成功但工具调用、JSON 输出或安全策略失败。第三类故障是 streaming 沿用普通 HTTP 超时和缓冲配置，长回答被中断，或者客户端取消后服务端继续生成。

第四类故障是灰度只看 5xx，不看质量和成本。模型升级后错误率不变，但回答风格、引用准确率或 token 消耗变化，业务已经受影响。第五类故障是错误码映射不统一，应用无法区分限流、配额、模型失败、安全拒绝和 provider 故障。第六类故障是策略分散在应用、网关和模型服务中，出现问题时没人知道最终生效规则是什么。

排查 AI Gateway 故障时，应从请求 trace 入手：调用者是谁，命中哪个 key 和租户，目标模型能力是否匹配，配额是否通过，路由规则是什么，upstream 是哪个，是否 fallback，streaming 是否取消，错误来自网关还是后端。Gateway 故障的本质通常是策略与实际流量不匹配，而不是简单连接失败。

还有一类常见故障是网关和控制面状态不一致。控制面已经下线模型，数据面仍缓存旧规则；配额已经提升，某个网关实例仍使用旧配置；灰度规则只更新了部分区域。这类问题会表现为同一租户在不同请求中得到不同结果。配置版本和实例一致性检查，是 AI Gateway 运维的基本功。

第七类故障是缺少可回放的策略证据。某个请求被拒绝，日志只写了 `forbidden`；某次 fallback 触发，账单只看到 served model 变化；某个 key 被限流，用户不知道是 TPM、预算还是来源限制。缺少 `policy_decision_record` 时，平台只能靠人肉查配置和日志。解决方向是每次 allow、deny、route、fallback、rate_limit 和 safety reject 都留下结构化决策记录。

第八类故障是数据边界被观测系统绕过。Gateway 正确阻止了请求发往第三方 provider，但把完整 prompt 写入了 trace；RAG 正确做了文档权限过滤，但检索摘要进入了共享日志；安全策略拒绝了工具调用，但工具参数出现在告警通知中。AI Gateway 必须把 `data_boundary_policy` 同时应用到转发、日志、trace、审计和调试导出，而不是只应用到请求路由。

配置漂移要告警，告警应指向具体实例、策略版本和受影响租户。否则同一策略会在不同实例上表现不同，直接破坏排障可信度。

第九类故障是质量路由没有证据。平台把高价值请求路由给大模型，把低价值请求路由给小模型，但没有记录 `routing_quality_scorecard`，也没有说明任务切片如何识别。出现投诉时，团队无法判断是路由策略错、任务分类错、模型质量差，还是成本策略过于激进。质量路由必须像安全策略一样可回放：给定历史请求和当时 scorecard，应能重建候选模型、过滤原因、排序结果和最终决策。

第十类故障是实验对象不完整。灰度只写了“10% 流量到新模型”，没有随机化单元、护栏指标和停止条件；同一用户多轮对话被分到不同版本，反馈样本互相污染；回滚后没有把失败样本沉淀到评测集。AI Gateway 里的实验开关必须绑定 `online_experiment_record`，否则 A/B 会退化成不可解释的线上试错。

## 性能指标

AI Gateway 指标应覆盖入口流量、治理结果、模型后端和成本。入口指标包括请求数、连接数、streaming duration、网关处理延迟、请求体大小、响应体大小和客户端取消率。治理指标包括认证失败、权限拒绝、限流命中、配额拒绝、上下文超限、fallback 次数、灰度流量比例、策略拦截和错误码分布。

后端指标应按 upstream、模型、租户和服务等级切分，包括 TTFT、TPOT、E2E latency、后端错误率、队列等待、超时、健康状态和资源池压力。成本指标包括按路由策略的 input/output token、fallback 成本、租户成本、模型成本和异常 token 消耗。安全指标包括高风险工具请求、策略拒绝、异常 key 调用和跨租户访问拒绝。

安全指标要区分三类：身份风险、策略风险和数据风险。身份风险包括 key 来源异常、过期 key 使用、被禁用 key 重试、同一 key 多地域突增；策略风险包括 deny 率、dry-run 影响请求数、fallback 能力不匹配、越权模型访问尝试；数据风险包括 prompt logging downgrade、第三方 provider 拒绝、跨区域拒绝、trace 脱敏失败和高敏字段命中。把这些指标混成一个 `security_rejected_total`，无法指导安全团队行动。

指标要支持行动。若限流命中高，可能需要调整配额或优化应用；若 fallback 激增，可能是主集群故障或路由策略错误；若 streaming 取消率高，可能是客户端体验、超时或输出过慢；若某 upstream TTFT 升高，可能是队列、prefill 或资源池压力。Gateway dashboard 应帮助团队判断下一步，而不是只显示总请求数。

指标还要按策略版本切分。一次路由规则、限流阈值或 fallback 配置变更后，平台需要比较变更前后的延迟、错误、成本和质量代理指标。若没有策略版本维度，网关变更造成的影响会混在模型流量波动里。AI Gateway 是策略执行点，因此策略本身也应成为观测维度。

性能指标还应连接容量规划。Gateway 看到的是最早的租户和模型流量分布，能提前发现某些模型、区域或资源池的需求增长。若这些指标只用于告警，不进入容量评审，平台会重复在高峰期被动扩容。入口指标是业务需求进入 AI Factory 的第一手信号。

质量路由指标包括 scorecard 命中率、按任务切片的模型分布、质量 fallback 率、低质量反馈后的路由变更、实验样本量、实验护栏触发、回滚样本数和路由决策回放成功率。它们和普通网关指标不同：普通指标告诉你请求是否到达后端，质量路由指标告诉你请求是否被交给了合适能力的后端。若 `routing_quality_scorecard` 命中率低，说明大量请求仍在走默认路由；若护栏触发后没有自动冻结，说明实验治理还停留在人工观察。

对 Gateway 来说，错误指标必须能指导客户端行为。建议每个对外错误码至少带三类元数据：`retryable`、`billing_impact` 和 `owner_stage`。例如认证失败不可重试、无 token 费用、owner 是 identity；后端超时可能可重试、可能已有部分 token、owner 是 serving；客户端取消不可重试、可能已有 generated token、owner 是 client/runtime 边界。缺少这些元数据，应用会用同一种退避策略处理所有失败，平台也无法把故障归到正确团队。

| 错误类别 | 是否建议立即重试 | 是否可能有 token 成本 | 主要责任阶段 |
| --- | --- | --- | --- |
| `auth_failed` | 否 | 否 | identity |
| `quota_exceeded` | 否，等待配额窗口 | 否 | budget |
| `context_too_large` | 否，缩短输入 | 否 | capability / budget |
| `model_unavailable` | 可按策略重试或 fallback | 可能没有 | route / serving |
| `timeout_after_prefill` | 谨慎重试 | 是 | serving / runtime |
| `stream_client_cancelled` | 否 | 是，取决于账单策略 | client / streaming |
| `safety_rejected` | 否 | 通常无输出成本，可能有输入成本 | policy |

## 设计取舍

AI Gateway 可以做得很厚，也可以做得很薄。厚网关统一治理强，能集中实现鉴权、限流、路由、fallback、计量和安全策略，但容易变成复杂单点，变更风险高。薄网关更简单、性能路径短，但很多策略会散落到应用和模型服务中，导致口径不一致。成熟平台通常让 Gateway 负责入口治理和路由，把模型内部调度留给推理服务，把多步任务编排留给 Agent Platform。

第二个取舍是标准化与业务定制。使用 Envoy AI Gateway、Gateway API 等标准生态可以降低基础能力建设成本，但企业仍有自己的租户、计费、权限和模型目录。完全自研可以贴合业务，但长期维护成本高。可行路径是用标准网关承载通用数据面能力，用自研控制面管理 MaaS 对象和业务策略，并通过清晰接口连接二者。

第三个取舍是实时策略与稳定性。动态路由、实时健康、成本优化和自动 fallback 可以提高效率，但策略过于复杂会降低可解释性。生产系统应从简单、可验证的路由开始，再逐步引入实时信号。每个自动策略都要有 trace、告警和回滚。AI Gateway 的目标不是让流量自动“聪明”起来，而是让治理动作可控、可解释、可恢复。

最后还要在性能路径和治理深度之间取舍。每增加一次外部查询、策略计算或日志写入，都会增加网关延迟；每减少一项检查，又可能放大安全和成本风险。常见做法是把高频、低复杂度的策略放在数据面快速执行，把复杂审批和分析放在控制面异步处理。入口层必须快，但不能快到失去治理。

治理缺失最终会以更高故障成本偿还。

## 小结

- AI Gateway 是模型流量的治理入口，不只是 HTTP 反向代理。
- LLM 限流需要理解 token、上下文、streaming 和任务调用树。
- Fallback 和灰度必须考虑模型能力、质量、成本和可回滚性。
- 标准化网关生态正在形成，但生产系统仍需要结合租户、计费和可观测性闭环。
- `routing_quality_scorecard` 和 `online_experiment_record` 让网关从“按健康转发”升级为“在质量、安全、SLO 和成本约束下可解释地路由与实验”。

## 延伸阅读

- [Envoy AI Gateway documentation](https://aigateway.envoyproxy.io/)
- [Gateway API Inference Extension documentation](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Kubernetes Gateway API documentation](https://gateway-api.sigs.k8s.io/)
