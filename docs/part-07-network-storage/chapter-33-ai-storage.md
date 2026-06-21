# 第 33 章：AI 存储系统

## 33.1 导读

### 33.1.1 本章回答的问题

- AI Factory 中的数据集、checkpoint、模型权重和日志分别需要什么存储能力？
- Object Storage、Parallel File System、Local NVMe 和 cache 如何组合？
- 为什么存储问题经常表现为 GPU 利用率下降、训练变慢或推理冷启动慢？


### 33.1.2 本章上下文

- 层级定位：本章属于 `网络与存储层`，重点讨论AI 网络、scale-up/scale-out、RDMA、对象存储、PFS 和 NVMe。
- 前置依赖：建议先理解 第 32 章：Scale-out 网络 中的核心对象和路径。
- 后续关联：本章内容会继续连接到 第 34 章：GPU 服务器，并在系统地图、深度标准和读者测试中被交叉引用。
- 读完能力：读完本章后，读者应能把《AI 存储系统》中的概念映射到 AI Factory 的生产路径、工程对象、观测证据和设计取舍。


### 33.1.3 读者测试

- 机制题：读者能否解释 dataset storage、checkpoint storage、object storage、parallel file system 的核心机制，以及它们如何共同支撑《AI 存储系统》？
- 边界题：读者能否区分 网络、存储、runtime、调度和物理故障域 的责任边界，并说明哪些问题不能简单归因到本章组件？
- 路径题：读者能否从训练或推理性能症状追到网络路径、存储路径、checkpoint、缓存和 telemetry，并指出本章对象在路径中的位置？
- 排障题：当《AI 存储系统》相关生产症状出现时，读者能否列出第一层证据、下一跳证据、可能 owner 和止血动作？


### 33.1.4 一个真实场景

一个训练任务在前几个小时运行正常，开始保存 checkpoint 后 step time 周期性升高。GPU 监控显示每次 checkpoint 附近都有明显空转，训练日志显示写 checkpoint 耗时变长。存储团队看到总带宽没有打满，认为存储不是瓶颈。进一步分析后发现，多个训练任务在同一时间写入同一个目录层级，大量小文件和元数据操作集中到少数元数据服务，导致周期性等待。

另一个问题发生在推理集群。模型服务扩容时，新 replica 需要拉取大模型权重。GPU 已经分配，Pod 也启动了，但服务迟迟不能进入 ready。对象存储总带宽充足，单个 replica 拉取也正常，问题出现在高峰扩容时所有 replica 同时拉取同一批权重，远端存储、网络和节点本地磁盘一起出现尖峰。用户看到的是扩容慢和 TTFT 上升。

这些场景说明，AI 存储不是“容量够、带宽高”就结束。存储系统同时影响数据读取、训练恢复、模型发布、推理冷启动、成本和可靠性。不同数据类型访问模式差异极大：数据集可能是顺序读或小文件随机读，checkpoint 是周期性突发写，模型权重是发布和扩容时的高吞吐读，日志和 trace 是持续小写。

AI Factory 的存储设计必须分层。对象存储适合源数据和归档，并行文件系统适合热训练路径，本地 NVMe 适合缓存和 scratch，cache 适合权重和数据预热，model registry 管理模型发布语义。把所有数据都放到一个系统里，通常会在性能、成本或运维上付出代价。

排障时也要从 workload 结果反推存储路径。训练慢不是先问“存储带宽多少”，而是看 GPU 是否等待 data loader、checkpoint 是否拉长 step、恢复是否失败、模型服务是否卡在权重加载、缓存命中率是否下降。只有把 job、dataset、model、path、tenant 和时间线连起来，才能判断瓶颈在对象存储、并行文件系统、本地盘、网络、元数据服务还是应用数据格式。


## 33.2 基础模型

### 33.2.1 核心概念

AI 存储系统位于网络与存储层，向上支撑数据处理、预训练、微调、评测、模型注册、模型服务、RAG、日志和计费。它承载的数据类型包括 raw dataset、processed dataset、tokenized shard、embedding、checkpoint、model artifact、tokenizer、prompt log、trace、metrics 和 billing record。不同数据生命周期和访问模式完全不同。

访问模式是存储设计的核心。数据集读取可能是大吞吐顺序读，也可能是大量小文件随机读；checkpoint 写入通常是周期性突发和多 rank 并发；模型权重加载要求启动时高吞吐和低冷启动；日志和 trace 更像持续流式写入；RAG embedding 和向量库还有低延迟查询和更新需求。不能用单一 benchmark 代表所有场景。

存储分层包括 object storage、parallel file system、local NVMe、cache 和 registry。Object storage 提供容量、成本和生命周期管理；parallel file system 提供高吞吐和 POSIX 接口；local NVMe 提供靠近计算的高速临时空间；cache 减少重复远端读取；model registry 管理模型版本、元数据和发布状态。分层的目标是把数据放到适合的位置。

AI 存储还必须有数据治理。数据集版本、权限、加密、保留策略、清理策略、成本标签、租户边界和审计，都影响生产使用。没有数据版本，实验不可复现；没有生命周期，checkpoint 和日志会无限增长；没有标签，成本无法归因。存储系统既是性能系统，也是治理系统。

因此，AI 存储的核心不是某一个产品，而是一套数据路径契约。数据从哪里来，经过哪些处理，热路径在哪里，哪个系统是 source of truth，哪些内容可以丢弃，哪些内容必须长期保留，谁有权限读取，成本归到哪个租户，都要在平台层表达清楚。契约缺失时，用户会把路径、清理和缓存逻辑写进脚本，最终形成不可审计、不可复现、不可迁移的隐性系统。


### 33.2.2 系统架构

AI 存储架构通常从数据生命周期出发。Raw data 进入 object storage，经过清洗、去重、tokenization 和 sharding，生成 processed dataset；热数据被同步或缓存到 parallel file system 或 local NVMe；训练 worker 读取数据并周期性写 checkpoint；checkpoint 进入评测和转换流程；通过的模型 artifact 进入 model registry；推理集群根据模型版本拉取权重并缓存。

架构中至少有四条关键路径。第一是数据读取路径，决定 GPU 是否等待 data loader。第二是 checkpoint 写入和恢复路径，决定训练容错和抢占成本。第三是模型发布和权重加载路径，决定推理冷启动和扩容速度。第四是观测和审计路径，决定问题能否归因、成本能否分摊、数据能否被治理。

存储系统与网络紧密耦合。对象存储、并行文件系统、缓存和本地盘之间的数据流会占用 scale-out 网络。checkpoint、权重加载和数据预热如果与训练通信重叠，会造成网络和存储双重压力。存储架构不能脱离网络拓扑和调度策略设计。

观测层应按 job、tenant、model、dataset、checkpoint 和 path 打标签。只看存储集群总吞吐没有意义，因为瓶颈可能在单目录元数据、单客户端限速、某个租户突发、某个 rack 缓存未命中或某个模型权重加载。AI 存储必须支持从 GPU idle 反查到具体数据路径。

控制面负责把这些路径产品化。训练平台应生成标准目录、注入数据集版本、创建 checkpoint manifest、设置 cache 策略，并把 storage policy 写入作业元数据；模型平台应把模型 artifact、tokenizer、配置和安全状态作为一个版本发布。数据面则负责实际读写和缓存。控制面和数据面分离后，用户仍然使用熟悉的路径或 API，但平台可以统一做审计、清理、迁移和回放。

![图：33.2.2 系统架构](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-01.svg)

AI 存储架构还需要一条证据链，把“GPU 在等”映射到具体数据路径。训练任务慢时，平台应能从 job id 找到 dataset manifest、shard、client node、cache state、storage backend、metadata service、network path 和 checkpoint 事件；推理冷启动慢时，平台应能从 endpoint 找到 model artifact、digest、registry、weight cache、节点本地 NVMe 和 replica readiness。没有这条链，存储团队看到的是系统吞吐，模型团队看到的是 GPU idle，平台团队只能猜测中间发生了什么。

![图：33.2.2 系统架构](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-02.svg)

这条链路的关键是 manifest。路径只是位置，manifest 才是契约：它说明哪些 shard、哪些对象、哪些 checkpoint 分片、哪个模型权重 digest、哪个 tokenizer、哪些 checksum 和生命周期策略构成一次可复现的数据访问。路径可以迁移，manifest 不应丢失。AI Factory 应尽量让训练、评测和服务都引用 manifest，而不是让脚本自由扫描目录。

可以把 AI 存储理解成四类生产路径的组合，而不是一个统一文件系统。Dataset path 负责持续供给训练样本；checkpoint path 负责保存和恢复训练状态；artifact path 负责发布模型和扩容副本；cache path 负责把热数据靠近 GPU。每条路径都有不同 owner、指标、失败语义和成本口径。

![图：33.2.2 系统架构](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-03.svg)

这张图的重点是 intent 和 evidence。Intent 让平台在任务启动前知道需要什么数据路径；evidence 让平台在任务运行后证明发生了什么。没有 intent，调度器无法预热和准入；没有 evidence，事故和成本只能靠存储后端平均指标解释。存储系统的工程深度，取决于这两类对象是否贯穿训练和推理生命周期。


## 33.3 关键技术

### 33.3.1 dataset storage

Dataset storage 保存训练、微调、评测和 RAG 所需的数据。它要解决容量、版本、权限、吞吐、数据格式、sharding、生命周期和成本问题。训练数据从 raw data 到 processed dataset，通常经历清洗、去重、过滤、tokenization、切分和格式转换。每个阶段都应有版本和元数据。

数据集版本是可复现性的基础。模型质量变化不一定来自代码或参数，也可能来自数据清洗规则、去重策略、tokenizer、采样比例或 shard 顺序变化。平台应记录数据集版本、生成脚本、输入来源、token 数、过滤规则和权限。没有这些信息，训练实验无法被可靠复现。

读取路径要匹配训练框架。大量小文件会造成元数据压力；过大的 shard 会降低随机性和并行度；压缩格式会影响 CPU 解码；远端对象存储可能受请求限流影响；本地缓存可能提高吞吐但需要预热和清理。数据工程和存储工程必须共同设计数据格式，而不是训练脚本临时拼路径。

工程上，应为数据集建立访问画像：文件数量、平均对象大小、总容量、读取并发、顺序读比例、缓存命中率、data loader wait time 和 GPU idle time。存储系统不是只管理数据，还要管理数据如何被训练任务消费。访问画像能指导 sharding、缓存和存储选型。

数据集还要有质量和权限边界。训练数据往往来自多个来源，可能包含不同许可证、敏感字段、重复样本和质量标签。存储层不负责判断模型效果，但必须保存元数据，让上游治理和下游实验能追踪来源。企业场景还要区分公共数据、租户私有数据和受限数据，避免缓存或共享文件系统打破权限边界。Dataset storage 不是“文件堆”，而是训练证据链的一部分。

生产级数据集应有 `dataset_manifest`。它把数据内容、处理版本、分片、统计、权限和缓存策略固定下来。示例：

```yaml
dataset_manifest:
  dataset_id: corpus-v3.2
  version: immutable
  tokenizer: tokenizer-v3
  source:
    raw_inputs: recorded
    cleaning_pipeline: clean-pipeline@sha256:example
    dedup_profile: dedup-v4
  shards:
    format: indexed_binary
    count: 8192
    average_shard_size: measured
    checksums: required
  statistics:
    total_tokens: measured
    language_mix: summarized
    filtered_ratio: measured
    sequence_length_distribution: summarized
  access:
    tenant_scope: foundation-model-team
    pii_classification: restricted
    cache_policy: prewarm_for_training_prod
  lineage:
    parent_dataset: corpus-v3.1
    generated_at: recorded
```

有了 manifest，训练平台才能做三件事：启动前验证数据是否完整，运行中从 step 反查 shard，事故后比较数据版本和缓存状态。没有 manifest，数据集只是目录名，任何复制、清理、覆盖和权限变化都可能破坏复现。


### 33.3.2 checkpoint storage

Checkpoint storage 保存训练中间状态，用于容错、恢复、评测和模型发布。Checkpoint 可能包含模型参数、优化器状态、训练 step、随机数状态、数据加载位置、并行切分信息和训练配置。大模型 checkpoint 规模巨大，写入和恢复都可能成为训练系统关键路径。

Checkpoint 的工程目标有三个：写得足够快，不显著拖慢训练；保存得足够可靠，故障后能恢复；元数据足够完整，后续能用于评测、转换和发布。只保存权重不一定能恢复训练，只保存最新 checkpoint 又可能无法回溯模型质量。平台应区分恢复 checkpoint、里程碑 checkpoint 和发布候选 artifact。

Checkpoint 间隔决定抢占和故障成本。间隔太短，会增加存储写入压力、网络流量和训练 step 抖动；间隔太长，节点故障或抢占后会浪费更多 GPU 小时。最佳间隔取决于训练规模、故障率、写入耗时、恢复耗时和任务优先级。Checkpoint 策略应与调度和 preemption 结合。

工程上，checkpoint 写入应避免所有 rank 同时冲击同一目录或元数据服务。可以使用分片、分层写入、异步上传、临时目录原子提交、压缩、保留策略和写入错峰。Checkpoint 成功应有 manifest 和校验，恢复前应验证完整性。没有完整性校验，恢复失败会在最糟糕的时间暴露。

Checkpoint 还承担组织协作语义。训练团队需要知道哪个 checkpoint 可恢复，评测团队需要知道哪个 checkpoint 可评估，服务团队需要知道哪个 artifact 可发布，平台团队需要知道哪些中间状态可以清理。把所有文件都叫 checkpoint 会造成混乱。更合理的做法是把恢复点、评测候选、发布候选和归档产物区分开，并用 manifest 描述来源 step、并行切分、依赖配置、校验和转换状态。

Checkpoint 写入还应采用两阶段提交语义。第一阶段写入临时分片和局部 metadata，第二阶段校验所有 rank 分片并提交 manifest。只有 manifest 提交成功，checkpoint 才能被标记为 valid。训练恢复逻辑必须读取 latest valid，而不是最新目录。

![图：33.3.2 checkpoint storage](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-04.svg)

这个协议能避免最常见的隐性事故：目录存在但分片不完整，最新 checkpoint 覆盖了旧健康版本，恢复时才发现 optimizer 或 scheduler state 缺失。Checkpoint 的可用性必须由 manifest 和校验定义，而不是由文件夹是否存在定义。

生产实现还应记录 `checkpoint_commit_record`，把一次 checkpoint 的写入、校验、提交、latest 指针更新和恢复候选关系保留下来：

```yaml
checkpoint_commit_record:
  checkpoint_id: ckpt-step-120000
  workload_id: train-20260620-017
  storage_backend: parallel_file_system
  commit_protocol: two_phase_manifest_commit
  writes:
    expected_rank_shards: 512
    completed_rank_shards: measured
    slowest_rank_write_ms: measured
    metadata_ops: measured
  validation:
    shard_checksums: verified
    manifest_committed: true
    restore_smoke_test: passed
  index_update:
    previous_latest_valid: ckpt-step-119000
    new_latest_valid: ckpt-step-120000
    update_time_ms: measured
  impact:
    checkpoint_pause_ms: measured
    gpu_idle_seconds: calculated
```

这个记录让 checkpoint 从“周期性写文件”变成可运营对象。若 checkpoint 慢，平台能判断是 rank 写入长尾、metadata 服务、manifest 校验还是 index update；若恢复失败，平台能判断是否 latest 指针指向了未验证对象；若训练成本上升，Token Factory 可以把 checkpoint pause 和 GPU idle 计入训练 ROI。


### 33.3.3 object storage

Object Storage 适合存放大规模数据集、模型 artifact、日志归档、评测结果和跨集群共享数据。它的优势是容量大、成本相对可控、接口通用、生命周期管理成熟、跨地域和多租户能力较强。AI Factory 通常把对象存储作为源数据、长期归档和模型制品分发的基础。

对象存储的问题是语义和本地文件系统不同。训练框架如果假设 POSIX 文件系统，直接读对象存储可能效率不稳定。对象存储更适合大对象和并发访问，不适合大量细粒度 stat、rename 或小文件随机访问。通过 FUSE 模拟文件系统可以降低改造成本，但性能和一致性行为需要验证。

常见做法是通过数据加载器、缓存层、同步工具、预热流程或并行文件系统适配对象存储。Object storage 作为 source of truth，热训练路径通过 cache、parallel file system 或 local NVMe 加速。这样既保留容量和成本优势，又避免每个 training step 直接依赖远端对象存储。

工程上，对象存储需要关注请求量、错误率、限流、单对象吞吐、列表操作、跨地域复制、权限和生命周期。AI workload 可能在短时间发起大量 GET、PUT 或 LIST 请求。若没有 per-tenant 和 per-job 标签，热点和成本很难归因。对象存储治理必须进入平台观测。

对象存储还适合作为跨集群的数据边界。训练集群、评测集群、推理集群和离线分析系统可以通过对象存储交换 artifact，但交换的是版本化对象，而不是随意共享目录。这样做有利于权限审计和灾备，也让不同集群可以采用不同热路径。对象存储的弱点不应通过把所有 workload 都直接压到它上面解决，而应通过缓存、同步、预热和格式设计把它放在正确位置。


### 33.3.4 parallel file system

Parallel File System 用于提供高吞吐、并行访问和 POSIX 风格接口。AI 训练常用它承载热数据集、checkpoint、共享工作目录和中间产物。典型系统包括 Lustre、Weka、GPFS 类产品或其它并行文件系统。它们的共同目标，是让多节点训练以文件系统语义获得高吞吐。

并行文件系统的优势是对训练框架友好，吞吐高，支持多客户端并发，适合需要 POSIX 语义的 workload。挑战是成本、运维复杂度、元数据瓶颈、故障域、容量扩展和多租户隔离。文件系统性能不好时，GPU 会等待 data loader 或 checkpoint，而问题不一定显示为总带宽不足。

设计时要区分数据路径和元数据路径。大文件顺序读写、海量小文件 create/stat/delete、并发 checkpoint、随机读和目录扫描是完全不同的负载。许多训练慢不是带宽不够，而是元数据服务成为瓶颈。数据集和 checkpoint 格式应尽量减少无谓小文件和热点目录。

工程上，并行文件系统要有租户配额、目录规范、冷热分层、快照或备份、故障演练和监控标签。训练平台应提供推荐路径和模板，而不是让用户随意把 checkpoint 写到任意目录。路径规范是存储治理的一部分，也是成本归因的基础。

并行文件系统的运营风险在于“看起来像普通文件系统”。用户容易把临时文件、日志、checkpoint、数据集副本和模型产物都放在同一棵目录树，短期方便，长期会制造权限、成本和性能问题。平台应把热路径设计成受控入口：哪些目录适合读数据，哪些目录适合写 checkpoint，哪些目录会被自动清理，哪些目录需要申请配额。越接近训练关键路径，越需要明确规则。


### 33.3.5 local NVMe

Local NVMe 是节点本地高速存储，适合做数据缓存、模型权重缓存、临时文件、shuffle 空间、编译缓存和推理冷启动加速。它靠近计算，延迟低、吞吐高，可以显著减少远端存储和网络压力。对推理扩容和训练数据预热，local NVMe 经常非常有价值。

本地盘的问题是生命周期短、可靠性低、跨节点不可共享。节点故障、重装或回收时，本地数据可能丢失。因此，本地 NVMe 应默认视为 cache 或 scratch，而不是 durable storage。唯一 checkpoint、唯一数据集或唯一模型 artifact 不应只保存在本地盘上。

平台应明确 local NVMe 的语义。作为 cache，需要预热、淘汰、容量保护、版本隔离和命中率指标；作为 scratch，需要任务结束清理、租户隔离和磁盘水位控制；如果某些场景需要短期持久化，也要有备份或上游 source of truth。语义不清会造成数据丢失和磁盘污染。

工程上，本地盘管理要与调度结合。任务需要多少 cache 空间、模型权重是否已预热、节点磁盘是否足够、清理是否完成，都应影响调度和启动。推理服务冷启动慢，很多时候不是 GPU 不足，而是权重缓存未命中或镜像/模型拉取路径慢。

Local NVMe 还要防止“性能优化变成状态污染”。同一节点可能先运行训练任务，再运行推理任务；也可能在租户之间复用。如果本地缓存没有命名空间、权限和清理策略，既可能泄露数据，也可能让后续任务读到错误版本。生产系统应把本地盘视为受控资源：有容量配额、有水位保护、有任务边界、有启动前检查，也有节点回收时的清理验收。

对调度器来说，local NVMe 应像 GPU、CPU 和内存一样被建模。一个任务声明需要 2 TB scratch，另一个推理服务声明需要某模型权重已缓存，这些需求都应影响节点选择。若平台不理解本地盘，用户会在启动脚本中临时下载和清理，导致启动时间不可预测。把本地盘产品化后，平台才能做预热、复用、限额和故障隔离。


### 33.3.6 cache

Cache 用于把远端对象存储或并行文件系统中的热数据放到更靠近计算的位置。它可以是节点本地缓存、rack 级缓存、集群级 dataset cache、模型权重缓存、embedding cache 或 tokenizer cache。Cache 的价值是减少重复读取、降低冷启动、保护远端存储并提升 GPU 有效利用率。

推理服务尤其依赖 cache。模型服务启动时拉取大模型权重，如果每个 replica 都从远端存储读取，会造成启动慢和存储尖峰。权重缓存、镜像预热和模型分发策略可以显著改善扩容和故障恢复体验。对训练来说，dataset cache 可以减少反复读取同一批 shard。

Cache 的难点是一致性和容量。模型权重、tokenizer、数据 shard 和 embedding 都有版本。缓存必须按内容哈希、digest 或明确版本管理，避免服务读到旧文件。容量不足时要有淘汰策略，且淘汰不能破坏正在运行的 workload。缓存命中率低时，系统会退回远端存储瓶颈。

工程上，cache 应有观测和控制面。平台需要知道哪些模型或数据已预热，缓存占用多少，命中率如何，淘汰了什么，冷启动耗时多少。用户不应在脚本里手工实现缓存逻辑。缓存如果不可见，就会在故障时变成隐藏状态。

缓存策略还必须服务发布流程。模型权重缓存应以 model version、artifact digest 和 tokenizer version 为键，不能只按文件名或路径判断；数据缓存应绑定 dataset version 和 shard digest，不能被后续数据处理覆盖。预热也要有优先级：关键线上模型优先，高频数据集优先，低优临时实验可以接受 cache miss。缓存不是越多越好，而是要把有限的本地和近端容量给最能减少 GPU 等待和用户延迟的路径。

缓存控制面应记录 `cache_residency`，也就是哪些节点、rack 或资源池已经具备某个数据或模型版本。调度器可以优先把任务放到已有缓存的位置，或者在任务启动前触发预热。示例：

```yaml
cache_residency:
  object:
    type: model_artifact
    id: af-chat-large@sha256:example
  scope:
    resource_pool: inference-prod-h100
    nodes_ready: 128
    racks_ready: [rack-11, rack-12]
  policy:
    priority: production
    eviction: protect_while_endpoint_active
    refresh_on_release: true
  metrics:
    hit_ratio: measured
    cold_load_p95: measured
    prewarm_duration: measured
```

缓存一旦进入调度，就不再是临时优化，而是资源状态。它能解释为什么某些 replica ready 更快，也能解释为什么某个训练任务在 cache miss 后拖慢。


### 33.3.7 bandwidth vs IOPS

Bandwidth 衡量吞吐，IOPS 衡量每秒 I/O 操作次数。AI workload 同时需要二者，但场景不同。大 shard 顺序读关注 bandwidth；大量小文件和元数据操作关注 IOPS 与 metadata ops；checkpoint 写入关注突发吞吐和并发写；模型权重加载关注启动窗口内的高吞吐读。

只看总带宽容易误判。存储总带宽未打满时，训练仍可能被单目录元数据、单客户端限速、小文件随机读、对象存储 API 限流或 cache miss 拖慢。反过来，高 IOPS 系统也不一定能承载大 checkpoint 顺序写。性能指标必须匹配访问模式。

验收时应设计多类 benchmark：大文件顺序读写、小文件随机读、元数据操作、并发 checkpoint、模型权重加载、对象存储请求、cache miss 和真实 data loader。单一 `dd` 测试不能代表 AI 存储能力。测试还应覆盖多节点并发，而不是单客户端峰值。

工程上，要把存储指标和 GPU 指标关联。Data loader wait time、GPU idle、checkpoint pause、model loading time、cache hit ratio 和存储端指标一起看，才能判断瓶颈。存储系统的目标不是跑出漂亮 benchmark，而是减少 GPU 等待和服务冷启动。

容量指标也不能替代性能指标。一个系统容量还有很多，但元数据服务可能已经接近瓶颈；一个节点本地盘剩余空间足够，但单个 cache 目录可能达到水位；对象存储总体请求量正常，但某个租户的 LIST 请求可能被限流。AI 存储验收要把 bandwidth、IOPS、metadata ops、tail latency、错误率和 workload 时间线放在一起。只有这样，团队才不会在“带宽没满”和“GPU 在等”之间来回争论。

另一个常见误区是只测平均值。AI workload 对长尾很敏感：少数 shard 读取慢会拖住 data loader，少数 rank checkpoint 写入慢会拖住全局 step，少数 replica 权重加载慢会影响扩容窗口。验收和监控都要看 P95/P99、抖动和并发下的稳定性。带宽和 IOPS 是必要指标，但不是最终目标；最终目标是让训练 step 和推理 ready 时间稳定。


### 33.3.8 Weka、Lustre、Ceph、S3

Weka、Lustre、Ceph、S3 代表不同存储形态和接口。Lustre 常见于 HPC 和高性能并行文件系统场景；Weka 属于面向高性能文件与对象场景的商业系统；Ceph 提供对象、块和文件能力；S3 更常被用作对象存储接口或服务形态。它们不是同一层面的完全替代品。

本书不把某个产品绝对化。选型应看 workload、团队能力、成本、生态、可运维性和已有基础设施。训练热路径、长期归档、模型发布、日志分析、RAG 数据和推理缓存可能需要不同系统组合。一个系统适合源数据，不代表适合 checkpoint 热路径。

评估存储产品时，应从访问模式和运营能力出发。团队是否能运维并行文件系统，是否已有对象存储生态，是否需要 POSIX，是否需要跨地域，是否能支持多租户配额和标签，是否能提供可观测指标，是否能承受 checkpoint 突发。这些比产品名更重要。

合理的 AI 存储架构通常是分层的：对象存储做源和归档，并行文件系统做热训练路径，本地 NVMe 做缓存和 scratch，模型 registry 管理发布语义。产品选型应服务分层，而不是让一个系统承担所有职责。

评估这些系统时，还应避免把产品能力等同于落地能力。一个产品支持高吞吐，不代表本组织能以正确目录结构、网络拓扑、客户端版本和运维流程跑出稳定结果；一个系统支持对象和文件接口，也不代表它适合所有访问模式。更可靠的评估方法是用本书前面提到的 workload 场景验收：数据加载、checkpoint、模型发布、cache miss、多租户突发和故障恢复。产品名只是起点，证据才是选型依据。


## 33.4 工程落地

### 33.4.1 工程实现

工程实现应先定义存储策略，而不是让用户在脚本里临时拼路径。训练平台提交任务时，应声明数据集版本、checkpoint 路径、缓存策略、清理策略、权限和成本标签。示例：

```yaml
storage_policy:
  dataset:
    source: object-storage
    hot_path: parallel-file-system
    cache: local-nvme
  checkpoint:
    write_path: parallel-file-system
    retention:
      latest: keep
      milestone: keep
      intermediate: expire
  model_artifact:
    registry: model-registry
    serving_cache: local-nvme
  observability:
    metrics: enabled
    per_job_labels: ["tenant", "job", "model", "dataset"]
```

第二步是把存储策略接入调度和运行时。任务需要多大本地缓存、是否需要预热数据、checkpoint 写入路径是否可用、模型权重是否已在目标节点缓存，都应影响调度和启动。存储不是任务启动后的脚本细节，而是资源需求的一部分。

第三步是建立生命周期规则。Raw data、processed dataset、checkpoint、artifact、cache、log 和 trace 的保留时间、权限、归档、清理和成本归属不同。没有生命周期规则，checkpoint 和缓存会无限增长，最终把存储成本和节点磁盘问题推给运维。

第四步是建立真实验收。存储验收应覆盖 data loader、checkpoint、模型加载、cache miss、对象存储限流、小文件和多节点并发。验收结果要与资源池、训练平台和模型服务关联。只测单客户端吞吐，不能证明 AI 存储可用。

第五步是建立 `data_path_evidence`。它把 workload 的每次关键 I/O 与路径、manifest、cache、client 和指标绑定。示例：

```yaml
data_path_evidence:
  workload:
    type: distributed_training
    id: train-20260619-042
    phase: checkpoint_write
  path:
    kind: checkpoint
    manifest_id: ckpt-step-120000
    backend: parallel_file_system
    directory: managed_path
  clients:
    ranks: 512
    nodes: attached
    racks: attached
  cache:
    local_nvme: not_applicable
    pfs_client_cache: recorded
  telemetry:
    write_p99: measured
    metadata_ops: measured
    throttling: measured
    network_overlap: nccl_collective_window
  impact:
    step_time_delta: measured
    gpu_idle_seconds: measured
    wasted_gpu_hours: calculated
```

这个对象让存储问题能进入统一诊断包。它回答的不是“存储系统慢不慢”，而是“哪个 workload 因为哪条数据路径慢，浪费了多少 GPU 时间，应该由谁处理”。

更完整的 `storage_evidence` 应覆盖 dataset、checkpoint、artifact 和 cache 四种路径。它应包含 manifest、client、cache、backend、telemetry、网络重叠和业务影响。示例：

```yaml
storage_evidence:
  evidence_id: storage-ev-20260620-017
  workload:
    id: train-20260620-017
    type: distributed_training
    phase: checkpoint_write
  path:
    kind: checkpoint
    manifest: ckpt-step-120000
    backend: parallel_file_system
    namespace: managed_training_checkpoint
  clients:
    ranks: 512
    nodes: measured
    racks: measured
  cache:
    state: bypass_or_hit_or_miss
    local_nvme_pressure: measured_if_applicable
  telemetry:
    read_write_p99_ms: measured
    metadata_ops: measured
    throttle_events: measured
    backend_error_rate: measured
    network_overlap: nccl_or_artifact_or_none
  impact:
    gpu_idle_seconds: calculated
    checkpoint_pause_ms: measured
    model_ready_delay_ms: calculated_if_artifact
    affected_tenants: recorded
  decision:
    likely_cause: metadata_hotspot_or_cache_miss_or_backend_throttle
    action: reshard_prewarm_throttle_or_move_path
```

这个对象是第 37 章观测、第 38 章准入、第 39 章故障树和第 41 章成本账本的交叉点。它不要求所有存储后端使用同一种实现，但要求它们用相同关联键表达事实。否则训练、推理、存储和财务团队会各自拥有一部分证据，却无法形成共同结论。

在模型进入生产之前，存储层还应提供 `supply_chain_release_contract`。它不是发布单，也不是模型 registry 的一行记录，而是把 dataset lineage、checkpoint restore、artifact provenance、artifact distribution、cache residency、cache invalidation、storage security boundary 和 serving release gate 绑定在一起的机器可读契约。它回答的是：这个模型产物是否可以被这个租户、这个 endpoint、这个资源池和这个 cache 层消费；如果某个依赖被撤销，哪些 release、缓存和账单窗口必须同步失效。

```yaml
supply_chain_release_contract:
  contract_id: scrc-af-chat-large-20260620-r3
  scope:
    model_family: af-chat-large
    serving_release: af-chat-large-20260619-r3
    allowed_endpoints: [af-chat-large-prod, af-chat-large-canary]
    allowed_tenants: [enterprise-a, enterprise-b]
    resource_pools: [inference-premium-a]
  training_inputs:
    dataset_lineage_record: dlr-pretrain-202606
    dataset_manifest: dataset-manifest@sha256:example
    deletion_requests_closed_before_training: true
    tokenizer_training_version: tok-train-202606
  checkpoint:
    checkpoint_manifest: ckpt-step-120000
    checkpoint_commit_record: ccr-ckpt-step-120000
    checkpoint_restore_drill: crd-20260620-ckpt-120000
    first_effective_step_after_restore: pass
  artifact:
    model_artifact_provenance: map-af-chat-large-20260619-r3
    model_artifact_distribution: mad-af-chat-large-20260619-r3
    weights_digest: sha256:weights
    tokenizer_digest: sha256:tokenizer
    chat_template_digest: sha256:template
    signature_verification: pass
  cache_and_runtime:
    cache_residency_policy: premium_hot_capacity_required
    cache_invalidation_record_required_on_change: true
    serving_runtime_profiles: [vllm-h100-longctx-r2]
    incompatible_runtime_profiles: [legacy-template-parser-r1]
  boundaries:
    storage_security_boundary: ssb-model-and-training-prod
    export_policy: denied_without_release_approval
    prompt_trace_redaction_record: required_for_debug_export
  release_gates:
    supply_chain_acceptance_matrix: pass
    serving_quality_contract: pass
    serving_route_release_contract: pass
    billing_usage_schema: usage-openai-compatible-v6
  invalidation_policy:
    revoke_if_any_digest_changes: true
    block_new_replica_until_replacement_cache_ready: true
    billing_hold_if_tokenizer_or_template_changes: true
```

这个契约把“能加载”提升为“能生产”。模型服务只要能读到权重就能启动，但这不证明权重来自批准的 checkpoint，也不证明 tokenizer 和 template 与计量一致，更不证明 cache 可以被撤销。`supply_chain_release_contract` 应由 release control plane 生成，并被 Gateway、scheduler、model registry、cache prewarmer、billing replay 和 PRR 消费。若某个数据删除请求影响了训练 lineage，契约能列出受影响 checkpoint、artifact 和 release；若某个 tokenizer digest 需要撤销，契约能告诉调度器哪些热 cache 不能再作为 ready 条件。

![图：33.4.1 工程实现](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-05.svg)

实现上，契约应尽量小而硬。它不需要复制完整训练日志，也不需要包含每个对象的全部元数据；它需要保存不可变引用、摘要、状态、适用范围和失效策略。这样做的好处是发布时可以快速验证契约是否闭合，事故时可以快速计算影响面，审计时可以证明“线上 release 为什么被允许”。如果契约只存在于人工发布说明中，后续 cache、billing、rollback 和安全撤销都会重新依赖人工记忆。

数据和模型供应链事故还需要 `supply_chain_invalidation_evidence`。`cache_invalidation_record` 说明“应当失效什么”，但生产系统还要证明“失效动作是否已经到达所有会使用旧内容的地方”。这些地方包括 model registry 指针、节点本地 NVMe、rack cache、RAG index cache、训练 dataset cache、推理 replica 的已加载权重、autoscaler 的预热队列和调度器的节点可用状态。只改 registry 指针，不能证明旧权重、旧 tokenizer 或旧索引已经离开生产路径。

```yaml
supply_chain_invalidation_evidence:
  evidence_id: scie-20260620-af-chat-large-r3
  trigger:
    reason: artifact_recalled_or_tokenizer_bug_or_data_deletion_or_index_rebuild
    cache_invalidation_record: cir-af-chat-large-20260620-001
    storage_security_boundary: ssb-model-and-training-prod
  affected_objects:
    model_artifact_provenance: map-af-chat-large-20260619-r3
    model_artifact_distribution: mad-af-chat-large-20260619-r3
    tokenizer: tok-af-chat-large-20260619
    rag_index: optional
    dataset_manifest: optional_if_training_or_rag
  propagation:
    registry_pointer_updated: true
    scheduler_blocked_invalid_cache_nodes: true
    autoscaler_uses_replacement_cache_only: true
    running_replicas_drained_or_verified: measured
    local_nvme_cache_invalidated: measured
    rack_cache_invalidated: measured
  verification:
    digest_replay: pass
    sampled_node_cache_scan: pass
    serving_release_trace_replay: pass
    rag_context_snapshot_replay: pass_if_rag
    audit_log_immutable: true
  residual_risk:
    nodes_unreachable: []
    allowed_temporary_exemptions: []
    block_prr_until_closed: true
```

这个对象把“撤销”从控制面动作变成可验证事实。若某个 tokenizer 修复了 token count bug，旧 tokenizer cache 仍然被部分 replica 使用，账单和质量都会继续漂移；若某个模型 artifact 因签名或许可证问题被撤销，节点本地 cache 仍然 ready，调度器就可能把新 replica 放到不可信节点；若 RAG 索引因权限策略更新而重建，旧索引 cache 未撤销，敏感文档仍可能被检索。供应链撤销的验收标准不是“发布系统显示成功”，而是所有可能消费旧对象的路径都被阻断、替换或记录为豁免。

多模态链路还需要 `media_artifact_manifest`。文本请求通常只需要 prompt、response 和 trace；多模态请求会产生原始文件、页面图、缩略图、OCR 文本、ASR 文本、抽帧、布局块、表格结构、视觉 embedding、检索索引、模型输出和人工复核记录。若这些产物没有统一 manifest，系统无法证明答案来自哪个源文件、哪个页码、哪一帧或哪一段音频，也无法执行删除、保留、审计和计费。多模态存储的第一原则是：原始媒体和派生产物都必须可追溯。

```yaml
media_artifact_manifest:
  manifest_id: mam-claims-doc-20260620-001
  source:
    tenant: enterprise-a
    application: claims-document-review
    upload_id: upl-20260620-001
    original_object:
      uri: object://media/enterprise-a/claims/redacted.pdf
      digest: sha256:original
      media_type: application/pdf
      data_classification: restricted
      retention_policy: claims_doc_retention_v3
      permission_snapshot: acl-snap-20260620-001
  derived_artifacts:
    page_images:
      count: measured
      uri_prefix: object://media-derived/claims/page-images/
      digest_manifest: sha256:pages
    ocr_text:
      uri: object://media-derived/claims/ocr.jsonl
      engine_profile: ocr-runtime-v6
      language_hints: [zh, en]
      confidence_distribution: measured
    layout_blocks:
      uri: object://media-derived/claims/layout.json
      includes_tables: true
      coordinate_system: page_pixel_coordinates
    visual_embeddings:
      uri: object://media-derived/claims/embeddings.parquet
      embedding_model: vision-embed-202606
      index_version: optional_if_indexed
  lineage:
    media_processing_pipeline_record: mppr-claims-20260620-001
    multimodal_workload_profile: mwp-claims-document-review-202606
    preprocessing_image: registry/media-pipeline@sha256:example
    security_scan: pass
  controls:
    encryption: kms-key-enterprise-a
    export_requires_approval: true
    derived_artifact_delete_with_source: true
    training_use: denied_by_default
  usage:
    serving_requests: [req-20260620-001]
    quality_gate_executions: [mm-qge-claims-20260620]
    metering_events: [mmme-20260620-001]
```

这个 manifest 让媒体文件不再是“对象存储里的一堆文件”。当用户质疑答案时，系统能从回答引用回到页码、布局块和 OCR 文本；当客户要求删除原始文件时，系统能找到所有派生对象和索引条目；当成本异常时，账本能知道某次请求消耗了多少页渲染、OCR、embedding、存储和模型推理；当评测失败时，第 13 章的多模态门禁能重放同一组派生产物，而不是依赖当时临时目录是否还存在。

`media_processing_pipeline_record` 则记录这次媒体流水线实际如何执行。Manifest 描述结果，pipeline record 描述过程：每个 stage 的输入、输出、runtime、耗时、失败、重试和资源消耗。很多多模态事故发生在模型之前，例如 PDF 解码库升级导致表格错位，OCR 语言识别错误导致关键字段缺失，视频抽帧策略改变导致漏掉目标帧。如果没有 pipeline record，团队会把所有问题归因给模型“看错了”，实际根因可能是派生产物已经错误。

```yaml
media_processing_pipeline_record:
  pipeline_id: mppr-claims-20260620-001
  manifest_id: mam-claims-doc-20260620-001
  stages:
    upload_admission:
      status: pass
      checks: [content_type, size, tenant_quota, retention_policy]
    security_scan:
      status: pass
      scanner_version: media-scan-v4
    page_render:
      status: pass
      runtime_image: pdf-renderer@sha256:example
      pages_rendered: measured
      retries: measured
    ocr:
      status: pass_or_degraded
      runtime_image: ocr-runtime@sha256:example
      low_confidence_regions: measured
      output: ocr_text
    layout_analysis:
      status: pass
      tables_detected: measured
      output: layout_blocks
    embedding:
      status: pass
      model: vision-embed-202606
      output: visual_embeddings
  resource_usage:
    cpu_seconds: measured
    gpu_seconds: measured_if_used
    storage_written_bytes: measured
    queue_wait_ms_by_stage: measured
  failure_semantics:
    partial_outputs_committed: true_or_false
    safe_to_retry_from_stage: recorded
    cleanup_required: true_or_false
```

![图：33.4.1 工程实现](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-06.svg)

媒体 manifest 还必须进入安全边界。原始文件和派生产物的敏感度不一定相同：缩略图可能暴露个人信息，OCR 文本可能比原 PDF 更容易被搜索和泄露，embedding 可能泄露语义信息，抽帧可能包含人脸或工牌。删除原始文件但保留 OCR、embedding 或帧索引，通常不满足真实删除要求。因此，`storage_security_boundary` 应要求 `derived_artifact_delete_with_source`、训练使用默认拒绝、导出审批、不可变审计和按租户 KMS key 加密。多模态系统最危险的地方，常常不是模型，而是派生产物被当成普通缓存长期留存。

AI 存储还要提供 `storage_security_boundary`。训练数据、RAG 文档、checkpoint、模型权重、adapter、prompt log、trace 和 billing record 的敏感级别不同，访问者也不同。存储层不能只依赖上层应用“不要读错路径”，而要把命名空间、加密、KMS key、IAM/RBAC、审计、导出、保留和删除策略写成边界对象。尤其是 checkpoint 和模型 artifact，它们可能泄露训练数据特征或商业能力，不能被当作普通文件共享。

```yaml
storage_security_boundary:
  boundary_id: ssb-model-and-training-prod
  namespaces:
    dataset_raw:
      classification: restricted
      allowed_principals: [data-pipeline-prod]
      export_requires_approval: true
    checkpoint:
      classification: model_secret
      allowed_principals: [training-platform, model-release]
      delete_requires_two_person_review: true
    model_artifact:
      classification: model_secret
      allowed_principals: [model-registry, serving-control-plane]
      signature_required: true
    logs_and_trace:
      classification: mixed_sensitive
      redaction_required: true
  controls:
    encryption_at_rest: required
    kms_key_scope: per_tenant_or_per_domain
    immutable_audit_log: required
    lifecycle_policy: versioned
    break_glass_ttl: policy_defined
```

私有化和离线环境还需要 `offline_upgrade_rehearsal`。很多 AI Factory 产品在公有云环境里升级顺利，到了客户现场却失败，原因不是单个镜像坏了，而是离线 registry、模型 artifact、RAG 索引、缓存、数据库迁移、证书、KMS、对象存储路径和 GPU runtime baseline 没有作为一个整体演练。离线升级不能把“包能导入”当作成功标准，必须证明导入后服务能启动、旧数据能读、新旧模型能回滚、缓存不会复用过期内容、诊断包能导出。

```yaml
offline_upgrade_rehearsal:
  rehearsal_id: offline-upg-enterprise-a-202606
  target_environment:
    deployment: private_ai_factory
    network_mode: offline_or_restricted_egress
    registry: customer_offline_registry
  release_bundle:
    offline_release_bundle_manifest: orb-ai-factory-2026-06-enterprise-a
    import_record: required
    digest_reconciliation: pass
  package_integrity:
    image_manifest: signed_and_digest_pinned
    model_artifact_manifest: signed_and_digest_pinned
    helm_or_kustomize_bundle: signed
    sbom_and_vulnerability_report: included
  storage_and_data_checks:
    dataset_manifest_compatibility: pass_if_training_or_rag
    model_artifact_provenance: pass
    cache_invalidation_record_replay: pass
    rag_index_migration: pass_if_rag
    database_migration_dry_run: pass_if_stateful
    backup_restore_point: verified_before_upgrade
  runtime_checks:
    compatibility_matrix: pass
    gpu_container_runtime_report: pass
    representative_inference_endpoint: pass
    representative_training_or_finetuning_job: pass_if_supported
  rollback_drill:
    service_version_rollback: pass
    artifact_pointer_rollback: pass
    cache_rewarm_after_rollback: pass
    data_migration_rollback_or_forward_fix: documented
  support_evidence:
    diagnostic_bundle_export: pass
    redaction_policy: pass
    customer_approval_record: attached
```

这个对象应归档到交付和 SRE 系统，而不是停留在项目群截图里。它回答三个关键问题：升级包是否完整可信，客户环境是否满足版本矩阵，失败时是否能回到可服务状态。对存储层来说，最重要的是 artifact 和 cache 的一致性：模型权重、tokenizer、chat template、RAG 索引和本地 NVMe cache 必须跟 release 指针一致；如果回滚只改控制面版本，却没有撤销旧 cache 或重新预热，就会出现“版本显示已回滚，实际仍加载旧权重或旧索引”的隐性故障。

离线升级还要把“导入记录”作为一等证据。公有云发布时，平台可以查询中心 registry、对象存储和控制面；客户现场不一定允许远程访问，事后只能依赖现场导出的证据包。因此导入工具应在客户环境中生成 `offline_import_record`：记录离线包 manifest、目标 registry namespace、导入的镜像 digest、模型 artifact digest、chart 版本、配置 overlay、数据库 migration 版本、RAG index 版本、缓存预热结果和签名验证结果。它不是安装日志的替代品，而是用结构化字段证明现场状态可回放。

```yaml
offline_import_record:
  import_id: oir-enterprise-a-20260620-001
  offline_release_bundle_manifest: orb-ai-factory-2026-06-enterprise-a
  target:
    environment: customer_staging
    registry_namespace: customer-registry/ai-factory
    storage_namespace: customer-object/ai-factory
  imported_artifacts:
    container_images:
      expected_digests: from_bundle_manifest
      imported_digests: measured
      missing_or_extra: none
    model_artifacts:
      weights_digest: verified
      tokenizer_digest: verified
      chat_template_digest: verified
    rag_indices:
      index_version: kb-index-202606
      acl_snapshot_digest: verified_if_rag
    database_migrations:
      planned: [schema-202606-a]
      executed: [schema-202606-a]
      rollback_plan: recorded
  validation:
    signature_verification: pass
    cache_rewarm: pass
    representative_endpoint_smoke: pass
    diagnostic_bundle_export: pass
```

这个记录能把离线交付从“现场师傅装好了”变成可审计事实。若升级后出现模型加载旧权重，先对比 bundle manifest、import record、model registry pointer 和节点 cache residency；若 RAG 答案越权，先对比 RAG index digest、ACL snapshot 和导入时的迁移脚本；若回滚失败，先看 rollback plan 是否覆盖 schema、artifact pointer 和 cache rewarm。离线环境排障的第一原则是少猜，多对账。

![图：33.4.1 工程实现](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-07.svg)

这个边界对象能防止很多“存储便利性”演变成安全事故。研究人员不应直接从生产 checkpoint 目录复制权重到个人路径；推理节点不应加载未签名 artifact；RAG 索引构建不应把受限文档写进共享缓存；事故排查不应把未脱敏 trace 导出到普通对象桶。存储安全边界不是阻碍效率，而是让高价值数据的访问、复制、删除和导出可审计、可撤销、可解释。

存储边界还应包含 `secret_boundary_evidence`。AI Factory 中的 secret 不只是 API key，也包括 KMS key、registry token、provider credential、model artifact signing key、object storage STS token、BMC/Redfish 凭据和临时 break-glass token。它们经常跨越 CI/CD、训练任务、推理服务、节点初始化和事故排查路径。如果 secret 边界只靠运维习惯维护，就会在镜像、日志、Notebook、对象存储或 trace 中泄露。

```yaml
secret_boundary_evidence:
  evidence_id: sbe-20260620-001
  scope:
    tenant: enterprise-a
    resource_pool: inference-premium-a
    release: af-chat-large-202606
  secrets:
    provider_api_key:
      storage: secret_manager
      mounted_as: tmpfs_or_runtime_injection
      logged: prohibited
      rotation_state: current
    model_signing_key:
      storage: hsm_or_kms
      exportable: false
    object_storage_token:
      ttl: short_lived
      scope: read_model_artifact_only
  checks:
    image_scan_for_secret: pass
    log_scan_for_secret: pass
    trace_redaction: pass
    break_glass_review: pass_if_used
```

这个对象把 secret 从“配置项”提升为生产证据。模型服务能否访问第三方 provider、训练任务能否读取受限数据、推理节点能否拉取权重，都依赖 secret；同时这些 secret 一旦泄露，会直接变成数据、成本或供应链事故。准入和 PRR 应要求关键 secret 有边界证据，尤其是高敏租户、外部 provider 和模型签名链路。

把数据和模型产物串起来，可以形成一条供应链图。它从 raw dataset 进入 dataset lineage 和 manifest，经过训练生成 checkpoint 和 restore drill，再经过转换、评测和签名生成 model artifact provenance，最终通过 artifact distribution、cache residency 和 cache invalidation 进入 serving。任何一个环节缺证据，模型发布就不可证明。

![图：33.4.1 工程实现](../assets/diagrams/part-07-network-storage-chapter-33-ai-storage-08.svg)

这张图的读法是：存储层不是被动保存文件，而是保存供应链状态。训练异常、模型质量退化、安全撤销、缓存失效和成本归因，都沿这条链路查证。AI Factory 如果能让每个产物回答“来自哪里、谁批准、在哪里缓存、何时失效、能否恢复、成本归谁”，模型上线就从文件复制变成工程发布。

第六步是把清理和成本做成自动化。平台应定期识别过期 checkpoint、孤儿缓存、无引用 artifact、超配额目录和异常增长租户，并在保留策略允许的范围内执行清理或发出审批。成本标签应从任务提交开始继承到存储路径，而不是月底人工追账。AI 存储的可持续性来自生命周期闭环：创建时有元数据，使用时有观测，结束后有清理，归档时有审计。

实现时还要保留逃生通道。大型训练和紧急发布可能需要临时延长保留、固定缓存或提高配额，但这些例外必须有过期时间、审批记录和成本归属。没有例外机制，平台会被绕过；没有过期机制，例外会变成永久债务。好的存储平台不是禁止用户做特殊操作，而是让特殊操作可见、可审计、可回收。


### 33.4.2 常见故障

第一类故障是小文件过多。数据集包含大量小文件，元数据操作成为瓶颈，GPU 等待 data loader。解决方向是重新 sharding、使用适合训练的格式、缓存热数据并监控 metadata ops。不要只增加存储带宽。

第二类故障是 checkpoint 同步写入造成周期性 step time 尖刺。多个 rank 或多个 job 同时写入，冲击网络、元数据和存储服务端。解决方向是分片写入、错峰、异步上传、原子 manifest、保留策略和 checkpoint 间隔优化。

第三类故障是缓存版本错误。权重缓存没有按 digest 或版本隔离，推理服务加载旧模型；dataset cache 没有绑定数据版本，训练读到过期 shard。解决方向是内容寻址、版本化路径和发布系统接管缓存管理。

第四类故障是本地 NVMe 没有清理。节点磁盘满导致新任务失败，或者上一个租户的临时数据残留。解决方向是任务结束清理、水位告警、租户隔离和节点回收验证。本地盘不是无人管理的临时空间。

第五类故障是存储监控没有 tenant/job/model 标签。存储团队看到总吞吐和错误率，平台无法追踪哪个任务制造热点或成本。解决方向是标签治理和统一路径规范。没有标签，成本和热点都无法归因。

第六类故障是对象存储或远端文件系统被误用为低延迟本地路径。训练脚本在每个 step 中执行细粒度 LIST、stat 或小对象读取，单机测试看不出问题，多节点并发后请求被限流。解决方向是改造数据格式、引入 manifest、预热到热路径，并限制训练循环中的远端元数据操作。存储故障往往不是产品失效，而是访问模式和系统语义不匹配。


### 33.4.3 性能指标

数据读取指标包括读取吞吐、data loader wait time、GPU idle time、样本读取延迟、shard 分布、cache hit ratio、对象存储请求延迟和错误率。训练慢时，这些指标能判断 GPU 是否在等数据。单看存储端总吞吐不够。

Checkpoint 指标包括写入耗时、恢复耗时、成功率、失败原因、checkpoint 大小、间隔、manifest 完整性、保留数量和对 step time 的影响。Checkpoint 既是可靠性机制，也是性能负载。指标应帮助平台在恢复成本和写入成本之间调节。

模型发布和推理指标包括模型权重加载时间、cache hit ratio、冷启动时长、镜像与权重拉取耗时、registry 请求延迟和服务 ready 时间。推理扩容慢时，常常需要从这些指标判断是 GPU、镜像、权重、网络还是缓存问题。

运营指标包括存储容量、增长速度、生命周期清理量、租户成本、热点目录、元数据 QPS、对象请求量、限流次数、节点本地盘水位和缓存淘汰次数。AI 存储长期成本很高，运营指标决定系统是否可持续。

还应建立验收指标和告警指标的区别。验收指标用于判断新存储池、新数据格式或新发布路径能否入生产；告警指标用于判断正在运行的系统是否退化。前者需要可重复 benchmark 和 workload 回放，后者需要实时标签和基线对比。把两者混在一起，会导致上线时缺证据，故障时缺上下文。AI Factory 应保存历史基线，持续比较不同版本、不同 rack、不同租户和不同数据集的存储表现。

指标应能支持三类人。训练工程师要看到 data loader、checkpoint 和恢复是否拖慢训练；平台 SRE 要看到容量、水位、错误率、热点和清理是否健康；财务或业务负责人要看到租户、模型和项目的存储成本。三类视角共享同一套标签，但展示粒度不同。没有统一标签，技术排障和成本治理会形成两套彼此不一致的账。


### 33.4.4 设计取舍

第一个取舍是对象存储与热路径文件系统。对象存储容量和成本友好，但不总适合每个 training step 的直接读取；并行文件系统性能强、POSIX 友好，但成本和运维复杂度高。常见设计是对象存储做源，热数据进入文件系统或缓存。

第二个取舍是 checkpoint 频率。频率高，故障恢复损失小，但存储和网络压力大；频率低，训练效率高一些，但故障或抢占浪费更多 GPU 小时。Checkpoint 策略应结合故障率、任务优先级、写入耗时和恢复目标，而不是固定全局值。

第三个取舍是本地缓存与一致性。本地 NVMe 和 cache 能显著提升性能，但增加版本、淘汰和清理复杂度。模型权重和数据 shard 必须按版本管理，否则缓存会成为隐性状态。性能优化不能牺牲可复现和发布安全。

第四个取舍是统一存储与分层存储。统一系统简单，但可能在成本、性能或治理上折中；分层系统更高效，但需要数据流、权限、观测和生命周期管理。AI Factory 应把分层作为默认思路，再通过平台抽象降低用户复杂度。

最后一个取舍是用户自由度与平台约束。完全开放路径让高级用户灵活，但长期会产生不可治理的数据孤岛；完全封闭平台可以统一管理，但可能阻碍实验迭代。更好的方式是提供少数清晰抽象：数据集注册、checkpoint 策略、模型 artifact 发布、缓存策略和临时 scratch。用户表达意图，平台决定路径、清理和观测。这样既保留工程效率，也避免把存储系统变成脚本约定的集合。

存储设计还要在性能、可靠性和成本之间持续调节。所有数据都放在最高性能路径上，成本无法持续；所有数据都压到低成本对象存储，GPU 会等待；所有 checkpoint 永久保留，可靠性看似提高，治理却会崩溃。AI Factory 的存储策略应随数据生命周期变化：热时靠近计算，冷时回到低成本系统，进入发布链路时提高完整性和审计要求。


## 33.5 小结与延伸阅读

### 33.5.1 小结

- AI 存储影响数据读取、checkpoint、模型发布、推理冷启动和成本治理。
- 数据集、checkpoint、模型 artifact、日志和 cache 有不同访问模式，不能用单一存储策略覆盖。
- Object Storage、Parallel File System、Local NVMe 和 cache 应按生命周期分层组合。
- 存储验收要覆盖真实 data loader、checkpoint、模型加载、cache miss 和多节点并发。
- 存储指标必须绑定 tenant、job、model、dataset 和 path，才能支持排障和成本归因。


### 33.5.2 延伸阅读

- [Amazon S3 documentation](https://docs.aws.amazon.com/s3/)
- [Lustre Manual](https://doc.lustre.org/lustre_manual.xhtml)；[WEKA documentation](https://docs.weka.io/)；[Ceph documentation](https://docs.ceph.com/)
- [PyTorch Distributed Checkpoint documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [NVIDIA GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/)
