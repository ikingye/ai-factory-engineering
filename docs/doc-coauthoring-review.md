# Doc Coauthoring 审稿标准

本页把 `doc-coauthoring` 的三段式方法落成《AI Factory Engineering》的全书审稿制度。它不是额外正文，而是每次扩写、重构和发布前的工作流：先补齐上下文，再逐节精修，最后用读者问题测试章节是否真正可用。

## 1. 目标读者

默认读者不是 AI Infra 新手，而是已经具备云计算、Kubernetes、后端工程或基础设施经验的工程师。他们读本书的目标不是获得术语解释，而是能在真实 AI Factory 项目中做设计评审、上线验收、故障定位、成本归因和容量决策。

每章必须同时服务三类读者：

- 负责落地的人：需要知道对象、接口、配置、命令、状态机和验收动作。
- 负责排障的人：需要知道症状、第一层证据、跨层追踪路径和责任边界。
- 负责决策的人：需要知道方案取舍、成本影响、风险边界和停止条件。

## 2. Stage 1：Context Gathering

扩写任意章节前，先回答上下文问题：

- 本章属于 AI Factory 哪一层，和相邻层的边界是什么？
- 本章要解释哪条生产路径：推理请求、训练任务、数据链路、资源交付、故障处理，还是经济账本？
- 本章读者读完后应该能排查哪类问题，设计哪类系统，或做哪类验收？
- 本章依赖哪些前置章节，后续章节会继续展开哪些主题？
- 哪些结论需要官方文档、标准、论文、代码实现或公开工程案例支撑？
- 哪些内容变化快，必须用“常见做法是”“一种典型架构是”等中性表达？

上下文收集的输出不是长背景，而是本章开头的 `本章回答的问题`、`本章上下文`、`读者测试`、`一个真实场景` 和 `系统架构` 能否互相支撑。若场景和读者测试无法对应，本章需要重写。

## 3. Stage 2：Refinement & Structure

每章按固定骨架逐节精修：

| 模块 | 审稿问题 |
| --- | --- |
| 本章回答的问题 | 是否明确告诉读者本章解决什么工程问题，而不是列概念？ |
| 本章上下文 | 是否交代层级定位、前置依赖、后续关联和读完后能做什么？ |
| 读者测试 | 是否有机制、边界、路径、排障、验收或经济性问题？ |
| 一个真实场景 | 是否来自生产约束、故障、上线或容量压力？ |
| 核心概念 | 是否解释术语关系、责任边界和常见误解？ |
| 系统架构 | 是否有文字和图说明控制流、数据流或状态流？ |
| 分层编号正文节 | 是否使用 H2 大组和 H3 小节组织内容，且正文机制节逐节解释机制，而不是只写定义？ |
| 工程实现 | 是否给出 YAML、命令、伪代码、配置、契约或操作流程？ |
| 常见故障 | 是否能映射到日志、事件、指标或证据包？ |
| 性能指标 | 是否说明口径、标签、使用场景和误读风险？ |
| 设计取舍 | 是否说明何时选择 A，何时选择 B，选错会怎样？ |
| 小结 | 是否收束成可复用工程判断，而不是重复目录？ |

精修时优先删除三类内容：只重复标题的段落、没有落地证据的抽象判断、不能帮助读者做设计或排障的名词堆砌。允许章节变长，但增长必须来自机制、路径、配置、指标、故障树、验收矩阵或跨章节索引。

## 4. Stage 3：Reader Testing

每章都必须包含 `读者测试` 模块。它不是习题，而是作者对读者能力的承诺。读者测试至少覆盖四类问题：

- 机制题：读者是否能说明组件如何工作？
- 边界题：读者是否能说明组件不负责什么？
- 路径题：读者是否能追踪一个请求、任务、设备、数据或账单事件？
- 排障题：读者是否能从症状找到第一层证据和下一跳证据？

涉及验收、可靠性、安全或经济性的章节，还必须加入验收题、风险题或经济题。若读者测试提出的问题无法在本章或明确的跨章节路径中回答，本章视为未完成。

## 5. 全书级读者测试

完成一轮主题链路重构后，用以下问题测试全书是否闭环：

- 一个 Chat 请求从 Gateway 到 GPU/HBM 再到 token 计量，读者能否画出路径并定位 TTFT 异常？
- 一个训练任务 pending，读者能否区分 quota、gang scheduling、拓扑、镜像、PVC、节点健康和资源池状态？
- 一个容器内 `nvidia-smi` 正常但 NCCL 失败，读者能否追到 runtime、RDMA、NUMA、NCCL env 和 fabric telemetry？
- 一个模型发布后质量下降，读者能否追到 dataset、judge、serving bundle、route contract、RAG/Agent 和 rollback？
- 一个 GPU 集群验收通过后线上仍慢，读者能否区分准入基线、真实 workload、调度放置和观测缺口？
- 一个账单争议发生时，读者能否从 invoice 回放到 metering event、policy decision、served model、价格版本和成本账本？
- 一个私有化客户升级失败，读者能否追到离线包、导入记录、现场补丁、诊断导出和支持成本？
- 一个技术负责人决定扩容，读者能否把 workload profile、tokens/s、tokens/W、SLO、质量、GPU hours 和 P&L 放在同一张图里？

## 6. 主题链路读者测试矩阵

| 主题链路 | 主要章节 | Reader Test | 通过标准 |
| --- | --- | --- | --- |
| 推理请求链路 | 第 1、5、6、7、8、14、15、37、39、41、44 章 | 一个 Chat 请求 TTFT 升高、部分 streaming 中断、账单 usage 与用户感知不一致。 | 读者能追踪 Gateway admission、route、serving release、engine queue、prefill/decode、KV block、metering event、trace 和成本账本。 |
| 训练任务链路 | 第 10、16、17、18、20、23、24、33、37、38、39、41、44 章 | 一个 512 卡训练任务 pending 后启动，首个 step 慢，checkpoint 后 NCCL hang。 | 读者能区分 queue/quota/gang/topology、launcher/rendezvous、rank mapping、NCCL、storage overlap、baseline 和 training incident cost。 |
| GPU 容器链路 | 第 19、21、22、29、38、39、44 章 | Pod 申请 1 张 GPU，却在容器内看到错误设备；另一个 Pod `nvidia-smi` 正常但 NCCL 失败。 | 读者能从 driver、Container Toolkit、CRI、OCI hook/CDI/NRI、device plugin、RuntimeClass、GPU UUID/MIG、RDMA device、可见性对账和准入矩阵定位问题。 |
| 网络通信链路 | 第 18、30、31、32、37、38、39、41、44 章 | 某 rack 训练 step time 抖动，NCCL all_reduce 长尾，端口有 ECN/PFC 事件。 | 读者能把 rank、GPU/NIC、rail、switch port、fabric baseline、NCCL op、checkpoint overlap、拥塞故障树和 network cost ledger 连起来。 |
| 存储与供应链链路 | 第 10、14、29、33、37、38、39、41、44 章 | 模型发布后 tokenizer mismatch，旧 cache 继续服务，RAG ACL 变更未失效。 | 读者能追踪 dataset lineage、checkpoint restore、artifact provenance、cache invalidation、release contract、billing hold 和 supply-chain fault tree。 |
| 质量证据链路 | 第 2、3、6、13、14、37、39、41、44 章 | 评测 gate 曾经通过，但 route、judge、RAG index 或 tool policy 变化后线上质量下降。 | 读者能判断 quality evidence 是否过期，冻结 replay bundle，执行质量故障树，并把低质量 token 和人工接管写入成本账本。 |
| 安全与租户链路 | 第 5、6、7、8、27、33、37、39、40、41、44 章 | 合法 API key 被滥用触发 denial-of-wallet，trace 中又可能包含敏感 prompt。 | 读者能追踪 tenant boundary、policy decision、provider egress、prompt redaction、secret boundary、billing replay、安全证据和 abuse cost ledger。 |
| 商业与建设链路 | 第 4、5、7、28、36、40、41、42、43、44 章 | 技术负责人要决定扩容或私有化交付，但 GPU 利用率、SLA、支持成本和毛利互相矛盾。 | 读者能把 workload profile、customer onboarding、capacity activation、commercial P&L、release train、support taxonomy 和 PRR stop condition 放在同一决策框架中。 |

每轮大规模重构至少选择一条主题链路执行 Reader Test。若矩阵中的通过标准无法在章节中找到明确答案，应优先修正文档，而不是降低测试标准。

## 7. GPU 容器链路 Reader Test 记录

本轮以 GPU 容器链路作为第一条完整读者测试链路。测试假设读者没有当前对话上下文，只能阅读本书页面。

| Reader Question | 应阅读章节 | 期望读者能给出的答案 | 当前状态 |
| --- | --- | --- | --- |
| Pod 申请 1 张 GPU，但容器内看到 8 张 GPU，应该先查哪里？ | 第 21、22、38、39 章 | 先比较 Kubernetes 分配事实、device plugin Allocate 输出、OCI/CDI/runtime 注入差异和容器内 `nvidia-smi -L`；若可见设备多于分配设备，应隔离节点并触发 `gpu_device_visibility_reconciliation`。 | 已覆盖 |
| Docker `--gpus all` 正常，Kubernetes Pod 看不到 GPU，说明什么？ | 第 21、22、29、38 章 | Docker 路径正常不代表 kubelet 使用的 CRI/runtime 路径正常；应查 kubelet CRI endpoint、containerd/CRI-O runtime handler、RuntimeClass、device plugin、CDI spec 和 Toolkit 配置。 | 已覆盖 |
| 容器内 `nvidia-smi` 正常但 NCCL 失败，为什么不能直接判定 GPU 正常？ | 第 18、21、22、32、38、39 章 | `nvidia-smi` 只证明 NVML/driver 基础路径，NCCL 还依赖 CUDA/NCCL 版本、GPU/NIC/NUMA、RDMA device、NCCL env、rail、fabric 和 collective 一致性。 | 已覆盖 |
| 从 legacy hook 切到 CDI/NRI，PRR 应验什么？ | 第 21、22、38、44 章 | 验证非 GPU Pod 不可见 GPU、单 GPU/MIG/多卡 RDMA Pod 可见性与分配一致、OCI/CDI 注入差异可解释、DCGM 标签不断链、故障树和回滚路径可执行。 | 已覆盖 |
| 为什么 GPU Operator DaemonSet Ready 不等于 GPU runtime 变更成功？ | 第 19、22、29、38、44 章 | Operator Ready 只说明控制器和 DaemonSet 状态，仍需验证 driver/Toolkit/device plugin/DCGM/MIG/Runtime 策略、container runtime baseline、容器内事实和观测标签。 | 已覆盖 |
| GPU 容器链路事故如何进入成本账本？ | 第 39、41、44 章 | 扩容失败、设备错配、拓扑错配、回滚复测、观测断链和业务 fallback 会进入 `container_runtime_incident_cost_record`、可靠性成本、计量修正和 PRR gate 更新。 | 已覆盖 |

本轮结论：GPU 容器链路已经具备 doc-coauthoring 需要的读者问题、上下文、跨章节答案和审计入口。下一轮可继续选择推理请求链路或训练任务链路做同等强度 Reader Test。

## 8. 审计命令

每轮重构后运行：

```bash
python3 tools/audit_doc_coauthoring.py
python3 tools/audit_heading_numbering.py
python3 tools/audit_consistency.py
python3 tools/audit_depth.py --limit 160
git diff --check
python3 tools/audit_placeholders.py
mkdocs build --strict
```

其中 `audit_doc_coauthoring.py` 负责检查章节是否具备 `读者测试` 模块，以及该模块是否覆盖机制、边界、路径和排障问题。`audit_consistency.py` 负责检查全书高风险口径漂移、旧结构规则残留、公开站点 URL 和精确重复长段落。它们不能替代人工审稿，但能防止全书退回“有正文、无读者测试”或“局部正确、全书矛盾”的状态。

## 9. 完成定义

一章只有满足以下条件，才算符合本书的 doc-coauthoring 要求：

- 上下文清楚：本章定位、读者、生产路径和跨章节关系明确。
- 结构完整：章节模板完整，H2 大组和 H3 小节按章号连续编号。
- 读者可用：`读者测试` 的问题能在本章或明确链接的后续章节中回答。
- 机制具体：至少有一条路径、状态机、架构图、配置、命令或契约能落地。
- 证据可查：涉及版本、接口、产品能力、标准和性能口径时，不编造，不偷换。
- 审计通过：结构审计、标题审计、深度审计、占位检查和 MkDocs 构建全部通过。
