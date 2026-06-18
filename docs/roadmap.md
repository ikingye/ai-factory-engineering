# 写作路线图

## Phase 1：项目骨架和核心章节

完成 Material for MkDocs 工程、全书目录、章节模板、导论、核心示例章节、术语表和路线图。优先保证站点可构建、导航完整、后续可持续扩写。

## Phase 2：应用层与 Platform 层

扩写 Chat、RAG、Agent、行业应用、MaaS、AI Gateway、计量计费和平台可观测性。重点补充 token 消耗模式、请求治理、路由、限流、账单和 dashboard 的工程设计。

当前状态：已完成第 1-8 章的系统化初稿，后续需要补充官方引用、更多案例和跨章节索引。

## Phase 3：Model 与 Runtime 层

扩写大模型基础、预训练、后训练、微调、评测、模型服务、推理引擎、训练框架、分布式并行、通信原语和 GPU 软件栈。重点讲清模型机制如何影响显存、通信、吞吐和稳定性。

当前状态：已完成第 9-19 章的系统化初稿，后续需要补充官方引用、图表精修和更细的配置案例。

## Phase 4：调度、GPU IaaS、网络存储

扩写 AI workload、Kubernetes、Slurm、Volcano、Kueue、Ray、裸金属 GPU 云、虚拟化隔离、GPU 资源池、镜像驱动、AI 网络和存储系统。重点补充拓扑、配额、gang scheduling、RDMA、checkpoint 和数据读取路径。

当前状态：已完成第 20-33 章的系统化初稿，后续需要补充官方引用、网络/存储 benchmark 示例和跨章节故障树。

## Phase 5：物理基础设施、可靠性、验收

扩写 GPU 服务器、GPU 芯片与系统架构、AI 数据中心工程、AI Factory 可观测性、准入测试、故障诊断和 SRE 运维体系。重点沉淀验收基线、故障树和运行手册。

## Phase 6：经济性与案例

扩写 Token Factory 经济模型、AI Factory 商业模式、产业案例和从 0 到 1 建设路径。重点把 tokens/s、tokens/W、cost per token、revenue per token 与业务模型连接起来。

## Phase 7：审校、图表、索引和发布

统一术语、补充官方文档和经典论文链接、完善 Mermaid 图、增加索引与交叉引用，配置 GitHub Pages 发布流程，并进行全站构建检查。
