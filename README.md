# AI Factory Engineering

《AI Factory 工程：从应用到 GPU 基础设施》

## 项目定位

这是一本面向工程师的开源技术书，系统讲解 AI Factory 如何从应用请求、MaaS 平台、模型服务、训练推理运行时、资源编排、GPU IaaS、网络存储、物理基础设施一路落到可靠运行和商业化度量。

本项目不是一个简单 README，而是一个可长期扩写、可本地预览、可发布到 GitHub Pages 的 Material for MkDocs 书稿工程。

## 为什么写这本书

大模型时代的基础设施问题不再只是“买多少 GPU”或“怎么部署 Kubernetes”。真正的工程问题是：一个用户请求如何稳定地产生 token，一个训练任务如何变成可服务的模型，以及这些过程如何被调度、计量、诊断和商业化。

传统云计算知识仍然重要，但它不足以解释 GPU 拓扑、NCCL 通信、KV Cache、prefill/decode、准入测试、cost per token 等 AI Infra 的关键约束。本书试图把这些层次放在同一个系统里讨论。

## 这本书讲什么

- Application 层：Chat、RAG、Agent、行业应用如何消耗 token。
- Platform 层：MaaS、AI Gateway、计量计费、平台可观测性。
- Model 层：预训练、后训练、微调、评测、模型服务。
- AI Runtime 层：推理引擎、训练框架、分布式并行、CUDA/NCCL 软件栈。
- 资源编排与作业调度层：Container Runtime、Kubernetes、Slurm、Volcano、Kueue、Ray、GPU 调度。
- GPU IaaS 层：裸金属、虚拟化、资源池、镜像、驱动和初始化。
- 网络与存储层：NVLink、NVSwitch、InfiniBand、RoCE、RDMA、Object Storage、Parallel File System、Local NVMe。
- 物理基础设施层：机房、电力、制冷、GPU 服务器、NIC、Switch、Storage。
- 横切能力：可靠性、可观测性、准入验收、故障诊断、安全、成本和 Token Factory 经济性。

## 这本书不是什么

- 不是 LLM 应用开发速成教程。
- 不是单一 Kubernetes、Slurm 或 GPU 运维手册。
- 不是厂商产品白皮书。
- 不是把 AI Factory 简化成 GPU 集群的采购指南。
- 不是把 Token Factory 当成独立技术栈；本书把它作为 AI Factory 的产出度量和经济性视角。

## 全书结构

全书从“一个 Chat 请求如何变成 token”和“一个训练任务如何变成模型”两条主线展开，共 11 个 Part、45 章。当前仓库已经包含完整目录、章节骨架、写作规范、导论、Application 层和 Platform 层核心章节初版正文。

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve
```

浏览器访问本地输出的地址，通常是 `http://127.0.0.1:8000/`。

构建静态站点：

```bash
mkdocs build
```

## 贡献方式

欢迎围绕以下方向提交贡献：

- 扩写章节正文，补充真实工程场景。
- 按 `docs/style-guide.md` 统一章节结构、指标口径和图表风格。
- 校对术语、图表和架构关系。
- 补充配置示例、排障流程、验收清单。
- 增加可验证的官方文档、论文和工程案例链接。

建议一次 PR 聚焦一个章节或一个主题，避免大范围重排目录结构。

## License

正文建议采用 Creative Commons Attribution 4.0 International，示例配置和代码建议采用 MIT。正式发布前应在仓库中补充明确的 `LICENSE` 文件。
