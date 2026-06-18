# 术语表

## AI Factory

AI Factory 是把模型、数据、算力、平台和运维流程组织起来，持续生产 AI 能力和 token 的工程系统。它不是单一 GPU 集群，而是从应用、MaaS、模型服务、运行时、调度、GPU IaaS 到网络存储和物理设施的端到端生产链。

## Token Factory

Token Factory 是观察 AI Factory 产出的经济性视角。它把 token 当作可计量产出，关注 tokens/s、tokens/W、cost per token、revenue per token 和毛利结构。

## TokenFoundry

TokenFoundry 更适合作为产业组织或案例名词，而不是通用技术组件。讨论它时应关注组织如何围绕模型、算力、数据和客户需求组织 token 生产，而不是把它等同于 AI Factory。

## MaaS

MaaS 即 Model as a Service，指把模型能力以 API、SDK、控制台和配额体系交付给用户的平台形态。它通常包括模型目录、API Key、租户管理、路由、限流、计费和可观测性。

## LLM

LLM 即 Large Language Model，大语言模型。它通过大规模语料训练，学习文本 token 之间的统计关系，并在推理时根据上下文生成后续 token。

## Agent

Agent 是能够围绕目标进行规划、调用工具、读取上下文并多轮执行的 AI 应用形态。和普通 Chatbot 相比，Agent 往往产生更多中间调用、更多 token 放大和更复杂的可观测性需求。

## RAG

RAG 即 Retrieval-Augmented Generation，检索增强生成。它先从外部知识库检索相关内容，再把检索结果拼入上下文，让模型基于外部资料回答问题。

## Prompt

Prompt 是传给模型的输入指令、上下文和约束集合。它可以包含系统指令、用户消息、工具结果、检索片段和格式要求。

## Token

Token 是模型处理文本的基本单位，可以是词、子词、字符片段或符号。计量、限流、上下文长度和计费通常都围绕 token 展开。

## Tokenizer

Tokenizer 是把文本和 token 序列相互转换的组件。不同模型可能使用不同 tokenizer，因此同一段文本在不同模型下的 token 数可能不同。

## Context Window

Context Window 是模型一次推理可接收的最大上下文长度。它限制了 prompt、历史消息、工具结果和检索内容能同时放入模型的总 token 数。

## TTFT

TTFT 即 Time To First Token，表示请求发出后到第一个输出 token 返回的时间。它强烈受排队、路由、prefill 和首个 decode 步骤影响。

## TPOT

TPOT 即 Time Per Output Token，表示生成阶段每个输出 token 的平均耗时。它常用于衡量 decode 阶段的交互体验和吞吐能力。

## TPOP

TPOP 常用于描述 Time Per Output Prediction 或类似输出节奏指标，具体口径需要在团队内统一。工程上应明确它和 TTFT、TPOT、端到端延迟的边界，避免 dashboard 指标含义混乱。

## Prefill

Prefill 是推理中处理输入上下文并构建 KV Cache 的阶段。长上下文请求通常会显著增加 prefill 计算和显存压力。

## Decode

Decode 是模型逐 token 生成输出的阶段。它通常更受 KV Cache 访问、batching、调度策略和单步 kernel 效率影响。

## KV Cache

KV Cache 是保存 attention 中 Key 和 Value 张量的缓存，用于避免每次 decode 重复计算历史上下文。它提升生成效率，但会消耗大量 GPU HBM。

## Batching

Batching 是把多个请求合并执行以提高 GPU 利用率的技术。批处理越大通常吞吐越高，但排队和首 token 延迟可能变差。

## Continuous Batching

Continuous Batching 是推理引擎在 decode 过程中动态加入和移除请求的批处理方式。它比静态 batch 更适合不同长度请求混合的在线服务。

## Paged Attention

Paged Attention 是一种管理 KV Cache 的机制，用类似分页的方式减少显存碎片并提升长序列服务效率。它常见于现代 LLM 推理引擎。

## Speculative Decoding

Speculative Decoding 使用较小或较快的 draft 模型先生成候选 token，再由目标模型验证。它的目标是在保持输出一致性的同时降低生成延迟。

## PD 分离

PD 分离指把 prefill 和 decode 拆到不同资源池或服务实例上运行。它适合 prefill 与 decode 资源特征差异明显、且平台具备更精细流量治理的场景。

## CUDA

CUDA 是 NVIDIA GPU 的并行计算平台和编程模型。训练框架、推理引擎和许多算子库都依赖 CUDA 与驱动配合执行 GPU kernel。

## NCCL

NCCL 是 NVIDIA Collective Communications Library，用于 GPU 间集合通信。分布式训练中的 AllReduce、AllGather、ReduceScatter 等通信通常会依赖 NCCL。

## RDMA

RDMA 即 Remote Direct Memory Access，允许主机间绕过传统内核协议栈进行低延迟数据传输。AI 训练的跨节点通信经常依赖 RDMA 网络能力。

## InfiniBand

InfiniBand 是常见的高性能、低延迟网络技术。大规模训练集群常用它承载跨节点 GPU 通信和集合通信流量。

## RoCE

RoCE 即 RDMA over Converged Ethernet，让 RDMA 运行在以太网上。它需要更严格的网络配置、拥塞控制和丢包治理。

## NVLink

NVLink 是 GPU 与 GPU 或 GPU 与 CPU 间的高速互连技术。它主要解决节点内或近距离 scale-up 通信带宽问题。

## NVSwitch

NVSwitch 是连接多块 GPU 的交换芯片，使节点内 GPU 间获得更高带宽和更灵活的通信拓扑。它对大模型训练和推理并行效率很关键。

## GPU IaaS

GPU IaaS 指以基础设施方式交付 GPU 资源，包括裸金属、虚拟机、GPU 资源池、镜像、驱动、网络和存储挂载。它不同于 MaaS，后者交付的是模型 API 和平台能力。

## Bare Metal

Bare Metal 即裸金属服务器，用户或平台直接使用物理机而非普通虚拟机。大模型训练常偏好裸金属，以减少虚拟化开销并获得更明确的 GPU、NIC 和拓扑控制。

## Kubernetes

Kubernetes 是容器编排系统，负责 Pod 生命周期、服务发现、控制器和调度扩展。AI Factory 中它属于资源编排与作业调度层，而不是简单的 PaaS 标签。

## Slurm

Slurm 是 HPC 领域常用的作业调度系统。它擅长批式作业、队列、partition、资源分配和拓扑感知调度，经常用于大规模训练集群。

## Gang Scheduling

Gang Scheduling 要求分布式作业所需的一组 Pod 或进程同时获得资源后再启动。它避免训练任务只启动部分 worker 后长期占用资源却无法进展。

## Kueue

Kueue 是 Kubernetes 生态中的作业排队和配额管理组件。它适合为批处理、训练、推理等 workload 增加队列、准入和资源借用语义。

## Volcano

Volcano 是面向批处理和 AI/HPC workload 的 Kubernetes 调度系统。它提供 queue、gang scheduling、priority、preemption 等能力。

## GPU Operator

GPU Operator 用 Operator 模式自动化安装和管理 NVIDIA Driver、Device Plugin、DCGM Exporter、Container Toolkit 等组件。它降低了 GPU 节点软件栈运维复杂度。

## Device Plugin

Device Plugin 是 Kubernetes 暴露特殊硬件资源的扩展机制。NVIDIA GPU device plugin 会把 GPU、MIG 等资源注册给 kubelet，供 Pod 通过 resource request 使用。

## MIG

MIG 即 Multi-Instance GPU，可把部分 NVIDIA GPU 切分成多个隔离的 GPU 实例。它适合小模型推理、开发测试或多租户隔离，但会改变资源粒度和调度策略。

## DCGM

DCGM 即 Data Center GPU Manager，用于采集 GPU 健康、利用率、错误和诊断信息。它是 GPU 可观测性和故障排查的重要工具。

## HPL

HPL 即 High Performance Linpack，常用于衡量高性能计算系统的浮点计算能力。AI 集群验收会参考 HPL 或相关基准，但不能只看 HPL。

## NCCL Test

NCCL Test 是验证 GPU 集合通信性能和连通性的常用测试集。它能暴露跨 GPU、跨节点、跨网络链路的带宽和稳定性问题。

## nvbandwidth

nvbandwidth 是用于测量 NVIDIA GPU、NVLink、PCIe 等路径带宽的工具。它常用于节点级验收和拓扑异常诊断。

## cost per token

cost per token 表示生产一个 token 的单位成本。它需要把 GPU 折旧、电力、机房、网络存储、平台运维和利用率都纳入同一口径。

## tokens/s

tokens/s 表示单位时间产生或处理的 token 数。它是推理吞吐、训练数据处理和容量规划中最常用的产出指标之一。

## tokens/W

tokens/W 表示每瓦功耗可以产生的 token 数。它把能效纳入 AI Factory 的经济性分析，适合比较架构、模型、推理引擎和机房设计。
