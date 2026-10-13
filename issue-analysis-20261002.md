# vLLM Issue 难度分析

> 分析日期：2026-10-02
> Issue 来源：[vllm-project/vllm](https://github.com/vllm-project/vllm)

## Issue 清单（修正后）

| # | Issue | 标题 |
|---|-------|------|
| 1 | #59273 | ThinkingBudgetStateHolder.sync_batch 单向移动泄漏陈旧状态 |
| 2 | #59181 | VLLM_BATCH_INVARIANT=1 在 sm_120 上对 pooling 模型无效 |
| 3 | #59151 | ROCm mxfp4 MoE: TRITON_UNFUSED 未被自动选择 |
| 4 | #59116 | SMG MoRI-IO PD disagg 不工作 |
| 5 | #59114 | MoRIIO WRITE: producer 死后 decode 请求 hang，KV 占用不释放 |
| 6 | #59110 | NixlConnector 异构 TP 静默误读 KV 页面 |
| 7 | #59109 | PP on docker: gloo 控制平面解析到 bridge IP（535ms/次） |
| 8 | #59087 | `--tool-call-parser mistral` 在非 Mistral tokenizer 上导致所有请求 500 |
| 9 | #59086 | VLLM_BATCH_INVARIANT=1 在 sm8x 上对 AWQ 模型无效 |
| 10 | #59352 | [Doc] Hopper NVSwitch 节点 TP 不可复现，NCCL_NVLS_ENABLE=0 可修复 |
| 11 | #59152 | mimo_audio.py 在 ROCm 上导入 CUDA-only flash_attn 导致引擎崩溃 |

---

## 难度分级

### 🔴 难度：高（分布式系统 / 底层内核 / 跨平台架构）

#### #59110 — NixlConnector 异构 TP 静默误读 KV 页面
- **标签**：`rocm`
- **问题**：`NixlConnector` 在异构 TP（decode TP ≠ prefill TP）下，对使用 split K/V slot 布局的 ROCm attention backend 按连续字节范围拆分 KV block，导致数据损坏或引擎崩溃
- **难点**：涉及 KV cache 内存布局、异构 TP 下 K/V split-slot 的字节级拆分逻辑、handshake 元数据验证。需理解多种 ROCm attention backend 的内部页面格式
- **修复方向**：① 临时拒绝 `num_head_slots == 2` 时的异构 TP ② 实现 K/V slot 分别拆分 ③ 在 `NixlAgentMetadata` 中添加页面格式 tag

#### #59114 — MoRIIO WRITE: producer 死后 decode 请求 hang
- **标签**：`rocm`
- **问题**：MoRI-IO PD disaggregation 中，producer 节点死亡后，decode 端的 WRITE 请求不返回，客户端 hang 直到超时，KV cache 占用不释放
- **难点**：涉及 PD disaggregation 的分布式通信协议、InfiniBand RDMA 写入语义、错误恢复路径。需理解 MoRI-IO connector 的生命周期管理
- **修复方向**：添加 writer 端死亡检测、超时机制、KV 资源清理

#### #59181 — VLLM_BATCH_INVARIANT=1 在 sm_120（Blackwell）上对 pooling 模型无效
- **标签**：无
- **问题**：sm_120 GPU（RTX 5060 Ti, compute capability 12.0）上 pooling 模型评分结果因 batch 大小不同而差异 1 ulp
- **难点**：需深入理解 CUDA GEMM kernel 选择、cuBLAS workspace 配置、compute capability 特定行为。涉及 `torch._inductor` 的 batch-invariant override 注册机制
- **修复方向**：将 sm_120 加入 invariant GEMM 列表，或在启动时发出警告

#### #59086 — VLLM_BATCH_INVARIANT=1 在 sm8x 上对 AWQ 模型无效
- **标签**：`bug`, `quantization`
- **问题**：AWQ 模型在 sm8x（H100/RTX 4090）上使用 torch.compile 时，greedy 输出因 batch 内其他请求而变化。`--enforce-eager` 可绕过
- **难点**：根因在 PyTorch inductor 编译器层面，调试需深入 inductor 的 kernel 生成逻辑和 AWQ 量化路径的非确定性来源
- **修复方向**：定位 inductor 为 AWQ 生成的 kernel 中导致非确定性的 tile/reduction 策略

---

### 🟠 难度：中高（平台特定 / 状态管理 / 集成问题）

#### #59116 — SMG MoRI-IO PD disagg 不工作
- **标签**：`bug`
- **问题**：ROCm vLLM 使用 MoRI-IO PD Disagg Connector 与 SMG router 集成时无法正常工作
- **难点**：涉及 PD disaggregation 架构、外部 SMG router 集成、ROCm 平台兼容性。调试依赖外部组件，复现环境搭建复杂
- **修复方向**：排查 MoRI-IO connector 与 SMG router 的协议兼容性

#### #59109 — PP Docker 中 Gloo 控制平面解析到 bridge IP（535ms/次）
- **标签**：无
- **问题**：单容器 PP 部署中 gloo 将主机名解析为 Docker bridge IP 而非 loopback，导致每次 send ~535ms 延迟，长上下文 decode 吞吐严重下降
- **难点**：涉及 Docker 网络命名空间、Gloo 传输层行为。诊断需 profiling 网络栈
- **修复方向**：vLLM 检测同命名空间 rank 时默认 `GLOO_SOCKET_IFNAME=lo`，或发出警告

#### #59273 — ThinkingBudgetStateHolder.sync_batch 单向移动时泄漏陈旧状态
- **标签**：`bug`, `mrv1-only`
- **问题**：`sync_batch` 中无 thinking budget 的请求移入有 budget 的请求留下的槽位时，旧状态未被清除，新请求 token 被计入已离开请求的 budget
- **难点**：根因明确（未无条件 pop 目标槽位），但涉及 thinking budget 的核心调度路径，修改需谨慎验证
- **修复方向**：在 `sync_batch` 中无条件 pop 两端索引

---

### 🟡 难度：中（明确 bug / 修复路径清晰）

#### #59087 — `--tool-call-parser mistral` 在非 Mistral tokenizer 上导致所有请求 500
- **标签**：`bug`, `tool-calling`, `mistral`
- **问题**：接受 `--tool-call-parser mistral` 搭配非 Mistral tokenizer，服务启动正常但所有 chat 请求返回 HTTP 500。其他 parser（hermes/granite）能优雅降级
- **难点**：根因清晰（缺少 `[TOOL_CALLS]` token 时直接抛异常），修复方向明确但需考虑 parser 架构通用性
- **修复方向**：① 启动时校验 tokenizer 是否包含必需 token，缺失则报错退出 ② 运行时仅对涉及 tool 的请求返回错误

#### #59151 — ROCm mxfp4 MoE: TRITON_UNFUSED 未被自动选择
- **标签**：`rocm`
- **问题**：ROCm 上 mxfp4 MoE 自动后端选择遗漏 `TRITON_UNFUSED`，导致 pre-CDNA3 硬件（gfx90a/MI210/MI250）无可用原生后端
- **难点**：需理解 ROCm 后端选择优先级、CDNA 架构代际差异。修复明确但需评估 MTP 在 ROCm 上的副作用
- **修复方向**：在 `mxfp4.py` 的 `_get_priority_backends()` 中插入 `Mxfp4MoeBackend.TRITON_UNFUSED`

---

### 🟢 难度：低（局部修复 / 文档 / 简单 import 问题）

#### #59152 — mimo_audio.py 在 ROCm 上导入 CUDA-only flash_attn 导致引擎崩溃
- **标签**：`rocm`, `multi-modality`
- **问题**：`mimo_audio.py` 硬编码导入 `vllm.vllm_flash_attn`（CUDA 专用），ROCm 上处理音频请求时引擎崩溃。`mimo_v2_omni.py` 中同类问题已修复
- **难点**：简单 import 路径错误，修复方式已有先例
- **修复方向**：路由到 `MMEncoderAttention` 或添加兼容 shim

#### #59352 — [Doc] Hopper NVSwitch 节点 TP 不可复现，文档缺失
- **标签**：无
- **问题**：8×H20 + NVSwitch 上 NVLS all-reduce 结果在不同运行间不一致。`NCCL_NVLS_ENABLE=0` 可 1% 延迟代价解决（vs 完整 batch-invariant 的 76% 代价）
- **难点**：纯文档修改，无代码变更
- **修复方向**：在 `docs/usage/reproducibility.md` 和 `docs/features/batch_invariance.md` 中添加 NVLS 说明

---

## GPU 验证需求

### 🟢 不需要 GPU（本地可验证）

| Issue | 原因 |
|-------|------|
| #59352 | 纯文档修改 |
| #59087 | tool-call-parser 是纯 Python 文本解析，用 Qwen2.5-0.5B + CPU 模式即可复现 |
| #59273 | 纯 Python 状态管理逻辑，可写单元测试 mock 掉 GPU 依赖 |

### 🟡 需要 GPU，但不挑型号

| Issue | 原因 |
|-------|------|
| #59109 | Gloo 是 CPU 传输层，核心 bug 是网络接口选择，任意支持 PP 的 GPU 可复现 |

### 🔴 需要特定 GPU 型号

| Issue | 需要什么 GPU | 原因 |
|-------|-------------|------|
| #59152 | AMD ROCm（任意） | ROCm 平台 import 路径问题 |
| #59151 | AMD gfx90a（MI210/MI250） | 必须 pre-CDNA3 才触发 |
| #59116 | AMD ROCm + 多节点 | PD disaggregation + SMG router |
| #59114 | AMD MI308X + InfiniBand | MoRI-IO RDMA 写入问题 |
| #59110 | AMD ROCm（多张，不同 TP） | 异构 TP + split K/V backend |
| #59086 | NVIDIA sm8x（H100/RTX 4090） | AWQ + torch.compile 非确定性 |
| #59181 | NVIDIA sm_120（Blackwell） | 只在 Blackwell 架构触发 |

---

## 建议处理顺序（从易到难）

```
#59352 → #59152 → #59087 → #59273 → #59151 → #59109 → #59116 → #59086 → #59181 → #59114 → #59110
```

| 等级 | Issue 数 | 预估工作量 |
|------|----------|-----------|
| 🟢 低 | 3 | 各 30 分钟 ~ 1 小时 |
| 🟡 中 | 2 | 各 1-2 小时 |
| 🟠 中高 | 3 | 各 2-4 小时 |
| 🔴 高 | 4 | 各 1-3 天 |
