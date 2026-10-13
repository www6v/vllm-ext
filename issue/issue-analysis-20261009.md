# vLLM Issues 可复现性分析与修复评估

> 分析日期：2026-10-09
> 机器：RTX 3090 (sm_86, 24GB) / CUDA 13.0 / PyTorch 2.13.0+cu130 / vLLM 0.28.0 / src: vllm-main/

---

## 总览：难易程度排序

| 优先级 | Issue | 标题 | 难度 | 本机可验证？ | 可修复？ |
|:---:|:---:|------|:---:|:---:|:---:|
| 1 | #60703 | GLM-4.7 流式解析器吞 tool call | 简单 | CPU | YES |
| 2 | #60744 | pause_generation keep+clear_cache 报错 | 中等 | CPU | YES (PR#60764) |
| 3 | #60608 | Exllama GPTQ group_size%32!=0 静默错误 | 中高 | GPU | 需CUDA编译 |
| 4 | #60499 | RMSNorm+int8 未融合 prefill慢2x | 困难 | sm_86!=sm_89 | 性能优化 |
| X | #60755 | Mooncake KV Decode超时崩溃 | 困难 | 缺硬件 | NO |

---

## 本机环境

```
GPU:            NVIDIA GeForce RTX 3090 (1x), 24GB, sm_86
Driver:         580.178.04 / CUDA: 13.0
PyTorch:        2.13.0+cu130 / cuDNN: 9.20.0
Python:         3.10.12 (system) / 3.13 avail
vLLM (pip):     0.28.0 / vLLM (src): vllm-main/
OS:             Linux 6.8.0-138-generic
```

---

## Issue 详细分析

---

### Issue #60703 — GLM-4.7 流式解析器吞掉 tool call [简单]

**链接**: https://github.com/vllm-project/vllm/issues/60703
**标签**: `tool-calling` | **状态**: Open

#### 问题描述

GLM-4.7/5.x 模型在正文中意外输出 tool_call 特殊字符时，解析器打开"幻影"工具调用，然后**静默丢弃后续所有内容**：剩余正文、think标签、真实工具调用全部丢失。HTTP 200 返回、finish_reason=stop、无 tool_calls、无日志警告。约 3 周内命中 4 次。

#### 根本原因

`glm47_moe_config` 状态机缺少 (TOOL_NAME, TOOL_START) 转换。stray tag 把引擎置于 TOOL_NAME 状态后，后续正文 token 和真实 tool_call 的开始符都追加到幻影调用的 name 缓冲区。最终 _accept_tool_name 因 validate_tool_names=True 拒绝该名称，所有内容被丢弃。

#### 本机可复现性: YES

- **纯 CPU，不需 GPU/权重/服务器**
- 源码文件:
  - `vllm-main/vllm/parser/glm47_moe.py` — 解析器状态机配置
  - `vllm-main/vllm/tool_parsers/glm47_moe_tool_parser.py` — 工具解析器
  - `vllm-main/tests/parser/engine/test_glm47_moe.py` — 现有测试
  - `vllm-main/tests/tool_parsers/test_glm47_moe_tool_parser.py` — 现有测试

#### 修复方案

1. 在 glm47_moe.py 中添加 (TOOL_NAME, TOOL_START) -> TOOL_NAME 转换
2. 进入时 emit TOOL_CALL_END 关闭幻影调用，然后处理 TOOL_START
3. 注意 ling3.py 也导入了 glm47_moe_config，修复会影响它
4. 编写测试覆盖：stray+real call、双stray、arg_value中的字面量、干净调用

#### 修复难度评估

- **难度**: 低 — 只需添加一个状态机转换 + 测试
- **预计工作量**: 2-4 小时
- **风险**: 低 — 改动范围小，现有测试可验证

---

### Issue #60744 — pause_generation(keep, clear_cache=True) 流式 session 暂停报错 [中等]

**链接**: https://github.com/vllm-project/vllm/issues/60744
**标签**: 无 | **状态**: Open | **已有 PR**: #60764

#### 问题描述

调用 `pause_generation(mode="keep", clear_cache=True)` 时，如果存在一个在 turn 之间暂停的可恢复流式输入 session，会抛出 RuntimeError，并使引擎停留在 PAUSED_ALL 状态。

这是 `docs/training/async_rl.md` 中文档推荐的精确调用方式，也是四种 mode/clear_cache 组合中**唯一失败**的：

| mode | clear_cache | 结果 |
|---|---|---|
| keep | True | **RuntimeError** |
| keep | False | ok |
| abort | True | ok |
| abort | False | ok |

#### 根本原因（6 步代码路径）

1. 可恢复请求未结束 — `_handle_stopped_request` 对 resumable 请求返回 False (scheduler.py:2464)
2. 请求离开 `self.running` 进入 `kv_holding_waiting` (scheduler.py:2294, 2456)
3. `_holds_kv_blocks` 对该请求为 true (scheduler.py:2444)
4. `reset_prefix_cache(reset_running_requests=True)` 只清空 `self.running`，**不清空 `kv_holding_waiting`** (scheduler.py:2839, 2857-2859)
5. `block_pool.reset_prefix_cache()` 因 block 仍被使用返回 False (block_pool.py:827, 838, 839)
6. scheduler 在 scheduler.py:2868-2869 抛出 RuntimeError

#### 本机可复现性: YES

- **纯 CPU，不需 GPU/权重**
- issue 提供了完整的 scheduler 级和 engine 级复现代码
- 源码文件:
  - `vllm-main/vllm/v1/core/sched/scheduler.py` — scheduler 核心逻辑
  - `vllm-main/vllm/v1/engine/core.py` — engine core，pause_scheduler 方法
  - `vllm-main/vllm/entrypoints/rl/online/api_router.py` — HTTP API 入口

#### 修复方案（PR #60764 已提交）

1. `reset_prefix_cache(reset_running_requests=True)` 额外清空 `kv_holding_waiting`
2. `_update_request_as_session` 中的 resume fold 在 `num_computed_tokens==0` 时不再清除 trajectory
3. 修正 api_router.py 的文档字符串
4. 新增 4 个 scheduler 测试，225 个测试通过

#### 修复难度评估

- **难度**: 中 — 涉及 scheduler 状态机多个队列的交互
- **预计工作量**: 4-8 小时（自己实现）/ 1-2 小时（基于 PR#60764 cherry-pick）
- **建议**: PR 已开，可以先 review PR 代码，理解修复逻辑后在本机源码上验证
- **风险**: 中 — scheduler 是核心组件，需确保不影响其他路径

---

### Issue #60608 — Exllama GPTQ 内核 group_size%32!=0 时静默输出错误 [中高]

**链接**: https://github.com/vllm-project/vllm/issues/60608
**标签**: `rocm`, `quantization` | **状态**: Open

#### 问题描述

Exllama GPTQ GEMM/reconstruct 内核（4-bit 和 8-bit）以及 RDNA3 W4A16 内核以 32 为步长遍历 K 维度，仅在步长起点切换量化组。当 `group_size % 32 != 0`（如 group_size=16）时，`k` 永远不等于 `nextgroup`，导致整个 K 块使用第一组的 scales/zeros，**静默产生错误结果**。

**关键点**: 这是一个防御性正确性修复 — 目前已知发布的 INT4 GPTQ checkpoint 都用 group_size=32/64/128，没有 group_size=16 的。但系统接受该配置却不报错。

#### 根本原因

CUDA 内核 `csrc/libtorch_stable/quantization/gptq/q_gemm.cu` 中：
- 内核以 32 为步长遍历 K（4 packed int32 words x 8 nibbles）
- 组切换条件 `k == nextgroup` 在 group_size=16 时永不满足
- `can_implement` / MoE oracle 只检查 `group_size > 0` 和 `K % group_size == 0`
- 对 fp16 激活 + group_size=16，Marlin/Conch/Triton W4A16 都拒绝该配置
- 导致 `choose_mp_linear_kernel` 选中 ExllamaLinearKernel，返回错误结果

#### 本机可复现性: YES（有条件）

- RTX 3090 (sm_86) 是 CUDA 设备，Exllama 内核会被选中
- **但**: 需要构造 group_size=16 的 GPTQ 权重来触发
- 可以用合成的量化权重写单元测试，不需要真实模型
- 源码文件:
  - `vllm-main/vllm/model_executor/kernels/linear/mixed_precision/exllama.py` — 内核选择逻辑
  - `vllm-main/csrc/libtorch_stable/quantization/gptq/q_gemm.cu` — CUDA 内核

#### 修复方案

1. 在 `ExllamaLinearKernel.can_implement` 中添加 `group_size % 32 != 0` 的拒绝逻辑
2. 同样修复 RDNA3W4A16LinearKernel.can_implement 和 RDNA3 WNA16 MoE oracle
3. 编写测试：构造 group_size=16 的合成权重，验证被正确拒绝

#### 修复难度评估

- **难度**: 中高 — 需要理解量化内核选择逻辑和 CUDA 内核行为
- **预计工作量**: 4-8 小时
- **挑战**:
  - 需要重新编译 CUDA 扩展来验证（nvcc 未在 PATH 上）
  - 需要构造合适的测试权重
  - Python 层的 can_implement 修复相对简单，CUDA 层修复更复杂
- **建议**: 可以先只做 Python 层的 can_implement 拒绝（不需重编译 CUDA），验证配置被正确拒绝

---

### Issue #60499 — RMSNorm 与 int8 量化未融合，prefill 时慢 2x [困难]

**链接**: https://github.com/vllm-project/vllm/issues/60499
**标签**: `quantization` | **状态**: Open

#### 问题描述

对 int8 W8A8 模型（compressed-tensors，动态 per-token 激活），每个 decoder 层将 RMSNorm 和 `dynamic_scaled_int8_quant` 作为两个独立内核运行。RMSNorm-quant 融合 pass（`rms_quant_fusion.py`）注册了 FP8 和 NVFP4 量化 op，但**没有 int8 的**，尽管 `rms_norm_dynamic_per_token_quant` 已支持 int8 输出。

**性能数据**（RTX 4090, sm_89）：
- Decode 阶段端到端提升：1.2% → 2.2%（batch 1-128）
- Prefill（8 prompts x 512 tokens）提升：6.8%
- 但在 prefill 尺寸（1024+ tokens）下，现有融合 CUDA 内核比两个独立内核**慢 2 倍**

#### 本机可复现性: 部分可以

- RTX 3090 是 sm_86（Ampere），issue 报告基于 sm_89（Ada/4090）
- 融合 pass 缺失 int8 这一事实是**架构无关的**，可以在 sm_86 上验证
- 但性能数据可能不同，2x slowdown 可能特定于 sm_89
- issue 提供了独立可运行的 benchmark 脚本（只需 torch + vLLM）
- 源码文件:
  - `vllm-main/vllm/compilation/passes/fusion/rms_quant_fusion.py` — 融合 pass
  - `vllm-main/vllm/kernels/helion/ops/rms_norm_dynamic_per_token_quant.py` — 融合内核

#### 修复方案

1. 在 `rms_quant_fusion.py` 的 `QUANT_OPS` 中添加 int8 动态 per-token 模式
2. 定义 add+RMSNorm+int8-quant 和 SiLU-and-mul+int8-quant 的融合模式
3. 可选：编写更快的 Triton 内核（issue 作者已有参考实现）
4. 可选：修复现有融合 CUDA 内核在 prefill 尺寸下的性能问题

#### 修复难度评估

- **难度**: 高 — 涉及 torch.compile 融合 pass、Triton/CUDA 内核优化
- **预计工作量**: 1-3 天
- **挑战**:
  - 需要理解 vLLM 的 torch.compile 融合 pass 框架
  - 需要在 sm_86 上重新 benchmark（结果可能与 sm_89 不同）
  - 可能需要编写/优化 Triton 内核
  - 性能优化类 PR 的 review 标准更严格
- **建议**: 这是性能优化而非 bug fix，优先级较低。建议先从 #60703 或 #60744 入手

---

### Issue #60755 — Mooncake KV 传输 Decode 超时后崩溃 [困难/不可行]

**链接**: https://github.com/vllm-project/vllm/issues/60755
**标签**: `bug` | **状态**: Open

#### 问题描述

vLLM 0.31.0 + Mooncake KV transfer 环境中，当 P/D（Prefill/Decode）请求被取消或 Decode 不可用导致超时时，Decode EngineCore 在恢复后会崩溃。

**错误**: `assert req_id in self.requests` — scheduler.py:3148

**复现步骤**:
1. 运行分离的 Prefill 和 Decode workers + Mooncake KV transfer
2. 通过 P/D frontend 发送请求
3. 暂停或使 Decode worker 不可用
4. 让 frontend 请求超时
5. 恢复 Decode 并允许挂起的 KV 清理操作完成

#### 本机可复现性: NO

- **需要 Mooncake 传输引擎**（mooncake-transfer-engine-cuda13）— 专用硬件/网络
- **需要 P/D 分离部署** — 至少两个 worker（Prefill + Decode）
- **需要 vLLM 0.31.0** — 本机是 0.28.0
- **需要 KV connector 配置**: MooncakeConnector, tcp 协议, kv_consumer 角色
- 本机环境不具备以上任何条件

#### 修复方案

修复 scheduler 中的请求生命周期管理：
1. 当 KV transfer 超时/取消时，正确清理 scheduler 中的请求状态
2. 确保 `self.requests` 字典在 cleanup 后与实际请求集合一致
3. 处理 Decode 不可用期间的请求队列状态

#### 修复难度评估

- **难度**: 高
- **本机可行性**: 不可行 — 缺少所有必需硬件和软件依赖
- **建议**: 跳过，等具备 Mooncake 环境的团队处理

---

## 建议行动

### 优先做: #60703 (GLM-4.7 解析器)

- 难度最低，纯 CPU，不需要 GPU
- 代码改动小（添加一个状态机转换）
- 有现有测试基础设施
- 预计 2-4 小时可完成修复+测试

### 其次: #60744 (pause_generation 报错)

- 纯 CPU 可验证
- PR #60764 已提交，可以学习修复思路
- 建议先 review PR 代码，再在本机源码上验证

### 可选: #60608 (Exllama GPTQ)

- 有 GPU 可以验证
- 但需要构造合成权重、可能需要编译 CUDA 扩展
- Python 层修复（can_implement 添加检查）可以先做

### 暂不建议: #60499 和 #60755

- #60499: 性能优化，工作量大，sm_86 与 sm_89 结果可能不同
- #60755: 缺少 Mooncake 硬件和多节点环境

---

## 关键源码位置速查

```
Issue #60703:
  vllm-main/vllm/parser/glm47_moe.py
  vllm-main/vllm/tool_parsers/glm47_moe_tool_parser.py
  vllm-main/tests/parser/engine/test_glm47_moe.py

Issue #60744:
  vllm-main/vllm/v1/core/sched/scheduler.py  (line ~2839-2869)
  vllm-main/vllm/v1/engine/core.py           (line ~859-932)
  vllm-main/vllm/entrypoints/rl/online/api_router.py

Issue #60608:
  vllm-main/vllm/model_executor/kernels/linear/mixed_precision/exllama.py
  vllm-main/csrc/libtorch_stable/quantization/gptq/q_gemm.cu

Issue #60499:
  vllm-main/vllm/compilation/passes/fusion/rms_quant_fusion.py
  vllm-main/vllm/kernels/helion/ops/rms_norm_dynamic_per_token_quant.py

Issue #60755:
  vllm-main/vllm/v1/core/sched/scheduler.py  (line ~3148)
  (需要 Mooncake 环境)
```
