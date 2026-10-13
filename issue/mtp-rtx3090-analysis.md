# RTX 3090 MTP Issue 可行性分析

**GPU**: NVIDIA RTX 3090 (SM86, Ampere, 24GB GDDR6X, PCIe)
**分析日期**: 2026-10-13

---

## 一、总体结论

22 个 issue 中，经过逐一分析，**5 个** 可以在 RTX 3090 上验证/修复，**3 个** 在特定条件下可以尝试，其余 14 个因硬件限制无法复现。

---

## 二、RTX 3090 的硬件约束

| 维度 | RTX 3090 | 限制影响 |
|------|----------|---------|
| 架构 | SM86 (Ampere) | 不支持 NVFP4、SM89+ FP8、SM120+ 特性 |
| VRAM | 24GB | 无法加载 27B+ 全精度模型，只能用 TP 或小模型 |
| 互联 | PCIe (无 NVLink) | TP=2 带宽受限，TP≥4 不可行 |
| 平台 | NVIDIA CUDA | 所有 ROCm/Intel XPU 问题均不适用 |

---

## 三、✅ 可以验证/修复的 Issue（5 个）

### 🥇 #57755 — PP × Spec Decode: `sampled_token_ids` 广播契约在 spec decode 下失效

**推荐度: ★★★★★（最高优先级）**

- **核心问题**: Pipeline Parallelism 下，`_pp_broadcast_prev_sampled_token_ids` 断言 `shape[-1]==1`，但 spec decode 输出 `[num_reqs, 1+k]`；workaround（只广播 `[:,:1]`）在 async scheduling 下会导致输出静默损坏
- **原报告人硬件**: **RTX 3090 24GB + RTX A4000 16GB**，TP=1 和 PP=2，SM86 — **与你的硬件完全匹配**
- **复现条件**:
  - Qwen3.5-27B hybrid-GDN + MTP head
  - PP=2（需要两张卡）
  - `--kv-cache-dtype int4_per_token_head`
  - 量化后模型可以在 2×3090 上跑
- **修复可行性**: 高 — 问题定位清晰，代码路径明确（`scheduler.py` / `gpu_model_runner.py`），已有详细代码分析
- **需要你**: 2 张 GPU（3090 + 任意 16GB+ NVIDIA 卡），或纯代码修复 + 单卡验证

---

### 🥈 #53181 — XGrammar "Failed to advance FSM" 在 spec decode + structured output 下仍然触发

**推荐度: ★★★★☆**

- **核心问题**: `backend_xgrammar.py:168` 处 FSM advance 失败，token 271 总是出问题；#52805 的修复不覆盖非终止 token 的 advance 路径
- **硬件要求**: **任意 NVIDIA GPU**（SM86 完全兼容），纯软件 bug
- **复现条件**:
  - Qwen3.x 模型 + MTP spec decode
  - `--tool-call-parser` + xgrammar 结构化输出
  - 中等并发，长会话
- **修复可行性**: 高 — 问题定位到具体代码行，失败模式清晰（特定 token id 触发）
- **模型选择**: 可以用一个更小的 Qwen 模型（如 Qwen3-7B/14B 带 MTP）来在 3090 上复现
- **需要你**: 单卡 3090，小模型即可

---

### 🥉 #61233 — Full-graph spec decode: ≥2 并发序列杀死引擎 (SM75/SM86)

**推荐度: ★★★★☆**

- **核心问题**: CUDA Graph FULL 模式 + MTP 投机解码 + ≥2 并发序列 → 引擎崩溃；且单流时 logits 与 PIECEWISE 模式存在位级差异
- **原报告人硬件**: 2×RTX 2080 Ti (SM75)，但报告人明确说 "非仅 SM75 特有，是 MRV2 通用机制"
- **RTX 3090 兼容性**: SM86 与 SM75 同属 pre-Ada/pre-Hopper，CUDA Graph 行为一致，**高度兼容**
- **复现条件**:
  - 2 张 GPU (TP=2)，或单卡 + 小模型
  - MTP k=4
  - CUDA graph FULL 模式
  - ≥2 并发请求
- **修复可行性**: 高 — 报告人已经做了详细的 A/B 排查，定位到 `CUDAGraphManager.dispatch` 并发≥2时的图档选择逻辑
- **需要你**: 2 张 GPU 最佳；单卡 3090 配合小模型也可以尝试

---

### #53912 — Prefix caching + MTP 在 hybrid Mamba/GDN 模型上仍然损坏输出

**推荐度: ★★★★☆**

- **核心问题**: `--enable-prefix-caching` + MTP spec decode → hybrid (GDN/Mamba) 模型输出退化为重复标点或空输出
- **硬件**: 原报告在 H100 上，但 **问题机制是通用的**（host/device 异步 race condition），SM86 完全兼容
- **重要进展**: 评论区确认 **#50729 修复已解决此问题**（`v0.28.1rc0` 或 main 包含），但如果你用的版本不含此修复，可以在 3090 上验证修复是否有效
- **复现条件**:
  - Qwen3.5-27B FP8 + MTP head（需要量化到 24GB 内）
  - prefix caching + async scheduling 开启
  - 约 288 个请求的批量测试
- **修复可行性**: 中高 — 如果需要进一步修复，根因是 `MambaManager` 的 state copy race

---

### #61157 — GLM hybrid-SSM + MTP: 并发请求到达时 agent session 突然退化

**推荐度: ★★★☆☆**

- **核心问题**: 长 agent 会话（~225K token）正常运行 97 轮后，第二个并发请求到达的瞬间，输出突然退化为空参数/虚构路径
- **硬件**: 原报告 4×A800 (TP4)，但 **bug 在 GLM 的 MTP × hybrid-SSM state 机制**（`MambaManager` / `kpool_compress.py`），不是硬件相关的
- **RTX 3090 可行性**:
  - GLM-5.3-Flash-AWQ 量化后可能在单卡 24GB 上运行（需要 TP=1 + aggressive quantization）
  - 但需要极长上下文（225K+），24GB 很难容纳
  - 可以用更短的会话 + 更小的 GLM 模型来缩小复现
- **修复可行性**: 中 — 问题定位到 `MambaManager` 的 checkpoint/drop 机制，并发触发

---

## 四、🔶 条件允许时可尝试的 Issue（3 个）

### #58422 — TP=1 silent engine wedge on Qwen3.8 GDN-hybrid MoE (MTP4)

- **原硬件**: RTX PRO 6000 Blackwell (SM120)
- **问题**: ~10 个并发请求导致引擎静默卡死（GPU util 100% 但 power 为 idle 水平）
- **为何只是条件允许**: 触发机制（并发 burst）可能与 SM 版本无关，但 NVFP4-KV 和 FlashInfer SM120 路径在 3090 上不存在。**如果改用 FP16/FP8 KV + 标准 attention backend 可能复现核心 bug**

### #59926 — DeepSeek-V4-Flash tool-call 格式错误与 prefix-cache 复用相关

- **原硬件**: 8×H100 TP=8
- **问题**: prefix cache 命中形状影响 tool-call 格式化失败率
- **为何只是条件允许**: DeepSeek-V4-Flash 太大，但 **prefix cache + MTP + tool-call 的交互** 是通用机制，可用更小的 DeepSeek 模型 + 2 张 3090 尝试

### #56868 — GLM-5.3-Flash 长 decode 退化

- **原硬件**: B300, TP=1
- **问题**: 长 agent 会话（76+ 轮）后模型输出退化为 `= = = =` 或 `.0.0.0.0`
- **为何只是条件允许**: 评论指出是 hybrid 架构的 recurrent state 饱和问题，**机制是通用的**，但需要长上下文，24GB 可能不够

---

## 五、❌ 无法在 RTX 3090 上复现的 Issue（14 个）

| Issue | 原因 |
|-------|------|
| #58688 | ROCm (MI325X/MI300X) 专属 |
| #60494 | ROCm (MI300X) + AITER 后端专属 |
| #57429 | ROCm (MI325X/gfx942) 专属 |
| #57230 | ROCm (MI355X/gfx950) 性能优化 |
| #61300 | Intel Arc Pro B70 XPU 专属 |
| #61069 | GB10 (SM121) + NVFP4 + Marlin MoE 专属 |
| #59606 | SM121 (DGX Spark) + b12x NVFP4 MoE 专属 |
| #60119 | B300 (SM103) + NVFP4 专属 |
| #47047 | RTX PRO 6000 Blackwell (SM120) + NVFP4 性能回归 |
| #58391 | DGX Station GB300 (SM103) 专属 |
| #59724 | SM120 FLASHINFER_MLA_SPARSE 专属 |
| #45531 | MiniMax-M3 需要 TP=8，模型过大 |
| #46679 | GLM5.2-FP8 需要 TP=16，模型过大 |
| #56896 | GLM-5.3-Flash 移植问题，目标 SM120 |

---

## 六、推荐行动路径

### 快速收益（1-2 周）

1. **#53181 (XGrammar FSM)** — 单卡 3090，用 Qwen3-7B/14B + MTP + tool-call，复现门槛最低
   - 找到 `backend_xgrammar.py:168` 的 unfixed failure mode
   - 修复后贡献 PR

2. **#57755 (PP × Spec Decode)** — 如果你有第二张 GPU，这是最有价值的修复
   - 原报告人已经用 3090 验证过，代码路径清晰
   - 涉及 scheduler + broadcast 的 spec-decode 宽度适配

### 中等投入（2-4 周）

3. **#61233 (Full-graph + MTP 并发)** — 需要 2 张 GPU 或单卡 + 小模型
   - 修复 CUDA Graph dispatch 的并发安全
   - 同时研究 logits 位级差异的数值稳定性问题

4. **#53912 (Prefix cache + MTP corruption)** — 验证 #50729 修复是否完全覆盖
   - 如果有 2×3090，可以用 Qwen3.5-27B FP8 复现
   - 如果修复不完整，可以进一步排查 MambaManager race

### 长期探索

5. **#61157 (并发触发 agent 退化)** — 需要找到在 24GB 内跑 GLM hybrid-SSM + 长上下文的方法

---

## 七、模型选择建议（适配 24GB VRAM）

| 目标 Issue | 推荐模型 | 估计 VRAM | 备注 |
|-----------|---------|----------|------|
| #53181 | Qwen3-7B/14B + MTP | ~12-18GB | 小模型即可触发 xgrammar bug |
| #57755 | Qwen3.5-27B FP8 + MTP | ~20-24GB × 2 | 需要 2 张卡做 PP |
| #61233 | Qwen3.5-7B + MTP | ~12GB × 2 | 小模型 + TP=2 |
| #53912 | Qwen3.5-27B FP8 + MTP | ~20-24GB | 单卡极限 |
| #61157 | GLM-5.3-Flash-AWQ | ~24GB | 极限压缩，上下文受限 |
