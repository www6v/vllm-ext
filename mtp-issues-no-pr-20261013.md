# MTP-Related Issues Without PRs

**Repository:** vllm-project/vllm
**Last Updated:** 2026-10-13
**Total Issues:** 22

These are open MTP (Multi-Token Prediction) related issues updated after 2026-10-01 that do not have any submitted PRs.

---

## #58422: [Bug]: TP=1 silent engine wedge on Qwen3.8 GDN-hybrid MoE (MTP4, NVFP4-KV, FlashInfer, SM120) — frozen generation counter, running>0 waiting=0, health 200, no self-recovery on v0.27.2rc1.dev77

- **URL:** https://github.com/vllm-project/vllm/issues/58422
- **Last Updated:** 2026-10-13
- **Labels:** quantization

## #60494: [Bug]: GLM-5.2-FP8 + prefix caching + FP8 KV cache returns empty content (gibberish reasoning) on repeated identical prompts

- **URL:** https://github.com/vllm-project/vllm/issues/60494
- **Last Updated:** 2026-10-13
- **Labels:** bug, rocm, tool-calling, quantization, kv-cache-manager, glm

## #61300: [Bug]: Intel Arc Pro B70 XPU checkpoint-loading hang before readiness with Xe BCS faults on v0.31.0

- **URL:** https://github.com/vllm-project/vllm/issues/61300
- **Last Updated:** 2026-10-13
- **Labels:** intel-gpu, quantization

## #58688: [Bug][ROCm]: Qwen3.8-Flash-Next-FP8 cannot load on ROCm, the AMD Qwen4ExpNGramEmbedding has no FP8 weight_scale

- **URL:** https://github.com/vllm-project/vllm/issues/58688
- **Last Updated:** 2026-10-12
- **Labels:** bug, rocm

## #61233: [MRV2][spec-decode] Full-graph replay with >=2 concurrent sequences kills the engine on SM75; logits also diverge from PIECEWISE

- **URL:** https://github.com/vllm-project/vllm/issues/61233
- **Last Updated:** 2026-10-12
- **Labels:** None

## #61157: [Bug] GLM hybrid-SSM + MTP: deep-ctx agent session sharply degrades (fabricated tool args, self-amnesia) exactly when a concurrent request arrives — persists after #58454/#58368

- **URL:** https://github.com/vllm-project/vllm/issues/61157
- **Last Updated:** 2026-10-12
- **Labels:** tool-calling, quantization, glm

## #57429: [Bug][ROCm] Engine stops making progress inside an aiter fused MoE step on gfx942, then dies with no error (GLM-5.3, TP8, v0.29.0)

- **URL:** https://github.com/vllm-project/vllm/issues/57429
- **Last Updated:** 2026-10-12
- **Labels:** rocm, glm

## #61069: [Bug]: GB10 (sm_121), NVFP4 + Marlin MoE + MTP: engine dies daily ~2k sequential short requests into a batch job (Xid 13+31 write to VA 0x0; Xid 13+43 at a fixed address on all SMs)

- **URL:** https://github.com/vllm-project/vllm/issues/61069
- **Last Updated:** 2026-10-11
- **Labels:** quantization

## #53912: [Bug]: prefix caching + MTP still corrupts output on hybrid Mamba/GDN models in v0.28.0 (#43559 closed but unfixed)

- **URL:** https://github.com/vllm-project/vllm/issues/53912
- **Last Updated:** 2026-10-11
- **Labels:** kv-cache-manager

## #56868: [Bug]: GLM-5.3-Flash long-decode degeneration after accumulated reasoning decode

- **URL:** https://github.com/vllm-project/vllm/issues/56868
- **Last Updated:** 2026-10-10
- **Labels:** bug, tool-calling, quantization, glm

## #60119: [Bug]: GLM-5.3-Flash NVFP4 on B300: CUDA illegal access persists with MTP disabled; blocking/eager reports persistent_topk

- **URL:** https://github.com/vllm-project/vllm/issues/60119
- **Last Updated:** 2026-10-09
- **Labels:** glm

## #57230: [ROCm][AMD] GLM5.2/5.3 Performance Optimization on gfx950 / MI355X

- **URL:** https://github.com/vllm-project/vllm/issues/57230
- **Last Updated:** 2026-10-09
- **Labels:** feature request, rocm, quantization

## #53181: [Bug] Xgrammar "Failed to advance FSM" still fires after #52805 (0.27.2rc1.dev256+geac636a7f) — follow-up to #52852

- **URL:** https://github.com/vllm-project/vllm/issues/53181
- **Last Updated:** 2026-10-07
- **Labels:** structured-output

## #59724: [Bug][SM120] MTP speculative decoding acceptance rate drops to 0% on nightly with native FLASHINFER_MLA_SPARSE_SM120 backend (GLM-5.3-Flash)

- **URL:** https://github.com/vllm-project/vllm/issues/59724
- **Last Updated:** 2026-10-07
- **Labels:** speculative-decoding, quantization, glm

## #57755: Title

- **URL:** https://github.com/vllm-project/vllm/issues/57755
- **Last Updated:** 2026-10-06
- **Labels:** None

## #59926: [Bug] DeepSeek-V4-Flash tool-call format failures associated with prefix-cache reuse; worst for short cached head + long uncached suffix; `cache_salt` reduces failures but is confounded with hit shape (v0.27.1, fp8 KV, MTP)

- **URL:** https://github.com/vllm-project/vllm/issues/59926
- **Last Updated:** 2026-10-04
- **Labels:** rocm, tool-calling, deepseek, DSv4

## #56896: Port PR #55738 (GLM-5.3-Flash NoPE MLA prefill) to older vLLM trees + SM120 note

- **URL:** https://github.com/vllm-project/vllm/issues/56896
- **Last Updated:** 2026-10-04
- **Labels:** glm

## #58391: [Bug]: DeepSeek-V4.1-Flash teacher-forced logprob shift on tool/structured text between nightlies dc36fcce and cd10ed6f (SM103, fp8_ds_mla)

- **URL:** https://github.com/vllm-project/vllm/issues/58391
- **Last Updated:** 2026-10-03
- **Labels:** tool-calling, deepseek, DSv4.1

## #47047: Gemma-4-26B-A4B-NVFP4 + MTP4 is slower on v0.23.0 than v0.21.0 on RTX PRO 6000 Blackwell

- **URL:** https://github.com/vllm-project/vllm/issues/47047
- **Last Updated:** 2026-10-03
- **Labels:** stale

## #59606: [Bug]: --moe-backend b12x on SM121 (DGX Spark): illegal memory access in CUDA-graph capture with padded rows, worker crash on prefill

- **URL:** https://github.com/vllm-project/vllm/issues/59606
- **Last Updated:** 2026-10-02
- **Labels:** None

## #45531: [Bug]: minimax M3MXFP8 with mtp can not start success

- **URL:** https://github.com/vllm-project/vllm/issues/45531
- **Last Updated:** 2026-10-02
- **Labels:** bug, stale

## #46679: [Bug]: GLM5.2 RuntimeError: The expanded size of the tensor (10210) must match the existing size (16384) at non-singleton dimension 1.  Target sizes: [12, 10210].  Tensor sizes: [12, 16384]

- **URL:** https://github.com/vllm-project/vllm/issues/46679
- **Last Updated:** 2026-10-02
- **Labels:** bug, stale
