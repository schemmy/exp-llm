# Progress Tracker

**Started**: 2026-08-30

Convention: one line per week. Mark ✅ done, 🟡 partial, ❌ skipped. Add a link to any artifact (commit, blog draft, notebook) so future-you can jump back.

---

## Phase 1 — Inference foundations

| Wk | Dates | Task summary | Status | Artifact / notes |
|----|-------|--------------|--------|------------------|
| 1  | 08-30 → 09-05 | GPU dev env on Modal or RunPod | ✅ | HF baseline ~31 tok/s (Qwen2.5-0.5B, A100, fp16) · [repo](https://github.com/schemmy/exp-llm) |
| 2  | 09-06 → 09-12 | HF baseline for 7B model | ✅ | Qwen2.5-7B ~40 tok/s (A100, fp16) — 7B > 0.5B due to arithmetic intensity |
| 3  | 09-13 → 09-19 | vLLM install + baseline benchmark | ✅ | vLLM single-req ~80 tok/s (2x HF); batch(8) 748 tok/s (~18x HF serial) |
| 4  | 09-20 → 09-26 | vLLM + PagedAttention papers; batch sweep | ✅ | Linear throughput scaling: batch 32 → 2,408 tok/s, latency barely changes |
| 5  | 09-27 → 10-03 | Continuous batching experiments | ✅ | CB P50 latency 2.4s vs static 6.4s; TTFT 49ms; throughput similar (~710 tok/s) |
| 6  | 10-04 → 10-10 | Continuous batching cont'd | ✅ | Load pressure sweep：A100 未饱和，TTFT 33→39ms (5→50 req/s)，throughput 763→3273 tok/s；短 prompt 下 prefill 太快，需长 prompt 才能看到 TTFT 崩溃 |
| 7  | 10-11 → 10-17 | Prefix caching toggle | ✅ | TTFT P50: 829ms→156ms (5.3x)；throughput 835→1412 tok/s；~500-token shared RAG prefix，32 concurrent requests |
| 8  | 10-18 → 10-24 | Speculative decoding toggle | ✅ | acceptance rate 是主变量：copy 任务 2.78x (b=1) / 2.38x (b=16)，novel 任务 1.13x / **0.85x 净亏**。draft model 在 vLLM V1 不受支持 |
| 9  | 10-25 → 10-31 | Tensor parallelism: TP=1 vs 2 vs 4 | ✅ | TP=2 1.55x / TP=4 2.16x（效率 75% / 54%，跨 batch 恒定）；**每卡吞吐反而下降**——TP 买延迟不买性价比；KV 容量 1.05M→5.14M tokens |
| 10 | 11-01 → 11-07 | MoE inference + expert parallelism | ✅ | **MoE 优势随 batch 蒸发**：batch=1 达小稠密速度 67%，batch=32 只剩 6%——token 被打散到 60 个专家，`M=32` 碎成 60 个 `M≈2`。**多卡在 batch=32 是负收益**：TP=2 0.84x、+EP 0.56x（对比稠密模型 TP=2 是 1.53x）。GEMM 已经太小时，加并行只会更糟 |
| 11 | 11-08 → 11-14 | Roofline ridge point + CUDA graph + MFU/MBU + INT8/FP8 | ✅ | **CUDA graph 2.69x@b1**（原估算 1.2x 太保守，漏算 PyTorch eager 分发开销）；脊点 M≈153 落在实测 128(64.6%)-256(43.1%) 之间；MFU 封顶 ~51-52%。**FP8 量化符号反转**：带宽区 1.5x，算力区 **0.84x 净亏**，反转位置精确对上脊点。**INT8(`int8_per_channel_weight_only`) 几乎零加速**（全程 ~1.0x，`bitsandbytes` 在 vLLM v0.28 已不受支持）——同样是"减字节"但没有高效 kernel 路径承接，字节数减少≠速度提升 |
| 12 | 11-15 → 11-21 | INT4 quantization + tradeoff table | ✅ | **悬念解开**：INT4 via Marlin batch=1 达 **2.20-2.23x**（比 FP8 的 1.56x 还快）——Wk11 INT8 零加速确认是 kernel 问题不是格式问题。权重 5.19 GiB (36.6%，不是理论 25%)。算力区 INT4 反而比 FP8 亏得少（0.93-0.97x vs 0.84x，预测错了）。AWQ≈GPTQ 速度（同 kernel），质量相似度低但样本检查是措辞不同不是退化 |
| 13 | 11-22 → 11-28 | Blog #1 draft + repo README | ☐ | |
| 14 | 11-29 → 12-05 | **Publish Blog #1** | ☐ | |

**P1 retro** (fill after Wk 14): what worked / what didn't / adjust for P2?

---

## Phase 2 — Distributed training (Wk 15-26)

To be filled in at end of Wk 14.

---

## Phase 3 — GPU serving + platform (Wk 27-38)

To be filled in at end of Wk 26.

---

## Phase 4 — CUDA + systems depth (Wk 39-52)

To be filled in at end of Wk 38. Reevaluate whether to stick with CUDA vs. swap to OSS-contribution track.

---

## Global log

*(Note anything cross-cutting — realizations, direction changes, external events)*

- **2026-08-30**: Plan created. Not job-search-urgent; long-term skill build. Decided to focus on inference + platform tracks over pure model-training MLE.
- **2026-08-31**: Wk 1 complete. Modal set up, HF baseline 31 tok/s on A100 (Qwen2.5-0.5B). Kept everything in one repo (no separate vllm-benchmarks). PagedAttention orientation done.
- **2026-09-06**: Wk 2 complete. 7B baseline ~40 tok/s (Qwen2.5-7B-Instruct, A100, fp16). Surprising finding: 7B faster than 0.5B due to better GPU arithmetic intensity.
- **2026-09-06**: Wk 3 complete. vLLM running on Modal (needed nvidia/cuda devel image for nvcc). Single-req: 2x HF. Batch(8): ~18x HF serial throughput. Core lesson: continuous batching is where vLLM wins.
- **2026-09-04**: **Plan change — P1 extended to 14 weeks.** Audit found the original plan never left a single GPU for *inference*: P1 was single-A100, P2's multi-GPU work is training (DDP/FSDP/ZeRO), P3 orchestrates replicas of a single-GPU-sized model, P4 is kernels. "The model doesn't fit on one GPU" — tensor / pipeline / expert parallelism — was absent end to end, despite being table stakes at every target company. Inserted Wk 9 (tensor parallelism) and Wk 10 (MoE + expert parallelism); quantization → Wk 11-12, Blog #1 → Wk 13-14. All later phases shift +2 weeks; P4 absorbs the compression (the plan already flags it as the swappable phase).
