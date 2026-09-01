# Progress Tracker

**Started**: 2026-08-30

Convention: one line per week. Mark ✅ done, 🟡 partial, ❌ skipped. Add a link to any artifact (commit, blog draft, notebook) so future-you can jump back.

---

## Phase 1 — Inference foundations

| Wk | Dates | Task summary | Status | Artifact / notes |
|----|-------|--------------|--------|------------------|
| 1  | 08-30 → 09-05 | GPU dev env on Modal or RunPod | ✅ | HF baseline ~31 tok/s (Qwen2.5-0.5B, A100, fp16) · [repo](https://github.com/schemmy/mle-transition) |
| 2  | 09-06 → 09-12 | HF baseline for 7B model | ✅ | Qwen2.5-7B ~40 tok/s (A100, fp16) — 7B > 0.5B due to arithmetic intensity |
| 3  | 09-13 → 09-19 | vLLM install + baseline benchmark | ☐ | |
| 4  | 09-20 → 09-26 | vLLM + PagedAttention papers; batch sweep | ☐ | |
| 5  | 09-27 → 10-03 | Continuous batching experiments | ☐ | |
| 6  | 10-04 → 10-10 | Continuous batching cont'd | ☐ | |
| 7  | 10-11 → 10-17 | Prefix caching toggle | ☐ | |
| 8  | 10-18 → 10-24 | Speculative decoding toggle | ☐ | |
| 9  | 10-25 → 10-31 | INT8 / FP8 quantization | ☐ | |
| 10 | 11-01 → 11-07 | INT4 quantization + tradeoff table | ☐ | |
| 11 | 11-08 → 11-14 | Blog #1 draft + repo README | ☐ | |
| 12 | 11-15 → 11-21 | **Publish Blog #1** | ☐ | |

**P1 retro** (fill after Wk 12): what worked / what didn't / adjust for P2?

---

## Phase 2 — Distributed training (Wk 13-24)

To be filled in at end of Wk 12.

---

## Phase 3 — GPU serving + platform (Wk 25-36)

To be filled in at end of Wk 24.

---

## Phase 4 — CUDA + systems depth (Wk 37-52)

To be filled in at end of Wk 36. Reevaluate whether to stick with CUDA vs. swap to OSS-contribution track.

---

## Global log

*(Note anything cross-cutting — realizations, direction changes, external events)*

- **2026-08-30**: Plan created. Not job-search-urgent; long-term skill build. Decided to focus on inference + platform tracks over pure model-training MLE.
- **2026-08-31**: Wk 1 complete. Modal set up, HF baseline 31 tok/s on A100 (Qwen2.5-0.5B). Kept everything in one repo (no separate vllm-benchmarks). PagedAttention orientation done.
- **2026-09-06**: Wk 2 complete. 7B baseline ~40 tok/s (Qwen2.5-7B-Instruct, A100, fp16). Surprising finding: 7B faster than 0.5B due to better GPU arithmetic intensity.
