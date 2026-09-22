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
| 13 | 11-22 → 11-28 | Blog #1 draft + repo README | ✅ | Blog #1 pivoted to three interactive pieces (see 2026-09-19 log entry); repo README rewritten to point at them |
| 14 | 11-29 → 12-05 | **Publish Blog #1** | ✅ | Published: [The Journey of One Token](https://chenxin-ma.github.io/viz/p1.html) — landing page over `request_journey.html`, `scaling_topology.html`, `phase1.html` |

**P1 retro** (fill after Wk 14): what worked / what didn't / adjust for P2?

---

## Phase 2 — Distributed training (Wk 15-28)

Goal (from PLAN.md): DDP → FSDP → LoRA → DPO fine-tune a 7B model, understand
ZeRO stages. Anchor: **Stanford CS336** as reference material for concepts;
not followed assignment-by-assignment (P1 pattern: measured experiments on
rented GPUs, not from-scratch homework).

| Wk | Dates | Task summary | Status | Artifact / notes |
|----|-------|--------------|--------|------------------|
| 15-16 | 12-06 → 12-19 | PyTorch DDP baseline + bucket_cap_mb sweep on 2× A100 | ✅ | Scaling efficiency 96.56%→99.96% (batch 4→64) — DDP's comm/backward overlap hides nearly all overhead on GPT-2 124M (comm_fraction 4.41% @ bs16 vs. 1.15% actual loss). Follow-up sweep of `bucket_cap_mb` at 124M and 774M found no measurable effect at either size — in hindsight, largely implied by the overlap finding itself and didn't need its own dedicated week; see the 2026-09-21 retro note below. Real byproduct: a cross-container GPU noise lesson (a 774M scaling-efficiency comparison came out as a physically-impossible 108.54% purely from inter-container variance) |
| 17 | 12-20 → 12-26 | FSDP full-shard on 7B model (Qwen2.5-7B); measure memory vs DDP baseline | ☐ | |
| 18 | 12-27 → 01-02 | ZeRO-1 vs ZeRO-2 vs ZeRO-3 (via FSDP sharding strategies): memory vs throughput tradeoff | ☐ | |
| 19 | 01-03 → 01-09 | Activation checkpointing on/off; CPU offload on/off — orthogonal knobs added to FSDP | ☐ | |
| 20 | 01-10 → 01-16 | Pick the LoRA fine-tune target task (code repair vs SQL-gen vs domain classification); prep dataset | ☐ | |
| 21 | 01-17 → 01-23 | LoRA fine-tune the 7B on the chosen task; measure baseline eval | ☐ | |
| 22 | 01-24 → 01-30 | QLoRA (4-bit base + LoRA adapters): does quantization + LoRA compose? Compare eval and speed | ☐ | |
| 23 | 01-31 → 02-06 | LoRA rank / target-module sweep; find the pareto frontier for this task | ☐ | |
| 24 | 02-07 → 02-13 | Merged-vs-adapter serving; do the LoRA weights + merged model round-trip cleanly under vLLM? | ☐ | |
| 25 | 02-14 → 02-20 | DPO setup: build a preference-pair dataset from the Wk 20-24 task; DPO loss on top of the LoRA adapter | ☐ | |
| 26 | 02-21 → 02-27 | DPO vs SFT-only eval: does preference tuning actually move the metric the pairs were built for? | ☐ | |
| 27 | 02-28 → 03-06 | Blog #2 / interactive draft | ☐ | |
| 28 | 03-07 → 03-13 | **Publish Blog #2** | ☐ | |

**Format**: same as P1 — one `weekNN.md` per week with concrete tasks +
results, one `experiments/wkNN_*/` folder per week, results feed into
`benchmarks/README.md`, one-file-per-commit workflow.

**Open decisions**:
- LoRA target task (Wk 20 decision): PLAN.md lists code repair vs SQL-gen vs
  domain classification. Pick based on which has the cleanest eval harness —
  this task also has to support building a preference-pair dataset for DPO
  at Wk 25, so favor a task where "output A is better than output B" is
  cheap to judge (e.g. a rubric or a test suite), not one needing human
  labeling.
- Blog #2 format: presumed same as Blog #1 (interactive over prose). Confirm
  around Wk 24.

**Added 2026-09-19 — DPO in, full RLHF deferred.** Asked directly whether P2
should include RLHF. Decision: add **DPO only** (Wk 25-26), not full
PPO + reward-model RLHF. Reasoning: DPO reuses the exact fine-tuning
infrastructure this phase already builds (same LoRA setup, just a different
loss over preference pairs) — no reward model, no PPO, no KL-penalty
machinery to stand up. Full RLHF would add 4-5 weeks and pull P2 from 14
weeks to 18-19, on top of P1 already growing from 12 to 14. Direction is
still infra-leaning (Anthropic Fleet / Together / Fireworks-style roles),
where RLHF specifically is not the gap — DPO closes the "have I done any
preference optimization at all" gap at much lower cost.

**Revisit full RLHF (PPO + reward model) if:**
- the target-company list shifts toward roles where alignment/RLHF is
  actually asked about (e.g. OpenAI applied ML, model-behavior-adjacent
  roles) rather than pure infra/serving roles, or
- DPO at Wk 25-26 turns out unsatisfying / raises questions that only a
  full RL loop would answer (e.g. curiosity about reward hacking, KL
  control, on-policy vs off-policy tradeoffs).

If it comes back, it most naturally slots as an added P2.5 block (4-5 weeks)
or folds into P4's swappable slot (P4 already flags itself as OSS-vs-CUDA
swappable — RLHF could compete for that slot too) — not squeezed into the
existing P2 weeks.

**P2 retro** (fill after Wk 28): what worked / what didn't / adjust for P3?

---

## Phase 3 — GPU serving + platform (Wk 29-40)

To be filled in at end of Wk 28.

---

## Phase 4 — CUDA + systems depth (Wk 41-54)

To be filled in at end of Wk 40. Reevaluate whether to stick with CUDA vs. swap to OSS-contribution track (DPO's deferred RLHF sibling is now also a candidate for this swappable slot — see the P2 note above).

---

## Global log

*(Note anything cross-cutting — realizations, direction changes, external events)*

- **2026-08-30**: Plan created. Not job-search-urgent; long-term skill build. Decided to focus on inference + platform tracks over pure model-training MLE.
- **2026-08-31**: Wk 1 complete. Modal set up, HF baseline 31 tok/s on A100 (Qwen2.5-0.5B). Kept everything in one repo (no separate vllm-benchmarks). PagedAttention orientation done.
- **2026-09-06**: Wk 2 complete. 7B baseline ~40 tok/s (Qwen2.5-7B-Instruct, A100, fp16). Surprising finding: 7B faster than 0.5B due to better GPU arithmetic intensity.
- **2026-09-06**: Wk 3 complete. vLLM running on Modal (needed nvidia/cuda devel image for nvcc). Single-req: 2x HF. Batch(8): ~18x HF serial throughput. Core lesson: continuous batching is where vLLM wins.
- **2026-09-04**: **Plan change — P1 extended to 14 weeks.** Audit found the original plan never left a single GPU for *inference*: P1 was single-A100, P2's multi-GPU work is training (DDP/FSDP/ZeRO), P3 orchestrates replicas of a single-GPU-sized model, P4 is kernels. "The model doesn't fit on one GPU" — tensor / pipeline / expert parallelism — was absent end to end, despite being table stakes at every target company. Inserted Wk 9 (tensor parallelism) and Wk 10 (MoE + expert parallelism); quantization → Wk 11-12, Blog #1 → Wk 13-14. All later phases shift +2 weeks; P4 absorbs the compression (the plan already flags it as the swappable phase).
- **2026-09-08 → 09-13**: Built two interactive `viz/*.html` pages instead of a prose Blog #1 draft (the draft in `blog/` was explicitly rejected — too much text). `request_journey.html`: one inference request's journey through 12 stages on a single GPU. `scaling_topology.html`: the same question at >1T-parameter, multi-node scale — how weights, KV cache, and activations move (or don't) across data center → nodes → GPUs → SM. Both bilingual EN/中文, deployed to `chenxin-ma.github.io/viz/`. Design/content details for `scaling_topology.html` in [viz/scaling_topology.md](viz/scaling_topology.md).
- **2026-09-19**: **Phase 1 shipped as Blog #1.** Added `viz/p1.html` — a short bilingual landing page over three interactive pieces: `request_journey.html` (one request on one GPU), `scaling_topology.html` (one request at trillion-parameter scale), and the existing `phase1.html` (the technical cross-section + results table). Replaced the two-link homepage callout on `chenxin-ma.github.io` with a single entry pointing at `p1.html` so readers see a coherent starting point instead of guessing between links. Nav dropdown updated the same way. This is what Wk13-14 became — no long-form prose, and the "one connecting write-up" question is resolved: it's `p1.html`, which is intentionally very short (title, one thesis, three cards).
- **2026-09-19**: **P2 scoped, RLHF deferred.** Filled in the Wk 15-28 week table. Asked directly whether RLHF belongs in P2 — decided DPO only (Wk 25-26), full PPO+reward-model RLHF deferred (see the P2 section's dated note for the full reasoning and the conditions to revisit it).
- **2026-09-20 → 09-21**: **Wk 15-16 complete — merged into one record (2026-09-21 retro: Wk 16 didn't warrant its own dedicated week).** Wk 15 was the real finding: DDP scaling efficiency on 2× A100 (GPT-2 124M, synthetic tokens) came in at 96.56% (batch=4) to 99.96% (batch=64), overturning the week's own prediction (guessed 70-85% at batch=4). Mechanism: DDP overlaps the gradient all-reduce with backward compute, and at this model size that overlap hides nearly all of it (comm_fraction of raw CUDA time was 4.41% at batch=16, actual throughput loss only 1.15% — the gap between those two numbers *is* the overlap). Two bugs hit and fixed in-place: a profiler attribute renamed by a torch version bump (`cuda_time_total` → `device_time_total`), and a scaling-efficiency formula that double-divided by 2 (first run showed ~48-50% efficiency from a math error, not bad data). Wk 16 followed up by sweeping `bucket_cap_mb` at 124M and at GPT-2 Large 774M (~6x the gradient volume) — no measurable effect at either size, which in hindsight was largely implied by Wk 15's own overlap finding and didn't need a full dedicated week + GPU budget to confirm; user feedback (2026-09-21): "合着这两周没啥意义" — fair for Wk 16 specifically, agreed on rereading. Real byproduct kept from Wk 16: a cross-container GPU noise lesson (an attempted 774M scaling-efficiency comparison came out as a physically-impossible 108.54%, because the single-GPU and DDP legs ran in separate Modal containers and inter-container variance, ~8.5% here, exceeded the actual effect being chased — a cross-run throughput *ratio* is only trustworthy when the effect size clearly exceeds that noise floor; a same-run profiler measurement, like Wk 15's comm_fraction, is the reliable way to measure a small effect). **Process change going forward**: before giving a question its own dedicated week + GPU budget, check whether the answer is already strongly implied by a prior week's result — if so, fold it into that week as a quick side-check instead of a headline deliverable. Sets up Wk 17: DDP's overlap held at both sizes tested, so FSDP's justification is memory capacity (774M already used 66.6GB/80GB per rank in DDP), not communication efficiency. Results in `week15.md`, `week16.md`, and `benchmarks/README_training.md`.
