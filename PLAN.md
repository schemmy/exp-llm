# MLE Transition Plan — 12 Months

**Start**: 2026-08-30
**Target completion**: 2027-08-30
**Direction**: DS → **ML Platform / Infra MLE + Inference / Training Engineer**
**Cadence**: 6–10 hrs/week (evenings + weekends), part-time alongside Databricks DBR job
**Non-goal**: applied ML / model-training MLE (already halfway there via Migration Prediction Model)

---

## End state (Aug 2027)

- **4 flagship GitHub repos** with reproducible experiments
- **4 technical blog posts** on personal site (chenxin-ma.github.io)
- **≥1 merged OSS PR** on vLLM, SkyPilot, or Ray (in addition to the existing SkyPilot autoscaler PR)
- **Portfolio-ready** for Staff/Sr MLE roles at Anthropic Fleet, OpenAI Applied ML, Together, Fireworks, Physical Intelligence

Resume delta: 3-4 concrete "MLE / inference / systems" bullets on top of the existing DS profile.

---

## Phase overview

| Phase | Weeks | Theme | Flagship deliverable |
|---|---|---|---|
| **P1** | 1–12 | Inference foundations | Blog #1 + repo: *Benchmarking vLLM on a single A100* |
| **P2** | 13–24 | Distributed training | Blog #2 + repo: *Fine-tuning 7B with LoRA on multi-GPU* |
| **P3** | 25–36 | GPU serving + platform | Blog #3 + repo: *Autoscaling LLM inference in Kubernetes* + OSS PR |
| **P4** | 37–52 | CUDA + systems depth | Blog #4 + repo: *Understanding FlashAttention by reimplementing softmax* |

Each phase caps with a public artifact. If a phase runs long, extend rather than skip the artifact — a public artifact you can point to is the whole point.

---

## Phase 1 — Inference foundations (Wk 1–12)

**Goal**: from zero to serving a 7B model on GPU with vLLM, deeply understanding continuous batching and PagedAttention.

| Wk | Task | Output |
|---|---|---|
| 1 | GPU dev env on Modal ($30 free) or RunPod (A100 @ $1.50/hr). Hello-world job. | Reproducible dev container |
| 2 | HuggingFace transformers baseline: run Llama 3 8B or Qwen 2.5-7B inference | Baseline throughput / latency numbers |
| 3 | Install vLLM, run same model | vLLM vs HF baseline benchmark |
| 4 | Read vLLM + PagedAttention papers; batch size sweep | Batch-size × throughput plot |
| 5–6 | Continuous batching under different arrival patterns (Poisson, burst) | 3–4 figures |
| 7–8 | Toggle prefix caching + speculative decoding | Ablation table |
| 9–10 | Quantization: INT8 / FP8 / INT4 — accuracy vs latency | Quantization tradeoff table |
| 11–12 | Write blog + repo README; publish to GitHub Pages | **Blog #1 shipped** |

**Cost**: ~$80–150 of GPU time across 12 weeks.

**Success criteria**: blog is reproducible from the README; someone can `git clone && bash reproduce.sh` and get within 5% of your numbers.

---

## Phase 2 — Distributed training (Wk 13–24)

**Goal**: DDP → FSDP → LoRA fine-tune a 7B model, understand ZeRO stages.

- Wk 13–14: PyTorch DDP on 2 GPUs, small model
- Wk 15–18: FSDP for 7B model; ZeRO-1 vs ZeRO-2 vs ZeRO-3 tradeoffs
- Wk 19–22: LoRA / QLoRA fine-tune on real task (code repair, SQL generation, or a domain-specific classification set)
- Wk 23–24: Blog #2 + repo

**Anchor course**: Stanford **CS336 — Language Modeling from Scratch** (recorded lectures + assignments). Follow it on their cadence.

**Cost**: ~$200–400 GPU. Rent multi-A100 as needed.

---

## Phase 3 — GPU serving + platform (Wk 25–36)

**Goal**: Take the Wk-12 vLLM setup and productionize it — K8s, autoscaling, observability. **This is your leverage phase**: your DBR autoscaler / capacity knowledge translates directly here, and Blog #3 becomes the resume headline.

- Wk 25–28: Local K8s via kind/minikube; deploy vLLM as a K8s service
- Wk 29–32: Small GKE or EKS cluster ($50-100/mo); Prometheus + Grafana; observe GPU util, TTFT, TBT
- Wk 33–34: Find + fix an issue in vLLM, SkyPilot, or Ray Serve (aim for merged PR)
- Wk 35–36: Blog #3 — "Autoscaling GPU inference: what's actually hard"

**Why this phase matters most for you**: the story "I did autoscaling for CPU-based serverless compute at Databricks, then re-did it for GPU inference on K8s" is exactly the pitch for Anthropic Fleet / Together Capacity Engineering.

---

## Phase 4 — CUDA + systems depth (Wk 37–52)

**Note on prior work**: you already have a solid CUDA foundation via `~/projects/fun_cuda_kernals` (GEMM steps 01–05 done: naive → SMEM tiling → register tiling → tensor cores; plus FA1 teaching version). Step 03 matched cuBLAS SGEMM (3.1 vs 3.4 TFLOP/s on T4). P4 is *extending* this repo, not starting from scratch.

**Goal**: kernels that are directly relevant to LLM inference (not just GEMM ladder), tested on A100 not just T4, and one written up as a blog.

- Wk 37–40: Resume `fun_cuda_kernals` — pick from the paused-at options: multi-warp FlashAttention v1 (fix occupancy), FA2 outer-loop swap, or PagedAttention (vLLM's kernel). Test on A100 not T4.
- Wk 41–44: One more inference-relevant kernel (causal masking, GQA attention, or KV cache eviction)
- Wk 45–48: Profile with nsys / ncu; understand where cuBLAS wins vs your kernel
- Wk 49–52: Blog #4 — "Algorithm vs implementation: why my FlashAttention loses to cuBLAS at N=4096 (and how I fixed it)"

**Alternative if the kernel work gets stale**: swap P4 for either
- (a) A second serious OSS contribution to vLLM (kernel-adjacent PR), OR
- (b) A speculative decoding side project (implement draft-target pipeline for 7B), OR
- (c) SGLang / Ray Serve deep-dive with a second serving-platform blog

Any of these produces strong MLE signal. **Don't do CUDA for CUDA's sake** — you already have enough of a foundation that "more CUDA" has diminishing returns vs. things closer to a real serving stack.

---

## Decision points

- **End of P1 (Wk 12)**: If you're loving inference internals → P2 as planned. If K8s / systems is calling you louder → swap P2 and P3 order.
- **End of P2 (Wk 24)**: Check the job market. If you feel ready, start test-applying to 2-3 Anthropic Fleet / Together roles as calibration (don't commit).
- **End of P3 (Wk 36)**: Portfolio should be strong enough to apply seriously. Decide: keep going to P4, or start interviewing.

---

## Adjacent commitments (throughout)

- **Anthropic Colab agent-loop pattern**: keep it warm (30 min every 2 months) — it's a phone-screen signal you already have
- **SkyPilot PR**: get it merged, iterate on any review feedback
- **Read compute-efficiency papers** at leisure — 1 paper / month, tracked in `resources.md`
- **`fun_cuda_kernals` repo** — occasional maintenance / additional kernels as spare-cycle work, especially if a specific kernel becomes relevant in P1-P3 (e.g., you hit a weird PagedAttention behavior in Wk 4, that's a good excuse to peek at the vLLM kernel source)

---

## What this plan is NOT

- A job-application plan → that's `~/projects/job_hunting/`
- A DS L6 promo plan → that's your Databricks internal doc
- A survey of everything ML → this is 4 focused projects with public artifacts. Depth > breadth.

---

## How to use these files

- `PLAN.md` (this file) — reference for the arc
- `progress.md` — 1 line every week, checkpoint at end of each phase
- `resources.md` — curated links, updated as we go
- `week01.md`, `week02.md`, ... — created as needed with concrete task lists

When we next chat, tell me the week number (e.g. "Wk 3") and I'll pull up context from these files.
