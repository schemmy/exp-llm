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
| **P1** | 1–14 | Inference foundations | Blog #1 + repo: *Benchmarking vLLM: single GPU to multi-GPU* |
| **P2** | 15–26 | Distributed training | Blog #2 + repo: *Fine-tuning 7B with LoRA on multi-GPU* |
| **P3** | 27–38 | GPU serving + platform | Blog #3 + repo: *Autoscaling LLM inference in Kubernetes* + OSS PR |
| **P4** | 39–52 | CUDA + systems depth | Blog #4 + repo: *Understanding FlashAttention by reimplementing softmax* |

Each phase caps with a public artifact. If a phase runs long, extend rather than skip the artifact — a public artifact you can point to is the whole point.

---

## Phase 1 — Inference foundations (Wk 1–14)

**Goal**: from zero to serving a 7B model on GPU with vLLM, deeply understanding continuous batching and PagedAttention — then past the point where the model stops fitting on one GPU.

| Wk | Task | Output |
|---|---|---|
| 1 | GPU dev env on Modal ($30 free) or RunPod (A100 @ $1.50/hr). Hello-world job. | Reproducible dev container |
| 2 | HuggingFace transformers baseline: run Llama 3 8B or Qwen 2.5-7B inference | Baseline throughput / latency numbers |
| 3 | Install vLLM, run same model | vLLM vs HF baseline benchmark |
| 4 | Read vLLM + PagedAttention papers; batch size sweep | Batch-size × throughput plot |
| 5–6 | Continuous batching under different arrival patterns (Poisson, burst) | 3–4 figures |
| 7–8 | Toggle prefix caching + speculative decoding | Ablation table |
| 9 | **Tensor parallelism**: 32B model on 2×/4×A100, `tensor_parallel_size` sweep. Where does TP stop scaling, and why (NVLink vs PCIe, all-reduce cost)? | TP scaling curve + comms-overhead analysis |
| 10 | **MoE + expert parallelism**: Qwen1.5-MoE-A2.7B (14.3B total / 2.7B active) vs a dense model at matched active params. Memory vs speed tradeoff. | MoE-vs-dense table; EP notes |
| 11 | **Roofline ridge point**: batch sweep to 512, find where throughput leaves linear (predicted M≈153 on A100) + INT8/FP8 quantization | Batch × throughput curve with measured ridge; fp16 vs INT8 table |
| 12 | INT4 quantization + accuracy/latency tradeoff table | Quantization tradeoff table |
| 13–14 | Write blog + repo README; publish to GitHub Pages | **Blog #1 shipped** |

**Why Wk 9–10 were added** (2026-09-04): the original plan never left a single GPU for *inference*. P2's multi-GPU content is training (DDP/FSDP/ZeRO); P3 orchestrates replicas of a model that already fits on one card; P4 is kernels. So "the model doesn't fit on one GPU" — the defining constraint of frontier inference — appeared nowhere in 52 weeks. Every target company (Anthropic Fleet, Together, Fireworks) serves models where TP is table stakes, and it is the missing bridge between P1's single-GPU tuning and P3's production serving.

**Cost**: ~$120–200 of GPU time across 14 weeks (multi-GPU weeks run ~$5/hr for 2×A100).

**Success criteria**: blog is reproducible from the README; someone can `git clone && bash reproduce.sh` and get within 5% of your numbers.

---

## Phase 2 — Distributed training (Wk 15–26)

**Goal**: DDP → FSDP → LoRA fine-tune a 7B model, understand ZeRO stages.

- Wk 15–16: PyTorch DDP on 2 GPUs, small model
- Wk 17–20: FSDP for 7B model; ZeRO-1 vs ZeRO-2 vs ZeRO-3 tradeoffs
- Wk 21–24: LoRA / QLoRA fine-tune on real task (code repair, SQL generation, or a domain-specific classification set)
- Wk 25–26: Blog #2 + repo

**Anchor course**: Stanford **CS336 — Language Modeling from Scratch** (recorded lectures + assignments). Follow it on their cadence.

**Cost**: ~$200–400 GPU. Rent multi-A100 as needed.

---

## Phase 3 — GPU serving + platform (Wk 27–38)

**Goal**: Take the Wk-14 vLLM setup and productionize it — K8s, autoscaling, observability. **This is your leverage phase**: your DBR autoscaler / capacity knowledge translates directly here, and Blog #3 becomes the resume headline.

- Wk 27–30: Local K8s via kind/minikube; deploy vLLM as a K8s service
- Wk 31–34: Small GKE or EKS cluster ($50-100/mo); Prometheus + Grafana; observe GPU util, TTFT, TBT
- Wk 35–36: Find + fix an issue in vLLM, SkyPilot, or Ray Serve (aim for merged PR)
- Wk 37–38: Blog #3 — "Autoscaling GPU inference: what's actually hard"

**Builds on Wk 9–10**: a multi-GPU deployment is the unit K8s schedules here — a TP=4 replica needs 4 co-located GPUs with NVLink, which makes bin-packing and autoscaling meaningfully harder than the single-GPU case. That constraint is exactly the DBR capacity story, transplanted.

**Why this phase matters most for you**: the story "I did autoscaling for CPU-based serverless compute at Databricks, then re-did it for GPU inference on K8s" is exactly the pitch for Anthropic Fleet / Together Capacity Engineering.

---

## Phase 4 — CUDA + systems depth (Wk 39–52)

**Note on prior work**: you already have a solid CUDA foundation via `~/projects/fun_cuda_kernals` (GEMM steps 01–05 done: naive → SMEM tiling → register tiling → tensor cores; plus FA1 teaching version). Step 03 matched cuBLAS SGEMM (3.1 vs 3.4 TFLOP/s on T4). P4 is *extending* this repo, not starting from scratch.

**Goal**: kernels that are directly relevant to LLM inference (not just GEMM ladder), tested on A100 not just T4, and one written up as a blog.

- Wk 39–42: Resume `fun_cuda_kernals` — pick from the paused-at options: multi-warp FlashAttention v1 (fix occupancy), FA2 outer-loop swap, or PagedAttention (vLLM's kernel). Test on A100 not T4.
- Wk 43–46: One more inference-relevant kernel (causal masking, GQA attention, or KV cache eviction)
- Wk 47–49: Profile with nsys / ncu; understand where cuBLAS wins vs your kernel
- Wk 50–52: Blog #4 — "Algorithm vs implementation: why my FlashAttention loses to cuBLAS at N=4096 (and how I fixed it)"

*P4 absorbs the 2-week shift from the P1 extension (16 weeks → 14). It is the designated flex phase — see the swap options below.*

**Alternative if the kernel work gets stale**: swap P4 for either
- (a) A second serious OSS contribution to vLLM (kernel-adjacent PR), OR
- (b) A speculative decoding side project (implement draft-target pipeline for 7B), OR
- (c) SGLang / Ray Serve deep-dive with a second serving-platform blog

Any of these produces strong MLE signal. **Don't do CUDA for CUDA's sake** — you already have enough of a foundation that "more CUDA" has diminishing returns vs. things closer to a real serving stack.

---

## Decision points

- **End of P1 (Wk 14)**: If you're loving inference internals → P2 as planned. If K8s / systems is calling you louder → swap P2 and P3 order.
- **End of P2 (Wk 26)**: Check the job market. If you feel ready, start test-applying to 2-3 Anthropic Fleet / Together roles as calibration (don't commit).
- **End of P3 (Wk 38)**: Portfolio should be strong enough to apply seriously. Decide: keep going to P4, or start interviewing.

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
