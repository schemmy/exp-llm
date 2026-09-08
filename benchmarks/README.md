# LLM Inference Benchmarks

Reproducible benchmarks for LLM serving on a single GPU.
Comparing HuggingFace transformers baseline vs vLLM with continuous
batching, PagedAttention, prefix caching, and quantization.

**Status**: Week 10 of 14 — MoE and expert parallelism complete.

## MoE vs Dense (Week 10)

Three Qwen1.5 models, one A100-80GB, fp16, 256 output tokens:

| Model | total / active | batch=1 | batch=32 |
|-------|----------------|---------|----------|
| Qwen1.5-1.8B | 1.8B / 1.8B | 314.5 tok/s | 7,312.9 tok/s |
| **Qwen1.5-MoE-A2.7B** | 14.3B / 2.7B | **228.5 tok/s** | **1,829.6 tok/s** |
| Qwen1.5-14B | 14B / 14B | 54.5 tok/s | 1,487.3 tok/s |

Where the MoE lands between its controls (0% = as slow as the 14B dense,
100% = as fast as the 1.8B dense): **67% at batch=1, 6% at batch=32.**

*The MoE's advantage evaporates with batch size, and GEMM shape explains it. At
batch=1 both models issue an M=1 GEMM, so only weight traffic differs and the
MoE's 2.7B active beats the 14B dense by 4.19x. At batch=32 the dense model gets
one M=32 GEMM, while the MoE scatters 32 tokens x top-4 across 60 experts —
roughly 2 tokens each, i.e. sixty M≈2 GEMMs. Batching is the one lever MoE
cannot pull.*

*Serving implication: what matters is per-expert batch. Reaching a dense-equivalent
M=32 per expert would need roughly 480 concurrent requests on this model.*

## Expert Parallelism (Week 10)

Qwen1.5-MoE-A2.7B, 2x A100-80GB:

| Config | batch=1 | batch=32 |
|--------|---------|----------|
| 1 GPU (reference) | 228.5 tok/s | **1,829.6 tok/s** |
| TP=2 | 271.0 tok/s (1.19x) | 1,530.2 tok/s (**0.84x**) |
| TP=2 + expert parallel | 239.8 tok/s (1.05x) | 1,029.3 tok/s (**0.56x**) |

*Adding GPUs makes this MoE slower at batch=32. Contrast the dense Qwen2.5-7B in
Week 9, where TP=2 gave a steady 1.53x: dense GEMMs are large enough to survive
being sliced, while an MoE expert's M≈2 GEMM is not. Slicing an already-tiny GEMM
and paying all-reduce on top is a net loss.*

*Expert parallelism loses throughout, and by more at larger batch (0.89x → 0.67x).
EP's benefit is a function of expert count and GPU count, not batch size: with 60
experts over 2 GPUs each GPU still holds 30 experts each seeing ~2 tokens, so M is
unchanged while two all-to-alls are added. EP starts paying only when experts number
in the hundreds (DeepSeek-V3 has 256) and TP would shred each one.*

---

## Tensor Parallelism (Week 9)

Qwen2.5-7B-Instruct, A100-80GB, fp16, 256 output tokens, vLLM v0.28 / Python 3.12.
Only `tensor_parallel_size` changes between rows.

| TP | batch=1 | batch=8 | batch=32 | TPOT (b=1) | weights/GPU | KV cache capacity |
|----|---------|---------|----------|------------|-------------|-------------------|
| 1 | 97.2 tok/s | 786.3 tok/s | 2,950.2 tok/s | 10.3 ms | 14.29 GiB | 1,052,560 tok |
| 2 | 150.6 tok/s | 1,101.6 tok/s | 4,501.9 tok/s | 6.6 ms | 7.16 GiB | 2,431,792 tok |
| 4 | 210.0 tok/s | 1,727.7 tok/s | 6,283.3 tok/s | 4.8 ms | 3.63 GiB | 5,135,248 tok |

Speedup vs TP=1: **1.40–1.55x at TP=2, 2.13–2.20x at TP=4** — roughly 75% and 54%
parallel efficiency, and notably *constant across batch size*. The shortfall is the
56 all-reduces per forward pass (28 layers x 2).

*Per-GPU throughput falls as TP rises: 2,950 → 2,251 → 1,571 tok/s per GPU at
batch=32. Tensor parallelism buys latency and capacity, not throughput per dollar —
four TP=1 replicas beat one TP=4 replica for pure throughput.*

*KV cache capacity scales better than throughput (1.05M → 5.14M tokens, 32x → 157x
max concurrency), because sharding frees both weight memory and KV heads.*

*TP=4 is the ceiling for this model: Qwen2.5-7B has 4 KV heads under GQA, so TP=4
leaves exactly one per GPU.*

---

## Speculative Decoding (Week 8)

vLLM n-gram speculative decoding (`num_speculative_tokens=5`), Qwen2.5-7B-Instruct, A100 80GB, fp16.

Two task shapes over the *same* passage — only the instruction text differs, engine config identical:

| Task | batch | baseline | n-gram | speedup |
|------|-------|----------|--------|---------|
| novel (rewrite in own words) | 1 | 80.7 tok/s | 91.1 tok/s | 1.13x |
| **copy** (reproduce verbatim) | 1 | 80.7 tok/s | **224.7 tok/s** | **2.78x** |
| novel | 16 | 923.7 tok/s | 789.0 tok/s | **0.85x** |
| **copy** | 16 | 1,067.2 tok/s | **2,545.2 tok/s** | **2.38x** |

*Speculative decoding's payoff is a function of acceptance rate, not of the engine.
Baseline decode speed is identical across both tasks at batch=1 (80.7 tok/s), so the
entire copy-task speedup comes from accepted speculations.*

*Batch size is an amplifier, not the deciding factor: at batch=16 the GPU is
compute-bound, so rejected tokens cost real FLOPs — low acceptance turns into a
15% net loss, while high acceptance still returns 2.38x.*

*Generic draft-model speculation is not supported by the vLLM V1 engine
(n-gram / Medusa / EAGLE / MTP only).*

---

## Prefix Caching (Week 7)

32 concurrent requests sharing a ~500-token prefix, Qwen2.5-7B-Instruct, A100 80GB, fp16:

| | TTFT P50 | TTFT P99 | Latency P50 | Throughput |
|---|---|---|---|---|
| cache OFF | 829 ms | 1,403 ms | 2.27 s | 835 tok/s |
| cache ON | **156 ms** | **346 ms** | **1.10 s** | **1,412 tok/s** |

*5.3x TTFT reduction. Cached requests skip prefill for the shared prefix entirely;
throughput rises because the reclaimed compute goes to decode.*

---

## Load Pressure (Week 6)

Arrival-rate sweep, fixed 200 output tokens/req, A100 80GB:

| Load | TTFT P50 | TTFT P99 | Latency P99 | Throughput |
|------|----------|----------|-------------|------------|
| ~5 req/s | 33 ms | 43 ms | 2.46 s | 763 tok/s |
| ~20 req/s | 37 ms | 49 ms | 2.56 s | 1,585 tok/s |
| ~20 req/s (64 req) | 38 ms | 50 ms | 2.65 s | 2,291 tok/s |
| ~50 req/s | 39 ms | 50 ms | 2.74 s | 3,273 tok/s |

*The A100 never saturates here: TTFT moves only 33→39 ms across a 10x arrival-rate
increase. Short prompts make prefill trivial — saturating this GPU requires long
prompts, where prefill cost grows with sequence length.*

---

## Continuous vs Static Batching (Week 5)

32 requests, Poisson arrivals (mean 0.2 s), 50-300 output tokens:

| | P50 latency | P99 latency | TTFT P50 | End-to-end throughput |
|---|---|---|---|---|
| static (wait, then batch) | 6.36 s | 8.18 s | — | 717 tok/s |
| continuous | **2.41 s** | **3.91 s** | 49 ms | 705 tok/s |

*Throughput is a wash; latency is not. Continuous batching's value here is
eliminating queueing delay, not raising GPU utilization.*

---

## Batch Size Sweep (Week 4)

vLLM, Qwen2.5-7B-Instruct, A100 80GB, fp16, 1024 output tokens/req:

| batch | total tok/s | wall latency | notes |
|-------|-------------|-------------|-------|
| 1 | 79.9 | 1.25s | single request |
| 4 | 321.9 | 1.24s | ~linear |
| 8 | 642.5 | 1.25s | ~linear |
| 16 | 1,255.4 | 1.27s | ~linear |
| 32 | 2,408.8 | 1.33s | A100 not yet saturated |

*Near-linear throughput scaling with constant wall latency — each additional request costs almost nothing in latency.*

---

## Baseline (Week 1)

| Model | Backend | GPU | dtype | tok/s |
|-------|---------|-----|-------|-------|
| Qwen2.5-0.5B | HuggingFace transformers | A100 40GB | fp16 | ~31 | single req |
| Qwen2.5-7B-Instruct | HuggingFace transformers | A100 40GB | fp16 | ~40 | single req |
| Qwen2.5-7B-Instruct | vLLM | A100 40GB | fp16 | ~80 | single req |
| Qwen2.5-7B-Instruct | HuggingFace transformers | A100 40GB | fp16 | 41 | 8 req serial, total |
| Qwen2.5-7B-Instruct | vLLM | A100 80GB | fp16 | **748** | 8 req batch, total |

*Greedy decode, 100 output tokens.*
*vLLM single-req 2x: CUDA graph + torch.compile (inductor).*
*vLLM batch 18x over HF serial: continuous batching saturates GPU. (~9x if normalizing for 80GB vs 40GB GPU).*

---

## Roadmap

| Week | Topic |
|------|-------|
| 1    | Environment setup, HF baseline |
| 2    | 7B model baseline |
| 3    | vLLM install + baseline benchmark |
| 4    | PagedAttention deep-dive + batch sweep |
| 5-6  | Continuous batching |
| 7    | Prefix caching |
| 8    | Speculative decoding |
| 9-10 | INT8 / INT4 quantization |
| 11-12 | Write-up + publish |

---

## Reproducing

```bash
pip install modal
modal run experiments/wk01_env_check/a100_baseline.py
```

Requires a Modal account (modal.com). A100 GPU time costs ~$3/hr.
