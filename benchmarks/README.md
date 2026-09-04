# LLM Inference Benchmarks

Reproducible benchmarks for LLM serving on a single GPU.
Comparing HuggingFace transformers baseline vs vLLM with continuous
batching, PagedAttention, prefix caching, and quantization.

**Status**: Week 8 of 12 — speculative decoding complete.

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
