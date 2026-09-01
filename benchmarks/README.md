# LLM Inference Benchmarks

Reproducible benchmarks for LLM serving on a single GPU.
Comparing HuggingFace transformers baseline vs vLLM with continuous
batching, PagedAttention, prefix caching, and quantization.

**Status**: Week 3 of 12 — vLLM single-request baseline complete.

---

## Baseline (Week 1)

| Model | Backend | GPU | dtype | tok/s |
|-------|---------|-----|-------|-------|
| Qwen2.5-0.5B | HuggingFace transformers | A100 | fp16 | ~31 |
| Qwen2.5-7B-Instruct | HuggingFace transformers | A100 | fp16 | ~40 |
| Qwen2.5-7B-Instruct | vLLM | A100 | fp16 | ~80 |

*Greedy decode, 100 output tokens, single request, steady-state (run 2+).*
*7B HF > 0.5B HF: larger matmuls better saturate A100 tensor cores.*
*vLLM 2x over HF: CUDA graph capture + torch.compile (inductor) eliminate Python/kernel-launch overhead.*

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
