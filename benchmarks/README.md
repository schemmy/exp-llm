# LLM Inference Benchmarks

Reproducible benchmarks for LLM serving on a single GPU.
Comparing HuggingFace transformers baseline vs vLLM with continuous
batching, PagedAttention, prefix caching, and quantization.

**Status**: Week 1 of 12 — dev environment setup.

---

## Baseline (Week 1)

| Model | Backend | GPU | dtype | tok/s |
|-------|---------|-----|-------|-------|
| Qwen2.5-0.5B | HuggingFace transformers | A100 | fp16 | ~31 |

*Greedy decode, 100 output tokens, steady-state (run 2+).*

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
