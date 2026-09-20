# Distributed Training Benchmarks

Phase 2. Reproducible benchmarks for training LLMs across multiple GPUs —
DDP, FSDP/ZeRO, and LoRA/QLoRA/DPO fine-tuning. Companion to
[`README.md`](README.md) (Phase 1, single- and multi-GPU *inference*).

**Status**: Week 15 of 28 — DDP scaling baseline complete.

## DDP Scaling: 2× A100 vs 1× A100 (Week 15)

GPT-2 124M (random init, synthetic random-token inputs — the point is the
DDP communication mechanism, not a trained model), seq_len 512, 100-step
warmup + 200-step measurement, per-GPU batch size held constant across the
single-GPU and DDP legs so the DDP column is a genuine doubled-batch
comparison.

| batch (per-GPU) | single-GPU tok/s | DDP total tok/s (2×) | scaling efficiency |
|---|---|---|---|
| 4  | 17,921.3 | 34,608.9 | **96.56%** |
| 16 | 19,633.2 | 38,814.8 | **98.85%** |
| 64 | 20,603.2 | 41,190.0 | **99.96%** |

*Scaling efficiency = DDP total throughput / (2 × single-GPU throughput).
DDP's `tokens_per_sec` is rank 0's own per-GPU number — both ranks are
lockstepped every step by the blocking gradient all-reduce, so rank 0's
cadence is the system's joint cadence and total throughput = 2× that number.*

**Predicted vs measured**: expected efficiency to be noticeably worse at
small batch (guessed 70-85% at batch=4, reasoning: less compute per step
means the fixed gradient all-reduce cost is a bigger fraction of the step).
**Measured 96.56% even at batch=4** — the prediction was wrong. GPT-2 124M's
gradient volume (~500MB fp32, synced once per step) is small enough, and the
interconnect fast enough, that DDP's overlap of communication with backward
computation hides nearly all of it even at the smallest batch tested here.

### Step profile at batch=16 (`torch.profiler`, 10 steps)

| Metric | Value |
|---|---|
| comm_fraction (CUDA time in nccl/all-reduce ops ÷ total CUDA time) | **4.41%** |

Top CUDA-time events (10-step aggregate): `aten::mm` (2,587.6ms, 990 calls),
`AddmmBackward0` (~1,567-1,591ms, 480 calls), `DistributedDataParallel.forward`
(~1,390-1,428ms, 10 calls — appears as two separate profiler entries with
close but non-identical totals, not yet explained), `aten::addmm` (780.0ms,
480 calls). Matmul/addmm backward dominates; nccl-tagged ops are a small
slice of total CUDA time, consistent with the near-100% scaling efficiency
above — measured comm_fraction (4.41%) is somewhat larger than the measured
efficiency loss at batch=16 (1.15%), which fits DDP's overlap design:
communication issued during backward is partly hidden behind still-running
backward compute, so its share of *wall-clock* step time is smaller than its
share of raw CUDA time.

*Script: [`experiments/wk15_ddp/ddp_scaling.py`](../experiments/wk15_ddp/ddp_scaling.py).*

---

## Roadmap

| Week | Topic |
|------|-------|
| 15-16 | PyTorch DDP: scaling baseline, gradient bucketing |
| 17-19 | FSDP full-shard on 7B; ZeRO-1/2/3; activation checkpointing + CPU offload |
| 20-24 | LoRA / QLoRA fine-tune on a real task |
| 25-26 | DPO on top of the same LoRA setup |
| 27-28 | Write-up + publish (Blog #2) |

---

## Reproducing

```bash
pip install modal
modal run --detach experiments/wk15_ddp/ddp_scaling.py
```

Requires a Modal account (modal.com). 2× A100 GPU time costs more than 1×;
budget ~$8-10 for the full Wk 15 sweep (6 legs).
