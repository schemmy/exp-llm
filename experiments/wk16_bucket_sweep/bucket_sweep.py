"""
Does DDP's bucket_cap_mb actually matter, and at what model size does it
start to?

Wk 15 found DDP scaling efficiency on GPT-2 124M already at 96.6%-99.96% —
the gradient all-reduce overlaps with backward closely enough that there's
almost no room left for a knob like bucket_cap_mb (the size threshold DDP
uses to decide when to flush a bucket of gradients and fire its all-reduce)
to improve on. This script tests that directly, two ways:

  1. Sweep bucket_cap_mb on the same 124M model Wk 15 used — if overlap is
     really saturated, all four bucket sizes should land within noise of
     each other.
  2. Repeat the same sweep on GPT-2 Large (774M, ~6x the gradient volume)
     to see whether a bigger model gives bucket_cap_mb room to matter, and
     whether scaling efficiency itself degrades from Wk 15's near-100%.

Reuses wk15_ddp's model-building and timing-loop shape exactly so the 124M
numbers here are directly comparable to Wk 15's, not just structurally
similar.

Cost: roughly $10-14 total (2 model sizes x 4 bucket sizes x 2-GPU legs).
"""
import modal

app = modal.App("wk16-bucket-sweep")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("torch", "transformers")
)

RESULTS = modal.Volume.from_name("wk16-bucket-results", create_if_missing=True)
VOLUMES = {"/results": RESULTS}

SEQ_LEN = 512
VOCAB_SIZE = 50257
BATCH_SIZE = 16  # Wk 15's middle batch — this week's variables are model size
                 # and bucket size, not batch, so only one batch is tested.
N_WARMUP = 100
N_MEASURE = 200
BUCKET_SIZES_MB = [1, 25, 100, 500]  # 25 is PyTorch's DDP default; 500 is
                                      # ~1 bucket for the 124M model (whose
                                      # total gradient is ~500MB) but still
                                      # several buckets for the 774M model.
MODEL_SIZES = ["124m", "large"]
MASTER_PORT = "29500"

MODEL_CONFIGS = {
    # Matches wk15_ddp/ddp_scaling.py exactly — same model, for comparability.
    "124m": dict(n_layer=12, n_head=12, n_embd=768),
    # GPT-2 Large. Deliberately not GPT-2 XL (1.5B): back-of-envelope
    # activation memory at batch=16/seq_len=512 for XL (~47GB) plus weights +
    # gradients + AdamW state (~24GB) leaves too little headroom on an
    # 80GB card once CUDA context overhead is included. Large (774M) keeps
    # a comfortable margin (~40GB total) while still giving ~6x the
    # gradient volume of the 124M baseline.
    "large": dict(n_layer=36, n_head=20, n_embd=1280),
}


def build_model(device, model_size):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = MODEL_CONFIGS[model_size]
    config = GPT2Config(
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        vocab_size=VOCAB_SIZE, n_positions=SEQ_LEN,
    )
    return GPT2LMHeadModel(config).to(device)


def run_steps(model, optimizer, device, n_warmup, n_measure):
    import torch
    import time

    model.train()
    for step in range(n_warmup + n_measure):
        if step == n_warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = BATCH_SIZE * SEQ_LEN * n_measure
    return {
        "tokens_per_sec": tokens / elapsed,
        "step_time_ms": elapsed / n_measure * 1000,
        "max_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def count_allreduce_calls(model, optimizer, device, n_steps=10):
    """A quick profiled run just to confirm smaller buckets really do fire
    more all-reduce calls — sanity-checks the measurement method itself,
    independent of whether the throughput hypothesis holds."""
    import torch
    from torch.profiler import profile, ProfilerActivity

    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    events = prof.key_averages()
    nccl_events = [e for e in events if "nccl" in e.key.lower() or "allreduce" in e.key.lower()]
    return sum(e.count for e in nccl_events)


def ddp_worker(rank, world_size, model_size, bucket_cap_mb, out_path):
    """Top-level function — required for torch.multiprocessing.spawn to
    pickle it by reference; a closure would not work."""
    import os
    import json
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", MASTER_PORT)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    model = DDP(build_model(device, model_size), device_ids=[rank], bucket_cap_mb=bucket_cap_mb)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    result = run_steps(model, optimizer, device, N_WARMUP, N_MEASURE)
    result["allreduce_calls"] = count_allreduce_calls(model, optimizer, device)

    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(result, f)

    dist.destroy_process_group()


def bench(model_size: str, bucket_cap_mb: int):
    import torch.multiprocessing as mp
    import json
    import pathlib

    out_path = f"/results/_worker_{model_size}_b{bucket_cap_mb}.json"
    mp.spawn(ddp_worker, args=(2, model_size, bucket_cap_mb, out_path), nprocs=2, join=True)
    return json.loads(pathlib.Path(out_path).read_text())


def bench_single(model_size: str):
    """Single-GPU baseline — needed to compute scaling efficiency for the
    "large" model, which (unlike 124m) has no Wk 15 baseline to reuse."""
    import torch
    device = "cuda:0"
    model = build_model(device, model_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return run_steps(model, optimizer, device, N_WARMUP, N_MEASURE)


def _run_and_save(model_size: str, bucket_cap_mb: int, result: dict):
    import json
    import pathlib
    pathlib.Path("/results").mkdir(exist_ok=True)
    out = {"model_size": model_size, "bucket_cap_mb": bucket_cap_mb, **result}
    pathlib.Path(f"/results/{model_size}_b{bucket_cap_mb}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  -> saved /results/{model_size}_b{bucket_cap_mb}.json: {result}")
    return out


# One @app.function per (model_size, bucket_cap_mb) leg, hand-declared rather
# than generated in a loop — same partial-progress pattern as wk09/wk15: each
# leg is independent, a dropped client costs nothing already measured, and
# LEGS_ONLY lets you re-run just the leg that failed.

@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_124m_b1():
    return _run_and_save("124m", 1, bench("124m", 1))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_124m_b25():
    return _run_and_save("124m", 25, bench("124m", 25))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_124m_b100():
    return _run_and_save("124m", 100, bench("124m", 100))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_124m_b500():
    return _run_and_save("124m", 500, bench("124m", 500))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_large_b1():
    return _run_and_save("large", 1, bench("large", 1))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_large_b25():
    return _run_and_save("large", 25, bench("large", 25))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_large_b100():
    return _run_and_save("large", 100, bench("large", 100))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def bucket_large_b500():
    return _run_and_save("large", 500, bench("large", 500))


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=1800)
def single_large():
    import json
    import pathlib
    result = bench_single("large")
    pathlib.Path("/results").mkdir(exist_ok=True)
    out = {"model_size": "large", **result}
    pathlib.Path("/results/single_large.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  -> saved /results/single_large.json: {result}")
    return out


LEGS = {
    "124m_b1": bucket_124m_b1, "124m_b25": bucket_124m_b25,
    "124m_b100": bucket_124m_b100, "124m_b500": bucket_124m_b500,
    "large_b1": bucket_large_b1, "large_b25": bucket_large_b25,
    "large_b100": bucket_large_b100, "large_b500": bucket_large_b500,
    "single_large": single_large,
}


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = []
    for ms in MODEL_SIZES:
        for b in BUCKET_SIZES_MB:
            p = pathlib.Path(f"/results/{ms}_b{b}.json")
            if p.exists():
                rows.append(json.loads(p.read_text()))
    single_large_p = pathlib.Path("/results/single_large.json")
    single_large_row = json.loads(single_large_p.read_text()) if single_large_p.exists() else None
    return rows, single_large_row


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("LEGS_ONLY")  # e.g. LEGS_ONLY=large_b1
    names = [n for n in LEGS if not only or n == only]

    print(f"Wk16 bucket_cap_mb sweep: {len(names)} leg(s).\n")
    for name in names:
        try:
            LEGS[name].remote()
        except Exception as e:
            print(f"\n!!! {name} failed: {type(e).__name__}: {e}\n")

    rows, single_large_row = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    for ms in MODEL_SIZES:
        ms_rows = {r["bucket_cap_mb"]: r for r in rows if r["model_size"] == ms}
        if not ms_rows:
            continue
        print(f"\n{'='*70}\n  {ms.upper()} — bucket_cap_mb sweep (batch={BATCH_SIZE})\n{'='*70}")
        print(f"  {'bucket_mb':<12}{'tok/s':<14}{'step_ms':<12}{'allreduce_calls':<18}{'max_mem_gb':<12}")
        for b in BUCKET_SIZES_MB:
            r = ms_rows.get(b)
            if not r:
                print(f"  {b:<12}(missing)")
                continue
            print(f"  {b:<12}{r['tokens_per_sec']:<14,.1f}{r['step_time_ms']:<12.2f}"
                  f"{r['allreduce_calls']:<18}{r['max_memory_gb']:<12.2f}")

    # The number Wk 16 actually wants: does "large" hold Wk 15's near-100%
    # scaling efficiency, or does a 6x bigger gradient start to cost something?
    # Compared at bucket_cap_mb=25 (PyTorch's default) for an apples-to-apples
    # read against Wk 15's own default-bucket 124M number.
    large_default = {r["bucket_cap_mb"]: r for r in rows if r["model_size"] == "large"}.get(25)
    if single_large_row and large_default:
        eff = large_default["tokens_per_sec"] / single_large_row["tokens_per_sec"]
        print(f"\n{'='*70}\n  LARGE (774M) SCALING EFFICIENCY @ bucket_cap_mb=25\n{'='*70}")
        print(f"  single-GPU tok/s: {single_large_row['tokens_per_sec']:,.1f}")
        print(f"  DDP (rank0) tok/s: {large_default['tokens_per_sec']:,.1f}")
        print(f"  scaling efficiency: {eff:.2%}  (Wk15's 124M @ bs16 was 98.85%)")
    elif large_default and not single_large_row:
        print("\n(single_large baseline missing — can't compute 774M scaling "
              "efficiency yet; run LEGS_ONLY=single_large)")
