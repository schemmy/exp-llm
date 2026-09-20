"""
PyTorch DDP baseline: what does replicating a model across 2 GPUs actually buy?

Same model, same per-GPU batch size, only the GPU count changes. The naive
expectation is 2x throughput — but after every backward pass, DDP has to
all-reduce the gradients across both GPUs before optimizer.step() can run,
and that's pure communication, not compute. This script measures how much
of the theoretical 2x survives at three batch sizes, and profiles one DDP
step to see where the time actually goes (forward / backward / comm / optim).

Model is GPT-2 124M (random init, no download) on synthetic random tokens —
the point this week is the DDP mechanism, not a trained model or real data.
124M was chosen specifically because it's a pure-DDP scenario: it comfortably
fits on one GPU with room to spare, so the only new cost this week is the
gradient all-reduce, not sharding (that's Wk 17's FSDP problem).

Cost: roughly $8-10 total (single-GPU legs + 2-GPU legs, few minutes each).
"""
import modal

app = modal.App("wk15-ddp-scaling")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("torch", "transformers")
)

RESULTS = modal.Volume.from_name("wk15-ddp-results", create_if_missing=True)
VOLUMES = {"/results": RESULTS}

SEQ_LEN = 512
VOCAB_SIZE = 50257
BATCH_SIZES = [4, 16, 64]
N_WARMUP = 100
N_MEASURE = 200
MASTER_PORT = "29500"


def build_model(device):
    # Random init, not from_pretrained — no download, and weight *values* don't
    # matter for a throughput/comm measurement, only the model's shape does.
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        n_layer=12, n_head=12, n_embd=768,
        vocab_size=VOCAB_SIZE, n_positions=SEQ_LEN,
    )
    return GPT2LMHeadModel(config).to(device)


def run_steps(model, optimizer, batch_size, device, n_warmup, n_measure):
    """Shared timing loop — used identically by the single-GPU and DDP legs so
    the resulting tokens/sec numbers are directly comparable."""
    import torch
    import time

    model.train()
    for step in range(n_warmup + n_measure):
        if step == n_warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=device)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = batch_size * SEQ_LEN * n_measure
    return {
        "tokens_per_sec": tokens / elapsed,
        "step_time_ms": elapsed / n_measure * 1000,
        "max_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def profile_ddp_step(model, optimizer, batch_size, device, n_steps=10):
    """Runs a few profiled DDP steps and buckets CUDA time into forward /
    backward / comm / optim by matching on event name. DDP overlaps the
    gradient all-reduce with backward (it fires per-bucket as soon as that
    bucket's gradients are ready, not after the whole backward pass) so this
    is a real trace, not a clean sequential breakdown — comm shows up
    interleaved with backward ops, which is itself the finding worth seeing."""
    import torch
    from torch.profiler import profile, ProfilerActivity

    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=device)
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    events = prof.key_averages()
    comm_us = sum(e.cuda_time_total for e in events
                  if "nccl" in e.key.lower() or "allreduce" in e.key.lower())
    total_us = sum(e.cuda_time_total for e in events)
    top10 = sorted(events, key=lambda e: -e.cuda_time_total)[:10]

    return {
        "comm_fraction": comm_us / total_us if total_us else None,
        "top_events": [
            {"name": e.key, "cuda_time_us": e.cuda_time_total, "count": e.count}
            for e in top10
        ],
    }


def ddp_worker(rank, world_size, batch_size, do_profile, out_path):
    """Entry point for each spawned process. Must stay a top-level function —
    torch.multiprocessing.spawn pickles it by reference, a closure won't work."""
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

    model = DDP(build_model(device), device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    result = run_steps(model, optimizer, batch_size, device, N_WARMUP, N_MEASURE)

    if do_profile:
        # Every rank must run the same profiled steps (they're still doing
        # synchronous collectives together) — only rank 0 keeps the trace.
        prof_result = profile_ddp_step(model, optimizer, batch_size, device)
        if rank == 0:
            result["profile"] = prof_result

    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(result, f)

    dist.destroy_process_group()


def bench_single(batch_size: int):
    import torch
    device = "cuda:0"
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return run_steps(model, optimizer, batch_size, device, N_WARMUP, N_MEASURE)


def bench_ddp(batch_size: int, do_profile: bool = False):
    import torch.multiprocessing as mp
    import json
    import pathlib

    out_path = f"/results/_ddp_worker_bs{batch_size}.json"
    mp.spawn(ddp_worker, args=(2, batch_size, do_profile, out_path), nprocs=2, join=True)
    return json.loads(pathlib.Path(out_path).read_text())


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=1800)
def single_bs4():
    return _run_and_save("single", 4, bench_single(4))


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=1800)
def single_bs16():
    return _run_and_save("single", 16, bench_single(16))


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=1800)
def single_bs64():
    return _run_and_save("single", 64, bench_single(64))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def ddp_bs4():
    return _run_and_save("ddp", 4, bench_ddp(4))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def ddp_bs16():
    # The middle batch size carries the Task 3 profiler trace — no need to
    # pay for it three times, one representative trace is the point.
    return _run_and_save("ddp", 16, bench_ddp(16, do_profile=True))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def ddp_bs64():
    return _run_and_save("ddp", 64, bench_ddp(64))


def _run_and_save(kind: str, batch_size: int, result: dict):
    import json
    import pathlib
    pathlib.Path("/results").mkdir(exist_ok=True)
    out = {"kind": kind, "batch_size": batch_size, **result}
    pathlib.Path(f"/results/{kind}_bs{batch_size}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  -> saved /results/{kind}_bs{batch_size}.json: {result}")
    return out


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    """Read whatever legs have completed, across any number of past sessions —
    same partial-progress pattern as wk09's TP sweep."""
    import json
    import pathlib
    rows = []
    for kind in ("single", "ddp"):
        for bs in BATCH_SIZES:
            p = pathlib.Path(f"/results/{kind}_bs{bs}.json")
            if p.exists():
                rows.append(json.loads(p.read_text()))
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("LEGS_ONLY")  # e.g. LEGS_ONLY=ddp_bs16 to re-run just that leg
    legs = {
        "single_bs4": single_bs4, "single_bs16": single_bs16, "single_bs64": single_bs64,
        "ddp_bs4": ddp_bs4, "ddp_bs16": ddp_bs16, "ddp_bs64": ddp_bs64,
    }
    names = [n for n in legs if not only or n == only]

    print("Wk15 DDP scaling sweep: 3 single-GPU legs, then 3 DDP legs.\n")
    for name in names:
        try:
            legs[name].remote()
        except Exception as e:
            print(f"\n!!! {name} failed: {type(e).__name__}: {e}\n")

    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    singles = {r["batch_size"]: r for r in rows if r["kind"] == "single"}
    ddps = {r["batch_size"]: r for r in rows if r["kind"] == "ddp"}

    print(f"\n{'='*70}\n  THROUGHPUT (tokens/sec)\n{'='*70}")
    print(f"  {'batch':<8}{'single-GPU':<16}{'2-GPU DDP':<16}{'scaling eff.':<14}")
    for bs in BATCH_SIZES:
        s = singles.get(bs)
        d = ddps.get(bs)
        s_tps = f"{s['tokens_per_sec']:,.1f}" if s else "-"
        d_tps = f"{d['tokens_per_sec']:,.1f}" if d else "-"
        eff = f"{d['tokens_per_sec'] / (2 * s['tokens_per_sec']):.2%}" if s and d else "-"
        print(f"  {bs:<8}{s_tps:<16}{d_tps:<16}{eff:<14}")

    if 16 in ddps and "profile" in ddps[16]:
        prof = ddps[16]["profile"]
        print(f"\n{'='*70}\n  DDP STEP PROFILE (batch=16, comm/total by CUDA time)\n{'='*70}")
        print(f"  comm_fraction: {prof['comm_fraction']:.2%}" if prof['comm_fraction'] is not None else "  comm_fraction: n/a")
        print(f"\n  Top CUDA-time events:")
        for e in prof["top_events"]:
            print(f"    {e['name']:<40} {e['cuda_time_us']/1000:>10.2f} ms  (x{e['count']})")
