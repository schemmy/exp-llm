"""
Does Qwen2.5-7B actually fit for training on one A100-80GB, and does FSDP's
full-shard (ZeRO-3 equivalent) fix it?

Wk 16 concluded "FSDP solves a memory problem, not a communication-efficiency
problem" as a prediction from DDP's near-saturated overlap at smaller model
sizes. This script tests that prediction directly on the actual model P1
spent 14 weeks benchmarking for inference (Qwen2.5-7B-Instruct), rather than
assuming it:

  1. single_gpu_probe: try to load the model in fp32 on ONE A100-80GB, build
     an AdamW optimizer, and run one forward/backward/step, catching an OOM
     at whichever stage it happens and reporting how far it got.
  2. fsdp_fp32: same fp32 precision, but FSDP FULL_SHARD across 2 GPUs — the
     apples-to-apples comparison against (1): same precision, sharded vs not.
  3. fsdp_fp16: FULL_SHARD with the model loaded directly in fp16 (not cast
     after loading) — the "safe bet" leg, expected to comfortably fit, so
     this week has at least one working FSDP throughput number regardless of
     how (2) goes. Also profiled with torch.profiler to get FSDP's
     comm_fraction (all-gather + reduce-scatter), comparable to Wk 15's DDP
     comm_fraction (4.41% at 124M).

Static memory arithmetic (fp32): weights 7.6B x 4B = 30.4GB, gradients
another 30.4GB, AdamW's exp_avg + exp_avg_sq (fp32, same dtype as params by
default) another 60.8GB. Total ~121.6GB — over a single 80GB card's budget
before any activations. FULL_SHARD across 2 GPUs would put ~60.8GB of that
per rank, which may or may not leave enough headroom for activations +
transient all-gather buffers; that's genuinely unknown going in, which is
the point of running it rather than assuming.

Cost: roughly $12-16 total (1x + 2x A100 legs, ~15GB model download shared
via the same hf-cache volume P1's experiments used).
"""
import modal

app = modal.App("wk17-fsdp")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("torch", "transformers", "accelerate")
)

# Reuses P1's hf-cache volume — Qwen2.5-7B-Instruct may already be cached
# from the Wk1-14 inference experiments, avoiding a redundant ~15GB download.
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("wk17-fsdp-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": HF_CACHE, "/results": RESULTS}

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEQ_LEN = 512
BATCH_SIZE = 1  # 7B is far more memory-hungry per token than Wk15-16's
                # models; batch=1 is the right starting point, not a
                # simplification chosen for convenience.
N_WARMUP = 20
N_MEASURE = 50  # smaller than Wk15/16 — 7B steps are much slower per step,
                # no need for as many samples to get a stable average.
MASTER_PORT = "29500"


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=1200)
def single_gpu_probe():
    """Load Qwen2.5-7B in fp32 on one GPU and try one full training step,
    catching an OOM at whichever stage it happens rather than letting the
    whole container crash uninformatively."""
    import json
    import pathlib
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    device = "cuda:0"
    result = {"stage_reached": None, "max_memory_gb": None, "error": None}

    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
        vocab_size = config.vocab_size

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
        result["stage_reached"] = "model_loaded"
        result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result["stage_reached"] = "optimizer_created"
        result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
        out = model(input_ids=input_ids, labels=input_ids)
        result["stage_reached"] = "forward_done"
        result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        out.loss.backward()
        result["stage_reached"] = "backward_done"
        result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        # Adam lazily allocates exp_avg/exp_avg_sq on the FIRST step() call —
        # this is the most likely point to actually OOM, not model loading.
        optimizer.step()
        result["stage_reached"] = "optimizer_step_done"
        result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

    except torch.cuda.OutOfMemoryError as e:
        result["error"] = f"OOM after stage '{result['stage_reached']}': {str(e)[:300]}"
        try:
            result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        except Exception:
            pass
    except Exception as e:
        result["error"] = f"{type(e).__name__} after stage '{result['stage_reached']}': {str(e)[:300]}"

    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path("/results/single_gpu_probe.json").write_text(json.dumps(result))
    RESULTS.commit()
    print(f"  -> saved /results/single_gpu_probe.json: {result}")
    return result


def device_us(e):
    """PyTorch renamed FunctionEventAvg's cuda_time_total to device_time_total
    somewhere around 2.1 — same fix as Wk 15's profiler code."""
    for attr in ("device_time_total", "cuda_time_total"):
        if hasattr(e, attr):
            return getattr(e, attr)
    return 0


def fsdp_worker(rank, world_size, dtype_name, do_profile, out_path):
    """Top-level function — required for torch.multiprocessing.spawn to
    pickle it by reference; a closure would not work.

    Note: if one rank OOMs mid-collective while the other doesn't, the
    surviving rank can hang waiting on that collective rather than erroring
    cleanly — a known rough edge of testing OOM behavior in a multi-process
    setting. The Modal function's timeout is the backstop for that case."""
    import os
    import json
    import time
    import functools
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoConfig

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", MASTER_PORT)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    result = {"dtype": dtype_name, "error": None}
    try:
        dtype = torch.float32 if dtype_name == "fp32" else torch.float16
        config = AutoConfig.from_pretrained(MODEL_ID)
        vocab_size = config.vocab_size

        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e7))
        model = FSDP(
            base_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        for step in range(N_WARMUP + N_MEASURE):
            if step == N_WARMUP:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tokens = BATCH_SIZE * SEQ_LEN * N_MEASURE
        result["tokens_per_sec"] = tokens / elapsed
        result["step_time_ms"] = elapsed / N_MEASURE * 1000
        result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        if do_profile:
            from torch.profiler import profile, ProfilerActivity
            model.train()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
                    out = model(input_ids=input_ids, labels=input_ids)
                    out.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
            events = prof.key_averages()
            comm_keys = ("nccl", "all_gather", "allgather", "reduce_scatter", "reducescatter", "allreduce")
            comm_us = sum(device_us(e) for e in events if any(k in e.key.lower() for k in comm_keys))
            total_us = sum(device_us(e) for e in events)
            result["comm_fraction"] = comm_us / total_us if total_us else None

    except torch.cuda.OutOfMemoryError as e:
        result["error"] = f"OOM: {str(e)[:300]}"
        try:
            result["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        except Exception:
            result["max_memory_gb"] = None
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(result, f)

    dist.destroy_process_group()


def bench_fsdp(dtype_name: str, do_profile: bool = False):
    import torch.multiprocessing as mp
    import json
    import pathlib

    out_path = f"/results/_worker_fsdp_{dtype_name}.json"
    mp.spawn(fsdp_worker, args=(2, dtype_name, do_profile, out_path), nprocs=2, join=True)
    return json.loads(pathlib.Path(out_path).read_text())


def _run_and_save(name: str, result: dict):
    import json
    import pathlib
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/{name}.json").write_text(json.dumps(result))
    RESULTS.commit()
    print(f"  -> saved /results/{name}.json: {result}")
    return result


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=2400)
def fsdp_fp32():
    return _run_and_save("fsdp_fp32", bench_fsdp("fp32"))


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=2400)
def fsdp_fp16():
    # Profiler runs on this leg specifically (not conditionally on whichever
    # succeeds) — deterministic and simple, and this is the leg expected to
    # reliably succeed regardless of how fsdp_fp32 goes.
    return _run_and_save("fsdp_fp16", bench_fsdp("fp16", do_profile=True))


LEGS = {
    "single_gpu_probe": single_gpu_probe,
    "fsdp_fp32": fsdp_fp32,
    "fsdp_fp16": fsdp_fp16,
}


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = {}
    for name in LEGS:
        p = pathlib.Path(f"/results/{name}.json")
        if p.exists():
            rows[name] = json.loads(p.read_text())
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("LEGS_ONLY")  # e.g. LEGS_ONLY=fsdp_fp16
    names = [n for n in LEGS if not only or n == only]

    print(f"Wk17 FSDP: {len(names)} leg(s).\n")
    for name in names:
        try:
            LEGS[name].remote()
        except Exception as e:
            print(f"\n!!! {name} failed: {type(e).__name__}: {e}\n")

    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    print(f"\n{'='*70}\n  SINGLE-GPU PROBE (fp32, batch=1, 1x A100)\n{'='*70}")
    p = rows.get("single_gpu_probe")
    if p:
        print(f"  stage reached: {p.get('stage_reached')}")
        print(f"  max_memory_gb at that point: {p.get('max_memory_gb')}")
        print(f"  error: {p.get('error')}")
    else:
        print("  (missing)")

    print(f"\n{'='*70}\n  FSDP FULL_SHARD (2x A100, batch=1)\n{'='*70}")
    print(f"  {'dtype':<8}{'status':<10}{'tok/s':<12}{'max_mem_gb':<12}{'comm_frac':<10}")
    for name in ("fsdp_fp32", "fsdp_fp16"):
        r = rows.get(name)
        if not r:
            print(f"  {name:<8}(missing)")
            continue
        status = "OK" if not r.get("error") else "FAILED"
        tps = f"{r['tokens_per_sec']:,.1f}" if r.get("tokens_per_sec") else "-"
        mem = f"{r['max_memory_gb']:.2f}" if r.get("max_memory_gb") is not None else "-"
        cf = f"{r['comm_fraction']:.2%}" if r.get("comm_fraction") is not None else "-"
        dtype = r.get("dtype", name)
        print(f"  {dtype:<8}{status:<10}{tps:<12}{mem:<12}{cf:<10}")
        if r.get("error"):
            print(f"    error: {r['error']}")
