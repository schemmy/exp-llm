"""
Tensor parallelism sweep: what does splitting one model across GPUs actually buy?

Same model, same work, only `tensor_parallel_size` changes. TP=2 means each GPU
holds half the weights and reads half as much per decode step — which is the real
mechanism, since decode is bandwidth-bound (see wk08). The bill is 2 all-reduces
per transformer layer: 28 layers x 2 = 56 collective ops per forward pass.

Expect sub-linear scaling. The gap between measured and Nx *is* the communication cost.

Note on the TP ceiling: column parallelism splits by attention head, so TP must
divide num_key_value_heads. Qwen2.5-7B has 4 KV heads (GQA), so only 1/2/4 are legal.
TP=8 is rejected by vLLM at config time — see test_tp8_fails().

Cost: roughly $3 total (1 + 2 + 4 A100-minutes, a few minutes each).
"""
import modal
import time

app = modal.App("vllm-tp-sweep")

# Python 3.12, not 3.11 as in wk01-08. flashinfer's comm module annotates a helper
# with `array.array[int]`; array.array only became subscriptable in 3.12, and the
# annotation is evaluated at import time, so on 3.11 the import raises
#   TypeError: type 'array.array' is not subscriptable
# vLLM imports that module while building the CUDA communicator, which happens only
# for TP>1 — which is exactly why every single-GPU week ran fine on 3.11 and TP=2
# died at worker init. All three legs share this image so the sweep stays controlled.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 8, 32]

# Without these, every leg re-downloads 15 GB of weights and re-runs inductor from
# scratch — three legs of that is ~20 minutes of foreground time, which is how a
# single network blip took down the whole sweep. Volumes are not part of the image
# cache key, so adding them does not trigger a rebuild.
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)

# Each leg writes its own result file the moment it finishes. A dropped client no
# longer costs anything already measured — re-run only the missing legs (TP_ONLY=2)
# and the report still assembles the full table from whatever is on disk.
RESULTS = modal.Volume.from_name("tp-sweep-results", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": HF_CACHE,
    "/root/.cache/vllm": VLLM_CACHE,
    "/results": RESULTS,
}

PROMPTS = [
    "Explain gradient descent:", "What is the CAP theorem:",
    "How does TCP work:", "Describe transformer architecture:",
    "What is PagedAttention:", "Explain backpropagation:",
    "How does Docker work:", "What is eventual consistency:",
    "Explain attention mechanism:", "How does DNS resolution work:",
    "What is a hash table:", "Explain public key cryptography:",
    "How does garbage collection work:", "What is microservices:",
    "Explain process vs thread:", "How does a compiler work:",
    "What is vector embedding:", "Explain OLTP vs OLAP:",
    "How does load balancing work:", "What is REST vs GraphQL:",
    "Explain database indexing:", "How does HTTPS work:",
    "What is functional programming:", "Explain Linux scheduler:",
    "How does a neural network learn:", "SQL vs NoSQL:",
    "Explain CNNs:", "How does Kubernetes work:",
    "What is reinforcement learning:", "Explain recursion:",
    "Memory management in C:", "Supervised vs unsupervised learning:",
]


def bench(tp: int):
    """Shared body. Runs inside a container that already has `tp` GPUs attached."""
    import os
    import subprocess

    # Set before importing vllm so the workers inherit them.
    #   spawn: fork hands a broken CUDA context to workers once the parent has
    #          touched CUDA. Standard for vLLM TP.
    #   cumem: defensive — NCCL's cuMem allocator wants IPC handles some container
    #          runtimes withhold. Not the cause of the 3.11 failure, kept as a guard.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

    from vllm import LLM, SamplingParams

    print(f"\n{'='*62}\n  tensor_parallel_size = {tp}\n{'='*62}")

    # Interconnect matters more than GPU count: NVLink is ~900 GB/s, PCIe ~64 GB/s.
    # All-reduce latency is what turns TP scaling sub-linear, so record the topology.
    for cmd, label in ((["nvidia-smi", "topo", "-m"], "topology"),
                       (["df", "-h", "/dev/shm"], "/dev/shm")):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            print(f"--- {label} ---\n{out}")
        except Exception as e:
            print(f"({label} unavailable: {e})")

    llm = LLM(model=MODEL_ID, dtype="float16", tensor_parallel_size=tp)

    # Persist the weights now rather than at function exit, so a failure in the
    # benchmark below does not throw away a 15 GB download.
    HF_CACHE.commit()

    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)

    # warmup absorbs torch.compile + CUDA graph capture + NCCL handshake
    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=16, temperature=0))

    results = {}
    for batch in BATCH_SIZES:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0

        total = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total / elapsed
        tpot = elapsed / (total / batch) * 1000
        results[batch] = tps
        print(
            f"  TP={tp}  batch={batch:<3} "
            f"wall={elapsed:.2f}s  total={tps:.1f} tok/s  "
            f"per-req={tps/batch:.1f} tok/s  TPOT={tpot:.1f}ms"
        )

    out = {"tp": tp, "throughput": results}

    import json
    import pathlib
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/tp{tp}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  → saved /results/tp{tp}.json")

    return out


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=1800)
def tp1():
    return bench(1)


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=1800)
def tp2():
    return bench(2)


@app.function(gpu="A100-80GB:4", image=image, volumes=VOLUMES, timeout=1800)
def tp4():
    return bench(4)


@app.function(gpu="A100-80GB:4", image=image, volumes=VOLUMES, timeout=600)
def test_tp8_fails():
    """TP is capped by num_key_value_heads (4 for Qwen2.5-7B), not by GPU count."""
    from vllm import LLM
    try:
        LLM(model=MODEL_ID, dtype="float16", tensor_parallel_size=8)
        print("\n[!] TP=8 unexpectedly succeeded — check the model's KV head count.")
    except Exception as e:
        print(f"\n[expected] TP=8 rejected:\n  {type(e).__name__}: {e}")


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    """Read whatever legs have completed, across any number of past sessions."""
    import json
    import pathlib
    rows = []
    for tp in (1, 2, 4):
        p = pathlib.Path(f"/results/tp{tp}.json")
        if p.exists():
            r = json.loads(p.read_text())
            # JSON turns the int batch keys into strings on the way out
            r["throughput"] = {int(k): v for k, v in r["throughput"].items()}
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("TP_ONLY")          # e.g. TP_ONLY=2 to re-run just that leg
    cases = [(tp1, 1), (tp2, 2), (tp4, 4)]
    if only:
        cases = [c for c in cases if str(c[1]) == only]

    print("TP sweep on Qwen2.5-7B — runs 1, 2, then 4 GPUs sequentially.\n")
    for fn, tp in cases:
        try:
            fn.remote()
        except Exception as e:
            print(f"\n!!! TP={tp} failed: {type(e).__name__}: {e}\n")

    # Build the table from the volume, not from this session's return values, so
    # legs measured in earlier runs still count.
    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return
    if not any(r["tp"] == 1 for r in rows):
        print("TP=1 baseline missing — run it before the speedup table means anything.")
        for r in rows:
            print(f"  TP={r['tp']}: {r['throughput']}")
        return

    base = next(r for r in rows if r["tp"] == 1)["throughput"]

    print(f"\n{'='*62}\n  THROUGHPUT (tok/s)\n{'='*62}")
    print("  TP   " + "".join(f"batch={b:<12}" for b in BATCH_SIZES))
    for r in rows:
        line = f"  {r['tp']:<5}"
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]:,.1f}".ljust(18)
        print(line)

    print(f"\n{'='*62}\n  SPEEDUP vs TP=1\n{'='*62}")
    print("  TP   " + "".join(f"batch={b:<12}" for b in BATCH_SIZES))
    for r in rows:
        line = f"  {r['tp']:<5}"
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]/base[b]:.2f}x".ljust(18)
        print(line)
