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
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 8, 32]

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

    # Set before importing vllm so the workers inherit them. Doing this here rather
    # than via Image.env() keeps the image cache key untouched — adding a layer would
    # force a full vllm reinstall on every script in this repo.
    #   spawn: fork breaks once the parent has touched CUDA, which is how TP>1 dies
    #          during WorkerProc startup
    #   cumem: NCCL's cuMem allocator needs IPC handles that many container runtimes
    #          do not grant; disabling it falls back to a path that works in sandboxes
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

    return {"tp": tp, "throughput": results}


@app.function(gpu="A100-80GB:1", image=image, timeout=1800)
def tp1():
    return bench(1)


@app.function(gpu="A100-80GB:2", image=image, timeout=1800)
def tp2():
    return bench(2)


@app.function(gpu="A100-80GB:4", image=image, timeout=1800)
def tp4():
    return bench(4)


@app.function(gpu="A100-80GB:4", image=image, timeout=600)
def test_tp8_fails():
    """TP is capped by num_key_value_heads (4 for Qwen2.5-7B), not by GPU count."""
    from vllm import LLM
    try:
        LLM(model=MODEL_ID, dtype="float16", tensor_parallel_size=8)
        print("\n[!] TP=8 unexpectedly succeeded — check the model's KV head count.")
    except Exception as e:
        print(f"\n[expected] TP=8 rejected:\n  {type(e).__name__}: {e}")


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("TP_ONLY")          # e.g. TP_ONLY=2 to re-run just that leg
    cases = [(tp1, 1), (tp2, 2), (tp4, 4)]
    if only:
        cases = [c for c in cases if str(c[1]) == only]

    print("TP sweep on Qwen2.5-7B — runs 1, 2, then 4 GPUs sequentially.\n")
    rows = []
    for fn, tp in cases:
        try:
            rows.append(fn.remote())
        except Exception as e:
            print(f"\n!!! TP={tp} failed: {type(e).__name__}: {e}\n")

    if not rows:
        return

    base = rows[0]["throughput"]
    print(f"\n{'='*62}\n  SPEEDUP vs TP=1\n{'='*62}")
    header = "  TP   " + "".join(f"batch={b:<12}" for b in BATCH_SIZES)
    print(header)
    for r in rows:
        line = f"  {r['tp']:<5}"
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]/base[b]:.2f}x".ljust(18)
        print(line)
