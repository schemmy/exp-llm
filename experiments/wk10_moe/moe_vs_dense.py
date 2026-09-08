"""
MoE vs dense: does an MoE really decode at its *active* parameter speed?

Three models from one family, so architecture and tokenizer stay comparable:

  Qwen1.5-1.8B        1.8B total /  1.8B active   <- active-param control
  Qwen1.5-MoE-A2.7B  14.3B total /  2.7B active   <- subject
  Qwen1.5-14B          14B total /   14B active   <- total-param control

The claim under test: memory follows *total* params (every expert must stay
resident, since you cannot know which one the next token routes to) while decode
speed follows *active* params (only the routed experts get read from HBM).

If that holds, the MoE should land near the 1.8B on tok/s and near the 14B on
memory. It will not match the 1.8B exactly — the router costs something, shared
experts fire every step, and MoE grouped-GEMM kernels are less efficient than a
plain dense GEMM. Where it lands between the two controls is the whole result.

Same hardening as wk09: Python 3.12 (flashinfer needs it for any multi-GPU path),
cached weights, per-leg result files so a dropped client costs nothing measured.
"""
import modal
import time

app = modal.App("vllm-moe-vs-dense")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("vllm")
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("moe-results", create_if_missing=True)
VOLUMES = {
    "/root/.cache/huggingface": HF_CACHE,
    "/root/.cache/vllm": VLLM_CACHE,
    "/results": RESULTS,
}

# (key, model id, total params B, active params B)
MODELS = [
    ("dense_1_8b", "Qwen/Qwen1.5-1.8B", 1.8, 1.8),
    ("moe_a2_7b", "Qwen/Qwen1.5-MoE-A2.7B", 14.3, 2.7),
    ("dense_14b", "Qwen/Qwen1.5-14B", 14.0, 14.0),
]

OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 32]

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


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=2400)
def bench_model(key: str, model_id: str, total_b: float, active_b: float):
    import json
    import os
    import pathlib

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

    import torch
    from vllm import LLM, SamplingParams

    print(f"\n{'='*66}\n  {model_id}\n"
          f"  {total_b}B total / {active_b}B active\n{'='*66}")

    llm = LLM(model=model_id, dtype="float16", trust_remote_code=True)
    HF_CACHE.commit()

    # Weight residency is the half of the trade that MoE does *not* improve.
    weight_gib = torch.cuda.memory_allocated() / 1024**3

    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)
    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=16, temperature=0))

    results = {}
    for batch in BATCH_SIZES:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0

        total = sum(len(o.outputs[0].token_ids) for o in outputs)
        results[batch] = total / elapsed
        print(
            f"  batch={batch:<3} wall={elapsed:.2f}s  "
            f"total={total/elapsed:,.1f} tok/s  "
            f"per-req={total/elapsed/batch:.1f} tok/s  "
            f"TPOT={elapsed/(total/batch)*1000:.1f}ms"
        )

    out = {
        "key": key, "model": model_id,
        "total_b": total_b, "active_b": active_b,
        "weight_gib": weight_gib, "throughput": results,
    }
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/{key}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  weights resident: {weight_gib:.2f} GiB  → saved /results/{key}.json")
    return out


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = []
    for key, *_ in MODELS:
        p = pathlib.Path(f"/results/{key}.json")
        if p.exists():
            r = json.loads(p.read_text())
            r["throughput"] = {int(k): v for k, v in r["throughput"].items()}
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("MODEL_ONLY")     # e.g. MODEL_ONLY=moe_a2_7b
    todo = [m for m in MODELS if not only or m[0] == only]

    for key, model_id, total_b, active_b in todo:
        try:
            bench_model.remote(key, model_id, total_b, active_b)
        except Exception as e:
            print(f"\n!!! {key} failed: {type(e).__name__}: {e}\n")

    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    print(f"\n{'='*78}")
    print(f"  {'model':<22}{'total/active':<16}{'weights':<11}"
          + "".join(f"batch={b:<10}" for b in BATCH_SIZES))
    print(f"{'='*78}")
    for r in rows:
        line = f"  {r['key']:<22}"
        line += f"{r['total_b']}B / {r['active_b']}B".ljust(16)
        line += f"{r['weight_gib']:.1f} GiB".ljust(11)
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]:,.1f}".ljust(16)
        print(line)

    # The result is where the MoE sits between its two controls.
    by_key = {r["key"]: r for r in rows}
    if {"dense_1_8b", "moe_a2_7b", "dense_14b"} <= by_key.keys():
        print(f"\n{'='*78}\n  MoE positioned between its controls\n{'='*78}")
        for b in BATCH_SIZES:
            small = by_key["dense_1_8b"]["throughput"][b]
            moe = by_key["moe_a2_7b"]["throughput"][b]
            big = by_key["dense_14b"]["throughput"][b]
            # 0% = as slow as the 14B dense, 100% = as fast as the 1.8B dense
            pos = (moe - big) / (small - big) * 100 if small != big else float("nan")
            print(f"  batch={b:<3} 1.8B {small:,.0f}  |  MoE {moe:,.0f}  |  14B {big:,.0f}"
                  f"   → MoE is {pos:.0f}% of the way to small-dense speed")
