"""
Expert parallelism vs tensor parallelism on the same MoE model.

Both split the model across 2 GPUs, but along different axes:

  TP      slices *each expert's* weight matrices by column/row. Every GPU holds a
          shard of every expert, every expert computation involves every GPU, and
          each layer ends in an all-reduce — same communication shape as a dense
          model.

  TP+EP   hands *whole experts* to individual GPUs. The router decides where each
          token goes, an all-to-all ships tokens to the owning GPU, and a second
          all-to-all brings results back. Each GPU then runs a full dense GEMM
          over the experts it owns.

The trade: EP keeps expert GEMMs large and efficient and stores 1/N of the experts
per GPU, but pays all-to-all and — the real failure mode — becomes hostage to
routing skew. A GPU holding a popular expert stalls everyone else.

At 60 experts this is a close call. At DeepSeek-V3's 256 experts, TP would slice
each expert so thin the GEMMs degenerate, and EP stops being optional.
"""
import modal
import time

app = modal.App("vllm-expert-parallel")

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

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
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


@app.function(gpu="A100-80GB:2", image=image, volumes=VOLUMES, timeout=2400)
def bench(expert_parallel: bool):
    import json
    import os
    import pathlib

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

    from vllm import LLM, SamplingParams

    key = "ep_on" if expert_parallel else "ep_off"
    label = "TP=2 + expert parallel" if expert_parallel else "TP=2 (tensor parallel only)"
    print(f"\n{'='*66}\n  {label}\n{'='*66}")

    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        tensor_parallel_size=2,
        enable_expert_parallel=expert_parallel,
        trust_remote_code=True,
    )
    HF_CACHE.commit()

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
            f"TPOT={elapsed/(total/batch)*1000:.1f}ms"
        )

    out = {"key": key, "label": label, "throughput": results}
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/{key}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  → saved /results/{key}.json")
    return out


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = []
    for key in ("ep_off", "ep_on"):
        p = pathlib.Path(f"/results/{key}.json")
        if p.exists():
            r = json.loads(p.read_text())
            r["throughput"] = {int(k): v for k, v in r["throughput"].items()}
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("EP_ONLY")     # EP_ONLY=on | off
    cases = [(False, "off"), (True, "on")]
    if only:
        cases = [c for c in cases if c[1] == only]

    for ep, name in cases:
        try:
            bench.remote(ep)
        except Exception as e:
            print(f"\n!!! EP={name} failed: {type(e).__name__}: {e}\n")

    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    print(f"\n{'='*70}\n  {'config':<28}"
          + "".join(f"batch={b:<12}" for b in BATCH_SIZES) + f"\n{'='*70}")
    for r in rows:
        line = f"  {r['label']:<28}"
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]:,.1f}".ljust(18)
        print(line)

    by = {r["key"]: r for r in rows}
    if {"ep_off", "ep_on"} <= by.keys():
        print(f"\n  EP vs TP-only:")
        for b in BATCH_SIZES:
            ratio = by["ep_on"]["throughput"][b] / by["ep_off"]["throughput"][b]
            print(f"    batch={b:<3} {ratio:.2f}x")
