"""
fp16 vs FP8 vs INT8, at batch sizes chosen to straddle the roofline ridge.

The prediction from this week's ridge-point sweep: quantization halves the
weight bytes moved per decode step, so its payoff should track where you sit
on the roofline. Left of the ridge (bandwidth-bound, batch=1) that halving
should show up almost directly as a speedup. Right of the ridge (compute-bound,
batch=256, near the measured M~153 turnover) the model is no longer waiting on
weight traffic, so the same halving should buy far less.

Two quantization paths, both loadable directly from the fp16 HF checkpoint
(no separate pre-quantized model needed):

  fp8   vLLM's on-the-fly FP8 weight quantization (quantization="fp8").
        A100 is sm80 -- no native FP8 tensor cores (those start at sm89,
        Ada/Hopper) -- so this is a storage-format quantization: half the
        bytes move over HBM, then upconvert to fp16 for the actual matmul.
        The win this measures is bandwidth, not compute.

  int8  bitsandbytes on-the-fly INT8 quantization (quantization="bitsandbytes",
        load_format="bitsandbytes"). Same idea, 8-bit storage instead of 8-bit
        float.

This script uses its own image (adds bitsandbytes), separate from the
vllm-only image the rest of the repo shares -- consistent with wk09/10, which
already diverged (Python 3.12) for their own reasons. It still mounts the
shared hf-cache volume, so the fp16 checkpoint download is free.
"""
import modal
import time

app = modal.App("vllm-quant-compare")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm", "bitsandbytes")
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("roofline-results", create_if_missing=True)
VOLUMES = {
    "/root/.cache/huggingface": HF_CACHE,
    "/root/.cache/vllm": VLLM_CACHE,
    "/results": RESULTS,
}

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_TOKENS = 256
# One point left of the measured ridge (bandwidth-bound), one point at it,
# one point past it (compute-bound) -- see wk11's roofline_sweep.py.
BATCH_SIZES = [1, 32, 256]

PROMPTS = [
    "Explain gradient descent:", "What is the CAP theorem:",
    "How does TCP work:", "Describe transformer architecture:",
    "What is PagedAttention:", "Explain backpropagation:",
    "How does Docker work:", "What is eventual consistency:",
    "Explain attention mechanism:", "How does DNS resolution work:",
    "What is a hash table:", "Explain public key cryptography:",
    "How does garbage collection work:", "What is microservices:",
    "Explain process vs thread:", "How does a compiler work:",
]

# (key, vLLM kwargs)
CONFIGS = {
    "fp16": {},
    "fp8": {"quantization": "fp8"},
    "int8_bnb": {"quantization": "bitsandbytes", "load_format": "bitsandbytes"},
}


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=3600)
def bench(key: str):
    import glob
    import json
    import os
    import pathlib

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    kwargs = dict(model=MODEL_ID, dtype="float16", **CONFIGS[key])
    print(f"\n{'='*66}\n  {key}: {kwargs}\n{'='*66}")

    llm = LLM(**kwargs)
    HF_CACHE.commit()

    weight_gib = 0.0
    try:
        files = glob.glob(
            f"/root/.cache/huggingface/**/models--{MODEL_ID.replace('/', '--')}"
            "/**/*.safetensors", recursive=True)
        weight_gib = sum(os.path.getsize(f) for f in files) / 1024**3
    except Exception as e:
        print(f"(weight size unavailable: {e})")

    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)
    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=16, temperature=0))

    # A short quality spot-check: same greedy prompt, read side by side across
    # configs afterward. Not a rigorous eval -- just enough to catch outright
    # breakage from quantization, which greedy fp16 vs int8/fp8 sometimes shows
    # as repetition or truncation.
    sample = llm.generate(
        ["Explain what a hash table is, in two sentences:"],
        SamplingParams(max_tokens=60, temperature=0),
    )[0].outputs[0].text

    results = {}
    for batch in BATCH_SIZES:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0

        total = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total / elapsed
        results[batch] = tps
        print(f"  batch={batch:<4} {tps:>9,.1f} tok/s")

    out = {"key": key, "weight_gib": weight_gib, "throughput": results, "sample": sample}
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/quant_{key}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  weights: {weight_gib:.2f} GiB  → saved /results/quant_{key}.json")
    return out


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = []
    for key in CONFIGS:
        p = pathlib.Path(f"/results/quant_{key}.json")
        if p.exists():
            r = json.loads(p.read_text())
            r["throughput"] = {int(k): v for k, v in r["throughput"].items()}
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("QUANT_ONLY")     # fp16 | fp8 | int8_bnb
    todo = [k for k in CONFIGS if not only or k == only]

    for key in todo:
        try:
            bench.remote(key)
        except Exception as e:
            print(f"\n!!! {key} failed: {type(e).__name__}: {e}\n")

    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    print(f"\n{'='*70}\n  {'config':<12}{'weights':<11}"
          + "".join(f"batch={b:<10}" for b in BATCH_SIZES) + f"\n{'='*70}")
    for r in rows:
        line = f"  {r['key']:<12}{r['weight_gib']:.1f} GiB".ljust(23)
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]:,.1f}".ljust(16)
        print(line)

    by = {r["key"]: r for r in rows}
    if "fp16" in by:
        print(f"\n  speedup vs fp16 (prediction: shrinks left-to-right across the ridge):")
        for key in by:
            if key == "fp16":
                continue
            print(f"    {key}:")
            for b in BATCH_SIZES:
                ratio = by[key]["throughput"][b] / by["fp16"]["throughput"][b]
                print(f"      batch={b:<4} {ratio:.2f}x")

    print(f"\n  sample outputs (greedy, same prompt) -- eyeball check for breakage:")
    for r in rows:
        print(f"    [{r['key']}] {r['sample'][:150]!r}")
