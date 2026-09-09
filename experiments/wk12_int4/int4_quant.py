"""
INT4 (AWQ vs GPTQ) vs the fp16/FP8/INT8 results from wk11.

wk11 found FP8 delivers a clean bandwidth-region win that flips to a loss past
the ridge, while `int8_per_channel_weight_only` barely moves the needle
anywhere -- not even at batch=1, deep in the bandwidth-bound region. The open
question: is that an INT8-the-format problem, or a that-specific-kernel
problem?

This week tests the alternative kernel path: pre-quantized checkpoints (Qwen's
official AWQ / GPTQ-Int4 releases) loaded through vLLM's Marlin kernels
(awq_marlin / gptq_marlin), which are the best-optimized INT4xfp16 GEMM path
vLLM has for Ampere+. If INT4-via-Marlin shows a real speedup where wk11's
INT8 didn't, that points at kernel quality, not the quantization format
itself, as the deciding factor.

Also fixes a measurement gap from wk11: these are genuinely pre-quantized
checkpoints on disk (not quantized-on-load from an fp16 checkpoint), so the
weight_gib probe here measures something real.

Same batch points as wk11's quant_compare.py (1/32/256) for direct
comparability, plus a lightweight quality check: 5 fixed prompts, greedy
decode, character-level similarity against a fresh fp16 reference (not a
rigorous eval -- just a more systematic signal than eyeballing one prompt).
"""
import modal
import time

app = modal.App("vllm-int4-quant")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("roofline-results", create_if_missing=True)
VOLUMES = {
    "/root/.cache/huggingface": HF_CACHE,
    "/root/.cache/vllm": VLLM_CACHE,
    "/results": RESULTS,
}

OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 32, 256]

# (key, model id, vLLM kwargs)
CONFIGS = {
    "fp16_ref": ("Qwen/Qwen2.5-7B-Instruct", {}),
    "awq_int4": ("Qwen/Qwen2.5-7B-Instruct-AWQ", {"quantization": "awq_marlin"}),
    "gptq_int4": ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", {"quantization": "gptq_marlin"}),
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
]

# Fixed set used for the quality/similarity check across configs.
QUALITY_PROMPTS = [
    "Explain what a hash table is, in two sentences:",
    "What is the difference between TCP and UDP?",
    "Summarize how garbage collection works in three sentences:",
    "What is the CAP theorem? Answer concisely:",
    "Explain backpropagation to a beginner in two sentences:",
]


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=3600)
def bench(key: str):
    import glob
    import json
    import os
    import pathlib

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    model_id, kwargs = CONFIGS[key]
    llm_kwargs = dict(model=model_id, dtype="float16", **kwargs)
    print(f"\n{'='*66}\n  {key}: {llm_kwargs}\n{'='*66}")

    llm = LLM(**llm_kwargs)
    HF_CACHE.commit()

    # Unlike wk11's fp8/int8 (quantized on load from an fp16 checkpoint, so
    # disk size was always 14.2 GiB regardless), these repos are genuinely
    # pre-quantized -- this measurement means something this time.
    weight_gib = 0.0
    try:
        files = glob.glob(
            f"/root/.cache/huggingface/**/models--{model_id.replace('/', '--')}"
            "/**/*.safetensors", recursive=True)
        weight_gib = sum(os.path.getsize(f) for f in files) / 1024**3
    except Exception as e:
        print(f"(weight size unavailable: {e})")

    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)
    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=16, temperature=0))

    # Quality check: fixed prompts, greedy, saved for cross-config comparison
    # in collect() (character-level similarity needs both sides present).
    quality_outputs = [
        o.outputs[0].text
        for o in llm.generate(QUALITY_PROMPTS, SamplingParams(max_tokens=150, temperature=0))
    ]

    # fp16_ref only exists to produce a matching-prompt baseline for the
    # similarity check -- the throughput numbers for fp16 are already
    # established in wk11 at these exact batch points, so skip the sweep here.
    results = {}
    if key != "fp16_ref":
        for batch in BATCH_SIZES:
            prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed = time.perf_counter() - t0
            total = sum(len(o.outputs[0].token_ids) for o in outputs)
            results[batch] = total / elapsed
            print(f"  batch={batch:<4} {total/elapsed:>9,.1f} tok/s")

    out = {
        "key": key, "model": model_id, "weight_gib": weight_gib,
        "throughput": results, "quality_outputs": quality_outputs,
    }
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/int4_{key}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  weights: {weight_gib:.2f} GiB  → saved /results/int4_{key}.json")
    return out


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = []
    for key in CONFIGS:
        p = pathlib.Path(f"/results/int4_{key}.json")
        if p.exists():
            r = json.loads(p.read_text())
            r["throughput"] = {int(k): v for k, v in r["throughput"].items()}
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("QUANT_ONLY")     # fp16_ref | awq_int4 | gptq_int4
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

    by = {r["key"]: r for r in rows}

    print(f"\n{'='*70}\n  {'config':<14}{'weights':<12}"
          + "".join(f"batch={b:<10}" for b in BATCH_SIZES) + f"\n{'='*70}")
    for r in rows:
        if not r["throughput"]:
            print(f"  {r['key']:<14}{r['weight_gib']:.2f} GiB".ljust(26) + "(reference only, no sweep)")
            continue
        line = f"  {r['key']:<14}{r['weight_gib']:.2f} GiB".ljust(26)
        for b in BATCH_SIZES:
            line += f"{r['throughput'][b]:,.1f}".ljust(16)
        print(line)

    # wk11's fp16 numbers at these exact batch points, for the speedup ratio.
    FP16_WK11 = {1: 93.8, 32: 2835.1, 256: 10659.4}
    print(f"\n  speedup vs wk11 fp16 baseline:")
    for r in rows:
        if not r["throughput"]:
            continue
        print(f"    {r['key']}:")
        for b in BATCH_SIZES:
            ratio = r["throughput"][b] / FP16_WK11[b]
            print(f"      batch={b:<4} {ratio:.2f}x")

    if "fp16_ref" in by:
        import difflib
        ref_texts = by["fp16_ref"]["quality_outputs"]
        print(f"\n  quality similarity vs fp16 (char-level, 5 fixed prompts):")
        for r in rows:
            if r["key"] == "fp16_ref":
                continue
            sims = [
                difflib.SequenceMatcher(None, ref, cand).ratio()
                for ref, cand in zip(ref_texts, r["quality_outputs"])
            ]
            avg = sum(sims) / len(sims)
            print(f"    {r['key']}: avg={avg:.2f}  per-prompt={[f'{s:.2f}' for s in sims]}")

    print(f"\n  sample outputs (greedy, same prompt) -- eyeball check:")
    for r in rows:
        print(f"    [{r['key']}] {r['quality_outputs'][0][:150]!r}")
