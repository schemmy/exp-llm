"""
Roofline ridge point + CUDA graph ablation + MFU/MBU, in one sweep.

Three questions, one batch axis, because they're all reading the same knob
(arithmetic intensity = batch size, for a weight-stationary decode GEMM):

  1. Where does throughput leave the linear (bandwidth-bound) region?
     Predicted ridge: peak_flops / peak_bw = 312e12 / 2039e9 ~= 153 (batch~=M).

  2. How much of TPOT is CUDA graph launch-overhead savings? Launch overhead
     (~300-400 kernels/token, ~5-10us each) is roughly fixed per step while
     compute per step grows with batch, so graph benefit should decay as batch
     grows -- same axis as (1), just a different lens on it.

  3. What do MFU and MBU actually say, batch by batch, next to what
     nvidia-smi's GPU-Util would have said (~100% at every single point,
     which is why that number is not used anywhere in this repo).

Stops at 512 deliberately: that's vLLM's max_cudagraph_capture_size for this
model. Past it, the engine falls back to eager regardless of the
enforce_eager flag, which would confound the sweep with a second variable.

Same hardening as wk09/wk10: shared image/weights cache, per-leg result files.
"""
import modal
import time

app = modal.App("vllm-roofline-sweep")

# Plain pip_install("vllm"), same as wk01-08 -- single GPU only here, so the
# flashinfer/array.array Python-3.12 requirement from wk09 (TP>1 only) does
# not apply. Sharing this image and the hf-cache volume with earlier weeks
# means the Qwen2.5-7B weights are already resident from wk09/wk10.
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

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]

# A100-80GB SXM specs used throughout this repo's roofline discussion.
PEAK_FLOPS = 312e12   # fp16 dense tensor-core FLOPS
PEAK_BW = 2039e9      # HBM2e bytes/sec
# Qwen2.5-7B-Instruct, per the model card. Used for MFU; MBU instead uses the
# weight size measured on disk at runtime (see bench()), same approach as wk10.
PARAMS_B = 7.61e9

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


def bench(engine_key: str, enforce_eager: bool):
    import glob
    import json
    import os
    import pathlib

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    label = "eager (no CUDA graph)" if enforce_eager else "default (CUDA graph)"
    print(f"\n{'='*66}\n  {label}\n{'='*66}")

    llm = LLM(model=MODEL_ID, dtype="float16", enforce_eager=enforce_eager)
    HF_CACHE.commit()

    weight_gib = 0.0
    try:
        files = glob.glob(
            f"/root/.cache/huggingface/**/models--{MODEL_ID.replace('/', '--')}"
            "/**/*.safetensors", recursive=True)
        weight_gib = sum(os.path.getsize(f) for f in files) / 1024**3
    except Exception as e:
        print(f"(weight size unavailable: {e})")
    weight_bytes = weight_gib * 1024**3

    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)
    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=16, temperature=0))

    results = {}
    for batch in BATCH_SIZES:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0

        total = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total / elapsed
        steps_per_sec = tps / batch     # one decode step emits `batch` tokens

        # MFU: (FLOPs needed per second) / (peak FLOPs per second)
        # 2x params is one multiply-add per weight per token -- see the FLOPs
        # note from this week's chat.
        mfu = (2 * PARAMS_B * tps) / PEAK_FLOPS

        # MBU: (bytes read per second) / (peak bytes per second). Weight-stationary
        # decode reads the full weight set once per step, not once per token.
        mbu = (weight_bytes * steps_per_sec) / PEAK_BW if weight_bytes else None

        results[batch] = {"tok_s": tps, "mfu": mfu, "mbu": mbu}
        mbu_str = f"{mbu*100:.1f}%" if mbu is not None else "n/a"
        print(
            f"  batch={batch:<4} {tps:>9,.1f} tok/s   "
            f"MFU={mfu*100:>6.2f}%   MBU={mbu_str:>7}"
        )

    out = {"key": engine_key, "enforce_eager": enforce_eager,
           "weight_gib": weight_gib, "results": results}
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/{engine_key}.json").write_text(json.dumps(out))
    RESULTS.commit()
    print(f"  → saved /results/{engine_key}.json")
    return out


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=3600)
def graph_sweep():
    return bench("graph", enforce_eager=False)


@app.function(gpu="A100-80GB:1", image=image, volumes=VOLUMES, timeout=3600)
def eager_sweep():
    return bench("eager", enforce_eager=True)


@app.function(image=image, volumes=VOLUMES, timeout=120)
def collect():
    import json
    import pathlib
    rows = []
    for key in ("graph", "eager"):
        p = pathlib.Path(f"/results/{key}.json")
        if p.exists():
            r = json.loads(p.read_text())
            r["results"] = {int(k): v for k, v in r["results"].items()}
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import os
    only = os.environ.get("ENGINE_ONLY")     # graph | eager
    cases = [(graph_sweep, "graph"), (eager_sweep, "eager")]
    if only:
        cases = [c for c in cases if c[1] == only]

    for fn, name in cases:
        try:
            fn.remote()
        except Exception as e:
            print(f"\n!!! {name} failed: {type(e).__name__}: {e}\n")

    rows = collect.remote()
    if not rows:
        print("No results on the volume yet.")
        return

    by = {r["key"]: r for r in rows}
    print(f"\n{'='*90}\n  ROOFLINE + CUDA GRAPH + MFU/MBU\n{'='*90}")
    header = f"  {'batch':<7}{'tok/s (graph)':<16}{'MFU':<9}{'MBU':<9}"
    if "eager" in by:
        header += f"{'tok/s (eager)':<16}{'graph gain':<12}"
    print(header)

    ref = by.get("graph")
    if not ref:
        for r in rows:
            print(f"  {r['key']}: {r['results']}")
        return

    for b in BATCH_SIZES:
        g = ref["results"][b]
        line = f"  {b:<7}{g['tok_s']:<16,.1f}{g['mfu']*100:<9.2f}"
        line += f"{g['mbu']*100:<9.1f}" if g["mbu"] is not None else f"{'n/a':<9}"
        if "eager" in by:
            e = by["eager"]["results"][b]
            gain = g["tok_s"] / e["tok_s"]
            line += f"{e['tok_s']:<16,.1f}{gain:.2f}x"
        print(line)

    # Ridge-point check: throughput should scale ~linearly with batch while
    # arithmetic intensity (== batch) stays under ~153; report the deviation.
    print(f"\n{'='*90}\n  LINEARITY vs BATCH=1 (ridge point predicted near batch=153)\n{'='*90}")
    base = ref["results"][1]["tok_s"]
    for b in BATCH_SIZES:
        actual = ref["results"][b]["tok_s"]
        linear_prediction = base * b
        pct_of_linear = actual / linear_prediction * 100
        print(f"  batch={b:<4} actual={actual:>9,.1f}  "
              f"linear_predicted={linear_prediction:>10,.1f}  "
              f"{pct_of_linear:>6.1f}% of linear")
