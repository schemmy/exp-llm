"""
Follow-up to spec_decode.py: isolate acceptance rate as the variable.

spec_decode.py measured ngram on a summarization task and got only +8% at batch=1.
Hypothesis: the gain was small because the model writes *new* prose, so n-gram
lookup into the prompt rarely matches — low acceptance rate.

This script keeps the engine identical and changes only the task shape:
  - "novel"  : explain/summarize — output is newly composed  (low acceptance expected)
  - "copy"   : reproduce the passage verbatim with small edits (high acceptance expected)

Same baseline-vs-ngram comparison on both. If the copy-heavy task shows a much
larger ngram win, that pins the speedup to acceptance rate rather than to the engine.
"""
import modal
import time

app = modal.App("vllm-ngram-acceptance")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_SPEC_TOKENS = 5
OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 16]

PASSAGE = """The vLLM inference engine introduces PagedAttention, a technique inspired by
virtual memory and paging in operating systems. Instead of allocating one contiguous
buffer per sequence, PagedAttention splits the KV cache into fixed-size blocks that
can live anywhere in GPU memory. A per-sequence block table maps logical positions to
physical blocks. This reduces internal and external fragmentation from 60-80 percent
down to under 4 percent, letting the engine hold far more sequences in the same amount
of memory."""

# Two task shapes over the SAME passage — only the instruction differs.
TASKS = {
    # Output is newly composed prose. n-gram lookup rarely matches.
    "novel": [
        "Explain the passage above in your own words, without reusing its phrasing.",
        "What problem does this solve? Answer in fresh wording.",
        "Describe the operating system analogy differently than the passage does.",
        "Rewrite the core idea for a reader who knows nothing about GPUs.",
    ],
    # Output copies the input almost verbatim. n-gram lookup should hit constantly.
    "copy": [
        "Reproduce the passage above exactly as written, changing nothing.",
        "Repeat the passage above verbatim, but replace every occurrence of "
        "'PagedAttention' with 'PagedAttn'. Leave all other text untouched.",
        "Output the passage above word for word, but capitalize the word 'blocks' "
        "everywhere it appears. Change nothing else.",
        "Copy the passage above exactly, then append one short sentence at the end.",
    ],
}

SPEC_CONFIGS = {
    "baseline": None,
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": N_SPEC_TOKENS,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    },
}


def build_prompts(task_kind, n):
    qs = TASKS[task_kind]
    return [f"{PASSAGE}\n\n{qs[i % len(qs)]}" for i in range(n)]


@app.function(gpu="A100", image=image, timeout=1800)
def run_config(config_name: str):
    from vllm import LLM, SamplingParams

    spec = SPEC_CONFIGS[config_name]
    kwargs = dict(model=MODEL_ID, dtype="float16")
    if spec is not None:
        kwargs["speculative_config"] = spec

    llm = LLM(**kwargs)
    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)

    # warmup — first call pays torch.compile + CUDA graph capture
    llm.generate(build_prompts("novel", 1), SamplingParams(max_tokens=16, temperature=0))

    print(f"\n=== config: {config_name} ===")
    for task_kind in TASKS:
        for batch in BATCH_SIZES:
            prompts = build_prompts(task_kind, batch)
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed = time.perf_counter() - t0

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            print(
                f"  task={task_kind:<6} batch={batch:<3} "
                f"wall={elapsed:.2f}s  "
                f"total={total_tokens/elapsed:.1f} tok/s  "
                f"per-req={total_tokens/elapsed/batch:.1f} tok/s  "
                f"TPOT={elapsed/(total_tokens/batch)*1000:.1f}ms"
            )


@app.local_entrypoint()
def main():
    print("Running baseline then ngram over novel + copy tasks...")
    for name in SPEC_CONFIGS:
        run_config.remote(name)
