"""
Speculative decoding: does it help, and when does it stop helping?

Three configs, each tested at batch=1 (latency-bound) and batch=16 (throughput-bound):
  1. baseline      — plain decode
  2. draft model   — Qwen2.5-0.5B proposes 5 tokens, 7B verifies in one pass
  3. ngram         — propose by looking up repeated n-grams in prompt + output so far

Expected: big win at batch=1 (GPU is memory-bound, spare compute is free),
          shrinking or negative at batch=16 (GPU already compute-saturated,
          rejected tokens become pure waste).

Each config runs in its own Modal invocation so a failure in one
(spec decode APIs move around between vLLM versions) doesn't kill the rest.
"""
import modal
import time

app = modal.App("vllm-spec-decode")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # same tokenizer family — required
N_SPEC_TOKENS = 5
OUTPUT_TOKENS = 256
BATCH_SIZES = [1, 16]

# Summarization-shaped prompts: the answer copies heavily from the input,
# which is the regime where ngram lookup has a real chance of hitting.
DOCUMENT = """The vLLM inference engine introduces PagedAttention, a technique inspired by
virtual memory and paging in operating systems. Instead of allocating one contiguous
buffer per sequence, PagedAttention splits the KV cache into fixed-size blocks that
can live anywhere in GPU memory. A per-sequence block table maps logical positions to
physical blocks. This reduces internal and external fragmentation from 60-80 percent
down to under 4 percent, letting the engine hold far more sequences in the same amount
of memory. More resident sequences means larger batches, and larger batches mean higher
throughput on the decode path, which is otherwise bound by memory bandwidth rather than
compute. vLLM pairs this with continuous batching, where finished sequences leave the
batch immediately and waiting requests take their place, instead of the whole batch
stalling until its slowest member finishes."""

QUESTIONS = [
    "Summarize the passage above in three sentences.",
    "What problem does PagedAttention solve? Quote the relevant numbers.",
    "Explain the operating system analogy used in the passage.",
    "According to the passage, why do larger batches help throughput?",
    "Restate the passage's description of continuous batching.",
    "What fragmentation numbers does the passage give, and for what?",
    "List every technique named in the passage.",
    "How does the block table work, per the passage?",
    "Summarize the passage for someone who knows nothing about GPUs.",
    "What is the decode path bound by, according to the passage?",
    "Rewrite the passage's first two sentences more concisely.",
    "What happens when a sequence finishes, per the passage?",
    "Explain why contiguous allocation is wasteful, per the passage.",
    "Extract the passage's main claim in one sentence.",
    "What does the passage say about memory bandwidth?",
    "Paraphrase the passage's explanation of paging.",
]

SPEC_CONFIGS = {
    "baseline": None,
    "draft_model": {
        "model": DRAFT_ID,
        "num_speculative_tokens": N_SPEC_TOKENS,
    },
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": N_SPEC_TOKENS,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    },
}


def build_prompts(n):
    return [f"{DOCUMENT}\n\n{QUESTIONS[i % len(QUESTIONS)]}" for i in range(n)]


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
    llm.generate(build_prompts(1), SamplingParams(max_tokens=16, temperature=0))

    print(f"\n=== config: {config_name} ===")
    for batch in BATCH_SIZES:
        prompts = build_prompts(batch)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(
            f"  batch={batch:<3} "
            f"wall={elapsed:.2f}s  "
            f"total={total_tokens/elapsed:.1f} tok/s  "
            f"per-req={total_tokens/elapsed/batch:.1f} tok/s  "
            f"TPOT={elapsed/(total_tokens/batch)*1000:.1f}ms"
        )


@app.local_entrypoint()
def main():
    print("Running baseline / draft_model / ngram sequentially...")
    for name in SPEC_CONFIGS:
        try:
            run_config.remote(name)
        except Exception as e:
            # spec decode APIs shift between vLLM versions — keep the other configs alive
            print(f"\n!!! config '{name}' failed: {type(e).__name__}: {e}\n")
