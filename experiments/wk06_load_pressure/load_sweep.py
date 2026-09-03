"""
Load pressure sweep: vary request arrival rate, observe TTFT degradation.

At low load: GPU has spare capacity, TTFT stays low.
At saturation: requests queue up, TTFT climbs, P99 blows up.
Past saturation: queue grows unbounded (not tested here — we stop at 64 reqs).
"""
import modal
import time

app = modal.App("vllm-load-sweep")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_TOKENS = 200   # fixed output length for clean comparison

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
    "What is attention:", "Explain transformers:",
    "How does BERT work:", "What is fine-tuning:",
    "Explain tokenization:", "What is beam search:",
    "How does sampling work:", "What is temperature in LLMs:",
    "Explain KV cache:", "What is speculative decoding:",
    "How does quantization work:", "What is LoRA:",
    "Explain RLHF:", "What is constitutional AI:",
    "How does RAG work:", "What is chain of thought:",
    "Explain prompt engineering:", "What is in-context learning:",
    "How does flash attention work:", "What is tensor parallelism:",
    "Explain pipeline parallelism:", "What is model sharding:",
    "How does gradient checkpointing work:", "What is mixed precision:",
    "Explain ZeRO optimization:", "What is data parallelism:",
    "How does CUDA work:", "Explain GPU memory hierarchy:",
    "What is warp divergence:", "How does memory coalescing work:",
    "Explain SIMD:", "What is roofline model:",
]

# Scenarios: (n_requests, avg_interval_seconds, label)
SCENARIOS = [
    (32,  0.20, "low load   (32 req, 0.20s interval, ~5 req/s)"),
    (32,  0.05, "medium load (32 req, 0.05s interval, ~20 req/s)"),
    (64,  0.05, "high load   (64 req, 0.05s interval, ~20 req/s)"),
    (64,  0.02, "heavy load  (64 req, 0.02s interval, ~50 req/s)"),
]


@app.function(gpu="A100", image=image, timeout=1200)
def run_load_sweep():
    import asyncio
    import numpy as np
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    engine_args = AsyncEngineArgs(model=MODEL_ID, dtype="float16")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)

    def pct(lst, p):
        s = sorted(lst)
        return s[int(len(s) * p / 100)]

    async def run_scenario(n_requests, avg_interval, label):
        rng = np.random.default_rng(42)
        gaps = rng.exponential(avg_interval, size=n_requests).tolist()
        results = {}

        async def send(req_id, prompt):
            t0 = time.perf_counter()
            ttft = None
            async for out in engine.generate(prompt, params, request_id=f"{label}-{req_id}"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if out.finished:
                    results[req_id] = {
                        "latency": time.perf_counter() - t0,
                        "ttft": ttft,
                        "n_tokens": len(out.outputs[0].token_ids),
                    }

        tasks = []
        for i in range(n_requests):
            if i > 0:
                await asyncio.sleep(gaps[i])
            tasks.append(asyncio.create_task(send(i, PROMPTS[i % len(PROMPTS)])))
        await asyncio.gather(*tasks)

        latencies = [r["latency"] for r in results.values()]
        ttfts = [r["ttft"] for r in results.values()]
        total_tokens = sum(r["n_tokens"] for r in results.values())
        wall = max(latencies) + sum(gaps)

        print(f"\n--- {label} ---")
        print(f"  Latency  P50: {pct(latencies,50):.2f}s  P90: {pct(latencies,90):.2f}s  P99: {pct(latencies,99):.2f}s")
        print(f"  TTFT     P50: {pct(ttfts,50)*1000:.0f}ms  P90: {pct(ttfts,90)*1000:.0f}ms  P99: {pct(ttfts,99)*1000:.0f}ms")
        print(f"  Throughput: {total_tokens/wall:.1f} tok/s")

    async def main():
        print("\n=== vLLM Load Pressure Sweep ===")
        print(f"Model: {MODEL_ID}, output: {OUTPUT_TOKENS} tokens/req\n")
        out = engine.generate("warmup", SamplingParams(max_tokens=20), request_id="warmup")
        async for _ in out:
            pass

        for n, interval, label in SCENARIOS:
            await run_scenario(n, interval, label)
            await asyncio.sleep(2)  # let engine settle between scenarios

    asyncio.run(main())


@app.local_entrypoint()
def main():
    run_load_sweep.remote()
