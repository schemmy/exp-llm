"""
Continuous batching vs static batching comparison.

Continuous: requests arrive over time (Poisson), AsyncLLMEngine serves them as they come.
Static:     wait for all requests to arrive, then submit as one batch (LLM.generate).

Metrics: P50/P90/P99 latency, TTFT, TPOT, total throughput.
"""
import modal
import time

app = modal.App("cb-vs-static")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_REQUESTS = 32
AVG_ARRIVAL_INTERVAL = 0.2   # seconds between requests (Poisson mean)
MIN_TOKENS = 50
MAX_TOKENS = 300

PROMPTS = [
    "Explain gradient descent:",
    "What is the CAP theorem:",
    "How does TCP work:",
    "Describe transformer architecture:",
    "What is PagedAttention:",
    "Explain backpropagation:",
    "How does Docker work:",
    "What is eventual consistency:",
    "Explain attention mechanism:",
    "How does DNS resolution work:",
    "What is a hash table:",
    "Explain public key cryptography:",
    "How does garbage collection work:",
    "What is microservices architecture:",
    "Explain the difference between process and thread:",
    "How does a compiler work:",
    "What is vector embedding:",
    "Explain OLTP vs OLAP:",
    "How does load balancing work:",
    "What is the difference between REST and GraphQL:",
    "Explain database indexing:",
    "How does HTTPS work:",
    "What is functional programming:",
    "Explain the Linux kernel scheduler:",
    "How does a neural network learn:",
    "What is the difference between SQL and NoSQL:",
    "Explain convolutional neural networks:",
    "How does Kubernetes work:",
    "What is reinforcement learning:",
    "Explain the concept of recursion:",
    "How does memory management work in C:",
    "What is the difference between supervised and unsupervised learning:",
]


@app.function(gpu="A100", image=image, timeout=600)
def run_continuous():
    import asyncio
    import numpy as np
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    rng = np.random.default_rng(42)
    output_lengths = rng.integers(MIN_TOKENS, MAX_TOKENS + 1, size=N_REQUESTS).tolist()
    arrival_gaps = rng.exponential(AVG_ARRIVAL_INTERVAL, size=N_REQUESTS).tolist()

    engine_args = AsyncEngineArgs(model=MODEL_ID, dtype="float16")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    results = {}  # request_id -> {ttft, finish_time, n_tokens, start_time}

    async def send_request(req_id, prompt, n_tokens, t_send):
        params = SamplingParams(max_tokens=n_tokens, temperature=0)
        t_start = time.perf_counter()
        ttft = None
        n_out = 0
        async for out in engine.generate(prompt, params, request_id=str(req_id)):
            if ttft is None:
                ttft = time.perf_counter() - t_start
            n_out = len(out.outputs[0].token_ids)
            if out.finished:
                finish = time.perf_counter()
                results[req_id] = {
                    "latency": finish - t_start,
                    "ttft": ttft,
                    "tpot": (finish - t_start - ttft) / max(n_out - 1, 1),
                    "n_tokens": n_out,
                }

    async def main():
        tasks = []
        t0 = time.perf_counter()
        for i in range(N_REQUESTS):
            if i > 0:
                await asyncio.sleep(arrival_gaps[i])
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(asyncio.create_task(
                send_request(i, prompt, output_lengths[i], time.perf_counter() - t0)
            ))
        await asyncio.gather(*tasks)

    asyncio.run(main())

    latencies = sorted(r["latency"] for r in results.values())
    ttfts = sorted(r["ttft"] for r in results.values())
    tpots = sorted(r["tpot"] for r in results.values())
    total_tokens = sum(r["n_tokens"] for r in results.values())
    total_time = max(r["latency"] for r in results.values()) + sum(arrival_gaps)

    def pct(lst, p):
        return lst[int(len(lst) * p / 100)]

    print("\n=== Continuous Batching (AsyncLLMEngine) ===")
    print(f"Requests: {N_REQUESTS}, avg arrival: {AVG_ARRIVAL_INTERVAL}s, "
          f"output: {MIN_TOKENS}-{MAX_TOKENS} tokens")
    print(f"\nLatency (request start → last token):")
    print(f"  P50: {pct(latencies,50):.2f}s  P90: {pct(latencies,90):.2f}s  P99: {pct(latencies,99):.2f}s")
    print(f"\nTTFT (time to first token):")
    print(f"  P50: {pct(ttfts,50)*1000:.0f}ms  P90: {pct(ttfts,90)*1000:.0f}ms  P99: {pct(ttfts,99)*1000:.0f}ms")
    print(f"\nTPOT (ms/token after first):")
    print(f"  P50: {pct(tpots,50)*1000:.1f}ms  P90: {pct(tpots,90)*1000:.1f}ms")
    print(f"\nThroughput: {total_tokens/total_time:.1f} tok/s  ({total_tokens} tokens)")


@app.function(gpu="A100", image=image, timeout=600)
def run_static():
    import numpy as np
    from vllm import LLM, SamplingParams

    rng = np.random.default_rng(42)
    output_lengths = rng.integers(MIN_TOKENS, MAX_TOKENS + 1, size=N_REQUESTS).tolist()
    arrival_gaps = rng.exponential(AVG_ARRIVAL_INTERVAL, size=N_REQUESTS).tolist()

    llm = LLM(model=MODEL_ID, dtype="float16")

    # simulate waiting for all requests to arrive before submitting
    collection_time = sum(arrival_gaps)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(N_REQUESTS)]
    params_list = [SamplingParams(max_tokens=n, temperature=0) for n in output_lengths]

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params_list)
    batch_time = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    # each request waited collection_time before batch started, then batch_time to finish
    # (worst case: last-arriving request waited collection_time, then full batch_time)
    total_wall = collection_time + batch_time

    print("\n=== Static Batching (wait-then-submit) ===")
    print(f"Requests: {N_REQUESTS}, collection wait: {collection_time:.1f}s, batch time: {batch_time:.2f}s")
    print(f"\nLatency (simulated — from arrival to finish):")
    # request i arrived at sum(gaps[:i]), finished at collection_time + batch_time
    arrival_times = [sum(arrival_gaps[:i]) for i in range(N_REQUESTS)]
    latencies = [collection_time + batch_time - t for t in arrival_times]
    latencies.sort()
    def pct(lst, p): return lst[int(len(lst) * p / 100)]
    print(f"  P50: {pct(latencies,50):.2f}s  P90: {pct(latencies,90):.2f}s  P99: {pct(latencies,99):.2f}s")
    print(f"\nThroughput: {total_tokens/batch_time:.1f} tok/s  (batch only)")
    print(f"  End-to-end: {total_tokens/total_wall:.1f} tok/s  (incl. collection wait)")


@app.local_entrypoint()
def main():
    print("Submitting both jobs (they run on separate A100s in parallel)...")
    run_continuous.remote()
    run_static.remote()
