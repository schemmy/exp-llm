"""
Prefix caching experiment: compare TTFT with and without APC.

Setup: 32 requests share a ~500-token system prompt (long RAG context).
       Each request has a unique question suffix (~10 tokens).

With caching OFF: every request pays full ~500-token prefill cost.
With caching ON:  first request pays full cost; remaining 31 hit cache.
                  Their TTFT drops to the cost of ~10 suffix tokens only.
"""
import modal
import time

app = modal.App("vllm-prefix-cache")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_REQUESTS = 32
OUTPUT_TOKENS = 100

# ~500-token shared system prompt (acts as the "RAG document")
SHARED_PREFIX = """You are an expert assistant helping engineers understand distributed systems.

Here is the reference document you must use to answer questions:

=== DISTRIBUTED SYSTEMS REFERENCE ===

Section 1: Consistency Models
Consistency models define the guarantees a distributed system makes about the order and visibility of operations. Strong consistency (linearizability) ensures all nodes see the same data at the same time, as if there were only one copy. Eventual consistency allows temporary divergence between nodes but guarantees convergence over time. Causal consistency preserves cause-and-effect ordering: if operation A causally precedes B, all nodes see A before B. Sequential consistency requires that operations appear to execute in some sequential order consistent with each process's program order. Read-your-writes consistency guarantees a client always sees its own writes. Monotonic reads ensure a client never reads older values after reading a newer one.

Section 2: Consensus Algorithms
Consensus algorithms allow distributed nodes to agree on a single value despite failures. Paxos (Lamport, 1989) requires a quorum (majority) for both preparing and accepting proposals. Multi-Paxos optimizes by electing a stable leader to reduce message rounds. Raft (Ongaro & Ousterhout, 2014) simplifies Paxos by separating leader election, log replication, and safety into distinct sub-problems. Raft guarantees that a leader always has the most up-to-date log. PBFT (Practical Byzantine Fault Tolerance) handles Byzantine (malicious) failures and requires 3f+1 nodes to tolerate f Byzantine faults. Viewstamped Replication is an alternative to Paxos with similar guarantees.

Section 3: Replication Strategies
Single-leader replication routes all writes through one primary node; replicas follow asynchronously or synchronously. Multi-leader replication allows writes at multiple sites; conflict resolution (last-write-wins, CRDTs, application-level merging) is required. Leaderless replication (Dynamo-style) allows any node to accept writes; quorum reads and writes (R+W>N) ensure consistency. Chain replication routes writes through a chain of servers; reads always go to the tail, providing strong consistency with high read throughput.

Section 4: Partitioning (Sharding)
Hash-based partitioning assigns keys to nodes using a hash function, distributing load evenly but making range queries expensive. Range-based partitioning keeps adjacent keys together for efficient scans but can cause hotspots. Consistent hashing minimizes key remapping when nodes are added or removed. Virtual nodes (vnodes) improve load balance by assigning each physical node multiple positions on the ring. Directory-based partitioning uses a lookup service to track key locations, allowing flexible remapping at the cost of a single point of failure.

Section 5: Fault Tolerance
Timeouts and heartbeats detect node failures; setting the right timeout is critical—too short causes false positives, too long delays detection. Fencing tokens prevent split-brain: a node that is considered dead must not act on stale leases. Circuit breakers prevent cascading failures by stopping calls to a repeatedly failing service. Bulkheads isolate failures within bounded resource pools. Chaos engineering proactively injects failures to validate resilience.

=== END OF REFERENCE DOCUMENT ===

Answer the following question based only on the reference document above. Be concise and precise.

Question: """

QUESTIONS = [
    "What is linearizability?",
    "How does Raft differ from Paxos?",
    "What is eventual consistency?",
    "Explain quorum reads and writes.",
    "What is consistent hashing?",
    "How does chain replication work?",
    "What is causal consistency?",
    "How many nodes does PBFT require?",
    "What is a fencing token?",
    "Explain virtual nodes.",
    "What is the difference between leader and leaderless replication?",
    "What causes hotspots in range partitioning?",
    "How do circuit breakers prevent cascading failures?",
    "What is monotonic reads?",
    "Explain multi-leader conflict resolution.",
    "What is the role of heartbeats in fault detection?",
    "What is sequential consistency?",
    "How does Raft elect a leader?",
    "What is chaos engineering?",
    "Explain hash-based partitioning.",
    "What is read-your-writes consistency?",
    "How does single-leader replication work?",
    "What is a bulkhead?",
    "Explain the Raft log replication process.",
    "What is directory-based partitioning?",
    "How does Multi-Paxos reduce message rounds?",
    "What is the Byzantine fault tolerance requirement?",
    "Explain range-based partitioning.",
    "What is viewstamped replication?",
    "How does consistent hashing minimize remapping?",
    "What is the tradeoff of synchronous replication?",
    "Explain the split-brain problem.",
]


@app.function(gpu="A100", image=image, timeout=600)
def run_experiment(enable_cache: bool):
    import asyncio
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        dtype="float16",
        enable_prefix_caching=enable_cache,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0)

    results = {}

    async def send(req_id, prompt):
        t0 = time.perf_counter()
        ttft = None
        async for out in engine.generate(prompt, params, request_id=str(req_id)):
            if ttft is None:
                ttft = time.perf_counter() - t0
            if out.finished:
                results[req_id] = {
                    "ttft": ttft,
                    "latency": time.perf_counter() - t0,
                    "n_tokens": len(out.outputs[0].token_ids),
                }

    async def main():
        # warmup
        wout = engine.generate("hello", SamplingParams(max_tokens=5), request_id="warmup")
        async for _ in wout:
            pass

        # send all requests concurrently (simulate burst)
        prompts = [SHARED_PREFIX + q for q in QUESTIONS[:N_REQUESTS]]
        tasks = [asyncio.create_task(send(i, p)) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)

    asyncio.run(main())

    def pct(lst, p):
        s = sorted(lst)
        return s[int(len(s) * p / 100)]

    ttfts = [r["ttft"] for r in results.values()]
    latencies = [r["latency"] for r in results.values()]
    total_tokens = sum(r["n_tokens"] for r in results.values())
    wall = max(latencies)

    label = "ON " if enable_cache else "OFF"
    print(f"\n=== Prefix Caching {label} ===")
    print(f"  TTFT    P50: {pct(ttfts,50)*1000:.0f}ms  P90: {pct(ttfts,90)*1000:.0f}ms  P99: {pct(ttfts,99)*1000:.0f}ms")
    print(f"  Latency P50: {pct(latencies,50):.2f}s  P90: {pct(latencies,90):.2f}s  P99: {pct(latencies,99):.2f}s")
    print(f"  Throughput: {total_tokens/wall:.1f} tok/s")


@app.local_entrypoint()
def main():
    print("Running prefix cache OFF then ON (separate A100s)...")
    # run sequentially so results are easy to read
    run_experiment.remote(enable_cache=False)
    run_experiment.remote(enable_cache=True)
