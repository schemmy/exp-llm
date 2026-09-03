import modal
import time

app = modal.App("vllm-batch-sweep")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_TOKENS = 1024
BATCH_SIZES = [1, 4, 8, 16, 32]

PROMPTS = [
    "Explain the concept of gradient descent in machine learning:",
    "What are the main differences between SQL and NoSQL databases:",
    "Describe the architecture of a transformer model:",
    "How does the TCP/IP protocol stack work:",
    "Explain the CAP theorem in distributed systems:",
    "What is the difference between process and thread:",
    "How does a convolutional neural network process images:",
    "Describe the key principles of RESTful API design:",
    "What is the difference between supervised and unsupervised learning:",
    "Explain how HTTPS encryption works:",
    "What are the key features of the Rust programming language:",
    "How does garbage collection work in Java:",
    "Explain the concept of database indexing:",
    "What is the difference between TCP and UDP:",
    "Describe how a hash table works:",
    "What are the key principles of microservices architecture:",
    "Explain the concept of attention mechanism in transformers:",
    "How does Docker containerization work:",
    "What is the difference between REST and GraphQL:",
    "Explain how public key cryptography works:",
    "What are the main types of machine learning algorithms:",
    "How does a compiler work at a high level:",
    "Explain the concept of eventual consistency:",
    "What is the difference between monolithic and distributed systems:",
    "How does the Linux kernel manage processes:",
    "Explain the concept of backpropagation:",
    "What are the key differences between Python and Go:",
    "How does a load balancer work:",
    "Explain the concept of vector embeddings:",
    "What is the difference between OLTP and OLAP systems:",
    "Describe how DNS resolution works:",
    "What are the key principles of functional programming:",
]


@app.function(gpu="A100", image=image, timeout=1200)
def run_sweep():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_ID, dtype="float16")
    params = SamplingParams(max_tokens=N_TOKENS, temperature=0)

    # warmup
    llm.generate([PROMPTS[0]], params)

    print(f"\n{'batch':>6}  {'tok/s':>10}  {'latency/req':>12}  {'total_time':>10}")
    print("-" * 46)

    results = []
    for bs in BATCH_SIZES:
        batch_prompts = PROMPTS[:bs]
        times = []
        for _ in range(2):  # 2 runs, take average
            t0 = time.perf_counter()
            outputs = llm.generate(batch_prompts, params)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        print(outputs[0].outputs[0].text)
        avg_time = sum(times) / len(times)
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / avg_time * (bs / len(batch_prompts))  # normalize
        total_tps = (bs * N_TOKENS) / avg_time
        latency_per_req = avg_time  # wall-clock latency (all reqs finish together)

        print(f"{bs:>6}  {total_tps:>10.1f}  {latency_per_req:>11.2f}s  {avg_time:>9.2f}s")
        results.append((bs, total_tps, latency_per_req))

    print("\nSummary (copy to week04.md):")
    print(f"{'batch':>6}  {'total tok/s':>12}  {'wall latency':>13}")
    for bs, tps, lat in results:
        print(f"{bs:>6}  {tps:>12.1f}  {lat:>12.2f}s")


@app.local_entrypoint()
def main():
    run_sweep.remote()
