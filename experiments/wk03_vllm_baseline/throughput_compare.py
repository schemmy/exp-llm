import modal
import time

app = modal.App("throughput-compare")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm", "transformers", "accelerate")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_TOKENS = 100
N_REQUESTS = 8
PROMPTS = [
    "Explain the concept of gradient descent in machine learning:",
    "What are the main differences between SQL and NoSQL databases:",
    "Describe the architecture of a transformer model:",
    "How does the TCP/IP protocol stack work:",
    "Explain the CAP theorem in distributed systems:",
    "What is the difference between process and thread:",
    "How does a convolutional neural network process images:",
    "Describe the key principles of RESTful API design:",
]


@app.function(gpu="A100", image=image, timeout=600)
def hf_sequential():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).cuda()

    # one warmup run
    inputs = tokenizer(PROMPTS[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False)

    # 8 requests, one at a time
    t0 = time.perf_counter()
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False)
    elapsed = time.perf_counter() - t0

    total_tokens = N_REQUESTS * N_TOKENS
    print(f"HF sequential  ({N_REQUESTS} requests): "
          f"{total_tokens/elapsed:.1f} tok/s  ({elapsed:.1f}s, {total_tokens} tokens)")


@app.function(gpu="A100", image=image, timeout=600)
def vllm_batch():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_ID, dtype="float16")
    params = SamplingParams(max_tokens=N_TOKENS, temperature=0)

    # one warmup run
    llm.generate([PROMPTS[0]], params)

    # 8 requests in one batch call
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"vLLM batch     ({N_REQUESTS} requests): "
          f"{total_tokens/elapsed:.1f} tok/s  ({elapsed:.1f}s, {total_tokens} tokens)")


@app.local_entrypoint()
def main():
    print("=== Throughput comparison: HF sequential vs vLLM batch (8 requests) ===")
    hf_sequential.remote()
    vllm_batch.remote()
