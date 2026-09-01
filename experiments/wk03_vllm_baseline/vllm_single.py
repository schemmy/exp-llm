import modal
import time

app = modal.App("vllm-7b-baseline")
image = (
    modal.Image.debian_slim()
    .pip_install("vllm")
    .env({"VLLM_USE_V1": "0"})  # v1 engine uses multiprocessing that breaks in Modal containers
)

@app.function(gpu="A100", image=image, timeout=600)
def run_vllm():
    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    llm = LLM(model=model_id, dtype="float16")

    params = SamplingParams(max_tokens=100, temperature=0)
    prompt = "The key to efficient LLM inference is"

    for i in range(3):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params)
        elapsed = time.perf_counter() - t0
        actual = len(outputs[0].outputs[0].token_ids)
        print(f"Run {i+1}: {actual/elapsed:.1f} tok/s  ({elapsed:.2f}s, {actual} tokens)")
        print(outputs[0].outputs[0].text)

@app.local_entrypoint()
def main():
    run_vllm.remote()
