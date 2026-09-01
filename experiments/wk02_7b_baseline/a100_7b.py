import modal
import time

app = modal.App("a100-7b-baseline")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate")
)

@app.function(gpu="A100", image=image, timeout=600)
def run_7b():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16
    ).cuda()

    prompt = "The key to efficient LLM inference is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    n_tokens = 100

    for i in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        elapsed = time.perf_counter() - t0
        print(f"Run {i+1}: {n_tokens/elapsed:.1f} tok/s  ({elapsed:.2f}s)")

@app.local_entrypoint()
def main():
    run_7b.remote()
