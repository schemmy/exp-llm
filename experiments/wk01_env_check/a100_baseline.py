import modal
import time

app = modal.App("a100-baseline")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate")
)

@app.function(gpu="A100", image=image, timeout=300)
def run_baseline():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "Qwen/Qwen2.5-0.5B"   # deliberately tiny
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()

    prompt = "The key to efficient LLM inference is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    n_tokens = 100

    # 3 runs
    for i in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        elapsed = time.perf_counter() - t0
        tps = n_tokens / elapsed
        print(f"Run {i+1}: {tps:.1f} tok/s  ({elapsed:.2f}s for {n_tokens} tokens)")

@app.local_entrypoint()
def main():
    run_baseline.remote()