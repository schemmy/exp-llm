import modal

app = modal.App("gpu-test")
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="T4", image=image)
def check_gpu():
    import torch
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

@app.local_entrypoint()
def main():
    check_gpu.remote()
