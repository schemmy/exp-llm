"""Inspect a training step, not a peak-throughput benchmark.

Run: modal run --detach experiments/wk15_training_baseline/train_step.py::bench
Results: Modal volume training-baseline-results, one JSON per run.
Pinned educational baseline: PyTorch 2.6.0 / CUDA 12.4 / FP32, no compile/AMP.
"""
import modal

app = modal.App("training-step-baseline")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124"
)
results = modal.Volume.from_name("training-baseline-results", create_if_missing=True)
CONFIG = dict(vocab=4096, width=256, heads=4, layers=4, sequence=256,
              batch=8, seed=42, lr=0.001, warmup=3, repeats=10)


def build_model(config):
    import torch
    from torch import nn

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(config["vocab"], config["width"])
            self.positions = nn.Embedding(config["sequence"], config["width"])
            # Separate constructors give each layer its own initialization.
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    config["width"], config["heads"], 4 * config["width"],
                    dropout=0.0, batch_first=True, norm_first=True,
                ) for _ in range(config["layers"])
            ])
            self.norm = nn.LayerNorm(config["width"])
            self.head = nn.Linear(config["width"], config["vocab"], bias=False)

        def forward(self, tokens):
            size = tokens.shape[1]
            x = self.tokens(tokens) + self.positions(torch.arange(size, device=tokens.device))
            mask = torch.ones(size, size, dtype=torch.bool, device=tokens.device).triu(1)
            for layer in self.layers:
                x = layer(x, src_mask=mask)
            return self.head(self.norm(x))

    return TinyLM()


@app.function(image=image, gpu="A100-80GB:1", volumes={"/results": results}, timeout=900)
def bench():
    import json
    import platform
    import statistics
    import subprocess
    import time
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Keep the no-grad control on the same Transformer implementation.
    torch.backends.mha.set_fastpath_enabled(False)
    model = build_model(CONFIG).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], foreach=False, fused=False)
    data = torch.randint(CONFIG["vocab"], (CONFIG["batch"], CONFIG["sequence"] + 1), device="cuda")
    inputs, targets = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    del data

    def nbytes(tensors):
        return sum(t.numel() * t.element_size() for t in tensors)

    def snapshot():
        return dict(allocated=torch.cuda.memory_allocated(),
                    reserved=torch.cuda.memory_reserved())

    def state_bytes():
        return nbytes(v for state in optimizer.state.values() for v in state.values()
                      if isinstance(v, torch.Tensor) and v.is_cuda)

    def measure(name, fn):
        torch.cuda.synchronize()
        before = snapshot()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        value = fn()
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        after = snapshot()
        row = dict(stage=name, cuda_ms=start.elapsed_time(end), wall_ms=wall_ms,
                   before=before, after=after,
                   peak_allocated=torch.cuda.max_memory_allocated(),
                   peak_reserved=torch.cuda.max_memory_reserved(),
                   gradient_bytes=nbytes(p.grad for p in model.parameters() if p.grad is not None),
                   optimizer_cuda_bytes=state_bytes())
        row["allocated_delta"] = after["allocated"] - before["allocated"]
        return value, row

    def loss_fn():
        logits = model(inputs)
        return F.cross_entropy(logits.reshape(-1, CONFIG["vocab"]), targets.reshape(-1))

    def no_grad_loss():
        # Same input, weights, train mode and dropout=0; only autograd differs.
        with torch.no_grad():
            return loss_fn()

    initial = snapshot()
    parameter_bytes = nbytes(model.parameters())
    control = []
    for _ in range(CONFIG["warmup"] + CONFIG["repeats"]):
        value, row = measure("no_grad_forward_loss", no_grad_loss)
        control.append(row)
        del value

    runs = []
    checks = {}
    for step in range(1 + CONFIG["warmup"] + CONFIG["repeats"]):
        rows = []
        if step == 0:
            # CPU reference avoids contaminating measured GPU allocations.
            reference = model.head.weight.detach().cpu().clone()
            with torch.no_grad():
                expected_loss = loss_fn().item()
        loss, row = measure("forward_loss", loss_fn)
        rows.append(row)
        loss_value = loss.item()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")
        _, row = measure("backward", loss.backward)
        rows.append(row)
        if step == 0:
            grads = [p.grad for p in model.parameters()]
            checks["all_gradients_present_and_finite"] = all(
                g is not None and torch.isfinite(g).all().item() for g in grads
            )
            checks["no_grad_loss_matches"] = abs(loss_value - expected_loss) < 1e-5
            del grads
        _, row = measure("optimizer_step", optimizer.step)
        rows.append(row)
        if step == 0:
            checks["head_weights_changed"] = not torch.equal(reference, model.head.weight.detach().cpu())
            del reference
            if not all(checks.values()):
                raise RuntimeError(f"Training checks failed: {checks}")
        _, row = measure("zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
        rows.append(row)
        del loss  # Also release the scalar loss and its exhausted graph.
        runs.append(dict(step=step, loss=loss_value, stages=rows, after_cleanup=snapshot()))

    steady = runs[1 + CONFIG["warmup"]:]
    summary = []
    for index in range(4):
        samples = [r["stages"][index] for r in steady]
        summary.append(dict(
            stage=samples[0]["stage"],
            median_cuda_ms=statistics.median(r["cuda_ms"] for r in samples),
            median_wall_ms=statistics.median(r["wall_ms"] for r in samples),
            median_after_allocated=statistics.median(r["after"]["allocated"] for r in samples),
            max_peak_allocated=max(r["peak_allocated"] for r in samples),
        ))
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False).stdout.strip()
    report = dict(
        config=CONFIG, timestamp=datetime.now(timezone.utc).isoformat(),
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                         python=platform.python_version(), modal=modal.__version__,
                         gpu=torch.cuda.get_device_name(), driver=driver,
                         dtype="float32", tf32=False, optimizer="AdamW", foreach=False, fused=False),
        parameters=sum(p.numel() for p in model.parameters()), parameter_bytes=parameter_bytes,
        initial=initial, no_grad_control=control, training=runs, steady_summary=summary,
        checks=checks,
        notes=["Memory is bytes; times are milliseconds. Allocated != reserved != nvidia-smi.",
               "Stage synchronization is diagnostic overhead; not an optimized throughput benchmark.",
               "Forward allocation delta includes loss intermediates; it is not pure activation size.",
               "First training step follows no-grad warmup but includes first backward/Adam allocation.",
               "FP32 synthetic next-token task; no AMP, KV-cache decode, quality evaluation or checkpoint.",
               "No-grad control precedes Adam state allocation; compare within-stage deltas/peaks with care."],
    )
    path = Path("/results") / f"train_step_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    results.commit()

    print(f"\nModel: {report['parameters']:,} parameters | FP32 | AdamW | {report['environment']['gpu']}")
    print("All memory columns below are MiB. End = live allocated, not reserved.")
    for title, rows in [("FIRST TRAINING STEP", runs[0]["stages"]),
                        ("LAST MEASURED STEP", runs[-1]["stages"])]:
        print(f"\n{title}\n  stage                 CUDA ms   wall ms   end MiB   peak MiB   grad MiB   Adam MiB")
        for r in rows:
            print(f"  {r['stage']:<21}{r['cuda_ms']:8.2f}{r['wall_ms']:10.2f}"
                  f"{r['after']['allocated']/2**20:10.1f}{r['peak_allocated']/2**20:11.1f}"
                  f"{r['gradient_bytes']/2**20:11.1f}{r['optimizer_cuda_bytes']/2**20:11.1f}")
    print("\nSTEADY MEDIANS (10 steps, after first step + 3 warmups)")
    for r in summary:
        print(f"  {r['stage']:<21}CUDA {r['median_cuda_ms']:.2f} ms | wall {r['median_wall_ms']:.2f} ms")
    print(f"  no-grad forward/loss: {statistics.median(r['cuda_ms'] for r in control[CONFIG['warmup']:]):.2f} ms CUDA")
    print(f"Checks: {checks}\nSaved {path} on volume training-baseline-results")
    return str(path)
