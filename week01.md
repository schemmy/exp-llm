# Week 1 — Set up GPU dev environment

**Dates**: 2026-08-30 → 2026-09-05
**Goal**: End the week with a reproducible dev container that runs `python inference.py` on a rented GPU and prints a token/sec number for *any* Hugging Face model. Nothing fancy — just proof the plumbing works.

**Time budget**: 4-6 hours across evenings + one weekend session.

---

## Tasks

### Task 1 — Modal account + first job (60 min)

- [x] Sign up at https://modal.com (uses GitHub OAuth). Get your $30 free credit.
- [x] Install CLI: `pip install modal && modal setup`
- [x] Run their hello-world example (their landing docs walk you through). Should complete in <1 min.
- [x] Run a GPU hello-world:
  ```python
  import modal
  app = modal.App("gpu-test")
  image = modal.Image.debian_slim().pip_install("torch")
  @app.function(gpu="T4", image=image)
  def check_gpu():
      import torch
      print(torch.cuda.get_device_name(0))
      print(torch.cuda.get_device_properties(0))
  ```
  Confirm you see the T4 name print. Cost so far: ~$0.05.

**Success**: Modal shows the T4 job succeeded in the dashboard. ✅ Done 2026-08-30.

---

### Task 2 — Rented A100, load a small model (2 hrs)

- [ ] Modify the above to use `gpu="A100"` and load a small model (Qwen 2.5 0.5B or Llama 3.2 1B — deliberately tiny for iteration speed)
- [ ] Use HuggingFace `transformers`, generate 100 tokens from a fixed prompt, print the tokens/sec
- [ ] Repeat 3 times, note variance

**Success**: You get a tokens/sec number for a 1B model on A100. Doesn't need to be optimized — this is the CPU-baseline equivalent, we're just wiring things up.

**Note**: This will cost ~$5-10 depending on how much you iterate.

---

### Task 3 — Create the flagship repo (30 min)

- [ ] `mkdir ~/projects/vllm-benchmarks && cd $_ && git init`
- [ ] Add README.md with a stub:
  ```markdown
  # LLM Inference Benchmarks
  
  Reproducible benchmarks for LLM serving on a single GPU.
  Comparing HuggingFace transformers baseline vs vLLM with continuous
  batching, PagedAttention, prefix caching, and quantization.
  
  **Status**: Week 1 of 12 — dev environment setup.
  ```
- [ ] Commit the modal T4/A100 test scripts under `experiments/wk01_env_check/`
- [ ] Push to a new public GitHub repo (mle-transition or vllm-benchmarks — your call)
- [ ] Add link to `~/projects/mle-transition/progress.md`

**Success**: Repo is public, has a real commit, README states the 12-week plan.

---

### Task 4 — Read + skim (60-90 min)

- [ ] Skim the vLLM landing docs: https://docs.vllm.ai
- [ ] Skim PagedAttention section of the vLLM paper (https://arxiv.org/abs/2309.06180) — deep-read scheduled for Wk 4, this is just orientation
- [ ] Bookmark the paper in `resources.md` (already done)

**Success**: You can articulate in one sentence what PagedAttention is optimizing (memory fragmentation in the KV cache). Don't try to understand the details yet.

---

### Task 5 — Log the week (10 min)

- [ ] Open `~/projects/mle-transition/progress.md`
- [ ] Update Wk 1 row: status ✅ (or 🟡 with note if you didn't finish everything)
- [ ] Add repo link to the artifact column
- [ ] Add one sentence to the "Global log" section: any realizations, surprises, blockers

---

## What "done" for Wk 1 looks like

You end the week with:
1. A Modal account you're actually using
2. A public GitHub repo with 1-2 commits
3. Tokens/sec baseline for a 1B model on A100 (a real number in your notes)
4. Comfort running short GPU jobs from your laptop

You do *not* need:
- vLLM installed (that's Wk 3)
- To understand PagedAttention (that's Wk 4)
- A 7B model working (that's Wk 2)

Move fast, don't over-engineer. Wk 1 is entirely about making sure the environment is real.

---

## Common failure modes

- **Spending 2 hrs on Modal free-tier limits** — if you hit a limit, just add a credit card ($10 buffer). Time is more valuable than $10.
- **Trying to run a 7B model** — do the 1B first. 7B on A100 has memory quirks not worth debugging until you're comfortable with the loop.
- **Getting distracted by "which model" choice** — pick Qwen 2.5 1B or Llama 3.2 1B and move on. All the interesting variance is in the serving stack, not the model.
- **Not committing to git** — if it's not in a public repo by Sunday night, it didn't happen.
