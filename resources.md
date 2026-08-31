# Resources

## GPU platforms (rented, on-demand)

| Provider | Best for | Pricing (approx, Aug 2026) | Notes |
|---|---|---|---|
| **Modal** | Getting started, short jobs | $30 free credit; A100 ~$4/hr | Best DX, Python-native, no infra to manage |
| **RunPod** | Cheap sustained work | A100 40GB ~$1.50/hr, H100 ~$3/hr | Community cloud can be flaky; secure cloud is stable |
| **Lambda** | Longer-running training | A100 ~$1.10/hr on-demand | Reservations get cheaper; good for P2 |
| **Together / Fireworks / Baseten** | Testing hosted inference | Per-token pricing | Use for comparison baseline in Blog #1 |
| **Colab Pro+** | Casual experimenting | $50/mo, A100 access | Limited runtime; not for serious benchmarks |
| **GKE / EKS small cluster** | P3 K8s work | ~$50-100/mo with 1 GPU node | Only when you hit P3 |

**Budget guide**: 
- P1: ~$80-150 total GPU spend
- P2: ~$200-400
- P3: ~$100 GPU + ~$50-100/mo K8s cluster for a couple months
- P4: minimal ($50-100)

## Anchor courses (walk through, don't skim)

- **Stanford CS336** — Language Modeling from Scratch (P2 anchor)
- **Stanford CS149** — Parallel Computing (P4, optional)
- **UW CSE 599W** — Systems for ML (P3 supplement)

## Anchor books

- **Programming Massively Parallel Processors (PMPP)**, 4th ed. — P4 anchor
- **Designing Machine Learning Systems** by Chip Huyen — background reading anytime

## Must-read papers (in reading order)

### P1
- [vLLM / PagedAttention (SOSP '23)](https://arxiv.org/abs/2309.06180) — Wk 4
- [FlashAttention v1 (NeurIPS '22)](https://arxiv.org/abs/2205.14135) — skim in P1, deep-dive P4
- [Efficient Memory Management for LLM Serving with PagedAttention (blog)](https://blog.vllm.ai)
- [Continuous batching in Anyscale blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)

### P2
- [ZeRO: Memory optimization towards training trillion parameter models](https://arxiv.org/abs/1910.02054)
- [FSDP: PyTorch's Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- [LoRA (ICLR '22)](https://arxiv.org/abs/2106.09685)
- [QLoRA (NeurIPS '23)](https://arxiv.org/abs/2305.14314)

### P3
- KServe docs
- vLLM production deployment guide
- SkyPilot serve docs

### P4
- FlashAttention v2, v3 papers
- Aleksa Gordić matmul blog (already read once — reread in P4)

## Blogs / substacks to follow

- Sebastian Raschka (ahead-of-ai substack) — LLM systems
- Aleksa Gordić (personal blog) — CUDA / matmul internals
- HazyResearch group blog — FlashAttention, Mamba, state-space
- vLLM blog
- Together / Anyscale / Modal engineering blogs

## OSS projects to know (and eventually contribute to)

- **vLLM** — https://github.com/vllm-project/vllm — highest-leverage OSS PR target for MLE brand
- **SkyPilot** — https://github.com/skypilot-org/skypilot — you already know the codebase; low friction for a 2nd PR
- **Ray / Ray Serve** — https://github.com/ray-project/ray
- **KServe** — https://github.com/kserve/kserve
- **llama.cpp / ggml** — for the CPU/edge inference perspective
- **TensorRT-LLM** — NVIDIA's stack; know it exists but not required

## Blog platform

- **GitHub Pages** on chenxin-ma.github.io — recommended (SEO, permanent, ATS-scannable from resume)
- Alternative: Substack (better distribution, but content ownership weaker)

## Tracker for compute-efficiency papers (leisure reading, 1/mo)

- [ ] Alpa: Automating Inter- and Intra-Operator Parallelism
- [ ] Megatron-LM (original + subsequent)
- [ ] DeepSpeed-Inference
- [ ] SGLang
- [ ] SpecInfer / Medusa (speculative decoding)
- [ ] Ring Attention
- [ ] TVM / MLIR overview

*(add papers as you find them; drop the ones that turn out to be low-yield)*
