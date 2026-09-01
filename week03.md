# Week 3 — vLLM 安装 + Baseline Benchmark

**Dates**: 2026-09-13 → 2026-09-19
**Goal**: 在 A100 上用 vLLM 跑 Qwen2.5-7B，和 Wk2 的 HF baseline (~40 tok/s) 直接对比，记录提升倍数。

**Time budget**: 3-4 小时

---

## Tasks

### Task 1 — Modal 环境装 vLLM (30 min)

- [x] 新建脚本，image 里装 vllm（需要用 `nvidia/cuda:12.4.0-devel-ubuntu22.04` 基础镜像，debian_slim 没有 nvcc，FlashInfer JIT 编译会失败）
- [x] 用 `vllm.LLM` 加载模型，`SamplingParams` 控制生成参数
- [x] 跑 3 次，记录 tok/s

**Results** — Qwen2.5-7B-Instruct, vLLM (fp16), A100, 100 tokens, greedy decode:
| Run | tok/s | Notes |
|-----|-------|-------|
| 1   | 78.2  | first run (CUDA graph already warmed up by vLLM init) |
| 2   | 80.6  | steady state |
| 3   | 80.6  | steady state |

**Steady-state: ~80 tok/s** ✅ Done 2026-09-06.

**坑**: `VLLM_USE_V1=0` 在新版 vLLM 里是无效变量，必须用带 nvcc 的 CUDA devel 镜像解决 FlashInfer JIT 问题。

---

### Task 2 — 对比 HF vs vLLM，单请求 (30 min)

- [x] 对比表：
  | 模型 | Backend | tok/s | 倍数 |
  |------|---------|-------|------|
  | Qwen2.5-7B | HF transformers | ~40 | 1x |
  | Qwen2.5-7B | vLLM | ~80 | **2x** |

**Success**: ✅ Done 2026-09-06. 单请求 2x 提升，来自 CUDA graph + torch.compile（inductor）。比预期高，通常说的"单请求提升有限"是指没有编译优化的版本。

---

### Task 3 — 多并发压测，看 throughput 差距 (60 min)

这才是 vLLM 真正发力的地方。

- [ ] HF 方式：串行跑 8 条请求，计算总 throughput (tokens/s)
- [ ] vLLM 方式：同时发 8 条请求（batch），计算总 throughput
- [ ] 记录对比

**Success**: 看到 vLLM 在多并发下的真实优势（预期 3-5x throughput 提升）。

---

### Task 4 — 更新 benchmarks/README.md + 日志 (20 min)

- [ ] 把 vLLM 结果加入基准表格
- [ ] 更新 `progress.md` Wk3 行

---

## What "done" for Wk 3 looks like

1. vLLM 在 Modal A100 上跑通
2. 单请求 HF vs vLLM 对比数字
3. 多并发 throughput 对比数字
4. 能解释：为什么单请求提升有限，多并发提升显著

---

## 常见坑

- **vLLM 镜像大**：第一次 Modal 构建要 5-10 分钟，正常
- **显存占用更高**：vLLM 会预分配 KV cache block pool，启动时显存占用比 HF 高
- **单请求 latency 可能更慢**：vLLM 有调度 overhead，单请求不是它的主场
