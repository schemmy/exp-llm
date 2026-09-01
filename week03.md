# Week 3 — vLLM 安装 + Baseline Benchmark

**Dates**: 2026-09-13 → 2026-09-19
**Goal**: 在 A100 上用 vLLM 跑 Qwen2.5-7B，和 Wk2 的 HF baseline (~40 tok/s) 直接对比，记录提升倍数。

**Time budget**: 3-4 小时

---

## Tasks

### Task 1 — Modal 环境装 vLLM (30 min)

- [ ] 新建脚本，image 里装 vllm：
  ```python
  image = (
      modal.Image.debian_slim()
      .pip_install("vllm")
  )
  ```
  vLLM 会自动带 torch，不需要单独装。
- [ ] 用 `vllm.LLM` 加载模型，`SamplingParams` 控制生成参数
- [ ] 跑 3 次，记录 tok/s

**Success**: vLLM 在 Modal 上跑通，没有报错。

---

### Task 2 — 对比 HF vs vLLM，单请求 (30 min)

- [ ] 填入对比表：
  | 模型 | Backend | tok/s | 倍数 |
  |------|---------|-------|------|
  | Qwen2.5-7B | HF transformers | ~40 | 1x |
  | Qwen2.5-7B | vLLM | ??? | ???x |
- [ ] 预期：vLLM 单请求提升不大（1-2x），主要优势在多并发

**Success**: 有一个可以解释的数字，知道为什么单请求提升有限。

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
