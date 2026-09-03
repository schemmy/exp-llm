# Week 4 — PagedAttention 精读 + Batch Size Sweep

**Dates**: 2026-09-20 → 2026-09-26
**Goal**: 深度理解 PagedAttention 的原理，跑 batch size sweep（1/4/8/16/32），画出 throughput 曲线，量化"并发越多 vLLM 优势越大"。

**Time budget**: 4-6 小时

---

## Tasks

### Task 1 — 精读 PagedAttention 论文 §3-4 (90 min)

论文：https://arxiv.org/abs/2309.06180

- [x] §3 Background: 理解 KV cache 在 attention 计算中的角色
- [x] §3.2 Memory Challenges: 搞清楚 internal/external fragmentation 的定义
- [x] §4 PagedAttention: 核心算法——block table、physical/logical block 的映射
- [x] §4.3 Scheduling and Preemption: 了解抢占机制（recompute vs swap）
- [x] 看完后用自己的话写 2-3 句总结，记在本文件底部

**Success**: ✅ Done 2026-09-02.

---

### Task 2 — Batch Size Sweep (90 min)

用 vLLM 跑不同并发数，画出 throughput 曲线。

- [x] 测试 batch size: 1, 4, 8, 16, 32
- [x] 每个 batch size 跑 2 次取均值
- [x] 记录 total tok/s 和 wall latency

**Success**: ✅ Done 2026-09-02.

**结果** (Qwen2.5-7B, vLLM, A100 80GB, fp16, 1024 tokens/req):

| batch | total tok/s | wall latency |
|-------|-------------|-------------|
| 1 | 79.9 | 1.25s |
| 4 | 321.9 | 1.24s |
| 8 | 642.5 | 1.25s |
| 16 | 1,255.4 | 1.27s |
| 32 | 2,408.8 | 1.33s |

**关键发现**：throughput 近乎线性扩展（4× batch → ~4× throughput），wall latency 几乎不变（1.25s → 1.33s）。这说明：
1. GPU 在并行处理所有请求的 decode 步骤，没有排队等待
2. batch=32 时 A100 80GB 仍未饱和（compute ceiling 还有余量）
3. 真实生产场景下，提高并发是免费的 throughput——latency 几乎不涨

---

### Task 3 — 更新 benchmarks/README.md (20 min)

- [ ] 加入 batch size sweep 数据表
- [ ] commit + push

---

### Task 4 — 论文笔记 + 日志 (20 min)

- [ ] 在本文件底部写 PagedAttention 3 句话总结
- [ ] 更新 `progress.md` Wk4 行

---

## What "done" for Wk 4 looks like

1. PagedAttention 论文核心机制能用自己的话解释
2. Batch size 1/4/8/16/32 的 throughput 数据
3. 理解 throughput/latency trade-off 曲线的形状和原因

---

## PagedAttention 3句话总结

> 1. **问题**：传统 LLM serving 按最大长度预分配 KV Cache，60–80% 显存成为碎片，严重限制并发。
> 2. **方案**：PagedAttention 用 block table 实现 KV Cache 的非连续分页分配，碎片率降至 <4%，物理 block 按需分配、即时释放。
> 3. **效果**：配合 continuous batching，同等显存下并发请求数大幅提升，throughput 比同期系统高 1.8–2.2×。

---

## Batch Sweep 结果

| batch size | total tok/s | wall latency |
|-----------|-------------|--------------|
| 1 | 79.9 | 1.25s |
| 4 | 321.9 | 1.24s |
| 8 | 642.5 | 1.25s |
| 16 | 1,255.4 | 1.27s |
| 32 | 2,408.8 | 1.33s |
