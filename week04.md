# Week 4 — PagedAttention 精读 + Batch Size Sweep

**Dates**: 2026-09-20 → 2026-09-26
**Goal**: 深度理解 PagedAttention 的原理，跑 batch size sweep（1/4/8/16/32），画出 throughput 曲线，量化"并发越多 vLLM 优势越大"。

**Time budget**: 4-6 小时

---

## Tasks

### Task 1 — 精读 PagedAttention 论文 §3-4 (90 min)

论文：https://arxiv.org/abs/2309.06180

- [ ] §3 Background: 理解 KV cache 在 attention 计算中的角色
- [ ] §3.2 Memory Challenges: 搞清楚 internal/external fragmentation 的定义
- [ ] §4 PagedAttention: 核心算法——block table、physical/logical block 的映射
- [ ] §4.3 Scheduling and Preemption: 了解抢占机制（recompute vs swap）
- [ ] 看完后用自己的话写 2-3 句总结，记在本文件底部

**Success**: 能解释 block table 是什么，以及为什么 physical block 不需要连续。

---

### Task 2 — Batch Size Sweep (90 min)

用 vLLM 跑不同并发数，画出 throughput 曲线。

- [ ] 测试 batch size: 1, 4, 8, 16, 32
- [ ] 每个 batch size 跑 2 次取均值
- [ ] 记录 total tok/s（所有请求合计）
- [ ] 同时记录 per-request latency（单条请求平均耗时）

**预期形状**:
- throughput 随 batch size 增大而增大（到某点趋于平稳）
- per-request latency 随 batch size 增大而增大（trade-off）

**Success**: 有一张 throughput vs batch size 的数据表，能解释曲线形状。

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

## PagedAttention 3句话总结（读完后填）

> （待填）

---

## Batch Sweep 结果（跑完后填）

| batch size | total tok/s | per-req latency (s) |
|-----------|-------------|---------------------|
| 1 | | |
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |
