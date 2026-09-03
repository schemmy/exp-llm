# Week 5 — Continuous Batching 实验

**Dates**: 2026-09-27 → 2026-10-03
**Goal**: 模拟真实流量——请求在不同时间到达、长度各异，对比 static batching 和 continuous batching 的 GPU 利用率和 throughput 差异。

**Time budget**: 4-6 小时

---

## 背景

Week 3-4 的实验是"静态 batch"：所有请求同时到达，vLLM 一次性处理。
这周模拟**真实场景**：请求按泊松分布到达，长度随机，系统需要动态调度。

关键区别：

| | Static Batch | Continuous Batching |
|---|---|---|
| 请求到达 | 同时 | 随时 |
| GPU 空闲时机 | 批次结束前一直满 | 几乎不空闲 |
| 短请求结束后 | 等长请求跑完 | 立即放入新请求 |
| 延迟敏感度 | 高（等凑批） | 低（即到即处理） |

---

## Tasks

### Task 1 — 用 vLLM 的 AsyncLLMEngine 模拟流式请求 (90 min)

vLLM 的 `LLM`（同步）是 static batch，`AsyncLLMEngine`（异步）才是真正的 continuous batching。

- [ ] 用 `AsyncLLMEngine` + `asyncio` 发送请求流：
  - 32 条请求，按泊松分布到达（平均间隔 0.2s）
  - 输出长度随机（50-500 tokens，模拟真实分布）
- [ ] 记录每条请求的 latency（从发出到收到最后一个 token）
- [ ] 计算整体 throughput（总 tokens / 总时间）

**Success**: 有 per-request latency 分布（P50/P90/P99）和总 throughput。

---

### Task 2 — 对比：同样请求流，用同步串行模拟 static batching (45 min)

- [ ] 同样的 32 条请求，改成等所有请求到达后一次性 batch 发送
- [ ] 对比 P50/P90/P99 latency 和 throughput

**Success**: 数字证明 continuous batching 在真实流量下的优势。

---

### Task 3 — 理解 time-to-first-token (TTFT) vs time-per-output-token (TPOT) (30 min)

这是推理系统的两个核心延迟指标：
- **TTFT**：从发出请求到收到第一个 token（prefill 时间）
- **TPOT**：之后每个 token 的平均时间（decode 速度）

- [ ] 在 Task 1 的脚本里同时记录这两个指标
- [ ] 思考：大 batch 对 TTFT 和 TPOT 分别有什么影响？

**Success**: 能解释为什么 TTFT 和 TPOT 需要分开优化。

---

### Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk5 行
- [ ] commit + push

---

## What "done" for Wk 5 looks like

1. 有真实流量模拟下的 latency 分布（P50/P90/P99）
2. Continuous batching vs static batching 的对比数字
3. 知道 TTFT 和 TPOT 是什么，能解释各自的 bottleneck

---

## Latency 结果（跑完后填）

### Continuous Batching (AsyncLLMEngine)
| 指标 | 数值 |
|------|------|
| P50 latency | |
| P90 latency | |
| P99 latency | |
| Throughput | |
| TTFT P50 | |
| TPOT P50 | |

### Static Batching（等所有请求到达后一次 batch）
| 指标 | 数值 |
|------|------|
| P50 latency | |
| P90 latency | |
| P99 latency | |
| Throughput | |
