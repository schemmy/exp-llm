# Week 6 — Load Pressure Sweep

**Dates**: 2026-10-04 → 2026-10-10  
**Goal**: 提高请求到达速率，观察 GPU 在接近饱和时 TTFT 和 latency 如何变化。

**Time budget**: 2-3 小时

---

## 实验设计

固定每条请求输出 200 tokens，sweeping 4 个负载场景：

| 场景 | 请求数 | 平均间隔 | 到达率 |
|------|--------|----------|--------|
| low load | 32 | 0.20s | ~5 req/s |
| medium load | 32 | 0.05s | ~20 req/s |
| high load | 64 | 0.05s | ~20 req/s |
| heavy load | 64 | 0.02s | ~50 req/s |

脚本：`experiments/wk06_load_pressure/load_sweep.py`

---

## 结果

| 场景 | Latency P50 | Latency P99 | TTFT P50 | TTFT P99 | Throughput |
|------|-------------|-------------|----------|----------|------------|
| low load (~5 req/s) | 2.45s | 2.46s | 33ms | 43ms | 763 tok/s |
| medium load (~20 req/s) | 2.54s | 2.56s | 37ms | 49ms | 1,585 tok/s |
| high load (~20 req/s, 64 req) | 2.63s | 2.65s | 38ms | 50ms | 2,291 tok/s |
| heavy load (~50 req/s) | 2.71s | 2.74s | 39ms | 50ms | **3,273 tok/s** |

---

## 核心发现：A100 没有被打满

**预期**：TTFT 随负载升高会急剧上升，latency P99 炸开。  
**实际**：TTFT P50 只从 33ms 爬到 39ms，latency P99 几乎纹丝不动（2.46s → 2.74s）。

**为什么没饱和？**

这个实验的 prompt 都很短（~10-20 tokens），prefill 计算量极小，每条请求的 prefill 在几毫秒内就完成。A100 80GB 的算力远超这种负载——即便 50 req/s，GPU 仍然游刃有余。

**要真正打满 A100，需要：**
1. **长 prompt**（thousands of tokens）——prefill 是 O(n²) attention，长 prompt 会让 prefill 时间暴增
2. **更高的到达率**（比如 200+ req/s）
3. **或更小的 GPU**（如 T4、A10G）

**Throughput 的线性扩展**说明 continuous batching 工作完美：从 5 req/s 到 50 req/s，吞吐量从 763 → 3,273 tok/s，近 4.3x，几乎和到达率成正比。GPU 的计算能力在这个范围内是线性可用的。

---

## 关键洞察：TTFT 的本质

TTFT 由两部分决定：
1. **排队时间**：请求到达时 GPU 在处理别的请求，需要等待调度
2. **Prefill 时间**：GPU 实际计算这条请求的 KV cache

当 prompt 很短时，prefill 只需几毫秒，即使排队也不会显著影响 TTFT。  
**要看到 TTFT 崩溃，必须让 prefill 本身变慢**——即长 prompt。

下周实验方向：用长 prompt（512/1024 tokens）重跑，观察真实的 prefill bottleneck。

---

## 与 Week 5 对比

| 指标 | Wk5 CB (低负载) | Wk6 heavy load |
|------|-----------------|----------------|
| 请求数/时间段 | 32 req / 6.4s | 64 req / 1.28s |
| TTFT P50 | 49ms | 39ms |
| Latency P50 | 2.41s | 2.71s |
| Throughput | 705 tok/s | 3,273 tok/s |

heavy load 下 TTFT 反而更低（39ms vs 49ms）——Week 5 的 50-300 token 可变输出导致调度不均，而这里固定 200 tokens 更整齐。

---

## Tasks

- [x] 写 load_sweep.py，sweep 4 个负载场景
- [x] 分析结果：A100 在短 prompt 场景下未饱和
- [x] 更新 progress.md
- [x] commit + push
