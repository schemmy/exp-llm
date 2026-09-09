# Week 12 — INT4 量化 + Tradeoff 总表

**Dates**: 2026-11-15 → 2026-11-21  
**Goal**: 测 INT4 的速度/质量权衡，同时回答 Wk 11 留下的一个悬念——
INT8 几乎零加速，是量化格式本身的问题，还是那条 kernel 路径没优化好？

**Time budget**: 3-4 小时 ｜ **GPU 成本**: ~$2

---

## 为什么这周能验证 Wk 11 的悬念

Wk 11 的结论：`int8_per_channel_weight_only` 全程 ~1.0x，**没有兑现"减字节=加速"的承诺**。
猜测是它走的是通用 dequant kernel，没有专门优化。

**这周换一条完全不同的路**：不用 vLLM 的"加载时动态量化"，改用**预量化 checkpoint**
（Qwen 官方发布的 AWQ / GPTQ-Int4 版本），并显式指定 `awq_marlin` / `gptq_marlin`——
这是 vLLM 里公认优化最好的量化 kernel 之一（Marlin，专为 Ampere+ 设计的 INT4×fp16 GEMM）。

**如果 INT4 这次真的兑现了加速**，就说明 Wk 11 INT8 的问题确实出在 kernel 上，
不是"量化本身注定没用"。**如果 INT4 也不加速**，那说明 A100 上做权重量化这件事
本身就有更根本的限制，值得重新审视整个假设。

---

## 权重体积这次是真的——和 Wk 11 不同

Wk 11 的 fp8/int8 都是**加载时动态量化**：下载下来的 checkpoint 本来就是 fp16，
量化发生在读进显存之后，所以"权重体积"那一列测的是下载体积，没有意义。

这周用的是 Qwen 官方**预量化**的 checkpoint——磁盘上的文件本来就是 4-bit。
`weight_gib` 这次测的是真实的量化后体积，Wk 11 的那个测量局限本周不存在。

---

## Task 1 — AWQ vs GPTQ-Int4 vs fp16，batch sweep

- [ ] 跑 `experiments/wk12_int4/int4_quant.py`
- [ ] 模型：`Qwen/Qwen2.5-7B-Instruct-AWQ`（`quantization="awq_marlin"`）
      和 `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`（`quantization="gptq_marlin"`）
- [ ] batch = 1 / 32 / 256（和 Wk 11 完全对齐，直接可比）

**预期**：如果 Marlin kernel 确实高效，batch=1 应该比 fp16 快不少
（4-bit 存储只有 fp16 的 1/4，理论上限比 FP8 的 2x 更高）。
batch=256 的算力区应该重演 FP8 那种"净亏"——甚至可能亏得更多，
因为 4-bit 反量化到 fp16 的开销比 8-bit 更大。

---

## Task 2 — 质量抽查：5 个固定 prompt，相似度对比 fp16

- [ ] 同样的 5 个问题，greedy 解码，AWQ / GPTQ / fp16 各生成一次
- [ ] 用字符级相似度（`difflib.SequenceMatcher`）粗略量化"输出偏离 fp16 多少"
- [ ] 肉眼过一遍，看有没有真正的退化（重复、跑题、语法错误）

这不是严格的困惑度评测，只是比 Wk 11 那种单个 prompt 肉看更systematic 一点的信号。

---

## Task 3 — 汇总 Phase 1 的完整量化 tradeoff 表

把 Wk 11 + Wk 12 的数字拼成一张总表：fp16 / FP8 / INT8 / AWQ-INT4 / GPTQ-INT4，
每行标注权重体积、三档 batch 的吞吐、相对 fp16 的加速比、质量相似度。

这张表是 Phase 1 quantization 部分唯一的交付物，也是 Blog #1 会直接引用的素材。

---

## Task 4 — 日志

- [ ] 更新 `progress.md`
- [ ] commit + push

---

## What "done" for Wk 12 looks like

1. 有 AWQ 和 GPTQ-Int4 的完整 batch sweep 数字
2. 回答了 Wk 11 的悬念：INT8 零加速是 kernel 问题还是格式问题
3. 有一张跨 5 种精度的完整 tradeoff 表，可以直接拿去写 Blog #1

---

## 结果（跑完后填）

### Task 1 — 吞吐

| 精度 | 权重体积 | batch=1 | batch=32 | batch=256 |
|------|---------|---------|----------|-----------|
| fp16（参照，Wk 11） | 14.2 GiB | 93.8 | 2,835.1 | 10,659.4 |
| AWQ-INT4 | | | | |
| GPTQ-INT4 | | | | |

加速比（相对 fp16）：

| 精度 | batch=1 | batch=32 | batch=256 |
|------|---------|----------|-----------|
| AWQ-INT4 | | | |
| GPTQ-INT4 | | | |

### Task 2 — 质量相似度

| 精度 | 平均相似度 vs fp16 | 肉眼检查 |
|------|-------------------|---------|
| AWQ-INT4 | | |
| GPTQ-INT4 | | |

### Task 3 — Phase 1 量化总表

| 精度 | 权重体积 | batch=1 加速 | batch=32 加速 | batch=256 加速 | 质量相似度 |
|------|---------|-------------|--------------|---------------|-----------|
| fp16 | 14.2 GiB | 1.00x | 1.00x | 1.00x | — |
| FP8 | 14.2 GiB* | 1.56x | 1.51x | 0.84x | — |
| INT8 | 14.2 GiB* | 1.03x | 1.03x | 1.06x | — |
| AWQ-INT4 | | | | | |
| GPTQ-INT4 | | | | | |

*FP8/INT8 的权重体积是 Wk 11 遗留的测量局限（测的是下载体积不是量化后体积）。
