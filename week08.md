# Week 8 — Speculative Decoding

**Dates**: 2026-10-18 → 2026-10-24  
**Goal**: 理解投机解码的原理，实测它在什么 batch size 下有效、什么时候反而变慢。

**Time budget**: 4-6 小时

---

## 原理

Decode 阶段的本质瓶颈：**memory bandwidth，不是算力**。

每生成 1 个 token，GPU 要把整个 7B 模型的权重（14 GB fp16）从 HBM 读进 SM。这个读取时间远大于矩阵乘法本身的时间。batch=1 时 GPU 的算力利用率可能不到 5%。

**投机解码利用的就是这个空闲算力：**

1. **Draft**：小模型（0.5B）自回归生成 k 个候选 token —— 便宜，因为模型小
2. **Verify**：大模型（7B）用**一次 forward pass** 并行验证这 k+1 个位置
3. **Accept/Reject**：从头开始逐个比对，接受到第一个不匹配为止

关键点：验证 k 个 token 和生成 1 个 token 的**耗时几乎一样**，因为都只需把权重读一遍。多出来的计算量填进了本来空闲的算力。

**数学保证**：用 rejection sampling，输出分布和纯大模型完全一致——不是近似，是精确等价。

---

## 两种 draft 来源

| 方式 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Draft model** | 用小模型（Qwen2.5-0.5B）预测 | 通用，任何输入都能猜 | 占显存，acceptance rate 依赖模型对齐度 |
| **N-gram / prompt lookup** | 在 prompt + 已生成内容里找重复 n-gram | 零显存开销，零额外计算 | 只在有大量复制的场景有效（摘要、代码编辑、RAG） |

---

## 核心预期：batch size 决定成败

这是本周最重要的洞察：

| Batch size | GPU 状态 | 投机解码效果 |
|-----------|---------|-------------|
| 1 | memory-bound，算力大量空闲 | **大幅加速**（预期 1.5-2.5x） |
| 8-16 | 开始接近 compute-bound | 收益缩小 |
| 32+ | compute-bound，算力已饱和 | **可能变慢**（被拒绝的 token = 纯浪费） |

Week 4 的 batch sweep 已经证明：batch 32 时 throughput 2,408 tok/s，GPU 在做真正的工作。这种状态下再塞投机解码的额外计算，只会抢占资源。

---

## Tasks

### Task 1 — 单请求 latency：baseline vs draft model vs ngram (90 min)

- [ ] batch=1，对比三种配置的 tok/s 和 TPOT
- [ ] 记录 acceptance rate（vLLM 日志里有）

### Task 2 — 高并发下的反转 (60 min)

- [ ] batch=16，对比 baseline vs draft model
- [ ] 验证"高 batch 下投机解码收益消失甚至变负"

### Task 3 — 理解 acceptance rate (30 min)

acceptance rate = 平均每次验证接受的 token 数 / k

- [ ] 思考：什么样的输入 acceptance rate 高？什么样的低？
- [ ] 为什么 draft model 和 target model 必须同 tokenizer？

### Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk8 行
- [ ] commit + push

---

## What "done" for Wk 8 looks like

1. 有 batch=1 和 batch=16 两组对比数字
2. 能解释为什么投机解码在低 batch 有效、高 batch 失效
3. 知道 draft model 和 n-gram 各自的适用场景

---

## 结果（跑完后填）

### Batch = 1（延迟敏感场景）

| 配置 | tok/s | TPOT | 加速比 |
|------|-------|------|--------|
| baseline | | | 1.0x |
| draft model (0.5B, k=5) | | | |
| ngram (k=5) | | | |

### Batch = 16（吞吐敏感场景）

| 配置 | total tok/s | 加速比 |
|------|-------------|--------|
| baseline | | 1.0x |
| draft model (0.5B, k=5) | | |
