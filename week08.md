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

- [x] batch=1，对比三种配置的 tok/s 和 TPOT（draft model 不受支持，见下）
- [ ] 记录 acceptance rate（vLLM 日志里有）

### Task 2 — 高并发下的反转 (60 min)

- [x] batch=16，对比 baseline vs ngram
- [x] 验证"高 batch 下投机解码收益消失甚至变负" —— 确认，0.89x

### Task 3 — 理解 acceptance rate (30 min)

acceptance rate = 平均每次验证接受的 token 数 / k

- [ ] 思考：什么样的输入 acceptance rate 高？什么样的低？（`ngram_acceptance.py` 待跑）
- [x] 为什么 draft model 和 target model 必须同 tokenizer？—— 见下方 vocab_size 分析

### Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk8 行
- [ ] commit + push

---

## What "done" for Wk 8 looks like

1. 有 batch=1 和 batch=16 两组对比数字
2. 能解释为什么投机解码在低 batch 有效、高 batch 失效
3. 知道 draft model 和 n-gram 各自的适用场景

---

## 结果

任务：~150 token 文档 + summarize/explain 类问题，输出 256 tokens，A100 fp16。

| 配置 | batch | total tok/s | per-req tok/s | TPOT | vs baseline |
|------|-------|-------------|---------------|------|-------------|
| baseline | 1 | 80.6 | 80.6 | 12.4ms | 1.00x |
| ngram (k=5) | 1 | 87.4 | 87.4 | 11.4ms | **1.08x** ✅ |
| baseline | 16 | 784.3 | 49.0 | 20.4ms | 1.00x |
| ngram (k=5) | 16 | 695.4 | 43.5 | 23.0ms | **0.89x** ❌ |

**反转出现了**：同一个开关，batch=1 赚 8%，batch=16 亏 11%。

原因和 Week 4 的 batch sweep 直接对得上：batch=16 时 GPU 已经在做真实工作，算力不再空闲。
此时每个被拒绝的投机 token 都是从真实计算里抢来的，命中收益 < 浪费成本，净亏。

---

## draft model 没跑成 —— 两个失败原因

### 1. 当前 vLLM V1 engine 不支持通用 draft model

```
ValueError: Speculative decoding with draft model is not supported yet.
Please consider using other speculative decoding methods such as
ngram, medusa, eagle, or deepseek_mtp.
```

投机解码的实现正在从"任意小模型当 draft"收敛到**和 target 模型一起训练的专用 draft head**
（EAGLE、Medusa、MTP）。原因是专用 head 共享 target 的 hidden state，acceptance rate
远高于独立小模型，而且不额外占一份完整权重。

### 2. vocab_size 不匹配（顺带暴露的机制细节）

```
Target model vocab_size=152064, Draft model vocab_size=151936
```

Qwen2.5-7B 和 Qwen2.5-0.5B **用的是同一个 tokenizer**，真实 token 数一致（~151,643）。
差的 128 是 **embedding 矩阵的 padding**——7B 补齐到 256 的倍数，0.5B 补到 128 的倍数，
为了 tensor core 对齐和张量并行切分。

**所以"draft 和 target 必须同 tokenizer"在工程上真正的要求是"必须同 vocab_size"**：
rejection sampling 要把两个 logits 向量逐位比对，长度不同就无法比较。语义兼容不够。

→ 这个错误正好回答了 Task 3 的第二个问题。

---

## 遗留问题：8% 太小了

ngram 在 batch=1 只赚 8%，远低于文献里报的 2x。假设是 **acceptance rate 太低**：
本实验的任务是 summarize/explain，模型生成的是**新组织的语言**，而 ngram 的猜测方式是
"在 prompt 和已生成内容里找重复 n-gram"——猜测大多被拒绝。

补充实验 `ngram_acceptance.py`：保持引擎不变，只换任务形态（逐字复制 vs 自由生成），
看加速比是否随之变化。若 copy 任务的加速显著高于 novel 任务，即可确认
**投机解码的收益是 acceptance rate 的函数，与引擎无关，与任务形态强相关**。

结果：

| 任务 | 配置 | batch | total tok/s | vs baseline |
|------|------|-------|-------------|-------------|
| novel | baseline | 1 | | 1.00x |
| novel | ngram | 1 | | |
| copy | baseline | 1 | | 1.00x |
| copy | ngram | 1 | | |
| novel | baseline | 16 | | 1.00x |
| novel | ngram | 16 | | |
| copy | baseline | 16 | | 1.00x |
| copy | ngram | 16 | | |
