# Week 7 — Prefix Caching

**Dates**: 2026-10-11 → 2026-10-17  
**Goal**: 理解 vLLM 的 prefix caching 机制，用实验验证它在什么场景下有效，TTFT 能降多少。

**Time budget**: 4-6 小时

---

## 背景

**Prefix caching（也叫 Automatic Prefix Caching / APC）**：当多条请求共享相同的 token 前缀时，vLLM 把这些 token 的 KV cache 存下来复用，后续请求跳过 prefill，直接从缓存读。

典型场景：
- **RAG**：所有请求都带同一份长文档作为 context
- **多轮对话**：每轮请求包含完整的历史对话
- **System prompt**：同一个 system prompt 用于成千上万条请求

**核心问题**：shared prefix 的 TTFT 从 O(prefix_len) 降到 O(1)（只需 prefix 之后的新 token 做 prefill）。

---

## Tasks

### Task 1 — 读懂 prefix caching 的开关和原理 (30 min)

vLLM 中开启方式：
```python
AsyncEngineArgs(model=..., enable_prefix_caching=True)
```

关键实现细节：
- KV cache block 按 hash 标识；相同内容的 block 共享物理内存
- 第一条请求仍需完整 prefill，结果写入 cache
- 后续请求命中 cache 则直接跳过对应 tokens 的 prefill

- [ ] 理解 block hash 机制
- [ ] 知道 cache miss vs cache hit 的 TTFT 差异来自哪里

---

### Task 2 — 实验：长 shared prefix + prefix caching ON vs OFF (90 min)

设计：
- Shared prefix：~500 token 的长 system prompt（一篇文章 / 一份文档）
- 32 条请求，每条 prefix 相同，suffix 不同（不同问题）
- 对比 `enable_prefix_caching=False` vs `True`
- 指标：TTFT P50/P90/P99，latency，throughput

预期：cache OFF 时每条都要跑完整 500-token prefill；cache ON 时第一条 cache miss，后续 31 条 cache hit，TTFT 暴跌。

- [ ] 写 `experiments/wk07_prefix_cache/prefix_cache_exp.py`
- [ ] 跑实验，记录结果

---

### Task 3 — 理解边界条件 (30 min)

Prefix caching 不是万能的：
- Prefix 必须完全相同（token 级别）——哪怕一个 token 不同就 cache miss
- Cache 占用 GPU 内存，影响 max batch size
- 动态 prefix（每条请求 prefix 都不同）无收益

- [ ] 思考：RAG 场景下 prefix caching 的局限是什么？

---

### Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk7 行
- [ ] commit + push

---

## What "done" for Wk 7 looks like

1. 有 TTFT 对比数字：prefix caching ON vs OFF，至少 3x 差距（预期）
2. 能解释 cache miss（第一条）和 cache hit（后续）的区别
3. 知道 prefix caching 的适用场景和局限

---

## 结果（跑完后填）

| 场景 | TTFT P50 | TTFT P99 | Latency P50 | Throughput |
|------|----------|----------|-------------|------------|
| cache OFF | | | | |
| cache ON | | | | |
