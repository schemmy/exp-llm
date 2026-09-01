# Week 2 — 7B 模型 HF Baseline

**Dates**: 2026-09-06 → 2026-09-12
**Goal**: 在 A100 上跑通一个 7B 模型的 HuggingFace baseline，记录稳定的 tok/s，作为 Wk3 vLLM 对比的基准。

**Time budget**: 3-4 小时（环境已经通了，这周主要是模型层面的调参）

---

## Tasks

### Task 1 — 选模型 + 跑通推理 (60 min)

- [x] 用 `Qwen/Qwen2.5-7B-Instruct`（推荐：中文友好，权重小，下载快）
- [x] 复用 Wk1 的 Modal 脚本框架，改成 7B：
  - `gpu="A100"` 保持不变（7B fp16 约需 14GB，A100 40GB 够用）
  - `dtype=torch.float16`
  - 生成 100 tokens，greedy decode，跑 3 次
- [x] 记录 3 次 tok/s，取稳态（Run 2/3）

**Results** — Qwen2.5-7B-Instruct, HuggingFace transformers (fp16), A100, 100 tokens, greedy decode:
| Run | tok/s | Notes |
|-----|-------|-------|
| 1   | 19.2  | CUDA warmup — discard |
| 2   | 39.3  | steady state |
| 3   | 39.9  | steady state |

**Steady-state baseline: ~40 tok/s** ✅ Done 2026-09-06.

**反直觉现象**: 7B (40 tok/s) 比 0.5B (31 tok/s) 更快。原因：0.5B 模型太小，每次 forward pass 的矩阵乘法规模不足以塞满 A100 的 tensor core，kernel launch overhead 占比更高。7B 的矩阵足够大，GPU 利用率更高，反而每 token 更快。→ 这说明 tok/s 不只取决于参数量，还取决于 GPU 的 arithmetic intensity。

---

### Task 2 — 对比 0.5B vs 7B，理解差距来源 (30 min)

- [x] 把两个数字放在一起：
  | 模型 | 参数量 | tok/s | 备注 |
  |------|--------|-------|------|
  | Qwen2.5-0.5B | 0.5B | ~31 | Wk1 baseline |
  | Qwen2.5-7B-Instruct | 7B | ~40 | Wk2 baseline |
- [x] 思考差距来源（见 Task 1 备注）

**Success**: ✅ Done 2026-09-06. 一句话：tok/s 由 GPU arithmetic intensity 决定，不只是参数量。

---

### Task 3 — 更新 benchmarks/README.md (20 min)

- [x] 把 7B 结果加入基准表格
- [x] commit + push

**Success**: `benchmarks/README.md` 基准表有两行数据。✅ Done 2026-09-06.

---

### Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk2 行，标 ✅，加 tok/s 数字

---

## What "done" for Wk 2 looks like

1. 7B 模型在 A100 上跑通，有稳态 tok/s 数字
2. 0.5B vs 7B 对比表格在 README 里
3. 能解释为什么更大的模型更慢

你不需要：
- 优化任何东西（那是 Wk3+ 的事）
- 跑 batch inference（先搞清楚单请求）
- 换其他模型（Qwen2.5-7B 就够了）

---

## 常见坑

- **模型下载慢** — Modal 每次冷启动都会重新下载，可以用 `modal.Volume` 缓存，但 Wk2 先不管，直接下就行
- **OOM** — 7B fp16 需要 ~14GB，A100 40GB 绝对够，不会 OOM
- **Run 1 很慢** — 和 Wk1 一样，第一次跑有 CUDA warmup，丢弃，看 Run 2/3
