# Week 15 — PyTorch DDP baseline：2× A100 上的第一次分布式训练

**Dates**: 2026-12-06 → 2026-12-12
**Goal**: 在 2 张 A100 上跑 PyTorch DDP，测出**scaling efficiency**（双卡吞吐相对
单卡的比值），并回答一个 P1 没碰过的问题——**通信开销**在什么条件下会把双卡的
理论 2 倍收益吃掉。

**Time budget**: 4-5 小时 ｜ **GPU 成本**: ~$8-10（Modal 上 2× A100，1-1.5 小时）

---

## 从 P1 带过来的钩子

P1 的 12 周里，所有吞吐测量都是**在一张 GPU 上**，横轴是 batch size，纵轴是
tokens/second。轴够简单，因为唯一在竞争的是这张卡的算力和带宽。

进入 P2 训练场景，纵轴还是 tokens/sec，但横轴多了一个**GPU 数量**。天真的
期望是：GPU 数量 × 单卡吞吐 = 总吞吐。实际不会这样——每一步反向传播完之后，
DDP 都要在所有卡之间做一次 **all-reduce**（把每张卡算出来的梯度平均）。
这一步是纯通信，不算数，加进去之后 scaling efficiency 一定 < 1。

**核心问题**：这个 efficiency 到底是 0.99（可以忽略），还是 0.5（一半时间在
等）？答案取决于两个量的比例——**每一步的算量**（batch × model FLOPs）和
**每一步的梯度体积**（模型参数量 × 4B fp32 或 2B fp16）。计算越多、梯度越少
的时候，DDP 越接近理想线性 scaling。

这周就是把这条曲线亲手画出来。

---

## 模型选型：为什么用 GPT-2 124M 而不是 Qwen2.5-7B

- **7B 放不进单卡训练**：DDP 的前提是"模型能装进一张卡"，7B 训练要 ~28GB
  权重 + ~28GB 梯度 + ~56GB 优化器状态 = ~112GB，一张 A100-80GB 装不下。
  那是 Wk 17 FSDP 的题目，不是 Wk 15 DDP 的题目。
- **124M 是纯 DDP 场景**：GPT-2 small 训练时占 ~2GB，两张卡各放一份完整
  副本还有大量余量，通信路径清晰——只有反向传播后的梯度 all-reduce，没有
  参数 shard 的复杂性。**先把这条最干净的 baseline 跑出来**，Wk 16-19 才有
  东西对比。
- **Karpathy 的 nanoGPT 是公开可复现的参考实现**：不用自己从零写一个训练
  循环，专注在 DDP 本身。

---

## Task 1 — 单卡 baseline：先把"没通信"的天花板测出来

- [ ] 写 `experiments/wk15_ddp/train_single.py`
- [ ] GPT-2 124M（12 层、768 hidden、12 heads），合成随机 token 输入
      （数据加载不是这周的题目，别在 tokenizer 和 dataloader 上耗时间）
- [ ] batch size sweep：`per_gpu_bs = 4 / 16 / 64`
- [ ] 测 100 步 warmup + 200 步稳定态的 **tokens/sec** 和 **step_time_ms**
- [ ] 记录 GPU 显存占用（`torch.cuda.max_memory_allocated`）

**预期**：batch 越大吞吐越高，因为算量对固定 kernel launch 开销的稀释率越高。
和 P1 Wk 4 的 batch sweep 曲线形状类似，但绝对数字比推理低很多（训练的
反向传播 FLOPs 是前向的 2 倍左右，序列长度也影响 attention 的二次项）。

---

## Task 2 — 双卡 DDP：同样的 per-GPU batch，测 scaling efficiency

- [ ] 写 `experiments/wk15_ddp/train_ddp.py`——用 `torch.distributed` 起 DDP
- [ ] Modal 上开一个 `gpu="A100-80GB"` 且 `gpu_count=2` 的 Function
- [ ] **关键控制**：per-GPU batch size 保持和 Task 1 一样（4/16/64），这样
      全局 batch 就是 2 倍。这样测出来的 tokens/sec 直接除以单卡数字，就是
      scaling efficiency
- [ ] 同样测 100 warmup + 200 稳定
- [ ] 记录：`tokens_per_sec_ddp`、`ddp_step_time_ms`、`per_step_comm_ms`
      （用 `torch.cuda.Event` 打点，把 all-reduce 那段圈出来）

**核心指标**：

```
scaling_efficiency = tokens_per_sec_ddp / (tokens_per_sec_single × 2)
comm_fraction = per_step_comm_ms / ddp_step_time_ms
```

**预期**：
- batch=4：算量小、每步很快、通信占比高 → efficiency 可能只有 0.7-0.85
- batch=64：算量大、每步耗时长、通信被摊薄 → efficiency 应该 ≥ 0.95
- 通信时间应该基本恒定（跟 batch 无关，只跟参数量有关）——这是这周
  最想验证的一个直观预测

如果实测反了，就说明我对 DDP 通信机制的理解有漏洞，比测数字本身更值得挖。

---

## Task 3 — 对着 `torch.profiler` 看一步 DDP 的时间构成

- [ ] 用 `torch.profiler.profile` 抓一步 DDP 的详细 trace
- [ ] 找出：forward、backward、all-reduce、optimizer.step 各自占多少
- [ ] 画一张 stacked bar：不同 batch size 下这四段的绝对时间

这个 trace 是 Wk 16 的钩子——Wk 16 要动 `bucket_cap_mb`（DDP 把小梯度攒成
大 bucket 再发的阈值），需要先有一张"通信开销当前占多少"的图作为基准。

---

## Task 4 — 日志

- [ ] 更新 `progress.md` Wk 15 那一行
- [ ] `benchmarks/README.md` 加 P2 段落 + Wk 15 结果表
- [ ] commit + push（每个文件单独一次，一如既往）

---

## What "done" for Wk 15 looks like

1. 有一张跨 3 档 batch 的 scaling efficiency 表：单卡吞吐、双卡吞吐、efficiency 比值
2. 有一张 profiler 的时间构成图：forward / backward / comm / optim 各占多少
3. 回答了本周开头的核心问题："什么条件下 DDP 通信开销大到值得关注"
4. Wk 16 可以直接从这些数字接过去，不用重跑 baseline

---

## 结果（跑完后填）

### Task 1 — 单卡 baseline

### Task 2 — 双卡 DDP + scaling efficiency

### Task 3 — profiler 时间构成

### 复盘：预测对了什么、错了什么
