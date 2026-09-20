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

- [x] 用 `torch.profiler.profile` 抓一步 DDP 的详细 trace（batch=16，10 步）
- [x] 找出 comm（nccl/all-reduce）占总 CUDA 时间的比例：**4.41%**
- [ ] ~~画一张 stacked bar：不同 batch size 下这四段的绝对时间~~ ——
      只在 batch=16 抓了一次 trace（脚本设计如此，避免三档 batch 各抓一次
      拉长运行时间），没有跨 batch 的对比图。如果 Wk 16 需要，可以再补

这个 trace 是 Wk 16 的钩子——Wk 16 要动 `bucket_cap_mb`（DDP 把小梯度攒成
大 bucket 再发的阈值），需要先有一张"通信开销当前占多少"的图作为基准。

---

## Task 4 — 日志

- [x] 更新 `progress.md` Wk 15 那一行
- [x] 新建 `benchmarks/README_training.md`（P2 专用，`README.md` 是 P1 inference-only 范围，标题名不副实所以没往里塞）+ Wk 15 结果表
- [x] commit + push（每个文件单独一次，一如既往）

---

## What "done" for Wk 15 looks like

1. 有一张跨 3 档 batch 的 scaling efficiency 表：单卡吞吐、双卡吞吐、efficiency 比值
2. 有一张 profiler 的时间构成图：forward / backward / comm / optim 各占多少
3. 回答了本周开头的核心问题："什么条件下 DDP 通信开销大到值得关注"
4. Wk 16 可以直接从这些数字接过去，不用重跑 baseline

---

## 结果

### Task 1 — 单卡 baseline

| batch | tok/s | step time | 显存峰值 |
|---|---|---|---|
| 4  | 17,921.3 | 114.28ms | 5.48 GB |
| 16 | 19,633.2 | 417.25ms | 17.37 GB |
| 64 | 20,603.2 | 1590.43ms | 64.95 GB |

### Task 2 — 双卡 DDP + scaling efficiency

| batch | 单卡 tok/s | DDP 总 tok/s（2×） | scaling efficiency |
|---|---|---|---|
| 4  | 17,921.3 | 34,608.9 | **96.56%** |
| 16 | 19,633.2 | 38,814.8 | **98.85%** |
| 64 | 20,603.2 | 41,190.0 | **99.96%** |

*DDP 存的 `tokens_per_sec` 是 rank 0 自己那张卡的数字（自己的 batch / 自己的
墙钟时间），不是系统总吞吐——但因为每步都靠阻塞式 all-reduce 强制同步，两张卡
的节奏锁在一起，rank 0 的节奏就代表了整个系统的节奏，所以总吞吐 = rank0 数字 × 2。*

### Task 3 — profiler 时间构成（batch=16）

**comm_fraction（nccl/all-reduce 占总 CUDA 时间）: 4.41%**

耗时最长的几类 CUDA 操作（10 步累计）：`aten::mm` 2587.6ms（990 次）、
`AddmmBackward0`（自动微分节点）~1567-1591ms（480 次）、
`DistributedDataParallel.forward` ~1390-1428ms（10 次，奇怪的是这个名字
出现了两条条目、数字很接近但不完全一样——没深挖，留作疑点）、
`aten::addmm` 780.0ms（480 次）。矩阵乘和它的反向占大头，nccl 相关操作
只是一小片，跟上面接近 100% 的 scaling efficiency 是一致的。

### 复盘：预测对了什么、错了什么

**预测错了（week15.md 开头写的）**：以为 batch 小的时候通信开销占比高，
efficiency 会明显掉，猜 batch=4 大概只有 0.7-0.85。**实测 batch=4 就有
96.56%**，跟 batch=64 的 99.96% 差距很小。

**为什么错**：GPT-2 124M 太小了——梯度体积只有 ~500MB（fp32，一步同步一次），
DDP 的 bucket 机制会在反向传播还没走完的时候就开始异步发出已经算好的那些
bucket 的 all-reduce，跟还在跑的反向计算重叠起来。只要模型不是大到让梯度
体积和算力不成比例，这个重叠机制几乎能把通信开销全部藏起来——**这周真正
验证的是"DDP 的重叠设计有多有效"，而不是"通信开销有多大"**。

这也解释了 comm_fraction（4.41%）比 batch=16 实测的 efficiency 损失
（100% - 98.85% = 1.15%）大：4.41% 是不重叠情况下 comm 会占的原始 CUDA
时间比例，重叠掉了大部分之后，真正体现在墙钟时间上的损失更小。

**跑的过程中撞到两个 bug**（都在 `ddp_scaling.py` 的 commit 历史里）：
1. `torch.profiler` 的 `FunctionEventAvg.cuda_time_total` 属性在当前 torch
   版本被改名成 `device_time_total`，导致 `ddp_bs16` 那一腿直接崩溃，
   没存下任何结果——重跑了这一腿才修好。
2. 脚本里打印 scaling efficiency 的公式多除了一次 2（`d_tps/(2*s_tps)`），
   第一次跑出来的数字（48%/50%）比真实值（96.6%/100%）低了将近一半——
   这个是纯算式错误，不是数据本身有问题，改完公式后原始数据直接复用，
   不用重跑 single/ddp_bs4/ddp_bs64 那几腿。

**给 Wk 16 的钩子**：Wk 16 要动 `bucket_cap_mb`（bucket 分组阈值），
既然这周发现"重叠"是 DDP 几乎不掉速的主因，Wk 16 更该问的问题是——
调小 bucket 阈值（更早触发 all-reduce，重叠窗口更长）还是调大
（更少次数的 all-reduce，但重叠窗口更短）对这个几乎已经 100% 高效的
场景还有没有意义？还是说 GPT-2 124M 这个规模下已经是重叠饱和了，
调 bucket 大小根本测不出差别，得换更大的模型才能看到效果？
