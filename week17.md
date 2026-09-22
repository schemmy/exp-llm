# Week 17 — FSDP：Qwen2.5-7B 真的能在 DDP 下训练吗？

**Dates**: 2026-12-20 → 2026-12-26
**Goal**: Wk16 的结论是"FSDP 解决的是显存问题，不是通信效率问题"——这周直接
验证这句话，而不是假设它。用 P1 全程测过的同一个模型（Qwen2.5-7B），
真的去撞一次 DDP 的显存墙，再看 FSDP 的 full-shard（ZeRO-3 等价）
能不能把它接住。

**跟 Wk15-16 的区别**：这周的两个核心问题（DDP 装不装得下 7B、FSDP fp32
装不装得下 7B）**目前都不知道答案**——不是"已经被上周结果暗示"的问题，
是真的要跑了才知道，符合上周定下的"只有真不确定的问题才配拿一整周"的
筛选标准。

**Time budget**: 4-5 小时 ｜ **GPU 成本**: ~$12-16（Modal 上 1× + 2× A100，
含 ~15GB 模型下载，复用 P1 的 `hf-cache` volume 省重复下载）

---

## 为什么用 fp32，不用 P1 那种 fp16 推理精度

P1 全程用 fp16/量化做**推理**，这周刻意先用**最简单、最占显存的 fp32**
做训练，原因是要把"权重 + 梯度 + 优化器状态"这三块显存账算清楚、算干净：

- 权重（fp32）：7.6B × 4B ≈ 30.4 GB
- 梯度（fp32，反向传播后跟权重同 dtype）：≈ 30.4 GB
- AdamW 状态（`exp_avg` + `exp_avg_sq`，fp32，各跟参数同 dtype）：≈ 60.8 GB
- **合计 ≈ 121.6 GB** —— 一张 A100-80GB 单卡理论上限的 1.5 倍

这个数字算出来就已经不可能装进 80GB，不需要跑就能预判 Task 1 会失败。
但"预判会失败"和"亲眼看到在哪一步失败、失败前显存冲到多少"是两回事——
后者才是这周真正要收集的数据。

---

## Task 1 — 单卡尝试：Qwen2.5-7B 会在哪一步炸显存

- [ ] 加载 Qwen2.5-7B-Instruct（fp32，复用 `hf-cache` volume）到 1 张 A100-80GB
- [ ] 依次尝试：模型加载 → 建 `AdamW` optimizer → 一次 forward → 一次
      backward → 一次 `optimizer.step()`，每一步都 try/except 包起来
- [ ] 无论在哪一步失败，都记录：到达的最后一步 + 失败前
      `torch.cuda.max_memory_allocated()`

**预期**：模型加载（30.4GB）大概率能成功；forward 应该也能过（激活值在
batch=1 时不大）；backward 后梯度落地会推到 ~60.8GB，可能还没炸；
`optimizer.step()` 第一次调用时 Adam 才真正分配 `exp_avg`/`exp_avg_sq`
（fp32，各 30.4GB），这一步大概率是真正炸显存的地方。但这只是猜测，
具体在哪步炸、炸之前实际吃了多少显存，是这个 Task 要测的东西，不是
预设的结论。

---

## Task 2 — FSDP full-shard（fp32），2× A100，batch=1

- [ ] `FullyShardedDataParallel`，`sharding_strategy=FULL_SHARD`（ZeRO-3
      等价：权重、梯度、优化器状态全部切分）
- [ ] 跟 Task 1 完全相同的 fp32 精度，方便直接对比"同样的精度设置，
      切了 vs 没切"的显存差距
- [ ] 测：per-GPU 显存峰值、tokens/sec

**这个 Task 的结果本身有两种可能，都有信息量**：
- **如果能跑**：121.6GB 总量切成 2 份，per-GPU 静态显存 ≈ 60.8GB，
  加上激活值应该在 80GB 以内——干净地证明"同样的精度，纯靠切分就能把
  装不下变成装得下"
- **如果还是炸**：说明单靠 FULL_SHARD 切分，在 fp32 精度下 2 张卡还不够——
  这不是失败，这是给 Wk 18（ZeRO 档位对比）和 Wk 19（activation
  checkpointing / CPU offload）提前探好路：如果连 ZeRO-3 都不够，
  Wk 18 测 ZeRO-1/ZeRO-2（切得更少，显存只会更紧）就没有意义，
  得优先看 Wk 19 那些进一步省显存的技巧

---

## Task 3 — FSDP full-shard（fp16），2× A100，batch=1

- [ ] 跟 Task 2 完全一样，只是模型直接以 `torch_dtype=torch.float16`
      加载（不是 fp32 转 fp16，是从一开始就以 fp16 常驻）——权重、梯度、
      AdamW 状态全部走 fp16 存储
- [ ] 理论静态显存：(15.2+15.2+30.4) / 2 ≈ 30.4GB per GPU，应该很宽松地
      装得下
- [ ] 测：per-GPU 显存峰值、tokens/sec，跟 Task 2 对比"精度减半，显存
      差多少"

这一档是这周的"保底"——不管 Task 2 是成是败，Task 3 大概率能跑成功，
保证这周至少有一组完整可用的 FSDP 吞吐数字，也顺带量化了混合精度对
显存的真实收益（不是理论上的 2x，因为激活值、通信 buffer 等不完全
随精度线性缩放）。

---

## Task 4 — profiler：FSDP 的通信开销多大

- [ ] 对 Task 2/Task 3 里跑成功的那一档（优先 Task 3，因为更可能成功），
      抓 `torch.profiler` trace
- [ ] FSDP 的通信原语跟 DDP 不一样——DDP 只有 all-reduce，FSDP 还有
      **all-gather**（forward/backward 前临时聚合当前层的完整参数）和
      **reduce-scatter**（反向传播后把梯度切分聚合），事件名里找
      `all_gather` / `reduce_scatter` / `nccl`
- [ ] 算一个 comm_fraction，跟 Wk 15 的 DDP comm_fraction（4.41% @ 124M）
      放在一起看——FSDP 因为多了 all-gather，comm_fraction 大概率明显
      更高，这是"切分换显存"要付的通信代价，值得量化出来而不是只停留在
      "应该更贵"这种定性判断

---

## Task 5 — 日志

- [ ] 更新 `progress.md` Wk 17 那一行
- [ ] `benchmarks/README_training.md` 加 Wk 17 结果表
- [ ] commit + push（每个文件单独一次）

---

## What "done" for Wk 17 looks like

1. 确切知道 Qwen2.5-7B 在 fp32、单卡上会在哪一步炸显存，炸之前吃了多少
2. 至少有一组 FSDP（fp16）的真实吞吐 + 显存数字
3. 知道 fp32 FULL_SHARD 在 2 张卡上够不够——这个结果直接决定 Wk 18
   该怎么设计（如果不够，Wk 18 的 ZeRO-1/2/3 对比要么全部在 fp16 下做，
   要么要先处理 Wk19 的省显存技巧）
4. 有 FSDP 的 comm_fraction 数字，能跟 Wk 15 的 DDP 数字做一个"切分的
   通信代价有多大"的定量对比

---

## 结果（跑完后填）

### Task 1 — 单卡 fp32：在哪一步炸显存

### Task 2 — FSDP full-shard fp32：装得下吗

### Task 3 — FSDP full-shard fp16：保底数据

### Task 4 — FSDP 的 comm_fraction

### 复盘：预测对了什么、错了什么
