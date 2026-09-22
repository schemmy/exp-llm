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

- [x] 加载 Qwen2.5-7B-Instruct（fp32，复用 `hf-cache` volume）到 1 张 A100-80GB
- [x] 依次尝试：模型加载 → 建 `AdamW` optimizer → 一次 forward → 一次
      backward → 一次 `optimizer.step()`，每一步都 try/except 包起来
- [x] 无论在哪一步失败，都记录：到达的最后一步 + 失败前
      `torch.cuda.max_memory_allocated()`

**预期**：模型加载（30.4GB）大概率能成功；forward 应该也能过（激活值在
batch=1 时不大）；backward 后梯度落地会推到 ~60.8GB，可能还没炸；
`optimizer.step()` 第一次调用时 Adam 才真正分配 `exp_avg`/`exp_avg_sq`
（fp32，各 30.4GB），这一步大概率是真正炸显存的地方。但这只是猜测，
具体在哪步炸、炸之前实际吃了多少显存，是这个 Task 要测的东西，不是
预设的结论。

---

## Task 2 — FSDP full-shard（fp32），2× A100，batch=1

- [x] `FullyShardedDataParallel`，`sharding_strategy=FULL_SHARD`（ZeRO-3
      等价：权重、梯度、优化器状态全部切分）
- [x] 跟 Task 1 完全相同的 fp32 精度，方便直接对比"同样的精度设置，
      切了 vs 没切"的显存差距
- [x] 测：per-GPU 显存峰值、tokens/sec

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

- [x] 跟 Task 2 完全一样，只是模型直接以 `torch_dtype=torch.float16`
      加载（不是 fp32 转 fp16，是从一开始就以 fp16 常驻）——权重、梯度、
      AdamW 状态全部走 fp16 存储
- [x] 理论静态显存：(15.2+15.2+30.4) / 2 ≈ 30.4GB per GPU，应该很宽松地
      装得下
- [x] 测：per-GPU 显存峰值、tokens/sec，跟 Task 2 对比"精度减半，显存
      差多少"

这一档是这周的"保底"——不管 Task 2 是成是败，Task 3 大概率能跑成功，
保证这周至少有一组完整可用的 FSDP 吞吐数字，也顺带量化了混合精度对
显存的真实收益（不是理论上的 2x，因为激活值、通信 buffer 等不完全
随精度线性缩放）。

---

## Task 4 — profiler：FSDP 的通信开销多大

- [x] 对 Task 2/Task 3 里跑成功的那一档（优先 Task 3，因为更可能成功），
      抓 `torch.profiler` trace
- [x] FSDP 的通信原语跟 DDP 不一样——DDP 只有 all-reduce，FSDP 还有
      **all-gather**（forward/backward 前临时聚合当前层的完整参数）和
      **reduce-scatter**（反向传播后把梯度切分聚合），事件名里找
      `all_gather` / `reduce_scatter` / `nccl`
- [x] 算一个 comm_fraction，跟 Wk 15 的 DDP comm_fraction（4.41% @ 124M）
      放在一起看——FSDP 因为多了 all-gather，comm_fraction 大概率明显
      更高，这是"切分换显存"要付的通信代价，值得量化出来而不是只停留在
      "应该更贵"这种定性判断

---

## Task 5 — 日志

- [x] 更新 `progress.md` Wk 17 那一行
- [x] `benchmarks/README_training.md` 加 Wk 17 结果表
- [x] commit + push（每个文件单独一次）

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

## 结果

### Task 1 — 单卡 fp32：在哪一步炸显存

**预测精确命中**：模型加载成功、optimizer 建好成功、forward 成功、
backward 也跑完了（`stage_reached: backward_done`）——真正炸的地方是
**`optimizer.step()` 第一次调用**（Adam 这时才懒加载 `exp_avg`/`exp_avg_sq`）。
炸之前吃到 **78.14 GiB**（卡总容量 79.25 GiB，只剩 181.94 MiB 空闲），
跟week17.md 开头算的 ~121.6GB 理论需求量级吻合——backward 结束时
权重+梯度已经占了 ~60.8GB，Adam 状态哪怕只分配了一小部分张量就直接
顶到墙。

### Task 2 — FSDP full-shard fp32：装得下吗——装是装下了，但不轻松

| | 值 |
|---|---|
| tokens/sec | 238.3 |
| step time | 2148.99ms |
| 显存峰值 per GPU | 76.82 GB = 71.54 GiB |
| 剩余余量 | ~7.7 GiB（卡总量 79.25 GiB 的 ~9.7%） |

**能跑，回答了这周最想要的问题之一**：同样 fp32 精度，单卡装不下的东西，
纯靠 FULL_SHARD 切成两份就能塞进 80GB——这是 FSDP"解决显存问题"最直接的
证据。

**但日志里有个不该跳过的细节**：这一档从模型加载完到存结果，中间
~2 分 27 秒里**持续不断**地打印显存分配失败又重试成功的警告（`memory
allocation failed with OOM ... ` 后紧跟着继续跑下去），时间跨度覆盖了
从 warmup 到 measure 的几乎每一步。换算一下：70 步 × ~2.1s/步 ≈ 150s，
跟警告持续的时间基本对上——**说明不是某一步偶然卡了一下，是几乎每一步
都在跟显存上限打架**，靠 PyTorch caching allocator 的重试机制勉强撑过去。
最终报出来的 238.3 tok/s 是真实测到的数字，但这不是一次"轻松装下"的
干净运行，是一次"卡着边缘反复挣扎但没有真的崩"的运行——这个区别值得
记下来，不能只看最后那行数字。

### Task 3 — FSDP full-shard fp16：保底数据，而且干净利落

| | 值 |
|---|---|
| tokens/sec | 1,098.4 |
| step time | 466.13ms |
| 显存峰值 per GPU | 38.33 GB = 35.69 GiB |
| comm_fraction | 22.91% |

日志里**没有任何 OOM 重试警告**——相比 Task 2 卡在边缘反复挣扎，这一档
是舒舒服服跑完的，剩余余量超过 40 GiB。

**跟 fp32 一对比，两个数字都很干净**：
- **显存**：38.33 / 76.82 = **49.9%**——几乎精确地"减半"，理论预期
  （fp16 是 fp32 字节数的一半）被结结实实地验证了
- **吞吐**：1,098.4 / 238.3 = **4.61x**——比纯"字节减半"能解释的幅度大
  不少，说明除了搬得少，fp16 tensor core 算得也比 fp32 CUDA core 快；
  另外 Task 2 那种反复 OOM 重试本身也在拖慢 fp32 的实际墙钟时间，
  所以这 4.61x 里有多少是"精度带来的真实计算/带宽收益"、多少是
  "fp32 那边被重试拖累"，两者搅在一起，**没法从这两个数字里干净拆开**——
  如实记录这个局限，不硬拆成一个精确的归因

### Task 4 — FSDP 的 comm_fraction：22.91%，比 DDP 高出一截

Wk 15 的 DDP comm_fraction（124M 模型）是 4.41%；这次 FSDP（7B 模型）
是 **22.91%**，高了 ~5.2 倍。方向上支持 week17.md 的预期——FSDP 比 DDP
多了 all-gather（每次算之前先把当前层完整参数聚合回来），通信原语
比 DDP 单纯的一次 all-reduce 更贵。

**但这不是一个干净的对照实验**：变了两个变量，不只是"通信机制从
all-reduce 换成 all-gather+reduce-scatter"，模型大小也从 124M 变成了
7B（参数量 ~61 倍），层数、通信次数、单次通信的数据量全都不一样。
**"FSDP 比 DDP 通信贵 5.2 倍"这个说法不成立**——成立的只是"这次测到的
FSDP 通信占比确实比那次测到的 DDP 通信占比高不少，机制上也确实解释得通
为什么会更贵"，两回事，后者是能站住脚的结论，前者是过度引申。

### 复盘：预测对了什么、错了什么

**预测对的部分**：
1. 单卡 fp32 Qwen2.5-7B 装不下——而且不仅方向对了，连**具体在哪一步
   炸**（`optimizer.step()` 首次调用）都精确命中
2. fp16 FULL_SHARD 是"保底"选项——干净跑完，零 OOM 警告，符合预期

**没预测到、算是意外发现的部分**：
1. fp32 FULL_SHARD **居然真的跑成功了**——week17.md 写这个 Task 的时候
   把"跑不成功"当成一个同样有信息量的可能结果认真准备了，结果它成功了，
   但成功的方式（持续贴着显存上限反复重试）比"成功"或"失败"这个二元
   判断本身更有信息量
2. fp32→fp16 的 4.61x 吞吐提升幅度超过纯字节减半能解释的范围，但因为
   fp32 那组数据本身混杂了 OOM 重试的干扰，没法干净拆解这个超出部分
   有多少是"真实精度收益"、多少是"重试开销"——如实标注为未拆解，
   不强行给出一个精确的归因数字

**给 Wk 18 的钩子**：这周确认了 fp32 FULL_SHARD 技术上能跑，但是贴着
显存墙反复挣扎；Wk 18 要对比 ZeRO-1/ZeRO-2/ZeRO-3（对应 FSDP 不同的
`sharding_strategy`）——**ZeRO-1/2 切得比 FULL_SHARD（ZeRO-3）少**，
显存只会比这周的 76.82GB 更紧，大概率在 fp32 下直接 OOM，没有"贴着边缘
硬撑"的余地。这意味着 Wk 18 的 ZeRO 档位对比可能得整体切到 fp16 精度下
才能公平地把三档都跑出结果——这是这周的数据直接决定 Wk 18 该怎么设计，
不是凭空猜的。

### 复盘：预测对了什么、错了什么
