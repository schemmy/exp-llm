# Week 9 — Tensor Parallelism

**Dates**: 2026-10-25 → 2026-10-31  
**Goal**: 理解一个模型如何被切开放到多张卡上，实测 TP 的收益和它的代价（通信）。

**Time budget**: 4-6 小时 ｜ **GPU 成本**: ~$3

---

## 为什么这周存在

前 8 周所有实验都在**一张卡**上。但真实场景里，你要服务的模型大概率装不下单卡——
70B fp16 是 140 GB，1T MoE 是 1 TB。这时候唯一的办法是**把模型本身切开**。

这也是 P1（单卡调优）和 P3（K8s 编排）之间缺的那一环：
P3 要调度的最小单位不是"一个容器"，而是"**4 张 NVLink 互联的卡**"——那比调度单卡副本难得多。

---

## 原理：Megatron 式切分

对一个线性层 `Y = XW`，有两种切法：

**列并行（column parallel）**：按列切 W = [W₁ | W₂]
- GPU i 算 `XWᵢ`，得到输出的一部分
- 各自独立，**不需要通信**
- 但结果是分片的，下一层要用就得拼

**行并行（row parallel）**：按行切 W，同时按列切 X
- GPU i 算 `XᵢWᵢ`，得到**部分和**
- 必须 **all-reduce** 把部分和加起来才是正确结果

**Megatron 的关键技巧**：把两者串起来，让中间那次通信省掉。

```
MLP:   [列并行 up_proj] → GELU → [行并行 down_proj] → all-reduce
                ↑ 输出是分片的，正好直接喂给行并行的输入，中间不用通信

Attn:  [列并行 QKV] → attention → [行并行 out_proj] → all-reduce
                ↑ 每张卡拿一部分 head，独立算完自己的 attention
```

所以**每层只需 2 次 all-reduce**（attention 一次、MLP 一次），不是 4 次。
Qwen2.5-7B 有 28 层 → **每次 forward pass 有 56 次 all-reduce**。

---

## 核心假设

**TP 对 decode 应该是大幅加速**，理由回到 Wk 8 那个结论：decode 是**带宽瓶颈**。

TP=2 时每张卡只需读**一半权重**（7 GB 而不是 14 GB），搬运时间直接减半。
这不是"多张卡算得快"，是"**每张卡要搬的东西变少了**"。

代价是 56 次 all-reduce 的延迟。batch=1 时每次 all-reduce 的载荷极小
（hidden 3584 × 2 bytes ≈ 7 KB），所以瓶颈是**延迟不是带宽**——NVLink 单次约几微秒。

| TP | 每卡权重 | 每 forward 的 all-reduce | 预期 |
|----|---------|------------------------|------|
| 1 | 14 GB | 0 | baseline |
| 2 | 7 GB | 56 | 接近 2x，但被通信吃掉一部分 |
| 4 | 3.5 GB | 56（载荷更小，跳数更多） | 明显低于 4x |

---

## 一个硬约束：TP 上限由 KV head 数决定

Qwen2.5-7B 的配置：
- `num_attention_heads = 28`
- `num_key_value_heads = 4` ← GQA

列并行是**按 head 切**的，所以 **TP 必须整除 KV head 数**。

→ **Qwen2.5-7B 的合法 TP 只有 1 / 2 / 4。TP=8 会直接报错。**

这是个很实际的约束：你不能因为有 8 张卡就开 TP=8，模型架构说了算。

---

## Tasks

### Task 1 — TP sweep (90 min)

- [ ] 跑 `experiments/wk09_tensor_parallel/tp_sweep.py`
- [ ] TP = 1 / 2 / 4，每个测 batch = 1 / 8 / 32
- [ ] 记录 tok/s、TPOT

**要回答的问题**：
1. TP=2 的加速比是多少？离 2x 差多远？
2. batch=1 和 batch=32 哪个 TP 收益更大？为什么？
3. TP=4 相比 TP=2 的边际收益如何？

---

### Task 1.5 — 踩到的坑：Python 3.11 跑不了 TP>1

第一次跑 TP=2 时，两个 worker 都死在 `init_device`：

```
flashinfer/comm/fd_exchange.py line 55
    def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
TypeError: type 'array.array' is not subscriptable
```

`array.array[int]` 这个泛型写法 **Python 3.12 才合法**（PEP 585 对 `array.array` 的支持是 3.12 加的）。
它写在**函数注解**里，模块导入时就求值，所以 3.11 直接抛错。

链路：TP>1 → 构造 `cuda_communicator` → 导入 `flashinfer_all_reduce` → 导入 `flashinfer.comm` → 炸。

**为什么 Wk 1-8 从没遇到**：单卡不走 `cuda_communicator`，永远不会导入这个模块。
这个 bug 只在多卡路径上存在。

**修复**：wk09 的镜像用 `add_python="3.12"`。三档 TP 共用同一镜像，保证对比受控。

（顺带确认 `/dev/shm` 有 80 GB，不是共享内存不足——那是我最初的错误猜测。）

---

### Task 2 — 验证 TP=8 会失败 (15 min)

- [ ] 手动改成 `tensor_parallel_size=8` 跑一次（或直接读 vLLM 源码的校验逻辑）
- [ ] 记录报错信息，确认是 KV head 整除约束

---

### Task 3 — 理解通信拓扑 (30 min)

- [ ] 在容器里跑 `nvidia-smi topo -m`，看卡间是 NVLink 还是 PCIe
- [ ] 思考：如果 TP 跨机器（NVLink 900 GB/s → InfiniBand 50 GB/s），会发生什么？

---

### Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk9 行
- [ ] 把结果填进 `viz/phase1.html` 第 06 幕的"待测"行
- [ ] commit + push

---

## What "done" for Wk 9 looks like

1. 有 TP × batch 的完整表格
2. 能解释 TP 为什么对 decode 特别有效（带宽，不是算力）
3. 知道 TP 的上限由什么决定，以及为什么不能跨机器开 TP

---

## 结果（跑完后填）

| TP | batch=1 | batch=8 | batch=32 | 每卡权重 |
|----|---------|---------|----------|---------|
| 1 | | | | 14 GB |
| 2 | | | | 7 GB |
| 4 | | | | 3.5 GB |

加速比（相对 TP=1）：

| TP | batch=1 | batch=8 | batch=32 |
|----|---------|---------|----------|
| 2 | | | |
| 4 | | | |
