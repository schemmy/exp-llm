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

- [x] 跑 `experiments/wk09_tensor_parallel/tp_sweep.py`
- [x] TP = 1 / 2 / 4，每个测 batch = 1 / 8 / 32
- [x] 记录 tok/s、TPOT

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

- [x] 确认 KV head 整除约束 —— TP=4 时每卡恰好 1 个 KV head，TP=8 需要 0.5 个，不可能
- [ ] （可选）实跑 `test_tp8_fails()` 拿到确切报错文本

---

### Task 3 — 理解通信拓扑 (30 min)

- [x] 在容器里跑 `nvidia-smi topo -m` —— **失败**，Modal 容器里报 `Failed to run topology matrix`，
      拓扑未能确认。留作未解问题。
- [x] 思考：如果 TP 跨机器（NVLink 900 GB/s → InfiniBand 50 GB/s），会发生什么？
      →  本机 TP=4 已经只有 54% 效率；跨机带宽降 18 倍，56 次 all-reduce 会直接主导整个
         forward pass。这就是为什么 TP 几乎从不跨节点，跨节点用的是 pipeline parallelism
         （每个 micro-batch 只传一次激活，通信量小得多）。

---

### Task 4 — 日志 (10 min)

- [x] 更新 `progress.md` Wk9 行
- [x] 把结果填进 `viz/phase1.html` 第 06 幕的"待测"行
- [x] commit + push

---

## What "done" for Wk 9 looks like

1. 有 TP × batch 的完整表格
2. 能解释 TP 为什么对 decode 特别有效（带宽，不是算力）
3. 知道 TP 的上限由什么决定，以及为什么不能跨机器开 TP

---

## 结果（跑完后填）

环境：Python 3.12，vLLM v0.28.0，A100-80GB，fp16，输出 256 tokens。
**注意**：这组数字不能和 Wk 3-8 直接比——Python 版本、vLLM 版本、输出长度都变了，
而且 v0.28 默认开启了 prefix caching 和 chunked prefill。组内对比才有效。

### 吞吐（tok/s）

| TP | batch=1 | batch=8 | batch=32 | TPOT (b=1) | 每卡权重 |
|----|---------|---------|----------|-----------|---------|
| 1 | 97.2 | 786.3 | 2,950.2 | 10.3 ms | 14.29 GiB |
| 2 | 150.6 | 1,101.6 | 4,501.9 | 6.6 ms | 7.16 GiB |
| 4 | 210.0 | 1,727.7 | 6,283.3 | 4.8 ms | 3.63 GiB |

### 加速比（相对 TP=1）

| TP | batch=1 | batch=8 | batch=32 | 并行效率 |
|----|---------|---------|----------|---------|
| 2 | **1.55x** | 1.40x | 1.53x | ~75% |
| 4 | **2.16x** | 2.20x | 2.13x | ~54% |

### KV cache 容量（第二个收益，比吞吐更明显）

| TP | 每卡 KV 显存 | 总 token 容量 | 最大并发 |
|----|-------------|--------------|---------|
| 1 | 56.21 GiB | 1,052,560 | 32.1x |
| 2 | 64.94 GiB | 2,431,792 | 74.2x |
| 4 | 68.56 GiB | 5,135,248 | 156.7x |

---

## 结论

### 1. 加速比远低于 TP 倍数，而且缺口稳定

TP=2 只有 1.4-1.55x，TP=4 只有 2.13-2.20x。**这个缺口就是通信成本**，
每次 forward 要跑 56 次 all-reduce（28 层 × 2 次）。

值得注意的是**并行效率在三个 batch 上几乎不变**（TP=2 恒在 ~75%，TP=4 恒在 ~54%）。
说明通信开销和计算量是等比例增长的——batch 变大，载荷变大，但计算也变多，比例守恒。
这和 Wk 8 投机解码那种"随 batch 反转"的行为完全不同。

### 2. 每卡吞吐是**下降**的 —— TP 买的是延迟，不是性价比

| TP | 总吞吐 (b=32) | 每卡吞吐 |
|----|--------------|---------|
| 1 | 2,950 | **2,950** |
| 2 | 4,502 | 2,251 |
| 4 | 6,283 | 1,571 |

用 4 张卡只拿到 2.13x，**每张卡的产出掉到单卡的 53%**。

→ **生产判断**：TP 只在两种情况下开——(a) 模型装不下单卡，(b) 需要更低的 TPOT。
纯粹追求吞吐/成本时，**开 4 个 TP=1 的副本比开 1 个 TP=4 划算得多**。

### 3. KV cache 容量的增长快过吞吐

token 容量 1.05M → 2.43M → 5.14M（2.3x / 4.9x），最大并发 32x → 157x。

两个原因叠加：权重被切走后每卡腾出显存，同时 KV head 也被切分
（4 个 KV head：TP=2 每卡 2 个，TP=4 每卡 1 个），每 token 的 KV 开销也随之下降。

**这是 TP 被低估的收益**：不只是跑得快，是能同时装下多得多的请求。
长上下文场景（Wk 6 提到的"要长 prompt 才能打满"）这一点尤其关键。

### 4. 顺带确认了 TP=4 就是这个模型的上限

TP=4 时每卡恰好分到 **1 个 KV head**。TP=8 需要每卡 0.5 个 —— 不可能。
硬约束从日志里直接可见，不用等报错。

---

## 环境细节

- `SymmMemCommunicator: Device capability 8.0 not supported` —— A100 是 sm80，
  用不了对称内存 all-reduce，实际走的是 `['CUSTOM', 'PYNCCL']`。H100 (sm90) 上会更快。
- `nvidia-smi topo -m` 在容器里报 `Failed to run topology matrix`，
  **没能确认卡间是 NVLink 还是 PCIe**。Task 3 这条留作未解。
- torch.compile 时间随 TP 增长：19s (TP=1) → 68s (TP=2) → 107s (TP=4)，
  因为每个 rank 都要各自编译一遍。

加速比（相对 TP=1）：

| TP | batch=1 | batch=8 | batch=32 |
|----|---------|---------|----------|
| 2 | | | |
| 4 | | | |
