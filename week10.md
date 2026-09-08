# Week 10 — MoE 推理 + Expert Parallelism

**Dates**: 2026-11-01 → 2026-11-07  
**Goal**: 实测 MoE 的核心交易——用大模型的显存换小模型的速度，以及 EP 和 TP 的区别。

**Time budget**: 4-6 小时 ｜ **GPU 成本**: ~$3

---

## 为什么这周存在

前沿模型基本都是 MoE：DeepSeek-V3（671B/37B 激活）、Kimi K2（1T/32B 激活）。
你 Wk 9 算过的那笔账——1T 参数要 16 张 H100——**存储看总参数，速度看激活参数**，
这个分裂是 MoE 推理的全部要点。

---

## 原理

稠密模型每个 token 都要过**全部** FFN 权重。MoE 把 FFN 换成 N 个专家 + 一个路由器：
每个 token 只被送去 **top-k 个专家**（通常 k=2~8），其余专家这一步完全不参与计算。

```
稠密 FFN:     x → [一个巨大的 FFN] → y          每 token 读全部 FFN 权重
MoE FFN:      x → router → 选 top-k 个专家 → y   每 token 只读 k/N 的 FFN 权重
```

**关键的不对称**：

| | 看什么 | 为什么 |
|---|--------|--------|
| **显存占用** | **总参数量** | 无法预知下个 token 激活哪些专家，全部专家必须常驻 |
| **decode 速度** | **激活参数量** | 每步只需把激活的那部分权重从 HBM 搬进 SM |

Wk 8 已经证明 decode 卡在带宽上。MoE 正是直接砍这个带宽需求——
**但砍的是"每 token 要读多少"，不是"要存多少"**。

---

## Task 1 — MoE vs 稠密：三方对比 (90 min)

单卡，同一个 Qwen1.5 家族，三个模型：

| 模型 | 总参数 | 激活参数 | 角色 |
|------|--------|---------|------|
| Qwen1.5-1.8B | 1.8B | 1.8B | 激活量对照 |
| **Qwen1.5-MoE-A2.7B** | **14.3B** | **2.7B** | 被测对象 |
| Qwen1.5-14B | 14B | 14B | 总参数量对照 |

- [ ] 跑 `experiments/wk10_moe/moe_vs_dense.py`
- [ ] 记录 tok/s、TPOT、显存占用

**假设**：MoE 的速度应该**接近 1.8B**（只读激活权重），
显存应该**接近 14B**（全部专家常驻）。

**但不会完全等于 1.8B**，因为：
1. 路由器本身有开销
2. 共享专家（shared expert）每步都激活
3. MoE 的 grouped GEMM + scatter/gather 比稠密 GEMM 的 kernel 效率低

**真正要看的是 MoE 落在 1.8B 和 14B 之间的哪个位置。**
越靠近 1.8B，说明"用显存换速度"这笔交易做得越成功。

---

## Task 2 — Expert Parallelism vs Tensor Parallelism (60 min)

多卡切 MoE 有两种切法，**切的维度完全不同**：

**TP（张量并行）**：把**每个专家**的权重按列/行切开
- 每张卡都持有所有专家的一部分
- 每个专家的计算都需要全部 GPU 参与 + all-reduce
- 通信模式和稠密模型一样

**EP（专家并行）**：把**专家本身**分给不同 GPU
- GPU 0 拿专家 0-29，GPU 1 拿专家 30-59（完整的专家）
- 路由器决定 token 去哪，然后 **all-to-all** 把 token 发到对应 GPU
- 算完再 all-to-all 送回来

```python
LLM(model=..., tensor_parallel_size=2)                            # 纯 TP
LLM(model=..., tensor_parallel_size=2, enable_expert_parallel=True)  # TP+EP
```

- [ ] 跑 `experiments/wk10_moe/expert_parallel.py`
- [ ] 对比 TP=2 vs TP=2+EP 的吞吐

**EP 的取舍**：
- ✅ 省显存（每卡只存 1/N 的专家），且专家计算是完整的稠密 GEMM，kernel 效率高
- ❌ all-to-all 通信量大，且**负载可能不均**——热门专家所在的 GPU 会成为瓶颈

**规模决定谁赢**：专家少的时候 TP 更简单；专家多到几百个（DeepSeek-V3 有 256 个）
EP 几乎是唯一选择，否则每个专家被切得太碎，GEMM 退化成小矩阵乘。

---

## Task 3 — 理解负载不均 (30 min)

- [ ] 思考：如果 90% 的 token 都路由到同一个专家，EP 会发生什么？
- [ ] 查一下什么是 EPLB（Expert Parallelism Load Balancer）——
      Wk 9 的日志里其实出现过 `EPLB rank N/A` 这一项

---

## Task 4 — 日志 (10 min)

- [ ] 更新 `progress.md` Wk10 行
- [ ] 填 `viz/phase1.html` 第 06 幕的 MoE 行
- [ ] commit + push

---

## What "done" for Wk 10 looks like

1. 有 MoE / 小稠密 / 大稠密 的三方数字，能说清 MoE 落在哪
2. 有 TP vs EP 的对比，能解释两者切的维度不同
3. 知道 EP 的失效模式是负载不均，不是通信量

---

## 结果（跑完后填）

### Task 1 — 单卡三方对比

| 模型 | 总/激活参数 | batch=1 | batch=32 |
|------|-----------|---------|----------|
| Qwen1.5-1.8B | 1.8B / 1.8B | 314.5 tok/s | 7,312.9 tok/s |
| **Qwen1.5-MoE-A2.7B** | **14.3B / 2.7B** | **228.5 tok/s** | **1,829.6 tok/s** |
| Qwen1.5-14B | 14B / 14B | 54.5 tok/s | 1,487.3 tok/s |

**MoE 在两个对照之间的位置**（0% = 和大稠密一样慢，100% = 和小稠密一样快）：

| batch | 位置 | vs 14B 稠密 | vs 1.8B 稠密 |
|-------|------|------------|-------------|
| 1 | **67%** | 4.19x 更快 | 0.73x |
| 32 | **6%** | 1.23x 更快 | 0.25x |

> 权重显存那一列这次没测到（全是 0.0 GiB）：`torch.cuda.memory_allocated()` 跑在父进程里，
> 而模型实际住在 EngineCore 子进程。已改成直接统计 safetensors 文件体积，见脚本。

---

## 核心发现：MoE 的优势随 batch 增大而蒸发

从 **67% 掉到 6%**。这不是噪声，是机制性的。

### 为什么

回到 GEMM 的形状。**batching 恰恰是 MoE 唯一无法利用的东西。**

| | 稠密模型 | MoE |
|---|---------|-----|
| batch=1 | 一个 `M=1` 的 GEMM | 一个 `M=1` 的 GEMM，但只读 2.7B 权重 |
| batch=32 | 一个 `M=32` 的 GEMM ✅ | **60 个 `M≈2` 的小 GEMM** ❌ |

**batch=1 时**：两边的 GEMM 形状同样糟（M=1），谁也没法用 tensor core。
此时纯粹比拼**要搬多少权重**——MoE 只读 2.7B 而 14B 稠密要读 14B，
所以 MoE 快 4.19x。这是 Wk 8 那条"decode 卡带宽"的直接推论。

**batch=32 时**：稠密模型把 32 个 token 攒成一个 `M=32` 的 GEMM，
算术强度涨 32 倍，tensor core 终于被喂饱。

但 MoE 把这 32 个 token **散到 60 个专家上**——32 token × top-4 = 128 个配对，
平均每个专家只分到 **约 2 个 token**。稠密模型的 `M=32` 在 MoE 这里变成了
60 个 `M≈2`。**batch 增大带来的 GEMM 效率红利，MoE 几乎一点都吃不到。**

于是 MoE 只比 14B 稠密快 1.23x，而 1.8B 稠密已经跑到它的 4 倍。

### 生产含义

**MoE 需要**远**大于稠密模型的 batch 才能发挥。**

要让每个专家也拿到 `M=32`，总 batch 需要约 `32 × 60 / 4 ≈ 480`。
这就是为什么前沿 MoE 服务都跑在极高并发上——不是为了摊薄成本，
是**专家级别的 batch 不够大的话，MoE 的 GEMM 根本喂不饱 GPU**。

这也预测了 Task 2 的结果方向：**EP 在大 batch 下应该比 TP 更有优势**，
因为 EP 让每张卡专注于自己那份专家，专家级 batch 相对更大。

---

## 顺带验证：decode 速度和权重体积成反比

| 模型 | 激活参数 | batch=1 tok/s |
|------|---------|--------------|
| Qwen2.5-7B（Wk 9） | 7B | 97.2 |
| Qwen1.5-14B | 14B | 54.5 |

参数量 2x，速度 1.78x 慢——**接近线性**。

这是对"decode 是带宽瓶颈"最干净的一次独立验证：
搬运时间正比于权重体积，而搬运时间就是 decode 时间。

### Task 2 — TP vs EP（Qwen1.5-MoE-A2.7B，2×A100）

| 配置 | batch=1 | batch=32 | 每卡权重 |
|------|---------|----------|---------|
| TP=2 | | | |
| TP=2 + EP | | | |
