# Week 16 — Bucket 大小重要吗？以及重要的门槛在哪

**Dates**: 2026-12-13 → 2026-12-19
**Goal**: 回答 Wk 15 结尾留下的问题——DDP 的通信/反向重叠在 GPT-2 124M 这个
体量下已经接近饱和（scaling efficiency 96.6%-99.96%），那 `bucket_cap_mb`
这个调重叠粒度的参数**在这个体量下还测不测得出差别**？如果测不出，
**多大的模型才能让它测出来**？

**Time budget**: 4-5 小时 ｜ **GPU 成本**: ~$10-14（Modal 上 2× A100，
两档模型 × 4 档 bucket size）

---

## 从 Wk 15 带过来的钩子

Wk 15 的结论：GPT-2 124M 的梯度体积（~500MB fp32）相对于每步的算量太小，
DDP 默认的 bucket 机制（25MB 一个 bucket，反向传播算完一个 bucket 就
立刻发它的 all-reduce，不用等整个反向传播结束）已经把几乎全部通信时间
藏在了计算背后——comm_fraction 只有 4.41%，实测吞吐损失只有 1.15%。

这周直接问：**调 `bucket_cap_mb` 这个参数，在这个已经快到 100% 效率的
场景下还有没有意义？**

两种可能，都值得记录：
- **没意义**：如果重叠已经饱和，bucket 切多大都不会改变最终吞吐——
  这本身就是一个诚实的负面结果，直接印证 Wk 15 的假设
- **有意义但方向反直觉**：比如 bucket 太小导致 all-reduce 调用次数暴涨，
  每次调用的固定开销（kernel launch、NCCL 握手）开始压过通信本身，
  于是"更细粒度的重叠"反而更慢——这是教科书上会提到但没亲手测过的效应

---

## 为什么要把模型从 124M 换成 774M（GPT-2 Large）

如果 Task 1（还在 124M 上测）真的测不出差别，光停在"测不出"没法往下走——
需要一个**加大梯度体积**的对照实验，才能回答"多大才测得出"。

GPT-2 Large（36 层、1280 hidden、20 heads，774M 参数）梯度体积 ~3.1GB，
是 124M 的 ~6 倍，但仍然稳稳装进一张 A100-80GB 做 DDP（不需要 FSDP 分片——
这是 Wk 17 的题目）。如果 bucket 大小在 774M 上开始有可测的影响，
而在 124M 上没有，那就直接定位了"重叠饱和"这件事跟模型大小的关系，
也顺带给 Wk 17 的 FSDP 铺了一个自然的台阶：**当模型大到连这种重叠
技巧都不够用的时候，才轮到分片出场**。

---

## Task 1 — GPT-2 124M 上先测一遍：bucket_cap_mb 有没有可测的影响

- [ ] 复用 `experiments/wk15_ddp/ddp_scaling.py` 里的 `ddp_worker`/`run_steps`，
      加一个 `bucket_cap_mb` 参数传给 `DDP(...)` 构造函数
- [ ] 固定 batch=16（Wk 15 的中间档，也是抓过 profiler trace 的那档，
      方便直接对比）
- [ ] `bucket_cap_mb` 扫 4 个值：**1 / 25（默认）/ 100 / 500（约等于全模型
      一个 bucket，因为整个模型梯度只有 ~500MB）**
- [ ] 每档都测 tokens/sec + `all_reduce` 调用次数（`torch.profiler` 里数
      nccl 相关事件的 `count`）

**预期**：如果 Wk 15 的假设对，四档吞吐应该几乎一样（重叠已经饱和，
切多细都无所谓）。`all_reduce` 调用次数应该随 bucket 变小而增多——
这个是必然的，机制层面就是这样，用来验证测量方法本身没错，不是用来
验证假设的。

---

## Task 2 — GPT-2 Large（774M）上重复同一个 sweep

- [ ] 同样的四档 `bucket_cap_mb`（1 / 25 / 100 / 500），换成 GPT-2 Large
- [ ] batch 只测一档（16，per-GPU）——这周的变量是 bucket size 和模型大小，
      不是 batch，没必要重新扫三档 batch 把成本拉高
- [ ] 顺带测一下 774M 在默认 bucket_cap_mb=25 下的 scaling efficiency，
      跟 Wk 15 的 124M 数字放在一起看——**这是本周最想要的一个数字**：
      同样是"重叠机制"，模型大 6 倍之后 efficiency 还能不能维持在
      99% 附近？

**预期**：774M 的 efficiency 应该比 124M 略低（梯度体积变大，需要重叠的
通信量变多），且 bucket_cap_mb 在这个体量上开始表现出可测的差异——
太小的 bucket（1MB）应该因为 all-reduce 调用次数太多、单次开销占比升高
而变慢；默认或更大的 bucket 应该更接近理想。

---

## Task 3 — 汇总：重叠饱和的门槛在哪

- [ ] 一张表：模型大小（124M / 774M）× bucket_cap_mb（4 档）→ tokens/sec
      + scaling efficiency
- [ ] 回答本周开头的问题：bucket 大小什么时候开始重要，什么时候不重要
- [ ] 如果两档模型都测不出 bucket_cap_mb 的影响——那也是一个完整的结论，
      如实记录，不要为了"有发现"而过度解读噪声

---

## Task 4 — 日志

- [ ] 更新 `progress.md` Wk 16 那一行
- [ ] `benchmarks/README_training.md` 加 Wk 16 结果表
- [ ] commit + push（每个文件单独一次）

---

## What "done" for Wk 16 looks like

1. 有 124M 和 774M 两个模型规模下、各 4 档 bucket_cap_mb 的吞吐数字
2. 明确回答了"bucket 大小在什么条件下值得调"
3. 774M 的 scaling efficiency 数字，跟 Wk 15 的 124M 放在一起能看出
   "模型越大、DDP 重叠机制的余量越小"这条趋势是否成立
4. 为 Wk 17 FSDP 做了一个自然的铺垫——如果连 774M 都还能被 DDP 的
   重叠机制撑住，那"什么时候必须用 FSDP"这个问题本身也需要重新想清楚
   （答案大概率不是"模型大了 DDP 就不够用"，而是"模型大到一张卡装不下"
   ——FSDP 解决的是显存问题，不是通信效率问题，这个区分值得在 Wk 17
   开头讲清楚）

---

## 结果（跑完后填）

### Task 1 — GPT-2 124M：bucket_cap_mb sweep

### Task 2 — GPT-2 Large 774M：bucket_cap_mb sweep + scaling efficiency

### Task 3 — 汇总与结论

### 复盘：预测对了什么、错了什么
