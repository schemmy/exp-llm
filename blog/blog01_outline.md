# Blog #1 大纲 — 待你审

**目标站点**：chenxin-ma.github.io ｜ **目标发布**：Wk 14（~2026-12-05）
**预计篇幅**：3,500-4,500 词，长文深度版
**素材来源**：`benchmarks/README.md`（全部数据表）、`viz/phase1.html`（Act 1-5 交互图）、
`viz/week11_roofline.html`（roofline + CUDA graph + 量化图）

---

## 候选标题（选一个或提新的）

1. *Benchmarking vLLM: Single GPU to Multi-GPU* —— PLAN.md 原定题目，中性、准确
2. *What Actually Makes LLM Inference Fast* —— 更吸引点击，突出"实测驱动"
3. *12 Weeks, One A100: A Roofline Tour of vLLM* —— 强调方法论（roofline 框架）和约束（单卡起步）
4. *The GPU Was at 100% Util and 0.47% MFU* —— 用开篇那个反例直接当标题，最挑衅但最抓人

**我的推荐**：3。理由——"roofline tour"准确描述了全文的组织逻辑（每节都在同一张图上挪位置），
"12 周、一张 A100"点出这是真实的、可复现的实验记录，不是综述。2 和 4 更像标题党，
和这个仓库"不用面试/营销语气"的既有原则（`feedback_repo_tone.md`）不太搭。

---

## 全文骨架：镜像 `viz/phase1.html` 的 Act 结构

读者读完文章，点进交互图，应该感觉是同一套心智模型——不是另一份摘要。

### 开篇（~300 词）

**钩子**：Wk 8 那个反例——同一时刻，`nvidia-smi` 显示 GPU-Util 100%，
但 MFU 只有 0.47%，MBU 却有 73%。三个数字同时成立，讲的是三件不同的事。

**这句话就是全文的论点**：LLM 推理的每一项优化，本质上都在做同一件事——
**提高算术强度（让每次搬运的权重服务更多计算），或者减少要搬运的字节数**。
搞懂这条轴，10 项看似无关的技术（batching、caching、投机解码、并行、量化）
会变成同一个故事的不同章节。

**范围声明**：Qwen2.5-7B-Instruct，A100 80GB，vLLM，12 周，全部可复现
（链接到 `experiments/` 和 `modal run` 命令）。

---

### Act 0 — 两个阶段的物理学（~250 词）

- Prefill：算力密集，一次算完整个 prompt
- Decode：带宽密集，每步只出 1 个 token 却要搬全部权重
- 引出 roofline 的基本词汇：算术强度、MFU、MBU
- **数据**：Wk 8 的 80.6 tok/s、TPOT 12.4ms、14 GB 权重搬运时间估算

这节短，只是立词汇表，不展开——后面每节都会回来用这套语言。

---

### Act 1 — 连续批处理：买的是排队，不是算力（~400 词）

- 静态 batch vs continuous batching 的调度差异
- **反直觉点**：端到端吞吐几乎没变（717 vs 705 tok/s），但 P50 延迟差 2.6 倍
- **数据**：Wk 5 全部结果 + TTFT/TPOT 定义
- **图**：嵌入/链接 `viz/phase1.html#act2`

---

### Act 2 — PagedAttention：把显存当虚拟内存用（~400 词）

- 连续预分配 vs 分页分配，操作系统类比
- 碎片率 60-80% → <4%
- **数据**：Wk 4 batch sweep（80→2,408 tok/s 近线性），Wk 9 的 KV 容量数字
      （1.05M → 5.14M tokens，TP 切权重顺带腾出的容量）
- **图**：`viz/phase1.html#act3`

---

### Act 3 — 前缀缓存：算过的不再算（~350 词）

- block hash 机制，cache hit/miss 的 TTFT 差异
- **数据**：Wk 7，TTFT 829→156ms（5.3x），吞吐 835→1,412 tok/s
- 局限：必须逐 token 相同，跨实例需要前缀感知路由

---

### Act 4 — 投机解码：把 decode 变回 prefill（~450 词）

- causal mask 为什么保证验证结果精确等价，不是近似
- **核心反例**：acceptance rate 才是唯一变量,不是"投机解码"这个技巧本身
- **数据**：Wk 8，copy 任务 2.78x/2.38x vs novel 任务 1.13x/**0.85x 净亏**
- **修正一个初始结论**："高 batch 下投机解码失效"是错的，
  正确版本是"batch 放大 acceptance rate 的好坏，不决定符号"

---

### Act 5 — 模型装不下一张卡怎么办（~600 词，合并 Wk9+10）

**5a. 张量并行**
- Megatron 式切分（列并行+行并行省掉一次通信）
- **反直觉点**：每卡吞吐反而下降（2,950→1,571 tok/s/GPU），TP 买的是延迟和容量,不是性价比
- **数据**：Wk 9，TP=2 1.55x/TP=4 2.16x，效率 75%/54% 且跨 batch 恒定
- KV 容量随 TP 涨得比吞吐快（1.05M→5.14M tokens）—— 被低估的收益

**5b. MoE 与专家并行**
- 总参数决定显存，激活参数决定速度——一句话解释 MoE
- **核心反例**：MoE 优势随 batch 蒸发（67%→6%），因为 token 被打散到 60 个专家，
  `M=32` 碎成 60 个 `M≈2`——**batching 是 MoE 唯一吃不到的红利**
- EP vs TP 的预测翻车：我预测 EP 该在大 batch 下赢，实测两者在这个规模下都是净亏
  （TP 0.84x，+EP 0.56x）——诚实记录预测被证伪
- **数据**：Wk 10 全部结果

---

### Act 6 — Roofline：吞吐撞上天花板的地方（~500 词）

- 算术强度=batch 这个恒等式怎么来的（M/K/N 推导，简版）
- A100 脊点 M≈153，实测转折区间 128-256，吻合
- MFU 封顶 ~52%，不是理论 100%——这个数字本身就是一个基准
- CUDA graph：预测 1.2x，实测 2.69x，漏算了 PyTorch 分发开销的教训
- **图**：`viz/week11_roofline.html` 的第一、二张图直接嵌入/截图

---

### Act 7 — 量化：把字节数变小,前提是有人接得住（~600 词）

- 三层模型：位宽 / 算法（AWQ、GPTQ）/ kernel（Marlin）—— 这是这节的核心框架
- **Wk 11 的谜团**：INT8 全程 ~1.0x，几乎零加速，尽管"字节数减半"理论成立
- **Wk 12 揭晓**：AWQ/GPTQ-INT4 走 Marlin，batch=1 达 2.2x——比 FP8 还快
- **精确解释为什么**：理论天花板 2.73x vs 2.00x，两者兑现率都在 78-81%——
  不是 kernel 更强，是起跑线更靠前
- FP8 在脊点精确翻脸（1.51x→0.84x），INT4 在算力区反而亏得更少——
  一个预测对了、一个预测错了，都诚实写出来
- **数据**：Wk 11 + Wk 12 完整 tradeoff 表
- **图**：`viz/week11_roofline.html` 的量化柱状图

---

### 收尾 — 一张表说完 12 周（~300 词）

直接搬 `viz/phase1.html` 结尾那张总表（7 行：连续批处理/PagedAttention/前缀缓存/
投机解码/TP/MoE/量化），每行加一句"这项技术动的是算术强度还是字节数"。

**结尾钩子**：Phase 2 是分布式训练——同一套 roofline 语言在训练侧会怎么变形
（反向传播的通信模式和推理完全不同），埋个引子但不展开。

**可复现性**：链接 repo，`modal run experiments/wkXX_*/*.py` 一行跑起来。

---

## 关于素材的一个决定点

Wk 1-3（环境搭建、HF baseline、vLLM 首测）**不打算单独成节**——
这些是热身，结论已经被后面的实验覆盖或吸收（比如 Wk 2 的"7B 比 0.5B 快"
可以在 Act 0 里一句话带过，作为算术强度的引子）。如果你觉得应该保留，
说一声，我可以加一个简短的"起点"小节。

---

## 你需要确认的几件事

1. 标题选哪个（或者你有更好的）
2. Act 5（TP+MoE）和 Act 7（量化）篇幅最大，是否要拆更细，还是保持现在的密度
3. 图表策略：直接把 `viz/*.html` 里的图截图嵌进 markdown，还是在博客里放
   iframe/链接指向可交互版本（取决于 chenxin-ma.github.io 的技术栈支持什么）
4. Wk 1-3 要不要单独留痕
