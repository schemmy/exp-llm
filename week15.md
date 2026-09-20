# Week 15 — 从单卡训练到两卡 DDP

状态：开始学习，尚未运行训练实验。对应 PLAN.md 的 Wk 15–16，不新增一周。

## 要回答的问题

同一个模型，为什么推理能放进显存，训练却可能放不下？第二张 GPU 能解决哪一部分问题？

## Task 1：读懂一个训练 step

- [ ] 画出 `batch → forward → loss → backward → optimizer.step → zero_grad`。
- [ ] 区分参数、激活、梯度、优化器状态：各自是什么、何时产生、何时释放。
- [ ] 解释为什么训练的 backward 需要 forward 的中间结果，而通常的推理不需要保存这些结果来求梯度。

先理解：loss 是当前预测的误差；backward 计算各参数对 loss 的梯度；optimizer 根据梯度更新参数。一个 step 会改变权重，普通推理不会。

## Task 2：单卡训练基线（脚本已准备，待运行）

脚本：[experiments/wk15_training_baseline/train_step.py](experiments/wk15_training_baseline/train_step.py)。随机初始化的四层 causal Transformer，词表 4096、hidden size 256、batch 8、序列长度 256。先用 FP32 + AdamW，关闭 TF32，不使用 AMP 或 compile，便于解释显存组成。

```bash
cd ~/projects/exp-llm
modal run --detach experiments/wk15_training_baseline/train_step.py::bench
```

直接运行远程 `bench`，整个实验和结果保存都在该函数内完成，不依赖本地 entrypoint 继续调度。新训练镜像只安装固定版本 `torch==2.6.0`（CUDA 12.4 wheel），第一次需要构建；不复用 vLLM 镜像。脚本本地通过语法和 Modal 定义加载检查；CUDA 执行尚未验证。

每次运行保存独立 JSON 至 Modal Volume `training-baseline-results`，文件名在日志末尾。可在 Modal 控制台下载，也可以运行 `modal volume get training-baseline-results <日志里的文件名，不含/results/> ./`。

先观察三个问题：

1. backward 后 `grad MiB` 是否接近 FP32 参数字节数？
2. 首次 optimizer step 后 `Adam MiB` 是否接近参数字节数的两倍，之后是否保持稳定？
3. `zero_grad(set_to_none=True)` 后梯度占用是否消失，而 Adam 状态仍在？

首次 step 与预热后 10 次的中位数分开报告。阶段之间同步会影响性能，因此这是诊断实验，不作为最大训练吞吐指标。JSON 保留 allocated/reserved/peak 和各阶段前后数据；forward 的分配增量不全是模型激活，也包含 loss 所需的中间 tensor。

测量接口参考：[PyTorch CUDA memory](https://docs.pytorch.org/docs/stable/cuda.html)、[AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)。

使用 Modal 单卡 A100 和一个小型 PyTorch 模型，先测完整训练过程，再扩大规模。无需一开始加载 7B。

- [ ] 固定随机种子、模型规模、batch size、序列长度、精度和优化器，记录环境版本。
- [ ] 分别测 forward/loss、backward、optimizer step 的耗时和显存。
- [ ] GPU 计时使用 CUDA events 或同步边界，避免只测 CPU 提交时间。
- [ ] 分开记录首次 step 和预热后的多次 step；Adam 的状态可能在首次更新时才分配。
- [ ] 同时记录 allocated、reserved 和阶段 peak；不要把缓存分配器保留的显存全部当作活跃 tensor。
- [ ] 用同一模型、同一输入做无梯度 forward 对照。这是训练开销对照，不等同于自回归生成 benchmark。
- [ ] 输出 JSON 结果，检查 loss/梯度有限且参数发生更新；合成数据仅验证训练与性能，不代表真实任务质量。

## Task 3：两卡 DDP（接续实验）

- [ ] 理解每张卡保存完整模型，各自处理不同数据；backward 时同步梯度。
- [ ] 用相同 global batch 比较单卡与两卡，再单独测固定 per-GPU batch 的吞吐扩展。
- [ ] 报告 global batch = per-GPU batch × GPU 数 × 梯度累积次数。
- [ ] 比较 step time、全局 samples/tokens per second、每卡显存；说明通信开销与计算重叠。
- [ ] 回答：为什么 DDP 通常不能解决单个完整模型训练状态放不进一张卡的问题？用它引出后续 FSDP/ZeRO。

## 完成标准

能解释一次参数更新的数据流；有可复现的单卡基线与两卡对照；分清模型复制和状态分片。没有实测前不填写性能结论。

## Phase 1 收尾仍保留

交互网页已经发布。Wk 13–14 的 README 复现说明、测量口径和总结仍需核对，不因进入训练学习就自动标为完成。
