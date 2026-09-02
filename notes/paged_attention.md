# PagedAttention 精读笔记

> 论文：Efficient Memory Management for Large Language Model Serving with PagedAttention  
> arxiv: https://arxiv.org/abs/2309.06180  
> 关键词：KV Cache、显存碎片、Block Table、Continuous Batching

---

## 一、问题背景：KV Cache 为什么浪费显存？

LLM 生成每个 token 时，需要用到之前所有 token 的 Key 和 Value 向量（即 KV Cache）。
这些向量必须放在 GPU 显存里，生成过程中持续累积。

**传统做法的问题**：

```
请求 A：最多可能生成 512 tokens → 预留 512 个 token 的 KV 空间
         实际只生成了 200 tokens → 剩余 312 个 token 的空间白白占用

请求 B：最多可能生成 512 tokens → 再预留 512 个 token 的 KV 空间
         实际只生成了 50 tokens  → 剩余 462 个 token 的空间白白占用
```

两种碎片（论文 §3.2）：

| 类型 | 含义 | 类比 |
|------|------|------|
| **Internal fragmentation** | 预留了但没用完的空间（上面的例子） | 订了一桌10人席，来了3个人 |
| **External fragmentation** | 空闲空间太碎，装不下新请求 | 停车场有空位但都不连续，大车停不进去 |

论文测量结果：传统方式下 **60-80% 的 KV Cache 显存被浪费**。

---

## 二、PagedAttention：把操作系统的虚拟内存搬进来

核心思路：借鉴 OS 的**分页内存管理**，把 KV Cache 切成固定大小的 block，按需分配。

### 类比对照表

| 操作系统虚拟内存 | PagedAttention |
|----------------|---------------|
| 进程的虚拟地址空间 | 一条请求的 KV Cache 逻辑空间 |
| 物理内存页（Page） | KV Cache Block（固定大小，如 16 tokens） |
| 页表（Page Table） | **Block Table**（逻辑 block → 物理 block 的映射） |
| 物理页不需要连续 | 物理 KV block 不需要连续 |
| 按需分配物理页 | 生成时才分配 KV block |

### Block Table 是什么？

每条请求维护一个 Block Table，记录"我的第 N 个逻辑 block 实际存在哪个物理 block"。

```
请求 A 的 Block Table：
  逻辑 block 0 → 物理 block #7
  逻辑 block 1 → 物理 block #23   ← 不需要连续！
  逻辑 block 2 → 物理 block #2

请求 B 的 Block Table：
  逻辑 block 0 → 物理 block #15
  逻辑 block 1 → 物理 block #3
```

Attention 计算时，GPU kernel 通过 block table 找到正确的物理位置读取 KV 值。

### 为什么物理 block 不需要连续？

传统 attention 计算假设 KV cache 在内存中连续排列，直接用指针偏移访问。  
PagedAttention 修改了 attention kernel，改为先查 block table，再访问对应物理位置。  
代价：attention kernel 稍复杂；收益：显存利用率从 ~30% 提升到 **>96%**。

---

## 三、调度与抢占（§4.3）

当显存不够时，vLLM 有两种处理方式：

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| **Swap** | 把 KV block 移到 CPU 内存 | 有足够 CPU 内存时 |
| **Recompute** | 直接丢弃，下次从头重算 KV | CPU 内存也紧张时 |

实际生产中通常用 Swap，Recompute 是最后手段。

---

## 四、Continuous Batching 怎么和 PagedAttention 配合？

```
时间线：
  t=0: 请求A(200tok), 请求B(50tok), 请求C(400tok) 同时进入
  
  传统 batch：等所有请求都结束才释放显存
    → B 在 t=50 结束，但显存一直占到 t=400（C 结束）
  
  vLLM continuous batching + PagedAttention：
    → B 在 t=50 结束，立即释放 B 的 KV blocks
    → t=50 时新请求 D 进入，直接复用 B 腾出的 blocks
    → GPU 始终满载
```

PagedAttention 是 continuous batching 的**基础设施**：没有按需分配的 block，
continuous batching 就没办法在请求结束时精准释放显存。

---

## 五、关键数字（论文结果）

| 指标 | 数值 |
|------|------|
| KV cache 碎片率（传统） | 60–80% |
| KV cache 碎片率（vLLM） | < 4% |
| vs FasterTransformer throughput 提升 | 2.2× |
| vs Orca throughput 提升 | 1.8× |
| block size 推荐值 | 16 tokens（论文 §6.2 ablation） |

---

## 六、三句话总结

1. **问题**：传统 LLM serving 按最大长度预分配 KV Cache，60–80% 显存成为碎片，严重限制并发。
2. **方案**：PagedAttention 用 block table 实现 KV Cache 的非连续分页分配，碎片率降至 <4%，物理 block 按需分配、即时释放。
3. **效果**：配合 continuous batching，同等显存下并发请求数大幅提升，throughput 比同期系统高 1.8–2.2×。

---

## 七、读完论文后可以追问自己的问题

- block size 设为 16 tokens 是怎么选出来的？（见 §6.2 ablation）
- prefix caching（Week 7）是怎么复用 block 的？（block table 的 copy-on-write）
- 如果一条请求的 KV cache 被 swap 到 CPU，下次调度回来需要多久？
