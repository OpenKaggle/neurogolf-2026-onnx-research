# biohack44 6115 / 6113-bundle 调研

来源：[`biohack44/neurogolf-2026-fp16-surgery-prune-blend-6115`](public_kernels/biohack44_6115/notebook.py)

## 1. 6115 = 6113 + 已知技巧

- Anchor 数据集：`biohack44/neurogolf-6113-bundle`（biohack 自己的 6113 分预制包）
- Kernel 只做两件事：
  - **fp16 graph surgery v2**（haoranran 同款，我们已实现 [`tools/fp16_surgery.py`](tools/fp16_surgery.py)）
  - **prune unused initializers**（去掉无引用 init / value_info / dead nodes）
- 没有新算法、没有新 hand-build；提升全在 6113 anchor 本身

## 2. Prune trick 在我们 6103 包上的独立增益

跑 [`reports/prune_probe.json`](reports/prune_probe.json)：

- 400 个 task 扫一遍，只有 **12 个**能瘦身且本地通过
- 累计 local-pred uplift = **+0.065**
- 结论：**忽略不计**，我们的 6103 已经够干净（fp16 surgery 路径基本已经触发到 prune）

## 3. 6113-bundle 每 task 对比（vs 我们 6103）

完整结果：[`reports/biohack44_6113_score_all.csv`](reports/biohack44_6113_score_all.csv)

整包 local 总分：
- ours 6103: **6103.94** (396/400 local pass)
- biohack 6113: **6037.70** (388/400 local pass) — 官方 6113 但 6 个高分 task 在我们 visible 例上直接 0 分

### 全盘 swap 的真实成本

| | 数 | 累计 Δlocal |
|---|---|---|
| biohack 严格更好（pass + Δ≥+0.05） | 11 | **+16.19** |
| 我们严格更好（pass + bio fail OR Δ≤-0.05） | 25 | **-84.27** |
| 净 swap-all | — | **-68** |

**结论**：不能整包换。

### 值得单独看的 11 个 biohack-赢 task

| tid | ours score | bio score | Δ | ours cost | bio cost | 备注 |
|-----|-----------|-----------|---|-----------|----------|------|
| **151** | 15.16 | 18.19 | **+3.03** | 18,805 | 910 | **22 节点 → 1 Conv**！纯结构压缩 |
| **028** | 13.30 | 15.90 | **+2.60** | 120,690 | 8,994 | Slice/ReduceMax 路径，比我们 Mul/ReduceSum 省内存 |
| 200 | 13.68 | 15.74 | +2.06 | 82,365 | 10,511 | 我们 SEM，bio 用 Less/Mul/Div 不同算法 |
| 258 | 18.19 | 19.93 | +1.74 | 910 | 160 | 同样 1 Conv，bio init 更小 |
| 111 | 16.58 | 18.32 | +1.74 | 4,532 | 797 | Concat-merge 拓扑 |
| 155 | 16.55 | 18.02 | +1.47 | 4,675 | 1,076 | 几乎同款，少一个 ReduceSum |
| 150 | 16.55 | 18.02 | +1.47 | 4,674 | 1,076 | 同 155 |
| 014 | 12.63 | 13.76 | +1.13 | 235,688 | 76,080 | 166 节点（50 Constant）→ 46 节点 |
| 204 | 12.36 | 12.82 | +0.46 | 308,848 | 195,823 | 我们 Python 已 268/268，bio ONNX 仍更小 |
| 084 | 14.69 | 15.12 | +0.44 | 30,176 | 19,513 | 我们 SEM，bio 用 Where/Conv |
| 322 | 17.97 | 18.05 | +0.07 | 1,128 | 1,047 | 边际 |

合计本地上限 **+16.19**。

### 隐藏集风险（不要忽视）

[`reports/research.md`](reports/research.md) 已记录两条警告：
1. Octavi 报告："cheap public swaps that locally predicted gains but scored much worse online"
2. 我们自己经历过：`task303` 的 fp16 在 hidden 上崩盘

所以即使本地 +16，hidden 极有可能折损。**单 task 概率比整批安全得多**。

## 4. 主线（我们自己改）的真正收获

biohack 的结构是**模板灵感**，不一定要 swap：

- **task151 = 1 个 Conv**：这种"全图归一化为单 Conv"的金币级压缩技巧，可以学着用到 task258/155/150 之外的其他几何 task
- **task014 / task028**：用 Slice 切单 channel 替代 broadcast，再 ReduceMax 收尾，比 one-hot 全通道操作省一个数量级内存
- **task200/084**：bio 的 Less + Where 路径值得看作我们 hand-build 的"另一实现"，可能解锁更低 cost 的 ONNX 重写

这些都不算 public-package swapping，是"看了好图自己改写"。

## 5. 推荐下一步（按风险/收益分层）

### A 路线（零风险，纯启发）
读这两个 ONNX 的 graph 结构，自己重写 hand-build：
- `submissions/biohack44_6113_onnx/task151.onnx`（1-Conv pattern）
- `submissions/biohack44_6113_onnx/task028.onnx`（Slice + ReduceMax）

如果我们能复现同款 cost、本地全过，就替换并提交。

### B 路线（小赌注 oracle 探测）
**最多挑 2–3 个 biohack 网络作为单 task 替换**，每个独立打包成候选 zip 提交，让 Kaggle 当 hidden-set oracle：
- 首选：`task151`（+3.03，结构最简单也最容易 hidden 上挂）
- 次选：`task028`（+2.60）

注意：每提交一次消耗一次额度，且需保留命名副本，不覆盖 best。

### C 路线（不做）
- 整包 swap：净 -68 local，否决
- 全部 11 个 winner 一起换：风险叠加大，hidden 折损不可控
