# SA-NNO-DRL 论文复现可行性评估

评估对象：`doc/Fan 等 - 2025 - Multi-UAV reconnaissance mission planning via deep reinforcement learning with simulated annealing.pdf`
（Swarm and Evolutionary Computation 93 (2025) 101858）

评估日期：2026-10-20
评估方式：论文全文精读 + 源码逐项比对 + 在 CPU 上实际跑通主链路（训练 → checkpoint → SA 推理 → 通用 eval）。

---

## 0. 结论

**一句话：仓库能支撑“方法本身跑通”的复现，不能支撑“论文结果表”的复现。**

| 维度 | 结论 | 依据 |
| --- | --- | --- |
| 端到端主链路（数据→训练→保存→加载→SA 多 UAV 规划） | ✅ 可跑通（已实测） | 见 §4 |
| 论文主表（表 2/3） | ❌ 只能产出 SA-NNO-DRL 一列，6 个对照均无可用实现 | 见 §3.1 G1 |
| 泛化实验（表 4/表 5） | ⚠️ 数据与推理具备，但对照列缺失；表 5 的 U3/U6/U7 受实现偏差影响 | 见 §3.2 G4 |
| 消融（图 8/9） | ❌ 三个变体都没有开关 | 见 §3.1 G6 |
| 单 UAV 路由对比（表 6） | ❌ 11 个方法中 9 个缺失（NNO-DRL 两行可跑） | 见 §3.1 G1 |
| 附录（A 敏感性 / B Wilcoxon / C TOP benchmark） | ❌ 硬编码、无脚本、无数据 | 见 §3.1 |
| 算法保真度 | ⚠️ 至少 3 处实质偏差（dummy hub 未 mask、增强方式、SA shift 算子） | 见 §3.2 |
| 结果可复现性（种子、结果落盘、聚合） | ❌ SA 入口无种子、不写结果、无 gap/表格脚本 | 见 §3.1 G2/G5 |

粗略分级（不是精度承诺）：

- **方法实现层面：约 70–80%**。网络结构、目标分配、SA 骨架、训练目标的主体都在，且能跑。
- **论文实验矩阵层面：约 20–30%**。缺全部对照基线、缺消融开关、缺 3 类附录实验、缺指标聚合，训练配方也未在仓库中固定。
- **论文数值层面：不可声称可复现**。即使补全上述内容，现有 3 处算法偏差也足以让表 2/4/5 的小数位（gap 0.00%–0.27% 量级）偏离。

> 注意：`doc/main-flow-repair-plan.md` 第 6 节已明确“不把完整训练收敛或论文指标复现作为本轮验收条件”。本次评估与该文件的自我定位一致，但把它推到了“能不能做论文复现”这个问题上，结论是不能，需补料。

---

## 1. 论文对“复现”的硬性要求（从原文提取）

| 要素 | 论文要求 | 出处 |
| --- | --- | --- |
| 问题 | MURMPP ≈ 多 UAV TOP：单 hub、每目标至多访问一次、航程 $L_{max}=2$、目标坐标 $[0,1]^2$ 均匀、hub 在中心 | §5.1 |
| 收益 | const=1；unif ~ U[0,1] | §5.1 |
| 训练集 | 每 UAV 分到的扇区作为训练样本；规模 20/20/20/20/30/40（T40U2…T200U5）；$\lvert D\rvert=1{,}280{,}000$，$\lvert D_v\rvert=10{,}000$ | §5.1 / Table 1 |
| 模型 | AM 编码器（6 层、8 头、$d_h$=128）+ 投影 MLP（2 FC + IN + GELU）+ k-NN 多起点解码器 | §4.1.2–4.1.4 |
| 增强 | 绕图中心随机旋转，$M=4$，$s_1=s$ | §4.1.1 |
| 解码 | 训练采样、测试 greedy；起点为 hub 的 $k=\text{int}(size/4)$ 近邻 | §4.1.4 / §5.1 |
| 训练 | CL（$k_{CL}=0.1$）+ REINFORCE（候选均值 baseline）、Adam、lr $10^{-4}$、衰减 0.995、batch 64、10 epoch | Table 1 / Alg. 1 |
| 分配 | 按相对 hub 角度等分 $m$ 组；按角度中位线分左右两半 | §4.2 |
| SA | $T_s=80$、$T_e=60$、$L=4$、$\sigma=0.9$；Metropolis 接受；shift 算子偶数轮顺时针、奇数轮逆时针 | Alg. 2 / Table 1 / Fig. 3 |
| dummy | 组大小不齐时在序列尾部补 dummy hub，**且在 kNN 与解码中被 mask** | §5.1 |
| 测试集 | 每组 30 个实例，所有方法共享 | §5.2 |
| 对照 | Gurobi、OR-Tools、GAMMA(Greedy/1280)、AM(Greedy/1280) | §5.2 |
| 指标 | Obj.（30 实例均值）、Gap（对全体方法最优值）、Time | Table 2 |
| 其它 | 表 4 大规模 300–1000、表 5 变 UAV 数 2–8、图 7 变 hub 位置、图 8/9 消融、表 6 单 UAV 路由、附录 A/B/C | §5.3–5.5 / App. |

---

## 2. 仓库现状速览

- 代码是作者公开仓库（`github.com/mingfan321/SA-NNO-DRL`）的 fork：作者原始上传 7 个 commit，之后是本仓库的 README/LICENSE 与 6 个“主流程修复”commit。
- 已有：OP 数据契约与归一化、POMO 多候选聚合、版本化 checkpoint、可配置入口、7 个单元测试、30 实例的 const/unif 测试集（40–1000）。
- 没有：预训练权重、任何对照基线、任何实验驱动/指标聚合脚本、消融开关、种子控制、结果落盘。

已纳入版本库的测试数据（`op_data/{const,unif}/*.pkl`）每个文件 30 条，字段序为 `(loc, prize, depot, max_length)`，`depot=(1,2)=[0.5,0.5]`、`max_length=2.0`，首条坐标与 `np.random.seed(12345)` 的 `uniform((30,40,2))` 完全一致。这与论文的 30 实例测试集设定吻合，是复现的有利条件。

---

## 3. 缺口清单

### 3.1 阻断级（不补就无法产出论文结果）

#### G1. 对照基线整体缺失（最严重）

论文表 2/3/4/5/6 的价值在于**横向比较**，而仓库里：

- `problems/op/op_baseline.py:177`、`:213`、`:244` 分别 `from problems.op.op_gurobi / op_ortools / tsiligirides import ...`，但**这三个模块在仓库里和上游公开仓库里都不存在**（只剩 Python 3.7/3.8 的 `.pyc`）。
- GAMMA、AM、MDAM、POMO 没有可切换的实现：仓库里的 attention 模型本身就是 NNO-DRL（k-NN 多起点 + 投影头），无法切回 AM 的单起点、POMO 的“全节点起点”或 MDAM 的多解码器语义（论文表 2 的 6 个对照列、表 6 的 9 个对照行）。
- `pyproject.toml` 未声明 `gurobipy`/`ortools`，也没有 Compass 的构建产物（`problems/op/install_compass.sh` 需要联网 clone + 编译）。
- 顺带：`op_baseline.py` 按 `(depot, loc, prize, max_length)` 解包数据（`run_all_in_pool` 把 tuple 直接展开），而仓库数据是 `(loc, prize, depot, max_length)`——即使补回缺失模块，也要先修数据序。

**后果**：Gap 列无法计算（Gap 定义依赖“全体方法最优值”），表 3 的“最优解计数”无法计算，表 6 无法成表。

#### G2. 没有实验驱动与指标聚合

- 没有“12 组 × 2 收益类型”的批量运行脚本；`SA-NNO-DRL.py` 一次只吃一个 `.pkl`，且只打印均值和均值耗时。
- 逐实例结果没有落盘：`SA-NNO-DRL.py:370-382` 建了 `results/drl_small_data` 目录却把 `save_dataset` 注释掉了，无法算 std、Gap、Wilcoxon。
- 没有论文级绘图脚本（图 4/5/6/7/8/9）。
- `eval.py` 是通用单 UAV OP 评估器，不含多 UAV/SA；且 `eval.py:209` 在未给 `--width` 时直接 `TypeError`（实测），必须先 `--width 0` 才能跑 greedy。

#### G3. 没有预训练权重，训练配方也未固定

- 仓库不含任何 `.pt`；README 只给 smoke 命令，默认 `--graph_size 20 --epoch_size 640000`，与论文的 40 节点 / 1,280,000 不一致。
- 论文没有明确“训练几个模型”（每个问题规模一个？每种收益类型一个？还是全部规模共用一个），§5.1 说“超参共享”，§5.3.1 又说“只用 40 节点训练”，两种读法对应 2 个或 12 个模型，差一个数量级的算力。仓库没有消解这个歧义（无配置矩阵、无训练脚本）。
- 我实测：模型 1.32 M 参数；40 节点扇区、batch 8×4 增强时 CPU 前向 0.049 s、反向 0.069 s。按此粗外推，论文的 20 万次更新在单张 RTX 5080 上是“数小时/模型”量级——可承受，但没有实测，且要先决定训练矩阵。

#### G4. dummy hub 未 mask（实现与论文明确描述不符）

论文 §5.1 原文：补在序列尾部的 dummy hub“在 kNN 选择与解码过程中被 mask，不会被选中”。
仓库：`problems/op/state_op.py:90-97` 的 `get_k_nearest_neighbor()` 只按距离排序并把超预算节点置 0，**没有任何 dummy mask**；`get_mask()` 也没有区分 dummy。dummy 是 hub 的副本，距离为 0，因此必然排在 kNN 第一位。

我用 40 目标 / 3 UAV（分组 14/13/13）实测：

```
UAV1 real nodes=13 expected_kNN_start=[12, 6, 5]  actual start ids=[14, 12, 6]
UAV2 real nodes=13 expected_kNN_start=[8, 2, 7]   actual start ids=[14, 8, 2]
```

即：**被补齐的 UAV 的第 1 条轨迹实际从 hub 出发，第 k 近邻起点被挤掉**；解码途中也可能选中 dummy（浪费一步，且模型内部长度把“绕回 hub”算进去，导致少访问目标）。最后 `line` 过滤会把它剔除，所以不破坏合法性和目标值计算，但损失解质量。

受影响的是所有 $n \bmod m \neq 0$ 的组：表 5 的 T200U3、T200U6、T200U7（表 2 的 12 组恰好都能整除，不受影响；表 4 的 300–1000 / U5 也能整除）。

#### G5. SA 推理不可复现（无种子）

- `SA-NNO-DRL.py` 没有 `--seed`，也没有 `torch.manual_seed`/`np.random.seed`（只有被注释掉的一行 `:344`）。
- 但是推理时 `train.augment()` 每次调用都会 `torch.rand(N-1)` 生成新的随机旋转（`train.py:164`），SA 的 `np.random.random()` 也是无种子（`:287`）。
- 结果：同一模型、同一数据、同一 UAV 数，两次运行的目标值不同；论文的 0.00% gap 不可能靠这种入口复现或核对。

#### G6. 消融变体没有开关

图 8/9 需要 4 个变体：

| 变体 | 需要的改动 | 仓库现状 |
| --- | --- | --- |
| SA-NNO-DRL w/o CL | 关掉 CL 项重训 | CL 系数硬编码在 `train.py:251` 的 `- 0.1 * similarity.mean()`，无 CLI |
| SA-DRL | 换成 AM 单起点解码器 | 解码器恒为 k-NN 多起点（`attention_model.py:242,255`），无切换 |
| NNO-DRL | 只跑初始分配+路由，不做 SA | SA 循环硬编码（`SA-NNO-DRL.py:174-291`），无开关 |
| NNO-DRL(M=1) | M=1 | 训练有 `--N_aug`，但约束 `>=2`（`options.py:98`）；SA 里 `N_aug=4` 写死（`:62`） |

#### G7. 训练超参/SA 超参不能全部通过命令行给定

- SA 的 $T_s/T_e/L/\sigma$ 硬编码在 `SA-NNO-DRL.py:174-177`，附录 A 的敏感性分析（Ts∈{70,80,90}、Te∈{55,60,65}、L∈{2,4,6}、σ∈{0.85,0.9,0.95}）需要改代码跑 9+ 组。
- `k` 由 `pomo_size = int((n+1)/4)` 隐式决定（`attention_model.py:242`，注意用的是含 depot 的节点数），无法复现图 4(b) 对 $k$ 的扫描，也不是严格的 `int(size/4)`（$n \equiv 3 \pmod 4$ 时会多一个候选）。

---

### 3.2 保真度级（能跑，但结果会和论文有系统偏差）

| # | 论文 | 仓库 | 影响 |
| --- | --- | --- | --- |
| F1 | 绕**图中心**随机旋转，仅旋转 | `train.py:138-175`：中心写死 `(0.5,0.5)`；`idx>=0.5` 时交换 x/y（镜像）；每次 batch 内所有实例共用同一随机角 | 训练分布略有不同；CL 的正样本几何性质改变（仍是等距变换，最优解不变，但表征学习难度不同） |
| F2 | shift 算子：偶数轮把**右半**交给右邻居，左半“remain unassigned”；奇数轮反之 | `SA-NNO-DRL.py:204-273`：偶数轮每架 UAV 保留自己的右半 + 接收下一架的左半；奇数轮保留自己的左半 + 接收上一架的右半。**两半都留在系统里**，且方向与论文相反 | SA 的邻域结构与论文不同；迭代轨迹、接受率、最终解都会偏 |
| F3 | 投影 MLP 用 **InstanceNorm** | `attention_model.py:117` 改为 `LayerNorm(affine=False)` | 我实测两者在 `[B,N,E]` 上数值等价（最大差 4.8e-7）。**但作者原始代码的 `InstanceNorm1d(128)` 作用在 `[B,N,E]` 时也是沿特征轴归一化**，所以是论文文字与作者代码本就不一致；当前实现“等价于作者代码、不等价于论文文字”，需在复现报告里显式声明 |
| F4 | 式(30)：$L_{CL}=-\frac{k_{CL}}{M-1}\sum_i \cos(\cdot)$ | `train.py:238-251` 对 batch 和节点都取 mean | 论文未写明节点维是否求和；若按节点求和，则 CL 相对权重会差约 $(n+1)$ 倍。当前实现给出的是“作者发布代码的语义”，需用训练曲线反推确认，不能直接当作论文语义 |
| F5 | 式(24–28) 概率由 masked softmax 给出，对数概率自然为 $-\infty$ | `train.py:71-75` 把概率 clamp 到 `tiny` 再取 log，零概率变成 −87.6 | 梯度不再是“零概率不被强化”，而是给出一个有限大梯度；可能引入伪梯度 |
| F6 | unif 收益 ~ U[0,1] | `problem_op.py:102`：`(1+randint(0,100))/100` → [0.01,1.00] | 均值偏高 0.01，可忽略；但“U[0,1]”的表述应修正 |
| F7 | 测试集由论文流程生成 | `generate_data.py:43-53` 用 `MAX_LENGTHS` 字典（只有 20/50/100）、depot 随机（`:29`）；实测 `--graph_sizes 40` 直接 `KeyError: 40` | **仓库自带的 30 实例测试集无法用仓库脚本重建**（数据本身在，能复现；但数据生成链断裂，无法扩展到新 seed/新 hub 位置） |
| F8 | 每个基线都有明确配置 | `op_baseline.py` 的 `run_all_in_pool` 用 `(depot, loc, prize, max_length)` 展开，与仓库数据序不符 | 即使补回 gurobi/ortools 模块，也会把 loc 当 depot |
| F9 | 表 2 时间 <0.5 s（GTX3090） | 本机 CPU 上 40 目标 0.32 s、200 目标 1.9 s（未训练模型，路由短，不代表真实值） | 时间列只能在新硬件上重测，不能与论文直接比 |

---

### 3.3 论文实验 × 仓库支持度

| 论文产物 | 仓库能做什么 | 缺口 |
| --- | --- | --- |
| 图 4（lr/k 曲线） | 可训练、有 TensorBoard/cost.csv | 无网格运行脚本；k 不可配 |
| 图 5/6（训练曲线） | 可训练、可记录 | 无绘图脚本；训练矩阵未定 |
| 表 2（12 组 × 7 方法） | 只跑得出 SA-NNO-DRL 的 Obj./Time | 对照基线缺失 → 无 Gap、无表 3 |
| 表 4（300–1000 泛化） | 数据齐、SA 可跑（U5） | 基线缺失；模型须按 40 节点扇区训练 |
| 表 5（变 UAV 数 2–8） | 数据齐、`--uav-num` 可跑 | 基线缺失；U3/U6/U7 受 G4 影响 |
| 图 7（hub 位置） | 数据/脚本都没有；`change_node()`（`SA-NNO-DRL.py:31`）存在但从未被调用 | 需生成非中心 hub 的 30 实例 + 实验脚本 |
| 图 8/9（消融） | 无 | 见 G6 |
| 表 6（单 UAV 路由 20/50 C/U） | 只能跑 NNO-DRL(M=4) 一行；`eval.py --width 1280` 可做采样版 | 其余 9 个方法缺失；没有“按 UAV 拆分扇区”的评测集生成脚本 |
| 附录 A（SA 敏感性） | 需改源码 | 超参不可配（G7） |
| 附录 B（Wilcoxon） | `scipy` 已是依赖 | 无脚本、无基线数据 |
| 附录 C（TOP Set_33_234） | 无 | 无数据、无加载器、无 32 节点改造脚本 |

---

## 4. 我实际验证了什么

环境：仓库根目录，Python 3.11.14（uv 管理），`torch==2.14.1+cpu`，numpy 2.4.6。**所有验证都在 CPU、未训练的 1 epoch smoke 模型上完成**，因此目标值不代表方法质量，只证明链路可跑。

| 检查 | 命令（要点） | 结果 |
| --- | --- | --- |
| 依赖安装 | `uv sync --extra cpu` | 成功；但 `--locked` 在本机因 pip 镜像与 lock 中 pypi.org 不一致而失败（环境问题，非仓库缺陷） |
| 单元测试 | `python -m pytest tests -q` | **7 passed**（数据契约、POMO 候选选择、sample_many 聚合、零概率 log 保护） |
| 配置自检 | `run.py --check_config --device cpu` | 正常退出，不建目录不加载数据 |
| 训练 smoke | `--graph_size 20 --batch_size 2 --epoch_size 2 --n_epochs 1 --device cpu` | 4.4 s 完成，梯度范数 434.7（裁剪到 1.0），checkpoint + args.json 落盘 |
| SA 推理（T40U2C） | `SA-NNO-DRL.py --dataset op_data/const/op_const40… --uav-num 2` | 完成，逐实例目标值 + 均值 + 耗时输出，`validate_allocation` 通过 |
| SA 推理（T200U5C） | 同上 `--uav-num 5` | 完成（93.5，未训练模型），1.9 s/实例 |
| SA 推理（U3/U8） | 同上 `--uav-num 3 / 8` | 均完成，补齐组不崩溃（但目标受损，见 G4） |
| 通用 eval | `eval.py … --decode_strategy greedy` | **未给 `--width` 时 TypeError**；加 `--width 0` 后可跑（且只是单 UAV OP 评估） |
| 数据生成 | `generate_data.py --problem op --graph_sizes 40` | `KeyError: 40`，无法重建自带测试集 |
| dummy 起点 | 40 目标/3 UAV，打印每架 UAV 的 POMO 起点 | 补齐组首条轨迹起点 = dummy（14），预期第 3 近邻被挤掉 |
| IN/IN 等价性 | `InstanceNorm1d(128)([2,7,16])` vs `LayerNorm(16,affine=False)` | 最大绝对差 4.8e-7（F3 的结论） |

复现我这些结论的最小命令：

```bash
uv sync --extra cpu
uv run --extra cpu python -m pytest tests -q
uv run --extra cpu python run.py --no_tensorboard --no_progress_bar --device cpu \
  --graph_size 20 --batch_size 2 --epoch_size 2 --val_size 2 --eval_batch_size 2 \
  --n_epochs 1 --checkpoint_epochs 1 --output_dir outputs/smoke
uv run --extra cpu python SA-NNO-DRL.py --model outputs/smoke/op_20/<run>/epoch-0.pt \
  --dataset op_data/const/op_const40_len2_seed12345.pkl --num-samples 3 --uav-num 2 --device cpu
```

---

## 5. 若要把“不能”变成“能”：建议路线

### P0（不做则论文表格无法产出，约 3–5 人日）

1. **决定基线策略**。两个选项：
   a. 只复现“本方法列”，在论文复现报告中明确降级为“方法可运行性 + 相对自建基线的对比”；
   b. 用 OR-Tools（`ortools` 可直接 pip 安装）重建 OR-Tools 列，用公开 AM/POMO 实现重建 AM 列，Gurobi 需许可证。GAMMA 需另找作者代码。无论哪条，都要先修 `op_baseline.py` 的数据序。
2. **实验驱动 + 指标聚合**：一个 `scripts/run_matrix.py`（12 组 × 2 收益 × 方法），把逐实例 `(obj, time, tour)` 写成 JSON/CSV，再算 Obj./Gap/Time、表 3 计数、Wilcoxon。顺带把 `SA-NNO-DRL.py` 的 `save_dataset` 恢复。
3. **SA 入口加 `--seed` 与超参 CLI**，并在 `cal_fitness` 里用局部 RNG（或 `torch.Generator`），保证同 seed 同结果。
4. **修 dummy mask**：在 `get_k_nearest_neighbor()` 和 `get_mask()` 里按“末位补齐节点”索引 mask；补一个“补齐组首起点不是 dummy”的回归测试。
5. **落一份预训练 checkpoint + args.json + 训练命令**（至少 const/unif 两个 40 节点模型），否则任何人复现都要先烧几十 GPU 小时。

### P1（保真度，约 3–5 人日 + GPU 训练）

6. 增强改为“只旋转、按实例采样角度、绕图中心（或固定 hub）”，并与论文 §4.1.1 对齐。
7. shift 算子按 Alg. 2 / Fig. 3 重写，或至少在文档中声明“实现为等价变体”并做 A/B 对照。
8. CL 归一化（F4）与零概率 log（F5）二选一：要么改回论文语义并重训，要么记录“这是作者发布实现的语义”。
9. 用修正后的配方重训，确认训练曲线（图 5/6 的形态）和 T40U2C 目标值落在论文附近；若偏差 >1%，逐项回退定位。

### P2（补齐实验矩阵）

10. 非中心 hub 的测试集生成 + 图 7。
11. 表 6 的“按 UAV 拆分扇区”测试集 + 采样宽度 1280 的评测（`eval.py --decode_strategy sample --width 1280`）。
12. TOP benchmark `Set_33_234` 下载/裁剪/加载。
13. 消融开关（`--cl_coef 0`、`--decoder {knn,am}`、`--no_sa`、`--N_aug 1`）与四条曲线。

---

## 6. 风险与判断

- **最大风险不是代码质量，而是“不可获得的对照实现”**。表 2 的 4 类基线（Gurobi、OR-Tools、GAMMA、AM）里只有 OR-Tools 容易重建；表 6 还额外需要 Tsili、GA、Compass、MDAM、POMO。GAMMA/AM/MDAM 需要外部代码，Gurobi 需要许可证。这意味着“复现论文表 2”在工程上不是一个仓库内可完成的任务。
- **其次是训练配方的不确定性**：2 个还是 12 个模型，论文没写死；这直接决定算力预算和“用哪个模型出哪一行”。
- **再次是 3 处算法偏差**（G4/F1/F2）。它们不会让程序报错，但会让数值系统性偏移；在 0.00%–0.27% gap 的精度要求下，必须先修再谈“复现”。
- 有利条件：测试数据在库、SA 骨架与论文一一对应、checkpoint/入口已工程化、7 个回归测试保证了数据契约和候选选择不出错。**主链路已经不是一个“跑不起来”的仓库，而是一个“跑得起来但只跑了一半实验”的仓库。**

---

## 7. 附：论文超参 vs 仓库默认值

| 项 | 论文 | 仓库默认 | 位置 |
| --- | --- | --- | --- |
| 编码器层/头/维度 | 6 / 8 / 128 | 6 / 8 / 128 | `options.py:29-32` |
| tanh clip | 10 | 10 | `options.py:33` |
| batch | 64 | 64 | `options.py:21` |
| epoch 数 | 10 | 10 | `options.py:43` |
| epoch 样本数 | 1,280,000 | 640,000 | `options.py:22` |
| 验证集 | 10,000 | 10,000 | `options.py:23` |
| lr / 衰减 | 1e-4 / 0.995 | 1e-4 / 0.995 | `options.py:39-41` |
| 增强数 M | 4 | 4（训练）/ 4（SA，写死） | `options.py:20`、`SA-NNO-DRL.py:62` |
| k（起点数） | int(size/4) | int((n+1)/4) | `attention_model.py:242` |
| 训练节点数 | 20/20/20/20/30/40 | 20（须手改 40） | `options.py:16` |
| 航程 L | 2 | 2 | `options.py:18` |
| SA Ts/Te/L/σ | 80/60/4/0.9 | 80/60/4/0.9（写死） | `SA-NNO-DRL.py:174-177` |
| CL 系数 | 0.1 | 0.1（写死） | `train.py:251` |
| 测试实例/组 | 30 | 30（数据文件） | `op_data/*` |
| 种子 | 未给（数据 seed 12345） | 训练 1234，SA 无 | `options.py:44` |
