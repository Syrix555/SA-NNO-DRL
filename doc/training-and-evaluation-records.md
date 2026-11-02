# 训练与复现实验记录

更新日期：2026-11-02。四组实验已完成。

## 1. 复现范围与代码来源

复现对象为 Fan 等人的 *Multi-UAV reconnaissance mission planning via deep reinforcement learning with simulated annealing*，发表于 Swarm and Evolutionary Computation 93 (2025)，文章号 101858，DOI：`10.1016/j.swevo.2025.101858`。本仓库基于作者的 `mingfan321/SA-NNO-DRL` 实现，补充运行环境适配、补齐节点屏蔽、路线可行性与收益一致性检查、推理结果导出和路线绘图。

本次范围为 SA-NNO-DRL 在以下四个场景下的自行训练、正式测试、训练曲线和路线可视化。每组独立训练一个模型，统一使用最后一个 epoch 的 checkpoint，不根据测试集收益选择模型。

| 场景 | 目标数量 | UAV 数量 | 收益分布 | 训练子问题规模 `graph_size` | 测试实例数 |
| --- | --- | --- | --- | --- | --- |
| T100U5C | 100 | 5 | const | 20 | 30 |
| T100U5U | 100 | 5 | unif | 20 | 30 |
| T200U5C | 200 | 5 | const | 40 | 30 |
| T200U5U | 200 | 5 | unif | 40 | 30 |

这里 `C` 表示每个目标收益恒为 1；`U` 表示收益取自 `{0.01, 0.02, …, 1.00}` 的离散均匀分布。本次没有开展其他算法对照、消融或多训练种子实验，不报告 Gap、最优解数量或显著性检验，不将四组结果表述为论文全部实验的数值复现。

## 2. 运行环境

正式推理的 `summary.json` 记录了以下环境：

| 项目 | 实际记录 |
| --- | --- |
| 系统 | Linux 7.0.0-38-generic，x86_64，glibc 2.43 |
| GPU / 设备 | NVIDIA GeForce RTX 5080 / `cuda:0` |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| PyTorch | 2.14.1+cu130 |
| CUDA / cuDNN | 13.0 / 92400（PyTorch 返回的版本整数） |
| 确定性算法开关 | `false` |

驱动版本 580.159.03 来自此前的环境验证记录，正式推理元数据没有逐次采集驱动版本。安装和 GPU 检查命令为：

```bash
uv sync --locked --extra cu130
uv run --extra cu130 python scripts/verify_environment.py --require-cuda
```

固定随机种子有助于同一环境下重跑；它不保证跨设备、跨库版本逐位一致。

## 3. 模型与训练参数

四个模型均为自行训练，`load_path` 和 `resume` 均为 `null`。正式测试固定使用 `epoch-9.pt`。最终权重与配套 `args.json` 原样存放在仓库的 `weights/<场景>/` 目录；下表同时保留 GPU 设备上的原始训练目录作为来源记录。

| 场景 | 原始训练目录（相对 `outputs/paper/`） | 仓库内最终权重 |
| --- | --- | --- |
| T100U5C | `op_20/t20_const_full_20261101T215416/` | `weights/T100U5C/epoch-9.pt` |
| T100U5U | `op_20/t20_unif_full_20261102T114336/` | `weights/T100U5U/epoch-9.pt` |
| T200U5C | `op_40/t40_const_full_20261028T220058/` | `weights/T200U5C/epoch-9.pt` |
| T200U5U | `op_40/t40_unif_full_20261029T030049/` | `weights/T200U5U/epoch-9.pt` |

训练证据包括各组 `args.json`、`cost.csv` 和 `logs/op_20/` 或 `logs/op_40/` 下同名运行的 TensorBoard events。正式推理的 `model_args.json` 与对应训练 `args.json` 内容一致；推理元数据记录了实际加载权重的路径和 SHA-256。

除子问题规模、收益分布和运行名外，四组共享以下训练设置，参数以实际 `args.json` 为准：

| 参数 | 设置 |
| --- | --- |
| 问题 / 模型 | `op` / `attention` |
| UAV 数量 / 最大往返航程 | 5 / 2.0 |
| `batch_size` / `epoch_size` | 64 / 1,280,000 |
| `n_epochs` / `epoch_start` | 10 / 0 |
| `N_aug` | 4 |
| `embedding_dim` / `hidden_dim` | 128 / 128 |
| `n_encode_layers` / 注意力头数 | 6 / 8（头数使用模型默认值） |
| `normalization` / `tanh_clipping` | `batch` / 10.0 |
| 优化器 / `lr_model` / `lr_decay` | Adam / 0.0001 / 0.995 |
| `max_grad_norm` | 1.0 |
| `baseline` / `bl_warmup_epochs` | `null` / 0 |
| `val_size` / `eval_batch_size` | 10,000 / 1,024 |
| `log_step` / `checkpoint_epochs` | 50 / 1 |
| 训练 seed | 1234 |

`graph_size` 指单 UAV 路由训练子问题的目标数。生成器先生成 `graph_size × uav_num` 个目标的母问题，再按角度划分为 5 个扇区。因此 T100 使用 20 节点子问题，T200 使用 40 节点子问题；正式推理时 SA 处理完整的 100 或 200 个目标。这四组使用各自训练的模型，不属于跨规模零样本测试。

每个 epoch 有 `1,280,000 / 64 = 20,000` 次参数更新，总计 200,000 次。配置中的 `baseline=null` 不表示完全没有基线项，训练损失内部仍使用候选收益的均值基线和一致性学习项。当前常规训练循环没有逐 epoch 调用 `validate()`，所以 `val_size` 等配置不构成独立验证曲线的证据。

以下是依据实际配置重建的 T100U5C 等效训练命令，用于重跑，不作为历史终端命令的原始记录：

```bash
uv run --extra cu130 python -u run.py \
  --device cuda:0 \
  --graph_size 20 --uav_num 5 --data_distribution const --max_length 2.0 \
  --batch_size 64 --epoch_size 1280000 \
  --val_size 10000 --eval_batch_size 1024 \
  --embedding_dim 128 --hidden_dim 128 --n_encode_layers 6 \
  --normalization batch --tanh_clipping 10 \
  --lr_model 0.0001 --lr_decay 0.995 --max_grad_norm 1.0 \
  --n_epochs 10 --N_aug 4 --log_step 50 \
  --checkpoint_epochs 1 --seed 1234 \
  --output_dir outputs/paper --run_name t20_const_full
```

其余三组分别替换 `graph_size / data_distribution / run_name` 为 `20 / unif / t20_unif_full`、`40 / const / t40_const_full`、`40 / unif / t40_unif_full`。程序会给运行名追加新的时间标识。

## 4. 数据集与正式测试流程

目标坐标位于单位正方形 `[0,1] × [0,1]`，基地固定在 `(0.5, 0.5)`。每架 UAV 从基地出发并返回基地，完整路线长度上限为 2.0。测试使用以下四份数据，每份 30 个实例：

```text
op_data/const/op_const100_len2_seed12345.pkl
op_data/unif/op_unif100_len2_seed12345.pkl
op_data/const/op_const200_len2_seed12345.pkl
op_data/unif/op_unif200_len2_seed12345.pkl
```

文件名中的数据 seed 为 12345，训练 seed 和正式推理 seed 均为 1234，三者用途不同。保存路线中的坐标、收益和基地已与原始测试集逐实例核对，数据文件哈希与正式推理元数据一致。没有将这些测试文件作为训练输入；训练样本由训练生成器生成。

推理先按角度分配目标，再用训练好的单 UAV 模型评估路线，并由 SA 调整分配。当前实现使用初温 80、终温 60、降温系数 0.9，每个温度 4 次邻域尝试，路由评估使用四重增强和贪心多起点解码。

从仓库根目录使用 `weights/` 中的最终权重，运行以下命令可重跑四组测试：

```bash
while read -r CASE DISTRIBUTION TARGETS; do
  uv run --extra cu130 python SA-NNO-DRL.py \
    --model "weights/${CASE}/epoch-9.pt" \
    --dataset "op_data/${DISTRIBUTION}/op_${DISTRIBUTION}${TARGETS}_len2_seed12345.pkl" \
    --num-samples 30 --uav-num 5 --device cuda:0 \
    --seed 1234 --weight-source self-trained \
    --output-dir "results/reruns/${CASE}" || break
done <<'CASES'
T100U5C const 100
T100U5U unif 100
T200U5C const 200
T200U5U unif 200
CASES
```

正式结果批次为 `results/course_report/20261102_172404/`。重跑使用独立目录；不覆盖该批次结果。每个实例完成校验后保存，四组正式记录的状态均为 `completed`，请求数量和完成数量均为 30。

## 5. 输出、评价指标与可行性检查

每次推理在输出根目录下生成带时间和随机后缀的独立目录：

| 文件 | 内容 |
| --- | --- |
| `instances.csv` | 数据集、实例编号、模型、seed、权重来源、目标/UAV 数量、航程预算、收益、耗时、访问目标数、总路线长度 |
| `routes.json` | 目标坐标与收益、基地、每架 UAV 的分配目标、访问顺序、往返长度和收益 |
| `summary.json` | 运行状态、请求/完成数量、收益与耗时的均值和标准差、调用参数、文件哈希、环境和推理 Git 信息 |
| `model_args.json` | checkpoint 配套训练配置的原样快照 |

实例、目标及 UAV 编号从 0 开始。目标编号直接对应原始 `loc/prize`；基地独立保存。路线含义为“基地 → 访问目标序列 → 基地”，空序列表示不飞行。分配给某 UAV 的目标可以多于它实际访问的目标。

收益为实际访问目标的正收益之和，const 场景下等于访问目标数。收益和耗时的标准差均为测试实例之间的样本标准差（`ddof=1`），不表示多次训练或多个种子的波动。单实例计时覆盖 SA 和内部路线校验，并执行 CUDA 同步；不含模型/数据加载、导出或绘图，包含首实例冷启动，没有独立预热。

目标覆盖率为“访问目标数 / 目标总数”；收益获取率为“获得收益 / 全部目标收益之和”，先按实例计算再取均值。全部目标收益之和只用于描述收益获取比例，不是已知最优可行解，不能据此计算相对最优解的 Gap。

已对保存结果的全部 120 个实例、600 条 UAV 路线进行独立重算检查：

- 分配集合两两不交，并覆盖全部真实目标；访问序列属于对应分配集合。
- 单条路线及不同 UAV 之间均无目标重复访问。
- 使用 float64 重算完整往返路线，全部满足长度上限 2.0。
- 重算收益、访问数量和路线长度与 CSV、JSON、汇总记录一致；最大收益差小于 `8e-6`，最大路线长度差小于 `3e-7`。

上述检查基于保存的数据与路线进行，不等同于重新加载四个正式权重执行一轮 GPU 推理。SA 中用于补齐不同长度集合的基地副本通过显式 `valid_nodes` 屏蔽，起点选择和动作选择均排除这些补齐节点；最终收益还会与路线重算收益进行一致性检查。

## 6. 正式结果与分析

| 场景 | 实例数 | 收益均值 | 收益样本标准差 | 平均耗时（秒/实例） | 耗时样本标准差 |
| --- | --- | --- | --- | --- | --- |
| T100U5C | 30 | 98.1667 | 1.2888 | 0.1698 | 0.0637 |
| T100U5U | 30 | 50.4427 | 3.5040 | 0.1702 | 0.0594 |
| T200U5C | 30 | 170.8000 | 3.9427 | 0.2639 | 0.0643 |
| T200U5U | 30 | 90.7850 | 4.4567 | 0.2538 | 0.0604 |

| 场景 | 平均访问目标数 | 平均目标覆盖率 | 平均收益获取率 | 平均单 UAV 航程 | 最大单 UAV 航程 |
| --- | --- | --- | --- | --- | --- |
| T100U5C | 98.1667 | 98.17% | 98.17% | 1.900333 | 1.999954 |
| T100U5U | 96.4333 | 96.43% | 99.04% | 1.921776 | 1.999575 |
| T200U5C | 170.8000 | 85.40% | 85.40% | 1.979882 | 1.999886 |
| T200U5U | 158.4333 | 79.22% | 89.73% | 1.985051 | 1.999972 |

100 目标场景中，5 架 UAV 能覆盖绝大多数目标；200 目标场景的平均单 UAV 航程接近预算上限，覆盖率下降。该现象与目标密度上升、航程约束更紧的情况一致，但这些结果不能说明解已达到最优。

unif 场景的收益获取率高于目标覆盖率，表明遗漏目标总体偏向低收益目标，与模型最大化收益的目标一致。const 和 unif 的收益单位分布不同，不能直接用收益绝对值高低评价两类场景谁更好。

本设备上的平均单实例求解耗时约为 0.17 秒（T100）和 0.25–0.26 秒（T200）。这是当前实现和计时口径下的单方法结果，不用于声称相对其他算法的加速或精度优势。

## 7. 训练曲线与路线可视化

四组训练均有 4,000 个记录点，step 从 0 到 199,950，间隔为 50。`cost.csv` 无表头，首列是 pandas 索引，第二列为 `avg_cost`；精确 step 来自 TensorBoard events。绘图使用 `training_reward = -avg_cost`，横轴 epoch 位置为 `step / 20,000`，平滑窗口为 20 个记录点。

| 场景 | 第 1 个 epoch 的平均训练收益 | 第 10 个 epoch 的平均训练收益 |
| --- | --- | --- |
| T100U5C | 17.9849 | 18.9711 |
| T100U5U | 9.2762 | 9.7032 |
| T200U5C | 28.6313 | 31.9328 |
| T200U5U | 15.5102 | 16.9135 |

表中每个 epoch 的均值来自 400 个日志记录点，并非全部 20,000 次更新的逐批次均值。训练收益是单 UAV 子问题上的候选/增强统计，不等于多 UAV 正式测试的总收益。曲线总体提高、后期增幅减小，可以描述为逐渐趋于平稳；缺少独立验证曲线和多训练种子，不能据此证明无过拟合或跨种子稳定收敛。

查看原始 TensorBoard 曲线：

```bash
uv run --extra cu130 tensorboard --logdir logs --host 127.0.0.1
```

正式材料中的 `training_curves/training_curves.png`、`training_curves.pdf` 及四份 `*_training.csv` 保存了曲线与数据。`route_figures/` 下的四组 PNG/PDF 展示各组实例 0、1、2，直接来自正式测试保存的路线。

路线图采用论文 Fig. 7 中 SA 方法的风格：蓝点为目标，黑方块为基地，不同颜色和箭头表示各 UAV 的闭合路线，标题为总收益。本次四组均使用中心基地，没有 OR-Tools 对照行，也不声称与论文 Fig. 7 的原始目标坐标或基地设置完全相同。

若要通过现有推理入口生成新的三实例示意图，可执行：

```bash
uv run --extra cu130 python SA-NNO-DRL.py \
  --model weights/T200U5C/epoch-9.pt \
  --dataset op_data/const/op_const200_len2_seed12345.pkl \
  --num-samples 3 --uav-num 5 --device cuda:0 \
  --seed 1234 --weight-source self-trained \
  --output-dir results/figure-reruns/T200U5C \
  --plot-dir results/fig7 --plot-name T200U5C --plot-format png
```

将 `--plot-format` 改为 `pdf` 可输出矢量图。该命令会重新推理，不是读取正式批次的历史路线；报告中的正式路线图以已保存的对应实例为准。

## 8. 来源与复现边界

四组推理记录的 Git revision 均为 `0142df989696cd893a9ea9313f5e5a02b1437464`，且 `inference_git_dirty=true`。这标识推理时的基础提交和存在工作区改动的事实，不能把该提交单独当作当时源码的完整快照，也不能据此认定训练代码版本。训练命令在本登记中按配置重建，原始训练时的精确代码版本没有完整记录。

当前实现与论文仍有细节差异，包括预测头的归一化、坐标增强方式和 SA shift 算子的方向语义；论文原始测试样本身份也未建立对应。因此，本次结论限定为：在所列环境、数据和配置下，完成了 SA-NNO-DRL 四组场景的训练与测试，得到满足约束的路线、可核查的收益与耗时，以及训练和路线可视化，不宣称与论文表格数值完全一致。
