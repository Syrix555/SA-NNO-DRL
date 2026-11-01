# 训练材料与推理结果登记

更新日期：2026-11-01。

本登记依据用户在 GPU 设备上提供的目录树和此前的运行输出。当前本地工作区没有这两轮训练的 `outputs/`、`logs/`，因此下表表示已报告的材料，文件内容和哈希仍需在 GPU 设备上核验。

## 1. 正式训练材料

用户报告两轮均为自己训练，模型来源在推理时填写 `--weight-source self-trained`。

| 分布 | 训练目录 | 已报告文件 |
| --- | --- | --- |
| const | `outputs/paper/op_40/t40_const_full_20261028T220058/` | `args.json`、`cost.csv`、`epoch-0.pt` 至 `epoch-9.pt` |
| unif | `outputs/paper/op_40/t40_unif_full_20261029T030049/` | `args.json`、`cost.csv`、`epoch-0.pt` 至 `epoch-9.pt` |

对应 TensorBoard 材料：

```text
logs/op_40/t40_const_full_20261028T220058/
  events.out.tfevents.1793196058.Long-Jiang-MS-7E45.2624298.0
logs/op_40/t40_unif_full_20261029T030049/
  events.out.tfevents.1793214049.Long-Jiang-MS-7E45.2751562.0
```

`gpu_smoke`、`gpu_resume` 和 `pilot` 是环境、恢复训练或试跑材料，正式指标使用 `paper` 下的对应模型。默认选用两轮的 `epoch-9.pt`；若以后按其他标准选模型，需要记录选择规则。

目前能确认材料已覆盖最终模型、逐 epoch checkpoint、训练配置、CSV 训练曲线及 TensorBoard 日志。还需要核实原始训练命令与训练时的代码版本。目录名中的时间是运行标识，不能证明当时的代码或完整命令。当前推理程序记录的 Git 版本属于推理阶段。

此前报告的参数为：

```text
--graph_size 40
--uav_num 5
--data_distribution const / unif
--batch_size 64
--epoch_size 1280000
--val_size 10000
--eval_batch_size 1024
--n_epochs 10
--N_aug 4
--checkpoint_epochs 1
--seed 1234
--output_dir outputs/paper
```

这是一份依据对话整理的配置摘要，原始执行命令尚未留存到本登记。实际设置以各自 `args.json` 为准，还应核对 `load_path`、`resume`、网络层数、学习率、baseline 等字段。若原始终端记录已不可得，使用“依据 args.json 重建的等效命令”标注补记，不能将重建命令或当前 Git HEAD 当作历史证据。已有模型可以继续用于修复后的推理，当前修改没有改变模型参数结构。

## 2. 在 GPU 设备上归档

从仓库根目录执行以下命令，保留全部正式训练 checkpoint 和日志，并生成内容校验清单：

```bash
mkdir -p results/training-evidence

find \
  outputs/paper/op_40/t40_const_full_20261028T220058 \
  outputs/paper/op_40/t40_unif_full_20261029T030049 \
  logs/op_40/t40_const_full_20261028T220058 \
  logs/op_40/t40_unif_full_20261029T030049 \
  -type f -exec sha256sum {} + \
  > results/training-evidence/SHA256SUMS

tar -czf results/training-evidence/t40-training-artifacts.tar.gz \
  outputs/paper/op_40/t40_const_full_20261028T220058 \
  outputs/paper/op_40/t40_unif_full_20261029T030049 \
  logs/op_40/t40_const_full_20261028T220058 \
  logs/op_40/t40_unif_full_20261029T030049
```

完整归档需要额外磁盘空间，约为被打包材料的压缩后大小。原有 `outputs/`、`logs/` 与 `results/` 均被 Git 忽略，归档应另行备份。

训练曲线已经有两份来源：`cost.csv` 和 TensorBoard events。查看 events：

```bash
uv run --extra cu130 tensorboard --logdir logs/op_40 --host 127.0.0.1
```

在远程设备上使用 SSH 端口转发访问 TensorBoard。报告中可从 TensorBoard 导出 `avg_cost` 的图和数据，并与归档放在一起。当前 `cost.csv` 无表头，首列是 pandas 写入的索引，第二列是 `avg_cost`；它没有显式 step/epoch 字段，精确横轴优先使用 TensorBoard events。

`avg_cost` 是训练批次的负收益统计，更负表示训练候选通常收集到更多收益。当前训练循环虽创建验证数据集，但没有在每个 epoch 调用 `validate()`，因此这些曲线不能标为独立验证集指标，也不能用作论文表中的 Obj./Gap。

## 3. 正式推理与结果格式

const 示例：

```bash
uv run --extra cu130 python SA-NNO-DRL.py \
  --model outputs/paper/op_40/t40_const_full_20261028T220058/epoch-9.pt \
  --dataset op_data/const/op_const40_len2_seed12345.pkl \
  --num-samples 30 --uav-num 5 --device cuda:0 \
  --seed 1234 --weight-source self-trained \
  --output-dir results/sa
```

unif 示例：

```bash
uv run --extra cu130 python SA-NNO-DRL.py \
  --model outputs/paper/op_40/t40_unif_full_20261029T030049/epoch-9.pt \
  --dataset op_data/unif/op_unif40_len2_seed12345.pkl \
  --num-samples 30 --uav-num 5 --device cuda:0 \
  --seed 1234 --weight-source self-trained \
  --output-dir results/sa
```

每次生成独立的 `results/sa/<数据集>_<UTC时间>_<随机后缀>/`，终端打印具体路径：

| 文件 | 用途 |
| --- | --- |
| `instances.csv` | 逐实例的 dataset、instance_id、model、seed、weight_source、target_count、uav_num、max_length、reward、duration_seconds、visited_target_count、total_route_length |
| `routes.json` | 目标坐标与收益、基地、每架 UAV 的分配目标和实际访问顺序、返回基地后的长度、重算收益 |
| `summary.json` | 请求/实际完成数量、状态、收益/耗时均值和样本标准差、运行配置、调用参数、文件 SHA-256、环境及推理 Git 状态 |
| `model_args.json` | 与实际加载 checkpoint 配套的 `args.json` 原样快照 |

实例 ID、目标 ID 和 UAV ID 均从 0 开始。目标 ID 指向 `loc/prize`，不使用 SA 内部的局部编号；基地独立存储。路线语义为“基地 → visited_target_ids → 基地”，空序列表示该 UAV 不飞行。分配集合 `assigned_target_ids` 可能包含没有被访问的目标。

保存前沿用 SA 的路线校验：检查全局目标重复访问、往返航程和 `best_fit` 与重算收益的一致性。每完成一个实例刷新结果；正常结束状态为 `completed`，可捕获的中断/异常分别标为 `interrupted`/`failed`，强制终止可能留下 `running`。各文件独立原子替换，汇总最后写入；正式汇总时检查 `status` 和实际完成数量。

收益与耗时的标准差均为同一测试集中跨实例的样本标准差（`ddof=1`）；不足两个实例时为 `null`。这不代表跨随机种子训练的波动。耗时覆盖单实例 SA 与路线校验，并在 GPU 上同步；排除加载、导出和绘图，但包含首实例冷启动。训练 seed 与推理 seed 是不同字段：推理未传 `--seed` 时记录为 `null`。固定 seed 不保证跨硬件/库版本逐位一致。

`--model` 可传文件或目录；传目录时会解析到 epoch 编号最大的 checkpoint，保存其真实绝对路径和哈希。`--weight-source` 是用户声明的来源，默认 `unspecified`，不根据目录名猜测。元数据的 Git 信息描述推理代码；工作树为 dirty 时，单个 commit 不能完整表示运行代码，正式实验应保留对应源码状态。

`--plot-dir` 仍可用于生成 Fig. 7 风格的 SA 路线图。图像与结构化结果可同时输出；图像把全部实例放在一行，建议只选少量实例用于展示。

## 4. 本轮验收与后续边界

本轮在本地 CPU 上使用临时随机权重和小数据集执行了真实 CLI 推理，验证了目录模型解析、CSV/JSON 导出、路线全局编号、航程与收益一致性、统计计算、空路线、相同 seed 重复运行以及可选 Fig. 7 绘图。随机权重测试仅验证程序行为。GPU 设备上的正式权重推理结果需按上述命令生成。

此前两轮 T40、U5 的 30 实例终端输出属于初步运行证据；它们产生于补齐节点屏蔽修复之前，缺少逐实例路线记录与显式推理 seed。后续正式表格应使用修复后重新推理的结构化结果。

已具备保存本方法 Obj./Time 原始记录和跨实例标准差的能力；完整实验矩阵驱动、多方法合并、Gap、最优解计数与显著性检验还需后续实现。Gap 依赖对照结果，当前汇总不生成 Gap。
