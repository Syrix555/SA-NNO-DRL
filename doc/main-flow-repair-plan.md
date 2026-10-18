# SA-NNO-DRL 主流程修复计划

日期：2026-10-17

状态：待实施。本文件记录修复计划，不代表代码已修复或完整训练已通过。

阶段一状态：已实施代码与 uv 项目配置；目标 GPU 上的训练和 SA 端到端验证待执行。

阶段二状态：源码修改与候选选择测试已准备；目标机器验证待执行。本机不执行测试。

阶段一环境命令（目标 RTX 5080 机器）：

```bash
uv sync --extra cu130
uv run --extra cu130 python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CPU 逻辑检查使用：

```bash
uv sync --extra cpu
uv run --extra cpu python -m pytest
```

阶段二目标 GPU 验证使用：

```bash
uv run --extra cu130 pytest tests/test_stage1_data.py -q
```

该测试现在也覆盖 POMO 候选成本与路线的同步选择，以及 `sample_many` 的按原始样本聚合；本机按用户要求不执行。

两个 extra 互斥；不要同时启用 `cpu` 和 `cu130`。`uv.lock` 已锁定 Python 3.11 的解析结果，GPU extra 从 PyTorch CUDA 13.0 索引安装 `torch==2.14.1+cu130`。

## 1. 目标与运行环境

打通以下链路：

`OP 数据加载/生成 → 小批次训练 → checkpoint 保存 → 恢复训练/加载推理 → SA 多 UAV 任务规划`

目标环境固定为用户已验证的组合：

| 组件 | 版本/型号 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA 运行时 | 13.0 |
| NVIDIA 驱动 | 580.159.03 |
| 显卡 | NVIDIA GeForce RTX 5080，计算能力 12.0 |

目标机器已通过 CUDA 可用性、`sm_120` 架构支持和 GPU 矩阵运算检查，但尚未安装 NumPy，也尚未验证仓库训练。

当前审查机器没有 PyTorch，不能替代目标机器完成前向、反向和 CUDA 验证。最终依赖版本以目标环境实际通过的结果锁定。

## 2. 范围与原则

- 优先支持 `--problem op --model attention`，覆盖已有 const/unif 数据、训练、模型恢复和 SA 推理。
- 保留现有 POMO 多起点、数据增强和一致性损失的算法意图；改变损失或归一化语义前，先核对论文和可获得的原始实现，并记录依据。
- 统一数据、张量形状和模型输出约定，再处理调用方，避免仅消除报错却选错路线。
- TSP/VRP/PCTSP、pointer 模型、beam search、外部求解器和 TensorFlow 基线暂不修复；暂不支持的选项应尽早给出明确错误。
- 不把完整训练收敛或论文指标复现作为本轮验收条件，也不承诺旧权重经过行为调整后仍能复现原指标。

## 3. 已确认的问题与处理方式

### 3.1 依赖与入口

涉及：`run.py`、`utils/log_utils.py`、`options.py`、`SA-NNO-DRL.py`，以及新增依赖说明。

问题：仓库没有依赖清单；训练入口强制导入旧 `tensorboard_logger`；全局屏蔽 warning；GPU 与推理模型路径写死。

计划：

1. 将日志迁移到 `torch.utils.tensorboard.SummaryWriter`，同步替换调用并关闭 writer；关闭 TensorBoard 时不要求导入该可选依赖。
2. 声明主流程需要的 NumPy、SciPy、pandas、matplotlib、tqdm 和 TensorBoard；PyTorch CUDA 包单独说明，避免安装普通依赖时意外替换已验证的构建。
3. 移除全局 warning 屏蔽，仅在有依据时局部过滤具体警告。
4. 通过参数配置模型、数据、输出路径和设备；尊重外部 `CUDA_VISIBLE_DEVICES`，保留 CPU 小规模检查入口。
5. 缺少 checkpoint 或 `args.json` 时在启动阶段报出明确路径和处理方法。当前仓库不含默认指定的 `outputs/T200U5const/epoch-9.pt`，需先训练或提供权重。

验收：训练与 SA 入口能显示帮助；关闭日志可正常启动；无模型时给出明确错误；启动检查不生成训练数据或加载大型模型。

### 3.2 数据格式、dtype 与样本数量

涉及：`make_dataset.py`、`generate_data.py`、`problems/op/problem_op.py`。

问题：

- `make_dataset.py` 使用已移除的 `np.float`，已在 NumPy 2.2.6 复现加载失败。
- 抽查的仓库数据使用 `(loc, prize, depot, max_length)`，但 `OPDataset` 和 `generate_op_data()` 使用 `(depot, loc, prize, max_length)`。仓库同时存在两种格式。
- 部分已有数据的 depot 带有额外的单元素维度，需要规范化。
- OP 生成器写死 5 架 UAV，并将 `num_samples` 放大为 5 倍，未遵循命令行 `uav_num`。
- 训练生成器和离线生成器的航程默认值不同，需要明确配置来源。

计划：

1. 内部统一为字段明确的字典：`loc`、`prize`、`depot`、`max_length`；通过共同加载逻辑规范化两类旧 tuple，不重写原始数据文件。
2. 兼容判断依据结构和形状，遇到歧义或不合法输入直接报错，不静默交换字段。
3. 使用明确的浮点类型，消除主流程中的 `np.float`；统一模型输入为 float32，路线索引为 int64。
4. 定义 `num_samples` 为最终返回的训练实例数，`graph_size` 为每个分区的节点数；将 `uav_num` 传入生成器，处理不能整除的尾部样本。记录这与旧实现每轮实际样本数的差别。
5. 航程由文件数据或明确的生成参数提供，并写入运行配置。

内部单批次约定：

| 字段 | 形状 |
| --- | --- |
| `loc` | `[B, N, 2]` |
| `prize` | `[B, N]` |
| `depot` | `[B, 2]` |
| `max_length` | `[B]` |

验收：两种 tuple 格式加载后语义一致；const/unif 已有数据通过形状与有限值校验；生成数据长度严格等于请求数；切换 UAV 数量生效。

### 3.3 POMO 输出、验证和 baseline

涉及：`nets/attention_model.py`、`train.py`、`reinforce_baselines.py`、`utils/functions.py`、`eval.py`。

问题：模型已采用 POMO 多候选输出，但通用评估和旧 baseline 部分仍假设每个实例只有一条路线。`sample_many()` 的 reshape/选择逻辑未完整适配；rollout baseline 展平多候选成本后可能与数据集长度不一致。训练损失实际使用候选平均奖励，未采用配置的 rollout/critic baseline；正常 epoch 结束也没有执行 baseline 的 epoch callback。`rollout()` 注释为 greedy，实际启用 sampling。

计划：

1. 明确成本与选中动作概率为 `[B, K]` 和 `[B, K, T]`，路线为 `[B, K, T]`；避免继续把概率命名为对数概率。
2. 推理时仅在同一个原始样本内合并 POMO、增强和采样候选，选择最小成本，并用相同索引取得路线。OP 成本为负奖励，因此最小成本对应最大奖励。
3. 默认沿用当前损失中的候选平均奖励 baseline；取消默认 rollout baseline 的无效模型复制和预计算。旧 baseline 选项先明确限制，避免显示启用但不参与损失。
4. 验证默认采用 greedy，对每个实例聚合最佳候选，再计算均值和标准误；采样评估作为显式选项。
5. 同步检查 `N_aug`、batch 大小和末批次的 reshape；限制不支持的一致性损失配置，防止 `N_aug=1` 除零。
6. 修正 OP 仅一步路线的成本返回形状，保证与通常的 `[B, K]` 一致；为过小图规模的 POMO 数量提供校验。

验收：用可人工判断的候选成本与路线验证选择结果；批量与逐实例评估结果对应；训练不会额外预计算未使用的 rollout baseline；增强与候选维度不串样本。

### 3.4 训练稳定性与现代 API

涉及：`nets/attention_model.py`、`train.py`。

计划：

1. 给激活检查点显式传入 `use_reentrant=False`；验证开启与关闭时输出、梯度在容差内一致。
2. 核对投影头 `InstanceNorm1d` 的轴：当前输入是 `[B, N, E]`，层声明的通道数是 `E`。先确认预期归一化轴，再修正转置或归一化层；不得未经核对直接替换为 `LayerNorm`。
3. 检查采样概率到对数概率的数值稳定性，避免零概率导致 `inf/NaN`；不以大范围 clamp 掩盖非法动作或概率错误。
4. 将学习率调度与优化器更新顺序对齐，明确按 epoch 衰减和恢复时的 epoch 语义。
5. 明确随机种子，并记录 Python、NumPy、PyTorch 与设备配置；涉及数据增强的随机张量按需要显式设置 dtype/device。

验收：小批次损失和梯度均有限；至少一个可训练参数在优化后改变；训练状态和梯度检查不被 warning 屏蔽。

### 3.5 checkpoint 保存、加载和恢复

涉及：`train.py`、`run.py`、`utils/functions.py`、`reinforce_baselines.py`。

计划：

1. 采用带格式版本的 checkpoint，保存模型/优化器/调度器 state_dict、epoch、必要配置和随机状态；不保存整个模型或数据集实例。
2. 将格式设计为可用 `weights_only=True` 加载，统一为 CPU 加载后迁移设备，去掉重复加载模型文件的步骤。
3. 保留可信旧 checkpoint 的显式兼容入口；只有用户明确指定可信旧文件时才允许完整 pickle 对象加载。普通推理不默认关闭 `weights_only`。
4. 默认严格校验权重键与形状。当前将缺失权重用随机初始化参数补齐的方式可能掩盖不完整模型，应改为清楚报告差异。
5. 保存配置与权重结构版本；对投影头或数据语义变化提供迁移说明，不承诺静默兼容。
6. 保存恢复训练所需的 epoch、调度器和随机状态；CPU/CUDA 状态按设备可用性处理。

验收：新格式可以安全加载；固定输入和 greedy 模式下，保存前后成本与路线一致；恢复后 epoch、优化器和学习率正确，且可继续更新参数。

### 3.6 SA 推理与端到端检查

涉及：`SA-NNO-DRL.py`、共享数据加载与模型加载函数。

计划：

1. SA 使用统一加载器和模型输出约定，显式转换 float32 并迁移到目标设备。
2. 保留 depot 与任务、奖励的对应关系；校验按角度分区、补齐节点和还原索引后的结果。
3. 检查任务数与 UAV 数不整除、任务数小于 UAV 数、空分区及零跨度坐标的边界；不支持的情况应在分区前明确拒绝。
4. 提供小规模 smoke 配置，限制实例数与 SA 迭代次数，支持无图形界面运行并输出路径、奖励与耗时。
5. 验证路线长度不超过预算、真实任务不重复分配、补齐节点不产生额外奖励，输出奖励与原始数据重新计算一致。

验收：使用本轮生成的新 checkpoint 完成至少一个多 UAV 实例的完整 SA 推理，不依赖仓库外的默认绝对路径。

## 4. 实施顺序与阶段产物

| 阶段 | 内容 | 完成标准 |
| --- | --- | --- |
| 1 | 依赖、入口、统一数据加载和生成配置 | 数据可读，形状与样本数正确，入口可配置 |
| 2 | POMO 输出、候选选择、训练 baseline 和验证逻辑 | 单批次前向/反向成功，评估不串样本 |
| 3 | 激活检查点、归一化轴、数值稳定性、学习率调度 | 梯度有限、参数更新、相关警告得到处理 |
| 4 | 新 checkpoint 格式与恢复机制 | 保存加载一致，恢复后能继续训练 |
| 5 | SA 入口与路线约束检查 | 新模型能完成小规模多 UAV 规划 |
| 6 | 目标 GPU 验证、依赖锁定和运行说明 | RTX 5080 上端到端通过，记录可复现命令 |

每一阶段先完成最小验证，再进入下一阶段。算法语义未明确的部分应记录待核对事项，避免以随意更改实现替代验证。

## 5. 验证与交付要求

仅针对关键行为增加回归检查：数据格式转换、候选选择、损失与梯度、checkpoint 往返及路线合法性。日志和参数等低风险修改通过入口 smoke 检查，不编写机械重复实现的测试。

目标机器执行顺序：

1. 记录环境版本并安装主流程依赖，确认 GPU 计算仍正常。
2. 加载仓库已有数据，并生成少量 const/unif 数据，核对统一约定。
3. 在 CPU 完成最小逻辑检查，在 RTX 5080 完成小批次前向、反向与参数更新。
4. 完成小规模训练，保存并恢复模型，比较固定输入的 greedy 结果。
5. 使用保存的模型执行受限 SA 规划，并独立重算路线成本和奖励。

最终交付包括代码修复、关键回归检查、依赖版本记录和运行命令。验证记录必须写明实际执行的环境、命令、结果与尚未覆盖的分支，区分“源码检查”“本地验证”和“目标 GPU 验证”。

## 6. 参考

- [PyTorch 2.14.1 Python 支持与发布包](https://pypi.org/project/torch/2.14.1/)
- [PyTorch 序列化与 weights_only 行为](https://docs.pytorch.org/docs/stable/notes/serialization)
- [PyTorch 激活检查点接口](https://docs.pytorch.org/docs/stable/checkpoint)
- [NVIDIA CUDA 驱动兼容规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
