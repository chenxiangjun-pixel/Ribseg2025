## RibSeg 肋骨点云分割 — 训练与测试指南

本项目实现了肋骨二分类点云分割（背景/肋骨）。包含训练、验证、测试与端点检测导出、TensorBoard 可视化与早停机制等。

### 1. 环境准备
- **Python**: 3.8+（建议 3.8–3.10）
- **PyTorch**: 1.10+（与本机 CUDA 版本匹配）
- 其他依赖：numpy, PyYAML, tqdm, tensorboard

安装示例（CPU，仅供参考）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pyyaml tqdm tensorboard
```

安装示例（CUDA，请到 PyTorch 官网选择与你的 CUDA 匹配的指令）：
```bash
# 例：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pyyaml tqdm tensorboard
```

### 2. 数据准备
目录结构（默认根目录 `data/pn/`，可在配置/参数中修改）：
```
data/pn/
  data_pn/
    train/   # 训练点云 .npy，每个文件形如 [N, C]，至少包含 xyz 三列
    val/     # 验证点云 .npy
    test/    # 测试点云 .npy
  label_pn/
    train/   # 训练标签 .npy，每个文件 [N]，前景肋骨标为1，背景为0
    val/
    test/
```
说明：
- 加载器 `data_utils/RibFracDataLoader_1cls.py` 会对每个样本随机采样 `npoints`（默认 30000），并做归一化。
- 若标签存在多值，代码会将非 0 全部映射为 1（即二分类）。

### 3. 关键脚本与配置
- 训练脚本：`train.py`
- 测试脚本：`test.py`
- 模型实现：`models/ribseg_model.py`
- 数据加载：`data_utils/RibFracDataLoader_1cls.py`
- 配置文件：`config/train_config.yaml`

日志与权重默认写入：`log/part_seg/<experiment_name>/`

### 4. 训练
基础用法：
```bash
python train.py --config config/train_config.yaml
```

可选参数（覆盖配置文件对应项）：
- `--batch_size` 覆盖 `data.batch_size`
- `--epochs` 覆盖 `training.epochs`
- `--lr` 覆盖 `training.learning_rate`
- `--device` 指定 `cuda`/`cpu`/`auto`（默认 auto）

示例：
```bash
python train.py --config config/train_config.yaml --batch_size 8 --epochs 50 --lr 1e-3 --device auto
```

训练输出：
- `log/part_seg/<experiment_name>/training.log`：训练/验证日志
- `log/part_seg/<experiment_name>/tensorboard/`：TensorBoard 事件文件
- `log/part_seg/<experiment_name>/best_model.pth`：验证 Dice 最优权重（自动断点续训）
- `log/part_seg/<experiment_name>/endpoints/val_endpoints_epoch_*.json`：每个验证周期的端点检测结果导出

早停与学习率：
- 配置于 `config/train_config.yaml` 中 `early_stopping` 与 `training.step_size/lr_decay`。

Windows 注意事项：
- 训练与测试均已针对 Windows 将 `DataLoader` 的 `num_workers` 设为 0，避免多进程问题。

断点续训：
- 若 `log/part_seg/<experiment_name>/best_model.pth` 已存在，训练会自动从其中的状态恢复（包括优化器与起始 epoch）。

### 5. TensorBoard 可视化
训练时会记录学习率、loss 与各类指标：
```bash
tensorboard --logdir log/part_seg/<experiment_name>/tensorboard
```
浏览器访问命令行提示的本地地址查看曲线。

### 6. 测试
基础用法：
```bash
python test.py \
  --data_root ./data/pn \
  --split test \
  --num_point 30000 \
  --batch_size 8
```

权重加载策略：
- 若指定 `--weights path/to/best_model.pth`，将优先使用该路径
- 否则从 `log/part_seg/<experiment_name>/best_model.pth` 加载（`--experiment_name` 默认 `ribseg_experiment`）

其他常用参数：
- `--device`：`cuda`/`cpu`/`auto`
- `--normal_channel`：是否启用法向量通道（需与数据一致）
- `--use_attention`：是否启用模型内的注意力模块

测试输出：
- 终端打印聚合指标：Acc、Dice、IoU、Precision、Sensitivity、Specificity、F1

### 7. 配置文件说明（`config/train_config.yaml`）
核心字段（部分）：
- `model.name`: `pointnet2_rib_seg`（当前模型）
- `model.num_part`: 2（二分类）
- `model.normal_channel`: 是否使用法向量
- `model.use_attention`: 是否启用注意力
- `data.root`: 数据根目录，默认 `./data/pn/`
- `data.npoint`: 每样本采样点数
- `data.batch_size`, `data.num_workers`, `data.pin_memory`
- `training.epochs`, `training.learning_rate`, `training.optimizer(Adam/AdamW/SGD)`, `training.weight_decay`, `training.step_size`, `training.lr_decay`
- `early_stopping.patience`, `early_stopping.min_delta`
- `experiment.name`: 实验名称（决定日志/权重写入目录）

提示：`train.py` 中还支持 `training.use_provider_aug`（若未出现在配置中，默认启用），用于 CPU 端的点云随机增强（旋转/缩放/平移/抖动/打乱）。

### 8. 端点检测导出
验证阶段会根据预测掩码执行端点检测（`data_utils/endpoint_utils.py`）：
- 全局端点：`detect_endpoints_from_points`
- 按连通域端点：`detect_endpoints_per_component`
- 24×2 端点：`detect_24x2_endpoints`

导出路径：`log/part_seg/<experiment_name>/endpoints/val_endpoints_epoch_#.json`

### 9. 常见问题
- 指定的权重不存在：确认 `--weights` 路径或 `--experiment_name` 下是否存在 `best_model.pth`
- CUDA 未被使用：将 `--device` 设为 `cuda`，或检查 PyTorch/CUDA 安装是否匹配
- 数据维度不一致：确保点云 `.npy` 至少包含 xyz 三列，标签 `.npy` 为形如 `[N]` 的整数数组（0/1）

### 10. 复现示例
1) 准备数据至 `data/pn/`（见第 2 节）

2) 启动训练（自动写日志与最优权重）：
```bash
python train.py --config config/train_config.yaml
```

3) 使用最优权重在测试集评估：
```bash
python test.py --data_root ./data/pn --split test --num_point 30000 --batch_size 8 --device auto --experiment_name ribseg_experiment
```

如需自定义实验名：修改 `config/train_config.yaml` 中 `experiment.name`，或复制该文件创建新的配置并在训练时指定。

