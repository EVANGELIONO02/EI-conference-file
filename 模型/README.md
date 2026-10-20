# 风电功率重构实验

在 `diffusion312` 环境中运行：

```powershell
conda run -n diffusion312 python src/run_experiment.py --config config.json
```

程序自动完成数据准备、LSTM 与两种扩散模型训练、线性插值评价，并只在 `results/` 中生成四目标分组重构对比表 `reconstruction_grouped_comparison.csv`。表中四种方法横向并列；不再生成总体 `reconstruction_comparison.csv`、预测支持表或消融表。训练过程和检查点位于 `work_v2/`。

## 模型与缺失条件

比较四种重构方法：

- `linear_interpolation`：不训练的线性插值基线；缺失段内部线性插值，边界使用最近观测值。
- `lstm_imputer`：统一训练的双向 LSTM 时序重构模型。它使用全部训练窗口，同时输入功率、掩码和 13 个气象特征，不按小风、中风或大风工况拆分训练；分组表中的风况列只用于测试误差分层。`lstm_bidirectional` 可在 `config.json` 中关闭双向结构。
- `diffusion_all_weather`：气象输入扩散基线。它输入全部 13 个气象特征，并把所有风况窗口放在一个训练任务中；“all weather”表示纳入全部气象条件，不表示删除气象变量。
- `proposed_weather_diffusion`：气象输入扩散模型。它同样输入全部 13 个气象特征，但按照小风、中风、大风三个气象簇分别训练三个独立子模型，测试时按窗口末端气象簇选择对应子模型。

两个扩散网络单体使用完全相同的 25 维输入（4 通道噪声功率、4 通道观测功率、4 通道掩码和 13 个气象特征）以及 `256/8层/8头` 网络容量。这样对比的唯一核心差异是：基线不做气象工况分类，Proposed Diffusion 按三个气象簇分别训练；Proposed Diffusion 的总参数量为三个同容量子模型之和。

所有方法使用完全相同的测试窗口和掩码。掩码分开评价，不再使用混合缺失：

- `random_point`：按 5% 配置随机点缺失。由于每个窗口有 24 个小时点且每通道至少遮蔽 1 点，实际遮蔽率为 1/24（约 4.17%）。
- `random_continuous`：每个功率通道随机遮蔽连续 1、2、3 小时。

`mask_type` 标识掩码类别；随机点条件另外写入 `missing_rate`，连续条件写入 `missing_length`。

## 结果表

评价函数只计算风况分组指标，不再维护总体评价的重复数据。预测支持函数及相关配置已移除；未使用的 `mask_types`、`mixed_precision` 配置已清理，掩码场景由 `point_missing_rates` 和 `continuous_lengths` 控制。主入口不依赖 `rebuild_grouped_comparison.py`。训练日志仍写入 `work_v2/`，不属于结果对比表。

误差指标为最大绝对误差 `abs_error_max`、最小绝对误差 `abs_error_min` 和平均绝对误差 `MAE`，其中 MAE 是主要比较指标。数值越小表示重构越准确。

### `reconstruction_grouped_comparison.csv`

四种方法的横向对比表。每行对应一个功率字段、掩码条件、风况和训练种子；`weather_regime` 为小风、中风或大风。每种方法的最大误差、最小误差和 MAE 分别放在以方法名为前缀的列中，例如 `lstm_imputer_MAE`。`train_seed` 为 11、22、33；同一行各方法使用相同的随机测试掩码。线性插值不训练，按每个种子的掩码重复评价以保持逐行公平比较。LSTM 和 `diffusion_all_weather` 都在所有风况上统一训练，分组仅用于报告测试误差；只有 `proposed_weather_diffusion` 按三个风况分别训练子模型。

## 建议分析顺序

在 `reconstruction_grouped_comparison.csv` 中按 `target_field`、`mask_type` 和 `weather_regime` 横向比较四种方法的 MAE，并查看不同训练种子是否呈现一致趋势。每行已对齐相同测试条件和随机种子，便于直接比较。
