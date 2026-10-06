# 风电功率重构实验

在 `diffusion312` 环境中运行：

```powershell
conda run -n diffusion312 python src/run_experiment.py --config config.json
```

程序自动完成数据准备、GAN 与两种扩散模型训练、线性插值基线评价、四目标重构评价和预测支持评价，并将四张结果表写入 `results/`。训练过程和检查点位于 `work_v2/`。

## 模型与缺失条件

比较四种重构方法：

- `linear_interpolation`：不训练的线性插值基线；缺失段内部线性插值，边界使用最近观测值。
- `gan_imputer`：带气象条件的生成对抗重构模型。
- `diffusion_no_weather`：不输入气象变量的扩散模型。
- `proposed_weather_diffusion`：输入气象变量的扩散模型。

所有方法使用完全相同的测试窗口和掩码。掩码分开评价，不再使用混合缺失：

- `random_point`：每个功率通道随机遮蔽 5% 的时间点。
- `random_continuous`：每个功率通道随机遮蔽连续 1、2、3 小时。

`mask_type` 标识掩码类别；随机点条件另外写入 `missing_rate`，连续条件写入 `missing_length`。

## 结果表

四张表只保留最大绝对误差 `abs_error_max`、最小绝对误差 `abs_error_min` 和平均绝对误差 `MAE`，其中 MAE 是主要比较指标。数值越小表示重构或预测越准确。

### `reconstruction_comparison.csv`

总体重构结果。每行对应一种方法、一个功率字段、一个训练种子和一种掩码条件。`target_field` 为 `Power_0`、`Power_15`、`Power_30` 或 `Power_45`；`n_missing_points` 是参与评价的缺失点数量。用于比较四种方法在不同功率目标和缺失条件下的整体表现。

### `reconstruction_grouped_comparison.csv`

按气象工况分组的重构结果。除上述字段外，`weather_regime` 表示低风、中风或高风工况。该表用于判断模型在不同气象条件下的稳定性，掩码类别仍由 `mask_type` 区分。

### `forecast_support_comparison.csv`

预测支持结果。它用窗口末行的真实功率字段作为目标，比较完整历史、直接遮蔽历史和重构历史对下游预测的影响。`history_setting` 为 `complete`、`masked` 或 `reconstructed`，`reconstruction_method` 记录具体重构方法。

### `ablation_comparison.csv`

气象条件消融结果。`remove_weather` 表示移除气象输入的扩散模型；将其与同一条件下的 `proposed_weather_diffusion` 对比，可评估气象变量的贡献。其他训练、采样、窗口和掩码条件保持一致。

## 建议分析顺序

先在 `reconstruction_comparison.csv` 中按 `target_field` 和 `mask_type` 比较四种方法的 MAE；再查看 `reconstruction_grouped_comparison.csv` 的低、中、高风工况差异；随后检查预测支持表中重构历史是否接近完整历史；最后使用消融表判断气象条件是否带来稳定收益。
