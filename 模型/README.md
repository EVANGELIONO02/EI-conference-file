# 数据扩充与风电功率预测实验

## 实验设计

本实验评估不同数据扩充方法对LSTM风电功率预测准确率的影响。

### 核心逻辑

1. **Diffusion模型仅用于数据扩充**：训练Diffusion模型来填充训练数据中的缺失值
2. **LSTM模型用于最终预测**：使用扩充后的完整数据训练LSTM预测模型
3. **对比评估标准**：以LSTM在测试集上的预测准确率为准

### 实验流程

#### 阶段1：训练数据扩充模型

训练两个Diffusion模型用于数据扩充：
- **无气象条件的Diffusion模型** (`diffusion_no_weather`)
- **带气象条件的Diffusion模型** (`diffusion_with_weather`)

#### 阶段2：数据扩充与LSTM预测模型训练

对每种扩充方法：

1. **数据扩充**：
   - **线性插值** (`linear_interpolation`)：使用线性插值填充缺失值
   - **Diffusion模型**：使用训练好的Diffusion模型填充缺失值
   - **带气象条件的Diffusion模型**：使用气象信息引导的Diffusion模型填充缺失值

2. **训练LSTM预测模型**：
   - 输入：历史12小时的功率和气象数据
   - 输出：未来12小时的功率预测
   - 使用扩充后的完整训练数据

3. **评估**：
   - 在测试集上评估LSTM预测准确率
   - 按风速分组（低风、中风、高风）报告结果

### 对比指标

最终表格对比三种数据扩充方法的效果，评估指标：
- **MAE (Mean Absolute Error)**：平均绝对误差
- **RMSE (Root Mean Square Error)**：均方根误差

结果按气象工况分组报告：
- `low_wind`：低风速条件
- `medium_wind`：中风速条件  
- `high_wind`：高风速条件

## 文件说明

### 核心代码文件

- `lstm_forecaster.py`：LSTM预测模型实现
- `train_and_evaluate.py`：新的训练和评估流程
- `run_experiment.py`：实验运行入口
- `diffusion_model.py`：Diffusion扩充模型（复用原有）
- `data_pipeline.py`：数据处理流程（复用原有）

### 配置文件

`config.json` 中的关键参数：

```json
{
  "forecast_input_steps": 12,     // LSTM输入历史长度（小时）
  "forecast_steps": 12,            // LSTM预测未来长度（小时）
  "lstm_hidden_dim": 427,          // LSTM隐藏层维度
  "lstm_layers": 2,                // LSTM层数
  "lstm_bidirectional": true,      // 是否使用双向LSTM
  "diffusion_model_dim": 256,      // Diffusion模型维度
  "diffusion_layers": 8,           // Diffusion Transformer层数
  "train_seeds": [11, 22, 33]      // 随机种子（重复实验）
}
```

## 运行实验

### 完整实验

```bash
cd 模型/src
python run_experiment.py --config ../config.json
```

### 快速测试

```bash
python run_experiment.py --config ../config.json --quick
```

## 结果输出

### 主要结果文件

- `results/forecast_augmentation_comparison.csv`：对比表格（主要结果）

表格结构示例：

| weather_regime | train_seed | linear_interpolation_MAE | diffusion_no_weather_MAE | diffusion_with_weather_MAE | linear_interpolation_RMSE | ... |
|----------------|------------|--------------------------|--------------------------|----------------------------|---------------------------|-----|
| low_wind       | 11         | 0.125                    | 0.118                    | 0.112                      | 0.156                     | ... |
| medium_wind    | 11         | 0.143                    | 0.139                    | 0.131                      | 0.178                     | ... |
| high_wind      | 11         | 0.167                    | 0.159                    | 0.148                      | 0.203                     | ... |

### 日志和检查点

- `work/logs/`：训练日志
  - `train_diffusion_no_weather_seed*.csv`：Diffusion模型训练日志
  - `train_diffusion_with_weather_seed*.csv`：带气象Diffusion训练日志
  - `train_lstm_forecaster_*_seed*.csv`：LSTM预测模型训练日志
  - `run_summary.json`：实验总结

- `work/checkpoints/`：模型检查点
  - `diffusion_no_weather_seed*.pt`：Diffusion扩充模型
  - `diffusion_with_weather_seed*.pt`：带气象Diffusion模型
  - `lstm_forecaster_*_seed*.pt`：LSTM预测模型

## 实验设计理念

### 与原实验的区别

**原实验逻辑**：
- 各模型（LSTM、Diffusion）都直接用于缺失值填充
- 对比各模型在填充任务上的重建准确率

**新实验逻辑**：
- Diffusion模型专门用于数据扩充（生成高质量的完整训练数据）
- LSTM模型专门用于预测任务
- 对比不同扩充方法对预测模型最终性能的影响

### 关键创新点

1. **两阶段分离**：扩充和预测任务分离，各司其职
2. **端到端评估**：直接评估最终预测准确率，而非中间填充准确率
3. **实用性导向**：更接近实际应用场景（用完整数据训练预测模型）

## 预期结果

预期 `diffusion_with_weather` 方法应该取得最好的预测准确率，因为：
1. 利用了气象信息的引导
2. 生成的扩充数据更符合真实物理规律
3. 为LSTM提供了更高质量的训练数据

## 注意事项

1. **计算资源**：完整实验需要GPU，预计运行时间数小时
2. **数据要求**：需要完整的风电功率和气象数据
3. **随机种子**：使用多个种子重复实验以确保结果可靠性
4. **内存占用**：数据扩充阶段会复制训练数据，需注意内存使用

## 后续分析

结果表格可用于：
- 绘制不同扩充方法在各气象工况下的对比柱状图
- 统计显著性检验
- 分析气象条件对预测准确率的影响
- 评估计算成本与性能的权衡
