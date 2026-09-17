# 1 引言

在双碳目标和新型电力系统建设的推动下，风电装机规模持续增长，其出力在电力系统中的影响日益显著。风能具有随机性、间歇性和时变性，风电功率随气象条件快速波动，给机组组合、备用配置、电力交易和实时调度带来挑战。准确可靠的风电功率预测能够为调度决策提供前瞻性信息，降低系统平衡成本并提高新能源消纳水平[1,2]。因此，面向实际运行条件开展鲁棒风电功率预测具有重要的工程价值。

数据驱动预测方法的有效性依赖于历史观测的完整性和质量。然而，风电场监督控制与数据采集（supervisory control and data acquisition，SCADA）系统在长期运行中易受传感器故障、通信中断、设备检修和环境干扰等因素影响，使功率序列出现零散或连续缺失[3]-[5]。缺失数据会削弱模型对功率时序依赖和变化趋势的学习能力；特别是在风速突变时，连续缺失会使关键运行信息中断。直接删除不完整样本会损失有效数据，而线性插值、均值填补等方法又难以描述非线性、强波动的功率演化过程，进而将重构误差传递至预测环节[3,6]。由此可见，缺失数据处理不能局限于降低插补误差，还应以提升最终功率预测性能为导向。

针对风机数据缺失问题，已有研究从统计插值、时间序列建模和深度生成模型等角度开展了探索。GAN 和掩码自编码器等模型能够挖掘多变量 SCADA 数据之间的复杂相关性，在缺失值重构任务中取得了较好的效果[4,5]。但在填补与预测串行处理中，填补模型往往以局部数值误差为主要目标，重构序列未必能够保留对预测有价值的时序结构和功率变化规律[6,7]。与此同时，风速、风向、温度、气压和湿度等气象因素共同塑造风机的运行状态和功率响应[1,8]。不同气象条件下的功率分布及其演化特征并不相同，若忽略这种差异，训练样本易受常见工况主导，模型对小风、大风等样本相对稀少或波动更为显著的工况难以获得充分学习。

扩散模型通过逐步加噪和反向去噪学习复杂数据分布，能够在条件信息约束下生成与观测数据一致的样本。条件得分扩散模型已在时间序列插补中展现出对观测关联和不确定性的刻画能力[9]，扩散式时间序列模型在多变量概率预测中也表现出良好的适用性[10]。对于存在缺失的风电功率序列，已观测片段和缺失掩码能够为反向生成过程提供约束，使模型在恢复缺失值的同时保持与整体功率演化相一致的时序特征。这为气象信息引导下的缺失序列重构与功率预测提供了合适的建模基础。

基于上述分析，本文提出一种基于气象信息引导与条件扩散模型的风电功率预测方法。首先，为增强模型对不同气象条件及缺失模式的适应性，原始功率样本与同期气象变量相结合，并以风速为主导特征结合 K-Means 无监督聚类算法，形成具有差异化功率特征的气象信息样本集合；然后，通过分层扩充构造随机点缺失和随机连续缺失情形。最后，条件扩散模型以功率观测片段和缺失掩码为条件，对缺失序列与未来功率进行联合建模。该方法通过提高缺失区间重构的时序一致性和变化合理性，提升缺失观测条件下的风电功率预测能力。

# 参考文献

[1] A. M. Foley, P. G. Leahy, A. Marvuglia, and E. J. McKeogh, “Current methods and advances in forecasting of wind power generation,” *Renewable Energy*, vol. 37, no. 1, pp. 1-8, Jan. 2012.

[2] P. Pinson, “Wind energy: Forecasting challenges for its operational management,” *Statistical Science*, vol. 28, no. 4, pp. 564-585, 2013.

[3] R. Tawn, J. Browell, and I. Dinwoodie, “Missing data in wind farm time series: Properties and effect on forecasts,” *Electric Power Systems Research*, vol. 189, Dec. 2020, Art. no. 106640.

[4] F. Qu, J. Liu, Y. Ma, D. Zang, and M. Fu, “A novel wind turbine data imputation method with multiple optimizations based on GANs,” *Mechanical Systems and Signal Processing*, vol. 139, May 2020, Art. no. 106610.

[5] Y. Fan, C. Feng, R. Wu, C. Liu, and D. Jiang, “Multiscale-attention masked autoencoder for missing data imputation of wind turbines,” *Knowledge-Based Systems*, vol. 299, Sep. 2024, Art. no. 112114.

[6] H. Wen, P. Pinson, J. Gu, and Z. Jin, “Wind energy forecasting with missing values within a fully conditional specification framework,” *International Journal of Forecasting*, vol. 40, no. 1, pp. 77-95, Jan.-Mar. 2024.

[7] Z. Meng, Y. Guo, and C. Zhao, “Probabilistic wind power forecasting with missing data tolerance: An end-to-end nonparametric approach,” *IEEE Transactions on Sustainable Energy*, vol. 17, no. 2, pp. 1202-1213, Apr. 2026.

[8] J.-Y. Ryu, B. Lee, S. Park, *et al.*, “Evaluation of weather information for short-term wind power forecasting with various types of models,” *Energies*, vol. 15, no. 24, Dec. 2022, Art. no. 9403.

[9] Y. Tashiro, J. Song, Y. Song, and S. Ermon, “CSDI: Conditional score-based diffusion models for probabilistic time series imputation,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021.

[10] K. Rasul, C. Seward, I. Schuster, and R. Vollgraf, “Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting,” in *Proceedings of the 38th International Conference on Machine Learning*, vol. 139, 2021, pp. 8857-8868.
