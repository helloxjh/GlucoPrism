# GlucoPrism Master Glossary

> Status: locked for all subsequent English translation in this task. Each Chinese term has one canonical English rendering. Acronyms, capitalization, and hyphenation must remain unchanged.

## A. Paper-Specific Names and Modules

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| GlucoPrism | GlucoPrism | 模型专名，不翻译、不加连字符 |
| 面向多模态可穿戴生理信号的生理上下文图引导血糖预测网络 | physiological-context-graph-guided network for blood glucose prediction from multimodal wearable physiological signals | 论文标题核心表述 |
| 生理上下文 | physiological context | 全文唯一用法 |
| 生理上下文图 | physiological context graph | 全文图概念的统一名称 |
| 生理上下文图引导 | physiological-context-graph-guided | 作前置复合修饰语时加连字符 |
| 多尺度时间编码模块 | Multiscale Temporal Encoder (MS-TE) | 正式模块名；multiscale 全文写作一个词 |
| 生理上下文图编码模块 | Physiological Context Graph Encoder (PC-GE) | 正式模块名；统一替代正文中 PP-GSE 等并行写法 |
| 双向交叉注意力融合模块 | Bidirectional Cross-Attention Fusion Module (BC-AFM) | 正式模块名 |
| 多预测视界输出头 | Multi-Horizon Prediction Head | 正式模块名 |
| 局部动态增强特征 | local dynamics-enhanced features | CGM 与生理信号的输入增强 |
| 生理先验引导的自适应图编码机制 | physiological-prior-guided adaptive graph encoding mechanism | PC-GE 内部机制 |
| 双向交叉注意力交互 | bidirectional cross-attention interaction | BC-AFM 内部交互 |
| 置信度门控机制 | confidence-gating mechanism | BC-AFM 内部机制 |
| 置信度门控融合 | confidence-gated fusion | 融合过程 |
| 多预测视界输出 | multi-horizon output | 同时输出多个未来时刻 |
| 直接多预测视界预测 | direct multi-horizon prediction | 非递归式预测 |
| 预测视界查询向量 | horizon query | 可学习查询向量 |
| 可学习预测视界查询 | learnable horizon query | 复数写作 learnable horizon queries |
| 视界感知查询 | horizon-aware query | 查询角色 |
| 视界条件编码函数 | horizon-conditioning function | 对归一化时间距离编码 |
| 视界特异性上下文表示 | horizon-specific context representation | 每个预测视界独立检索得到 |
| 视界感知特征细化 | horizon-aware feature refinement | 输出头中的轻量细化过程 |
| 长期上下文校准 | long-term context calibration | 长视界校准过程 |
| 渐进式特征细化 | progressive feature refinement | 仅复用特征表示 |
| 残差锚点 | residual anchor | 最后一个历史 CGM 观测值 |
| 残差变化量 | residual change | 相对残差锚点的预测变化 |
| 残差式输出 | residual prediction | 输出设计 |

## B. Glucose Monitoring and Physiological Signals

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 血糖 | blood glucose | 任务目标的统一表述 |
| 血糖水平 | blood glucose level | 临床叙述 |
| 血糖值 | blood glucose value | 单个观测或预测值 |
| 血糖浓度 | blood glucose concentration | 带单位的浓度表述 |
| 血糖预测 | blood glucose prediction | 任务名称 |
| 连续血糖预测 | continuous glucose prediction | 基于连续监测序列的预测任务 |
| 连续血糖监测 | continuous glucose monitoring (CGM) | 标准术语 |
| 连续血糖监测设备 | continuous glucose monitor | 指硬件设备 |
| CGM 序列 | CGM sequence | 时间序列输入 |
| 历史 CGM 序列 | historical CGM sequence | 历史输入 |
| CGM 观测值 | CGM observation | 单个历史采样值 |
| 估算血糖值 | estimated glucose value (EGV) | Dexcom 数据字段 |
| 自我血糖监测 | self-monitoring of blood glucose (SMBG) | 标准临床术语 |
| 糖化血红蛋白 | glycated hemoglobin (HbA1c) | 统一写 HbA1c |
| 组织间液葡萄糖浓度 | interstitial glucose concentration | CGM 的直接测量对象 |
| 葡萄糖动力学 | glucose dynamics | 动态变化机制 |
| 葡萄糖代谢 | glucose metabolism | 生理过程 |
| 葡萄糖代谢紊乱 | impaired glucose metabolism | 论文研究人群语境 |
| 血糖波动 | glycemic fluctuations | 临床与结果讨论 |
| 血糖变异性 | glycemic variability | 数据集及临床概念 |
| 高血糖 | hyperglycemia | 临床术语 |
| 低血糖 | hypoglycemia | 临床术语 |
| 隐匿性高血糖 | unrecognized hyperglycemia | CGM 检出语境 |
| 隐匿性低血糖 | unrecognized hypoglycemia | CGM 检出语境 |
| 多模态可穿戴生理信号 | multimodal wearable physiological signals | 全文核心输入表述 |
| 多源外周生理信号 | multisource peripheral physiological signals | 多传感器输入表述 |
| 异构生理信号 | heterogeneous physiological signals | 强调模态差异 |
| 生理信号通道 | physiological signal channel | 单个模态通道 |
| 加速度信号 | acceleration signal | 通用表述 |
| 三轴加速度 | tri-axial acceleration (ACC) | 三个轴向数据 |
| 三轴加速度测量 | tri-axial accelerometry | 数据集采集模态 |
| 加速度模长 | acceleration magnitude (ACC-L2) | 三轴加速度的欧氏范数 |
| 皮肤电活动 | electrodermal activity (EDA) | 不使用皮电反应作同义替换 |
| 体表温度 | skin temperature (TEMP) | 与 PhysioNet 数据集命名一致 |
| 心率 | heart rate (HR) | 单位 bpm |
| 血容量脉搏 | blood volume pulse (BVP) | 不写 blood volume pulse wave |
| 心搏间期 | inter-beat interval (IBI) | 保留连字符 |
| 光电容积描记法 | photoplethysmography (PPG) | 腕戴光学传感 |
| 惯性测量单元 | inertial measurement unit (IMU) | 加速度传感器 |
| 自主神经系统 | autonomic nervous system | 生理机制 |
| 交感神经活动 | sympathetic nervous activity | 生理机制 |
| 副交感神经活动 | parasympathetic nervous activity | 生理机制 |
| 外周血液循环 | peripheral circulation | 生理机制 |
| 心血管活动 | cardiovascular activity | 生理上下文 |
| 体温调节 | thermoregulation | 生理上下文 |
| 身体活动强度 | physical activity intensity | ACC-L2 所表征状态 |
| 生理应激 | physiological stress | EDA 相关语境 |
| 代谢状态 | metabolic state | 生理上下文 |
| 碳水化合物摄入 | carbohydrate intake | 外部扰动因素 |
| 胰岛素剂量 | insulin dose | 外部干预变量 |
| 胰岛素干预 | insulin intervention | 治疗语境 |
| 糖尿病前期 | prediabetes | 受试者风险状态 |
| 1 型糖尿病 | type 1 diabetes (T1D) | 疾病名称 |
| 2 型糖尿病 | type 2 diabetes (T2D) | 疾病名称 |
| 个体化健康管理 | personalized health management | 应用场景 |
| 可穿戴健康监测 | wearable health monitoring | 应用场景 |
| 智能医疗 | smart healthcare | 应用领域 |
| 真实世界监测 | real-world monitoring | 数据采集环境 |

## C. Graph Learning and Structured Modeling

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 图神经网络 | graph neural network (GNN) | 标准术语 |
| 图卷积网络 | graph convolutional network (GCN) | 标准术语 |
| 图注意力网络 | graph attention network (GAT) | 标准术语 |
| GATv2 风格图注意力编码器 | GATv2-style graph attention encoder | 本文编码器表述 |
| 时空图神经网络 | spatiotemporal graph neural network (ST-GNN) | 统一写 spatiotemporal |
| 时空图卷积网络 | spatiotemporal graph convolutional network (STGCN) | 基线模型 |
| 扩散卷积循环神经网络 | diffusion convolutional recurrent neural network (DCRNN) | 基线模型 |
| 时空元网络 | spatiotemporal meta-network (STMetaNet) | 相关工作模型 |
| 图多注意力网络 | graph multi-attention network (GMAN) | 相关工作模型 |
| 基于注意力的时空图卷积网络 | attention-based spatiotemporal graph convolutional network (ASTGCN) | 相关工作模型 |
| 图结构 | graph structure | 通用概念 |
| 图拓扑 | graph topology | 节点与边的组织关系 |
| 拓扑约束 | topological constraint | 图结构约束 |
| 拓扑搜索空间 | topology search space | 先验掩码限定范围 |
| 图节点 | graph node | 生理信号作为节点 |
| 生理节点 | physiological node | 生理上下文图中的节点 |
| 节点集合 | node set | 图定义 |
| 边集合 | edge set | 图定义 |
| 自连接 | self-loop | 图中节点自身连接 |
| 邻域集合 | neighborhood | 掩码限定的相邻节点集合 |
| 邻接矩阵 | adjacency matrix | 标准术语 |
| 生理先验图 | physiological prior graph | 由生理知识初始化 |
| 生理先验邻接矩阵 | physiological prior adjacency matrix | 记作 A_prior |
| 先验掩码 | prior mask | 图学习约束 |
| 先验掩码矩阵 | prior mask matrix | 记作 M |
| 先验连接强度 | prior connection strength | A_prior 的元素含义 |
| 先验边 | prior edge | 生理知识支持的边 |
| 可学习先验权重 | learnable prior weight | 记作 lambda |
| 自适应图结构 | adaptive graph structure | 先验约束下学习 |
| 自适应图学习 | adaptive graph learning | 图结构学习范式 |
| 自适应残差图学习 | adaptive residual graph learning | 本文具体机制 |
| 可学习残差邻接矩阵 | learnable residual adjacency matrix | 记作 Delta A |
| 残差边权 | residual edge weight | 残差邻接组成项 |
| 残差边缩放系数 | residual-edge scaling factor | 记作 gamma |
| 边重要性学习 | edge-importance learning | 本文具体机制 |
| 边重要性权重 | edge-importance weight | 可学习边参数 |
| 邻接边偏置 | adjacency-derived edge bias | 注意力相关性组成项 |
| 边偏置缩放参数 | edge-bias scaling parameter | 每个注意力头的参数 |
| 图结构学习 | graph structure learning | 通用任务 |
| 数据驱动修正 | data-driven refinement | 对先验图的有限调整 |
| 图拉普拉斯矩阵 | graph Laplacian | GCN 相关概念 |
| 消息传递 | message passing | GNN 基本机制 |
| 邻域信息聚合 | neighborhood aggregation | GNN 基本机制 |
| 加权聚合 | weighted aggregation | 注意力加权汇聚 |
| 节点特征矩阵 | node feature matrix | 记作 P0 |
| 节点表示 | node representation | 编码后的节点特征 |
| 跨节点依赖关系 | inter-node dependencies | 生理节点间关系 |
| 结构化依赖关系 | structured dependencies | 全文统一用复数泛指 |
| 空间依赖关系 | spatial dependencies | 图建模语境 |
| 拓扑结构依赖 | topological dependencies | 图结构语境 |
| 图结构生理特征 | graph-structured physiological features | 融合模块输入 |
| 生理图上下文表示 | physiological context representation | 图编码输出；由生理上下文图产生 |
| 图结构增强的生理上下文表示 | graph-enhanced physiological context representation | 记作 P_g |
| 固定先验图 | fixed prior graph | 对比概念 |
| 数据驱动自适应图 | data-driven adaptive graph | 对比概念 |
| 伪相关关系 | spurious correlations | 有限样本风险 |
| 医学可解释性 | medical interpretability | 图先验目标 |

## D. Temporal Modeling, Attention, and Fusion

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 时间序列 | time series | 名词；作定语不加连字符 |
| 多变量时间序列 | multivariate time series | 标准术语 |
| 多模态时间序列 | multimodal time series | 标准术语 |
| 时间序列预测 | time-series forecasting | 任务类别 |
| 时间依赖 | temporal dependency | 全文统一用 temporal |
| 长程依赖 | long-range dependency | 远距离时间关系 |
| 全局依赖 | global dependency | Transformer 语境 |
| 序列依赖 | sequential dependency | RNN 语境 |
| 时空依赖 | spatiotemporal dependency | 统一写 spatiotemporal |
| 时间动态 | temporal dynamics | 血糖或生理序列动态 |
| 血糖时间动态特征 | temporal glucose-dynamics features | BC-AFM 输入 |
| 全局时间上下文 | global temporal context | 历史窗口全局信息 |
| 局部时间模式 | local temporal pattern | 卷积分支提取对象 |
| 短期局部波动 | short-term local fluctuation | 短尺度动态 |
| 长期趋势 | long-term trend | 长尺度动态 |
| 时间滞后相关性 | lagged correlation | 生理信号与血糖关系 |
| 非平稳波动 | nonstationary fluctuation | 血糖快速变化 |
| 多尺度时间建模 | multiscale temporal modeling | 方法概念 |
| 多尺度联合建模 | joint multiscale modeling | 多时间尺度联合表示 |
| 特征嵌入 | feature embedding | 输入映射 |
| 输入嵌入函数 | input embedding function | 线性层、GELU 与层归一化组成 |
| 局部时间卷积 | local temporal convolution | 短期分支 |
| 空洞时间卷积 | dilated temporal convolution | 长程分支 |
| 短期局部时间卷积分支 | short-range local temporal-convolution branch | MS-TE 分支 |
| 长程空洞时间卷积分支 | long-range dilated temporal-convolution branch | MS-TE 分支 |
| Transformer 全局自注意力分支 | Transformer-based global self-attention branch | MS-TE 分支 |
| 感受野 | receptive field | 卷积术语 |
| 扩展感受野 | enlarged receptive field | 空洞卷积作用 |
| 位置编码 | positional encoding | Transformer 术语 |
| 正弦位置编码 | sinusoidal positional encoding | 标准位置编码 |
| 多分辨率位置编码 | multi-resolution positional encoding | 本文术语 |
| 多周期位置编码 | multi-period positional encoding | 周期 8、24、72 |
| 可学习位置嵌入 | learnable positional embedding | 本文输入表示组成项 |
| 分支门控机制 | branch-gating mechanism | 三个时间分支自适应加权 |
| 多尺度融合表示 | multiscale fused representation | MS-TE 中间表示 |
| 残差时间细化 | residual temporal refinement | MS-TE 输出细化 |
| 特征复用门控权重 | feature-reuse gate | 记作 rho |
| 注意力机制 | attention mechanism | 通用概念 |
| 自注意力 | self-attention | 单序列内部注意力 |
| 缩放点积注意力 | scaled dot-product attention | Q/K/V 注意力形式 |
| 多头注意力 | multi-head attention (MHA) | 保留连字符 |
| 交叉注意力 | cross-attention | 保留连字符 |
| 查询 | query (Q) | 注意力角色 |
| 键 | key (K) | 注意力角色 |
| 值 | value (V) | 注意力角色 |
| 注意力头 | attention head | 多头注意力子空间 |
| 注意力相关性 | attention logit | Softmax 前的相关性分数 |
| 注意力权重 | attention weight | Softmax 后的归一化权重 |
| 注意力向量 | attention vector | GAT 参数 |
| 特征相容性 | feature compatibility | 节点对相关性组成项 |
| 线性投影 | linear projection | Q/K/V 或跨模态映射 |
| 统一隐空间投影 | projection into a shared latent space | BC-AFM 输入对齐 |
| 表示对齐 | representation alignment | 跨模态对齐 |
| 时间轴对齐 | temporal alignment | 生理上下文对齐到 CGM 时间轴 |
| 跨模态交互 | cross-modal interaction | 模态间信息交换 |
| 跨模态融合 | cross-modal fusion | 模态表示融合 |
| 条件依赖关系 | conditional dependencies | 两类模态间关系 |
| 条件检索 | conditional retrieval | Query 从另一模态检索信息 |
| 反向调制 | reverse modulation | 生理到血糖方向作用 |
| 血糖到生理方向 | glucose-to-physiology direction | BC-AFM 第一方向 |
| 生理到血糖方向 | physiology-to-glucose direction | BC-AFM 第二方向 |
| 特征拼接 | feature concatenation | 数学符号为 double vertical bar |
| 浅层融合 | shallow fusion | 对比方法 |
| 静态加权 | static weighting | 对比方法 |
| 模态表示 | modality representation | 融合分支 |
| 最小模态权重 | minimum modality weight | 记作 eta |
| 模态塌陷 | modality collapse | 融合风险 |
| 时空融合表征 | spatiotemporal fused representation | 输出头输入 |
| 递归式多步预测 | recursive multi-step prediction | 对比预测策略 |
| 自回归误差累积 | autoregressive error accumulation | 递归预测风险 |

## E. Neural Networks, Operations, and Mathematical Terms

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 深度学习 | deep learning | 标准术语 |
| 机器学习 | machine learning | 标准术语 |
| 监督学习 | supervised learning | 任务形式 |
| 多模态学习 | multimodal learning | 方法类别 |
| 端到端训练 | end-to-end training | 联合优化方式 |
| 端到端框架 | end-to-end framework | 模型整体 |
| 数据驱动方法 | data-driven method | 通用表述 |
| 循环神经网络 | recurrent neural network (RNN) | 标准术语 |
| 长短期记忆网络 | long short-term memory network (LSTM) | 标准术语 |
| 双向长短期记忆网络 | bidirectional long short-term memory network (BiLSTM) | 标准术语 |
| 堆叠长短期记忆网络 | stacked long short-term memory network | 标准术语 |
| 门控循环单元 | gated recurrent unit (GRU) | 标准术语 |
| 扩张循环神经网络 | dilated recurrent neural network (DRNN) | 相关工作模型 |
| 卷积神经网络 | convolutional neural network (CNN) | 标准术语 |
| 一维卷积神经网络 | one-dimensional convolutional neural network (1D-CNN) | 基线模型 |
| 时间卷积网络 | temporal convolutional network (TCN) | 标准术语 |
| 卷积循环神经网络 | convolutional recurrent neural network (CRNN) | 基线模型 |
| 证据循环神经网络 | evidential recurrent neural network (E3NN) | 相关工作模型 |
| Transformer 编码器 | Transformer encoder | Transformer 大写 |
| 前馈网络 | feed-forward network (FFN) | 保留连字符 |
| 多层感知机 | multilayer perceptron (MLP) | 标准术语 |
| 全连接层 | fully connected layer | 图中用语 |
| 线性层 | linear layer | PyTorch 层 |
| 输入投影 | input projection | 输出头输入映射 |
| 输出网络 | output network | 每个视界独立设置 |
| 恒等映射 | identity mapping | 维度一致时的退化情形 |
| 层归一化 | layer normalization | 写作 LayerNorm 仅限代码名 |
| 随机失活 | dropout | 训练正则化 |
| 残差连接 | residual connection | 网络结构 |
| GELU 激活函数 | Gaussian error linear unit (GELU) | 激活函数 |
| LeakyReLU 激活函数 | leaky rectified linear unit (LeakyReLU) | 激活函数 |
| Sigmoid 函数 | sigmoid function | 小写 sigmoid 用于正文 |
| Softmax 归一化 | softmax normalization | 小写 softmax 用于正文 |
| Softplus 函数 | softplus function | 小写 softplus 用于正文 |
| 特征空间 | feature space | 表示空间 |
| 隐空间 | latent space | 统一度量空间 |
| 隐藏表示 | hidden representation | 编码输出 |
| 隐藏特征维度 | hidden dimension | 记作 d |
| 批量大小 | batch size | 记作 B |
| 时间步 | time step | 序列索引 |
| 历史时间步长度 | number of historical time steps | 记作 T |
| 输入特征维度 | input feature dimension | 记作 F |
| 可学习参数 | learnable parameter | 模型参数 |
| 投影矩阵 | projection matrix | 线性投影参数 |
| 非线性映射函数 | nonlinear mapping function | 预测函数 |
| 非线性激活函数 | nonlinear activation function | 通用概念 |
| 逐元素乘法 | element-wise multiplication | 数学运算 |
| 加权求和 | weighted sum | 融合与聚合运算 |
| 转置 | transpose | 矩阵运算 |
| 缩放因子 | scaling factor | 注意力计算 |
| 非负对称矩阵 | nonnegative symmetric matrix | A_prior 的性质 |
| 数值稳定项 | numerical-stability term | 记作 epsilon |
| 全局均值 | global mean | 输出头长期上下文 |
| 线性趋势 | linear trend | 输出头长期上下文 |
| 平均池化 | average pooling | 聚合操作 |
| 最大池化 | max pooling | 聚合操作 |
| 时间维度加权池化 | temporal attentive pooling | 生理节点构建 |

## F. Data, Datasets, and Preprocessing

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| BIG IDEAs Lab 血糖变异性与可穿戴设备数据集 v1.1.3 | BIG IDEAs Lab Glycemic Variability and Wearable Device Data v1.1.3 | PhysioNet 官方名称 |
| PhysioNet 公共医学数据库 | PhysioNet repository | 数据托管平台 |
| BIG IDEAs 实验室 | BIG IDEAs Laboratory | 数据发布机构 |
| Dexcom G6 连续血糖监测设备 | Dexcom G6 continuous glucose monitor | CGM 设备 |
| Empatica E4 可穿戴设备 | Empatica E4 wristband | 生理信号设备 |
| OhioT1DM 数据集 | OhioT1DM dataset | 基线研究数据集 |
| ABC4D 数据集 | ABC4D dataset | 相关工作数据集 |
| ARISES 数据集 | ARISES dataset | 相关工作数据集 |
| 原始多模态数据 | raw multimodal data | 预处理输入 |
| 数据预处理 | data preprocessing | 实验流程 |
| 时间戳 | timestamp | 数据字段 |
| 时间同步 | temporal synchronization | 多设备时钟同步 |
| 时间对齐 | temporal alignment | 映射到统一时间轴 |
| 时间分辨率 | temporal resolution | 本文为 5 min |
| 采样频率 | sampling frequency | 单位 Hz |
| 采样间隔 | sampling interval | 记作 Delta t |
| 重采样 | resampling | 映射到 5 min 网格 |
| 重采样聚合 | resampling and aggregation | CGM 与生理信号处理 |
| 滑动窗口 | sliding window | 样本构建方式 |
| 滑动窗口构建 | sliding-window construction | 预处理步骤 |
| 历史观测窗口 | historical observation window | 输入窗口 |
| 历史窗口长度 | historical window length | 本文为 24 个时间步 |
| 预测窗口 | prediction window | 未来总体区间 |
| 预测视界 | prediction horizon (PH) | 指定未来时间点或范围 |
| 多预测视界 | multiple prediction horizons | 15、30、45、60 min |
| 前瞻窗口 | prediction horizon (PH) | 与本文预测任务统一 |
| 缺失值 | missing value | 数据质量问题 |
| 缺失值填补 | missing-value imputation | 通用处理 |
| 短缺失填充 | short-gap imputation | 短时缺失策略 |
| 长缺失剔除 | long-gap exclusion | 长时缺失策略 |
| 信号掉线 | signal dropout | 设备数据中断 |
| 测量噪声 | measurement noise | CGM 误差来源 |
| Z-score 标准化 | z-score normalization | 小写 z |
| 标准化统计量 | normalization statistics | 训练集均值和标准差 |
| 无量纲化处理 | feature scaling | 消除量纲差异 |
| 数据划分 | data partitioning | 训练、验证、测试划分 |
| 训练集 | training set | 实验划分 |
| 验证集 | validation set | 模型选择与早停 |
| 测试集 | test set | 最终评估 |
| 信息泄露 | information leakage | 受试者与预处理层面 |
| 受试者级划分 | subject-level partitioning | LOSO-CV 原则 |
| 未见受试者 | unseen subject | 测试对象 |
| 跨受试者泛化 | cross-subject generalization | 核心评估目标 |
| 留一受试者交叉验证 | leave-one-subject-out cross-validation (LOSO-CV) | 保留完整缩写 |
| 真实世界数据集 | real-world dataset | 数据属性 |
| 多模态数据流 | multimodal data stream | 连续采集数据 |

## G. Training, Baselines, and Evaluation

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 实现细节 | implementation details | 实验小节标题 |
| 模型训练 | model training | 标准术语 |
| 模型推理 | model inference | 标准术语 |
| 联合优化 | joint optimization | 端到端训练 |
| AdamW 优化器 | AdamW optimizer | 优化器名称 |
| 初始学习率 | initial learning rate | 训练超参数 |
| 权重衰减系数 | weight-decay coefficient | 训练超参数 |
| 训练轮数 | number of training epochs | epoch 首次出现可解释 |
| 线性预热 | linear warm-up | 保留连字符 |
| 余弦退火学习率调度器 | cosine-annealing learning-rate scheduler | 训练调度 |
| 梯度裁剪 | gradient clipping | 训练稳定策略 |
| 裁剪阈值 | clipping threshold | 梯度裁剪参数 |
| 早停 | early stopping | 训练策略 |
| 耐心轮数 | patience | 早停超参数 |
| 模型收敛 | model convergence | 训练状态 |
| 过拟合 | overfitting | 泛化风险 |
| 超参数 | hyperparameter | 模型或训练配置 |
| 超参数选择 | hyperparameter selection | 使用验证集 |
| 随机种子 | random seed | 重复实验设置 |
| 多次独立运行 | multiple independent runs | 稳定性评估 |
| 均值 ± 标准差 | mean ± standard deviation | 结果报告格式 |
| 基线模型 | baseline model | 对比对象 |
| 传统深度学习基线 | conventional deep learning baseline | LSTM、1D-CNN |
| 长时序预测模型 | long-term time-series forecasting model | Informer 等类别 |
| 时空联合建模基线 | spatiotemporal modeling baseline | Graph WaveNet、DCRNN |
| 混合架构基线 | hybrid-architecture baseline | CRNN |
| 消融实验 | ablation study | 标准论文用语 |
| 模块消融 | module ablation | 删除单个模块 |
| 复合损失函数 | composite loss function | 方法部分计划项 |
| 预测性能 | predictive performance | 结果总称 |
| 预测精度 | predictive accuracy | 数值误差语境 |
| 预测稳定性 | prediction stability | 跨视界或跨受试者 |
| 泛化能力 | generalizability | 模型在未见对象上的表现 |
| 鲁棒性 | robustness | 噪声和掉线语境 |
| 计算复杂度 | computational complexity | 模型效率 |
| 统计精度评价 | statistical accuracy evaluation | 评价体系第一层 |
| 临床风险评估 | clinical risk assessment | 评价体系第二层 |
| 回归评价指标 | regression metric | MAE 等 |
| 平均绝对误差 | mean absolute error (MAE) | 单位 mg/dL |
| 均方根误差 | root mean square error (RMSE) | 单位 mg/dL |
| 决定系数 | coefficient of determination (R²) | 保留上标 2 |
| 平均绝对百分比误差 | mean absolute percentage error (MAPE) | 百分比指标 |
| 平均绝对相对差 | mean absolute relative difference (MARD) | 与 MAPE 不混用，须按代码确认 |
| 克拉克误差网格分析 | Clarke error grid analysis (CEGA) | 临床风险评价 |
| 克拉克误差网格 | Clarke error grid (CEG) | 图或网格本体 |
| A 区 | Zone A | 高度准确 |
| B 区 | Zone B | 不导致错误治疗 |
| C 区 | Zone C | 可能导致不必要干预 |
| D 区 | Zone D | 未识别危险状态 |
| E 区 | Zone E | 可能导致严重错误治疗 |
| 临床可接受 | clinically acceptable | 指 Zone A 与 Zone B |
| 真实值 | ground-truth value | 评价与图示 |
| 预测值 | predicted value | 评价与图示 |
| 预测误差 | prediction error | 预测与真实值偏差 |
| 临床治疗决策 | clinical treatment decision | CEG 语境 |

## H. Named Models and Methods Cited in the Paper

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 自回归积分滑动平均模型 | autoregressive integrated moving average (ARIMA) | 标准全称 |
| 支持向量回归 | support vector regression (SVR) | 标准术语 |
| 随机森林 | random forest (RF) | 标准术语 |
| 卡尔曼平滑 | Kalman smoothing | 方法名 |
| 格兰杰因果分析 | Granger causality analysis | 方法名 |
| 混合效应逻辑回归 | mixed-effects logistic regression | 统计模型 |
| Informer | Informer | 模型专名 |
| 概率稀疏注意力 | ProbSparse self-attention | Informer 机制 |
| Autoformer | Autoformer | 模型专名 |
| 序列分解 | series decomposition | Autoformer 机制 |
| 自相关机制 | autocorrelation mechanism | Autoformer 机制 |
| PatchTST | PatchTST | 模型专名 |
| 时间序列分块 | time-series patching | PatchTST 机制 |
| 图小波网络 | Graph WaveNet | 模型专名，不直译用于正文 |
| 自适应图卷积 | adaptive graph convolution | Graph WaveNet 机制 |
| 扩散图卷积 | diffusion graph convolution | DCRNN 机制 |
| 频谱域消息传递 | spectral-domain message passing | GNN 相关工作 |
| 空间域消息传递 | spatial-domain message passing | GNN 相关工作 |

## I. Figure and Workflow Labels

| 中文术语 | 英文术语 | 说明（可选） |
|---|---|---|
| 数据输入 | data input | 流程图标签 |
| 数据清洗 | data cleaning | 预处理流程 |
| 特征提取 | feature extraction | 通用流程 |
| CGM 时序特征提取 | CGM temporal feature extraction | 图中分支标签 |
| 生理信号特征提取 | physiological signal feature extraction | 图中分支标签 |
| 分层时序划分 | hierarchical temporal partitioning | 图中预处理标签 |
| 多尺度滑动窗口 | multiscale sliding window | 图中样本构建标签 |
| 时间编码 | temporal encoding | 图中编码步骤 |
| 窗口统计 | window statistics | 图中特征类型 |
| 滞后特征 | lag feature | 动态增强特征 |
| 差分特征 | differenced feature | 动态增强特征 |
| 频域特征 | frequency-domain feature | 图中特征类型 |
| 动态特征 | dynamic feature | 图中特征类型 |
| 时间特征序列 | temporal feature sequence | 图中记作 X_T |
| 空间特征序列 | spatial feature sequence | 图中记作 X_S |
| 短时间尺度 | short timescale | 图中尺度标签 |
| 长时间尺度 | long timescale | 图中尺度标签 |
| 生理信号空间编码器 | physiological-signal spatial encoder | 图中分支名称 |
| 图卷积层 | graph convolutional layer | 图中网络层 |
| GNN 层 | GNN layer | 图中网络层 |
| 全局节点池化 | global node pooling | 图中图级聚合 |
| 时间维池化 | temporal pooling | 图中时间聚合 |
| 时间到空间交叉注意力 | temporal-to-spatial cross-attention | 图中 Time-to-Space 分支 |
| 空间到时间交叉注意力 | spatial-to-temporal cross-attention | 图中 Space-to-Time 分支 |
| 时空联合注意力机制 | joint spatiotemporal attention mechanism | 图中融合机制 |
| 时空融合模块 | spatiotemporal fusion module | 图中模块标签 |
| 时间编码分支 | temporal encoding branch | 图中分支标签 |
| 空间编码分支 | spatial encoding branch | 图中分支标签 |
| 融合多层感知机 | fusion MLP | 图中融合层 |
| 尺度融合 | scale fusion | 图中融合步骤 |
| 先验图结构 | prior graph structure | 图中邻接关系 |
| 图邻接矩阵 | graph adjacency matrix | 图中输入 |
| 生理信号空间输入 | spatial physiological-signal input | 图中记作 X_S |
| CGM 时间序列输入 | CGM time-series input | 图中记作 X_T |
| 时间特征 | temporal features | 图中记作 H_T |
| 空间特征 | spatial features | 图中记作 H_S |
| 预测头 | prediction head | 图中输出模块 |
| 血糖预测输出 | blood glucose prediction output | 图中最终输出 |
| 预测血糖 | predicted blood glucose | 图中曲线或数值 |
