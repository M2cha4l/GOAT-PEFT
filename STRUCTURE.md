# GOAT-PEFT 库结构说明

## 概述

GOAT-PEFT 是一个基于 LoRA (Low-Rank Adaptation) 的 Mixture-of-Experts (MoE) 框架，用于大规模模型的参数高效微调。该框架通过集成自适应奇异值先验和理论优化对齐，在 MoE 架构中提升 LoRA 的性能。

## 目录结构

```
GOAT-PEFT/
├── goat/                          # 核心代码目录
│   ├── peta/                      # PEFT (Parameter-Efficient Fine-Tuning) 实现
│   │   ├── src/                   # 核心源代码
│   │   │   ├── goat/              # GOAT 模型核心实现
│   │   │   │   ├── __init__.py
│   │   │   │   ├── config.py      # GOAT 配置类
│   │   │   │   ├── layer.py       # GOAT 层实现（专家层、路由层等）
│   │   │   │   └── model.py       # GOAT 模型主类
│   │   │   ├── config.py          # PEFT 基础配置
│   │   │   ├── mapping.py          # 模型映射
│   │   │   ├── peft_model.py      # PEFT 模型封装
│   │   │   ├── trainer.py          # 自定义训练器（支持辅助损失）
│   │   │   └── utils/              # 工具函数
│   │   │       ├── peft_types.py  # PEFT 类型定义
│   │   │       ├── save_and_load.py # 模型保存和加载
│   │   │       └── test_utils.py  # 测试工具
│   │   ├── tasks/                  # 任务相关代码
│   │   │   ├── cv/                 # 计算机视觉任务
│   │   │   │   ├── *.py           # 各种 CV 数据集实现
│   │   │   │   ├── registry.py    # 任务注册表
│   │   │   │   └── templates.py   # 模板定义
│   │   │   ├── data.py            # 数据加载和预处理
│   │   │   ├── datasets_preprocess.py # 数据集预处理
│   │   │   ├── glue_preprocess.py # GLUE 任务预处理
│   │   │   └── llama_dataset_preprocessing.py # LLaMA 数据集预处理
│   │   └── utils/                  # 工具模块
│   │       ├── args.py             # 参数解析
│   │       ├── logging/            # 日志模块
│   │       │   └── rich.py        # Rich 日志格式化
│   │       └── timeit.py          # 时间测量工具
│   ├── config/                     # 配置文件
│   │   └── deepspeed_zero2.json   # DeepSpeed ZeRO-2 配置
│   ├── train_vit.py               # Vision Transformer 训练脚本
│   ├── train_nlg.py               # 自然语言生成任务训练脚本
│   ├── train_nlu.py               # 自然语言理解任务训练脚本
│   ├── eval_nlg.py                # NLG 任务评估脚本
│   └── download_data.py           # 数据下载脚本
├── dataset/                        # 数据集目录
│   ├── ~data/                      # 原始数据下载脚本
│   │   ├── download_dataset.sh    # 数据集下载脚本
│   │   └── process_*.py           # 数据集处理脚本
│   ├── commonsense_170k.json      # 常识推理数据集
│   └── [各种任务数据集]/           # 各任务的数据集文件
│       ├── *.json                 # JSON 格式数据集
│       └── test.json/train.json  # 训练/测试集
├── *.sh                           # 训练脚本
│   ├── vit.sh                     # ViT 训练脚本
│   ├── math.sh                    # 数学推理训练脚本
│   ├── code.sh                    # 代码生成训练脚本
│   ├── chat.sh                    # 聊天任务训练脚本
│   ├── glue.sh                    # GLUE 任务训练脚本
│   └── cr.sh                      # 常识推理训练脚本
├── requirements.txt               # Python 依赖
├── README.md                      # 项目说明
└── LICENSE                        # 许可证

```

## 核心模块详解

### 1. GOAT 模型核心 (`goat/peta/src/goat/`)

#### `config.py`
- **GOATConfig**: GOAT 方法的配置类，继承自 `LoraConfig`
- 关键参数：
  - `num_experts`: MoE 中专家数量（默认 8）
  - `top_k`: Top-K 路由中的 K 值（默认 2）
  - `init_type`: 初始化类型（如 "goat", "pissa", "milora" 等）
  - `init_cof`: 初始化系数（默认 1.0）

#### `layer.py`
包含 GOAT 的核心层实现：

- **TopKGOATLayer**: Top-K 路由的 MoE 层
  - 实现专家选择和加权聚合
  - 计算负载均衡损失（layer_loss）
  - 支持专家相似度损失计算

- **GOATExpert**: 单个 LoRA 专家
  - 包含 `lora_A`, `lora_B`, `lora_dropout` 和 `scaling` 参数
  - 实现 LoRA 的前向传播

- **GOATLayer**: GOAT 层基类
  - 管理多个专家和路由门控网络
  - 支持多种初始化方法（SVD 分解、高斯初始化等）
  - 实现自适应奇异值初始化（GOAT 核心特性）

- **LinearGOATLayer**: 线性层的 GOAT 实现
  - 包装 `torch.nn.Linear` 层
  - 集成 MoE 机制到线性层

#### `model.py`
- **GOATModel**: GOAT 模型主类，继承自 `LoraModel`
  - 负责将目标层替换为 GOAT 层
  - 管理适配器的创建和替换
  - 提供辅助损失计算接口

### 2. 训练器 (`goat/peta/src/trainer.py`)

- **PeftTrainer**: 自定义训练器，继承自 Transformers 的 `Trainer`
  - 支持 MoE 的辅助损失（负载均衡损失）
  - 可配置辅助损失系数 `aux_loss_coeff`
  - 自动处理模型保存和加载

### 3. 任务模块 (`goat/peta/tasks/`)

#### CV 任务 (`cv/`)
包含多个计算机视觉数据集的实现：
- **分类任务**: CIFAR-10/100, ImageNet, MNIST, FashionMNIST 等
- **细粒度分类**: Cars, Flowers102, Food101 等
- **场景分类**: SUN397, RESISC45, EuroSAT 等
- **特殊任务**: DTD (纹理分类), GTSRB (交通标志), FER2013 (表情识别) 等

每个任务文件包含：
- 数据集加载逻辑
- 数据预处理流程
- 评估指标计算

#### 数据预处理
- `data.py`: 通用数据加载和预处理
- `glue_preprocess.py`: GLUE 基准测试预处理
- `llama_dataset_preprocessing.py`: LLaMA 模型数据集预处理
- `datasets_preprocess.py`: 其他数据集预处理

### 4. 训练脚本

#### `train_vit.py`
- Vision Transformer 模型训练
- 支持多种 CV 数据集
- 使用 CLIP ViT 作为基础模型

#### `train_nlg.py`
- 自然语言生成任务训练
- 支持数学推理、代码生成等任务
- 使用 LLaMA 等因果语言模型

#### `train_nlu.py`
- 自然语言理解任务训练
- 支持 GLUE 基准测试
- 分类和序列标注任务

### 5. 配置文件

#### `config/deepspeed_zero2.json`
- DeepSpeed ZeRO-2 优化配置
- 用于分布式训练的内存优化

## 关键特性

### 1. 自适应奇异值初始化
- 将预训练权重的 SVD 分解为多个片段
- 每个 MoE 专家使用不同的奇异值段初始化
- 通过路由动态选择专家

### 2. 理论优化对齐
- 推导闭式缩放因子和残差校正
- 对齐 LoRA 的低秩适应梯度与全量微调轨迹

### 3. Mixture-of-Experts 架构
- Top-K 路由机制
- 负载均衡损失
- 专家多样性保证

## 数据流

1. **数据加载**: `tasks/data.py` → 加载和预处理数据集
2. **模型初始化**: `train_*.py` → 加载基础模型，应用 GOAT 配置
3. **层替换**: `model.py` → 将目标层替换为 GOAT 层
4. **训练循环**: `trainer.py` → 执行训练，计算主损失和辅助损失
5. **评估**: 在验证集上评估性能

## 使用流程

1. **环境准备**: 安装依赖 (`requirements.txt`)
2. **数据准备**: 下载数据集到 `dataset/` 目录
3. **配置训练**: 修改对应的 `.sh` 脚本（如 `vit.sh`, `math.sh`）
4. **执行训练**: 运行训练脚本
5. **结果查看**: 结果保存在指定的输出目录，可通过 WandB 查看

## 支持的初始化方法

- `goat`: GOAT 自适应奇异值初始化（默认推荐）
- `pissa`: PiSSA 初始化方法
- `milora`: MiLoRA 初始化方法
- `lora`: 标准 LoRA 初始化
- `hydralora`: HydraLoRA 初始化
- `mole`: MoLE 初始化

## 支持的模型类型

- **视觉模型**: CLIP ViT, Vision Transformer
- **语言模型**: LLaMA, GPT 系列
- **多模态模型**: 支持 Transformers 库中的大部分模型

## 扩展性

- 易于添加新的任务：在 `tasks/cv/` 或 `tasks/` 中添加新任务文件
- 支持自定义初始化方法：在 `layer.py` 的 `update_layer` 方法中添加
- 灵活的配置系统：通过 `GOATConfig` 和命令行参数配置

## 注意事项

1. 需要设置 Kaggle API 密钥用于下载某些数据集
2. 训练脚本需要配置 `BASE_DIR` 和 `OUT_DIR` 环境变量
3. 某些任务需要特定的 GPU 内存配置
4. 建议使用 DeepSpeed 进行大规模模型训练

