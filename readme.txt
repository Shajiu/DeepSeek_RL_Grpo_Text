llama2_project/
│
├── config/                      # 配置文件夹
│   └── config.py                # 存放所有配置的参数，比如模型路径、数据路径等
│
├── data/                        # 数据存储文件夹
│   ├── download_base_data.py    # 下载数据的脚本(可执行/可不执行)
│   ├── train_data/              # 训练数据
│   ├── val_data/                # 验证数据
│   └── test_data/               # 测试数据
│
├── dataset/                     # 数据处理模块
│   ├── __init__.py              # 初始化脚本，可以将数据处理相关的函数或类导入
│   ├── dataset_loader.py        # 负责加载数据集的脚本
│   └── preprocess.py            # 数据预处理脚本，比如数据清洗、格式化等
│
├── models/                      # 模型定义文件夹
│   ├── __init__.py              # 初始化脚本，方便导入模型定义
│   ├── grpo.py                  # 定义grpo算法逻辑函数
│   ├── download_base_models.py  # 下载基座模型的脚本(可执行/可不执行)
│   ├── reward.py                # 定义奖励函数
│   └── model.py                 # 定义Llama2模型架构的脚本
│
├── scripts/                     # 运行脚本文件夹
│   ├── train.py                 # 模型训练脚本
│   ├── evaluate.py              # 模型评估脚本
│   └── predict.py               # 模型预测脚本
│
├── utils/                       # 工具类文件夹
│   ├── __init__.py              # 初始化脚本
│   ├── metrics.py               # 定义评估指标的脚本
│   └── logger.py                # 日志记录脚本
│
└── requirements.txt             # 项目依赖文件，列出所有必需的库

