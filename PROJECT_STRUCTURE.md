# LeNet Journey 项目结构说明

## 📁 项目目录结构

```
lenet-journey/
├── 📁 核心文件
│   ├── config.py                    # 项目配置文件
│   ├── utils.py                     # 工具函数
│   ├── train.py                     # 基础训练脚本
│   ├── evaluate.py                  # 模型评估脚本
│   ├── inference.py                 # 推理脚本
│   ├── visualization.py             # 可视化工具
│   ├── run.py                       # 主运行脚本
│   ├── quick_start.py               # 快速开始脚本
│   └── requirements.txt             # 依赖包列表
│
├── 📁 models/                       # 模型定义
│   ├── __init__.py
│   ├── lenet.py                     # LeNet-5 模型
│   ├── conv_comparison.py           # 卷积层对比模型
│   └── adaptive_conv_comparison.py  # 自适应卷积模型
│
├── 📁 data/                         # 数据处理
│   ├── __init__.py
│   ├── dataset.py                   # 数据集加载器
│   └── sample_images/               # 示例图片
│
├── 📁 scripts/                      # 脚本集合
│   ├── 📁 comparison/               # 模型对比脚本
│   │   ├── conv_comparison_analysis.py      # 卷积层对比分析
│   │   ├── dual_dataset_comparison.py       # 双数据集对比
│   │   ├── train_comparison.py              # 对比训练
│   │   └── adaptive_train_comparison.py     # 自适应训练
│   │
│   ├── 📁 visualization/            # 可视化脚本
│   │   ├── visualize_comparison.py          # 对比可视化
│   │   ├── visualize_model_architecture.py  # 架构可视化 v1
│   │   ├── visualize_model_architecture_v2.py # 架构可视化 v2
│   │   └── visualize_model_architecture_v3.py # 架构可视化 v3
│   │
│   ├── 📁 demo/                     # 演示脚本
│   │   ├── demo.py                          # 基础演示
│   │   ├── enhanced_demo.py                 # 增强演示
│   │   ├── simple_test.py                   # 简单测试
│   │   └── generate_animation.py            # 动画生成
│   │
│   └── 📁 test/                     # 测试脚本
│       ├── test_setup.py                    # 环境测试
│       ├── test_chinese_display.py          # 中文显示测试
│       ├── test_multiple_7s.py              # 多样本测试
│       └── setup_chinese_font.py            # 中文字体设置
│
├── 📁 docs/                         # 文档
│   ├── 📁 guides/                   # 指南文档
│   │   ├── ANIMATION_GUIDE.md               # 动画功能指南
│   │   ├── ARCHITECTURE_VISUALIZATION_GUIDE.md # 架构可视化指南
│   │   ├── CHINESE_FONT_SOLUTION.md         # 中文字体解决方案
│   │   └── CONV_COMPARISON_GUIDE.md         # 卷积对比指南
│   │
│   └── 📁 reports/                  # 报告文档
│       ├── FINAL_COMPARISON_SUMMARY.md      # 最终对比总结
│       └── PROJECT_SUMMARY.md               # 项目总结
│
├── 📁 results/                      # 结果输出
│   ├── 📁 dual_dataset_comparison/  # 双数据集对比结果
│   ├── 📁 evaluation/               # 评估结果
│   └── *.png, *.gif                 # 各种可视化图片
│
├── 📁 checkpoints/                  # 模型检查点
│   ├── 📁 comparison/               # 对比模型检查点
│   └── 📁 adaptive_comparison/      # 自适应模型检查点
│
├── 📁 logs/                         # 训练日志
│   ├── 📁 comparison_*/             # 对比训练日志
│   └── 📁 adaptive_comparison_*/    # 自适应训练日志
│
├── 📁 large_files/                  # 大文件存储
│   ├── 📁 animations/               # 动画文件
│   └── 📁 images/                   # 图片文件
│
├── 📁 lenet_env/                    # Python虚拟环境
│
├── 📄 README.md                     # 项目主文档
├── 📄 INSTALL.md                    # 安装指南
├── 📄 LARGE_FILES_README.md         # 大文件说明
├── 📄 PROJECT_STRUCTURE.md          # 本文件
└── 📄 .gitignore                    # Git忽略文件
```

## 🎯 主要功能模块

### 1. 核心功能
- **config.py**: 统一配置管理
- **utils.py**: 通用工具函数
- **train.py**: 基础模型训练
- **evaluate.py**: 模型评估
- **inference.py**: 单图推理
- **visualization.py**: 可视化工具

### 2. 模型定义 (models/)
- **lenet.py**: 经典LeNet-5实现
- **conv_comparison.py**: 1/2/3层卷积对比模型
- **adaptive_conv_comparison.py**: 支持不同输入尺寸的自适应模型

### 3. 脚本集合 (scripts/)

#### 对比分析 (comparison/)
- **conv_comparison_analysis.py**: 完整的卷积层对比分析
- **dual_dataset_comparison.py**: MNIST vs CIFAR-10双数据集对比
- **train_comparison.py**: 对比训练器
- **adaptive_train_comparison.py**: 自适应训练器

#### 可视化 (visualization/)
- **visualize_comparison.py**: 对比结果可视化
- **visualize_model_architecture_v3.py**: 模型架构图（推荐使用v3）

#### 演示 (demo/)
- **demo.py**: 基础功能演示
- **enhanced_demo.py**: 增强功能演示
- **simple_test.py**: 简化测试
- **generate_animation.py**: 动画生成

#### 测试 (test/)
- **test_setup.py**: 环境验证
- **test_chinese_display.py**: 中文显示测试
- **setup_chinese_font.py**: 字体配置

### 4. 文档 (docs/)

#### 指南 (guides/)
- **ANIMATION_GUIDE.md**: 动画功能使用指南
- **ARCHITECTURE_VISUALIZATION_GUIDE.md**: 架构可视化指南
- **CHINESE_FONT_SOLUTION.md**: 中文字体问题解决方案
- **CONV_COMPARISON_GUIDE.md**: 卷积层对比使用指南

#### 报告 (reports/)
- **FINAL_COMPARISON_SUMMARY.md**: 最终对比分析报告
- **PROJECT_SUMMARY.md**: 项目功能总结

## 🚀 快速开始

### 1. 基础使用
```bash
# 快速开始
python quick_start.py

# 基础演示
python scripts/demo/demo.py

# 训练模型
python train.py
```

### 2. 对比分析
```bash
# 卷积层对比分析
python scripts/comparison/conv_comparison_analysis.py

# 双数据集对比
python scripts/comparison/dual_dataset_comparison.py
```

### 3. 可视化
```bash
# 模型架构可视化
python scripts/visualization/visualize_model_architecture_v3.py

# 对比结果可视化
python scripts/visualization/visualize_comparison.py
```

### 4. 测试验证
```bash
# 环境测试
python scripts/test/test_setup.py

# 中文显示测试
python scripts/test/test_chinese_display.py
```

## 📊 主要特性

1. **多模型支持**: LeNet-5, 1/2/3层卷积网络
2. **多数据集支持**: MNIST, CIFAR-10
3. **自适应架构**: 支持不同输入尺寸和通道数
4. **丰富可视化**: 训练曲线、激活图、架构图、动画
5. **对比分析**: 详细的性能对比和效率分析
6. **中文支持**: 完整的中文字体显示解决方案

## 🔧 维护说明

- **results/**: 自动生成，包含所有输出结果
- **checkpoints/**: 自动生成，包含训练好的模型
- **logs/**: 自动生成，包含训练日志
- **large_files/**: 手动管理，包含大文件（被git忽略）
- **lenet_env/**: Python虚拟环境，不要手动修改

## 📝 注意事项

1. 首次运行前请先执行 `python scripts/test/test_setup.py` 验证环境
2. 中文显示问题请参考 `docs/guides/CHINESE_FONT_SOLUTION.md`
3. 大文件（动画、图片）存储在 `large_files/` 目录，可通过脚本重新生成
4. 所有脚本都支持相对路径，建议在项目根目录运行
