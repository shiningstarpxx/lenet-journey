# 大文件说明

## 概述
由于GitHub对单个文件大小有限制（100MB），本项目中的大文件已被移动到 `large_files/` 目录中，该目录已被 `.gitignore` 忽略。

## 大文件内容

### 动画文件 (`large_files/animations/`)
- `multiple_7s_animation.gif` - 多个数字7样本的激活动画
- `detailed_channels_7_animation.gif` - 详细通道激活动画
- `conv_comparison_animation_7.gif` - 卷积层对比动画

### 图片文件 (`large_files/images/`)
- 各种模型架构图
- 激活可视化图片
- 对比分析图表
- 中文字体测试图片

## 如何获取大文件

### 方法1：重新生成
运行相应的脚本可以重新生成这些文件：

```bash
# 生成动画
python generate_animation.py

# 生成架构图
python visualize_model_architecture_v3.py

# 运行完整对比分析
python conv_comparison_analysis.py
```

### 方法2：从其他来源获取
如果需要这些预生成的文件，可以通过以下方式获取：
- 联系项目维护者
- 从其他存储位置下载
- 使用Git LFS（如果配置了的话）

## 文件大小统计
- 总大小：约12MB
- 动画文件：约6MB
- 图片文件：约6MB

## 注意事项
- 这些文件不影响项目的核心功能
- 所有可视化功能都可以通过运行脚本重新生成
- 建议在本地开发时重新生成这些文件以获得最新结果
