# LeNet-5 动图生成指南

## 🎬 动图功能概述

本项目现在支持生成多个数字7的多层输出动图，可以动态展示不同样本在LeNet-5网络各层的激活情况。

## 📁 生成的文件

### 静态图片
- `multiple_7s_simple_test.png` - 多个7样本的静态可视化
- `detailed_conv_channels_simple.png` - 详细卷积通道的静态可视化

### 动图文件
- `multiple_7s_animation.gif` - 多个7样本的多层输出动图 (3.06 MB)
- `detailed_channels_7_animation.gif` - 详细卷积通道动图 (3.09 MB)

## 🚀 使用方法

### 1. 运行简化测试（包含动图生成）
```bash
python simple_test.py
```

### 2. 运行专门的动图生成器
```bash
python generate_animation.py
```

### 3. 运行增强版演示
```bash
python enhanced_demo.py
```

## 🎯 动图内容说明

### 多个7样本动图 (`multiple_7s_animation.gif`)
- **展示内容**: 6个不同的数字7样本
- **布局**: 4行6列网格
  - 第1行: 原始图像
  - 第2行: Conv1激活 (6通道平均)
  - 第3行: Conv2激活 (16通道平均)
  - 第4行: 激活统计信息
- **动画效果**: 循环展示不同样本，每个样本停留1秒
- **文件大小**: 约3MB

### 详细卷积通道动图 (`detailed_channels_7_animation.gif`)
- **展示内容**: 4个不同的数字7样本
- **布局**: 3行8列网格
  - 第1行: 原始图像 + Conv1的6个通道
  - 第2行: Conv2的前8个通道
  - 第3行: 激活统计信息
- **动画效果**: 循环展示不同样本，每个样本停留1.5秒
- **文件大小**: 约3MB

## 🔍 观察要点

### 1. 激活模式差异
- 不同数字7样本在Conv1和Conv2层表现出不同的激活模式
- 可以观察到网络如何提取不同的特征

### 2. 特征层次
- Conv1层: 提取边缘、线条等低级特征
- Conv2层: 提取更复杂的形状和模式

### 3. 激活强度
- 热力图颜色越亮表示激活越强
- 不同样本的激活强度分布不同

## 🛠️ 自定义选项

### 修改目标数字
在脚本中修改 `target_digit` 参数：
```python
# 生成数字3的动图
anim = generate_multiple_7s_animation(num_samples=6, target_digit=3)
```

### 修改样本数量
```python
# 生成8个样本的动图
anim = generate_multiple_7s_animation(num_samples=8, target_digit=7)
```

### 修改动画速度
```python
# 更快的动画 (2fps)
anim = generate_multiple_7s_animation(num_samples=6, target_digit=7, fps=2)
```

## 📊 技术细节

### 网络结构
- **Conv1**: 6个5×5卷积核，输出6个28×28特征图
- **Conv2**: 16个5×5卷积核，输出16个10×10特征图
- **激活函数**: ReLU
- **池化**: 2×2最大池化

### 数据处理
- **输入**: 28×28灰度图像
- **标准化**: 使用MNIST标准参数
- **设备**: 自动检测CPU/GPU

### 可视化技术
- **热力图**: 使用'hot'颜色映射
- **动画**: matplotlib.animation.FuncAnimation
- **保存格式**: GIF (使用pillow writer)

## 🎓 教学价值

### 1. 理解CNN工作原理
- 观察卷积层如何提取特征
- 理解特征从低级到高级的层次结构

### 2. 分析网络行为
- 比较不同样本的激活模式
- 理解网络的泛化能力

### 3. 可视化技术学习
- 学习如何可视化神经网络内部状态
- 掌握动图生成技术

## 🔧 故障排除

### 常见问题

1. **动图生成失败**
   - 检查matplotlib版本
   - 确保pillow库已安装
   - 检查磁盘空间

2. **文件过大**
   - 减少样本数量
   - 降低fps参数
   - 减小图像尺寸

3. **内存不足**
   - 减少同时处理的样本数
   - 使用CPU而不是GPU

### 性能优化

1. **减少文件大小**
   ```python
   # 使用更低的fps
   anim.save(save_path, writer='pillow', fps=0.5)
   ```

2. **提高生成速度**
   ```python
   # 减少样本数量
   anim = generate_multiple_7s_animation(num_samples=4, target_digit=7)
   ```

## 📚 扩展建议

1. **添加更多可视化**
   - 全连接层的激活
   - 梯度可视化
   - 注意力图

2. **支持更多数据集**
   - CIFAR-10
   - 自定义数据集

3. **交互式功能**
   - 用户选择样本
   - 实时参数调整
   - 导出功能

---

**享受探索LeNet-5网络内部工作机制的乐趣！** 🎉
