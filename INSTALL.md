# LeNet-5 项目安装指南

## 🚀 快速开始

### 1. 环境要求
- Python 3.7 或更高版本
- pip 包管理器
- 可选：CUDA (用于GPU加速)

### 2. 安装步骤

#### 方法一：使用pip安装依赖
```bash
# 安装项目依赖
pip install -r requirements.txt

# 或者使用pip3
pip3 install -r requirements.txt
```

#### 方法二：使用conda创建虚拟环境（推荐）
```bash
# 创建conda环境
conda create -n lenet python=3.8

# 激活环境
conda activate lenet

# 安装PyTorch (CPU版本)
conda install pytorch torchvision -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

#### 方法三：使用venv创建虚拟环境
```bash
# 创建虚拟环境
python3 -m venv lenet_env

# 激活环境 (macOS/Linux)
source lenet_env/bin/activate

# 激活环境 (Windows)
# lenet_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 验证安装
```bash
# 运行测试脚本
python3 test_setup.py

# 或者运行演示
python3 demo.py
```

## 🔧 常见问题解决

### 问题1：ModuleNotFoundError: No module named 'torch'
**解决方案：**
```bash
# 安装PyTorch
pip install torch torchvision

# 或者指定版本
pip install torch==2.0.0 torchvision==0.15.0
```

### 问题2：CUDA相关错误
**解决方案：**
- 如果不需要GPU，安装CPU版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- 如果需要GPU，确保安装了正确版本的CUDA：
```bash
# 查看CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题3：权限错误
**解决方案：**
```bash
# 使用用户安装
pip install --user -r requirements.txt

# 或者使用sudo (不推荐)
sudo pip install -r requirements.txt
```

### 问题4：网络连接问题
**解决方案：**
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或者使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 📦 依赖包说明

### 核心依赖
- **torch**: PyTorch深度学习框架
- **torchvision**: 计算机视觉工具包
- **numpy**: 数值计算库
- **matplotlib**: 绘图库
- **seaborn**: 统计绘图库

### 辅助依赖
- **tqdm**: 进度条显示
- **Pillow**: 图像处理
- **scikit-learn**: 机器学习工具
- **tensorboard**: 训练过程可视化

## 🎯 安装后测试

安装完成后，运行以下命令测试：

```bash
# 1. 测试基本功能
python3 test_setup.py

# 2. 运行演示
python3 demo.py

# 3. 启动交互式菜单
python3 run.py
```

如果所有测试都通过，恭喜你！项目已经成功安装。

## 🚀 开始使用

安装完成后，你可以：

1. **运行演示**：`python3 demo.py`
2. **开始训练**：`python3 train.py`
3. **评估模型**：`python3 evaluate.py`
4. **单张图片推理**：`python3 inference.py --image your_image.jpg`
5. **使用交互式菜单**：`python3 run.py`

## 📚 更多帮助

- 查看 [README.md](README.md) 了解项目详情
- 查看 [config.py](config.py) 了解配置选项
- 查看各个脚本的注释了解具体功能

---

**祝你学习愉快！** 🎉
