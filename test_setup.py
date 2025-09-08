"""
LeNet-5 项目设置测试脚本
验证所有依赖和功能是否正常工作
"""

import sys
import importlib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试导入...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'seaborn', 'tqdm', 'PIL', 'sklearn'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ 以下包导入失败: {failed_imports}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依赖包导入成功!")
        return True

def test_pytorch():
    """测试PyTorch功能"""
    print("\n🔍 测试PyTorch...")
    
    try:
        # 测试基本操作
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print(f"  ✅ 张量运算: {z.shape}")
        
        # 测试设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  ✅ 设备: {device}")
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA版本: {torch.version.cuda}")
            print(f"  ✅ GPU数量: {torch.cuda.device_count()}")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch测试失败: {e}")
        return False

def test_project_modules():
    """测试项目模块"""
    print("\n🔍 测试项目模块...")
    
    modules_to_test = [
        'config',
        'models.lenet',
        'data.dataset',
        'utils'
    ]
    
    failed_modules = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ 以下模块导入失败: {failed_modules}")
        return False
    else:
        print("✅ 所有项目模块导入成功!")
        return True

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        from models import LeNet5, LeNet5_CIFAR
        
        # 测试MNIST模型
        model_mnist = LeNet5(input_channels=1, num_classes=10)
        print(f"  ✅ LeNet5 (MNIST): {sum(p.numel() for p in model_mnist.parameters()):,} 参数")
        
        # 测试CIFAR模型
        model_cifar = LeNet5_CIFAR(input_channels=3, num_classes=10)
        print(f"  ✅ LeNet5_CIFAR: {sum(p.numel() for p in model_cifar.parameters()):,} 参数")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model_mnist(dummy_input)
        print(f"  ✅ 前向传播: {dummy_input.shape} -> {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ 模型测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        from data import DatasetLoader
        
        # 测试MNIST数据加载器创建
        loader = DatasetLoader(dataset_name='MNIST', data_dir='./data', batch_size=32)
        print("  ✅ MNIST数据加载器创建成功")
        
        # 测试数据集信息
        info = loader.get_dataset_info()
        print(f"  ✅ 数据集信息: {info['dataset_name']}, {info['num_classes']} 类")
        
        return True
    except Exception as e:
        print(f"  ❌ 数据加载测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n🔍 测试可视化...")
    
    try:
        # 测试matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✅ Matplotlib绘图成功")
        
        # 清理测试文件
        import os
        if os.path.exists('test_plot.png'):
            os.remove('test_plot.png')
        
        return True
    except Exception as e:
        print(f"  ❌ 可视化测试失败: {e}")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n🔍 测试目录结构...")
    
    required_files = [
        'config.py',
        'train.py',
        'evaluate.py',
        'inference.py',
        'demo.py',
        'utils.py',
        'requirements.txt',
        'README.md',
        'models/__init__.py',
        'models/lenet.py',
        'data/__init__.py',
        'data/dataset.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ 以下文件缺失: {missing_files}")
        return False
    else:
        print("✅ 所有必需文件存在!")
        return True

def main():
    """主测试函数"""
    print("🧪 LeNet-5 项目设置测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("PyTorch测试", test_pytorch),
        ("项目模块测试", test_project_modules),
        ("模型创建测试", test_model_creation),
        ("数据加载测试", test_data_loading),
        ("可视化测试", test_visualization),
        ("目录结构测试", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 出现异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! 项目设置正确，可以开始使用。")
        print("\n🚀 接下来你可以:")
        print("  1. 运行 'python run.py' 启动交互式菜单")
        print("  2. 运行 'python demo.py' 查看演示")
        print("  3. 运行 'python train.py' 开始训练")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息并修复。")
        print("\n🔧 常见解决方案:")
        print("  1. 安装依赖: pip install -r requirements.txt")
        print("  2. 检查Python版本: python --version (需要3.7+)")
        print("  3. 检查PyTorch安装: pip install torch torchvision")

if __name__ == '__main__':
    import os
    main()
