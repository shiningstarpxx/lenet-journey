#!/usr/bin/env python3
"""
LeNet-5 快速开始脚本
自动检查环境并引导用户完成设置
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (需要3.7+)")
        return False

def check_pip():
    """检查pip是否可用"""
    print("\n📦 检查pip...")
    try:
        import pip
        print("  ✅ pip 可用")
        return True
    except ImportError:
        print("  ❌ pip 不可用")
        return False

def install_requirements():
    """安装依赖包"""
    print("\n📥 安装依赖包...")
    
    if not os.path.exists('requirements.txt'):
        print("  ❌ 找不到 requirements.txt 文件")
        return False
    
    try:
        # 尝试安装依赖
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ 依赖包安装成功")
            return True
        else:
            print(f"  ❌ 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ 安装过程出错: {e}")
        return False

def test_imports():
    """测试关键包导入"""
    print("\n🔍 测试包导入...")
    
    critical_packages = ['torch', 'torchvision', 'numpy', 'matplotlib']
    failed_packages = []
    
    for package in critical_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_packages.append(package)
    
    return len(failed_packages) == 0

def test_project_modules():
    """测试项目模块"""
    print("\n🔍 测试项目模块...")
    
    try:
        from config import Config
        from models import LeNet5
        from data import DatasetLoader
        from utils import set_seed
        print("  ✅ 所有项目模块导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 项目模块导入失败: {e}")
        return False

def run_demo():
    """运行演示"""
    print("\n🎮 运行演示...")
    try:
        subprocess.run([sys.executable, 'demo.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 演示运行失败: {e}")
        return False
    except FileNotFoundError:
        print("  ❌ 找不到 demo.py 文件")
        return False

def main():
    """主函数"""
    print("🚀 LeNet-5 快速开始向导")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ Python版本不符合要求，请升级到Python 3.7+")
        return
    
    # 检查pip
    if not check_pip():
        print("\n❌ pip不可用，请安装pip")
        return
    
    # 询问是否安装依赖
    install_deps = input("\n是否安装项目依赖? (y/n): ").lower().strip() == 'y'
    
    if install_deps:
        if not install_requirements():
            print("\n❌ 依赖安装失败，请手动安装:")
            print("pip install -r requirements.txt")
            return
    
    # 测试导入
    if not test_imports():
        print("\n❌ 关键包导入失败，请检查安装")
        return
    
    if not test_project_modules():
        print("\n❌ 项目模块导入失败，请检查文件完整性")
        return
    
    print("\n✅ 环境检查完成!")
    
    # 询问是否运行演示
    run_demo_choice = input("\n是否运行演示? (y/n): ").lower().strip() == 'y'
    
    if run_demo_choice:
        if run_demo():
            print("\n🎉 演示运行成功!")
        else:
            print("\n⚠️ 演示运行失败，但项目设置可能正常")
    
    # 显示下一步操作
    print("\n" + "=" * 50)
    print("🎯 下一步操作:")
    print("1. 运行训练: python3 train.py")
    print("2. 运行评估: python3 evaluate.py")
    print("3. 单张推理: python3 inference.py --image your_image.jpg")
    print("4. 交互菜单: python3 run.py")
    print("5. 查看文档: cat README.md")
    
    print("\n🎓 学习建议:")
    print("- 先运行演示了解项目功能")
    print("- 然后开始训练模型")
    print("- 最后尝试评估和推理")
    
    print("\n📚 更多帮助:")
    print("- 查看 INSTALL.md 了解详细安装说明")
    print("- 查看 README.md 了解项目功能")
    print("- 查看 config.py 了解配置选项")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查环境设置或查看错误信息")
