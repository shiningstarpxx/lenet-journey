#!/usr/bin/env python3
"""
LeNet-5 项目启动脚本
提供交互式菜单来选择不同的功能
"""

import os
import sys
import subprocess
import argparse

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    LeNet-5 深度学习入门项目                    ║
    ║                                                              ║
    ║  🎯 完整的CNN实现和深度学习教学项目                            ║
    ║  📊 包含训练、评估、可视化和分析功能                           ║
    ║  🚀 专为深度学习初学者设计                                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """打印主菜单"""
    menu = """
    📋 请选择要执行的操作:
    
    1. 🎮 运行演示 (demo.py)
    2. 🏋️  训练模型 (train.py)
    3. 📊 评估模型 (evaluate.py)
    4. 🔮 单张图片推理 (inference.py)
    5. 📁 查看项目结构
    6. 📖 查看README
    7. 🛠️  安装依赖
    8. ❌ 退出
    
    """
    print(menu)

def run_demo():
    """运行演示"""
    print("🎮 运行LeNet-5演示...")
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 演示运行失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到demo.py文件")

def run_training():
    """运行训练"""
    print("🏋️ 开始训练LeNet-5模型...")
    print("注意: 训练可能需要几分钟到几十分钟，请耐心等待...")
    
    # 询问是否使用GPU
    use_gpu = input("是否使用GPU加速? (y/n): ").lower().strip() == 'y'
    
    try:
        if use_gpu:
            print("🚀 使用GPU加速训练...")
        else:
            print("💻 使用CPU训练...")
        
        subprocess.run([sys.executable, "train.py"], check=True)
        print("✅ 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")

def run_evaluation():
    """运行评估"""
    print("📊 开始评估模型...")
    
    # 检查模型文件是否存在
    model_path = os.path.join("models", "best_model.pth")
    if not os.path.exists(model_path):
        print("⚠️  未找到训练好的模型文件")
        print("请先运行训练 (选项2) 或确保模型文件存在于 models/ 目录中")
        return
    
    try:
        subprocess.run([sys.executable, "evaluate.py"], check=True)
        print("✅ 评估完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败: {e}")

def run_inference():
    """运行推理"""
    print("🔮 单张图片推理")
    
    # 检查模型文件
    model_path = os.path.join("models", "best_model.pth")
    if not os.path.exists(model_path):
        print("⚠️  未找到训练好的模型文件")
        print("请先运行训练 (选项2)")
        return
    
    # 获取图片路径
    image_path = input("请输入图片路径: ").strip()
    if not image_path:
        print("❌ 未提供图片路径")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    # 询问数据集类型
    dataset = input("数据集类型 (MNIST/CIFAR10) [默认: MNIST]: ").strip()
    if not dataset:
        dataset = "MNIST"
    
    try:
        cmd = [sys.executable, "inference.py", "--image", image_path, "--dataset", dataset]
        subprocess.run(cmd, check=True)
        print("✅ 推理完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败: {e}")

def show_project_structure():
    """显示项目结构"""
    print("📁 项目结构:")
    print("""
    Lenet-journey/
    ├── models/                 # 模型定义
    │   ├── __init__.py
    │   └── lenet.py           # LeNet-5网络架构
    ├── data/                  # 数据处理
    │   ├── __init__.py
    │   └── dataset.py         # 数据加载和预处理
    ├── config.py              # 配置文件
    ├── train.py               # 训练脚本
    ├── evaluate.py            # 评估脚本
    ├── inference.py           # 推理脚本
    ├── demo.py                # 演示脚本
    ├── utils.py               # 工具函数
    ├── run.py                 # 启动脚本 (当前文件)
    ├── requirements.txt       # 依赖包
    └── README.md             # 项目说明
    
    运行后会自动创建的目录:
    ├── data/                  # 数据集存储
    ├── models/                # 模型保存
    ├── logs/                  # TensorBoard日志
    └── results/               # 评估结果和可视化
    """)

def show_readme():
    """显示README内容"""
    readme_path = "README.md"
    if os.path.exists(readme_path):
        print("📖 README.md 内容:")
        print("=" * 60)
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 只显示前1000个字符
            if len(content) > 1000:
                print(content[:1000] + "...")
                print(f"\n(显示前1000个字符，完整内容请查看 {readme_path})")
            else:
                print(content)
    else:
        print("❌ 找不到README.md文件")

def install_dependencies():
    """安装依赖"""
    print("🛠️ 安装项目依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ 依赖安装完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到requirements.txt文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LeNet-5 项目启动脚本')
    parser.add_argument('--auto', choices=['demo', 'train', 'eval'], 
                       help='自动运行指定功能')
    
    args = parser.parse_args()
    
    # 如果指定了自动运行
    if args.auto:
        print_banner()
        if args.auto == 'demo':
            run_demo()
        elif args.auto == 'train':
            run_training()
        elif args.auto == 'eval':
            run_evaluation()
        return
    
    # 交互式菜单
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input("请输入选项编号 (1-8): ").strip()
            
            if choice == '1':
                run_demo()
            elif choice == '2':
                run_training()
            elif choice == '3':
                run_evaluation()
            elif choice == '4':
                run_inference()
            elif choice == '5':
                show_project_structure()
            elif choice == '6':
                show_readme()
            elif choice == '7':
                install_dependencies()
            elif choice == '8':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选项，请输入1-8之间的数字")
            
            # 等待用户按键继续
            if choice in ['1', '2', '3', '4']:
                input("\n按Enter键继续...")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == '__main__':
    main()
