#!/usr/bin/env python3
"""
LeNet Journey 主运行脚本
提供统一的入口点来运行各种功能
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='LeNet Journey - 深度学习入门项目')
    parser.add_argument('--mode', type=str, default='help', 
                       choices=['help', 'quick', 'train', 'compare', 'visualize', 'demo', 'test'],
                       help='运行模式')
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10'],
                       help='数据集选择')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    if args.mode == 'help':
        print_help()
    elif args.mode == 'quick':
        run_quick_start()
    elif args.mode == 'train':
        run_training(args.dataset, args.epochs)
    elif args.mode == 'compare':
        run_comparison(args.epochs)
    elif args.mode == 'visualize':
        run_visualization()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'test':
        run_tests()

def print_help():
    """打印帮助信息"""
    print("🎯 LeNet Journey - 深度学习入门项目")
    print("=" * 50)
    print()
    print("📚 可用模式:")
    print("  quick      - 快速开始演示")
    print("  train      - 训练LeNet模型")
    print("  compare    - 运行模型对比分析")
    print("  visualize  - 生成可视化图表")
    print("  demo       - 运行功能演示")
    print("  test       - 运行测试验证")
    print()
    print("🚀 使用示例:")
    print("  python main.py --mode quick")
    print("  python main.py --mode train --dataset MNIST --epochs 5")
    print("  python main.py --mode compare --epochs 3")
    print("  python main.py --mode visualize")
    print("  python main.py --mode demo")
    print("  python main.py --mode test")
    print()
    print("📖 详细文档:")
    print("  - README.md: 项目主文档")
    print("  - PROJECT_STRUCTURE.md: 项目结构说明")
    print("  - docs/guides/: 各种功能指南")

def run_quick_start():
    """运行快速开始"""
    print("🚀 运行快速开始演示...")
    os.system("python quick_start.py")

def run_training(dataset, epochs):
    """运行训练"""
    print(f"🏋️ 训练LeNet模型 (数据集: {dataset}, 轮数: {epochs})...")
    os.system(f"python train.py")

def run_comparison(epochs):
    """运行对比分析"""
    print(f"📊 运行模型对比分析 (轮数: {epochs})...")
    print("1. 运行卷积层对比分析...")
    os.system("python scripts/comparison/conv_comparison_analysis.py")
    print("2. 运行双数据集对比分析...")
    os.system("python scripts/comparison/dual_dataset_comparison.py")

def run_visualization():
    """运行可视化"""
    print("🎨 生成可视化图表...")
    print("1. 生成模型架构图...")
    os.system("python scripts/visualization/visualize_model_architecture_v3.py")
    print("2. 生成对比可视化...")
    os.system("python scripts/visualization/visualize_comparison.py")

def run_demo():
    """运行演示"""
    print("🎪 运行功能演示...")
    print("1. 基础演示...")
    os.system("python scripts/demo/demo.py")
    print("2. 增强演示...")
    os.system("python scripts/demo/enhanced_demo.py")

def run_tests():
    """运行测试"""
    print("🧪 运行测试验证...")
    print("1. 环境测试...")
    os.system("python scripts/test/test_setup.py")
    print("2. 中文显示测试...")
    os.system("python scripts/test/test_chinese_display.py")

if __name__ == "__main__":
    main()
