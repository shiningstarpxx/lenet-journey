#!/usr/bin/env python3
"""
LeNet-5 增强版演示脚本
专门展示增强的可视化功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import LeNet5
from config import Config
from utils import set_seed
from data import DatasetLoader
from visualization import (
    visualize_conv_filters, visualize_conv_activations_for_digit,
    visualize_conv_channels_detailed, create_activation_animation_enhanced,
    create_feature_evolution_animation, create_comprehensive_visualization
)

def demo_conv_filters():
    """演示卷积滤波器可视化"""
    print("🔍 演示1: 卷积滤波器可视化")
    print("=" * 50)
    
    # 创建模型
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    print("可视化Conv1层的6个滤波器...")
    visualize_conv_filters(model, 'conv1')
    
    print("可视化Conv2层的16个滤波器...")
    visualize_conv_filters(model, 'conv2')

def demo_digit_analysis():
    """演示特定数字的详细分析"""
    print("\n🔍 演示2: 数字7的详细分析")
    print("=" * 50)
    
    # 创建模型和数据加载器
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    
    print("分析数字7的多个样本...")
    visualize_conv_activations_for_digit(
        model, train_loader, Config.DEVICE, target_digit=7, num_samples=6
    )
    
    print("详细分析数字7的卷积通道...")
    visualize_conv_channels_detailed(
        model, train_loader, Config.DEVICE, target_digit=7, sample_idx=0
    )

def demo_activation_animations():
    """演示激活动画"""
    print("\n🎬 演示3: 网络激活动画")
    print("=" * 50)
    
    # 创建模型和数据加载器
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    class_names = data_loader.get_class_names()
    
    print("创建增强版激活动画...")
    print("(展示不同样本在网络中的激活过程)")
    anim1 = create_activation_animation_enhanced(
        model, train_loader, Config.DEVICE, class_names, num_samples=5
    )
    
    print("创建特征演化动画...")
    print("(展示数字7不同样本的特征变化)")
    anim2 = create_feature_evolution_animation(
        model, train_loader, Config.DEVICE, class_names, target_digit=7, num_samples=4
    )
    
    return anim1, anim2

def demo_comprehensive_analysis():
    """演示综合分析"""
    print("\n📊 演示4: 综合分析")
    print("=" * 50)
    
    # 创建模型和数据加载器
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    class_names = data_loader.get_class_names()
    
    print("创建所有可视化并保存到results目录...")
    anim1, anim2 = create_comprehensive_visualization(
        model, train_loader, Config.DEVICE, class_names
    )
    
    return anim1, anim2

def demo_interactive_analysis():
    """演示交互式分析"""
    print("\n🎮 演示5: 交互式分析")
    print("=" * 50)
    
    # 创建模型和数据加载器
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    class_names = data_loader.get_class_names()
    
    print("请选择要分析的数字 (0-9):")
    try:
        target_digit = int(input("输入数字: "))
        if 0 <= target_digit <= 9:
            print(f"分析数字 {target_digit}...")
            visualize_conv_activations_for_digit(
                model, train_loader, Config.DEVICE, target_digit=target_digit
            )
        else:
            print("无效输入，使用默认数字7")
            target_digit = 7
            visualize_conv_activations_for_digit(
                model, train_loader, Config.DEVICE, target_digit=target_digit
            )
    except ValueError:
        print("无效输入，使用默认数字7")
        target_digit = 7
        visualize_conv_activations_for_digit(
            model, train_loader, Config.DEVICE, target_digit=target_digit
        )

def main():
    """主函数"""
    print("🎨 LeNet-5 增强版可视化演示")
    print("=" * 60)
    print("本演示将展示LeNet-5网络的详细可视化功能")
    print("包括卷积滤波器、激活分析、动画展示等")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    try:
        # 演示各个功能
        demo_conv_filters()
        
        input("\n按Enter键继续下一个演示...")
        demo_digit_analysis()
        
        input("\n按Enter键继续下一个演示...")
        anim1, anim2 = demo_activation_animations()
        
        input("\n按Enter键继续下一个演示...")
        demo_comprehensive_analysis()
        
        input("\n按Enter键继续最后一个演示...")
        demo_interactive_analysis()
        
        print("\n" + "="*60)
        print("🎉 增强版演示完成!")
        print("="*60)
        print("\n📁 生成的文件:")
        print("- results/conv1_filters.png - Conv1滤波器可视化")
        print("- results/conv2_filters.png - Conv2滤波器可视化")
        print("- results/digit7_multiple_samples.png - 数字7多样本分析")
        print("- results/digit7_detailed_channels.png - 数字7详细通道分析")
        print("- results/enhanced_activation_animation.gif - 增强版激活动画")
        print("- results/feature_evolution_animation.gif - 特征演化动画")
        
        print("\n🚀 接下来你可以:")
        print("1. 运行 'python train.py' 训练模型")
        print("2. 运行 'python evaluate.py' 评估训练好的模型")
        print("3. 查看 results/ 目录中的可视化结果")
        print("4. 运行 'python demo.py' 查看基础演示")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
