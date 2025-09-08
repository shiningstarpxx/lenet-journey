#!/usr/bin/env python3
"""
测试多个数字7的多层输出结果展示
专门验证这个功能是否正常工作
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models import LeNet5
from config import Config
from utils import set_seed
from data import DatasetLoader

def test_multiple_7s_visualization():
    """测试多个7的多层输出可视化"""
    print("🔍 测试多个数字7的多层输出展示功能")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建模型
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    
    print("✅ 模型和数据加载器创建成功")
    
    # 收集数字7的样本
    target_digit = 7
    num_samples = 6
    target_samples = []
    target_labels = []
    
    print(f"🔍 收集数字 {target_digit} 的样本...")
    
    with torch.no_grad():
        for data, labels in train_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < num_samples:
                    target_samples.append(data[i])
                    target_labels.append(label.item())
            
            if len(target_samples) >= num_samples:
                break
    
    print(f"✅ 成功收集到 {len(target_samples)} 个数字 {target_digit} 的样本")
    
    if len(target_samples) == 0:
        print("❌ 未找到目标数字的样本")
        return False
    
    # 获取各层激活
    print("🔍 获取各层激活...")
    activations = []
    
    with torch.no_grad():
        for i, sample in enumerate(target_samples):
            print(f"  处理样本 {i+1}/{len(target_samples)}")
            sample_batch = sample.unsqueeze(0).to(Config.DEVICE)
            
            if hasattr(model, 'get_activations'):
                acts = model.get_activations(sample_batch)
                activations.append(acts)
            else:
                print("❌ 模型没有get_activations方法")
                return False
    
    print("✅ 成功获取所有样本的激活")
    
    # 创建可视化
    print("🎨 创建多层输出可视化...")
    
    # 创建大图
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, num_samples, figure=fig)
    
    # 定义层名和对应的网格位置
    layers_info = [
        ('conv1', 0, 'Conv1 激活\n(6通道平均)'),
        ('pool1', 1, 'Pool1 激活\n(6通道平均)'),
        ('conv2', 2, 'Conv2 激活\n(16通道平均)'),
        ('pool2', 3, 'Pool2 激活\n(16通道平均)')
    ]
    
    for sample_idx, (sample, acts) in enumerate(zip(target_samples, activations)):
        print(f"  绘制样本 {sample_idx + 1} 的可视化...")
        
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, sample_idx])
        img = sample.squeeze().cpu().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'样本 {sample_idx+1}\n数字 {target_digit}', fontsize=10)
        ax_orig.axis('off')
        
        # 各层激活
        for layer_name, row, title in layers_info:
            if layer_name in acts:
                ax = fig.add_subplot(gs[row, sample_idx])
                act = acts[layer_name].squeeze().cpu().numpy()
                
                if len(act.shape) == 3:  # 卷积层
                    # 显示所有通道的平均激活
                    act_avg = act.mean(axis=0)
                    im = ax.imshow(act_avg, cmap='hot')
                    ax.set_title(title, fontsize=9)
                else:  # 其他层
                    # 显示激活值的条形图
                    ax.bar(range(len(act)), act, color='skyblue')
                    ax.set_title(title, fontsize=9)
                
                ax.axis('off')
    
    plt.suptitle(f'数字 {target_digit} 的多个样本在各层的激活情况', fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(Config.RESULTS_DIR, 'multiple_7s_layers_visualization.png')
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图片已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n📊 激活统计信息:")
    for sample_idx, acts in enumerate(activations):
        print(f"\n样本 {sample_idx + 1}:")
        for layer_name, _, _ in layers_info:
            if layer_name in acts:
                act = acts[layer_name].squeeze().cpu().numpy()
                if len(act.shape) == 3:
                    print(f"  {layer_name}: 形状={act.shape}, 均值={act.mean():.4f}, 最大值={act.max():.4f}")
                else:
                    print(f"  {layer_name}: 形状={act.shape}, 均值={act.mean():.4f}, 最大值={act.max():.4f}")
    
    return True

def test_detailed_conv_channels():
    """测试详细的卷积通道可视化"""
    print("\n🔍 测试详细的卷积通道可视化")
    print("=" * 60)
    
    # 创建模型
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    
    # 获取一个数字7的样本
    target_digit = 7
    sample = None
    
    for data, labels in train_loader:
        for i, label in enumerate(labels):
            if label.item() == target_digit:
                sample = data[i]
                break
        if sample is not None:
            break
    
    if sample is None:
        print("❌ 未找到数字7的样本")
        return False
    
    # 获取激活
    with torch.no_grad():
        sample_batch = sample.unsqueeze(0).to(Config.DEVICE)
        acts = model.get_activations(sample_batch)
    
    # 创建详细可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 原始图像
    ax_orig = plt.subplot2grid((3, 8), (0, 0), colspan=2)
    img = sample.squeeze().cpu().numpy()
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title(f'原始图像\n数字 {target_digit}')
    ax_orig.axis('off')
    
    # Conv1 各通道
    conv1_act = acts['conv1'].squeeze().cpu().numpy()
    for i in range(6):
        ax = plt.subplot2grid((3, 8), (0, 2+i))
        ax.imshow(conv1_act[i], cmap='hot')
        ax.set_title(f'Conv1-{i+1}')
        ax.axis('off')
    
    # Conv2 各通道
    conv2_act = acts['conv2'].squeeze().cpu().numpy()
    for i in range(16):
        row = 1 + i // 8
        col = i % 8
        ax = plt.subplot2grid((3, 8), (row, col))
        ax.imshow(conv2_act[i], cmap='hot')
        ax.set_title(f'Conv2-{i+1}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'数字 {target_digit} 的详细卷积通道激活', fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(Config.RESULTS_DIR, 'detailed_conv_channels_7.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 详细通道可视化已保存到: {save_path}")
    
    plt.show()
    
    return True

def main():
    """主函数"""
    print("🧪 测试多个数字7的多层输出展示功能")
    print("=" * 60)
    
    try:
        # 测试1: 多个7的多层输出展示
        success1 = test_multiple_7s_visualization()
        
        if success1:
            print("\n✅ 测试1通过: 多个数字7的多层输出展示功能正常")
        else:
            print("\n❌ 测试1失败: 多个数字7的多层输出展示功能异常")
        
        # 测试2: 详细的卷积通道可视化
        success2 = test_detailed_conv_channels()
        
        if success2:
            print("\n✅ 测试2通过: 详细卷积通道可视化功能正常")
        else:
            print("\n❌ 测试2失败: 详细卷积通道可视化功能异常")
        
        print("\n" + "=" * 60)
        if success1 and success2:
            print("🎉 所有测试通过! 多个7的多层输出展示功能完全正常")
            print("\n📁 生成的文件:")
            print("- results/multiple_7s_layers_visualization.png")
            print("- results/detailed_conv_channels_7.png")
        else:
            print("⚠️ 部分测试失败，请检查错误信息")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
