#!/usr/bin/env python3
"""
简化的测试脚本 - 专门验证多个7的多层输出展示功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from models import LeNet5
from config import Config
from utils import set_seed
from data import DatasetLoader

def test_multiple_7s_simple():
    """简化的多个7测试"""
    print("🔍 测试多个数字7的多层输出展示功能")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建模型
    model = LeNet5(input_channels=1, num_classes=10)
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(dataset_name='MNIST', data_dir='./data', batch_size=64)
    train_loader, _ = data_loader.create_dataloaders()
    
    # 收集数字7的样本
    target_digit = 7
    num_samples = 4  # 减少样本数量以便快速测试
    target_samples = []
    
    with torch.no_grad():
        for data, labels in train_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < num_samples:
                    target_samples.append(data[i])
            if len(target_samples) >= num_samples:
                break
    
    print(f"✅ 收集到 {len(target_samples)} 个数字 {target_digit} 的样本")
    
    # 获取激活
    activations = []
    with torch.no_grad():
        for sample in target_samples:
            sample_batch = sample.unsqueeze(0)
            acts = model.get_activations(sample_batch)
            activations.append(acts)
    
    print("✅ 成功获取所有样本的激活")
    
    # 创建可视化
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, num_samples, figure=fig)
    
    for sample_idx, (sample, acts) in enumerate(zip(target_samples, activations)):
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, sample_idx])
        img = sample.squeeze().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'Sample {sample_idx+1}\nDigit {target_digit}')
        ax_orig.axis('off')
        
        # Conv1 激活
        ax_conv1 = fig.add_subplot(gs[1, sample_idx])
        conv1_act = acts['conv1'].squeeze().detach().numpy()
        conv1_avg = conv1_act.mean(axis=0)
        ax_conv1.imshow(conv1_avg, cmap='hot')
        ax_conv1.set_title('Conv1 Activation\n(6 channels avg)')
        ax_conv1.axis('off')
        
        # Conv2 激活
        ax_conv2 = fig.add_subplot(gs[2, sample_idx])
        conv2_act = acts['conv2'].squeeze().detach().numpy()
        conv2_avg = conv2_act.mean(axis=0)
        ax_conv2.imshow(conv2_avg, cmap='hot')
        ax_conv2.set_title('Conv2 Activation\n(16 channels avg)')
        ax_conv2.axis('off')
    
    plt.suptitle(f'Multiple Digit {target_digit} Samples - Layer Activations', fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('./results', exist_ok=True)
    save_path = './results/multiple_7s_simple_test.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图片已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n📊 激活统计信息:")
    for sample_idx, acts in enumerate(activations):
        print(f"\nSample {sample_idx + 1}:")
        conv1_act = acts['conv1'].squeeze().detach().numpy()
        conv2_act = acts['conv2'].squeeze().detach().numpy()
        print(f"  Conv1: shape={conv1_act.shape}, mean={conv1_act.mean():.4f}, max={conv1_act.max():.4f}")
        print(f"  Conv2: shape={conv2_act.shape}, mean={conv2_act.mean():.4f}, max={conv2_act.max():.4f}")
    
    return True

def test_conv_channels_detailed():
    """测试详细的卷积通道"""
    print("\n🔍 测试详细的卷积通道可视化")
    print("=" * 50)
    
    # 创建模型
    model = LeNet5(input_channels=1, num_classes=10)
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(dataset_name='MNIST', data_dir='./data', batch_size=64)
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
        sample_batch = sample.unsqueeze(0)
        acts = model.get_activations(sample_batch)
    
    # 创建详细可视化
    fig = plt.figure(figsize=(20, 8))
    
    # 原始图像
    ax_orig = plt.subplot2grid((2, 8), (0, 0), colspan=2)
    img = sample.squeeze().numpy()
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title(f'Original Image\nDigit {target_digit}')
    ax_orig.axis('off')
    
    # Conv1 各通道
    conv1_act = acts['conv1'].squeeze().detach().numpy()
    for i in range(6):
        ax = plt.subplot2grid((2, 8), (0, 2+i))
        ax.imshow(conv1_act[i], cmap='hot')
        ax.set_title(f'Conv1-{i+1}')
        ax.axis('off')
    
    # Conv2 各通道 (只显示前8个)
    conv2_act = acts['conv2'].squeeze().detach().numpy()
    for i in range(8):
        ax = plt.subplot2grid((2, 8), (1, i))
        ax.imshow(conv2_act[i], cmap='hot')
        ax.set_title(f'Conv2-{i+1}')
        ax.axis('off')
    
    plt.suptitle(f'Digit {target_digit} - Detailed Conv Channels', fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    save_path = './results/detailed_conv_channels_simple.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 详细通道可视化已保存到: {save_path}")
    
    plt.show()
    
    return True

def create_multiple_7s_animation():
    """创建多个7的多层输出动图"""
    print("\n🎬 创建多个7的多层输出动图")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建模型
    model = LeNet5(input_channels=1, num_classes=10)
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(dataset_name='MNIST', data_dir='./data', batch_size=64)
    train_loader, _ = data_loader.create_dataloaders()
    
    # 收集数字7的样本
    target_digit = 7
    num_samples = 6
    target_samples = []
    
    with torch.no_grad():
        for data, labels in train_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < num_samples:
                    target_samples.append(data[i])
            if len(target_samples) >= num_samples:
                break
    
    print(f"✅ 收集到 {len(target_samples)} 个数字 {target_digit} 的样本")
    
    # 获取激活
    activations = []
    with torch.no_grad():
        for sample in target_samples:
            sample_batch = sample.unsqueeze(0)
            acts = model.get_activations(sample_batch)
            activations.append(acts)
    
    print("✅ 成功获取所有样本的激活")
    
    # 创建动画
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, num_samples, figure=fig)
    
    def animate(frame):
        # 清除所有子图
        for ax in fig.axes:
            ax.clear()
        
        sample_idx = frame % num_samples
        sample = target_samples[sample_idx]
        acts = activations[sample_idx]
        
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, sample_idx])
        img = sample.squeeze().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'Sample {sample_idx+1}\nDigit {target_digit}')
        ax_orig.axis('off')
        
        # Conv1 激活
        ax_conv1 = fig.add_subplot(gs[1, sample_idx])
        conv1_act = acts['conv1'].squeeze().detach().numpy()
        conv1_avg = conv1_act.mean(axis=0)
        ax_conv1.imshow(conv1_avg, cmap='hot')
        ax_conv1.set_title('Conv1 Activation\n(6 channels avg)')
        ax_conv1.axis('off')
        
        # Conv2 激活
        ax_conv2 = fig.add_subplot(gs[2, sample_idx])
        conv2_act = acts['conv2'].squeeze().detach().numpy()
        conv2_avg = conv2_act.mean(axis=0)
        ax_conv2.imshow(conv2_avg, cmap='hot')
        ax_conv2.set_title('Conv2 Activation\n(16 channels avg)')
        ax_conv2.axis('off')
    
    plt.suptitle(f'Multiple Digit {target_digit} Samples - Layer Activations Animation', fontsize=16)
    plt.tight_layout()
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=num_samples*4, 
                                 interval=1000, repeat=True)
    
    # 保存动图
    os.makedirs('./results', exist_ok=True)
    save_path = './results/multiple_7s_animation.gif'
    print(f"🎬 正在生成动图: {save_path}")
    
    try:
        anim.save(save_path, writer='pillow', fps=1)
        print(f"✅ 动图已保存到: {save_path}")
    except Exception as e:
        print(f"❌ 动图保存失败: {e}")
        return None
    
    plt.show()
    
    return anim

def main():
    """主函数"""
    print("🧪 简化测试 - 多个数字7的多层输出展示功能")
    print("=" * 60)
    
    try:
        # 测试1: 多个7的多层输出展示
        success1 = test_multiple_7s_simple()
        
        if success1:
            print("\n✅ 测试1通过: 多个数字7的多层输出展示功能正常")
        else:
            print("\n❌ 测试1失败")
        
        # 测试2: 详细的卷积通道可视化
        success2 = test_conv_channels_detailed()
        
        if success2:
            print("\n✅ 测试2通过: 详细卷积通道可视化功能正常")
        else:
            print("\n❌ 测试2失败")
        
        # 测试3: 生成动图
        print("\n🎬 开始生成动图...")
        anim = create_multiple_7s_animation()
        
        if anim is not None:
            print("\n✅ 测试3通过: 动图生成功能正常")
            success3 = True
        else:
            print("\n❌ 测试3失败: 动图生成失败")
            success3 = False
        
        print("\n" + "=" * 60)
        if success1 and success2 and success3:
            print("🎉 所有测试通过! 多个7的多层输出展示功能完全正常")
            print("\n📁 生成的文件:")
            print("- results/multiple_7s_simple_test.png")
            print("- results/detailed_conv_channels_simple.png")
            print("- results/multiple_7s_animation.gif")
        else:
            print("⚠️ 部分测试失败")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
