#!/usr/bin/env python3
"""
专门的动图生成脚本
生成多个数字7的多层输出动图
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

def generate_multiple_7s_animation(num_samples=6, target_digit=7, fps=1, save_path=None):
    """生成多个数字7的多层输出动图"""
    print(f"🎬 生成数字{target_digit}的多层输出动图")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建模型
    model = LeNet5(input_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES)
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    
    # 收集目标数字的样本
    target_samples = []
    
    with torch.no_grad():
        for data, labels in train_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < num_samples:
                    target_samples.append(data[i])
            if len(target_samples) >= num_samples:
                break
    
    print(f"✅ 收集到 {len(target_samples)} 个数字 {target_digit} 的样本")
    
    if len(target_samples) == 0:
        print(f"❌ 未找到数字 {target_digit} 的样本")
        return None
    
    # 获取激活
    activations = []
    with torch.no_grad():
        for sample in target_samples:
            sample_batch = sample.unsqueeze(0).to(Config.DEVICE)
            acts = model.get_activations(sample_batch)
            activations.append(acts)
    
    print("✅ 成功获取所有样本的激活")
    
    # 创建动画
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, num_samples, figure=fig)
    
    def animate(frame):
        # 清除所有子图
        for ax in fig.axes:
            ax.clear()
        
        sample_idx = frame % num_samples
        sample = target_samples[sample_idx]
        acts = activations[sample_idx]
        
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, sample_idx])
        img = sample.squeeze().cpu().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'Sample {sample_idx+1}\nDigit {target_digit}', fontsize=12)
        ax_orig.axis('off')
        
        # Conv1 激活
        ax_conv1 = fig.add_subplot(gs[1, sample_idx])
        conv1_act = acts['conv1'].squeeze().detach().cpu().numpy()
        conv1_avg = conv1_act.mean(axis=0)
        im1 = ax_conv1.imshow(conv1_avg, cmap='hot')
        ax_conv1.set_title('Conv1 Activation\n(6 channels avg)', fontsize=10)
        ax_conv1.axis('off')
        
        # Conv2 激活
        ax_conv2 = fig.add_subplot(gs[2, sample_idx])
        conv2_act = acts['conv2'].squeeze().detach().cpu().numpy()
        conv2_avg = conv2_act.mean(axis=0)
        im2 = ax_conv2.imshow(conv2_avg, cmap='hot')
        ax_conv2.set_title('Conv2 Activation\n(16 channels avg)', fontsize=10)
        ax_conv2.axis('off')
        
        # 激活统计信息
        ax_stats = fig.add_subplot(gs[3, sample_idx])
        ax_stats.axis('off')
        
        # 计算统计信息
        stats_text = f"""
Sample {sample_idx + 1} Stats:
Conv1: μ={conv1_act.mean():.3f}
       σ={conv1_act.std():.3f}
       max={conv1_act.max():.3f}

Conv2: μ={conv2_act.mean():.3f}
       σ={conv2_act.std():.3f}
       max={conv2_act.max():.3f}
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=8, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.suptitle(f'Multiple Digit {target_digit} Samples - Layer Activations Animation', fontsize=16)
    plt.tight_layout()
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=num_samples*3, 
                                 interval=1000//fps, repeat=True)
    
    # 保存动图
    if save_path is None:
        os.makedirs('./results', exist_ok=True)
        save_path = f'./results/multiple_{target_digit}s_animation.gif'
    
    print(f"🎬 正在生成动图: {save_path}")
    
    try:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"✅ 动图已保存到: {save_path}")
        
        # 显示文件大小
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 动图保存失败: {e}")
        return None
    
    plt.show()
    
    return anim

def generate_detailed_channels_animation(target_digit=7, save_path=None):
    """生成详细卷积通道的动图"""
    print(f"🎬 生成数字{target_digit}的详细卷积通道动图")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建模型
    model = LeNet5(input_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES)
    model.eval()
    
    # 创建数据加载器
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    train_loader, _ = data_loader.create_dataloaders()
    
    # 获取目标数字的样本
    target_samples = []
    
    with torch.no_grad():
        for data, labels in train_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < 4:
                    target_samples.append(data[i])
            if len(target_samples) >= 4:
                break
    
    print(f"✅ 收集到 {len(target_samples)} 个数字 {target_digit} 的样本")
    
    if len(target_samples) == 0:
        print(f"❌ 未找到数字 {target_digit} 的样本")
        return None
    
    # 获取激活
    activations = []
    with torch.no_grad():
        for sample in target_samples:
            sample_batch = sample.unsqueeze(0).to(Config.DEVICE)
            acts = model.get_activations(sample_batch)
            activations.append(acts)
    
    # 创建动画
    fig = plt.figure(figsize=(20, 12))
    
    def animate(frame):
        # 清除所有子图
        for ax in fig.axes:
            ax.clear()
        
        sample_idx = frame % len(target_samples)
        sample = target_samples[sample_idx]
        acts = activations[sample_idx]
        
        # 原始图像
        ax_orig = plt.subplot2grid((3, 8), (0, 0), colspan=2)
        img = sample.squeeze().cpu().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'Sample {sample_idx+1}\nDigit {target_digit}')
        ax_orig.axis('off')
        
        # Conv1 各通道
        conv1_act = acts['conv1'].squeeze().detach().cpu().numpy()
        for i in range(6):
            ax = plt.subplot2grid((3, 8), (0, 2+i))
            ax.imshow(conv1_act[i], cmap='hot')
            ax.set_title(f'Conv1-{i+1}')
            ax.axis('off')
        
        # Conv2 各通道 (只显示前8个)
        conv2_act = acts['conv2'].squeeze().detach().cpu().numpy()
        for i in range(8):
            ax = plt.subplot2grid((3, 8), (1, i))
            ax.imshow(conv2_act[i], cmap='hot')
            ax.set_title(f'Conv2-{i+1}')
            ax.axis('off')
        
        # 激活统计
        ax_stats = plt.subplot2grid((3, 8), (2, 0), colspan=8)
        ax_stats.axis('off')
        
        stats_text = f"""
Sample {sample_idx + 1} - Conv1: mean={conv1_act.mean():.3f}, std={conv1_act.std():.3f}, max={conv1_act.max():.3f}
Sample {sample_idx + 1} - Conv2: mean={conv2_act.mean():.3f}, std={conv2_act.std():.3f}, max={conv2_act.max():.3f}
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.suptitle(f'Digit {target_digit} - Detailed Conv Channels Animation', fontsize=16)
    plt.tight_layout()
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=len(target_samples)*4, 
                                 interval=1500, repeat=True)
    
    # 保存动图
    if save_path is None:
        os.makedirs('./results', exist_ok=True)
        save_path = f'./results/detailed_channels_{target_digit}_animation.gif'
    
    print(f"🎬 正在生成详细通道动图: {save_path}")
    
    try:
        anim.save(save_path, writer='pillow', fps=1)
        print(f"✅ 详细通道动图已保存到: {save_path}")
        
        # 显示文件大小
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 动图保存失败: {e}")
        return None
    
    plt.show()
    
    return anim

def main():
    """主函数"""
    print("🎬 LeNet-5 动图生成器")
    print("=" * 50)
    
    try:
        # 生成多个7的多层输出动图
        print("1. 生成多个数字7的多层输出动图...")
        anim1 = generate_multiple_7s_animation(num_samples=6, target_digit=7, fps=1)
        
        if anim1 is not None:
            print("✅ 多个7动图生成成功")
        else:
            print("❌ 多个7动图生成失败")
        
        # 生成详细卷积通道动图
        print("\n2. 生成详细卷积通道动图...")
        anim2 = generate_detailed_channels_animation(target_digit=7)
        
        if anim2 is not None:
            print("✅ 详细通道动图生成成功")
        else:
            print("❌ 详细通道动图生成失败")
        
        print("\n" + "=" * 50)
        print("🎉 动图生成完成!")
        print("\n📁 生成的文件:")
        print("- results/multiple_7s_animation.gif")
        print("- results/detailed_channels_7_animation.gif")
        
        print("\n💡 使用提示:")
        print("- 动图展示了不同数字7样本在网络各层的激活情况")
        print("- 可以观察到不同样本在Conv1和Conv2层的激活模式差异")
        print("- 动图会自动循环播放，方便观察和分析")
        
    except Exception as e:
        print(f"❌ 动图生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
