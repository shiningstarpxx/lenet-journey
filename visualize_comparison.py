#!/usr/bin/env python3
"""
不同层数卷积网络的可视化对比
比较1层Conv、2层Conv和3层Conv的激活可视化
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from models.conv_comparison import get_model, get_model_info
from config import Config
from utils import set_seed
from data import DatasetLoader

class ComparisonVisualizer:
    """对比可视化器"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.data_loader = DatasetLoader(
            dataset_name=Config.DATASET,
            data_dir=Config.DATA_DIR,
            batch_size=Config.BATCH_SIZE
        )
        self.train_loader, _ = self.data_loader.create_dataloaders()
        
    def load_trained_models(self, model_types=['conv1', 'conv2']):
        """加载训练好的模型"""
        models = {}
        
        for model_type in model_types:
            checkpoint_path = f'./checkpoints/comparison/{model_type}_best.pth'
            
            if os.path.exists(checkpoint_path):
                # 加载模型
                model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                models[model_type] = {
                    'model': model,
                    'accuracy': checkpoint['best_accuracy'],
                    'info': get_model_info(model_type)
                }
                
                print(f"✅ 加载 {model_type} 模型，准确率: {checkpoint['best_accuracy']:.2f}%")
            else:
                print(f"⚠️ 未找到 {model_type} 模型检查点: {checkpoint_path}")
                # 创建未训练的模型用于演示
                model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
                model = model.to(self.device)
                model.eval()
                
                models[model_type] = {
                    'model': model,
                    'accuracy': 0.0,
                    'info': get_model_info(model_type)
                }
                
                print(f"⚠️ 使用未训练的 {model_type} 模型进行演示")
        
        return models
    
    def get_sample_data(self, target_digit=7, num_samples=4):
        """获取样本数据"""
        target_samples = []
        
        with torch.no_grad():
            for data, labels in self.train_loader:
                for i, label in enumerate(labels):
                    if label.item() == target_digit and len(target_samples) < num_samples:
                        target_samples.append(data[i])
                if len(target_samples) >= num_samples:
                    break
        
        print(f"✅ 收集到 {len(target_samples)} 个数字 {target_digit} 的样本")
        return target_samples
    
    def visualize_activations_comparison(self, models, target_digit=7, num_samples=4, save_path=None):
        """可视化不同模型的激活对比"""
        print(f"\n🎨 可视化不同模型的激活对比 (数字 {target_digit})")
        print("=" * 60)
        
        # 获取样本数据
        samples = self.get_sample_data(target_digit, num_samples)
        
        if not samples:
            print(f"❌ 未找到数字 {target_digit} 的样本")
            return None
        
        # 获取每个模型的激活
        all_activations = {}
        for model_type, model_data in models.items():
            model = model_data['model']
            activations = []
            
            with torch.no_grad():
                for sample in samples:
                    sample_batch = sample.unsqueeze(0).to(self.device)
                    acts = model.get_activations(sample_batch)
                    activations.append(acts)
            
            all_activations[model_type] = activations
            print(f"✅ 获取 {model_type} 模型的激活")
        
        # 创建可视化
        model_types = list(models.keys())
        num_models = len(model_types)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 为每个样本创建一行
        for sample_idx in range(num_samples):
            sample = samples[sample_idx]
            
            # 原始图像
            ax_orig = plt.subplot2grid((num_samples + 1, num_models + 1), (sample_idx, 0))
            img = sample.squeeze().cpu().numpy()
            ax_orig.imshow(img, cmap='gray')
            ax_orig.set_title(f'Sample {sample_idx+1}\nDigit {target_digit}', fontsize=12)
            ax_orig.axis('off')
            
            # 每个模型的激活
            for model_idx, model_type in enumerate(model_types):
                model_data = models[model_type]
                activations = all_activations[model_type][sample_idx]
                
                # 获取该模型的卷积层
                conv_layers = model_data['info']['layers']
                
                # 显示第一个卷积层的激活
                if conv_layers and conv_layers[0] in activations:
                    ax = plt.subplot2grid((num_samples + 1, num_models + 1), (sample_idx, model_idx + 1))
                    conv_act = activations[conv_layers[0]].squeeze().detach().cpu().numpy()
                    
                    # 如果是多通道，计算平均值
                    if len(conv_act.shape) == 3:
                        conv_avg = conv_act.mean(axis=0)
                    else:
                        conv_avg = conv_act
                    
                    im = ax.imshow(conv_avg, cmap='hot')
                    ax.set_title(f'{model_data["info"]["name"]}\n{conv_layers[0]} (Acc: {model_data["accuracy"]:.1f}%)', 
                               fontsize=10)
                    ax.axis('off')
        
        # 添加统计信息行
        ax_stats = plt.subplot2grid((num_samples + 1, num_models + 1), (num_samples, 0), colspan=num_models + 1)
        ax_stats.axis('off')
        
        # 计算统计信息
        stats_text = "模型对比统计:\n"
        for model_type, model_data in models.items():
            info = model_data['info']
            stats_text += f"{info['name']}: {info['params']}, 准确率: {model_data['accuracy']:.2f}%\n"
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.suptitle(f'不同层数卷积网络激活对比 - 数字 {target_digit}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            os.makedirs('./results', exist_ok=True)
            save_path = f'./results/conv_activations_comparison_{target_digit}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 激活对比图已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def create_comparison_animation(self, models, target_digit=7, num_samples=4, save_path=None):
        """创建对比动画"""
        print(f"\n🎬 创建不同模型的对比动画 (数字 {target_digit})")
        print("=" * 60)
        
        # 获取样本数据
        samples = self.get_sample_data(target_digit, num_samples)
        
        if not samples:
            print(f"❌ 未找到数字 {target_digit} 的样本")
            return None
        
        # 获取每个模型的激活
        all_activations = {}
        for model_type, model_data in models.items():
            model = model_data['model']
            activations = []
            
            with torch.no_grad():
                for sample in samples:
                    sample_batch = sample.unsqueeze(0).to(self.device)
                    acts = model.get_activations(sample_batch)
                    activations.append(acts)
            
            all_activations[model_type] = activations
        
        # 创建动画
        model_types = list(models.keys())
        num_models = len(model_types)
        
        fig = plt.figure(figsize=(20, 12))
        
        def animate(frame):
            # 清除所有子图
            for ax in fig.axes:
                ax.clear()
            
            sample_idx = frame % num_samples
            sample = samples[sample_idx]
            
            # 原始图像
            ax_orig = plt.subplot2grid((num_samples + 1, num_models + 1), (0, 0))
            img = sample.squeeze().cpu().numpy()
            ax_orig.imshow(img, cmap='gray')
            ax_orig.set_title(f'Sample {sample_idx+1}\nDigit {target_digit}', fontsize=12)
            ax_orig.axis('off')
            
            # 每个模型的激活
            for model_idx, model_type in enumerate(model_types):
                model_data = models[model_type]
                activations = all_activations[model_type][sample_idx]
                
                # 获取该模型的卷积层
                conv_layers = model_data['info']['layers']
                
                # 显示第一个卷积层的激活
                if conv_layers and conv_layers[0] in activations:
                    ax = plt.subplot2grid((num_samples + 1, num_models + 1), (0, model_idx + 1))
                    conv_act = activations[conv_layers[0]].squeeze().detach().cpu().numpy()
                    
                    # 如果是多通道，计算平均值
                    if len(conv_act.shape) == 3:
                        conv_avg = conv_act.mean(axis=0)
                    else:
                        conv_avg = conv_act
                    
                    im = ax.imshow(conv_avg, cmap='hot')
                    ax.set_title(f'{model_data["info"]["name"]}\n{conv_layers[0]} (Acc: {model_data["accuracy"]:.1f}%)', 
                               fontsize=10)
                    ax.axis('off')
            
            # 添加统计信息
            ax_stats = plt.subplot2grid((num_samples + 1, num_models + 1), (1, 0), colspan=num_models + 1)
            ax_stats.axis('off')
            
            # 计算当前样本的统计信息
            stats_text = f"Sample {sample_idx + 1} 激活统计:\n"
            for model_type, model_data in models.items():
                activations = all_activations[model_type][sample_idx]
                conv_layers = model_data['info']['layers']
                
                if conv_layers and conv_layers[0] in activations:
                    conv_act = activations[conv_layers[0]].squeeze().detach().cpu().numpy()
                    stats_text += f"{model_data['info']['name']}: mean={conv_act.mean():.3f}, std={conv_act.std():.3f}, max={conv_act.max():.3f}\n"
            
            ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.suptitle(f'不同层数卷积网络激活对比动画 - 数字 {target_digit}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=num_samples * 3, 
                                     interval=1500, repeat=True)
        
        # 保存动画
        if save_path is None:
            os.makedirs('./results', exist_ok=True)
            save_path = f'./results/conv_comparison_animation_{target_digit}.gif'
        
        print(f"🎬 正在生成对比动画: {save_path}")
        
        try:
            anim.save(save_path, writer='pillow', fps=1)
            print(f"✅ 对比动画已保存到: {save_path}")
            
            # 显示文件大小
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            print(f"📁 文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ 动画保存失败: {e}")
            return None
        
        plt.show()
        
        return anim
    
    def visualize_detailed_channels_comparison(self, models, target_digit=7, save_path=None):
        """可视化详细通道对比"""
        print(f"\n🔍 可视化详细通道对比 (数字 {target_digit})")
        print("=" * 60)
        
        # 获取一个样本
        samples = self.get_sample_data(target_digit, 1)
        if not samples:
            print(f"❌ 未找到数字 {target_digit} 的样本")
            return None
        
        sample = samples[0]
        
        # 获取每个模型的激活
        all_activations = {}
        for model_type, model_data in models.items():
            model = model_data['model']
            
            with torch.no_grad():
                sample_batch = sample.unsqueeze(0).to(self.device)
                acts = model.get_activations(sample_batch)
                all_activations[model_type] = acts
        
        # 创建可视化
        model_types = list(models.keys())
        num_models = len(model_types)
        
        fig = plt.figure(figsize=(24, 16))
        
        # 原始图像
        ax_orig = plt.subplot2grid((3, num_models + 1), (0, 0))
        img = sample.squeeze().cpu().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'Original\nDigit {target_digit}', fontsize=12)
        ax_orig.axis('off')
        
        # 每个模型的详细通道
        for model_idx, model_type in enumerate(model_types):
            model_data = models[model_type]
            activations = all_activations[model_type]
            conv_layers = model_data['info']['layers']
            
            if conv_layers and conv_layers[0] in activations:
                conv_act = activations[conv_layers[0]].squeeze().detach().cpu().numpy()
                
                # 显示所有通道
                num_channels = conv_act.shape[0]
                cols = min(6, num_channels)  # 最多显示6列
                rows = (num_channels + cols - 1) // cols
                
                for ch_idx in range(num_channels):
                    row = ch_idx // cols
                    col = ch_idx % cols
                    
                    ax = plt.subplot2grid((3, num_models + 1), (row, model_idx + 1))
                    ax.imshow(conv_act[ch_idx], cmap='hot')
                    ax.set_title(f'{model_data["info"]["name"]}\nCh{ch_idx+1}', fontsize=8)
                    ax.axis('off')
        
        plt.suptitle(f'详细通道对比 - 数字 {target_digit}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            os.makedirs('./results', exist_ok=True)
            save_path = f'./results/detailed_channels_comparison_{target_digit}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 详细通道对比图已保存到: {save_path}")
        
        plt.show()
        
        return fig

def main():
    """主函数"""
    print("🎨 不同层数卷积网络可视化对比")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建可视化器
    visualizer = ComparisonVisualizer()
    
    try:
        # 加载训练好的模型
        models = visualizer.load_trained_models(['conv1', 'conv2', 'conv3'])
        
        if not models:
            print("❌ 没有可用的模型")
            return
        
        # 可视化激活对比
        visualizer.visualize_activations_comparison(models, target_digit=7, num_samples=4)
        
        # 创建对比动画
        visualizer.create_comparison_animation(models, target_digit=7, num_samples=4)
        
        # 可视化详细通道对比
        visualizer.visualize_detailed_channels_comparison(models, target_digit=7)
        
        print("\n🎉 可视化对比完成!")
        print("\n📁 生成的文件:")
        print("- results/conv_activations_comparison_7.png")
        print("- results/conv_comparison_animation_7.gif")
        print("- results/detailed_channels_comparison_7.png")
        
    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
