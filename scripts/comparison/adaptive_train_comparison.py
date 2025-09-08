#!/usr/bin/env python3
"""
自适应不同层数卷积网络的对比训练脚本
支持不同输入尺寸和通道数
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.adaptive_conv_comparison import get_adaptive_model, get_adaptive_model_info
from config import Config
from utils import set_seed, load_checkpoint
from data import DatasetLoader

class AdaptiveComparisonTrainer:
    """自适应对比训练器"""
    
    def __init__(self, model_types=['conv1', 'conv2', 'conv3'], epochs=5, input_channels=1, input_size=28):
        self.model_types = model_types
        self.epochs = epochs
        self.device = Config.DEVICE
        self.input_channels = input_channels
        self.input_size = input_size
        self.results = {}
        
        # 创建数据加载器
        self.data_loader = DatasetLoader(
            dataset_name=Config.DATASET,
            data_dir=Config.DATA_DIR,
            batch_size=Config.BATCH_SIZE
        )
        self.train_loader, self.test_loader = self.data_loader.create_dataloaders()
        
        print(f"📊 数据集信息:")
        print(f"  训练集大小: {len(self.train_loader.dataset)}")
        print(f"  测试集大小: {len(self.test_loader.dataset)}")
        print(f"  批次大小: {Config.BATCH_SIZE}")
        print(f"  设备: {self.device}")
        print(f"  输入通道数: {self.input_channels}")
        print(f"  输入尺寸: {self.input_size}x{self.input_size}")
    
    def train_model(self, model_type):
        """训练单个模型"""
        print(f"\n🚀 开始训练 {get_adaptive_model_info(model_type)['name']}")
        print("=" * 60)
        
        # 创建模型
        model = get_adaptive_model(model_type, self.input_channels, Config.NUM_CLASSES, self.input_size)
        model = model.to(self.device)
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 创建TensorBoard writer
        log_dir = f'./logs/adaptive_comparison_{model_type}'
        writer = SummaryWriter(log_dir)
        
        # 训练历史
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        best_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%")
            
            # 计算训练准确率
            train_accuracy = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(self.train_loader)
            
            # 测试阶段
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
            
            # 计算测试准确率
            test_accuracy = 100. * test_correct / test_total
            avg_test_loss = test_loss / len(self.test_loader)
            
            print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
            print(f"  测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%")
            
            # 记录历史
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Test', avg_test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
            
            # 保存最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = model.state_dict().copy()
        
        # 保存最佳模型
        os.makedirs('./checkpoints/adaptive_comparison', exist_ok=True)
        model_path = f'./checkpoints/adaptive_comparison/{model_type}_best.pth'
        torch.save(best_model_state, model_path)
        
        print(f"\n✅ {get_adaptive_model_info(model_type)['name']} 训练完成!")
        print(f"   最佳测试准确率: {best_accuracy:.2f}%")
        print(f"   模型已保存到: {model_path}")
        
        # 关闭TensorBoard writer
        writer.close()
        
        # 返回结果
        return {
            'model_type': model_type,
            'best_accuracy': best_accuracy,
            'train_accuracy': train_accuracies[-1],
            'test_accuracy': test_accuracies[-1],
            'train_loss': train_losses[-1],
            'test_loss': test_losses[-1],
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'model_path': model_path
        }
    
    def train_all_models(self):
        """训练所有模型"""
        print("🎯 开始对比训练不同层数的卷积网络")
        print("=" * 80)
        
        for model_type in self.model_types:
            info = get_adaptive_model_info(model_type)
            print(f"\n📋 模型信息:")
            print(f"   名称: {info['name']}")
            print(f"   描述: {info['description']}")
            print(f"   参数: {info['params']}")
            print(f"   卷积层: {info['conv_layers']}")
            
            result = self.train_model(model_type)
            self.results[model_type] = result
        
        return self.results
    
    def plot_comparison(self):
        """绘制对比图"""
        if not self.results:
            print("❌ 没有训练结果可以绘制")
            return
        
        print("\n📊 绘制对比图表...")
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = [get_adaptive_model_info(model_type)['name'] for model_type in self.model_types]
        accuracies = [self.results[model_type]['test_accuracy'] for model_type in self.model_types]
        train_accuracies = [self.results[model_type]['train_accuracy'] for model_type in self.model_types]
        
        # 1. 测试准确率对比
        bars1 = ax1.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax1.set_title('测试准确率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. 训练vs测试准确率
        x = np.arange(len(model_names))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, train_accuracies, width, label='训练准确率', color='lightblue', alpha=0.8)
        bars3 = ax2.bar(x + width/2, accuracies, width, label='测试准确率', color='lightcoral', alpha=0.8)
        
        ax2.set_title('训练vs测试准确率对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 3. 训练损失曲线
        for model_type in self.model_types:
            train_losses = self.results[model_type]['train_losses']
            epochs = range(1, len(train_losses) + 1)
            ax3.plot(epochs, train_losses, marker='o', label=get_adaptive_model_info(model_type)['name'])
        
        ax3.set_title('训练损失曲线', fontsize=14, fontweight='bold')
        ax3.set_xlabel('训练轮数')
        ax3.set_ylabel('损失值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 测试准确率曲线
        for model_type in self.model_types:
            test_accuracies = self.results[model_type]['test_accuracies']
            epochs = range(1, len(test_accuracies) + 1)
            ax4.plot(epochs, test_accuracies, marker='s', label=get_adaptive_model_info(model_type)['name'])
        
        ax4.set_title('测试准确率曲线', fontsize=14, fontweight='bold')
        ax4.set_xlabel('训练轮数')
        ax4.set_ylabel('准确率 (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/adaptive_conv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 对比图表已保存到: ./results/adaptive_conv_comparison.png")

def main():
    """主函数"""
    set_seed(42)
    
    # 测试自适应模型
    print("🧪 测试自适应模型...")
    trainer = AdaptiveComparisonTrainer(
        model_types=['conv1', 'conv2', 'conv3'], 
        epochs=3,
        input_channels=1,
        input_size=28
    )
    
    results = trainer.train_all_models()
    trainer.plot_comparison()
    
    print("\n🎉 自适应对比训练完成!")
    print("\n📊 结果总结:")
    for model_type, result in results.items():
        print(f"  {get_adaptive_model_info(model_type)['name']}: {result['test_accuracy']:.2f}%")

if __name__ == "__main__":
    main()
