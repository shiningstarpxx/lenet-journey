#!/usr/bin/env python3
"""
不同层数卷积网络的对比训练脚本
比较1层Conv、2层Conv和3层Conv的效果
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

from models.conv_comparison import get_model, get_model_info
from config import Config
from utils import set_seed, load_checkpoint
from data import DatasetLoader

class ComparisonTrainer:
    """对比训练器"""
    
    def __init__(self, model_types=['conv1', 'conv2'], epochs=5):
        self.model_types = model_types
        self.epochs = epochs
        self.device = Config.DEVICE
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
    
    def train_model(self, model_type):
        """训练单个模型"""
        print(f"\n🚀 开始训练 {get_model_info(model_type)['name']}")
        print("=" * 60)
        
        # 创建模型
        input_channels = getattr(self, 'input_channels', Config.INPUT_CHANNELS)
        model = get_model(model_type, input_channels, Config.NUM_CLASSES)
        model = model.to(self.device)
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 创建TensorBoard writer
        log_dir = f'./logs/comparison_{model_type}'
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
                    print(f'  Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100. * train_correct / train_total:.2f}%')
            
            # 计算训练指标
            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = 100. * train_correct / train_total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # 测试阶段
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
            
            # 计算测试指标
            avg_test_loss = test_loss / len(self.test_loader)
            test_accuracy = 100. * test_correct / test_total
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
            
            print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
            print(f'  测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%')
        
        # 保存最佳模型
        os.makedirs('./checkpoints/comparison', exist_ok=True)
        checkpoint_path = f'./checkpoints/comparison/{model_type}_best.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'model_type': model_type,
            'best_accuracy': best_accuracy,
            'epochs': self.epochs,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        }, checkpoint_path)
        
        print(f"\n✅ {get_model_info(model_type)['name']} 训练完成!")
        print(f"   最佳测试准确率: {best_accuracy:.2f}%")
        print(f"   模型已保存到: {checkpoint_path}")
        
        # 关闭TensorBoard writer
        writer.close()
        
        # 返回结果
        return {
            'model': model,
            'best_accuracy': best_accuracy,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'checkpoint_path': checkpoint_path
        }
    
    def train_all_models(self):
        """训练所有模型"""
        print("🎯 开始对比训练不同层数的卷积网络")
        print("=" * 80)
        
        for model_type in self.model_types:
            info = get_model_info(model_type)
            print(f"\n📋 模型信息:")
            print(f"   名称: {info['name']}")
            print(f"   描述: {info['description']}")
            print(f"   参数: {info['params']}")
            print(f"   卷积层: {info['layers']}")
            
            # 训练模型
            result = self.train_model(model_type)
            self.results[model_type] = result
        
        return self.results
    
    def plot_comparison(self, save_path=None):
        """绘制对比图表"""
        if not self.results:
            print("❌ 没有训练结果可以绘制")
            return
        
        print("\n📊 绘制对比图表...")
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_type, result) in enumerate(self.results.items()):
            info = get_model_info(model_type)
            color = colors[i % len(colors)]
            epochs = range(1, len(result['train_losses']) + 1)
            
            # 训练和测试损失
            ax1.plot(epochs, result['train_losses'], color=color, linestyle='-', 
                    label=f"{info['name']} (训练)", linewidth=2)
            ax1.plot(epochs, result['test_losses'], color=color, linestyle='--', 
                    label=f"{info['name']} (测试)", linewidth=2)
            
            # 训练和测试准确率
            ax2.plot(epochs, result['train_accuracies'], color=color, linestyle='-', 
                    label=f"{info['name']} (训练)", linewidth=2)
            ax2.plot(epochs, result['test_accuracies'], color=color, linestyle='--', 
                    label=f"{info['name']} (测试)", linewidth=2)
        
        # 设置图表
        ax1.set_title('训练和测试损失对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('训练和测试准确率对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 最终准确率对比
        model_names = [get_model_info(mt)['name'] for mt in self.results.keys()]
        final_accuracies = [result['test_accuracies'][-1] for result in self.results.values()]
        best_accuracies = [result['best_accuracy'] for result in self.results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, final_accuracies, width, label='最终准确率', alpha=0.8)
        ax3.bar(x + width/2, best_accuracies, width, label='最佳准确率', alpha=0.8)
        ax3.set_title('最终准确率对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('模型')
        ax3.set_ylabel('准确率 (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 参数数量对比
        param_counts = []
        for model_type in self.results.keys():
            model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
            param_count = sum(p.numel() for p in model.parameters())
            param_counts.append(param_count)
        
        ax4.bar(model_names, param_counts, alpha=0.8, color=colors[:len(model_names)])
        ax4.set_title('模型参数数量对比', fontsize=14, fontweight='bold')
        ax4.set_xlabel('模型')
        ax4.set_ylabel('参数数量')
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for i, v in enumerate(param_counts):
            ax4.text(i, v + max(param_counts) * 0.01, f'{v:,}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            os.makedirs('./results', exist_ok=True)
            save_path = './results/conv_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比图表已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def print_summary(self):
        """打印训练总结"""
        if not self.results:
            print("❌ 没有训练结果可以总结")
            return
        
        print("\n" + "=" * 80)
        print("📊 训练结果总结")
        print("=" * 80)
        
        for model_type, result in self.results.items():
            info = get_model_info(model_type)
            print(f"\n{info['name']}:")
            print(f"  描述: {info['description']}")
            print(f"  最佳测试准确率: {result['best_accuracy']:.2f}%")
            print(f"  最终测试准确率: {result['test_accuracies'][-1]:.2f}%")
            print(f"  最终训练准确率: {result['train_accuracies'][-1]:.2f}%")
            print(f"  模型文件: {result['checkpoint_path']}")
        
        # 找出最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]['best_accuracy'])
        best_info = get_model_info(best_model[0])
        print(f"\n🏆 最佳模型: {best_info['name']}")
        print(f"   最佳准确率: {best_model[1]['best_accuracy']:.2f}%")

def main():
    """主函数"""
    print("🎯 不同层数卷积网络对比训练")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建训练器
    trainer = ComparisonTrainer(
        model_types=['conv1', 'conv2', 'conv3'],  # 可以调整要训练的模型
        epochs=5  # 可以调整训练轮数
    )
    
    try:
        # 训练所有模型
        results = trainer.train_all_models()
        
        # 绘制对比图表
        trainer.plot_comparison()
        
        # 打印总结
        trainer.print_summary()
        
        print("\n🎉 对比训练完成!")
        print("\n📁 生成的文件:")
        print("- results/conv_comparison.png (对比图表)")
        print("- checkpoints/comparison/ (模型检查点)")
        print("- logs/comparison_*/ (TensorBoard日志)")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
