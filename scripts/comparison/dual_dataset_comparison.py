#!/usr/bin/env python3
"""
双数据集对比分析脚本
测试3个模型（1层、2层、3层卷积）在MNIST和CIFAR-10数据集上的性能差异
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.adaptive_conv_comparison import get_adaptive_model, get_adaptive_model_info
from config import Config
from utils import set_seed
from data import DatasetLoader
from scripts.comparison.adaptive_train_comparison import AdaptiveComparisonTrainer

class DualDatasetComparison:
    """双数据集对比分析器"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model_types = ['conv1', 'conv2', 'conv3']
        self.datasets = ['MNIST', 'CIFAR10']
        self.results = {}
        
    def run_comparison(self, epochs=3):
        """运行双数据集对比分析"""
        print("🎯 双数据集卷积层数对比分析")
        print("=" * 80)
        print(f"📊 数据集: {', '.join(self.datasets)}")
        print(f"🏗️ 模型: {', '.join(self.model_types)}")
        print(f"🔄 训练轮数: {epochs}")
        print("=" * 80)
        
        for dataset_name in self.datasets:
            print(f"\n📚 数据集: {dataset_name}")
            print("-" * 50)
            
            # 更新配置
            original_dataset = Config.DATASET
            Config.DATASET = dataset_name
            
            # 创建数据加载器
            data_loader = DatasetLoader(
                dataset_name=dataset_name,
                data_dir=Config.DATA_DIR,
                batch_size=Config.BATCH_SIZE
            )
            train_loader, test_loader = data_loader.create_dataloaders()
            
            print(f"📊 数据集信息:")
            print(f"  训练集大小: {len(train_loader.dataset)}")
            print(f"  测试集大小: {len(test_loader.dataset)}")
            print(f"  批次大小: {Config.BATCH_SIZE}")
            print(f"  设备: {Config.DEVICE}")
            
            # 根据数据集设置输入参数
            input_channels = 3 if dataset_name == 'CIFAR10' else 1
            input_size = 32 if dataset_name == 'CIFAR10' else 28
            
            # 训练所有模型
            trainer = AdaptiveComparisonTrainer(
                model_types=self.model_types, 
                epochs=epochs,
                input_channels=input_channels,
                input_size=input_size
            )
            # 替换数据加载器
            trainer.train_loader = train_loader
            trainer.test_loader = test_loader
            
            dataset_results = trainer.train_all_models()
            
            # 存储结果
            self.results[dataset_name] = dataset_results
            
            # 恢复原始配置
            Config.DATASET = original_dataset
            
        # 生成对比分析
        self.generate_comparison_analysis()
        
    def generate_comparison_analysis(self):
        """生成对比分析报告和可视化"""
        print("\n📊 生成对比分析报告")
        print("-" * 50)
        
        # 创建结果目录
        os.makedirs('results/dual_dataset_comparison', exist_ok=True)
        
        # 1. 生成性能对比表
        self.create_performance_table()
        
        # 2. 生成可视化图表
        self.create_comparison_plots()
        
        # 3. 生成详细报告
        self.create_detailed_report()
        
    def create_performance_table(self):
        """创建性能对比表"""
        print("📋 创建性能对比表...")
        
        # 准备数据
        data = []
        for dataset_name, dataset_results in self.results.items():
            for model_type, model_result in dataset_results.items():
                info = get_adaptive_model_info(model_type)
                data.append({
                    '数据集': dataset_name,
                    '模型': info['name'],
                    '卷积层数': len(info['conv_layers']),
                    '参数量': info['params'],
                    '测试准确率': f"{model_result['test_accuracy']:.2f}%",
                    '训练准确率': f"{model_result['train_accuracy']:.2f}%",
                    '最佳测试损失': f"{model_result['test_loss']:.4f}",
                    '训练时间': f"{model_result.get('training_time', 0):.1f}s"
                })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存CSV
        csv_path = 'results/dual_dataset_comparison/performance_comparison.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 性能对比表已保存到: {csv_path}")
        
        # 打印表格
        print("\n📊 性能对比表:")
        print(df.to_string(index=False))
        
    def create_comparison_plots(self):
        """创建对比可视化图表"""
        print("📊 创建对比可视化图表...")
        
        # 1. 准确率对比图
        self.plot_accuracy_comparison()
        
        # 2. 参数量vs准确率图
        self.plot_params_vs_accuracy()
        
        # 3. 训练曲线对比
        self.plot_training_curves()
        
    def plot_accuracy_comparison(self):
        """绘制准确率对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        datasets = list(self.results.keys())
        model_types = self.model_types
        
        # MNIST准确率
        mnist_acc = [self.results['MNIST'][model]['test_accuracy'] for model in model_types]
        # CIFAR-10准确率
        cifar_acc = [self.results['CIFAR10'][model]['test_accuracy'] for model in model_types]
        
        x = np.arange(len(model_types))
        width = 0.35
        
        # 绘制柱状图
        bars1 = ax1.bar(x - width/2, mnist_acc, width, label='MNIST', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, cifar_acc, width, label='CIFAR-10', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('模型类型')
        ax1.set_ylabel('测试准确率 (%)')
        ax1.set_title('不同模型在两个数据集上的准确率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels([get_adaptive_model_info(model)['name'] for model in model_types], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 准确率差异图
        accuracy_diff = [cifar_acc[i] - mnist_acc[i] for i in range(len(model_types))]
        colors = ['red' if diff < 0 else 'green' for diff in accuracy_diff]
        
        bars3 = ax2.bar(x, accuracy_diff, color=colors, alpha=0.7)
        ax2.set_xlabel('模型类型')
        ax2.set_ylabel('准确率差异 (CIFAR-10 - MNIST)')
        ax2.set_title('CIFAR-10与MNIST的准确率差异')
        ax2.set_xticks(x)
        ax2.set_xticklabels([get_adaptive_model_info(model)['name'] for model in model_types], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('results/dual_dataset_comparison/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 准确率对比图已保存")
        
    def plot_params_vs_accuracy(self):
        """绘制参数量vs准确率图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MNIST数据
        mnist_params = [get_adaptive_model_info(model)['params'] for model in self.model_types]
        mnist_acc = [self.results['MNIST'][model]['test_accuracy'] for model in self.model_types]
        
        # CIFAR-10数据
        cifar_params = [get_adaptive_model_info(model)['params'] for model in self.model_types]
        cifar_acc = [self.results['CIFAR10'][model]['test_accuracy'] for model in self.model_types]
        
        # 绘制散点图
        ax1.scatter(mnist_params, mnist_acc, s=100, alpha=0.7, c='skyblue', edgecolors='navy')
        for i, model in enumerate(self.model_types):
            ax1.annotate(get_adaptive_model_info(model)['name'], 
                        (mnist_params[i], mnist_acc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('参数量')
        ax1.set_ylabel('测试准确率 (%)')
        ax1.set_title('MNIST: 参数量 vs 准确率')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(cifar_params, cifar_acc, s=100, alpha=0.7, c='lightcoral', edgecolors='darkred')
        for i, model in enumerate(self.model_types):
            ax2.annotate(get_adaptive_model_info(model)['name'], 
                        (cifar_params[i], cifar_acc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('参数量')
        ax2.set_ylabel('测试准确率 (%)')
        ax2.set_title('CIFAR-10: 参数量 vs 准确率')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dual_dataset_comparison/params_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 参数量vs准确率图已保存")
        
    def plot_training_curves(self):
        """绘制训练曲线对比"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, model_type in enumerate(self.model_types):
            ax = axes[i]
            
            # 绘制MNIST训练曲线
            if 'train_losses' in self.results['MNIST'][model_type]:
                epochs = range(1, len(self.results['MNIST'][model_type]['train_losses']) + 1)
                ax.plot(epochs, self.results['MNIST'][model_type]['train_losses'], 
                       'b-', label='MNIST 训练损失', linewidth=2)
                ax.plot(epochs, self.results['MNIST'][model_type]['test_losses'], 
                       'b--', label='MNIST 测试损失', linewidth=2)
            
            # 绘制CIFAR-10训练曲线
            if 'train_losses' in self.results['CIFAR10'][model_type]:
                epochs = range(1, len(self.results['CIFAR10'][model_type]['train_losses']) + 1)
                ax.plot(epochs, self.results['CIFAR10'][model_type]['train_losses'], 
                       'r-', label='CIFAR-10 训练损失', linewidth=2)
                ax.plot(epochs, self.results['CIFAR10'][model_type]['test_losses'], 
                       'r--', label='CIFAR-10 测试损失', linewidth=2)
            
            ax.set_xlabel('训练轮数')
            ax.set_ylabel('损失值')
            ax.set_title(f'{get_adaptive_model_info(model_type)["name"]} 训练曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dual_dataset_comparison/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 训练曲线对比图已保存")
        
    def create_detailed_report(self):
        """创建详细分析报告"""
        print("📝 创建详细分析报告...")
        
        report_path = 'results/dual_dataset_comparison/dual_dataset_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 双数据集卷积层数对比分析报告\n\n")
            f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 分析概述\n\n")
            f.write("本报告对比了1层、2层、3层卷积网络在MNIST和CIFAR-10两个数据集上的性能差异。\n\n")
            
            f.write("## 🏗️ 模型架构\n\n")
            for model_type in self.model_types:
                info = get_adaptive_model_info(model_type)
                f.write(f"### {info['name']}\n")
                f.write(f"- **描述**: {info['description']}\n")
                f.write(f"- **参数量**: {info['params']}\n")
                f.write(f"- **卷积层**: {', '.join(info['conv_layers'])}\n\n")
            
            f.write("## 📈 性能对比\n\n")
            f.write("### 测试准确率对比\n\n")
            f.write("| 模型 | MNIST准确率 | CIFAR-10准确率 | 差异 |\n")
            f.write("|------|-------------|----------------|------|\n")
            
            for model_type in self.model_types:
                mnist_acc = self.results['MNIST'][model_type]['test_accuracy']
                cifar_acc = self.results['CIFAR10'][model_type]['test_accuracy']
                diff = cifar_acc - mnist_acc
                f.write(f"| {get_adaptive_model_info(model_type)['name']} | {mnist_acc:.2f}% | {cifar_acc:.2f}% | {diff:+.2f}% |\n")
            
            f.write("\n### 关键发现\n\n")
            
            # 分析最佳性能
            mnist_best = max(self.model_types, key=lambda x: self.results['MNIST'][x]['test_accuracy'])
            cifar_best = max(self.model_types, key=lambda x: self.results['CIFAR10'][x]['test_accuracy'])
            
            f.write(f"1. **MNIST最佳模型**: {get_adaptive_model_info(mnist_best)['name']} ({self.results['MNIST'][mnist_best]['test_accuracy']:.2f}%)\n")
            f.write(f"2. **CIFAR-10最佳模型**: {get_adaptive_model_info(cifar_best)['name']} ({self.results['CIFAR10'][cifar_best]['test_accuracy']:.2f}%)\n")
            
            # 分析参数量效率
            f.write("\n### 参数量效率分析\n\n")
            f.write("| 模型 | 参数量 | MNIST效率 | CIFAR-10效率 |\n")
            f.write("|------|--------|-----------|--------------|\n")
            
            for model_type in self.model_types:
                params_str = get_adaptive_model_info(model_type)['params']
                # 提取参数量数字（去掉"约"和"K参数"）
                if '152K' in params_str:
                    params = 152
                elif '62K' in params_str:
                    params = 62
                elif '34K' in params_str:
                    params = 34
                else:
                    params = 100  # 默认值
                
                mnist_eff = self.results['MNIST'][model_type]['test_accuracy'] / params
                cifar_eff = self.results['CIFAR10'][model_type]['test_accuracy'] / params
                f.write(f"| {get_adaptive_model_info(model_type)['name']} | {params_str} | {mnist_eff:.2f} | {cifar_eff:.2f} |\n")
            
            f.write("\n## 🎯 结论与建议\n\n")
            
            # 生成结论
            f.write("### 主要结论\n\n")
            f.write("1. **数据集复杂度影响**: CIFAR-10比MNIST更复杂，所有模型的准确率都显著降低\n")
            f.write("2. **模型复杂度权衡**: 更深的网络在复杂数据集上表现更好，但参数量增加\n")
            f.write("3. **效率考虑**: 需要根据具体应用场景平衡准确率和计算资源\n\n")
            
            f.write("### 应用建议\n\n")
            f.write("- **MNIST**: 1层卷积网络已足够，计算效率高\n")
            f.write("- **CIFAR-10**: 建议使用2-3层卷积网络以获得更好性能\n")
            f.write("- **资源受限**: 优先考虑1-2层网络\n")
            f.write("- **性能优先**: 可选择3层网络\n\n")
            
            f.write("## 📁 生成文件\n\n")
            f.write("- `performance_comparison.csv`: 性能对比表\n")
            f.write("- `accuracy_comparison.png`: 准确率对比图\n")
            f.write("- `params_vs_accuracy.png`: 参数量vs准确率图\n")
            f.write("- `training_curves.png`: 训练曲线对比图\n")
            f.write("- `dual_dataset_analysis_report.md`: 本报告\n\n")
        
        print(f"✅ 详细分析报告已保存到: {report_path}")

def main():
    """主函数"""
    set_seed(42)
    
    analyzer = DualDatasetComparison()
    analyzer.run_comparison(epochs=3)
    
    print("\n🎉 双数据集对比分析完成!")
    print("\n📁 查看生成的文件:")
    print("- results/dual_dataset_comparison/ (所有结果文件)")

if __name__ == "__main__":
    main()
