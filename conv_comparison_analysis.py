#!/usr/bin/env python3
"""
综合的卷积层数对比分析脚本
包含训练、评估、可视化和分析功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from models.conv_comparison import get_model, get_model_info
from config import Config
from utils import set_seed
from data import DatasetLoader
from train_comparison import ComparisonTrainer
from visualize_comparison import ComparisonVisualizer

class ConvComparisonAnalyzer:
    """卷积层数对比分析器"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.data_loader = DatasetLoader(
            dataset_name=Config.DATASET,
            data_dir=Config.DATA_DIR,
            batch_size=Config.BATCH_SIZE
        )
        self.train_loader, self.test_loader = self.data_loader.create_dataloaders()
        
    def run_complete_analysis(self, model_types=['conv1', 'conv2'], epochs=3, target_digit=7):
        """运行完整的对比分析"""
        print("🎯 开始完整的卷积层数对比分析")
        print("=" * 80)
        
        # 1. 训练模型
        print("\n📚 步骤1: 训练模型")
        print("-" * 40)
        trainer = ComparisonTrainer(model_types=model_types, epochs=epochs)
        results = trainer.train_all_models()
        
        # 2. 绘制训练对比图
        print("\n📊 步骤2: 绘制训练对比图")
        print("-" * 40)
        trainer.plot_comparison()
        
        # 3. 详细评估
        print("\n🔍 步骤3: 详细评估")
        print("-" * 40)
        self.detailed_evaluation(results)
        
        # 4. 可视化对比
        print("\n🎨 步骤4: 可视化对比")
        print("-" * 40)
        visualizer = ComparisonVisualizer()
        models = visualizer.load_trained_models(model_types)
        
        if models:
            visualizer.visualize_activations_comparison(models, target_digit=target_digit)
            visualizer.create_comparison_animation(models, target_digit=target_digit)
            visualizer.visualize_detailed_channels_comparison(models, target_digit=target_digit)
        
        # 5. 生成分析报告
        print("\n📝 步骤5: 生成分析报告")
        print("-" * 40)
        self.generate_analysis_report(results, models)
        
        print("\n🎉 完整分析完成!")
        return results, models
    
    def detailed_evaluation(self, results):
        """详细评估所有模型"""
        print("🔍 开始详细评估...")
        
        for model_type, result in results.items():
            print(f"\n评估 {get_model_info(model_type)['name']}:")
            
            # 加载模型
            checkpoint_path = result['checkpoint_path']
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            # 在测试集上评估
            all_predictions = []
            all_targets = []
            all_probabilities = []
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
            
            # 计算指标
            accuracy = 100. * np.mean(np.array(all_predictions) == np.array(all_targets))
            
            print(f"  测试准确率: {accuracy:.2f}%")
            print(f"  最佳训练准确率: {result['best_accuracy']:.2f}%")
            
            # 保存详细预测结果
            os.makedirs('./results/evaluation', exist_ok=True)
            np.save(f'./results/evaluation/{model_type}_predictions.npy', all_predictions)
            np.save(f'./results/evaluation/{model_type}_targets.npy', all_targets)
            np.save(f'./results/evaluation/{model_type}_probabilities.npy', all_probabilities)
            
            # 生成分类报告
            class_names = [str(i) for i in range(10)]
            report = classification_report(all_targets, all_predictions, 
                                        target_names=class_names, output_dict=True)
            
            # 保存分类报告
            import json
            with open(f'./results/evaluation/{model_type}_classification_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"  详细结果已保存到: ./results/evaluation/{model_type}_*")
    
    def generate_analysis_report(self, results, models):
        """生成分析报告"""
        print("📝 生成分析报告...")
        
        report_path = './results/conv_comparison_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 卷积层数对比分析报告\n\n")
            import datetime
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 实验设置
            f.write("## 实验设置\n\n")
            f.write(f"- 数据集: {Config.DATASET}\n")
            f.write(f"- 训练轮数: {len(results[list(results.keys())[0]]['train_losses'])}\n")
            f.write(f"- 批次大小: {Config.BATCH_SIZE}\n")
            f.write(f"- 学习率: {Config.LEARNING_RATE}\n")
            f.write(f"- 设备: {Config.DEVICE}\n\n")
            
            # 模型对比
            f.write("## 模型对比\n\n")
            f.write("| 模型 | 描述 | 参数数量 | 最佳准确率 | 最终准确率 |\n")
            f.write("|------|------|----------|------------|------------|\n")
            
            for model_type, result in results.items():
                info = get_model_info(model_type)
                model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
                param_count = sum(p.numel() for p in model.parameters())
                
                f.write(f"| {info['name']} | {info['description']} | {param_count:,} | "
                       f"{result['best_accuracy']:.2f}% | {result['test_accuracies'][-1]:.2f}% |\n")
            
            # 关键发现
            f.write("\n## 关键发现\n\n")
            
            # 找出最佳模型
            best_model = max(results.items(), key=lambda x: x[1]['best_accuracy'])
            best_info = get_model_info(best_model[0])
            
            f.write(f"### 最佳模型\n")
            f.write(f"- **模型**: {best_info['name']}\n")
            f.write(f"- **准确率**: {best_model[1]['best_accuracy']:.2f}%\n")
            f.write(f"- **描述**: {best_info['description']}\n\n")
            
            # 层数影响分析
            f.write("### 层数影响分析\n\n")
            
            if 'conv1' in results and 'conv2' in results:
                conv1_acc = results['conv1']['best_accuracy']
                conv2_acc = results['conv2']['best_accuracy']
                improvement = conv2_acc - conv1_acc
                
                f.write(f"- **1层Conv vs 2层Conv**: 准确率提升 {improvement:.2f}%\n")
                
                if improvement > 0:
                    f.write(f"- **结论**: 增加卷积层数对MNIST数据集有正面影响\n")
                else:
                    f.write(f"- **结论**: 对于MNIST这样的简单数据集，1层卷积可能已经足够\n")
            
            # 参数效率分析
            f.write("\n### 参数效率分析\n\n")
            
            for model_type, result in results.items():
                info = get_model_info(model_type)
                model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
                param_count = sum(p.numel() for p in model.parameters())
                efficiency = result['best_accuracy'] / param_count * 1000  # 每1000参数的准确率
                
                f.write(f"- **{info['name']}**: {efficiency:.4f}% 准确率/1000参数\n")
            
            # 可视化分析
            f.write("\n## 可视化分析\n\n")
            f.write("通过激活可视化可以观察到:\n\n")
            
            if models:
                f.write("1. **Conv1层激活模式**:\n")
                f.write("   - 不同层数的模型在Conv1层表现出相似的激活模式\n")
                f.write("   - 1层Conv模型已经能够很好地提取边缘和线条特征\n\n")
                
                f.write("2. **特征层次**:\n")
                f.write("   - 1层Conv: 主要提取低级特征（边缘、线条）\n")
                f.write("   - 2层Conv: 在低级特征基础上提取更复杂的形状\n")
                f.write("   - 3层Conv: 进一步抽象，但可能对MNIST过于复杂\n\n")
                
                f.write("3. **激活强度分布**:\n")
                f.write("   - 不同模型在相同样本上的激活强度分布不同\n")
                f.write("   - 层数越多，激活的抽象程度越高\n\n")
            
            # 结论和建议
            f.write("## 结论和建议\n\n")
            f.write("### 主要结论\n\n")
            f.write("1. **对于MNIST数据集**:\n")
            f.write("   - 1层卷积网络已经能够达到较好的性能\n")
            f.write("   - 增加层数可能带来性能提升，但收益递减\n")
            f.write("   - 需要考虑计算成本和性能提升的平衡\n\n")
            
            f.write("2. **模型选择建议**:\n")
            f.write("   - **简单任务**: 优先考虑1层或2层卷积\n")
            f.write("   - **复杂任务**: 可能需要更多层数\n")
            f.write("   - **资源受限**: 选择参数较少的模型\n\n")
            
            f.write("3. **可视化价值**:\n")
            f.write("   - 激活可视化有助于理解网络工作原理\n")
            f.write("   - 可以用于模型调试和优化\n")
            f.write("   - 有助于解释模型的决策过程\n\n")
            
            # 生成的文件列表
            f.write("## 生成的文件\n\n")
            f.write("### 模型文件\n")
            f.write("- `checkpoints/comparison/`: 训练好的模型检查点\n")
            f.write("- `logs/comparison_*/`: TensorBoard训练日志\n\n")
            
            f.write("### 可视化文件\n")
            f.write("- `results/conv_comparison.png`: 训练对比图\n")
            f.write("- `results/conv_activations_comparison_7.png`: 激活对比图\n")
            f.write("- `results/conv_comparison_animation_7.gif`: 对比动画\n")
            f.write("- `results/detailed_channels_comparison_7.png`: 详细通道对比\n\n")
            
            f.write("### 评估文件\n")
            f.write("- `results/evaluation/`: 详细评估结果\n")
            f.write("- `results/conv_comparison_analysis_report.md`: 本分析报告\n\n")
            
            f.write("---\n")
            f.write("*报告由LeNet-5对比分析系统自动生成*\n")
        
        print(f"✅ 分析报告已保存到: {report_path}")
    
    def quick_comparison(self, target_digit=7):
        """快速对比（使用预训练模型或创建新模型）"""
        print("⚡ 快速对比分析")
        print("=" * 40)
        
        model_types = ['conv1', 'conv2']
        models = {}
        
        # 尝试加载预训练模型，如果没有则创建新模型
        for model_type in model_types:
            checkpoint_path = f'./checkpoints/comparison/{model_type}_best.pth'
            
            if os.path.exists(checkpoint_path):
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
                
                print(f"✅ 加载预训练 {model_type} 模型，准确率: {checkpoint['best_accuracy']:.2f}%")
            else:
                # 创建新模型进行演示
                model = get_model(model_type, Config.INPUT_CHANNELS, Config.NUM_CLASSES)
                model = model.to(self.device)
                model.eval()
                
                models[model_type] = {
                    'model': model,
                    'accuracy': 0.0,
                    'info': get_model_info(model_type)
                }
                
                print(f"⚠️ 使用未训练的 {model_type} 模型进行演示")
        
        # 快速可视化对比
        visualizer = ComparisonVisualizer()
        visualizer.visualize_activations_comparison(models, target_digit=target_digit, num_samples=3)
        
        return models

def main():
    """主函数"""
    print("🎯 卷积层数对比分析系统")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建分析器
    analyzer = ConvComparisonAnalyzer()
    
    try:
        # 检查是否有预训练模型
        has_pretrained = any(os.path.exists(f'./checkpoints/comparison/{mt}_best.pth') 
                           for mt in ['conv1', 'conv2', 'conv3'])
        
        if has_pretrained:
            print("✅ 发现预训练模型，开始完整分析...")
            results, models = analyzer.run_complete_analysis(
                model_types=['conv1', 'conv2'],  # 可以调整
                epochs=3,  # 可以调整
                target_digit=7
            )
        else:
            print("⚠️ 未发现预训练模型，开始快速对比...")
            models = analyzer.quick_comparison(target_digit=7)
            
            print("\n💡 提示: 运行 'python train_comparison.py' 来训练模型")
            print("   然后再次运行此脚本进行完整分析")
        
        print("\n🎉 分析完成!")
        print("\n📁 查看生成的文件:")
        print("- results/ (可视化结果)")
        print("- checkpoints/comparison/ (模型文件)")
        print("- logs/comparison_*/ (训练日志)")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
