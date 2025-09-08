"""
LeNet-5 模型评估脚本
包含完整的模型评估、可视化和分析功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from models import LeNet5, LeNet5_CIFAR
from data import DatasetLoader
from config import Config
from utils import (
    set_seed, load_checkpoint, evaluate_model, plot_confusion_matrix,
    plot_classification_report, visualize_predictions, analyze_hidden_features,
    create_activation_animation
)
from visualization import (
    visualize_conv_filters, visualize_conv_activations_for_digit,
    visualize_conv_channels_detailed, create_activation_animation_enhanced,
    create_feature_evolution_animation, create_comprehensive_visualization
)

class Evaluator:
    """LeNet评估器"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = config.DEVICE
        
        # 初始化模型
        self._init_model()
        
        # 加载预训练模型
        if model_path:
            self.load_model(model_path)
        
        # 初始化数据加载器
        self._init_data()
    
    def _init_model(self):
        """初始化模型"""
        if self.config.DATASET == 'MNIST':
            self.model = LeNet5(
                input_channels=self.config.INPUT_CHANNELS,
                num_classes=self.config.NUM_CLASSES
            )
        else:  # CIFAR10
            self.model = LeNet5_CIFAR(
                input_channels=self.config.INPUT_CHANNELS,
                num_classes=self.config.NUM_CLASSES
            )
        
        self.model = self.model.to(self.device)
    
    def _init_data(self):
        """初始化数据加载器"""
        self.data_loader = DatasetLoader(
            dataset_name=self.config.DATASET,
            data_dir=self.config.DATA_DIR,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS
        )
        
        _, self.test_loader = self.data_loader.create_dataloaders()
        self.class_names = self.data_loader.get_class_names()
    
    def load_model(self, model_path):
        """加载预训练模型"""
        if os.path.exists(model_path):
            load_checkpoint(model_path, self.model)
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"模型文件不存在: {model_path}")
            print("将使用随机初始化的模型进行评估")
    
    def evaluate(self):
        """完整评估流程"""
        print("开始模型评估...")
        print(f"数据集: {self.config.DATASET}")
        print(f"测试样本数: {len(self.test_loader.dataset)}")
        print("-" * 50)
        
        # 基本评估
        predictions, targets, probabilities, accuracy = evaluate_model(
            self.model, self.test_loader, self.device, self.class_names
        )
        
        print(f"测试准确率: {accuracy:.2f}%")
        
        # 创建结果目录
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        
        # 1. 混淆矩阵
        print("\n生成混淆矩阵...")
        plot_confusion_matrix(
            targets, predictions, self.class_names,
            save_path=os.path.join(self.config.RESULTS_DIR, 'confusion_matrix.png')
        )
        
        # 2. 分类报告
        print("生成分类报告...")
        plot_classification_report(
            targets, predictions, self.class_names,
            save_path=os.path.join(self.config.RESULTS_DIR, 'classification_report.png')
        )
        
        # 3. 预测结果可视化
        print("生成预测结果可视化...")
        visualize_predictions(
            self.model, self.test_loader, self.device, self.class_names,
            save_path=os.path.join(self.config.RESULTS_DIR, 'predictions.png')
        )
        
        # 4. 隐藏层特征分析
        print("分析隐藏层特征...")
        features_2d, labels = analyze_hidden_features(
            self.model, self.test_loader, self.device,
            save_path=os.path.join(self.config.RESULTS_DIR, 'hidden_features_tsne.png')
        )
        
        # 5. 网络激活动画
        if self.config.SHOW_ACTIVATIONS:
            print("生成网络激活动画...")
            anim = create_activation_animation(
                self.model, self.test_loader, self.device, self.class_names,
                save_path=os.path.join(self.config.RESULTS_DIR, 'activation_animation.gif')
            )
            
            # 6. 增强版可视化
            print("生成增强版可视化...")
            self._generate_enhanced_visualizations()
        
        # 6. 详细分类报告
        print("\n详细分类报告:")
        report = classification_report(targets, predictions, target_names=self.class_names)
        print(report)
        
        # 保存报告到文件
        with open(os.path.join(self.config.RESULTS_DIR, 'detailed_report.txt'), 'w') as f:
            f.write(f"LeNet-5 模型评估报告\n")
            f.write(f"数据集: {self.config.DATASET}\n")
            f.write(f"测试准确率: {accuracy:.2f}%\n")
            f.write(f"测试样本数: {len(self.test_loader.dataset)}\n")
            f.write("\n详细分类报告:\n")
            f.write(report)
        
        print(f"\n评估完成! 结果已保存到 {self.config.RESULTS_DIR}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'features_2d': features_2d,
            'labels': labels
        }
    
    def _generate_enhanced_visualizations(self):
        """生成增强版可视化"""
        try:
            # 可视化卷积滤波器
            visualize_conv_filters(
                self.model, 'conv1',
                os.path.join(self.config.RESULTS_DIR, 'conv1_filters.png')
            )
            visualize_conv_filters(
                self.model, 'conv2',
                os.path.join(self.config.RESULTS_DIR, 'conv2_filters.png')
            )
            
            # 可视化数字7的多个样本
            visualize_conv_activations_for_digit(
                self.model, self.test_loader, self.device, target_digit=7,
                save_path=os.path.join(self.config.RESULTS_DIR, 'digit7_multiple_samples.png')
            )
            
            # 详细可视化卷积通道
            visualize_conv_channels_detailed(
                self.model, self.test_loader, self.device, target_digit=7,
                save_path=os.path.join(self.config.RESULTS_DIR, 'digit7_detailed_channels.png')
            )
            
            # 创建增强版激活动画
            create_activation_animation_enhanced(
                self.model, self.test_loader, self.device, self.class_names,
                save_path=os.path.join(self.config.RESULTS_DIR, 'enhanced_activation_animation.gif')
            )
            
            # 创建特征演化动画
            create_feature_evolution_animation(
                self.model, self.test_loader, self.device, self.class_names,
                target_digit=7,
                save_path=os.path.join(self.config.RESULTS_DIR, 'feature_evolution_animation.gif')
            )
            
            print("✅ 增强版可视化生成完成")
            
        except Exception as e:
            print(f"⚠️ 增强版可视化生成失败: {e}")
    
    def analyze_misclassifications(self, num_samples=20):
        """分析错误分类的样本"""
        self.model.eval()
        
        misclassified_samples = []
        correct_samples = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                # 找出错误分类的样本
                misclassified_mask = predicted != target
                correct_mask = predicted == target
                
                if misclassified_mask.any():
                    misclassified_indices = torch.where(misclassified_mask)[0]
                    for idx in misclassified_indices:
                        if len(misclassified_samples) < num_samples:
                            misclassified_samples.append({
                                'image': data[idx].cpu(),
                                'true_label': target[idx].item(),
                                'predicted_label': predicted[idx].item(),
                                'confidence': torch.softmax(output[idx], dim=0)[predicted[idx]].item()
                            })
                
                if correct_mask.any():
                    correct_indices = torch.where(correct_mask)[0]
                    for idx in correct_indices:
                        if len(correct_samples) < num_samples:
                            correct_samples.append({
                                'image': data[idx].cpu(),
                                'true_label': target[idx].item(),
                                'predicted_label': predicted[idx].item(),
                                'confidence': torch.softmax(output[idx], dim=0)[predicted[idx]].item()
                            })
                
                if len(misclassified_samples) >= num_samples and len(correct_samples) >= num_samples:
                    break
        
        # 可视化错误分类样本
        self._plot_misclassified_samples(misclassified_samples, correct_samples)
        
        return misclassified_samples, correct_samples
    
    def _plot_misclassified_samples(self, misclassified_samples, correct_samples):
        """绘制错误分类样本"""
        fig, axes = plt.subplots(4, 10, figsize=(20, 8))
        
        # 错误分类样本
        for i, sample in enumerate(misclassified_samples[:10]):
            img = sample['image']
            if img.shape[0] == 1:
                axes[0, i].imshow(img.squeeze(), cmap='gray')
            else:
                img = img.permute(1, 2, 0)
                axes[0, i].imshow(img)
            
            true_label = self.class_names[sample['true_label']]
            pred_label = self.class_names[sample['predicted_label']]
            conf = sample['confidence']
            
            axes[0, i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {conf:.3f}', 
                               color='red', fontsize=8)
            axes[0, i].axis('off')
        
        # 正确分类样本
        for i, sample in enumerate(correct_samples[:10]):
            img = sample['image']
            if img.shape[0] == 1:
                axes[1, i].imshow(img.squeeze(), cmap='gray')
            else:
                img = img.permute(1, 2, 0)
                axes[1, i].imshow(img)
            
            true_label = self.class_names[sample['true_label']]
            pred_label = self.class_names[sample['predicted_label']]
            conf = sample['confidence']
            
            axes[1, i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {conf:.3f}', 
                               color='green', fontsize=8)
            axes[1, i].axis('off')
        
        # 隐藏多余的子图
        for i in range(2, 4):
            for j in range(10):
                axes[i, j].axis('off')
        
        plt.suptitle('错误分类样本 (上) vs 正确分类样本 (下)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'misclassified_samples.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 模型路径
    model_path = os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pth')
    
    # 创建评估器
    evaluator = Evaluator(Config, model_path)
    
    # 开始评估
    results = evaluator.evaluate()
    
    # 分析错误分类
    print("\n分析错误分类样本...")
    misclassified, correct = evaluator.analyze_misclassifications()

if __name__ == '__main__':
    main()
