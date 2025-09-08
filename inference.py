"""
LeNet-5 推理脚本
用于单张图片预测和批量预测
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from models import LeNet5, LeNet5_CIFAR
from config import Config
from utils import set_seed, load_checkpoint

class LeNetInference:
    """LeNet推理器"""
    
    def __init__(self, config, model_path):
        self.config = config
        self.device = config.DEVICE
        
        # 初始化模型
        self._init_model()
        
        # 加载模型
        self.load_model(model_path)
        
        # 获取类别名称
        self.class_names = self._get_class_names()
    
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
    
    def _get_class_names(self):
        """获取类别名称"""
        if self.config.DATASET == 'MNIST':
            return [str(i) for i in range(10)]
        elif self.config.DATASET == 'CIFAR10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    def load_model(self, model_path):
        """加载预训练模型"""
        if os.path.exists(model_path):
            load_checkpoint(model_path, self.model)
            print(f"模型已从 {model_path} 加载")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        # 加载图片
        image = Image.open(image_path)
        
        # 转换为RGB（如果是彩色图片）
        if image.mode != 'RGB' and self.config.DATASET == 'CIFAR10':
            image = image.convert('RGB')
        elif image.mode != 'L' and self.config.DATASET == 'MNIST':
            image = image.convert('L')
        
        # 调整尺寸
        if self.config.DATASET == 'MNIST':
            image = image.resize((28, 28))
        else:  # CIFAR10
            image = image.resize((32, 32))
        
        # 转换为tensor
        image_array = np.array(image)
        
        if self.config.DATASET == 'MNIST':
            # 灰度图
            image_tensor = torch.from_numpy(image_array).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # 添加通道维度
            # 标准化
            image_tensor = (image_tensor - 0.1307) / 0.3081
        else:  # CIFAR10
            # 彩色图
            image_tensor = torch.from_numpy(image_array).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
            # 标准化
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        
        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, image_array
    
    def predict_single(self, image_path, show_confidence=True):
        """预测单张图片"""
        # 预处理
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        # 显示结果
        if show_confidence:
            print(f"预测结果: {self.class_names[predicted_class]}")
            print(f"置信度: {confidence_score:.4f}")
            print(f"所有类别概率:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  {self.class_names[i]}: {prob:.4f}")
        
        # 可视化
        self._visualize_prediction(original_image, predicted_class, confidence_score, probabilities[0])
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence_score,
            'all_probabilities': probabilities[0].cpu().numpy()
        }
    
    def _visualize_prediction(self, original_image, predicted_class, confidence, probabilities):
        """可视化预测结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示原图
        if len(original_image.shape) == 2:  # 灰度图
            ax1.imshow(original_image, cmap='gray')
        else:  # 彩色图
            ax1.imshow(original_image)
        
        ax1.set_title(f'输入图像\n预测: {self.class_names[predicted_class]} (置信度: {confidence:.4f})')
        ax1.axis('off')
        
        # 显示概率分布
        bars = ax2.bar(range(len(self.class_names)), probabilities.cpu().numpy())
        bars[predicted_class].set_color('red')
        ax2.set_xlabel('类别')
        ax2.set_ylabel('概率')
        ax2.set_title('预测概率分布')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        # 添加数值标签
        for i, prob in enumerate(probabilities):
            ax2.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, image_paths, save_results=False):
        """批量预测"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, show_confidence=False)
                result['image_path'] = image_path
                results.append(result)
                print(f"{os.path.basename(image_path)}: {result['predicted_label']} ({result['confidence']:.4f})")
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        if save_results:
            self._save_batch_results(results)
        
        return results
    
    def _save_batch_results(self, results):
        """保存批量预测结果"""
        results_dir = os.path.join(self.config.RESULTS_DIR, 'batch_predictions')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存文本结果
        with open(os.path.join(results_dir, 'predictions.txt'), 'w', encoding='utf-8') as f:
            f.write("批量预测结果\n")
            f.write("=" * 50 + "\n")
            
            for result in results:
                if 'error' in result:
                    f.write(f"{result['image_path']}: 错误 - {result['error']}\n")
                else:
                    f.write(f"{result['image_path']}: {result['predicted_label']} (置信度: {result['confidence']:.4f})\n")
        
        print(f"批量预测结果已保存到 {results_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LeNet-5 推理脚本')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--batch', type=str, nargs='+', help='批量图片路径')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型文件名')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10'], 
                       default='MNIST', help='数据集类型')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 更新配置
    Config.DATASET = args.dataset
    if args.dataset == 'CIFAR10':
        Config.INPUT_CHANNELS = 3
    
    # 模型路径
    model_path = os.path.join(Config.MODEL_SAVE_DIR, args.model)
    
    # 创建推理器
    try:
        inference = LeNetInference(Config, model_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先训练模型或检查模型路径")
        return
    
    # 执行推理
    if args.image:
        print(f"预测单张图片: {args.image}")
        inference.predict_single(args.image)
    elif args.batch:
        print(f"批量预测 {len(args.batch)} 张图片")
        inference.predict_batch(args.batch, save_results=True)
    else:
        print("请提供 --image 或 --batch 参数")
        print("示例:")
        print("  python inference.py --image path/to/image.jpg")
        print("  python inference.py --batch path/to/image1.jpg path/to/image2.jpg")
        print("  python inference.py --image path/to/image.jpg --dataset CIFAR10")

if __name__ == '__main__':
    main()
