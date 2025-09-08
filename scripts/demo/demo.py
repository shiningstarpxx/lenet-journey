"""
LeNet-5 演示脚本
展示项目的主要功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from models import LeNet5
from config import Config
from utils import set_seed
from data import DatasetLoader
from visualization import (
    visualize_conv_filters, visualize_conv_activations_for_digit,
    visualize_conv_channels_detailed, create_activation_animation_enhanced,
    create_feature_evolution_animation, create_comprehensive_visualization
)

def create_sample_images():
    """创建一些示例图片用于演示"""
    print("创建示例图片...")
    
    # 创建示例目录
    sample_dir = './sample_images'
    os.makedirs(sample_dir, exist_ok=True)
    
    # 创建一些手写数字图片
    for digit in range(10):
        # 创建28x28的白色背景图片
        img = Image.new('L', (28, 28), 255)
        draw = ImageDraw.Draw(img)
        
        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制数字
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (28 - text_width) // 2
        y = (28 - text_height) // 2
        
        draw.text((x, y), text, fill=0, font=font)
        
        # 保存图片
        img.save(os.path.join(sample_dir, f'digit_{digit}.png'))
    
    print(f"示例图片已保存到 {sample_dir}")
    return sample_dir

def demo_data_loading():
    """演示数据加载功能"""
    print("\n" + "="*50)
    print("演示1: 数据加载和可视化")
    print("="*50)
    
    # 创建数据加载器
    data_loader = DatasetLoader(
        dataset_name=Config.DATASET,
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    
    # 获取数据集信息
    info = data_loader.get_dataset_info()
    print(f"数据集: {info['dataset_name']}")
    print(f"训练样本: {info['train_samples']}")
    print(f"测试样本: {info['test_samples']}")
    print(f"类别数: {info['num_classes']}")
    print(f"类别名称: {info['class_names']}")
    print(f"输入形状: {info['input_shape']}")
    
    # 可视化样本
    print("\n可视化数据样本...")
    data_loader.visualize_samples(num_samples=8)

def demo_model_architecture():
    """演示模型架构"""
    print("\n" + "="*50)
    print("演示2: 模型架构分析")
    print("="*50)
    
    # 创建模型
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型架构: LeNet-5")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"输入通道数: {Config.INPUT_CHANNELS}")
    print(f"输出类别数: {Config.NUM_CLASSES}")
    
    # 打印网络结构
    print("\n网络结构:")
    print(model)
    
    # 测试前向传播
    print("\n测试前向传播...")
    model.eval()
    with torch.no_grad():
        # 创建随机输入
        dummy_input = torch.randn(1, Config.INPUT_CHANNELS, 28, 28)
        output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出概率: {torch.softmax(output, dim=1).numpy()}")

def demo_activation_analysis():
    """演示激活分析功能"""
    print("\n" + "="*50)
    print("演示3: 网络激活分析")
    print("="*50)
    
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
        batch_size=1
    )
    train_loader, _ = data_loader.create_dataloaders()
    
    # 获取一个样本
    data_iter = iter(train_loader)
    image, label = next(data_iter)
    
    print(f"样本标签: {label.item()}")
    
    # 获取各层激活
    with torch.no_grad():
        activations = model.get_activations(image)
    
    # 分析激活
    print("\n各层激活分析:")
    for layer_name, activation in activations.items():
        print(f"{layer_name}: 形状={activation.shape}, "
              f"均值={activation.mean().item():.4f}, "
              f"标准差={activation.std().item():.4f}, "
              f"最大值={activation.max().item():.4f}, "
              f"最小值={activation.min().item():.4f}")
    
    # 可视化激活
    _visualize_activations(activations, image)

def _visualize_activations(activations, original_image):
    """可视化激活"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 显示原始图像
    img = original_image.squeeze().numpy()
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示各层激活
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    for i, layer_name in enumerate(layer_names):
        if layer_name in activations:
            act = activations[layer_name].squeeze().numpy()
            
            if len(act.shape) == 3:  # 卷积层
                # 显示所有通道的平均激活
                act_avg = act.mean(axis=0)
                axes[i+1].imshow(act_avg, cmap='hot')
            else:  # 全连接层
                # 显示激活值的条形图
                axes[i+1].bar(range(len(act)), act)
            
            axes[i+1].set_title(f'{layer_name.upper()} Activations')
            axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

def demo_enhanced_visualization():
    """演示增强版可视化功能"""
    print("\n" + "="*50)
    print("演示4: 增强版可视化功能")
    print("="*50)
    
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
    class_names = data_loader.get_class_names()
    
    print("1. 可视化卷积滤波器...")
    visualize_conv_filters(model, 'conv1')
    visualize_conv_filters(model, 'conv2')
    
    print("\n2. 可视化数字7的多个样本...")
    visualize_conv_activations_for_digit(model, train_loader, Config.DEVICE, target_digit=7)
    
    print("\n3. 详细可视化卷积通道...")
    visualize_conv_channels_detailed(model, train_loader, Config.DEVICE, target_digit=7)
    
    print("\n4. 创建增强版激活动画...")
    anim1 = create_activation_animation_enhanced(model, train_loader, Config.DEVICE, class_names)
    
    print("\n5. 创建特征演化动画...")
    anim2 = create_feature_evolution_animation(model, train_loader, Config.DEVICE, class_names, target_digit=7)
    
    return anim1, anim2

def demo_inference():
    """演示推理功能"""
    print("\n" + "="*50)
    print("演示5: 模型推理")
    print("="*50)
    
    # 创建模型
    model = LeNet5(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    model.eval()
    
    # 创建示例图片
    sample_dir = create_sample_images()
    
    print(f"\n使用随机初始化的模型进行推理演示...")
    print("(注意: 这是随机权重，实际使用时请加载训练好的模型)")
    
    # 对每个示例图片进行推理
    for digit in range(10):
        image_path = os.path.join(sample_dir, f'digit_{digit}.png')
        
        # 加载和预处理图片
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = (image_tensor - 0.1307) / 0.3081  # 标准化
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        # 推理
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        print(f"数字 {digit}: 预测={predicted.item()}, 置信度={confidence.item():.4f}")

def main():
    """主演示函数"""
    print("LeNet-5 深度学习入门项目演示")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    try:
        # 演示各个功能
        demo_data_loading()
        demo_model_architecture()
        demo_activation_analysis()
        demo_enhanced_visualization()
        demo_inference()
        
        print("\n" + "="*60)
        print("演示完成!")
        print("="*60)
        print("\n接下来你可以:")
        print("1. 运行 'python train.py' 开始训练模型")
        print("2. 运行 'python evaluate.py' 评估模型")
        print("3. 运行 'python inference.py --image your_image.jpg' 进行推理")
        print("4. 查看 README.md 了解更多功能")
        print("5. 运行 'python enhanced_demo.py' 查看增强版可视化")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")

if __name__ == '__main__':
    main()
