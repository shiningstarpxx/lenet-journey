#!/usr/bin/env python3
"""
自适应不同层数卷积网络的对比模型
支持不同输入尺寸和通道数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveConv1Layer(nn.Module):
    """1层卷积网络 - 自适应输入尺寸"""
    def __init__(self, input_channels=1, num_classes=10, input_size=28):
        super(AdaptiveConv1Layer, self).__init__()
        
        # 1层卷积
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        # 卷积后尺寸: (input_size + 2*padding - kernel_size) / stride + 1
        # 池化后尺寸: (conv_size - kernel_size) / stride + 1
        conv_size = (input_size + 2*2 - 5) // 1 + 1  # padding=2, kernel=5, stride=1
        pool_size = (conv_size - 2) // 2 + 1  # kernel=2, stride=2
        fc_input_size = 6 * pool_size * pool_size
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 用于存储激活
        self.activations = {}
        
    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.pool1(F.relu(self.conv1(x)))
        self.activations['conv1'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_activations(self, x):
        """获取中间层激活"""
        activations = {}
        
        # Conv1 + ReLU + Pool
        conv1_out = F.relu(self.conv1(x))
        activations['conv1_before_pool'] = conv1_out
        activations['conv1'] = self.pool1(conv1_out)
        
        # Flatten
        x_flat = activations['conv1'].view(activations['conv1'].size(0), -1)
        
        # 全连接层
        activations['fc1'] = F.relu(self.fc1(x_flat))
        activations['fc2'] = F.relu(self.fc2(activations['fc1']))
        activations['fc3'] = self.fc3(activations['fc2'])
        
        return activations

class AdaptiveConv2Layer(nn.Module):
    """2层卷积网络 - 自适应输入尺寸"""
    def __init__(self, input_channels=1, num_classes=10, input_size=28):
        super(AdaptiveConv2Layer, self).__init__()
        
        # 1层卷积
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2层卷积
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        conv1_size = (input_size + 2*2 - 5) // 1 + 1
        pool1_size = (conv1_size - 2) // 2 + 1
        conv2_size = (pool1_size + 2*2 - 5) // 1 + 1
        pool2_size = (conv2_size - 2) // 2 + 1
        fc_input_size = 16 * pool2_size * pool2_size
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 用于存储激活
        self.activations = {}
        
    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.pool1(F.relu(self.conv1(x)))
        self.activations['conv1'] = x
        
        # Conv2 + ReLU + Pool
        x = self.pool2(F.relu(self.conv2(x)))
        self.activations['conv2'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_activations(self, x):
        """获取中间层激活"""
        activations = {}
        
        # Conv1 + ReLU + Pool
        conv1_out = F.relu(self.conv1(x))
        activations['conv1_before_pool'] = conv1_out
        activations['conv1'] = self.pool1(conv1_out)
        
        # Conv2 + ReLU + Pool
        conv2_out = F.relu(self.conv2(activations['conv1']))
        activations['conv2_before_pool'] = conv2_out
        activations['conv2'] = self.pool2(conv2_out)
        
        # Flatten
        x_flat = activations['conv2'].view(activations['conv2'].size(0), -1)
        
        # 全连接层
        activations['fc1'] = F.relu(self.fc1(x_flat))
        activations['fc2'] = F.relu(self.fc2(activations['fc1']))
        activations['fc3'] = self.fc3(activations['fc2'])
        
        return activations

class AdaptiveConv3Layer(nn.Module):
    """3层卷积网络 - 自适应输入尺寸"""
    def __init__(self, input_channels=1, num_classes=10, input_size=28):
        super(AdaptiveConv3Layer, self).__init__()
        
        # 1层卷积
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2层卷积
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3层卷积
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        conv1_size = (input_size + 2*2 - 5) // 1 + 1
        pool1_size = (conv1_size - 2) // 2 + 1
        conv2_size = (pool1_size + 2*2 - 5) // 1 + 1
        pool2_size = (conv2_size - 2) // 2 + 1
        conv3_size = (pool2_size + 2*2 - 5) // 1 + 1
        pool3_size = (conv3_size - 2) // 2 + 1
        fc_input_size = 32 * pool3_size * pool3_size
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 用于存储激活
        self.activations = {}
        
    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.pool1(F.relu(self.conv1(x)))
        self.activations['conv1'] = x
        
        # Conv2 + ReLU + Pool
        x = self.pool2(F.relu(self.conv2(x)))
        self.activations['conv2'] = x
        
        # Conv3 + ReLU + Pool
        x = self.pool3(F.relu(self.conv3(x)))
        self.activations['conv3'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_activations(self, x):
        """获取中间层激活"""
        activations = {}
        
        # Conv1 + ReLU + Pool
        conv1_out = F.relu(self.conv1(x))
        activations['conv1_before_pool'] = conv1_out
        activations['conv1'] = self.pool1(conv1_out)
        
        # Conv2 + ReLU + Pool
        conv2_out = F.relu(self.conv2(activations['conv1']))
        activations['conv2_before_pool'] = conv2_out
        activations['conv2'] = self.pool2(conv2_out)
        
        # Conv3 + ReLU + Pool
        conv3_out = F.relu(self.conv3(activations['conv2']))
        activations['conv3_before_pool'] = conv3_out
        activations['conv3'] = self.pool3(conv3_out)
        
        # Flatten
        x_flat = activations['conv3'].view(activations['conv3'].size(0), -1)
        
        # 全连接层
        activations['fc1'] = F.relu(self.fc1(x_flat))
        activations['fc2'] = F.relu(self.fc2(activations['fc1']))
        activations['fc3'] = self.fc3(activations['fc2'])
        
        return activations

def get_adaptive_model(model_type, input_channels=1, num_classes=10, input_size=28):
    """获取指定类型的自适应模型"""
    if model_type == 'conv1':
        return AdaptiveConv1Layer(input_channels, num_classes, input_size)
    elif model_type == 'conv2':
        return AdaptiveConv2Layer(input_channels, num_classes, input_size)
    elif model_type == 'conv3':
        return AdaptiveConv3Layer(input_channels, num_classes, input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_adaptive_model_info(model_type):
    """获取自适应模型信息"""
    info = {
        'conv1': {
            'name': '1层卷积网络(自适应)',
            'description': 'Conv1 -> Pool -> FC (支持不同输入尺寸)',
            'params': '约152K参数',
            'conv_layers': ['conv1']
        },
        'conv2': {
            'name': '2层卷积网络(自适应)',
            'description': 'Conv1 -> Pool -> Conv2 -> Pool -> FC (支持不同输入尺寸)',
            'params': '约62K参数',
            'conv_layers': ['conv1', 'conv2']
        },
        'conv3': {
            'name': '3层卷积网络(自适应)',
            'description': 'Conv1 -> Pool -> Conv2 -> Pool -> Conv3 -> Pool -> FC (支持不同输入尺寸)',
            'params': '约34K参数',
            'conv_layers': ['conv1', 'conv2', 'conv3']
        }
    }
    return info.get(model_type, {})

if __name__ == '__main__':
    # 测试自适应模型创建
    print("测试自适应卷积网络...")
    
    for model_type in ['conv1', 'conv2', 'conv3']:
        # 测试MNIST (28x28, 1通道)
        model_mnist = get_adaptive_model(model_type, input_channels=1, input_size=28)
        info = get_adaptive_model_info(model_type)
        
        print(f"\n{info['name']} (MNIST):")
        print(f"  描述: {info['description']}")
        print(f"  参数: {info['params']}")
        print(f"  卷积层: {info['layers']}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model_mnist.parameters())
        print(f"  总参数: {total_params:,}")
        
        # 测试前向传播
        x_mnist = torch.randn(1, 1, 28, 28)
        output_mnist = model_mnist(x_mnist)
        print(f"  MNIST输出形状: {output_mnist.shape}")
        
        # 测试CIFAR-10 (32x32, 3通道)
        model_cifar = get_adaptive_model(model_type, input_channels=3, input_size=32)
        x_cifar = torch.randn(1, 3, 32, 32)
        output_cifar = model_cifar(x_cifar)
        print(f"  CIFAR-10输出形状: {output_cifar.shape}")
        
        # 测试激活获取
        activations = model_cifar.get_activations(x_cifar)
        print(f"  激活层: {list(activations.keys())}")
