#!/usr/bin/env python3
"""
不同层数卷积网络的对比模型
用于比较1层Conv和2层Conv的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1Layer(nn.Module):
    """1层卷积网络"""
    def __init__(self, input_channels=1, num_classes=10):
        super(Conv1Layer, self).__init__()
        
        # 1层卷积
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(6 * 14 * 14, 120)
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
        
        return activations

class Conv2Layer(nn.Module):
    """2层卷积网络（简化版LeNet）"""
    def __init__(self, input_channels=1, num_classes=10):
        super(Conv2Layer, self).__init__()
        
        # 2层卷积
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
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
        
        return activations

class Conv3Layer(nn.Module):
    """3层卷积网络"""
    def __init__(self, input_channels=1, num_classes=10):
        super(Conv3Layer, self).__init__()
        
        # 3层卷积
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 2 * 2, 120)
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
        
        return activations

def get_model(model_type, input_channels=1, num_classes=10):
    """获取指定类型的模型"""
    if model_type == 'conv1':
        return Conv1Layer(input_channels, num_classes)
    elif model_type == 'conv2':
        return Conv2Layer(input_channels, num_classes)
    elif model_type == 'conv3':
        return Conv3Layer(input_channels, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_info(model_type):
    """获取模型信息"""
    info = {
        'conv1': {
            'name': '1层卷积网络',
            'description': 'Conv1 -> Pool -> FC',
            'params': '约152K参数',
            'layers': ['conv1']
        },
        'conv2': {
            'name': '2层卷积网络',
            'description': 'Conv1 -> Pool -> Conv2 -> Pool -> FC',
            'params': '约62K参数',
            'layers': ['conv1', 'conv2']
        },
        'conv3': {
            'name': '3层卷积网络',
            'description': 'Conv1 -> Pool -> Conv2 -> Pool -> Conv3 -> Pool -> FC',
            'params': '约34K参数',
            'layers': ['conv1', 'conv2', 'conv3']
        }
    }
    return info.get(model_type, {})

if __name__ == '__main__':
    # 测试模型创建
    print("测试不同层数的卷积网络...")
    
    for model_type in ['conv1', 'conv2', 'conv3']:
        model = get_model(model_type)
        info = get_model_info(model_type)
        
        print(f"\n{info['name']}:")
        print(f"  描述: {info['description']}")
        print(f"  参数: {info['params']}")
        print(f"  卷积层: {info['layers']}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        print(f"  输出形状: {output.shape}")
        
        # 测试激活获取
        activations = model.get_activations(x)
        print(f"  激活层: {list(activations.keys())}")
