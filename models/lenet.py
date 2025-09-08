"""
LeNet-5 网络架构实现
经典的卷积神经网络，用于图像分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 网络架构
    
    网络结构:
    Input (1x28x28) -> Conv1 (6x5x5) -> ReLU -> MaxPool2d (2x2)
    -> Conv2 (16x5x5) -> ReLU -> MaxPool2d (2x2)
    -> Flatten -> FC1 (120) -> ReLU -> FC2 (84) -> ReLU -> FC3 (10)
    """
    
    def __init__(self, input_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 对于28x28输入
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout层（可选，用于正则化）
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一层卷积 + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二层卷积 + 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_activations(self, x):
        """
        获取各层的激活值，用于可视化分析
        """
        activations = {}
        
        # 第一层卷积 + 池化
        x = F.relu(self.conv1(x))
        activations['conv1'] = x.clone()
        x = F.max_pool2d(x, 2)
        activations['pool1'] = x.clone()
        
        # 第二层卷积 + 池化
        x = F.relu(self.conv2(x))
        activations['conv2'] = x.clone()
        x = F.max_pool2d(x, 2)
        activations['pool2'] = x.clone()
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        activations['fc1'] = x.clone()
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        activations['fc2'] = x.clone()  # 最后一个隐藏层
        x = self.dropout(x)
        x = self.fc3(x)
        activations['output'] = x.clone()
        
        return activations
    
    def get_last_hidden_features(self, x):
        """
        获取最后一个隐藏层（fc2）的输出特征
        用于分析网络学到的表示
        """
        # 前向传播到最后一个隐藏层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # 最后一个隐藏层
        
        return x

class LeNet5_CIFAR(nn.Module):
    """
    适用于CIFAR-10的LeNet-5变体
    调整了输入尺寸和网络结构
    """
    
    def __init__(self, input_channels=3, num_classes=10):
        super(LeNet5_CIFAR, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # 全连接层 (CIFAR-10: 32x32 -> 5x5 after conv layers)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_activations(self, x):
        """获取各层激活值"""
        activations = {}
        x = F.relu(self.conv1(x))
        activations['conv1'] = x.clone()
        x = F.max_pool2d(x, 2)
        activations['pool1'] = x.clone()
        x = F.relu(self.conv2(x))
        activations['conv2'] = x.clone()
        x = F.max_pool2d(x, 2)
        activations['pool2'] = x.clone()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        activations['fc1'] = x.clone()
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        activations['fc2'] = x.clone()
        x = self.dropout(x)
        x = self.fc3(x)
        activations['output'] = x.clone()
        return activations
    
    def get_last_hidden_features(self, x):
        """获取最后一个隐藏层特征"""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x
