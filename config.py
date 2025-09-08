"""
LeNet-5 配置文件
包含所有超参数和配置选项
"""

import torch

class Config:
    # 数据配置
    DATASET = 'MNIST'  # 可选: 'MNIST', 'CIFAR10'
    DATA_DIR = './data'
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    
    # 模型配置
    INPUT_CHANNELS = 1  # MNIST: 1, CIFAR10: 3
    NUM_CLASSES = 10
    
    # 训练配置
    EPOCHS = 20
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存配置
    MODEL_SAVE_DIR = './models'
    LOG_DIR = './logs'
    RESULTS_DIR = './results'
    
    # 可视化配置
    SAVE_PLOTS = True
    SHOW_ACTIVATIONS = True
    ANIMATION_FPS = 2
    
    # 随机种子
    RANDOM_SEED = 42
    
    def __init__(self):
        # 根据数据集调整配置
        if self.DATASET == 'CIFAR10':
            self.INPUT_CHANNELS = 3
            self.LEARNING_RATE = 0.01  # CIFAR10需要更大的学习率
