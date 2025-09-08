"""
LeNet-5 训练脚本
包含完整的训练循环、验证和模型保存功能
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import LeNet5, LeNet5_CIFAR
from data import DatasetLoader
from config import Config
from utils import set_seed, save_checkpoint, load_checkpoint

class Trainer:
    """LeNet训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # 创建目录
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化数据加载器
        self._init_data()
        
        # 初始化优化器和损失函数
        self._init_optimizer()
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
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
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"模型已初始化，参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _init_data(self):
        """初始化数据加载器"""
        self.data_loader = DatasetLoader(
            dataset_name=self.config.DATASET,
            data_dir=self.config.DATA_DIR,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS
        )
        
        self.train_loader, self.val_loader = self.data_loader.create_dataloaders()
        self.class_names = self.data_loader.get_class_names()
        
        print(f"数据集: {self.config.DATASET}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
    
    def _init_optimizer(self):
        """初始化优化器"""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """完整训练过程"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"学习率: {self.config.LEARNING_RATE}")
        print(f"批次大小: {self.config.BATCH_SIZE}")
        print("-" * 50)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            print(f'Epoch {epoch+1:2d}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc,
                    os.path.join(self.config.MODEL_SAVE_DIR, 'best_model.pth')
                )
                print(f'新的最佳验证准确率: {val_acc:.2f}%')
        
        # 训练完成
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"训练完成! 总用时: {total_time:.2f}秒")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        # 保存最终模型
        save_checkpoint(
            self.model, self.optimizer, self.config.EPOCHS-1, best_val_acc,
            os.path.join(self.config.MODEL_SAVE_DIR, 'final_model.pth')
        )
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        self.writer.close()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(os.path.join(self.config.RESULTS_DIR, 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """主函数"""
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建训练器
    trainer = Trainer(Config)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()
