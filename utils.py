"""
工具函数模块
包含各种辅助功能
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.animation as animation

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"模型已保存到: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    
    print(f"模型已从 {filepath} 加载")
    print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")
    
    return epoch, accuracy

def evaluate_model(model, test_loader, device, class_names):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    # 计算准确率
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    
    return all_preds, all_targets, all_probs, accuracy

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_classification_report(y_true, y_pred, class_names, save_path=None):
    """绘制分类报告"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 提取指标
    metrics = ['precision', 'recall', 'f1-score']
    classes = class_names
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_predictions(model, test_loader, device, class_names, num_samples=16, save_path=None):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    # 选择前num_samples个样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    predicted = predicted[:num_samples]
    probabilities = probabilities[:num_samples]
    
    # 绘制结果
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 显示图像
        if images[i].shape[0] == 1:  # 灰度图
            img = images[i].squeeze().cpu()
            axes[i].imshow(img, cmap='gray')
        else:  # 彩色图
            img = images[i].permute(1, 2, 0).cpu()
            # 反标准化
            img = torch.clamp(img, 0, 1)
            axes[i].imshow(img)
        
        # 设置标题
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        confidence = probabilities[i][predicted[i]].item()
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_hidden_features(model, test_loader, device, num_samples=1000, method='tsne', save_path=None):
    """分析最后一个隐藏层的特征"""
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i * test_loader.batch_size >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # 获取最后一个隐藏层特征
            if hasattr(model, 'get_last_hidden_features'):
                hidden_features = model.get_last_hidden_features(data)
            else:
                # 手动提取特征
                x = torch.relu(model.conv1(data))
                x = torch.max_pool2d(x, 2)
                x = torch.relu(model.conv2(x))
                x = torch.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = torch.relu(model.fc1(x))
                hidden_features = torch.relu(model.fc2(x))
            
            features.extend(hidden_features.cpu().numpy())
            labels.extend(target.cpu().numpy())
    
    features = np.array(features)
    labels = np.array(labels)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Last Hidden Layer Features ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return features_2d, labels

def create_activation_animation(model, test_loader, device, class_names, num_samples=5, save_path=None):
    """创建网络激活情况的动画"""
    model.eval()
    
    # 获取样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 获取各层激活
    activations = {}
    with torch.no_grad():
        for i, img in enumerate(images):
            img_batch = img.unsqueeze(0)
            if hasattr(model, 'get_activations'):
                acts = model.get_activations(img_batch)
                activations[i] = acts
    
    # 创建动画
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    def animate(frame):
        for ax in axes:
            ax.clear()
        
        sample_idx = frame % num_samples
        sample_acts = activations[sample_idx]
        
        # 显示原始图像
        img = images[sample_idx].cpu()
        if img.shape[0] == 1:
            axes[0].imshow(img.squeeze(), cmap='gray')
        else:
            img = img.permute(1, 2, 0)
            axes[0].imshow(img)
        axes[0].set_title(f'Input (Label: {class_names[labels[sample_idx]]})')
        axes[0].axis('off')
        
        # 显示各层激活
        layer_names = ['conv1', 'conv2', 'fc1', 'fc2', 'output']
        for i, layer_name in enumerate(layer_names):
            if layer_name in sample_acts:
                act = sample_acts[layer_name].squeeze().cpu()
                
                if len(act.shape) == 3:  # 卷积层
                    # 显示所有通道的平均激活
                    act_avg = act.mean(dim=0)
                    axes[i+1].imshow(act_avg, cmap='hot')
                else:  # 全连接层
                    # 显示激活值的条形图
                    axes[i+1].bar(range(len(act)), act)
                
                axes[i+1].set_title(f'{layer_name.upper()}')
                axes[i+1].axis('off')
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=num_samples*10, 
                                 interval=500, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
    
    plt.tight_layout()
    plt.show()
    
    return anim
