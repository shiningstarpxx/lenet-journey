"""
增强版可视化模块
包含更丰富的网络激活可视化和动画功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

def visualize_conv_filters(model, layer_name='conv1', save_path=None):
    """可视化卷积层的滤波器"""
    model.eval()
    
    # 获取指定层的权重
    if layer_name == 'conv1':
        conv_layer = model.conv1
    elif layer_name == 'conv2':
        conv_layer = model.conv2
    else:
        raise ValueError("只支持 conv1 和 conv2 层")
    
    # 获取滤波器权重
    filters = conv_layer.weight.data.cpu().numpy()
    
    # 计算子图布局
    num_filters = filters.shape[0]
    cols = min(6, num_filters)
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        
        filter_img = filters[i]
        
        if filter_img.shape[0] == 1:  # 单通道
            axes[row, col].imshow(filter_img[0], cmap='gray')
        else:  # 多通道
            # 对于多通道滤波器，显示每个通道的平均值
            filter_avg = filter_img.mean(axis=0)
            axes[row, col].imshow(filter_avg, cmap='gray')
        
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{layer_name.upper()} 卷积滤波器', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_conv_activations_for_digit(model, test_loader, device, target_digit=7, 
                                       num_samples=6, save_path=None):
    """针对特定数字（如7）可视化多个样本的卷积层激活"""
    model.eval()
    
    # 收集目标数字的样本
    target_samples = []
    target_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < num_samples:
                    target_samples.append(data[i])
                    target_labels.append(label.item())
            
            if len(target_samples) >= num_samples:
                break
    
    if len(target_samples) == 0:
        print(f"未找到数字 {target_digit} 的样本")
        return
    
    # 获取激活
    activations = []
    for sample in target_samples:
        sample_batch = sample.unsqueeze(0).to(device)
        if hasattr(model, 'get_activations'):
            acts = model.get_activations(sample_batch)
            activations.append(acts)
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, num_samples, figure=fig)
    
    for i, (sample, acts) in enumerate(zip(target_samples, activations)):
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, i])
        img = sample.squeeze().cpu().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'样本 {i+1}\n数字 {target_digit}')
        ax_orig.axis('off')
        
        # Conv1 激活
        ax_conv1 = fig.add_subplot(gs[1, i])
        conv1_act = acts['conv1'].squeeze().detach().cpu().numpy()
        # 显示所有通道的平均激活
        conv1_avg = conv1_act.mean(axis=0)
        im1 = ax_conv1.imshow(conv1_avg, cmap='hot')
        ax_conv1.set_title(f'Conv1 激活\n(6通道平均)')
        ax_conv1.axis('off')
        
        # Conv2 激活
        ax_conv2 = fig.add_subplot(gs[2, i])
        conv2_act = acts['conv2'].squeeze().detach().cpu().numpy()
        # 显示所有通道的平均激活
        conv2_avg = conv2_act.mean(axis=0)
        im2 = ax_conv2.imshow(conv2_avg, cmap='hot')
        ax_conv2.set_title(f'Conv2 激活\n(16通道平均)')
        ax_conv2.axis('off')
    
    plt.suptitle(f'数字 {target_digit} 的卷积层激活可视化', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_conv_channels_detailed(model, test_loader, device, target_digit=7, 
                                   sample_idx=0, save_path=None):
    """详细可视化卷积层的每个通道"""
    model.eval()
    
    # 获取目标样本
    target_samples = []
    for data, labels in test_loader:
        for i, label in enumerate(labels):
            if label.item() == target_digit:
                target_samples.append(data[i])
                if len(target_samples) > sample_idx:
                    break
        if len(target_samples) > sample_idx:
            break
    
    if len(target_samples) <= sample_idx:
        print(f"未找到足够的数字 {target_digit} 样本")
        return
    
    sample = target_samples[sample_idx].unsqueeze(0).to(device)
    
    # 获取激活
    with torch.no_grad():
        if hasattr(model, 'get_activations'):
            acts = model.get_activations(sample)
    
    # 创建详细可视化
    fig = plt.figure(figsize=(24, 16))
    
    # 原始图像
    ax_orig = plt.subplot2grid((4, 8), (0, 0), colspan=2)
    img = sample.squeeze().cpu().numpy()
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title(f'原始图像\n数字 {target_digit}')
    ax_orig.axis('off')
    
    # Conv1 各通道
    conv1_act = acts['conv1'].squeeze().detach().cpu().numpy()
    for i in range(6):
        ax = plt.subplot2grid((4, 8), (0, 2+i))
        ax.imshow(conv1_act[i], cmap='hot')
        ax.set_title(f'Conv1-{i+1}')
        ax.axis('off')
    
    # Conv2 各通道
    conv2_act = acts['conv2'].squeeze().detach().cpu().numpy()
    for i in range(16):
        row = 1 + i // 8
        col = i % 8
        ax = plt.subplot2grid((4, 8), (row, col))
        ax.imshow(conv2_act[i], cmap='hot')
        ax.set_title(f'Conv2-{i+1}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'数字 {target_digit} 的详细卷积通道激活', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_activation_animation_enhanced(model, test_loader, device, class_names, 
                                       num_samples=5, save_path=None):
    """增强版激活动画，展示更详细的计算过程"""
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
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    def animate(frame):
        # 清除所有子图
        for ax in fig.axes:
            ax.clear()
        
        sample_idx = frame % num_samples
        sample_acts = activations[sample_idx]
        current_label = labels[sample_idx].item()
        
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, 0])
        img = images[sample_idx].cpu()
        if img.shape[0] == 1:
            ax_orig.imshow(img.squeeze(), cmap='gray')
        else:
            img = img.permute(1, 2, 0)
            ax_orig.imshow(img)
        ax_orig.set_title(f'输入图像\n标签: {class_names[current_label]}', fontsize=12)
        ax_orig.axis('off')
        
        # Conv1 激活
        ax_conv1 = fig.add_subplot(gs[0, 1])
        conv1_act = sample_acts['conv1'].squeeze().cpu().numpy()
        conv1_avg = conv1_act.mean(axis=0)
        im1 = ax_conv1.imshow(conv1_avg, cmap='hot')
        ax_conv1.set_title('Conv1 激活\n(6通道平均)', fontsize=12)
        ax_conv1.axis('off')
        
        # Conv2 激活
        ax_conv2 = fig.add_subplot(gs[0, 2])
        conv2_act = sample_acts['conv2'].squeeze().cpu().numpy()
        conv2_avg = conv2_act.mean(axis=0)
        im2 = ax_conv2.imshow(conv2_avg, cmap='hot')
        ax_conv2.set_title('Conv2 激活\n(16通道平均)', fontsize=12)
        ax_conv2.axis('off')
        
        # 输出概率
        ax_output = fig.add_subplot(gs[0, 3])
        output_act = sample_acts['output'].squeeze().cpu().numpy()
        probs = torch.softmax(torch.tensor(output_act), dim=0).numpy()
        bars = ax_output.bar(range(len(class_names)), probs, color='skyblue')
        bars[current_label].set_color('red')
        ax_output.set_title('输出概率分布', fontsize=12)
        ax_output.set_xlabel('类别')
        ax_output.set_ylabel('概率')
        ax_output.set_xticks(range(len(class_names)))
        ax_output.set_xticklabels(class_names, rotation=45)
        
        # 添加数值标签
        for i, prob in enumerate(probs):
            ax_output.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # FC1 激活
        ax_fc1 = fig.add_subplot(gs[1, :2])
        fc1_act = sample_acts['fc1'].squeeze().cpu().numpy()
        ax_fc1.bar(range(len(fc1_act)), fc1_act, color='lightgreen')
        ax_fc1.set_title('FC1 激活 (120个神经元)', fontsize=12)
        ax_fc1.set_xlabel('神经元索引')
        ax_fc1.set_ylabel('激活值')
        
        # FC2 激活
        ax_fc2 = fig.add_subplot(gs[1, 2:])
        fc2_act = sample_acts['fc2'].squeeze().cpu().numpy()
        ax_fc2.bar(range(len(fc2_act)), fc2_act, color='orange')
        ax_fc2.set_title('FC2 激活 (84个神经元)', fontsize=12)
        ax_fc2.set_xlabel('神经元索引')
        ax_fc2.set_ylabel('激活值')
        
        # 激活统计信息
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        
        # 计算统计信息
        stats_text = f"""
        样本 {sample_idx + 1}/{num_samples} | 标签: {class_names[current_label]}
        
        激活统计:
        Conv1: 均值={conv1_act.mean():.4f}, 最大值={conv1_act.max():.4f}, 标准差={conv1_act.std():.4f}
        Conv2: 均值={conv2_act.mean():.4f}, 最大值={conv2_act.max():.4f}, 标准差={conv2_act.std():.4f}
        FC1: 均值={fc1_act.mean():.4f}, 最大值={fc1_act.max():.4f}, 标准差={fc1_act.std():.4f}
        FC2: 均值={fc2_act.mean():.4f}, 最大值={fc2_act.max():.4f}, 标准差={fc2_act.std():.4f}
        
        预测: {class_names[np.argmax(probs)]} (置信度: {np.max(probs):.4f})
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=num_samples*8, 
                                 interval=800, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1.5)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def create_feature_evolution_animation(model, test_loader, device, class_names, 
                                     target_digit=7, num_samples=4, save_path=None):
    """创建特征演化动画，展示不同样本的特征变化"""
    model.eval()
    
    # 收集目标数字的样本
    target_samples = []
    target_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            for i, label in enumerate(labels):
                if label.item() == target_digit and len(target_samples) < num_samples:
                    target_samples.append(data[i])
                    target_labels.append(label.item())
            
            if len(target_samples) >= num_samples:
                break
    
    if len(target_samples) == 0:
        print(f"未找到数字 {target_digit} 的样本")
        return
    
    # 获取所有样本的激活
    all_activations = []
    for sample in target_samples:
        sample_batch = sample.unsqueeze(0).to(device)
        if hasattr(model, 'get_activations'):
            acts = model.get_activations(sample_batch)
            all_activations.append(acts)
    
    # 创建动画
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig)
    
    def animate(frame):
        # 清除所有子图
        for ax in fig.axes:
            ax.clear()
        
        sample_idx = frame % num_samples
        acts = all_activations[sample_idx]
        sample = target_samples[sample_idx]
        
        # 原始图像
        ax_orig = fig.add_subplot(gs[0, 0])
        img = sample.squeeze().cpu().numpy()
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title(f'样本 {sample_idx + 1}\n数字 {target_digit}')
        ax_orig.axis('off')
        
        # Conv1 激活
        ax_conv1 = fig.add_subplot(gs[0, 1])
        conv1_act = acts['conv1'].squeeze().cpu().numpy()
        conv1_avg = conv1_act.mean(axis=0)
        im1 = ax_conv1.imshow(conv1_avg, cmap='hot')
        ax_conv1.set_title('Conv1 激活')
        ax_conv1.axis('off')
        
        # Conv2 激活
        ax_conv2 = fig.add_subplot(gs[0, 2])
        conv2_act = acts['conv2'].squeeze().cpu().numpy()
        conv2_avg = conv2_act.mean(axis=0)
        im2 = ax_conv2.imshow(conv2_avg, cmap='hot')
        ax_conv2.set_title('Conv2 激活')
        ax_conv2.axis('off')
        
        # 输出概率
        ax_output = fig.add_subplot(gs[0, 3])
        output_act = acts['output'].squeeze().cpu().numpy()
        probs = torch.softmax(torch.tensor(output_act), dim=0).numpy()
        bars = ax_output.bar(range(len(class_names)), probs, color='skyblue')
        bars[target_digit].set_color('red')
        ax_output.set_title('输出概率')
        ax_output.set_xlabel('类别')
        ax_output.set_ylabel('概率')
        ax_output.set_xticks(range(len(class_names)))
        ax_output.set_xticklabels(class_names, rotation=45)
        
        # 特征对比
        ax_compare = fig.add_subplot(gs[1, :])
        
        # 收集所有样本的FC2特征
        all_fc2 = []
        for act in all_activations:
            fc2_act = act['fc2'].squeeze().cpu().numpy()
            all_fc2.append(fc2_act)
        
        # 绘制特征对比
        x = np.arange(len(all_fc2[0]))
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, fc2 in enumerate(all_fc2):
            ax_compare.plot(x, fc2, color=colors[i], label=f'样本 {i+1}', alpha=0.7)
        
        ax_compare.set_title(f'数字 {target_digit} 不同样本的FC2特征对比')
        ax_compare.set_xlabel('特征维度')
        ax_compare.set_ylabel('激活值')
        ax_compare.legend()
        ax_compare.grid(True, alpha=0.3)
        
        # 高亮当前样本
        current_fc2 = all_fc2[sample_idx]
        ax_compare.plot(x, current_fc2, color=colors[sample_idx], linewidth=3, alpha=1.0)
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=num_samples*6, 
                                 interval=1000, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def create_comprehensive_visualization(model, test_loader, device, class_names, 
                                     results_dir='./results'):
    """创建综合可视化，包含所有增强功能"""
    os.makedirs(results_dir, exist_ok=True)
    
    print("🎨 开始创建综合可视化...")
    
    # 1. 可视化卷积滤波器
    print("1. 可视化卷积滤波器...")
    visualize_conv_filters(model, 'conv1', 
                          os.path.join(results_dir, 'conv1_filters.png'))
    visualize_conv_filters(model, 'conv2', 
                          os.path.join(results_dir, 'conv2_filters.png'))
    
    # 2. 可视化数字7的多个样本
    print("2. 可视化数字7的多个样本...")
    visualize_conv_activations_for_digit(model, test_loader, device, target_digit=7,
                                       save_path=os.path.join(results_dir, 'digit7_multiple_samples.png'))
    
    # 3. 详细可视化卷积通道
    print("3. 详细可视化卷积通道...")
    visualize_conv_channels_detailed(model, test_loader, device, target_digit=7,
                                   save_path=os.path.join(results_dir, 'digit7_detailed_channels.png'))
    
    # 4. 创建增强版激活动画
    print("4. 创建增强版激活动画...")
    anim1 = create_activation_animation_enhanced(model, test_loader, device, class_names,
                                               save_path=os.path.join(results_dir, 'enhanced_activation_animation.gif'))
    
    # 5. 创建特征演化动画
    print("5. 创建特征演化动画...")
    anim2 = create_feature_evolution_animation(model, test_loader, device, class_names,
                                             target_digit=7,
                                             save_path=os.path.join(results_dir, 'feature_evolution_animation.gif'))
    
    print(f"✅ 所有可视化已完成，结果保存在 {results_dir}")
    
    return anim1, anim2
