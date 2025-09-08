#!/usr/bin/env python3
"""
可视化不同层数卷积网络的结构图 - 美化版本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
import matplotlib
from matplotlib.patches import ConnectionPatch

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def draw_conv_block(ax, x, y, width, height, text, color='lightblue'):
    """绘制卷积块"""
    rect = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=10, fontweight='bold')

def draw_pool_block(ax, x, y, width, height, text, color='lightgreen'):
    """绘制池化块"""
    rect = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=10, fontweight='bold')

def draw_fc_block(ax, x, y, width, height, text, color='lightcoral'):
    """绘制全连接块"""
    rect = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=10, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

def visualize_conv1_architecture():
    """可视化1层卷积网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 输入
    draw_conv_block(ax, 0.5, 2.5, 1, 1, 'Input\n28×28×1', 'lightgray')
    
    # Conv1
    draw_conv_block(ax, 2.5, 2.5, 1.5, 1, 'Conv1\n5×5×6', 'lightblue')
    
    # Pool1
    draw_pool_block(ax, 4.5, 2.5, 1.5, 1, 'Pool1\n2×2', 'lightgreen')
    
    # FC layers
    draw_fc_block(ax, 6.5, 3.5, 1.5, 0.8, 'FC1\n120', 'lightcoral')
    draw_fc_block(ax, 6.5, 2.5, 1.5, 0.8, 'FC2\n84', 'lightcoral')
    draw_fc_block(ax, 6.5, 1.5, 1.5, 0.8, 'FC3\n10', 'lightcoral')
    
    # 箭头
    draw_arrow(ax, 1.5, 3, 2.5, 3)
    draw_arrow(ax, 4, 3, 4.5, 3)
    draw_arrow(ax, 6, 3, 6.5, 3.1)
    draw_arrow(ax, 6, 3, 6.5, 2.9)
    draw_arrow(ax, 6, 3, 6.5, 1.9)
    
    # 参数信息
    ax.text(5, 0.5, '参数数量: 152,410\n卷积层: 156\n全连接层: 152,254', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_title('1层卷积网络结构 (Conv1 → Pool → FC)', fontsize=16, fontweight='bold')
    
    return fig

def visualize_conv2_architecture():
    """可视化2层卷积网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 输入
    draw_conv_block(ax, 0.5, 2.5, 1, 1, 'Input\n28×28×1', 'lightgray')
    
    # Conv1
    draw_conv_block(ax, 2, 2.5, 1.5, 1, 'Conv1\n5×5×6', 'lightblue')
    
    # Pool1
    draw_pool_block(ax, 4, 2.5, 1.5, 1, 'Pool1\n2×2', 'lightgreen')
    
    # Conv2
    draw_conv_block(ax, 6, 2.5, 1.5, 1, 'Conv2\n5×5×16', 'lightblue')
    
    # Pool2
    draw_pool_block(ax, 8, 2.5, 1.5, 1, 'Pool2\n2×2', 'lightgreen')
    
    # FC layers
    draw_fc_block(ax, 10, 3.5, 1.5, 0.8, 'FC1\n120', 'lightcoral')
    draw_fc_block(ax, 10, 2.5, 1.5, 0.8, 'FC2\n84', 'lightcoral')
    draw_fc_block(ax, 10, 1.5, 1.5, 0.8, 'FC3\n10', 'lightcoral')
    
    # 箭头
    draw_arrow(ax, 1.5, 3, 2, 3)
    draw_arrow(ax, 3.5, 3, 4, 3)
    draw_arrow(ax, 5.5, 3, 6, 3)
    draw_arrow(ax, 7.5, 3, 8, 3)
    draw_arrow(ax, 9.5, 3, 10, 3.1)
    draw_arrow(ax, 9.5, 3, 10, 2.9)
    draw_arrow(ax, 9.5, 3, 10, 1.9)
    
    # 参数信息
    ax.text(6, 0.5, '参数数量: 61,706\n卷积层: 2,572\n全连接层: 59,134', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_title('2层卷积网络结构 (Conv1 → Pool → Conv2 → Pool → FC)', fontsize=16, fontweight='bold')
    
    return fig

def visualize_conv3_architecture():
    """可视化3层卷积网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 输入
    draw_conv_block(ax, 0.5, 2.5, 1, 1, 'Input\n28×28×1', 'lightgray')
    
    # Conv1
    draw_conv_block(ax, 2, 2.5, 1.5, 1, 'Conv1\n5×5×6', 'lightblue')
    
    # Pool1
    draw_pool_block(ax, 4, 2.5, 1.5, 1, 'Pool1\n2×2', 'lightgreen')
    
    # Conv2
    draw_conv_block(ax, 6, 2.5, 1.5, 1, 'Conv2\n5×5×16', 'lightblue')
    
    # Pool2
    draw_pool_block(ax, 8, 2.5, 1.5, 1, 'Pool2\n2×2', 'lightgreen')
    
    # Conv3
    draw_conv_block(ax, 10, 2.5, 1.5, 1, 'Conv3\n3×3×32', 'lightblue')
    
    # Pool3
    draw_pool_block(ax, 12, 2.5, 1.5, 1, 'Pool3\n2×2', 'lightgreen')
    
    # FC layers
    draw_fc_block(ax, 10, 4.5, 1.5, 0.8, 'FC1\n120', 'lightcoral')
    draw_fc_block(ax, 10, 3.5, 1.5, 0.8, 'FC2\n84', 'lightcoral')
    draw_fc_block(ax, 10, 2.5, 1.5, 0.8, 'FC3\n10', 'lightcoral')
    draw_fc_block(ax, 10, 1.5, 1.5, 0.8, 'FC4\n10', 'lightcoral')
    
    # 箭头
    draw_arrow(ax, 1.5, 3, 2, 3)
    draw_arrow(ax, 3.5, 3, 4, 3)
    draw_arrow(ax, 5.5, 3, 6, 3)
    draw_arrow(ax, 7.5, 3, 8, 3)
    draw_arrow(ax, 9.5, 3, 10, 3)
    draw_arrow(ax, 11.5, 3, 12, 3)
    draw_arrow(ax, 13.5, 3, 10, 4.1)
    draw_arrow(ax, 13.5, 3, 10, 3.9)
    draw_arrow(ax, 13.5, 3, 10, 2.9)
    draw_arrow(ax, 13.5, 3, 10, 1.9)
    
    # 参数信息
    ax.text(7, 0.5, '参数数量: 33,706\n卷积层: 7,212\n全连接层: 26,494', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_title('3层卷积网络结构 (Conv1 → Pool → Conv2 → Pool → Conv3 → Pool → FC)', fontsize=16, fontweight='bold')
    
    return fig

def visualize_comparison():
    """可视化所有模型结构的对比"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    
    # 1层卷积网络
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    
    # 绘制1层网络
    draw_conv_block(ax1, 0.5, 2.5, 1, 1, 'Input\n28×28×1', 'lightgray')
    draw_conv_block(ax1, 2.5, 2.5, 1.5, 1, 'Conv1\n5×5×6', 'lightblue')
    draw_pool_block(ax1, 4.5, 2.5, 1.5, 1, 'Pool1\n2×2', 'lightgreen')
    draw_fc_block(ax1, 6.5, 3.5, 1.5, 0.8, 'FC1\n120', 'lightcoral')
    draw_fc_block(ax1, 6.5, 2.5, 1.5, 0.8, 'FC2\n84', 'lightcoral')
    draw_fc_block(ax1, 6.5, 1.5, 1.5, 0.8, 'FC3\n10', 'lightcoral')
    
    draw_arrow(ax1, 1.5, 3, 2.5, 3)
    draw_arrow(ax1, 4, 3, 4.5, 3)
    draw_arrow(ax1, 6, 3, 6.5, 3.1)
    draw_arrow(ax1, 6, 3, 6.5, 2.9)
    draw_arrow(ax1, 6, 3, 6.5, 1.9)
    
    ax1.text(5, 0.5, '1层卷积网络\n参数: 152,410\n准确率: 98.38%', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax1.set_title('1层卷积网络结构', fontsize=14, fontweight='bold')
    
    # 2层卷积网络
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    
    # 绘制2层网络
    draw_conv_block(ax2, 0.5, 2.5, 1, 1, 'Input\n28×28×1', 'lightgray')
    draw_conv_block(ax2, 2, 2.5, 1.5, 1, 'Conv1\n5×5×6', 'lightblue')
    draw_pool_block(ax2, 3.5, 2.5, 1.5, 1, 'Pool1\n2×2', 'lightgreen')
    draw_conv_block(ax2, 5, 2.5, 1.5, 1, 'Conv2\n5×5×16', 'lightblue')
    draw_pool_block(ax2, 6.5, 2.5, 1.5, 1, 'Pool2\n2×2', 'lightgreen')
    draw_fc_block(ax2, 8, 3.5, 1.5, 0.8, 'FC1\n120', 'lightcoral')
    draw_fc_block(ax2, 8, 2.5, 1.5, 0.8, 'FC2\n84', 'lightcoral')
    draw_fc_block(ax2, 8, 1.5, 1.5, 0.8, 'FC3\n10', 'lightcoral')
    
    draw_arrow(ax2, 1.5, 3, 2, 3)
    draw_arrow(ax2, 3.5, 3, 3.5, 3)
    draw_arrow(ax2, 5, 3, 5, 3)
    draw_arrow(ax2, 6.5, 3, 6.5, 3)
    draw_arrow(ax2, 8, 3, 8, 3.1)
    draw_arrow(ax2, 8, 3, 8, 2.9)
    draw_arrow(ax2, 8, 3, 8, 1.9)
    
    ax2.text(5, 0.5, '2层卷积网络\n参数: 61,706\n准确率: 98.69%', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    ax2.set_title('2层卷积网络结构', fontsize=14, fontweight='bold')
    
    # 3层卷积网络
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    
    # 绘制3层网络
    draw_conv_block(ax3, 0.5, 2.5, 1, 1, 'Input\n28×28×1', 'lightgray')
    draw_conv_block(ax3, 1.8, 2.5, 1.2, 1, 'Conv1\n5×5×6', 'lightblue')
    draw_pool_block(ax3, 3.2, 2.5, 1.2, 1, 'Pool1\n2×2', 'lightgreen')
    draw_conv_block(ax3, 4.6, 2.5, 1.2, 1, 'Conv2\n5×5×16', 'lightblue')
    draw_pool_block(ax3, 6, 2.5, 1.2, 1, 'Pool2\n2×2', 'lightgreen')
    draw_conv_block(ax3, 7.4, 2.5, 1.2, 1, 'Conv3\n3×3×32', 'lightblue')
    draw_pool_block(ax3, 8.8, 2.5, 1.2, 1, 'Pool3\n2×2', 'lightgreen')
    draw_fc_block(ax3, 7.4, 4.2, 1.2, 0.6, 'FC1\n120', 'lightcoral')
    draw_fc_block(ax3, 7.4, 3.6, 1.2, 0.6, 'FC2\n84', 'lightcoral')
    draw_fc_block(ax3, 7.4, 3, 1.2, 0.6, 'FC3\n10', 'lightcoral')
    
    draw_arrow(ax3, 1.5, 3, 1.8, 3)
    draw_arrow(ax3, 3, 3, 3.2, 3)
    draw_arrow(ax3, 4.4, 3, 4.6, 3)
    draw_arrow(ax3, 5.8, 3, 6, 3)
    draw_arrow(ax3, 7.2, 3, 7.4, 3)
    draw_arrow(ax3, 8.6, 3, 8.8, 3)
    draw_arrow(ax3, 10, 3, 7.4, 4.2)
    draw_arrow(ax3, 10, 3, 7.4, 3.6)
    draw_arrow(ax3, 10, 3, 7.4, 3)
    
    ax3.text(5, 0.5, '3层卷积网络\n参数: 33,706\n准确率: 98.50%', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    ax3.set_title('3层卷积网络结构', fontsize=14, fontweight='bold')
    
    plt.suptitle('不同层数卷积网络结构对比', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    """主函数"""
    print("🎨 可视化不同层数卷积网络结构")
    print("=" * 60)
    
    import os
    os.makedirs('./results', exist_ok=True)
    
    try:
        # 1. 绘制1层卷积网络结构
        print("📊 绘制1层卷积网络结构...")
        fig1 = visualize_conv1_architecture()
        fig1.savefig('./results/conv1_architecture.png', dpi=300, bbox_inches='tight')
        print("✅ 1层卷积网络结构图已保存到: ./results/conv1_architecture.png")
        
        # 2. 绘制2层卷积网络结构
        print("📊 绘制2层卷积网络结构...")
        fig2 = visualize_conv2_architecture()
        fig2.savefig('./results/conv2_architecture.png', dpi=300, bbox_inches='tight')
        print("✅ 2层卷积网络结构图已保存到: ./results/conv2_architecture.png")
        
        # 3. 绘制3层卷积网络结构
        print("📊 绘制3层卷积网络结构...")
        fig3 = visualize_conv3_architecture()
        fig3.savefig('./results/conv3_architecture.png', dpi=300, bbox_inches='tight')
        print("✅ 3层卷积网络结构图已保存到: ./results/conv3_architecture.png")
        
        # 4. 绘制对比图
        print("📊 绘制结构对比图...")
        fig4 = visualize_comparison()
        fig4.savefig('./results/conv_architectures_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ 结构对比图已保存到: ./results/conv_architectures_comparison.png")
        
        # 显示图表
        plt.show()
        
        print("\n🎉 所有结构图绘制完成!")
        print("\n📁 生成的文件:")
        print("- results/conv1_architecture.png")
        print("- results/conv2_architecture.png") 
        print("- results/conv3_architecture.png")
        print("- results/conv_architectures_comparison.png")
        
        print("\n💡 结构图说明:")
        print("- 蓝色: 卷积层 (Conv)")
        print("- 绿色: 池化层 (Pool)")
        print("- 红色: 全连接层 (FC)")
        print("- 灰色: 输入层")
        
    except Exception as e:
        print(f"❌ 绘制过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
