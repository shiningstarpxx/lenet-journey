#!/usr/bin/env python3
"""
可视化不同层数卷积网络的结构图 - 修正FC层连接版本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np
import matplotlib

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义颜色方案
COLORS = {
    'input': '#E8F4FD',
    'conv': '#4A90E2', 
    'pool': '#7ED321',
    'fc': '#F5A623',
    'output': '#D0021B',
    'arrow': '#333333',
    'text': '#2C3E50'
}

def draw_layer_box(ax, x, y, width, height, text, layer_type, params=None):
    """绘制层块"""
    color = COLORS.get(layer_type, '#CCCCCC')
    
    # 创建带阴影的矩形
    shadow = FancyBboxPatch((x+0.05, y-0.05), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor='#CCCCCC',
                           edgecolor='none',
                           alpha=0.3)
    ax.add_patch(shadow)
    
    # 主矩形
    rect = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='white',
                         linewidth=2)
    ax.add_patch(rect)
    
    # 添加渐变效果
    gradient = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             facecolor='white',
                             edgecolor='none',
                             alpha=0.2)
    ax.add_patch(gradient)
    
    # 文本
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', 
            fontsize=11, fontweight='bold',
            color=COLORS['text'])
    
    # 参数信息
    if params:
        ax.text(x + width/2, y - 0.3, params, 
                ha='center', va='center', 
                fontsize=9, color=COLORS['text'],
                style='italic')

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    """绘制箭头"""
    # 箭头
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', 
                              lw=2.5, 
                              color=COLORS['arrow'],
                              shrinkA=5, shrinkB=5))
    
    # 箭头上的文本
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, text, 
                ha='center', va='center', 
                fontsize=8, color=COLORS['arrow'],
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

def visualize_conv1_architecture():
    """可视化1层卷积网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(6, 7.5, '1层卷积网络结构', 
            ha='center', va='center', 
            fontsize=20, fontweight='bold', color=COLORS['text'])
    
    # 输入层
    draw_layer_box(ax, 1, 5, 1.5, 1, 'Input\n28×28×1', 'input')
    
    # Conv1
    draw_layer_box(ax, 3.5, 5, 1.5, 1, 'Conv1\n5×5×6', 'conv', '156 params')
    
    # Pool1
    draw_layer_box(ax, 6, 5, 1.5, 1, 'Pool1\n2×2', 'pool')
    
    # FC layers - 串行连接
    draw_layer_box(ax, 8.5, 5, 1.5, 1, 'FC1\n120', 'fc', '120 params')
    draw_layer_box(ax, 8.5, 3.5, 1.5, 1, 'FC2\n84', 'fc', '84 params')
    draw_layer_box(ax, 8.5, 2, 1.5, 1, 'FC3\n10', 'output', '10 params')
    
    # 箭头 - 串行连接
    draw_arrow(ax, 2.5, 5.5, 3.5, 5.5)
    draw_arrow(ax, 5, 5.5, 6, 5.5)
    draw_arrow(ax, 7.5, 5.5, 8.5, 5.5)
    draw_arrow(ax, 8.5, 4.5, 8.5, 4.5)  # FC1 -> FC2
    draw_arrow(ax, 8.5, 3, 8.5, 3)      # FC2 -> FC3
    
    # 统计信息
    info_text = """模型统计:
• 总参数: 152,410
• 卷积层: 156 (0.1%)
• 全连接层: 152,254 (99.9%)
• 测试准确率: 98.38%"""
    
    ax.text(6, 0.5, info_text, 
            ha='center', va='center', 
            fontsize=12, color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', 
                     edgecolor='#DEE2E6', linewidth=1))
    
    return fig

def visualize_conv2_architecture():
    """可视化2层卷积网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(7, 7.5, '2层卷积网络结构 (LeNet-5)', 
            ha='center', va='center', 
            fontsize=20, fontweight='bold', color=COLORS['text'])
    
    # 输入层
    draw_layer_box(ax, 1, 5, 1.5, 1, 'Input\n28×28×1', 'input')
    
    # Conv1
    draw_layer_box(ax, 3.5, 5, 1.5, 1, 'Conv1\n5×5×6', 'conv', '156 params')
    
    # Pool1
    draw_layer_box(ax, 6, 5, 1.5, 1, 'Pool1\n2×2', 'pool')
    
    # Conv2
    draw_layer_box(ax, 8.5, 5, 1.5, 1, 'Conv2\n5×5×16', 'conv', '2,416 params')
    
    # Pool2
    draw_layer_box(ax, 11, 5, 1.5, 1, 'Pool2\n2×2', 'pool')
    
    # FC layers - 串行连接
    draw_layer_box(ax, 8.5, 3, 1.5, 1, 'FC1\n120', 'fc', '48,120 params')
    draw_layer_box(ax, 8.5, 1.5, 1.5, 1, 'FC2\n84', 'fc', '10,164 params')
    draw_layer_box(ax, 8.5, 0, 1.5, 1, 'FC3\n10', 'output', '850 params')
    
    # 箭头
    draw_arrow(ax, 2.5, 5.5, 3.5, 5.5)
    draw_arrow(ax, 5, 5.5, 6, 5.5)
    draw_arrow(ax, 7.5, 5.5, 8.5, 5.5)
    draw_arrow(ax, 10, 5.5, 11, 5.5)
    draw_arrow(ax, 12.5, 5.5, 8.5, 3.5)  # Pool2 -> FC1
    draw_arrow(ax, 8.5, 2.5, 8.5, 2.5)   # FC1 -> FC2
    draw_arrow(ax, 8.5, 1, 8.5, 1)       # FC2 -> FC3
    
    # 统计信息
    info_text = """模型统计:
• 总参数: 61,706
• 卷积层: 2,572 (4.2%)
• 全连接层: 59,134 (95.8%)
• 测试准确率: 98.69%"""
    
    ax.text(7, -1, info_text, 
            ha='center', va='center', 
            fontsize=12, color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', 
                     edgecolor='#DEE2E6', linewidth=1))
    
    return fig

def visualize_conv3_architecture():
    """可视化3层卷积网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(8, 7.5, '3层卷积网络结构', 
            ha='center', va='center', 
            fontsize=20, fontweight='bold', color=COLORS['text'])
    
    # 输入层
    draw_layer_box(ax, 1, 5, 1.5, 1, 'Input\n28×28×1', 'input')
    
    # Conv1
    draw_layer_box(ax, 3.5, 5, 1.5, 1, 'Conv1\n5×5×6', 'conv', '156 params')
    
    # Pool1
    draw_layer_box(ax, 6, 5, 1.5, 1, 'Pool1\n2×2', 'pool')
    
    # Conv2
    draw_layer_box(ax, 8.5, 5, 1.5, 1, 'Conv2\n5×5×16', 'conv', '2,416 params')
    
    # Pool2
    draw_layer_box(ax, 11, 5, 1.5, 1, 'Pool2\n2×2', 'pool')
    
    # Conv3
    draw_layer_box(ax, 13.5, 5, 1.5, 1, 'Conv3\n3×3×32', 'conv', '4,640 params')
    
    # Pool3
    draw_layer_box(ax, 11, 3, 1.5, 1, 'Pool3\n2×2', 'pool')
    
    # FC layers - 串行连接
    draw_layer_box(ax, 8.5, 3, 1.5, 1, 'FC1\n120', 'fc', '15,480 params')
    draw_layer_box(ax, 8.5, 1.5, 1.5, 1, 'FC2\n84', 'fc', '10,164 params')
    draw_layer_box(ax, 8.5, 0, 1.5, 1, 'FC3\n10', 'output', '850 params')
    
    # 箭头
    draw_arrow(ax, 2.5, 5.5, 3.5, 5.5)
    draw_arrow(ax, 5, 5.5, 6, 5.5)
    draw_arrow(ax, 7.5, 5.5, 8.5, 5.5)
    draw_arrow(ax, 10, 5.5, 11, 5.5)
    draw_arrow(ax, 12.5, 5.5, 13.5, 5.5)
    draw_arrow(ax, 15, 5.5, 11, 3.5)     # Conv3 -> Pool3
    draw_arrow(ax, 12.5, 3.5, 8.5, 3.5)  # Pool3 -> FC1
    draw_arrow(ax, 8.5, 2.5, 8.5, 2.5)   # FC1 -> FC2
    draw_arrow(ax, 8.5, 1, 8.5, 1)       # FC2 -> FC3
    
    # 统计信息
    info_text = """模型统计:
• 总参数: 33,706
• 卷积层: 7,212 (21.4%)
• 全连接层: 26,494 (78.6%)
• 测试准确率: 98.50%"""
    
    ax.text(8, -1, info_text, 
            ha='center', va='center', 
            fontsize=12, color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', 
                     edgecolor='#DEE2E6', linewidth=1))
    
    return fig

def visualize_comparison():
    """可视化所有模型结构的对比"""
    fig, axes = plt.subplots(3, 1, figsize=(18, 20))
    
    # 1层卷积网络
    ax1 = axes[0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    # 标题
    ax1.text(6, 7.5, '1层卷积网络', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # 绘制1层网络
    draw_layer_box(ax1, 1, 5, 1.5, 1, 'Input\n28×28×1', 'input')
    draw_layer_box(ax1, 3.5, 5, 1.5, 1, 'Conv1\n5×5×6', 'conv')
    draw_layer_box(ax1, 6, 5, 1.5, 1, 'Pool1\n2×2', 'pool')
    draw_layer_box(ax1, 8.5, 5, 1.5, 1, 'FC1\n120', 'fc')
    draw_layer_box(ax1, 8.5, 3.5, 1.5, 1, 'FC2\n84', 'fc')
    draw_layer_box(ax1, 8.5, 2, 1.5, 1, 'FC3\n10', 'output')
    
    draw_arrow(ax1, 2.5, 5.5, 3.5, 5.5)
    draw_arrow(ax1, 5, 5.5, 6, 5.5)
    draw_arrow(ax1, 7.5, 5.5, 8.5, 5.5)
    draw_arrow(ax1, 8.5, 4.5, 8.5, 4.5)
    draw_arrow(ax1, 8.5, 3, 8.5, 3)
    
    # 性能指标
    ax1.text(6, 1, '准确率: 98.38% | 参数: 152,410 | 效率: 0.65%', 
            ha='center', va='center', 
            fontsize=14, fontweight='bold', color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E3F2FD', 
                     edgecolor='#4A90E2', linewidth=2))
    
    # 2层卷积网络
    ax2 = axes[1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    # 标题
    ax2.text(6, 7.5, '2层卷积网络 (LeNet-5)', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # 绘制2层网络
    draw_layer_box(ax2, 1, 5, 1.5, 1, 'Input\n28×28×1', 'input')
    draw_layer_box(ax2, 3, 5, 1.5, 1, 'Conv1\n5×5×6', 'conv')
    draw_layer_box(ax2, 5, 5, 1.5, 1, 'Pool1\n2×2', 'pool')
    draw_layer_box(ax2, 7, 5, 1.5, 1, 'Conv2\n5×5×16', 'conv')
    draw_layer_box(ax2, 9, 5, 1.5, 1, 'Pool2\n2×2', 'pool')
    draw_layer_box(ax2, 7, 3.5, 1.5, 1, 'FC1\n120', 'fc')
    draw_layer_box(ax2, 7, 2, 1.5, 1, 'FC2\n84', 'fc')
    draw_layer_box(ax2, 7, 0.5, 1.5, 1, 'FC3\n10', 'output')
    
    draw_arrow(ax2, 2.5, 5.5, 3, 5.5)
    draw_arrow(ax2, 4.5, 5.5, 5, 5.5)
    draw_arrow(ax2, 6.5, 5.5, 7, 5.5)
    draw_arrow(ax2, 8.5, 5.5, 9, 5.5)
    draw_arrow(ax2, 10.5, 5.5, 7, 4)
    draw_arrow(ax2, 7, 3, 7, 3)
    draw_arrow(ax2, 7, 1.5, 7, 1.5)
    
    # 性能指标
    ax2.text(6, -0.5, '准确率: 98.69% | 参数: 61,706 | 效率: 1.60%', 
            ha='center', va='center', 
            fontsize=14, fontweight='bold', color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', 
                     edgecolor='#7ED321', linewidth=2))
    
    # 3层卷积网络
    ax3 = axes[2]
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    
    # 标题
    ax3.text(6, 7.5, '3层卷积网络', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # 绘制3层网络
    draw_layer_box(ax3, 1, 5, 1.5, 1, 'Input\n28×28×1', 'input')
    draw_layer_box(ax3, 2.8, 5, 1.2, 1, 'Conv1\n5×5×6', 'conv')
    draw_layer_box(ax3, 4.6, 5, 1.2, 1, 'Pool1\n2×2', 'pool')
    draw_layer_box(ax3, 6.4, 5, 1.2, 1, 'Conv2\n5×5×16', 'conv')
    draw_layer_box(ax3, 8.2, 5, 1.2, 1, 'Pool2\n2×2', 'pool')
    draw_layer_box(ax3, 10, 5, 1.2, 1, 'Conv3\n3×3×32', 'conv')
    draw_layer_box(ax3, 8.2, 3.5, 1.2, 1, 'Pool3\n2×2', 'pool')
    draw_layer_box(ax3, 6.4, 3.5, 1.2, 1, 'FC1\n120', 'fc')
    draw_layer_box(ax3, 6.4, 2, 1.2, 1, 'FC2\n84', 'fc')
    draw_layer_box(ax3, 6.4, 0.5, 1.2, 1, 'FC3\n10', 'output')
    
    draw_arrow(ax3, 2.5, 5.5, 2.8, 5.5)
    draw_arrow(ax3, 4, 5.5, 4.6, 5.5)
    draw_arrow(ax3, 5.8, 5.5, 6.4, 5.5)
    draw_arrow(ax3, 7.6, 5.5, 8.2, 5.5)
    draw_arrow(ax3, 9.4, 5.5, 10, 5.5)
    draw_arrow(ax3, 11.2, 5.5, 8.2, 4)
    draw_arrow(ax3, 9.4, 4, 6.4, 4)
    draw_arrow(ax3, 6.4, 3, 6.4, 3)
    draw_arrow(ax3, 6.4, 1.5, 6.4, 1.5)
    
    # 性能指标
    ax3.text(6, -0.5, '准确率: 98.50% | 参数: 33,706 | 效率: 2.92%', 
            ha='center', va='center', 
            fontsize=14, fontweight='bold', color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF3E0', 
                     edgecolor='#F5A623', linewidth=2))
    
    plt.suptitle('不同层数卷积网络结构对比 - 串行FC连接', fontsize=24, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    return fig

def main():
    """主函数"""
    print("🎨 可视化不同层数卷积网络结构 - 修正FC连接版本")
    print("=" * 60)
    
    import os
    os.makedirs('./results', exist_ok=True)
    
    try:
        # 1. 绘制1层卷积网络结构
        print("📊 绘制1层卷积网络结构...")
        fig1 = visualize_conv1_architecture()
        fig1.savefig('./results/conv1_architecture_v3.png', dpi=300, bbox_inches='tight')
        print("✅ 1层卷积网络结构图已保存到: ./results/conv1_architecture_v3.png")
        
        # 2. 绘制2层卷积网络结构
        print("📊 绘制2层卷积网络结构...")
        fig2 = visualize_conv2_architecture()
        fig2.savefig('./results/conv2_architecture_v3.png', dpi=300, bbox_inches='tight')
        print("✅ 2层卷积网络结构图已保存到: ./results/conv2_architecture_v3.png")
        
        # 3. 绘制3层卷积网络结构
        print("📊 绘制3层卷积网络结构...")
        fig3 = visualize_conv3_architecture()
        fig3.savefig('./results/conv3_architecture_v3.png', dpi=300, bbox_inches='tight')
        print("✅ 3层卷积网络结构图已保存到: ./results/conv3_architecture_v3.png")
        
        # 4. 绘制对比图
        print("📊 绘制结构对比图...")
        fig4 = visualize_comparison()
        fig4.savefig('./results/conv_architectures_comparison_v3.png', dpi=300, bbox_inches='tight')
        print("✅ 结构对比图已保存到: ./results/conv_architectures_comparison_v3.png")
        
        # 显示图表
        plt.show()
        
        print("\n🎉 所有修正FC连接的结构图绘制完成!")
        print("\n📁 生成的文件:")
        print("- results/conv1_architecture_v3.png")
        print("- results/conv2_architecture_v3.png") 
        print("- results/conv3_architecture_v3.png")
        print("- results/conv_architectures_comparison_v3.png")
        
        print("\n💡 修正内容:")
        print("- FC层改为串行连接")
        print("- 数据流向更加清晰")
        print("- 符合实际网络结构")
        
    except Exception as e:
        print(f"❌ 绘制过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
