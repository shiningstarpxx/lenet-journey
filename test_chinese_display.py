#!/usr/bin/env python3
"""
测试中文字体显示效果
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def test_chinese_display():
    """测试中文字体显示"""
    print("🔤 测试中文字体显示效果")
    print("=" * 50)
    
    # 创建测试数据
    models = ['1层卷积网络', '2层卷积网络', '3层卷积网络']
    accuracies = [98.38, 98.69, 98.79]
    params = [152410, 61706, 33706]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 准确率对比
    bars1 = ax1.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('不同层数卷积网络准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_ylim(98, 99)
    
    # 在柱状图上显示数值
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. 参数数量对比
    bars2 = ax2.bar(models, params, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('不同层数卷积网络参数数量对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('参数数量')
    
    # 在柱状图上显示数值
    for bar, param in zip(bars2, params):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(params)*0.01,
                f'{param:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 训练损失曲线
    epochs = [1, 2, 3]
    conv1_loss = [0.2254, 0.0761, 0.0547]
    conv2_loss = [0.2402, 0.0670, 0.0466]
    
    ax3.plot(epochs, conv1_loss, 'o-', label='1层卷积网络', linewidth=2, markersize=8)
    ax3.plot(epochs, conv2_loss, 's-', label='2层卷积网络', linewidth=2, markersize=8)
    ax3.set_title('训练损失变化趋势', fontsize=14, fontweight='bold')
    ax3.set_xlabel('训练轮数')
    ax3.set_ylabel('损失值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 参数效率对比
    efficiency = [acc/param*1000 for acc, param in zip(accuracies, params)]
    bars4 = ax4.bar(models, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('参数效率对比 (准确率/1000参数)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('效率值')
    
    # 在柱状图上显示数值
    for bar, eff in zip(bars4, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(efficiency)*0.01,
                f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('卷积层数对比分析 - 中文字体测试', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    import os
    os.makedirs('./results', exist_ok=True)
    save_path = './results/chinese_font_test.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 中文字体测试图已保存到: {save_path}")
    
    plt.show()
    
    return True

def test_activation_visualization():
    """测试激活可视化中的中文显示"""
    print("\n🎨 测试激活可视化中的中文显示")
    print("=" * 50)
    
    # 创建模拟的激活数据
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 模拟不同模型的激活
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            
            # 生成随机激活数据
            activation = np.random.rand(14, 14)
            
            im = ax.imshow(activation, cmap='hot')
            ax.set_title(f'模型{i+1} - 卷积层{j+1}激活', fontsize=12)
            ax.axis('off')
    
    plt.suptitle('不同模型卷积层激活可视化 - 中文字体测试', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = './results/activation_chinese_test.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 激活可视化中文测试图已保存到: {save_path}")
    
    plt.show()
    
    return True

def main():
    """主函数"""
    print("🔤 中文字体显示测试")
    print("=" * 60)
    
    try:
        # 测试基本中文显示
        test_chinese_display()
        
        # 测试激活可视化中文显示
        test_activation_visualization()
        
        print("\n🎉 中文字体测试完成!")
        print("\n📁 生成的文件:")
        print("- results/chinese_font_test.png")
        print("- results/activation_chinese_test.png")
        
        print("\n💡 如果中文显示正常，说明字体设置成功!")
        print("   如果仍有问题，请检查系统是否安装了中文字体")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
