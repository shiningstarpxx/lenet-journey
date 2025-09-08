#!/usr/bin/env python3
"""
设置matplotlib中文字体支持
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import os

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS常见中文字体
        fonts = [
            'PingFang SC',
            'Hiragino Sans GB', 
            'STHeiti',
            'SimHei',
            'Arial Unicode MS',
            'DejaVu Sans'
        ]
    elif system == "Windows":
        # Windows常见中文字体
        fonts = [
            'SimHei',
            'Microsoft YaHei',
            'SimSun',
            'KaiTi',
            'DejaVu Sans'
        ]
    else:  # Linux
        # Linux常见中文字体
        fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Source Han Sans SC',
            'DejaVu Sans'
        ]
    
    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = fonts
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 测试字体是否可用
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, '中文字体测试', fontsize=16, ha='center', va='center')
    ax.set_title('字体测试')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 保存测试图片
    test_path = './results/font_test.png'
    os.makedirs('./results', exist_ok=True)
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 中文字体设置完成")
    print(f"📁 测试图片已保存到: {test_path}")
    print(f"🔤 使用的字体: {matplotlib.rcParams['font.sans-serif']}")
    
    return True

def get_available_fonts():
    """获取系统可用字体"""
    from matplotlib.font_manager import FontManager
    
    fm = FontManager()
    fonts = [f.name for f in fm.ttflist]
    
    # 过滤中文字体
    chinese_fonts = []
    for font in fonts:
        if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'kai', 'song', 'ming']):
            chinese_fonts.append(font)
    
    print("🔍 系统中可用的中文字体:")
    for font in sorted(set(chinese_fonts)):
        print(f"  - {font}")
    
    return chinese_fonts

if __name__ == '__main__':
    print("🔤 设置matplotlib中文字体支持")
    print("=" * 50)
    
    # 获取可用字体
    available_fonts = get_available_fonts()
    
    # 设置字体
    setup_chinese_font()
    
    print("\n💡 如果中文显示仍有问题，请:")
    print("1. 安装中文字体包")
    print("2. 清除matplotlib字体缓存: rm -rf ~/.cache/matplotlib")
    print("3. 重启Python环境")
