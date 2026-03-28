import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

def plot_professional_tradeoff():
    # --- 1. 数据准备 ---
    # 压缩档位
    comp_levels = np.array(['2x', '4x', '8x', '16x'])
    x = np.arange(len(comp_levels))
    # 对应的准确率数据
    accuracy = np.array([68.41, 65.52, 64.71, 60.23])
    # 对应的 Peak(Token) 指标 (模拟数据，用于柱状图)
    peak_token = np.array([2041, 1515, 1051, 800])

    # --- 2. 风格配置 ---
    plt.style.use('seaborn-v0_8-whitegrid') # 使用干净的白网格背景
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=120)
    
    # 配色方案
    COLOR_LINE = '#004c6d' # 深海蓝 (专业、沉稳)
    COLOR_BAR = '#ffa600'  # 琥珀橙 (明亮、对比度高)

    # --- 3. 绘制次坐标轴 (背景柱状图：延迟/速度) ---
    ax2 = ax1.twinx()
    bars = ax2.bar(x, peak_token, width=0.55, color=COLOR_BAR, 
                   alpha=0.4, label='Peak(Token)', zorder=1)
    
    ax2.set_ylabel('Peak(Token)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 2500)
    ax2.grid(False) # 关闭次轴网格，避免干扰

    # --- 4. 绘制主坐标轴 (核心折线：准确率) ---
    # 使用 zorder=3 确保折线在柱子上方
    line, = ax1.plot(x, accuracy, color=COLOR_LINE, marker='o', 
                     linewidth=3, markersize=10, markerfacecolor='white', 
                     markeredgewidth=2.5, label='Top-1 Accuracy', zorder=3)

    # 添加数值标签 (Data Annotations)
    for x_i, y in zip(x, accuracy):
        ax1.annotate(f'{y}%', xy=(x_i, y), xytext=(0, 10), 
                     textcoords="offset points", ha='center', 
                     fontsize=10, fontweight='bold', color=COLOR_LINE)

    # --- 5. 坐标轴美化 ---
    ax1.set_xlabel('Compression Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    
    # 限制范围
    ax1.set_xlim(-0.6, len(comp_levels) - 0.4)
    ax1.set_ylim(50, 70)
    
    # 格式化刻度
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_levels)

    # --- 6. 图例与标题 ---
    # 合并两个轴的图例
    lines = [line, bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=11, facecolor='white')

    plt.title('Impact of Model Compression on Accuracy and Speed', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.figtext(0.5, 0.01, "(b) Trade-off visualization across compression techniques", 
                ha="center", fontsize=12, fontstyle='italic')

    # --- 7. 输出文件 ---
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存两个格式
    plt.savefig("compression_ratio.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("compression_ratio.png", format='png', dpi=300, bbox_inches='tight')
    
    print("已生成：compression_ratio.pdf 和 compression_ratio.png")
    plt.show()

if __name__ == "__main__":
    plot_professional_tradeoff()