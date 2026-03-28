import matplotlib.pyplot as plt
import numpy as np

def generate_dual_formats():
    # --- 1. 数据准备 ---
    models = ['Qwen-2.5-7B', 'Llama-3.1-8B']
    vanilla_data = [2934, 3095]
    h2o_data = [3863, 4020]
    lightthinker_data = [2714, 3034]
    memcot_data = [2678, 2835]

    x = np.arange(len(models))  
    width = 0.18

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- 2. 绘制柱状图 (带填充纹理) ---
    vanilla_bars = ax.bar(x - 1.5*width, vanilla_data, width, label='Vanilla', color='#F0C8C3', edgecolor='black', hatch='|')
    h2o_bars = ax.bar(x - 0.5*width, h2o_data, width, label='H2O', color='#FEE5D5', edgecolor='black', hatch='\\')
    lightthinker_bars = ax.bar(x + 0.5*width, lightthinker_data, width, label='LightThinker', color='#BFD3C2', edgecolor='black', hatch='/')
    memcot_bars = ax.bar(x + 1.5*width, memcot_data, width, label='MemCoT', color='#68A690', edgecolor='black',  hatch='-')

    # 在柱子顶部显示具体数值
    for bars in [vanilla_bars, h2o_bars, lightthinker_bars, memcot_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 35,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )

    # --- 3. 细节调整 ---
    ax.set_ylabel('Generated Tokens', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 4800)
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.set_yticklabels(['0', '1k', '2k', '3k', '4k'])
    ax.legend(ncol=2, loc='upper right')

    # 添加底部标题
    plt.figtext(0.5, 0.02, "(a) Average number of generated tokens across the four datasets", 
                ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # --- 4. 同时生成两个文件 ---
    # 保存为 PDF (矢量图，适合论文)
    plt.savefig("output_length.pdf", format='pdf', bbox_inches='tight')
    
    # 保存为 PNG (像素图，适合预览，设置 300 DPI 保证清晰度)
    plt.savefig("output_length.png", format='png', dpi=300, bbox_inches='tight')

    plt.show()
    print("已成功生成 result_chart.pdf 和 result_chart.png")

if __name__ == "__main__":
    generate_dual_formats()