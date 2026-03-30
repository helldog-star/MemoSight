import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def generate_combined_figure():
	plt.style.use('seaborn-v0_8-whitegrid')
	plt.rcParams['font.sans-serif'] = ['Arial']
	plt.rcParams['axes.unicode_minus'] = False

	fig = plt.figure(figsize=(22, 6), dpi=120)
	gs = fig.add_gridspec(1, 3, wspace=0.35)

	# (a) Output Length
	ax_a = fig.add_subplot(gs[0, 0])
	models = ['Qwen-2.5-7B', 'Llama-3.1-8B']
	vanilla_data = [2934, 3095]
	h2o_data = [3863, 4020]
	lightthinker_data = [2714, 3034]
	memcot_data = [2678, 2835]

	x_a = np.arange(len(models))
	width = 0.18

	vanilla_bars = ax_a.bar(
		x_a - 1.5 * width,
		vanilla_data,
		width,
		label='Vanilla',
		color='#F0C8C3',
		edgecolor='black',
		hatch='|',
	)
	h2o_bars = ax_a.bar(
		x_a - 0.5 * width,
		h2o_data,
		width,
		label='H2O',
		color='#FEE5D5',
		edgecolor='black',
		hatch='\\',
	)
	lightthinker_bars = ax_a.bar(
		x_a + 0.5 * width,
		lightthinker_data,
		width,
		label='LightThinker',
		color='#BFD3C2',
		edgecolor='black',
		hatch='/',
	)
	memcot_bars = ax_a.bar(
		x_a + 1.5 * width,
		memcot_data,
		width,
		label='MemCoT',
		color='#68A690',
		edgecolor='black',
		hatch='-',
	)

	for bars in [vanilla_bars, h2o_bars, lightthinker_bars, memcot_bars]:
		for bar in bars:
			height = bar.get_height()
			ax_a.text(
				bar.get_x() + bar.get_width() / 2,
				height + 35,
				f'{int(height)}',
				ha='center',
				va='bottom',
				fontsize=9,
			)

	ax_a.set_ylabel('Generated Tokens', fontsize=12, fontweight='bold')
	ax_a.set_xticks(x_a)
	ax_a.set_xticklabels(models, fontsize=11)
	ax_a.set_ylim(0, 4800)
	ax_a.set_yticks([0, 1000, 2000, 3000, 4000])
	ax_a.set_yticklabels(['0', '1k', '2k', '3k', '4k'])
	ax_a.legend(ncol=2, loc='upper right', fontsize=10, frameon=True)
	ax_a.set_title('(a) Average Generated Tokens', fontsize=13, fontweight='bold', pad=10)

	# (b) Compression Trade-off
	ax_b = fig.add_subplot(gs[0, 1])
	comp_levels = np.array(['2x', '4x', '8x', '16x'])
	x_b = np.arange(len(comp_levels))
	comp_acc = np.array([68.41, 65.52, 64.71, 61.72])
	comp_peak = np.array([2041, 1515, 1051, 944])

	ax_b_r = ax_b.twinx()
	bars_b = ax_b_r.bar(
		x_b,
		comp_peak,
		width=0.55,
		color='#ffa600',
		alpha=0.4,
		label='Peak(Token)',
		zorder=1,
	)
	line_b, = ax_b.plot(
		x_b,
		comp_acc,
		color='#004c6d',
		marker='o',
		linewidth=3,
		markersize=9,
		markerfacecolor='white',
		markeredgewidth=2.2,
		label='Accuracy',
		zorder=3,
	)

	for xi, yi in zip(x_b, comp_acc):
		ax_b.annotate(
			f'{yi:.2f}%',
			xy=(xi, yi),
			xytext=(0, 9),
			textcoords='offset points',
			ha='center',
			fontsize=9,
			fontweight='bold',
			color='#004c6d',
		)

	ax_b.set_xlabel('Compression Level', fontsize=12, fontweight='bold')
	ax_b.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
	ax_b_r.set_ylabel('Peak(Token)', fontsize=12, fontweight='bold')
	ax_b.set_xlim(-0.6, len(comp_levels) - 0.4)
	ax_b.set_ylim(50, 70)
	ax_b_r.set_ylim(0, 2500)
	ax_b.yaxis.set_major_formatter(mtick.PercentFormatter())
	ax_b.set_xticks(x_b)
	ax_b.set_xticklabels(comp_levels)
	ax_b_r.grid(False)
	ax_b.legend([line_b, bars_b], ['Accuracy', 'Peak(Token)'], loc='upper right', fontsize=10, frameon=True)
	ax_b.set_title('(b) Compression Trade-off', fontsize=13, fontweight='bold', pad=10)

	# (c) Offset Trade-off
	ax_c = fig.add_subplot(gs[0, 2])
	offset_levels = np.array(['d1', 'd2', 'd3', 'd4'])
	x_c = np.arange(len(offset_levels))
	offset_acc = np.array([64.94, 66.19, 66.08, 65.01])
	offset_peak = np.array([1357, 1295, 1210, 1186])

	ax_c_r = ax_c.twinx()
	bars_c = ax_c_r.bar(
		x_c,
		offset_peak,
		width=0.55,
		color='#ffa600',
		alpha=0.4,
		label='Peak(Token)',
		zorder=1,
	)
	line_c, = ax_c.plot(
		x_c,
		offset_acc,
		color='#004c6d',
		marker='o',
		linewidth=3,
		markersize=9,
		markerfacecolor='white',
		markeredgewidth=2.2,
		label='Accuracy',
		zorder=3,
	)

	for xi, yi in zip(x_c, offset_acc):
		ax_c.annotate(
			f'{yi:.2f}%',
			xy=(xi, yi),
			xytext=(0, 9),
			textcoords='offset points',
			ha='center',
			fontsize=9,
			fontweight='bold',
			color='#004c6d',
		)

	ax_c.set_xlabel('Offset Level', fontsize=12, fontweight='bold')
	ax_c.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
	ax_c_r.set_ylabel('Peak(Token)', fontsize=12, fontweight='bold')
	ax_c.set_xlim(-0.6, len(offset_levels) - 0.4)
	ax_c.set_ylim(50, 70)
	ax_c_r.set_ylim(0, 2000)
	ax_c.yaxis.set_major_formatter(mtick.PercentFormatter())
	ax_c.set_xticks(x_c)
	ax_c.set_xticklabels(offset_levels)
	ax_c_r.grid(False)
	ax_c.legend([line_c, bars_c], ['Accuracy', 'Peak(Token)'], loc='upper right', fontsize=10, frameon=True)
	ax_c.set_title('(c) Offset Trade-off', fontsize=13, fontweight='bold', pad=10)

	fig.suptitle('MemCoT Evaluation Overview', fontsize=18, fontweight='bold', y=1.02)
	plt.tight_layout()

	plt.savefig('output_offset_compress.pdf', format='pdf', bbox_inches='tight')
	plt.savefig('output_offset_compress.png', format='png', dpi=300, bbox_inches='tight')

	print('已生成：output_offset_compress.pdf 和 output_offset_compress.png')
	plt.show()


if __name__ == '__main__':
	generate_combined_figure()
