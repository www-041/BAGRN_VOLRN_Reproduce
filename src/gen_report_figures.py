"""
生成周报用图表
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

out_dir = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\report_figures'
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====== 图1：代码模块占比（环形图） ======
fig, ax = plt.subplots(figsize=(8, 6))
modules = ['BAGRN\n(bagrn.py)', 'VOLRN\n(volrn.py)', '主流程\n(main.py)',
           '评价指标\n(metrics.py)', 'IO工具\n(io_utils.py)',
           '重叠检测\n(overlap.py)', '测试代码\n(tests/)']
lines = [348, 696, 696, 333, 174, 152, 1057]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05)
wedges, texts, autotexts = ax.pie(lines, labels=modules, autopct='%1.1f%%',
                                    colors=colors, explode=explode,
                                    startangle=90, pctdistance=0.75,
                                    textprops={'fontsize': 9})
for t in autotexts:
    t.set_fontsize(8)
ax.set_title('BAGRN-VOLRN 项目代码分布（总计 3456 行）', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'code_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图1 已生成')

# ====== 图2：测试结果 ======
fig, ax = plt.subplots(figsize=(8, 5))
modules_t = ['重叠检测\n(overlap)', 'BAGRN', 'VOLRN', '指标\n(metrics)', '主流程\n(main)']
passed = [7, 9, 8, 18, 5]
colors_t = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
bars = ax.bar(modules_t, passed, color=colors_t, width=0.5, edgecolor='white', linewidth=1.5)
for bar, p in zip(bars, passed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(p),
            ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 22)
ax.set_ylabel('测试用例数', fontsize=12)
ax.set_title('各模块单元测试结果（47/47 全部通过）', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'test_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已生成')

# ====== 图3：处理流程 ======
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

steps = [
    (0.5, '读取\nGeoTIFF', '#45B7D1'),
    (2.0, '检测\n重叠区域', '#4ECDC4'),
    (3.5, 'BAGRN\n全局归一', '#FF6B6B'),
    (5.0, 'VOLRN\n局部归一', '#96CEB4'),
    (6.5, '写出\n归一化结果', '#DDA0DD'),
    (8.0, '评价指标\n& 日志', '#FFEAA7'),
]

for x, label, color in steps:
    rect = mpatches.FancyBboxPatch((x, 0.8), 1.2, 1.4, boxstyle='round,pad=0.15',
                                     facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + 0.6, 1.5, label, ha='center', va='center', fontsize=9, fontweight='bold', color='#333')

for i in range(len(steps)-1):
    x1 = steps[i][0] + 1.2
    x2 = steps[i+1][0]
    ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#888'))

ax.set_title('BAGRN-VOLRN 处理流水线', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'pipeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图3 已生成')

# ====== 图4：BAGRN 原理图解 ======
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

ax = axes[0]
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.axis('off')
rect1 = mpatches.FancyBboxPatch((0.3, 1.8), 1.8, 1.8, boxstyle='round,pad=0.1',
                                  facecolor='#FF6B6B', alpha=0.3, edgecolor='#FF6B6B')
rect2 = mpatches.FancyBboxPatch((1.5, 0.2), 1.8, 1.8, boxstyle='round,pad=0.1',
                                  facecolor='#4ECDC4', alpha=0.3, edgecolor='#4ECDC4')
ax.add_patch(rect1)
ax.add_patch(rect2)
overlap = mpatches.FancyBboxPatch((1.2, 1.4), 1.2, 1.0, boxstyle='round,pad=0.05',
                                    facecolor='#FFD93D', alpha=0.7, edgecolor='#E6A817')
ax.add_patch(overlap)
ax.text(0.4, 3.4, '影像 i', fontsize=10, fontweight='bold', color='#CC4444')
ax.text(2.6, 0.6, '影像 j', fontsize=10, fontweight='bold', color='#3AAFA9')
ax.text(1.8, 1.7, '重叠区\nmu_ij, sigma_ij', fontsize=8, ha='center', va='center', fontweight='bold')
ax.set_title('1 提取重叠区\nmu sigma 统计量', fontsize=10, fontweight='bold')

ax = axes[1]
ax.axis('off')
matrix_data = np.random.rand(6, 6) * 0.3 + 0.7
ax.imshow(matrix_data, cmap='Blues', aspect='auto', alpha=0.8)
eq_text = 'D * theta = L'
ax.text(0.5, 0.6, eq_text, transform=ax.transAxes, fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(0.5, 0.25, '加权最小二乘\nmin ||sqrt(P)(D*theta-L)||^2',
        transform=ax.transAxes, fontsize=10, ha='center', va='center')
ax.set_title('2 构建稀疏线性系统\n求解补偿值', fontsize=10, fontweight='bold')

ax = axes[2]
ax.axis('off')
x = np.linspace(0, 4, 100)
y1 = 0.3 * x + 0.5 + 0.05 * np.random.randn(100)
y2 = 0.3 * x + 0.8 + 0.05 * np.random.randn(100)
ax.scatter(x, y1, s=5, c='#FF6B6B', alpha=0.5, label='原始')
ax.scatter(x, y2, s=5, c='#4ECDC4', alpha=0.5, label='归一化后')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlabel('像素值', fontsize=8)
ax.set_ylabel('频数', fontsize=8)
ax.set_title("3 Moment Matching\nf' = w*f + v", fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('BAGRN 全局辐射归一化原理', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'bagrn_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图4 已生成')

# ====== 图5：VOLRN 原理图解 ======
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

ax = axes[0]
ax.axis('off')
for r in range(3):
    for c in range(3):
        rect = mpatches.FancyBboxPatch((c*1.1, 2-r*1.1), 1.0, 1.0, boxstyle='round,pad=0.05',
                                        facecolor='#45B7D1', alpha=0.3, edgecolor='#45B7D1')
        ax.add_patch(rect)
overlap_rect = mpatches.FancyBboxPatch((0.55, 0.95), 1.1, 1.1, boxstyle='round,pad=0.05',
                                         facecolor='#FF6B6B', alpha=0.4, edgecolor='#FF6B6B')
ax.add_patch(overlap_rect)
ax.text(1.1, 1.5, '重叠块对', ha='center', va='center', fontsize=8, fontweight='bold', color='#CC4444')
ax.set_title('1 规则网格分块\n(block_size x block_size)', fontsize=10, fontweight='bold')

ax = axes[1]
ax.axis('off')
eq1 = 'min 1/2 ||Bx||^2 + lambda ||Ax-b||_1'
eq2 = 'Bx: 块间差异项'
eq3 = '||Ax-b||_1: l1 稀疏保真项'
ax.text(0.5, 0.7, eq1, fontsize=13, ha='center', va='center', fontweight='bold')
ax.text(0.5, 0.45, eq2, fontsize=10, ha='center', va='center', color='#555')
ax.text(0.5, 0.25, eq3, fontsize=10, ha='center', va='center', color='#555')
ax.set_title('2 构建变分模型\nl1+l2 混合范数', fontsize=10, fontweight='bold')

ax = axes[2]
ax.axis('off')
admm_steps = ['x <- PCG\n(BtB+rAtA)^-1', 'z <- S_{l/r}\n(软阈值)', 'u <- u +\n(对偶更新)']
for i, (txt, color) in enumerate(zip(admm_steps, ['#FF6B6B', '#4ECDC4', '#45B7D1'])):
    rect = mpatches.FancyBboxPatch((0.05 + i*1.1, 0.5), 1.0, 1.0, boxstyle='round,pad=0.1',
                                    facecolor=color, alpha=0.3, edgecolor=color)
    ax.add_patch(rect)
    ax.text(0.55 + i*1.1, 1.0, txt, ha='center', va='center', fontsize=8, fontweight='bold')
    if i < 2:
        ax.annotate('', xy=(1.15 + i*1.1, 1.0), xytext=(1.05 + i*1.1, 1.0),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#888'))
ax.set_title('3 ADMM 交替迭代\nEq.(30)-(33)', fontsize=10, fontweight='bold')

plt.suptitle('VOLRN 局部辐射归一化原理', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'volrn_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图5 已生成')

print()
print('所有图片已生成至:', out_dir)
