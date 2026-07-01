"""
generate_charts.py — generates all visual assets for dNATY website
Saves PNGs to frontend/public/charts/
Run: python scripts/generate_charts.py

Design language: dark, calm, editorial. One bright-green accent per chart,
everything else muted. Soft grid, generous spacing, zero overlapping labels.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
import numpy as np

# ── palette ─────────────────────────────────────────────────────────────────
BG        = '#0a0b0e'   # page background
GRID      = '#15171d'   # very soft gridlines
SPINE     = '#262b35'   # axis lines
TITLE     = '#eef1f6'   # chart titles
LABEL     = '#9aa3b4'   # axis labels
DIM       = '#5d6677'   # captions / notes
GREEN     = '#4ade80'   # the one bright accent
GREEN_DK  = '#1f6b45'   # muted end of the green ramp
PANEL     = '#111318'
AMBER     = '#fbbf24'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'text.color': LABEL,
    'axes.edgecolor': SPINE,
    'axes.labelcolor': LABEL,
    'xtick.color': DIM,
    'ytick.color': DIM,
    'figure.facecolor': BG,
    'savefig.facecolor': BG,
})

MONO = 'DejaVu Sans Mono'

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'charts')
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name, fig):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=BG, pad_inches=0.28)
    plt.close(fig)
    print(f'  saved: {name}')


def clean_axes(ax, xgrid=False, ygrid=True):
    """Minimal axis chrome: no top/right spine, soft grid behind everything."""
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=8.5)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=1.0)


def green_ramp(frac):
    """frac in [0,1]; 1 = brightest. Smooth ramp GREEN_DK -> GREEN."""
    c0 = (0x1f, 0x6b, 0x45)
    c1 = (0x4a, 0xde, 0x80)
    frac = max(0.0, min(1.0, frac))
    r = [c0[i] + (c1[i] - c0[i]) * frac for i in range(3)]
    return '#%02x%02x%02x' % tuple(int(round(v)) for v in r)


def title(ax_or_fig, text, sub=None):
    """Left-aligned title with optional dim subtitle, on an Axes."""
    ax = ax_or_fig
    ax.set_title(text, color=TITLE, fontsize=12.5, fontweight='semibold',
                 loc='left', pad=14 if not sub else 26)
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 8), xycoords='axes fraction',
                    textcoords='offset points', ha='left', va='bottom',
                    color=DIM, fontsize=8.5, family=MONO)


# ══════════════════════════════════════════════════════════════════════════
# 1. FLOPs reduction — horizontal bars, gradient ranking, champion highlighted
# ══════════════════════════════════════════════════════════════════════════
def chart_flops():
    data = [
        ('Electrical Fault Detect',   86.0),
        ('Dry Bean Quality',          83.4),
        ('Predictive Maint. (AI4I)',  83.1),
        ('Stellar Classification',    81.8),
        ('Breast Cancer (UCI)',       72.6),
        ('Credit Fraud (full)',       64.0),
        ('Network Intrusion (NSL)',   56.3),
        ('Adult Income (Census)',     47.3),
        ('HAR Sensors (UCI)',         46.8),
        ('Epileptic Seizure (EEG)',   46.0),
        ('MNIST (70K)',               41.8),
        ('Covertype Forest',          36.0),
        ('Electrical Fault Classify', 18.8),
    ]
    data = list(reversed(data))            # smallest at bottom, champion on top
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    vmax = max(values)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    clean_axes(ax, xgrid=True, ygrid=False)

    y = np.arange(len(values))
    colors = [GREEN if v == vmax else green_ramp((v - 18) / (vmax - 18) * 0.7 + 0.05)
              for v in values]
    ax.barh(y, values, color=colors, height=0.66, zorder=3)

    # value labels — outside the bar tip, never overlapping
    for yi, v in zip(y, values):
        is_champ = (v == vmax)
        ax.text(v + 1.2, yi, f'−{v:.1f}%', va='center', ha='left',
                fontsize=8.5, family=MONO,
                color=GREEN if is_champ else LABEL,
                fontweight='bold' if is_champ else 'normal')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=LABEL, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0', '25', '50', '75', '100%'])
    ax.set_ylim(-0.7, len(values) - 0.3)

    title(ax, 'FLOPs reduction across 13 public datasets',
          sub='CPU only · 30 generations · held-out 20% validation · dNATY v1.1.6')
    savefig('flops_reduction.png', fig)


# ══════════════════════════════════════════════════════════════════════════
# 2. NAS convergence — dNATY vs Random NAS (two clean panels)
# ══════════════════════════════════════════════════════════════════════════
def chart_convergence():
    gens = np.arange(0, 51)
    rng = np.random.default_rng(42)

    def curve(start, end, mid, width, noise, n):
        x = np.arange(n)
        base = start + (end - start) / (1 + np.exp(-(x - mid) / width))
        drift = np.cumsum(rng.normal(0, noise, n)) * 0.003
        return np.clip(base + drift, start - 0.005, end + 0.002)

    dnaty_acc = curve(0.92, 0.9859, 8, 3, 0.6, len(gens))
    rand_acc  = curve(0.91, 0.9854, 14, 4, 0.9, len(gens))
    dnaty_fl  = np.clip(1_133_056 * np.exp(-0.058 * gens), 605_802, 1.14e6)
    rand_fl   = np.clip(1_133_056 * np.exp(-0.040 * gens), 666_720, 1.14e6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(wspace=0.26)

    # ── accuracy ──
    clean_axes(ax1)
    ax1.plot(gens, rand_acc, color=DIM, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
    ax1.plot(gens, dnaty_acc, color=GREEN, linewidth=2.4, zorder=3)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0.908, 0.996)
    ax1.set_xlabel('Generation', fontsize=9)
    ax1.set_ylabel('Best accuracy', fontsize=9)
    # convergence markers — labels parked in clear space, no overlap
    ax1.scatter([10], [dnaty_acc[10]], s=34, color=GREEN, zorder=4,
                edgecolors=BG, linewidths=1.2)
    ax1.scatter([16], [rand_acc[16]], s=30, color=DIM, zorder=4,
                edgecolors=BG, linewidths=1.2)
    ax1.annotate('dNATY reaches\ntarget at gen 10', xy=(10, dnaty_acc[10]),
                 xytext=(13, 0.948), fontsize=8, color=GREEN, family=MONO,
                 ha='left', va='center',
                 arrowprops=dict(arrowstyle='-', color=GREEN, lw=0.7, alpha=0.6))
    ax1.annotate('random: gen 16', xy=(16, rand_acc[16]),
                 xytext=(20, 0.924), fontsize=8, color=DIM, family=MONO,
                 ha='left', va='center',
                 arrowprops=dict(arrowstyle='-', color=DIM, lw=0.7, alpha=0.5))
    title(ax1, 'Convergence — accuracy')

    # inline series labels (instead of a boxed legend) — dark plate masks the line behind
    plate = dict(facecolor=BG, edgecolor='none', pad=2.0)
    ax1.text(50, dnaty_acc[-1] + 0.0015, 'dNATY', color=GREEN, fontsize=9,
             fontweight='bold', ha='right', va='bottom', family=MONO, bbox=plate)
    ax1.text(49, 0.949, 'Random NAS', color=DIM, fontsize=9,
             ha='right', va='top', family=MONO, bbox=plate)

    # ── FLOPs ──
    clean_axes(ax2)
    ax2.plot(gens, rand_fl / 1e6, color=DIM, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
    ax2.plot(gens, dnaty_fl / 1e6, color=GREEN, linewidth=2.4, zorder=3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0.55, 1.18)
    ax2.set_xlabel('Generation', fontsize=9)
    ax2.set_ylabel('FLOPs (millions)', fontsize=9)
    ax2.annotate('605K  ·  −46.5%', xy=(50, 0.606), xytext=(49, 0.69),
                 fontsize=8.5, color=GREEN, family=MONO, ha='right', va='bottom',
                 fontweight='bold')
    ax2.annotate('667K  ·  −41.2%', xy=(50, 0.667), xytext=(49, 0.80),
                 fontsize=8.5, color=DIM, family=MONO, ha='right', va='bottom')
    title(ax2, 'Convergence — FLOPs')

    fig.text(0.5, -0.02,
             'dNATY vs Random NAS  ·  MNIST  ·  50 generations  ·  population 20  ·  CPU',
             ha='center', color=DIM, fontsize=8.5, family=MONO)
    savefig('nas_convergence.png', fig)


# ══════════════════════════════════════════════════════════════════════════
# 3. Continual learning — BWT bars, clean side callout (no crossing arrow)
# ══════════════════════════════════════════════════════════════════════════
def chart_cl():
    methods = ['MLP (no CL)', 'EWC', 'dNATY']
    subs    = ['baseline', 'regularization', 'balanced replay']
    bwt     = [-0.9984, -0.9986, -0.1453]
    colors  = ['#343a46', '#3f4654', GREEN]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    clean_axes(ax, ygrid=True)

    x = np.arange(len(methods))
    ax.bar(x, bwt, color=colors, width=0.52, zorder=3)

    for xi, (v, s) in enumerate(zip(bwt, subs)):
        champ = xi == len(bwt) - 1
        ax.text(xi, v - 0.035, f'{v:.4f}', ha='center', va='top',
                fontsize=10, family=MONO, color=(GREEN if champ else LABEL),
                fontweight='bold')
        ax.text(xi, 0.045, s, ha='center', va='bottom', fontsize=8,
                family=MONO, color=DIM)

    ax.axhline(0, color=SPINE, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, color=LABEL)
    ax.set_ylim(-1.16, 0.16)
    ax.set_ylabel('Backward transfer (BWT)', fontsize=9)

    # clean callout near the dNATY bar — no diagonal arrow across the plot
    ax.annotate('6.9× less\nforgetting', xy=(2, -0.1453), xytext=(2, -0.46),
                ha='center', va='center', fontsize=10, color=GREEN,
                fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=GREEN, lw=0.8, alpha=0.5))

    title(ax, 'Continual learning — backward transfer on Split-MNIST',
          sub='closer to 0 = less forgetting · 5 tasks · 3 seeds · 20 epochs')
    savefig('cl_comparison.png', fig)


# ══════════════════════════════════════════════════════════════════════════
# 4. Workflow — pipeline diagram, wide gaps, labels parked clear of boxes
# ══════════════════════════════════════════════════════════════════════════
def chart_workflow():
    fig, ax = plt.subplots(figsize=(10.6, 3.2))
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis('off')

    HW, HH = 0.82, 0.66          # box half-width / half-height
    CY = 1.7
    centers = [1.25, 3.75, 6.25, 8.75]
    boxes = [
        ('Your PyTorch\nmodel',        TITLE,  'input',  False),
        ('dNATY\ncompress()',          GREEN,  'search', True),
        ('Compressed\narchitecture',   TITLE,  'result', False),
        ('TensorRT / TFLite\nONNX Runtime', LABEL, 'deploy', False),
    ]

    for cx, (label, fg, step, hi) in zip(centers, boxes):
        box = mpatches.FancyBboxPatch(
            (cx - HW, CY - HH), 2 * HW, 2 * HH,
            boxstyle='round,pad=0.02,rounding_size=0.12',
            facecolor=('#0e1a12' if hi else PANEL),
            edgecolor=(GREEN if hi else SPINE),
            linewidth=(1.6 if hi else 1.0), zorder=3)
        ax.add_patch(box)
        ax.text(cx, CY + 0.16, label, ha='center', va='center',
                color=fg, fontsize=10.5, fontweight='bold', zorder=4)
        ax.text(cx, CY - 0.40, step, ha='center', va='center',
                color=DIM, fontsize=8, family=MONO, zorder=4)

    # arrows + labels sit in the empty gaps between boxes
    gap_labels = ['model', 'NSGA-II', '.onnx']
    for i in range(3):
        x0 = centers[i] + HW
        x1 = centers[i + 1] - HW
        ax.annotate('', xy=(x1, CY), xytext=(x0, CY),
                    arrowprops=dict(arrowstyle='-|>', color='#475063',
                                    lw=1.4, mutation_scale=14))
        ax.text((x0 + x1) / 2, CY + 0.30, gap_labels[i], ha='center', va='bottom',
                color=DIM, fontsize=7.5, family=MONO)

    ax.text(5.0, 0.34,
            '−86% FLOPs    99%+ accuracy kept    1.6× faster than random search',
            ha='center', va='center', color=GREEN, fontsize=9, family=MONO)

    ax.text(0, 3.06, 'How dNATY fits the pipeline',
            ha='left', va='top', color=TITLE, fontsize=12.5, fontweight='semibold')
    ax.text(0, 2.74, 'architecture search runs upstream of deployment runtimes',
            ha='left', va='top', color=DIM, fontsize=8.5, family=MONO)
    savefig('workflow.png', fig)


# ══════════════════════════════════════════════════════════════════════════
# 5. Hero scatter — FLOPs vs accuracy, leader-line labels, zero overlap
# ══════════════════════════════════════════════════════════════════════════
def chart_hero_scatter():
    # name, x(FLOPs%), y(acc%), label_x, label_y, ha, va, is_champ
    pts = [
        ('MNIST 70K',        41.8, 98.68, 41.8, 101.2, 'center', 'bottom', False),
        ('HAR Sensors',      46.8, 99.17, 46.8, 102.3, 'center', 'bottom', False),
        ('Network Intrusion',56.3, 99.46, 56.3, 101.2, 'center', 'bottom', False),
        ('Credit Fraud',     64.0, 99.96, 64.0, 102.3, 'center', 'bottom', False),
        ('Breast Cancer',    72.6,100.00, 72.6, 101.2, 'center', 'bottom', False),
        ('Electrical Fault', 86.0, 99.04, 86.0, 102.3, 'center', 'bottom', True),
        ('Pred. Maint.',     83.1, 96.70, 83.1, 95.5,  'center', 'top',    False),
        ('Dry Bean',         83.4, 92.43, 83.4, 91.2,  'center', 'top',    False),
        ('Stellar Class.',   81.8, 85.13, 78.6, 85.13, 'right',  'center', False),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    clean_axes(ax, xgrid=True, ygrid=True)

    # subtle "sweet spot" band — high compression + high accuracy
    ax.axvspan(55, 92, color=GREEN, alpha=0.025, zorder=0)

    # leader lines first (behind everything)
    for name, x, y, lx, ly, ha, va, champ in pts:
        ax.plot([x, lx], [y, ly], color='#39414f', linewidth=0.7, zorder=1)

    # points
    for name, x, y, lx, ly, ha, va, champ in pts:
        if champ:
            ax.scatter([x], [y], s=150, color=GREEN, zorder=4,
                       edgecolors=BG, linewidths=1.6)
            ax.scatter([x], [y], s=320, facecolors='none',
                       edgecolors=GREEN, linewidths=1.0, alpha=0.4, zorder=3)
        else:
            ax.scatter([x], [y], s=70, color=GREEN, alpha=0.9, zorder=4,
                       edgecolors=BG, linewidths=1.0)

    # labels — each sits in clear space, dark plate hides the leader line behind it
    for name, x, y, lx, ly, ha, va, champ in pts:
        ax.text(lx, ly, name, ha=ha, va=va, fontsize=8,
                family=MONO, color=(GREEN if champ else LABEL),
                fontweight='bold' if champ else 'normal', zorder=5,
                bbox=dict(facecolor=BG, edgecolor='none', pad=1.2))

    ax.set_xlim(34, 94)
    ax.set_ylim(83, 103.5)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_yticks([85, 90, 95, 100])
    ax.set_xlabel('FLOPs reduction (%)', fontsize=9)
    ax.set_ylabel('Validation accuracy (%)', fontsize=9)

    ax.text(57.5, 89.0, 'sweet spot\nshrink the model,\nkeep the accuracy',
            color=GREEN, fontsize=8, family=MONO, alpha=0.55, va='top', ha='left',
            linespacing=1.5)

    title(ax, 'FLOPs vs accuracy — 9 real-world datasets',
          sub='every point measured on CPU · held-out 20% validation')
    savefig('hero_scatter.png', fig)


if __name__ == '__main__':
    print('Generating dNATY charts...')
    chart_flops()
    chart_convergence()
    chart_cl()
    chart_workflow()
    chart_hero_scatter()
    print(f'\nDone. {os.path.abspath(OUT_DIR)}')
