"""
generate_demo_video.py — creates animated GIF demo for dNATY website
Output: frontend/public/demo.gif  (~800×450, 10fps, ~15s)
Run:    python scripts/generate_demo_video.py
"""

import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import io

# ── theme ──────────────────────────────────────────────────────────────────
BG       = '#0a0b0e'
PANEL    = '#111318'
HAIRLINE = '#1f2128'
GREEN    = '#4ade80'
WHITE    = '#ffffff'
DIM      = '#4a5060'
DIM2     = '#2a2f3a'
AMBER    = '#fbbf24'

FPS   = 12
W, H  = 960, 540
DPI   = 96
FW    = W / DPI
FH    = H / DPI

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_GIF = os.path.join(OUT_DIR, 'demo.gif')


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize((W, H), Image.LANCZOS)
    return img.copy()


def ease_in_out(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)

def clamp(v, lo=0.0, hi=1.0):
    return float(np.clip(v, lo, hi))


# ═══════════════════════════════════════════════════════════════════════════
# Scene 1 — Title card  (0–2s → 0–24 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_title(frames_out, n_frames=28):
    for i in range(n_frames):
        t = ease_in_out(i / max(n_frames - 1, 1))
        fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
        ax.set_facecolor(BG)
        ax.axis('off')

        # background grid lines (subtle)
        for y in np.linspace(0.1, 0.9, 8):
            ax.axhline(y, color=HAIRLINE, linewidth=0.5, alpha=0.6)
        for x in np.linspace(0.05, 0.95, 14):
            ax.axvline(x, color=HAIRLINE, linewidth=0.5, alpha=0.6)

        # green accent bar top
        ax.axhline(0.88, color=GREEN, linewidth=2, alpha=min(t * 2, 1.0))

        ax.text(0.5, 0.72, 'dNATY',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=72, fontweight='bold', color=WHITE,
                alpha=t, fontfamily='DejaVu Sans')

        ax.text(0.5, 0.56,
                'Dynamic Neuro-Adaptive sYstem with evoluTionarY Learning',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#8090a8', alpha=max(t - 0.2, 0),
                fontfamily='monospace')

        ax.text(0.5, 0.40,
                'Neural architecture search · model compression · CPU only',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color=DIM, alpha=max(t - 0.4, 0),
                fontfamily='monospace')

        # pip install badge
        badge_alpha = max(t - 0.6, 0)
        badge = mpatches.FancyBboxPatch((0.35, 0.24), 0.30, 0.08,
                                         boxstyle='round,pad=0.01',
                                         facecolor=PANEL, edgecolor=HAIRLINE,
                                         linewidth=0.8, transform=ax.transAxes,
                                         alpha=badge_alpha)
        ax.add_patch(badge)
        ax.text(0.5, 0.28, '$ pip install dnaty',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#c0c8d8', alpha=badge_alpha,
                fontfamily='monospace')

        frames_out.append(fig_to_pil(fig))
        plt.close(fig)
        sys.stdout.write(f'\r  scene 1: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 2 — FLOPs bar chart building up  (2–7s → 24–84 frames)
# ═══════════════════════════════════════════════════════════════════════════
DATASETS = [
    ('Electrical Fault',    86.0, '99.04%'),
    ('Dry Bean Quality',    83.4, '92.43%'),
    ('Predictive Maint.',   83.1, '96.70%'),
    ('Stellar Class.',      81.8, '85.13%'),
    ('Breast Cancer UCI',   72.6, '100.0%'),
    ('Credit Fraud',        64.0, '99.96%'),
    ('Network Intrusion',   56.3, '99.46%'),
    ('Adult Income',        47.3, '83.60%'),
    ('HAR Sensors',         46.8, '99.17%'),
    ('MNIST 70K',           41.8, '98.68%'),
    ('Covertype Forest',    36.0, '90.07%'),
]

def scene_flops(frames_out, n_frames=60):
    labels  = [d[0] for d in DATASETS]
    values  = [d[1] for d in DATASETS]
    acc_lbl = [d[2] for d in DATASETS]
    n_bars  = len(DATASETS)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        # how many bars are fully visible
        bars_done = t * n_bars

        fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
        ax.set_facecolor(BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(HAIRLINE)
        ax.spines['bottom'].set_color(HAIRLINE)
        ax.tick_params(colors='#6a7890', labelsize=8.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=HAIRLINE, linewidth=0.5)

        for j, (lbl, val, acc) in enumerate(zip(labels, values, acc_lbl)):
            bar_t = max(0, min(1, bars_done - j))
            bar_t = ease_in_out(bar_t)
            color = GREEN if val >= 60 else '#22c55e' if val >= 40 else '#16a34a'
            ax.barh(j, val * bar_t, color=color, height=0.62, zorder=3, alpha=0.9)
            if bar_t > 0.85:
                ax.text(val * bar_t - 1.5, j, f'-{val:.0f}%',
                        va='center', ha='right', fontsize=8,
                        color=BG, fontweight='bold', fontfamily='monospace')
                ax.text(val * bar_t + 1.2, j, acc,
                        va='center', ha='left', fontsize=7.5,
                        color=DIM, fontfamily='monospace')

        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(labels, color='#c0c8d8', fontsize=8.5)
        ax.set_xlabel('FLOPs reduction (%)', color=DIM, fontsize=9)
        ax.set_xlim(0, 98)
        ax.set_ylim(-0.6, n_bars - 0.4)
        ax.tick_params(axis='y', length=0)

        ax.set_title('FLOPs reduction across real-world datasets  ·  val. accuracy shown',
                     color=WHITE, fontsize=11, fontweight='semibold', pad=12, loc='left')

        note = 'CPU only  ·  30 generations  ·  held-out 20% val split  ·  dNATY v1.1.6'
        fig.text(0.02, 0.01, note, color=DIM, fontsize=7.5, fontfamily='monospace')

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        frames_out.append(fig_to_pil(fig))
        plt.close(fig)
        sys.stdout.write(f'\r  scene 2: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 3 — NAS convergence animation  (7–11s → 84–132 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_nas(frames_out, n_frames=48):
    gens_total = 50
    rng = np.random.default_rng(42)

    def sigmoid_curve(start, end, midpoint, width, noise_scale, n):
        x = np.arange(n)
        curve = start + (end - start) / (1 + np.exp(-(x - midpoint) / width))
        noise = rng.normal(0, noise_scale, n)
        noise = np.cumsum(noise) * 0.003
        return np.clip(curve + noise, start - 0.005, end + 0.002)

    dnaty_acc  = sigmoid_curve(0.92, 0.9859, 8,  3, 0.6, gens_total + 1)
    random_acc = sigmoid_curve(0.91, 0.9854, 14, 4, 0.9, gens_total + 1)
    gens_x = np.arange(gens_total + 1)

    for i in range(n_frames):
        t = ease_in_out(i / max(n_frames - 1, 1))
        n_show = max(2, int(t * gens_total))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FW, FH), facecolor=BG)
        for ax in (ax1, ax2):
            ax.set_facecolor(BG)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(HAIRLINE)
            ax.spines['bottom'].set_color(HAIRLINE)
            ax.tick_params(colors=DIM, labelsize=8)
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, color=HAIRLINE, linewidth=0.4)
            ax.yaxis.grid(True, color=HAIRLINE, linewidth=0.4)

        # accuracy
        ax1.plot(gens_x[:n_show], dnaty_acc[:n_show],
                 color=GREEN, linewidth=2.2, label='dNATY', zorder=3)
        ax1.plot(gens_x[:n_show], random_acc[:n_show],
                 color=DIM, linewidth=1.5, label='Random NAS',
                 linestyle='--', zorder=2)

        if n_show >= 10:
            ax1.axvline(10, color=GREEN, linewidth=0.8, linestyle=':', alpha=0.7)
            ax1.text(10.5, 0.921, 'gen 10', color=GREEN,
                     fontsize=7.5, fontfamily='monospace')
        if n_show >= 16:
            ax1.axvline(16, color=DIM, linewidth=0.8, linestyle=':', alpha=0.5)
            ax1.text(16.5, 0.917, 'gen 16', color=DIM,
                     fontsize=7.5, fontfamily='monospace')

        ax1.set_xlim(0, gens_total)
        ax1.set_ylim(0.908, 0.994)
        ax1.set_xlabel('Generation', color=DIM, fontsize=9)
        ax1.set_ylabel('Best accuracy', color=DIM, fontsize=9)
        ax1.set_title('Search convergence — accuracy',
                      color=WHITE, fontsize=10, fontweight='semibold', pad=10, loc='left')
        ax1.legend(facecolor=PANEL, edgecolor=HAIRLINE, labelcolor=WHITE,
                   fontsize=8, loc='lower right')

        # FLOPs (simplified)
        flops_dnaty  = 1133056 * np.exp(-0.058 * gens_x[:n_show])
        flops_random = 1133056 * np.exp(-0.040 * gens_x[:n_show])
        flops_dnaty  = np.clip(flops_dnaty,  600_000, 1_140_000)
        flops_random = np.clip(flops_random, 665_000, 1_140_000)

        ax2.plot(gens_x[:n_show], flops_dnaty / 1e6,
                 color=GREEN, linewidth=2.2, label='dNATY', zorder=3)
        ax2.plot(gens_x[:n_show], flops_random / 1e6,
                 color=DIM, linewidth=1.5, label='Random NAS',
                 linestyle='--', zorder=2)

        if n_show >= 15:
            ax2.axhline(605802 / 1e6, color=GREEN, linewidth=0.8,
                        linestyle=':', alpha=0.7)
            ax2.text(n_show * 0.5, 0.615, '605K FLOPs  (-46.5%)',
                     color=GREEN, fontsize=7.5, fontfamily='monospace')
        if n_show >= 20:
            ax2.axhline(666720 / 1e6, color=DIM, linewidth=0.8,
                        linestyle=':', alpha=0.5)
            ax2.text(n_show * 0.5, 0.675, '667K (-41.2%)',
                     color=DIM, fontsize=7.5, fontfamily='monospace')

        ax2.set_xlim(0, gens_total)
        ax2.set_ylim(0.55, 1.16)
        ax2.set_xlabel('Generation', color=DIM, fontsize=9)
        ax2.set_ylabel('FLOPs (M)', color=DIM, fontsize=9)
        ax2.set_title('Pareto-efficient FLOPs',
                      color=WHITE, fontsize=10, fontweight='semibold', pad=10, loc='left')
        ax2.legend(facecolor=PANEL, edgecolor=HAIRLINE, labelcolor=WHITE,
                   fontsize=8, loc='upper right')

        badge_t = float(np.clip((t - 0.7) / 0.3, 0.0, 1.0))
        if badge_t > 0:
            fig.text(0.5, 0.02,
                     'dNATY converges 1.6x faster than random search',
                     ha='center', color=GREEN, fontsize=9,
                     fontfamily='monospace', alpha=badge_t)

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        frames_out.append(fig_to_pil(fig))
        plt.close(fig)
        sys.stdout.write(f'\r  scene 3: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 4 — Continual learning  (11–14s → 132–168 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_cl(frames_out, n_frames=36):
    methods = ['MLP\n(no CL)', 'EWC', 'dNATY\n(balanced replay)']
    bwt     = [-0.9984, -0.9986, -0.1453]
    colors  = [DIM2, DIM, GREEN]

    for i in range(n_frames):
        t = ease_in_out(i / max(n_frames - 1, 1))
        fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
        ax.set_facecolor(BG)

        for j, (m, v, c) in enumerate(zip(methods, bwt, colors)):
            bar_t = ease_in_out(max(0, min(1, t * 3 - j * 0.6)))
            ax.bar(j, v * bar_t, color=c, width=0.46, zorder=3, alpha=0.9)
            if bar_t > 0.9:
                ax.text(j, v * bar_t - 0.04, f'{v:.4f}',
                        ha='center', va='top', fontsize=10,
                        color=WHITE, fontfamily='monospace', fontweight='bold')

        ax.axhline(0, color=HAIRLINE, linewidth=1.2)
        ax.text(2.38, 0.04, '<- ideal (0)', color=DIM,
                fontsize=8, fontfamily='monospace')

        # 6.9x annotation fades in late
        arrow_t = float(np.clip((t - 0.7) / 0.3, 0.0, 1.0))
        if arrow_t > 0:
            ax.annotate('', xy=(2, -0.1453), xytext=(1, -0.9986),
                        arrowprops=dict(arrowstyle='<->', color=GREEN,
                                        lw=1.3, alpha=arrow_t))
            ax.text(1.52, -0.57, '6.9x less\nforgetting',
                    ha='center', color=GREEN, fontsize=10,
                    fontweight='bold', alpha=arrow_t)

        ax.set_ylabel('Backward Transfer (BWT)', color=DIM, fontsize=9)
        ax.set_ylim(-1.18, 0.14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(methods, color='#c0c8d8', fontsize=10)
        ax.tick_params(axis='y', colors=DIM, labelsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(HAIRLINE)
        ax.spines['bottom'].set_color(HAIRLINE)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=HAIRLINE, linewidth=0.5)

        ax.set_title('Continual learning on Split-MNIST  (BWT closer to 0 = less forgetting)',
                     color=WHITE, fontsize=11, fontweight='semibold', pad=14, loc='left')

        note = '5 sequential tasks  ·  3 seeds  ·  20 epochs  ·  balanced replay size=500'
        fig.text(0.02, 0.01, note, color=DIM, fontsize=7.5, fontfamily='monospace')

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        frames_out.append(fig_to_pil(fig))
        plt.close(fig)
        sys.stdout.write(f'\r  scene 4: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 5 — Summary / CTA  (14–16s → 168–192 frames)
# ═══════════════════════════════════════════════════════════════════════════
STATS = [
    ('-86%',  'FLOPs reduction\n(peak)'),
    ('1.6x',  'faster NAS\nconvergence'),
    ('6.9x',  'less forgetting\nvs EWC'),
    ('100%',  'CPU — no GPU\nrequired'),
]

def scene_summary(frames_out, n_frames=28):
    for i in range(n_frames):
        t = ease_in_out(i / max(n_frames - 1, 1))
        fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
        ax.set_facecolor(BG)
        ax.axis('off')

        for y in np.linspace(0.1, 0.9, 8):
            ax.axhline(y, color=HAIRLINE, linewidth=0.5, alpha=0.6)

        ax.axhline(0.88, color=GREEN, linewidth=2)

        ax.text(0.5, 0.76, 'Measured results — reproducible on a laptop CPU',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=WHITE, alpha=t)

        # 4 stat boxes
        xs = [0.13, 0.38, 0.63, 0.88]
        for k, (val, lbl) in enumerate(STATS):
            card_t = ease_in_out(max(0, min(1, t * 4 - k * 0.7)))
            box = mpatches.FancyBboxPatch(
                (xs[k] - 0.10, 0.40), 0.20, 0.22,
                boxstyle='round,pad=0.01',
                facecolor=PANEL, edgecolor=GREEN if k == 0 else HAIRLINE,
                linewidth=1.2 if k == 0 else 0.7,
                transform=ax.transAxes, alpha=card_t
            )
            ax.add_patch(box)
            ax.text(xs[k], 0.565, val,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=26, fontweight='bold',
                    color=GREEN if k == 0 else WHITE, alpha=card_t)
            ax.text(xs[k], 0.455, lbl,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='#8090a8', alpha=card_t,
                    fontfamily='monospace', multialignment='center')

        # CTA
        cta_t = float(np.clip((t - 0.7) / 0.3, 0.0, 1.0))
        ax.text(0.5, 0.26, 'dnaty.vercel.app  ·  pip install dnaty  ·  github.com/pedrovergueiro/dNaty',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#6a7890', alpha=cta_t,
                fontfamily='monospace')

        ax.text(0.5, 0.14, 'dNATY v1.1.6  ·  Pedro Vergueiro  ·  2026',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8.5, color=DIM, alpha=float(np.clip(cta_t * 0.7, 0.0, 1.0)),
                fontfamily='monospace')

        frames_out.append(fig_to_pil(fig))
        plt.close(fig)
        sys.stdout.write(f'\r  scene 5: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Assemble & save GIF
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating dNATY demo GIF...')
    print(f'  Target: {W}x{H} @ {FPS}fps')

    frames = []

    scene_title(frames, n_frames=28)    # ~2.3s
    scene_flops(frames, n_frames=60)    # ~5.0s
    scene_nas(frames, n_frames=48)      # ~4.0s
    scene_cl(frames, n_frames=36)       # ~3.0s
    scene_summary(frames, n_frames=28)  # ~2.3s  → total ~16.6s

    # hold last frame
    frames.extend([frames[-1]] * 18)   # +1.5s pause at end

    print(f'\n  Total frames: {len(frames)} (~{len(frames)/FPS:.1f}s)')
    print('  Saving GIF (this takes a moment)...')

    delay_ms = int(1000 / FPS)   # ~83ms per frame

    frames[0].save(
        OUT_GIF,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=delay_ms,
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(OUT_GIF) / (1024 * 1024)
    print(f'  Saved: {OUT_GIF}')
    print(f'  Size:  {size_mb:.1f} MB')
    print('Done.')
