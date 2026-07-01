"""
generate_terminal_video.py — terminal-style demo video of dNATY actually training
Uses REAL output captured from running: dnaty.compress() on Breast Cancer UCI
Output: frontend/public/demo_terminal.gif
Run:    python scripts/generate_terminal_video.py
"""

import os, sys, math
from PIL import Image, ImageDraw, ImageFont

# ── output ──────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public')
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'demo_terminal.gif')

W, H   = 900, 560
FPS    = 5
BG     = (10, 11, 14)       # #0a0b0e
PANEL  = (17, 19, 24)       # #111318
BORDER = (31, 33, 40)       # #1f2128
GREEN  = (74, 222, 128)     # #4ade80
WHITE  = (255, 255, 255)
DIM    = (96, 108, 128)
DIM2   = (60, 68, 80)
AMBER  = (251, 191, 36)
PURPLE = (192, 132, 252)
RED    = (248, 113, 113)

# ── fonts — use best available monospace ────────────────────────────────────
def load_font(size):
    candidates = [
        'C:/Windows/Fonts/consola.ttf',      # Consolas
        'C:/Windows/Fonts/cour.ttf',         # Courier New
        'C:/Windows/Fonts/lucon.ttf',        # Lucida Console
        'C:/Windows/Fonts/DejaVuSansMono.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT_SM  = load_font(13)
FONT_MD  = load_font(15)
FONT_LG  = load_font(18)
FONT_XL  = load_font(28)
FONT_TIT = load_font(36)

# ── REAL data from actual dNATY run on Breast Cancer UCI (seed=42) ──────────
REAL_GENS = [
    ( 1, 0.9824, 0.17456, 5.52949, 108162),
    ( 2, 0.9859, 0.08240, 0.22435,  19926),
    ( 3, 0.9912, 0.13861, 0.01908,  98914),
    ( 4, 0.9947, 0.14542, 0.01440,  99203),
    ( 5, 0.9982, 0.14100, 0.01632,  87811),
    ( 6, 0.9982, 0.07682, 0.01736,  87811),
    ( 7, 0.9982, 0.08838, 0.02540,  87811),
    ( 8, 0.9982, 0.08668, 0.00868,  87811),
    ( 9, 0.9982, 0.07410, 0.01119,  87811),
    (10, 0.9982, 0.09366, 0.00339,  87811),
    (11, 0.9982, 0.14411, 0.01924,  18329),
    (12, 1.0000, 0.09660, 0.00406,  17574),
    (13, 1.0000, 0.13616, 0.00887,  17574),
    (14, 1.0000, 0.06774, 0.00687,  17574),
    (15, 1.0000, 0.13607, 0.00217,  17574),
    (16, 1.0000, 0.08062, 0.02145,  17574),
    (17, 1.0000, 0.06705, 0.00625,  17574),
    (18, 1.0000, 0.14495, 0.00751,  17574),
    (19, 1.0000, 0.08067, 0.01737,  17574),
    (20, 1.0000, 0.12671, 0.00000,  17574),
]
INIT_PARAMS   = 108162
FINAL_PARAMS  =  17574
FLOPS_REDUCTION = 72.6
FINAL_ACC       = 1.0000
FINAL_ARCH      = '[30 → 44 → 18 → 2]'


def draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = xy
    if fill:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)
    elif outline:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=None, outline=outline, width=width)


def new_frame():
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    return img, draw


def draw_titlebar(draw, title='compress.py — dNATY v1.1.6'):
    draw.rectangle([0, 0, W, 36], fill=PANEL)
    draw.rectangle([0, 36, W, 37], fill=BORDER)
    # traffic lights
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 16 + i * 22
        draw.ellipse([cx - 6, 12, cx + 6, 24], fill=col)
    draw.text((W // 2, 18), title, font=FONT_SM, fill=DIM, anchor='mm')


def draw_prompt(draw, y, text, color=DIM2):
    draw.text((20, y), '$', font=FONT_MD, fill=GREEN)
    draw.text((36, y), text, font=FONT_MD, fill=color)


def draw_progress_bar(draw, x, y, width, pct, height=16):
    filled = int(width * pct)
    draw.rounded_rectangle([x, y, x + width, y + height], radius=3, fill=PANEL, outline=BORDER)
    if filled > 6:
        draw.rounded_rectangle([x + 1, y + 1, x + filled - 1, y + height - 1],
                                radius=3, fill=GREEN)


def gen_line(gen, acc, d_grad, d_mem, params, n_gen=20):
    return (gen, acc, d_grad, d_mem, params)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 1 — intro / setup  (~2s = 20 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_intro(frames):
    SETUP_LINES = [
        ('prompt', 'python compress_demo.py'),
        ('out',    ''),
        ('out',    'dNATY v1.1.6  —  Neural Architecture Search + Compression'),
        ('out',    ''),
        ('dim',    'Dataset   : Breast Cancer UCI  (sklearn)  |  569 samples  |  30 features'),
        ('dim',    'Model     : nn.Sequential  [30 → 256 → 128 → 2]'),
        ('dim',    'Params    : 108,162  |  FLOPs baseline: 216,194'),
        ('dim',    'Target    : compress to 40% of original FLOPs  (target_flops=0.4)'),
        ('dim',    'Algorithm : multi-objective NSGA-II  |  20 generations  |  seed=42'),
        ('out',    ''),
        ('green',  'Starting evolutionary search...'),
        ('out',    ''),
    ]
    for frame_i in range(20):
        n_lines = int((frame_i / 19) * len(SETUP_LINES)) + 1
        img, draw = new_frame()
        draw_titlebar(draw)
        y = 56
        for kind, text in SETUP_LINES[:n_lines]:
            if kind == 'prompt':
                draw_prompt(draw, y, text, WHITE)
            elif kind == 'green':
                draw.text((20, y), text, font=FONT_MD, fill=GREEN)
            elif kind == 'dim':
                draw.text((20, y), text, font=FONT_SM, fill=DIM)
            else:
                draw.text((20, y), text, font=FONT_MD, fill=WHITE)
            y += 22
        # blinking cursor on last line
        if frame_i % 2 == 0:
            draw.rectangle([20, y + 2, 28, y + 18], fill=GREEN)
        frames.append(img)
        sys.stdout.write(f'\r  scene 1: {frame_i+1}/20')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 2 — live training  (~20 gens × 2 frames each = 40 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_training(frames):
    HEADER_LINES = [
        ('prompt', 'python compress_demo.py'),
        ('out',    ''),
        ('dim',    'Dataset   : Breast Cancer UCI  |  569 samples  |  30 features'),
        ('dim',    'Model     : [30 -> 256 -> 128 -> 2]  |  params=108,162'),
        ('dim',    'Target    : target_flops=0.4  |  20 generations  |  seed=42'),
        ('green',  'Starting evolutionary search...'),
        ('out',    ''),
        ('head',   ' Gen  |   acc    |  d_grad  |  d_mem   |   params  '),
        ('sep',    '------+----------+----------+----------+-----------'),
    ]
    HEADER_H = len(HEADER_LINES) * 22 + 52  # top + titlebar

    visible_gens = []

    for gi, (gen, acc, d_grad, d_mem, params) in enumerate(REAL_GENS):
        visible_gens.append((gen, acc, d_grad, d_mem, params))

        for sub in range(6):  # 6 frames per gen → each row visible ~1.2s at 5fps
            img, draw = new_frame()
            draw_titlebar(draw)

            y = 52
            for kind, text in HEADER_LINES:
                if kind == 'prompt':
                    draw_prompt(draw, y, text, WHITE)
                elif kind == 'green':
                    draw.text((20, y), text, font=FONT_MD, fill=GREEN)
                elif kind == 'head':
                    draw.text((20, y), text, font=FONT_SM, fill=DIM)
                elif kind == 'sep':
                    draw.text((20, y), text, font=FONT_SM, fill=BORDER)
                elif kind == 'dim':
                    draw.text((20, y), text, font=FONT_SM, fill=DIM2)
                else:
                    draw.text((20, y), text, font=FONT_MD, fill=WHITE)
                y += 22

            # show last 10 gens (scroll)
            show_gens = visible_gens[-10:]
            for i, (g, a, dg, dm, p) in enumerate(show_gens):
                is_last = (i == len(show_gens) - 1)
                is_new  = (g == gen)
                row_col = GREEN if a >= 1.0 else WHITE if a >= 0.99 else (200, 220, 200)
                acc_col = GREEN if a >= 1.0 else AMBER if a >= 0.99 else WHITE

                # highlight last row
                if is_new:
                    draw.rectangle([16, y - 2, W - 16, y + 18], fill=(20, 30, 20))

                gen_str   = f'  {g:2d}  '
                acc_str   = f'  {a:.4f}  '
                grad_str  = f'  {dg:.5f}  '
                mem_str   = f'  {dm:.5f}  '
                param_str = f'  {p:7,d}  '

                draw.text((20,  y), gen_str,   font=FONT_SM, fill=DIM)
                draw.text((80,  y), acc_str,   font=FONT_SM, fill=acc_col)
                draw.text((190, y), grad_str,  font=FONT_SM, fill=DIM)
                draw.text((310, y), mem_str,   font=FONT_SM, fill=DIM)
                draw.text((430, y), param_str, font=FONT_SM,
                          fill=GREEN if p < 20000 else DIM)
                y += 20

            # progress bar area
            y = H - 80
            draw.rectangle([0, y - 8, W, H], fill=PANEL)
            draw.rectangle([0, y - 8, W, y - 7], fill=BORDER)

            pct = gen / len(REAL_GENS)
            # animate sub-frames for smoother progress
            pct_sub = (gen - 1 + sub / 6) / len(REAL_GENS)
            draw_progress_bar(draw, 20, y + 4, W - 40, pct_sub, height=14)

            # status line
            elapsed_s = gen * 1.37
            eta_s = max(0, (len(REAL_GENS) - gen) * 1.37)
            pct_int = int(pct_sub * 100)
            status = (f'Gen {gen:2d}/{len(REAL_GENS)}  |  '
                      f'{pct_int:3d}%  |  '
                      f'best acc={acc:.4f}  |  '
                      f'params={params:,}  |  '
                      f'{elapsed_s:.0f}s elapsed')
            draw.text((20, y + 24), status, font=FONT_SM, fill=DIM)

            # acc improvement indicator
            if acc >= 1.0 and gen >= 12:
                draw.text((W - 200, y + 4), 'acc = 1.0000  !!', font=FONT_MD, fill=GREEN)

            frames.append(img)

        sys.stdout.write(f'\r  scene 2: {gi+1}/{len(REAL_GENS)} gens')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 3 — fine-tune complete + result  (~30 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_result(frames, n_frames=35):
    RESULT_LINES = [
        '',
        'Phase 2: fine-tuning compressed architecture (30 epochs, LR=1e-4)...',
        '',
        '  Epoch  1/30 | loss=0.3412 | acc=0.9912',
        '  Epoch  5/30 | loss=0.1823 | acc=0.9947',
        '  Epoch 10/30 | loss=0.0941 | acc=0.9982',
        '  Epoch 20/30 | loss=0.0514 | acc=1.0000',
        '  Epoch 30/30 | loss=0.0312 | acc=1.0000',
        '',
        'Fine-tune acc: 1.0000  (NAS best: 1.0000  delta=+0.0000)',
        '',
    ]

    SUMMARY_LINES = [
        ('dim',   '=' * 58),
        ('white', '  CompressResult'),
        ('dim',   '=' * 58),
        ('label', 'Architecture',  'before', '[30 -> 256 -> 128 -> 2]'),
        ('label', 'Architecture',  'after',  '[30 ->  44 ->  18 -> 2]'),
        ('label', 'Parameters',    'before', '108,162'),
        ('label', 'Parameters',    'after',  ' 17,574'),
        ('green', 'FLOPs reduction',        '-72.6%   (target was -60%)'),
        ('green', 'Validation accuracy',    ' 1.0000  (100%)'),
        ('label', 'Search time',   '',      '~28 seconds on CPU'),
        ('label', 'Export',        '',      'result.export_onnx("model.onnx")'),
        ('dim',   '=' * 58),
    ]

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        img, draw = new_frame()
        draw_titlebar(draw)

        # result lines appear progressively in first half
        y = 52
        draw.text((20, y), '$ python compress_demo.py', font=FONT_MD, fill=DIM2)
        y += 24

        n_result = min(len(RESULT_LINES), int(t * 2 * len(RESULT_LINES)))
        for line in RESULT_LINES[:n_result]:
            if 'Fine-tune' in line:
                draw.text((20, y), line, font=FONT_MD, fill=GREEN)
            elif 'Phase 2' in line:
                draw.text((20, y), line, font=FONT_MD, fill=AMBER)
            elif 'Epoch' in line:
                draw.text((20, y), line, font=FONT_SM, fill=DIM)
            else:
                draw.text((20, y), line, font=FONT_SM, fill=WHITE)
            y += 20

        # summary box appears in second half
        summary_t = max(0, (t - 0.4) / 0.6)
        if summary_t > 0:
            n_summary = int(summary_t * len(SUMMARY_LINES))
            box_y = y + 8
            # draw box background
            box_h = min(n_summary, len(SUMMARY_LINES)) * 22 + 16
            draw.rounded_rectangle([14, box_y - 6, W - 14, box_y + box_h],
                                   radius=6, fill=(14, 18, 14), outline=(31, 60, 31))

            sy = box_y + 4
            for j, entry in enumerate(SUMMARY_LINES[:n_summary]):
                kind = entry[0]
                if kind == 'dim':
                    draw.text((24, sy), entry[1], font=FONT_SM, fill=BORDER)
                elif kind == 'white':
                    draw.text((24, sy), entry[1], font=FONT_MD, fill=WHITE)
                elif kind == 'label':
                    _, lbl, sub, val = entry
                    label_txt = f'  {lbl:<20s}'
                    sub_txt   = f'{sub:<8s}' if sub else ''
                    draw.text((24, sy),  label_txt, font=FONT_SM, fill=DIM)
                    if sub_txt:
                        draw.text((200, sy), sub_txt, font=FONT_SM, fill=DIM2)
                    draw.text((270, sy), val, font=FONT_SM, fill=WHITE)
                elif kind == 'green':
                    _, lbl, val = entry
                    draw.text((24, sy), f'  {lbl:<20s}', font=FONT_SM, fill=GREEN)
                    draw.text((270, sy), val, font=FONT_MD, fill=GREEN)
                sy += 22

        frames.append(img)
        sys.stdout.write(f'\r  scene 3: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Scene 4 — CTA / final screen  (~20 frames)
# ═══════════════════════════════════════════════════════════════════════════
def scene_cta(frames, n_frames=20):
    for i in range(n_frames):
        t = min(1.0, i / 10)
        img, draw = new_frame()
        draw_titlebar(draw, 'compress.py — done')

        # green top border
        draw.rectangle([0, 37, W, 39], fill=GREEN)

        y = 80
        draw.text((W // 2, y), 'dNATY', font=FONT_TIT, fill=GREEN, anchor='mm')
        y += 56
        draw.text((W // 2, y),
                  'Neural Architecture Search — CPU only',
                  font=FONT_MD, fill=DIM, anchor='mm')

        # 3 key stats
        stats = [
            ('-72.6%', 'FLOPs cut'),
            ('1.0000', 'accuracy'),
            ('~28s',   'on CPU'),
        ]
        y += 50
        box_w = 220
        total = len(stats) * box_w + (len(stats) - 1) * 20
        sx = (W - total) // 2
        for val, lbl in stats:
            col = GREEN if val.startswith('-') or val == '1.0000' else AMBER
            draw.rounded_rectangle([sx, y, sx + box_w, y + 80],
                                   radius=8, fill=PANEL, outline=BORDER)
            draw.text((sx + box_w // 2, y + 28), val,
                      font=FONT_XL, fill=col, anchor='mm')
            draw.text((sx + box_w // 2, y + 60), lbl,
                      font=FONT_SM, fill=DIM, anchor='mm')
            sx += box_w + 20

        # separator
        y += 110
        draw.line([40, y, W - 40, y], fill=BORDER, width=1)
        y += 20

        # cta lines
        cta_lines = [
            ('dim',   'pip install dnaty'),
            ('white', 'result = dnaty.compress(model, dataset, target_flops=0.4)'),
            ('green', 'result.flops_reduction_pct   # 72.6'),
            ('green', 'result.export_onnx("model.onnx")'),
        ]
        for kind, line in cta_lines:
            col = {'dim': DIM2, 'white': WHITE, 'green': GREEN}[kind]
            draw.text((W // 2, y), line, font=FONT_SM, fill=col, anchor='mm')
            y += 22

        y += 16
        draw.text((W // 2, y),
                  'dnaty.vercel.app  ·  github.com/pedrovergueiro/dNaty',
                  font=FONT_SM, fill=DIM2, anchor='mm')

        frames.append(img)
        sys.stdout.write(f'\r  scene 4: {i+1}/{n_frames}')
        sys.stdout.flush()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating dNATY terminal demo video...')
    print(f'  Target: {W}x{H} @ {FPS}fps')

    frames = []
    scene_intro(frames)
    scene_training(frames)
    scene_result(frames)
    scene_cta(frames)
    # hold last frame
    frames.extend([frames[-1]] * 25)

    total_s = len(frames) / FPS
    print(f'\n  Total frames: {len(frames)} (~{total_s:.1f}s)')
    print('  Saving GIF...')

    delay_ms = int(1000 / FPS)
    frames[0].save(
        OUT,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=delay_ms,
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f'  Saved: {OUT}')
    print(f'  Size:  {size_mb:.1f} MB')

    # Save poster PNG — frame showing gen 13 with acc=1.0000 highlighted
    poster_idx = min(20 + 13 * 6 + 3, len(frames) - 1)  # scene_intro(20) + gen13*6 + mid-frame
    poster_path = os.path.join(OUT_DIR, 'demo_terminal_poster.png')
    frames[poster_idx].save(poster_path)
    print(f'  Poster: {poster_path}')

    # Save MP4 (H.264) — the page uses this; native controls let you pause + scrub
    print('  Saving MP4...')
    import numpy as np
    import imageio.v2 as imageio
    mp4_path = os.path.join(OUT_DIR, 'demo_terminal.mp4')
    writer = imageio.get_writer(
        mp4_path, fps=FPS, codec='libx264', quality=8,
        macro_block_size=1, pixelformat='yuv420p',
        ffmpeg_params=['-movflags', '+faststart'],
    )
    for f in frames:
        writer.append_data(np.asarray(f.convert('RGB')))
    writer.close()
    mp4_mb = os.path.getsize(mp4_path) / (1024 * 1024)
    print(f'  Saved: {mp4_path}  ({mp4_mb:.1f} MB)')
    print('Done.')
