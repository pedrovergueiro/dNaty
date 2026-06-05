"""
Testa o upload de ZIP com VARIAS estruturas diferentes.
Simula a logica de extracao/normalizacao do endpoint upload_dataset e
roda _load_dataset (treino loader) para confirmar que cada estrutura funciona.
"""
import sys, io, zipfile, tempfile, shutil, re
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dnaty_saas"))

import numpy as np
import pandas as pd
from PIL import Image
from routes.train import _read_table_df, _infer_label_from_filename, TABULAR_EXTS
from models.dnaty_model import _load_dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".ppm", ".pgm"}
_SKIP = {"__MACOSX", "__pycache__", ".git", "annotations", "val", "validation", "test", "testing"}
rng = np.random.default_rng(0)


def mkimg(path, color):
    base = np.array(color, dtype=np.uint8)
    noise = rng.integers(-25, 25, size=(32, 32, 3))
    Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).save(path)


# ── Replica a normalizacao do endpoint (espelha producao) ───────────────────
def normalize_zip(extract_dir: Path):
    def _dir_has_data(d):
        return any(f.is_file() and not f.name.startswith(".") and "__MACOSX" not in str(f)
                   for f in d.rglob("*"))
    def _qualifying_subdirs(d):
        return sorted(s.name for s in d.iterdir()
                      if s.is_dir() and s.name not in _SKIP and not s.name.startswith("_") and _dir_has_data(s))
    def _find_root(base, max_depth=6):
        for pr in ("train", "Train", "training", "images", "data", "dataset"):
            p = base / pr
            if p.exists() and p.is_dir():
                subs = _qualifying_subdirs(p)
                if len(subs) >= 2: return p, subs
        cur = base
        for _ in range(max_depth):
            subs = _qualifying_subdirs(cur)
            if len(subs) >= 2: return cur, subs
            if len(subs) == 1: cur = cur / subs[0]; continue
            break
        return base, []

    class_root, classes = _find_root(extract_dir)
    if not classes:
        all_files = [f for f in extract_dir.rglob("*") if f.is_file() and not f.name.startswith(".") and "__MACOSX" not in str(f)]
        flat_images = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]
        label_map = {}
        for f in flat_images:
            lbl = _infer_label_from_filename(f.name)
            if lbl: label_map.setdefault(lbl, []).append(f)
        usable = {k: v for k, v in label_map.items() if len(v) >= 2}
        if flat_images and len(usable) >= 2:
            for lbl, files in usable.items():
                d = extract_dir / lbl; d.mkdir(exist_ok=True)
                for f in files:
                    if not (d / f.name).exists(): shutil.copy2(str(f), str(d / f.name))
            classes = sorted(usable.keys())
        else:
            d = extract_dir / "dataset"; d.mkdir(exist_ok=True)
            for f in all_files:
                if not (d / f.name).exists(): shutil.copy2(str(f), str(d / f.name))
            classes = ["dataset"]
    elif class_root != extract_dir:
        for c in classes:
            src, dst = class_root / c, extract_dir / c
            if not dst.exists(): shutil.move(str(src), str(dst))

    # cleanup fantasmas
    _cset = set(classes)
    for item in list(extract_dir.iterdir()):
        try:
            if item.is_dir() and item.name not in _cset:
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
        except Exception:
            pass

    all_data_files = [f for c in classes for f in (extract_dir / c).rglob("*")
                      if f.is_file() and not f.name.startswith(".") and "__MACOSX" not in str(f)]
    has_images = any(f.suffix.lower() in IMAGE_EXTS for f in all_data_files)
    return classes, all_data_files, has_images


def build_and_test(name, builder):
    tmp = Path(tempfile.mkdtemp(prefix="ztest_"))
    extract = tmp / "extracted"; extract.mkdir(parents=True)
    builder(extract)
    try:
        classes, files, has_images = normalize_zip(extract)
        if has_images:
            tr, te, insize, ncls = _load_dataset("custom", "cpu", custom_path=str(extract))
            print(f"  [{name}] OK  IMAGE  classes={ncls} files={len(files)} train={len(tr.dataset)} test={len(te.dataset)}")
        else:
            # tabular path: read each tabular file
            frames = []
            for f in sorted(files):
                if f.suffix.lower() not in TABULAR_EXTS: continue
                df = _read_table_df(f)
                if df is not None and not df.empty:
                    if len(classes) > 1: df["_label_"] = f.parent.name
                    frames.append(df)
            if not frames:
                print(f"  [{name}] FALHA TABULAR: nenhum frame lido"); return
            combined = pd.concat(frames, ignore_index=True)
            print(f"  [{name}] OK  TABULAR classes={len(classes)} rows={len(combined)} cols={combined.shape[1]}")
    except Exception as e:
        import traceback
        print(f"  [{name}] ERRO: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Builders de estrutura ───────────────────────────────────────────────────
def s_imagefolder(ex):
    for cls, col in [("gato", (200,30,30)), ("cao", (30,200,30)), ("passaro", (30,30,200))]:
        d = ex/cls; d.mkdir()
        for i in range(8): mkimg(d/f"{i}.jpg", col)

def s_nested_train(ex):
    for cls, col in [("a", (200,30,30)), ("b", (30,200,30))]:
        d = ex/"train"/cls; d.mkdir(parents=True)
        for i in range(8): mkimg(d/f"{i}.png", col)

def s_nested_deep(ex):  # imagens em subpastas dentro da classe
    for cls, col in [("a", (200,30,30)), ("b", (30,200,30))]:
        d = ex/cls/"batch1"; d.mkdir(parents=True)
        for i in range(8): mkimg(d/f"{i}.jpg", col)

def s_flat_prefix(ex):  # Kaggle dogs-vs-cats: cat.0.jpg, dog.0.jpg flat
    for i in range(10): mkimg(ex/f"cat.{i}.jpg", (200,30,30))
    for i in range(10): mkimg(ex/f"dog.{i}.jpg", (30,200,30))

def s_flat_nolabel(ex):  # imagens soltas sem padrao → 1 classe
    for i in range(10): mkimg(ex/f"img{i:03d}.jpg", (120,120,120))

def s_single_xlsx(ex):  # ZIP com 1 planilha
    df = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100), "label": rng.integers(0,3,100)})
    df.to_excel(ex/"data.xlsx", index=False)

def s_single_parquet(ex):
    df = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100), "label": rng.integers(0,2,100)})
    df.to_parquet(ex/"data.parquet")

def s_csvs_per_class(ex):  # CSVs em pastas de classe
    for cls in ["positivo", "negativo"]:
        d = ex/cls; d.mkdir()
        df = pd.DataFrame({"x1": rng.normal(size=50), "x2": rng.normal(size=50)})
        df.to_csv(d/"part.csv", index=False)

def s_images_plus_junk(ex):  # imagens + README solto + __MACOSX
    for cls, col in [("a", (200,30,30)), ("b", (30,200,30))]:
        d = ex/cls; d.mkdir()
        for i in range(8): mkimg(d/f"{i}.jpg", col)
    (ex/"README.txt").write_text("dataset description")
    mac = ex/"__MACOSX"; mac.mkdir(); (mac/"._junk").write_text("x")

print("Testando estruturas de ZIP\n" + "="*70)
build_and_test("imagefolder padrao",      s_imagefolder)
build_and_test("nested train/classe",     s_nested_train)
build_and_test("imagens em subpasta",     s_nested_deep)
build_and_test("flat prefixo (cat/dog)",  s_flat_prefix)
build_and_test("flat sem rotulo",         s_flat_nolabel)
build_and_test("single .xlsx",            s_single_xlsx)
build_and_test("single .parquet",         s_single_parquet)
build_and_test("CSVs por classe",         s_csvs_per_class)
build_and_test("imagens + README + mac",  s_images_plus_junk)
print("="*70 + "\nFim.")
