import zipfile, os

def add_dir(zf, path, arcname):
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.pyc'): continue
            fp = os.path.join(root, file)
            arc = os.path.join(arcname, os.path.relpath(fp, path))
            zf.write(fp, arc)

with zipfile.ZipFile('dnaty_code.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    add_dir(zf, 'dnaty', 'dnaty')
    for f in ['run_experiments.py', 'requirements.txt', 'setup.py']:
        if os.path.exists(f):
            zf.write(f, f)

checks = [
    ('dnaty/operators/mutations.py', 'OPERATORS = list'),
    ('dnaty/core/arch.py', 'BatchNorm1d'),
    ('dnaty/training/local_train.py', 'CosineAnnealingLR'),
    ('dnaty/training/local_train.py', 'label_smoothing'),
    ('dnaty/experiments/fast_dataset.py', 'FastDataset'),
    ('dnaty/evolution/evolver.py', 'early_stop_patience'),
]

with zipfile.ZipFile('dnaty_code.zip') as zf:
    all_ok = True
    for path, token in checks:
        ok = token in zf.read(path).decode()
        name = path.split('/')[-1]
        print(f'{name} [{token[:25]}]: {"OK" if ok else "FALHOU"}')
        if not ok:
            all_ok = False
    print('ZIP OK' if all_ok else 'ZIP COM ERROS')
