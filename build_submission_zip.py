from pathlib import Path
import zipfile

root = Path('d:/Dissertation/Code')
zip_path = root / 'submission_package.zip'
include_paths = []

for name in [
    'README.md',
    'requirements.txt',
    'LICENSE.md',
    'continuous_driver.py',
    'discrete_driver.py',
    'encoder_init.py',
    'location.py',
    'parameters.py',
]:
    p = root / name
    if p.exists():
        include_paths.append(p)

for p in root.rglob('*'):
    if not p.is_file():
        continue
    s = str(p)
    if 'autoencoder' in s and 'autoencoder/model' not in s:
        include_paths.append(p)
    elif 'networks' in s or 'simulation' in s:
        include_paths.append(p)

pyproj = root / 'poetry' / 'pyproject.toml'
if pyproj.exists():
    include_paths.append(pyproj)

include_paths = sorted(set(include_paths))

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in include_paths:
        z.write(p, p.relative_to(root))

print(f'Created {zip_path}')
print(f'Size {zip_path.stat().st_size / 1024 / 1024:.2f} MB')
print(f'Files included: {len(include_paths)}')
