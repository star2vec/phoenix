# Windows (CUDA) setup — RTX 2000 Ada

Target: i9-13950HX / 32 GB / RTX 2000 Ada 8 GB. Training runs here; the Mac
clone (no vendor/, no ckpts/) is for code, documents, and CPU tests only.

```powershell
# 1. clone the project and the reference code (vendor/ is gitignored; re-clone)
git clone <REPO_URL> phoenix ; cd phoenix
git clone --depth 1 https://github.com/Ber666/reasoning-by-superposition.git vendor/reasoning-by-superposition

# 2. env — python 3.12, pins from the reference paper + CUDA torch
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/Scripts/python numpy==2.1.3 transformers==4.46.2 datasets==3.1.0 tqdm==4.67.0 pyyaml
```

Notes:
- Disable Windows sleep for long runs (Settings > Power); an idle-sleep once
  stopped a training run once.
- Check GPU co-residents with `nvidia-smi` before a run.
- On the Mac the same steps apply with `.venv/bin/python` and the default
  (CPU) torch wheel.
