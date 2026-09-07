# Laptop (Windows, offline): Amendment 1 training

The laptop has no GitHub access. Everything arrives in one transfer folder:

    phoenix.bundle          the whole repository, every commit (git bundle)
    LAPTOP.md               this file
    data-relabel\           the relabeled splits and manifest.json (fallback;
                            step 5 regenerates them from the seed and checks
                            the hashes against this manifest)
    wheels\                 empty unless step 3 reports a missing package

Assumed layout on the laptop, as in WINDOWS_SETUP.md: the RRR clone at
`<dev>\RRR` with its `.venv` (Python 3.12; torch 2.5.1 cu121, transformers
4.46.2, numpy 2.1.3, datasets 3.1.0, tqdm 4.67.0, pyyaml) and its
`vendor\reasoning-by-superposition`. phoenix becomes a sibling, `<dev>\phoenix`.
phoenix adds no dependency beyond RRR's pins. All commands are PowerShell,
run from `<dev>` unless the step says otherwise.

## 0. Variables

```powershell
$T   = "D:\transfer"                    # where the transfer folder was copied
$DEV = "C:\Users\<you>\Developer"       # parent of RRR
Set-Location $DEV
```

## 1. Clone from the bundle

```powershell
git clone "$T\phoenix.bundle" phoenix
Set-Location phoenix
git log --oneline -3
```

The clone's `origin` is the bundle file. A later bundle is applied with
`git pull "$T\phoenix.bundle" main` from inside `phoenix`. Nothing is
committed on the laptop; only checkpoints and logs travel back.

## 2. Copy vendor/ from RRR

```powershell
robocopy ..\RRR\vendor\reasoning-by-superposition vendor\reasoning-by-superposition /E /NFL /NDL
Get-ChildItem vendor\reasoning-by-superposition\data      # three prosqa_*_graph_4_coconut.json files
Get-ChildItem vendor\reasoning-by-superposition\configs   # symbol-2layer-8head-768dim.json
```

`vendor/` is gitignored in both repos; phoenix imports it by relative path.

## 3. Python environment (RRR's venv, no activation needed)

```powershell
$PY = "..\RRR\.venv\Scripts\python.exe"
& $PY scripts\check_env.py
```

Expected: four `ok` lines (torch 2.5.1, transformers 4.46.2, numpy 2.1.3,
datasets 3.1.0), `cuda available: True`, and `ENV: OK`. If you prefer an
activated shell: `..\RRR\.venv\Scripts\Activate.ps1`, then `python` in place
of `& $PY`.

If a line says MISSING or VERSION, the offline install path is: on the Mac,
`pip download <pkg>==<ver> --platform win_amd64 --python-version 3.12
--only-binary=:all: -d wheels` for that package, carry `wheels\` over, then
here `& $PY -m pip install --no-index --find-links "$T\wheels" <pkg>==<ver>`.
Do not reinstall torch: the cu121 wheel in RRR's venv is the one to keep.

## 4. Tests before training (README rule: on every machine, before training)

```powershell
& $PY tests\test_fast_equivalence.py    # must print EQUIVALENCE: PASS
& $PY tests\test_relabel.py             # must print RELABEL: PASS
```

## 5. Regenerate the relabeled data and check it against the manifest

```powershell
& $PY src\phoenix\relabel.py --seed 20260906
& $PY scripts\verify_relabel.py --manifest "$T\data-relabel\manifest.json"
```

Expected: three `match` lines and `RELABEL DATA: MATCH`. If a file
mismatches (Python's random module can differ between versions), use the
transfer copies instead and re-check:

```powershell
Copy-Item "$T\data-relabel\*.json" data\relabel\ -Force
& $PY scripts\verify_relabel.py --manifest "$T\data-relabel\manifest.json"
```

## 6. Train seed 0, then seed 1

Disable sleep (Settings > Power) and check `nvidia-smi` for co-residents first.

```powershell
.\scripts\train_relabel.ps1 -Python $PY
```

This runs `train.py` twice in sequence with the amendment's recipe
(`--data-dir data\relabel --fixed-early-stages --full-task-patience 15
--save-every 10`), run names `seed0/relabel` and `seed1/relabel`. About 6 to
10 hours per seed. Resumable: rerun the same command after an interruption.
Paths written (train.py and the script agree on these):

    ckpts\seed0\relabel\best.pt            best full-task checkpoint (needed)
    ckpts\seed0\relabel\latest_state.pt    resume state (optional to copy)
    ckpts\seed0\relabel\epoch_NNN.pt       stage ends and every 10 full-task epochs
    results\seed0\relabel\metrics.jsonl    per-epoch loss and validation accuracy
    results\seed0\relabel\best.json        epoch and accuracy of best.pt
    results\seed0\relabel\train.out        console log

and the same under `seed1`. About 1 GB per seed with the epoch checkpoints.

## 7. Copy checkpoints and logs back

```powershell
foreach ($s in 0, 1) {
  robocopy ckpts\seed$s\relabel   "$T\back\ckpts\seed$s\relabel"   /E /NFL /NDL
  robocopy results\seed$s\relabel "$T\back\results\seed$s\relabel" /E /NFL /NDL
}
```

On the Mac, drop `back\ckpts\seed*\relabel` into `phoenix/ckpts/` and
`back\results\seed*\relabel` into `phoenix/results/` at the same relative
paths. Gate 1 of Amendment 1 then runs `evaluate.py --run-name seed0/relabel
--data-dir data/relabel --serialization-seed {0,1,2,3}` (and seed 1) and
stops with the accuracy numbers.
