# Usage

## 1. Requirements

- Linux x86-64. Development and testing used Ubuntu 24.04 under WSL2 (Windows 11). The command-line entry point was
  written for that setup; see the platform note in Section 3.1.
- An NVIDIA GPU with CUDA 12.x. The default configuration was developed on an RTX 4090 (24 GB). A native-resolution
  OASIS volume ((z, y, x) = 192 × 224 × 160 at 1 mm) needs about 6–8 GB of host RAM and a few GB of GPU memory.
- Python 3.10 with the pinned packages in [`environment.yml`](../environment.yml):

| Package | Version |
|---|---|
| numpy | 2.0.2 |
| scipy | 1.15.2 |
| numba | 0.65.0 |
| torch | 2.5.1 (CUDA 12.1) |
| triton | 3.1.0 |
| nibabel | 5.4.0 |
| surface-distance | DeepMind, commit `ee651c8` (evaluation only) |

The certificate and all acceptance decisions run in binary64 on the CPU (Numba). Results were validated with exactly
these versions; other versions may change floating-point results.

```bash
conda env create -f environment.yml
conda activate seq-vp-3d
```

Single-threaded BLAS is recommended for tests. Registration runs used 8 threads:

```bash
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMBA_NUM_THREADS=8
```

## 2. Data

Image data are **not** distributed with this repository; the licences of OASIS, LPBA40 and IBSR18 do not allow it.
Obtain them from their providers.

### 2.1 Pair format

The solver reads *pair sets*: folders `data/three_d/<set>/` that contain a `selection.json` listing the pairs and,
for each pair, a folder with `pair.npz` and `manifest.json`.

| `pair.npz` key | Content |
|---|---|
| `fixed`, `moving` | float64 intensity volumes, shape `(nz, ny, nx)`, RAS storage order |
| `fixed_seg4`, `moving_seg4`, `fixed_seg35`, `moving_seg35` | label volumes (OASIS); used only for evaluation |
| `initial_u` | zeros, shape `(nz, ny, nx, 3)` |

`manifest.json` records:
- the grid: `shape_zyx`, `spacing_xyz`, `origin_xyz`, `unit` (mm);
- the weights: `weights.lam`, `weights.kappa`;
- the zero-displacement baselines: `baseline_label_metrics`, `baseline_native_mse`;
- for non-OASIS label sets, `label_sets` and `brain_mask_set`.

The displacement `u` lives on the fixed grid; the moving image is pulled back onto it (method.md §1).
Labels never enter the objective.

### 2.2 Preparation scripts

| Script | Pair set |
|---|---|
| `scripts/prepare_stage_3d_l2r_val.py` | the 19 Learn2Reg OASIS validation pairs (from the neurite-OASIS release) |
| `scripts/prepare_stage_3d_t0_tune.py` | 12 tuning pairs drawn by a fixed rule from the Learn2Reg training range |
| `scripts/prepare_stage_3d_lpba40.py` | LPBA40 (delineation space, 56 labels) |

The source data locations are constants at the top of each script (e.g. the neurite-OASIS folder); adjust them to your
installation. All scripts read with pure RAS storage reorientation, without resampling, and refuse to overwrite an
existing set.

To use a new pair set, add a `(name, folder)` entry to `PAIR_DATASETS` in `scripts/run_stage_3d_e48_pg.py`.

## 3. Running a registration

A run has two steps. `prepare` creates the run directory: it freezes a snapshot of the source code, copies the inputs
and writes the full configuration. `run` executes the registration.

### 3.1 Default configuration

```bash
OUT=/path/to/runs/my_pair_$(date -u +%Y%m%dT%H%M%SZ)
PAIR=OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1        # a pair registered in a pair set

python scripts/run_stage_3d_e48_pg.py prepare "$OUT" \
  --strength weak --lean-commits --resolution-exit \
  --pg-metric-policy-levels legacy_q1_jd_affine_always_v1,legacy_q1_jd_affine_always_v1,image_shift_ggn_equilibrated_jd_affine_v3,image_shift_ggn_equilibrated_jd_affine_v3 \
  --level-shares 0.020833,0.0625,0.145833,1 \
  --curvature-weight-levels 1e-3,1e-3,1e-3,0 \
  --geometry-weight-levels 1,1,0.1,0.03 \
  --pre-optimizer "fp32-v1:iterations=50/50/100/60,lr=0.5,cap=0.5,sigma_grad=1,sigma_field=1,mu=1/1/0.01/0.003,alpha=1e-3/1e-3/1e-3/0,kappa_pre=10,j0=0.2,seconds=60,taper=4" \
  --throughput v41 --export-reserve 1 --seconds 90 \
  --pair "$PAIR"

cd "$OUT/snapshot" && python -B scripts/run_stage_3d_e48_pg.py run "$OUT" > "$OUT/run.log" 2>&1; echo $? > "$OUT/exit_code"
```

- `--seconds 90` is the wall-clock budget, with 1 s reserved for export.
- With `--pre-optimizer`, levels 0–2 are replaced by the pre-optimiser, so their level shares have no effect.
- The first run of a new source version compiles Numba and Triton kernels and performs one-time consistency checks
  outside the timed budget. The results are cached in `~/.cache/seq_vp_3d/`; override the location with
  `JD_CURL_JIT_CACHE` and `JD_CURL_FIRST_USE_CACHE`.
- Running each registration as its own process is recommended, ideally under a resource-limited service manager
  such as `systemd-run --user` with a memory ceiling and an external watchdog.
- **Platform note.** Before and during a run, the entry point checks free space (at least 10 GB) on the output
  volume and on the Windows host volume that receives the writes under WSL2: `/mnt/d` when `OUT` is under `/mnt/d`,
  otherwise `/mnt/c`. If that path is not a mount point, the run stops with a resource error. On a native Linux
  system, adapt the `host` volume in `_run_pair` (`scripts/run_stage_3d_e48_pg.py`); this has not been tested.
  The library in `jd_curl/` has no such dependency.

### 3.2 Main options

| Option | Meaning |
|---|---|
| `--strength {weak,strong}` | geometry weight profile; `--geometry-weight-levels` overrides μ = ν per level |
| `--seconds T` | wall-clock budget (supported values: 30, 45, 60, 90, 120, 180, 240, 330, 600, 3600) |
| `--pre-optimizer SPEC` | FP32 pre-optimiser instead of the coarse levels (method.md §4.3) |
| `--pg-metric-policy-levels` | PG metric per level (method.md §4.2) |
| `--curvature-weight-levels` | curvature weight α per level |
| `--throughput vN` | execution profile; changes execution only, not the model |
| `--export-reserve S` | seconds reserved for exporting the final state |

Every other option is an explicit switch that defaults to the reference behaviour. They cover documented
alternatives (for example the LM inner solver `--native-inner lm`) and ablations (`--native-reference`,
`--native-geometry-weights`, `--native-metric-ablation`). `python scripts/run_stage_3d_e48_pg.py --help` lists them all.

## 4. Outputs

Inside `$OUT/pg_weak__four__90s/`:

| File | Content |
|---|---|
| `final_native.npz` | `u`: final displacement, `(nz, ny, nx, 3)`, mm, components (x, y, z); `warped`: warped moving image |
| `final_native.json` | certificate record of the final state (status, minimum JD lower bound, cell counts, configuration) |
| `summary.json` | status (convergence or budget termination), energies, GF and threshold, MSE ratio, counts, timings |
| `labels.json` | Dice per label set before and after registration (evaluation only) |
| `smoothness.json` | log-Jacobian statistics (diagnostic) |
| `commits.json`, `history.jsonl`, `levels.json` | every committed state with its certificate, the outer and inner history, level events |
| `progress.json` | live progress, updated about once per second |

`exit_code` and `run.log` are written in `$OUT`. The status is `converged_native_gradient` only if the original
gradient criterion is met (method.md §6); `native_budget_exhausted` means the budget ended the run.

## 5. Evaluation

- `scripts/evaluate_l2r_style.py out.json RUN_DIR...` computes the Learn2Reg OASIS measures from the final
  displacement: Dice over 35 labels, DSC30, HD95 and SDlogJ. On the Learn2Reg validation data they agree bit for bit
  with the official evaluation script.
- `scripts/folding_metrics.py` computes proxy folding measures for any displacement field: central-difference and
  24-tetrahedra counts. These measure different maps than the certified trilinear map and are reported only as
  diagnostics.

To re-certify any stored displacement independently:

```python
import json
import numpy as np
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.feasibility import certify_map, CompiledEnclosureCertificateConfig

m = json.load(open(f'{OUT}/inputs/{PAIR}/manifest.json'))
nz, ny, nx = m['shape_zyx']; hx, hy, hz = m['spacing_xyz']; x0, y0, z0 = m['origin_xyz']
grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit='mm')
u = np.load(f'{OUT}/pg_weak__four__90s/final_native.npz')['u']
proof = certify_map(u, grid, config=CompiledEnclosureCertificateConfig(delta=0.))
print(proof.status, proof.min_jd_lower_bound)   # 'certified_feasible' and a positive lower bound
```

## 6. Repository layout

| Path | Content |
|---|---|
| `jd_curl/` | library: 2D reference implementation and `three_d/` (geometry, energies, solvers, certificate, multiresolution, GPU backends, pre-optimiser) |
| `jd_curl/three_d/feasibility.py`, `compiled_enclosure.py`, `certificate_prescreen.py`, `interval.py` | full-cell certificate |
| `jd_curl/three_d/sequential.py`, `multiresolution.py`, `solvers/` | outer iteration, level driver, inner solvers |
| `jd_curl/three_d/backends/` | PyTorch / Triton execution backends (generated kernels are produced by `scripts/gen_*.py`) |
| `scripts/run_stage_3d_e48_pg.py` | command-line entry point (`prepare`, `run`) |
| `scripts/run_stage_3d_e19_mri.py` | registration worker |
| `scripts/prepare_stage_3d_*.py` | data preparation |
| `scripts/evaluate_l2r_style.py`, `scripts/folding_metrics.py` | evaluation |
| `docs/method.md`, `docs/usage.md` | documentation |
