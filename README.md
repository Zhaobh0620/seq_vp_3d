# seq_vp_3d

**3D deformable image registration with a certified positive Jacobian determinant on every cell.**

`seq_vp_3d` registers a moving 3D image to a fixed one with a trilinear (Q1) displacement field. Every state the
solver accepts, including the final result, carries a computer-verified proof that the Jacobian determinant of the
trilinear map is strictly positive on every cell. The proof covers every point of every closed cell, not only grid
nodes or sample points, and is computed in outward-rounded interval arithmetic.

## Method in brief

- **Model.** The displacement is Q1 on the voxel grid and zero on the domain boundary. The outer energy is
  image MSE + a local log-barrier on the Jacobian determinant (+ an optional curvature term). Each outer step
  minimises this energy plus proximity and **JD–curl** change penalties, which penalise the change of the Jacobian
  determinant and of the curl relative to the last accepted state (a sequential, self-referenced proximal scheme).
- **Solver.** Preconditioned gradient steps under a Poisson-type metric applied with discrete sine transforms, with
  image-curvature shift, GGN equilibration and Jacobian-aware damping. Steps use Barzilai–Borwein sizes, an exact cubic
  prediction of the first Jacobian root along the step, and a stable Armijo test. By default, the initial guess comes from a
  fast single-precision pre-optimiser run on an 8/4/2/1 pyramid. It is certified before the certified solver takes
  over on the native grid.
- **Certificate.** The Jacobian determinant is triquadratic on each cell. Its Bernstein coefficients are enclosed in
  interval arithmetic, with subdivision where needed. An edge-hull pre-proof and a floating-point prescreen with an
  a priori rounding-error bound make this fast without weakening it. Undecided cells count as failures.
- **Precision.** The model is evaluated in binary64 on CPU and GPU (PyTorch, Triton). The certificate is decided on
  the CPU in binary64 with outward rounding. Single precision appears only in the pre-optimiser, whose output is
  certified before use, and in an energy screen whose error band was calibrated empirically against binary64
  ([details](docs/method.md#7-precision-and-execution)).

The full description, with all formulas, is in [docs/method.md](docs/method.md).

## What is and is not claimed

- Every exported displacement has a certified Jacobian determinant > 0 on every closed cell of the trilinear map,
  vanishes on the boundary and samples only inside the image domain. `final_native.json` records the certificate,
  and [docs/usage.md](docs/usage.md#5-evaluation) shows how to re-certify any stored field independently.
- Global invertibility of the map follows from these properties by a degree argument. The formal proof is in
  preparation and is not claimed here.
- The default configuration stops on a wall-clock budget (90 s). A run is reported as *converged* only if the
  original gradient criterion is met; otherwise its status says the budget ended it.
- Segmentation labels are used only for evaluation and never enter the objective.

## Installation

Linux x86-64 with an NVIDIA GPU (CUDA 12.x) is required. The code was developed and validated with the pinned
versions in [environment.yml](environment.yml).

```bash
git clone https://github.com/Zhaobh0620/seq_vp_3d.git
cd seq_vp_3d
conda env create -f environment.yml
conda activate seq-vp-3d
```

## Quick start

Image data are not distributed with this repository. Prepare a pair set as described in
[docs/usage.md §2](docs/usage.md#2-data), then:

```bash
python scripts/run_stage_3d_e48_pg.py prepare "$OUT" <default options, see docs/usage.md §3.1> --pair "$PAIR"
cd "$OUT/snapshot" && python -B scripts/run_stage_3d_e48_pg.py run "$OUT"
```

The result is written to `$OUT/pg_weak__four__90s/final_native.npz` (displacement `u` in mm, and the warped
image), with the certificate in `final_native.json`.

## Documentation

| Document | Content |
|---|---|
| [docs/method.md](docs/method.md) | mathematical model, discretisation, algorithm, certificate, default parameters |
| [docs/usage.md](docs/usage.md) | requirements, data format, running, outputs, evaluation, repository layout |

## Notes on the code

- The code is the frozen version used for the results in the accompanying manuscript (tag `v1.0`). Apart from the
  public documentation, only comments and docstrings were edited for this release; the program logic is unchanged.
- Code comments refer to entries of the internal design record by identifier (for example `T76`, `E106`, `v41`).
  That record is not part of this release; [docs/method.md](docs/method.md) keeps the identifiers of the relevant
  model sections for orientation.
- Many components are explicit options that default to the reference behaviour. They are documented alternatives,
  execution profiles or ablation switches; the default configuration is listed in
  [docs/method.md §8](docs/method.md#8-default-configuration).
- The command-line entry point was written for Ubuntu under WSL2. See the platform note in
  [docs/usage.md §3.1](docs/usage.md#31-default-configuration).
- The regression test suite is not yet included in this release.

## Citation

A manuscript describing the method is in preparation. Until it is available, please contact the author before
using or citing this work.

## License

No license is granted at this time; all rights are reserved by the author.
