"""FX1 F2' (throughput-v35; results/verification/3d_stage_FX1_v1/plan/protocol.md
revision 2). The first-use bitwise checks of three fused-kernel families -- E93 witness-screen prefilter, E94 fused node JD
with E100's estimator overlap, E99 skip kernels and tuned launch shapes -- run before the task clock, in a dry pass of the
worker on a synthetic pair with the run's native grid geometry (shape, spacing, origin; never the pair's image data) and a
synthetic native seed with strongly compressed regions (barrier-active closed27 points, T69/T73 candidates).

harvest(backend) takes the keys that the dry pass's backend verified (geometry tuples or [kernel, grid id]; the tuned
shapes' register counts of this process's binaries); seed_backend(backend, registry) puts them into the run's new
backend, which then treats them as checked: its own first use of those keys runs the fast path directly. Values are
unchanged wherever the check passes on the run's data as well (every check compares a fast path with its reference
bit for bit); the difference of scope -- synthetic field instead of the run's first state -- is the user-approved
change. A family switched off in the dry pass contributes nothing (the run checks it itself, as before); every other
first-use check of the run is unchanged."""
import numpy as np

REGISTRY_POLICY = "first-use-prewarm-synthetic-v1"
FAMILIES = ("screen_prefilter", "fused_node_jd", "estimator_overlap", "barrier_skip", "tuned_launch")
# v38 (gpu_hull_first_use_seed): T84's synthetic first-use comparison (the half of the first use that does not depend on the
# run's input). Harvested and seeded only for a backend with the switch, so that the v35-v37 registries are unchanged.
EXTRA_FAMILIES = ("gpu_edge_hull",)


def _geometry_keys(keys):
    """{(shape, spacing, x)} -> JSON-ready sorted [[shape], [spacing], x]."""
    return sorted(([int(v) for v in k[0]], [float(v) for v in k[1]], k[2]) for k in keys)


def harvest(backend):
    out = dict(policy=REGISTRY_POLICY, families={}, skipped={})

    def take(name, on, record, keys):
        if not on:
            out["skipped"][name] = "not_in_profile"
        elif record.get("disabled"):
            out["skipped"][name] = "switched_off_in_dry_pass"
        elif not keys:
            out["skipped"][name] = "not_exercised_in_dry_pass"
        else:
            out["families"][name] = keys

    take("screen_prefilter", getattr(backend, "screen_prefilter", False), backend.screen_prefilter_record,
         [[a, b, float(c)] for a, b, c in _geometry_keys(backend._prefilter_verified)])
    take("fused_node_jd", getattr(backend, "fused_node_jd", False), backend.fused_node_jd_record,
         [[a, b, bool(c)] for a, b, c in _geometry_keys(backend._fused_node_verified)])
    take("estimator_overlap", getattr(backend, "estimator_overlap", False), backend.estimator_overlap_record,
         [[a, b, bool(c)] for a, b, c in _geometry_keys(backend._overlap_verified)])
    take("barrier_skip", getattr(backend, "barrier_skip", False), backend.barrier_skip_record,
         sorted([str(n), str(g)] for n, g in backend.barrier_skip_record["verified"]))
    tuned = getattr(backend, "tuned_launch_record", None)
    take("tuned_launch", getattr(backend, "tuned_launch", False), tuned or {},
         None if tuned is None or not tuned["verified"] or tuned.get("low_registers") else
         dict(verified=sorted([str(n), str(g)] for n, g in tuned["verified"]),
              registers={str(k): int(v) for k, v in tuned["registers"].items()}))
    if getattr(backend, "gpu_hull_first_use_seed", False):
        # v38: the (grid id, delta hex) keys whose synthetic folding field matched the CPU margins bit for bit in the dry pass
        hull = getattr(backend, "_gpu_hull", None)
        record = getattr(backend, "gpu_certificate_record", None) or {}
        take("gpu_edge_hull", getattr(backend, "gpu_edge_hull", False), record,
             [] if hull is None else sorted([str(g), str(d)] for (g, d), diff in hull._synthetic.items() if diff is None))
        if getattr(backend, "gpu_edge_hull", False) and record.get("disabled"):
            # failure evidence (hard constraint 5): a T84 mismatch or error in the dry pass is kept with its details
            out["evidence"] = dict(gpu_edge_hull={k: record.get(k) for k in ("mismatches", "mismatch_details", "errors",
                                                                             "error_types")})
    return out


def seed_backend(backend, registry):
    """Seed a new backend (same profile) with the dry pass's keys; returns the record for the run's summary."""
    fam = registry.get("families", {})
    seeded = {}
    shape = lambda k: (tuple(int(v) for v in k[0]), tuple(float(v) for v in k[1]))
    if getattr(backend, "screen_prefilter", False) and "screen_prefilter" in fam:
        backend._prefilter_verified.update(shape(k)+(float(k[2]),) for k in fam["screen_prefilter"])
        backend.screen_prefilter_record["preverified"] = fam["screen_prefilter"]
        seeded["screen_prefilter"] = len(fam["screen_prefilter"])
    if getattr(backend, "fused_node_jd", False) and "fused_node_jd" in fam:
        backend._fused_node_verified.update(shape(k)+(bool(k[2]),) for k in fam["fused_node_jd"])
        backend.fused_node_jd_record["preverified"] = fam["fused_node_jd"]
        seeded["fused_node_jd"] = len(fam["fused_node_jd"])
    if getattr(backend, "estimator_overlap", False) and "estimator_overlap" in fam:
        backend._overlap_verified.update(shape(k)+(bool(k[2]),) for k in fam["estimator_overlap"])
        backend.estimator_overlap_record["preverified"] = fam["estimator_overlap"]
        seeded["estimator_overlap"] = len(fam["estimator_overlap"])
    if getattr(backend, "barrier_skip", False) and "barrier_skip" in fam:
        # the membership test of triton_q1 reads record["verified"]; the seeded entries are also listed as preverified
        backend.barrier_skip_record["verified"].extend([n, g] for n, g in fam["barrier_skip"])
        backend.barrier_skip_record["preverified"] = fam["barrier_skip"]
        seeded["barrier_skip"] = len(fam["barrier_skip"])
    if getattr(backend, "tuned_launch", False) and "tuned_launch" in fam:
        record = backend.tuned_launch_record
        record["verified"].extend([n, g] for n, g in fam["tuned_launch"]["verified"])
        record["registers"].update(fam["tuned_launch"]["registers"])   # the same compiled binaries (this process)
        record["preverified"] = fam["tuned_launch"]
        seeded["tuned_launch"] = len(fam["tuned_launch"]["verified"])
    if (getattr(backend, "gpu_hull_first_use_seed", False) and getattr(backend, "gpu_edge_hull", False)
            and "gpu_edge_hull" in fam):
        # v38: the backend puts these keys into its GpuEdgeHull's synthetic comparisons when it creates it; the first use on
        # the run's own input is then the only comparison left (edge_hull_gpu.GpuEdgeHull.margins)
        backend._gpu_hull_seed = {(str(g), str(d)): None for g, d in fam["gpu_edge_hull"]}
        backend.gpu_certificate_record["synthetic_preverified"] = fam["gpu_edge_hull"]
        seeded["gpu_edge_hull"] = len(fam["gpu_edge_hull"])
    return dict(policy=REGISTRY_POLICY, seeded=seeded, skipped=registry.get("skipped", {}),
                **({"evidence": registry["evidence"]} if registry.get("evidence") else {}),      # v38: dry-pass T84 evidence
                role="first-use checks of these keys were done before the task clock on a synthetic field of this grid")


def synthetic_inputs(manifest, seed=35):
    """(arrays, grid): a synthetic pair on the run's native grid (shape_zyx, spacing_xyz, origin_xyz of its manifest, as
    e11's inputs() builds it): sums of 24 separable Gaussian blobs scaled to [0, 1]; the moving image has the same blobs
    shifted by (1.5, -1.0, 0.8) voxels (z, y, x). Zero initial field, zero label maps. Deterministic."""
    from ..grid import make_grid_3d
    nz, ny, nx = (int(v) for v in manifest["shape_zyx"])
    hx, hy, hz = (float(v) for v in manifest["spacing_xyz"])
    x0, y0, z0 = (float(v) for v in manifest["origin_xyz"])
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit="mm")
    rng = np.random.default_rng(seed)
    centres = rng.uniform(.15, .85, size=(24, 3))*np.array([nz-1, ny-1, nx-1])
    widths, amps = rng.uniform(6., 18., size=24), rng.uniform(.3, 1., size=24)
    axes = (np.arange(nz, dtype=np.float64), np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64))

    def volume(shift):
        out = np.zeros((nz, ny, nx))
        for c, w, a in zip(centres+np.asarray(shift), widths, amps):
            ez, ey, ex = (np.exp(-(t-ci)**2/(2*w*w)) for t, ci in zip(axes, c))
            out += a*ez[:, None, None]*ey[None, :, None]*ex[None, None, :]
        return out
    fixed, moving = volume((0., 0., 0.)), volume((1.5, -1., .8))
    top = max(fixed.max(), moving.max())
    labels = np.zeros((nz, ny, nx), dtype=np.int16)
    arrays = dict(fixed=fixed/top, moving=moving/top, initial_u=np.zeros((nz, ny, nx, 3)),
                  fixed_seg4=labels, moving_seg4=labels, fixed_seg35=labels, moving_seg35=labels)
    return arrays, grid


def synthetic_seed(grid, centres=((.3, .3, .3), (.7, .35, .6), (.35, .7, .5), (.6, .62, .72)), c=.75, sigma_vox=6.):
    """Radial compression bumps u = -c (X - X0) exp(-|X - X0|^2 / (2 s^2)), s = sigma_vox voxels, at the given fractions
    (z, y, x) of the grid: J = (1 - c)^3 = .0156 at each centre (closed27 points below the barrier activation .05, T69/T73
    candidates) and J > 0 everywhere (radial stretch 1 - c g (1 - r^2/s^2) >= 1 - c, tangential 1 - c g); the six faces
    are exactly zero (the pre-optimizer contract)."""
    X = grid.identity()
    s = sigma_vox*float(min(grid.spacing))
    u = np.zeros_like(X)
    shape = np.array(grid.shape)
    for f in centres:
        i = tuple(int(round(a*(n-1))) for a, n in zip(f, shape))
        d = X-X[i]
        u -= c*d*np.exp(-(d*d).sum(-1)/(2*s*s))[..., None]
    for axis in range(3):
        index = [slice(None)]*4
        for end in (0, -1):
            index[axis] = end
            u[tuple(index)] = 0.
    return u
