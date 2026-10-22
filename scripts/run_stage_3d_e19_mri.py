"""E19-B frozen single/four-level native PG time-budget comparison (T38)."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import errno
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stage_3d_e11_short as e11
from scripts.run_stage_3d_e17_pilot import PAIR, RecordingBackend
from scripts.run_stage_3d_c import dump
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.feasibility import (CompiledEnclosureCertificateConfig, ParallelEnclosureCertificateConfig,
                                         CompiledFallbackCertificateConfig)
from jd_curl.three_d.backends.device_batch import DeviceBatchConfig
from jd_curl.three_d.multiresolution import solve_four_level, TRANSFER_BACKTRACK_POLICY, TRANSFER_BACKTRACK_ALPHAS
from jd_curl.three_d.nifti import file_sha256
from jd_curl.three_d.problem import ParallelMemoryConfig, Weights3D, make_problem
from jd_curl.three_d.sequential import solve_sequential, SequentialReuseConfig3D, mse
from jd_curl.three_d.solvers.poisson_gradient import PGConfig3D, JDStepPGConfig3D
from jd_curl.three_d.work_control import WorkBudgetExceeded, work_session

INNER_STEPS, OUTER_STEPS = 5, 1000
EXPORT_RESERVE = 30.
EXPORT_RESERVE_SHORT = 3.      # S30-G1: budgets <= 60 s keep 3 s for the final export; the longer budgets keep 30 s
EXPORT_RESERVE_SHORT_MAX_SECONDS = 120.   # S30 schedule probe (2026-10-15): the 3 s reserve also covers the 90 s and 120 s
                                          # budgets (measured final export 3.1 s PRE / 4.5 s NM in G3c); 180 s and longer keep 30 s


def export_reserve_for(seconds):
    """S30-G1: the export reserve of a wall budget: 30 s for every pre-S30 budget (bit for bit the old constant),
    3 s for budgets of at most 60 s; S30 schedule probe (2026-10-15): the 3 s rule extends to budgets of at most
    EXPORT_RESERVE_SHORT_MAX_SECONDS (120 s)."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('wall_seconds must be a positive finite number')
    return EXPORT_RESERVE_SHORT if float(seconds) <= EXPORT_RESERVE_SHORT_MAX_SECONDS else EXPORT_RESERVE


def reset_process_counters():
    """S30-G1 (resident service): per-process diagnostic counters that a fresh process starts at zero are reset at
    the start of every pair, and the row-SoA certificate first-use registry is cleared so that its first-use
    comparison is repeated per pair. Records and first-use checks only; no numerical path reads them."""
    reset = []
    mod = sys.modules.get('jd_curl.three_d.solvers.jd_step')
    if mod is not None and hasattr(mod, '_SCREEN_STATS'):
        for key in list(mod._SCREEN_STATS):
            mod._SCREEN_STATS[key] = 0
        reset.append('jd_step._SCREEN_STATS')
    mod = sys.modules.get('jd_curl.three_d.compiled_enclosure')
    if mod is not None and hasattr(mod, '_SOA_VERIFIED'):
        mod._SOA_VERIFIED.clear()
        reset.append('compiled_enclosure._SOA_VERIFIED')
    return reset
CUMULATIVE_SHARES = (.05, .15, .35, 1.)
CPU_BUDGET_POLICY = 'fused-cpu-block-deadline-formal-export-v1'
SOURCE = ROOT/'results/mri_3d/stage_E18_reuse_io_v1/inputs'/PAIR
LOW_LOAD_POLICY = dict(cpu_workers=1, cell_tile_size=2048, spline_query_chunk=16384,
    certificate_tile_size=2048, torch_threads=1, cuda_allocator_limit_bytes=8*1024**3,
    cgroup_memory_high_bytes=6*1024**3, cgroup_memory_max_bytes=8*1024**3,
    cgroup_swap_max_bytes=0, cgroup_cpu_quota_cores=1, nice=10)
# E48 explicit execution profile: sizes/threads/IO only. CPU proofs, quadrature,
# F/E, acceptance, forcing and GF are unchanged; old low-load sizes stay default.
THROUGHPUT_POLICY = dict(name='throughput-v1', cell_tile_size=16384, spline_query_chunk=1 << 20,
    cpu_workers=8, device_tile_size=262144, certificate_tile_size=65536, torch_threads=8,
    noncommit_arrays=False, append_only_logs=True, progress_min_interval_seconds=1.)
# E50/S2: v1 plus the parallel whole-map certificate (same per-cell arithmetic and
# certificate fields) and CPU-proof/device-evaluation overlap (same decisions).
THROUGHPUT_POLICIES = {'v1': THROUGHPUT_POLICY,
    'v2': dict(THROUGHPUT_POLICY, name='throughput-v2', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True),
    # E51/S3: v2 plus fused Triton Q1 geometry kernels (roundoff-level, guarded GPU energies).
    'v3': dict(THROUGHPUT_POLICY, name='throughput-v3', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True),
    # E54 (T65): v3 plus the compiled Bernstein-fallback arrays (bitwise equal certificate).
    'v4': dict(THROUGHPUT_POLICY, name='throughput-v4', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True, compiled_fallback=True),
    # E55b: v4 plus the fused Triton global JD root (used only when T66 jd_step_global is on).
    'v5': dict(THROUGHPUT_POLICY, name='throughput-v5', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_jd_root=True),
    # E65: v4 plus the fused GGN diagonal (T59/T62/T69 metric builds only; roundoff-level vs torch).
    'v6': dict(THROUGHPUT_POLICY, name='throughput-v6', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True),
    # E69: v6 plus the device decision of clearly reached inner gradient targets (CPU recheck near the target).
    'v7': dict(THROUGHPUT_POLICY, name='throughput-v7', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
               cpu_gradient_band=.99),
    # E71/E72: v7 plus the rigorous float Bernstein certificate prescreen (T75) and the fused GGN action.
    'v8': dict(THROUGHPUT_POLICY, name='throughput-v8', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
               cpu_gradient_band=.99, prescreen_certificate=True, fused_hvp=True),
    # E76: v8 with bit-identical host-side shortcuts (inner solver, transfers, digests, bindings),
    # ordered background checkpoint writes, and a relative 1e-6 CPU recheck margin for the inner
    # gradient target instead of the [.99, 1.01] band (device-vs-CPU norms differ by ~1e-12).
    'v9': dict(THROUGHPUT_POLICY, name='throughput-v9', parallel_certificate=True, fallback_workers=8,
               overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
               cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
               lean_host=True, fast_host=True, async_commit_writes=True),
    # E81: v9 plus the compiled T69/T73 face/interior estimator (metric values bit-identical to v9).
    'v10': dict(THROUGHPUT_POLICY, name='throughput-v10', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True),
    # E89: v10 plus bit-identical host work removal: glibc malloc retention (P02), identity facts recorded
    # where they are established and copy-free trial arrays (P01), compiled fused step algebra (P03),
    # outer/inner boundary reuse (P05), lean metric rebuild (P07), lean outer checks (P03/P08).
    # Every value, decision and recorded identity equals v10's.
    'v11': dict(THROUGHPUT_POLICY, name='throughput-v11', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True),
    # E93 (G1/C3): v11 plus the rigorous Triton prefilter of the witness screen: tiles whose legacy closed27
    # minimum is proven > delta are skipped, all others run the legacy screen. Every value, decision and
    # recorded identity equals v11's.
    'v12': dict(THROUGHPUT_POLICY, name='throughput-v12', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True),
    # E94 (G2/C4): v12 plus the fused Triton T62/T69 per-cell measure of node_min_jd, bit for bit the torch path
    # (first use on every level grid checked bit for bit). Every value, decision and recorded identity equals v12's.
    'v13': dict(THROUGHPUT_POLICY, name='throughput-v13', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True),
    # E95 (G3a): v13 plus no rescans of proven-finite arrays (inner secants, direction, step bound, trial entry,
    # gradient norm), the lazy direction digest, and the equilibrated Poisson scalings on the device.
    # Every value, decision and recorded identity equals v13's.
    # E96: revision of the same profile (memory and gating only, values unchanged); runs before E96 have no field.
    'v14': dict(THROUGHPUT_POLICY, name='throughput-v14', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96'),
    # E97 (G3a second batch): v14 plus the fused axpy/step extrema, the early warped download, the trial's device u
    # built on the device and the JD prediction's root screen. Every value, decision and recorded identity equals v14's.
    'v15': dict(THROUGHPUT_POLICY, name='throughput-v15', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True),
    # E98 (G7 + bit-identical G8): v15 plus the row-SoA bit-step certificate kernel, the owned certificate front end,
    # the barrier energy/gradient kernels that skip inactive closed27 points, and the curvature Laplacian memo.
    # Every value, decision, certificate field and recorded identity equals v15's.
    'v16': dict(THROUGHPUT_POLICY, name='throughput-v16', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True),
    # E99 (G8, bit-identical): v16 plus the tuned launch shapes (soft-geometry energy with a base computed once per
    # cell, GGN diagonal maxnreg), the fused cubic B-spline sample (torch's reduction tree) and the skip kernels'
    # reference-first check. Every value, decision and recorded identity equals v16's.
    'v17': dict(THROUGHPUT_POLICY, name='throughput-v17', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True),
    # E100 (bit-identical): v17 plus the node estimator overlapped with the GGN diagonal kernel, the compiled stable
    # top-4 selection of the estimator's Newton starts, and the outer vector's device copy, finiteness and norm reused
    # by the next inner solve's first gradient. Every value, decision and recorded identity equals v17's.
    'v18': dict(THROUGHPUT_POLICY, name='throughput-v18', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True),
    # E101 (D9: T84 with plan D, user 2026-10-09): v18 plus the certificate's edge-hull margins on the GPU bit for bit
    # (inner-solve trial certificates only), every outer candidate re-certified on the CPU before it is committed,
    # and the curvature Laplacian memo with a reference slot and fused kernel. Every value, decision, certificate
    # field and recorded identity equals v18's.
    'v19': dict(THROUGHPUT_POLICY, name='throughput-v19', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=True,
                laplacian_cache=True, fused_laplacian=True),
    # E102 (W1, user 2026-10-09): v19 with each checkpoint serialized and hashed on the ordered writer thread (the
    # recorded SHA-256 is still that of the exact file bytes, files and indexes in the same order) and without T84 plan
    # D (closed by the user 2026-10-09; the first-use checks per grid and the independent CPU audit stay). Every
    # value, decision, certificate field and recorded identity equals v19's.
    'v20': dict(THROUGHPUT_POLICY, name='throughput-v20', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True),
    # E103 (M1, user 2026-10-10): v20 with the next metric build's host parts (the T73/T69 estimator over the prepared
    # rows and the T59 shift alpha) started in a background thread at the inner step budget and used by that build
    # when its base is the same frozen u and image gradient. Every value, decision and recorded identity equals v20's.
    'v21': dict(THROUGHPUT_POLICY, name='throughput-v21', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, metric_prestart=True),
    # E104 (G3b narrowed, user 2026-10-10): v20 with the inner slope g.p and the BB2 tail y.(p_old - p) from the device
    # copies of g and p (ROUND-OFF LEVEL: 1e-6 decision bands, host products inside them; first uses checked against
    # the a-priori dot-product bound) and the gradient's finite check on its device copy (exact). Not bit-identical
    # to v20: accepted under the G3b standard (rounding bound, bands, recorded states, T0 at equal native commits).
    'v22': dict(THROUGHPUT_POLICY, name='throughput-v22', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True),
    # E105 (user 2026-10-11, overnight plan): v22 with the metric build's T59 shift alpha by a compiled pass in NumPy's
    # own reduction order (8192-value buffers, pairwise sums, accumulated in order; first uses per level grid compared
    # bit for bit with the NumPy expression), and each inner gradient's equilibrated Poisson direction queued right after
    # the gradient kernels while the gradient downloads on a side stream (the same kernels in the same order). Bit for
    # bit v22's values; only the 'gradient'/'poisson' timings' synchronization points differ.
    'v23': dict(THROUGHPUT_POLICY, name='throughput-v23', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True),
    # E106 (C14 phase A): v23 with the Poisson metric's spectral solve M^-1 in float32
    # (the direction only; energies, comparisons, gradient, GF and certificates in float64), with range, finiteness,
    # first-use (relative 1e-4) and descent guards. NOT bit-identical to v23: accepted on T0 at equal native commits.
    'v24': dict(THROUGHPUT_POLICY, name='throughput-v24', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                fp32_poisson=True),
    # E107: v23 with the Numba kernels loaded from a content-addressed on-disk cache
    # (jd_curl.three_d.jit_cache: keyed by the whole jd_curl source tree, versions and kernel constants) instead of
    # compiled again in every process, before the task clock. The same machine code: every value as v23.
    'v25': dict(THROUGHPUT_POLICY, name='throughput-v25', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                jit_cache=True),
    # E108 (coarse levels): v25 with the outer CPU gradient recheck of sequential.norms
    # (at the coarse levels every outer whose inner solve met its forcing target) in one fused CPU pass: one sampling,
    # each tile's geometry once for the energy checks and the vectors, local products on the tile threads, the sampling
    # while the first tiles run. The same three vectors, so the same norms and decisions: every value as v25.
    'v26': dict(THROUGHPUT_POLICY, name='throughput-v26', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                jit_cache=True,
                fused_recheck=True),
    # E109 (B14; ROUNDING LEVEL, accepted by the G3b standard): v26 with an outer
    # recheck triggered by the forcing target alone (GF not within 1.01x of the threshold) decided on the device
    # norms outside a 1e-6 band; first uses per level grid checked against the CPU recheck (relative 1e-8).
    'v27': dict(THROUGHPUT_POLICY, name='throughput-v27', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                jit_cache=True,
                fused_recheck=True,
                device_forcing=True),
    # T89 (S30-G4, 2026-10-14): v26 with the active-set skipping (energy blocks, gradient cells/nodes and sampled nodes
    # of the frozen part cached instead of recomputed; every value as the full computation; meaningful only with the
    # active_set option, which v28 requires).
    'v28': dict(THROUGHPUT_POLICY, name='throughput-v28', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                jit_cache=True,
                fused_recheck=True,
                active_set_skip=True),
    # T90 (S30-G5; ROUNDING LEVEL with a band and float64
    # rechecks): v26 with the float32 screening of the trial's soft-geometry and barrier energy terms (v29), and v28 with it (v30).
    'v29': dict(THROUGHPUT_POLICY, name='throughput-v29', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                jit_cache=True,
                fused_recheck=True,
                fp32_screen=True),
    'v30': dict(THROUGHPUT_POLICY, name='throughput-v30', parallel_certificate=True, fallback_workers=8,
                overlap_certificate=True, fused_kernels=True, compiled_fallback=True, fused_diagonal=True,
                cpu_gradient_margin=1e-6, prescreen_certificate=True, fused_hvp=True,
                lean_host=True, fast_host=True, async_commit_writes=True, compiled_extension=True,
                malloc_retention=True, lean_identity=True, fused_steps=True, boundary_reuse=True,
                metric_host=True, lean_outer=True, screen_prefilter=True, fused_node_jd=True, lean_checks=True,
                lean_checks_revision='E96', step_extrema=True, early_warped=True, device_candidate=True,
                root_screen=True, soa_certificate=True, lean_certificate=True, barrier_skip=True,
                laplacian_memo=True, tuned_launch=True, fused_spline=True, skip_reference_first=True,
                estimator_overlap=True, top4_select=True, outer_reuse=True, gpu_edge_hull=True, commit_recheck=False,
                laplacian_cache=True, fused_laplacian=True, deferred_checkpoint_hash=True, device_scalars=True, fused_shift=True,
                direction_prefetch=True,
                jit_cache=True,
                fused_recheck=True,
                active_set_skip=True,
                fp32_screen=True)}
# S30 PRE speed (2026-10-15, implementation only): the pre-optimizer's execution switches (bit for bit
# the same field; preopt.PreoptExecution): v31 = v26 + P-1 lean_sync + P-2 static_grid; v32 = v31 + P-3 cuda_graph.
PREOPT_EXECUTION_SWITCHES = ('preopt_lean_sync', 'preopt_static_grid', 'preopt_cuda_graph',
                             'preopt_parallel_setup', 'preopt_analytic_grad')     # S30 P-12/P-6 (v33, v34)
THROUGHPUT_POLICIES['v31'] = dict(THROUGHPUT_POLICIES['v26'], name='throughput-v31', preopt_lean_sync=True, preopt_static_grid=True)
THROUGHPUT_POLICIES['v32'] = dict(THROUGHPUT_POLICIES['v31'], name='throughput-v32', preopt_cuda_graph=True)
# S30 P-12/P-6: v33 = v26 + P-12 (coarse setup in a thread pool, bit for bit);
# v34 = v33 + P-6 (analytic pre-objective gradient; ROUNDING LEVEL; first use per level checked against autograd).
THROUGHPUT_POLICIES['v33'] = dict(THROUGHPUT_POLICIES['v26'], name='throughput-v33', preopt_parallel_setup=True)
THROUGHPUT_POLICIES['v34'] = dict(THROUGHPUT_POLICIES['v33'], name='throughput-v34', preopt_analytic_grad=True)
# FX1: v35 = v34 + F2' first-use checks of the E93/E94(E100)/E99 fused kernels done
# before the task clock on a synthetic field of the native grid (first_use_prewarm, launcher dry pass), F3 final export to
# tmpfs with the queued checkpoints, full logs and the copy after e2e (fast_export), F4 no native outer that cannot finish
# before the cutoff (outer_gate). The formal commits are those of v34 up to the last one either run makes.
THROUGHPUT_POLICIES['v35'] = dict(THROUGHPUT_POLICIES['v34'], name='throughput-v35', first_use_prewarm=True,
                                  fast_export=True, fast_export_dir='/dev/shm', outer_gate=True, outer_gate_window=3)
# FC1: v36 = v35 + a disk cache of the first-use registry (jd_curl/three_d/backends/first_use_cache.py);
# the dry pass runs once per source, device, native grid, profile and options. Same registration results as v35.
THROUGHPUT_POLICIES['v36'] = dict(THROUGHPUT_POLICIES['v35'], name='throughput-v36', first_use_cache=True)
# HS1: v37 = v36 +
# the T90 float32 screening of the trial's soft-geometry and barrier energy terms (fp32_screen, as in v29/v30; ROUNDING LEVEL
# with the float64 recheck band). Nothing else differs from v36.
THROUGHPUT_POLICIES['v37'] = dict(THROUGHPUT_POLICIES['v36'], name='throughput-v37', fp32_screen=True)
# v38 (implementation only; docs/draft_v38_gate1_20261018.md): v37 +
# dedup_level_files (a transfer attempt whose field is an earlier saved map of this level, and the native level_completed
# equal to the last native commit's u, refer to that file instead of writing the same bytes again) + gpu_hull_first_use_seed
# (T84's synthetic first-use comparison, done in the dry pass, is harvested into the first-use registry and its disk cache
# and seeded into the run's backend; the comparison on the run's own input is unchanged). Decisions bit for bit as v37.
THROUGHPUT_POLICIES['v38'] = dict(THROUGHPUT_POLICIES['v37'], name='throughput-v38', dedup_level_files=True,
                                  gpu_hull_first_use_seed=True)
# v39 (implementation only; the warm-up of item 4 was measured at ~9.5 s
# outside the clock and dropped): v38 + guard_on_writer (the disk guard's per-checkpoint check runs on
# the ordered writer thread right before each checkpoint is written -- still before every write, fsync kept, same rows;
# the main thread no longer waits for it). Decisions bit for bit as v38.
THROUGHPUT_POLICIES['v39'] = dict(THROUGHPUT_POLICIES['v38'], name='throughput-v39', guard_on_writer=True)
# v41 (implementation only): H2 early_trial_hash -- the certificate's SHA-256 of a
# trial's u starts when u is injected (same digest). Decisions bit for bit as v39. (v40 was measured and not built: V40S0.)
THROUGHPUT_POLICIES['v41'] = dict(THROUGHPUT_POLICIES['v39'], name='throughput-v41', early_trial_hash=True)
BACKEND_SWITCHES_E97_E98 = ('step_extrema', 'early_warped', 'device_candidate', 'root_screen',
                            'lean_certificate', 'barrier_skip', 'laplacian_memo',
                            'tuned_launch', 'fused_spline', 'skip_reference_first',      # E99 (v17)
                            'estimator_overlap', 'top4_select', 'outer_reuse',           # E100 (v18)
                            'gpu_edge_hull', 'commit_recheck', 'laplacian_cache', 'fused_laplacian',   # E101 (v19)
                            'metric_prestart',                                                         # E103 (v21)
                            'device_scalars',                                                          # E104 (v22)
                            'fused_shift', 'direction_prefetch',                                       # E105 (v23)
                            'fp32_poisson',                                                            # E106 (v24)
                            'fused_recheck',                                                           # E108 (v26)
                            'device_forcing',                                                          # E109 (v27)
                            'active_set_skip',                                                         # T89 (v28)
                            'fp32_screen',                                                             # T90 (v29/v30)
                            'early_trial_hash')                                                        # v41 (H2)


def host_prewarm_kwargs(throughput):
    """E96: keyword arguments of host_kernels.prewarm for a throughput profile -- the E95 kernels are compiled
    only for the profiles that use them (lean_checks); every other profile gets E94's prewarm()."""
    profile = THROUGHPUT_POLICIES.get(throughput) if isinstance(throughput, str) else None
    if profile is None or not profile.get('lean_checks'):
        return {}
    return dict(lean_checks=True, **(dict(step_extrema=True) if profile.get('step_extrema') else {}),   # E97
                **(dict(fused_shift=True) if profile.get('fused_shift') else {}))                      # E105


STORAGE_RESERVE_BYTES = 10*1024**3
HOST_START_AVAILABLE_BYTES = 4*1024**3
HOST_RUNNING_AVAILABLE_BYTES = 1024**3


def device_batch_config(size, reuse_reference):
    if size is None:
        return None
    if type(size) is not int or size not in (8192,32768,65536) or reuse_reference is not True:
        raise ValueError('device_batch_size requires --reuse-reference and 8192/32768/65536')
    return DeviceBatchConfig(size,8)


def resource_policy(*, memory_limits=True, certificate_tile_size=None):
    if not isinstance(memory_limits, bool):
        raise ValueError('memory_limits must be boolean')
    policy = dict(LOW_LOAD_POLICY)
    if certificate_tile_size is not None:
        if type(certificate_tile_size) is not int or certificate_tile_size not in (2048,8192):
            raise ValueError('certificate_tile_size must be 2048 or 8192')
        policy['certificate_tile_size'] = certificate_tile_size
    if not memory_limits:
        for key in ('cuda_allocator_limit_bytes', 'cgroup_memory_high_bytes',
                    'cgroup_memory_max_bytes', 'cgroup_swap_max_bytes'):
            policy[key] = None  # no task-added limit; hardware/OS limits remain
    return policy


def storage_policy(host_volume, *, start_memory_gate=True, memory_limits=True):
    if not isinstance(start_memory_gate, bool):
        raise ValueError('start_memory_gate must be boolean')
    resource_policy(memory_limits=memory_limits)
    return dict(host_volume=str(Path(host_volume).resolve()),
                reserve_bytes=STORAGE_RESERVE_BYTES, poll_seconds=5.,
                host_start_available_bytes=(HOST_START_AVAILABLE_BYTES if start_memory_gate else HOST_RUNNING_AVAILABLE_BYTES) if memory_limits else None,
                host_running_available_bytes=HOST_RUNNING_AVAILABLE_BYTES if memory_limits else None)


def windows_available_bytes():
    """Actual Windows RAM headroom, not the guest's virtual MemAvailable."""
    command = ['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-NonInteractive', '-Command',
        '(Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreePhysicalMemory']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        value = int(result.stdout.strip())*1024
        if value < 0:
            raise ValueError('negative available memory')
        return value
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise e11.CampaignResourceError('cannot verify Windows memory: '+str(error)) from error


class StorageGuard:
    """Explicit host volume: WSL ext4 free space alone is not sufficient."""

    def __init__(self, out, host_volume, *, cgroup=None, monitor_host_memory=False, start_memory_gate=True, memory_limits=True):
        storage_policy(host_volume, start_memory_gate=start_memory_gate, memory_limits=memory_limits)
        self.out = Path(out)
        self.host = Path(host_volume)
        self.cgroup = None if cgroup is None else Path(cgroup)
        self.monitor_host_memory = monitor_host_memory and memory_limits
        self.start_memory_gate = start_memory_gate
        self._lock = threading.Lock()    # v39: checks may come from the writer thread and the main thread (one at a time)

    def check(self, phase, *, pending_bytes=0, task=None, worker_pid=None, checkpoint=None):
        with self._lock:
            return self._check(phase, pending_bytes=pending_bytes, task=task, worker_pid=worker_pid, checkpoint=checkpoint)

    def _check(self, phase, *, pending_bytes=0, task=None, worker_pid=None, checkpoint=None):
        target = self.out
        while not target.exists():
            target = target.parent
        row = dict(utc=e11.now(), phase=phase, pending_bytes=pending_bytes,
            reserve_bytes=STORAGE_RESERVE_BYTES, output_volume_path=str(target),
            host_volume=str(self.host), task=task, worker_pid=worker_pid,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        if checkpoint is not None:
            row['checkpoint'] = checkpoint     # v39: a writer-side check names the checkpoint it guards
        reason = None
        try:
            if not self.host.is_mount():
                raise OSError('host volume is not mounted: '+str(self.host))
            row['output_free_bytes'] = shutil.disk_usage(target).free
            row['host_free_bytes'] = shutil.disk_usage(self.host).free
            for key in ('output_free_bytes', 'host_free_bytes'):
                if row[key] < STORAGE_RESERVE_BYTES+pending_bytes:
                    reason = f'{key}={row[key]} below reserve plus pending write={STORAGE_RESERVE_BYTES+pending_bytes}'
                    break
        except OSError as error:
            reason = 'cannot verify storage: '+str(error)
        if self.monitor_host_memory and reason is None:
            required = HOST_START_AVAILABLE_BYTES if worker_pid is None and self.start_memory_gate else HOST_RUNNING_AVAILABLE_BYTES
            row['host_required_available_bytes'] = required
            try:
                row['host_available_bytes'] = windows_available_bytes()
                if row['host_available_bytes'] < required:
                    reason = f'host_available_bytes={row["host_available_bytes"]} below headroom={required}'
            except e11.CampaignResourceError as error:
                reason = str(error)
        if self.cgroup is not None:
            row['cgroup'] = str(self.cgroup)
            row['memory'] = {}
            for name in ('memory.current', 'memory.peak', 'memory.events', 'memory.swap.current'):
                try:
                    row['memory'][name] = (self.cgroup/name).read_text().strip()
                except OSError as error:
                    row['memory'][name] = 'unavailable: '+str(error)
        row['reason'] = reason
        if self.out.is_dir():
            # Append throughout the whole campaign, not only the first task.
            try:
                with (self.out/'resource_checks.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            except OSError as error:
                raise e11.CampaignResourceError('cannot persist resource check: '+str(error)) from error
        if reason is not None:
            raise e11.CampaignResourceError(reason)
        return row


def tasks(*, skip_first=False, weak_four_30min=False, local_barrier_hour=False, short_600=False):
    if not isinstance(short_600, bool) or (short_600 and not local_barrier_hour):
        raise ValueError('short_600 requires explicit local_barrier_hour')
    if not isinstance(weak_four_30min, bool):
        raise ValueError('weak_four_30min must be boolean')
    if not isinstance(local_barrier_hour, bool) or (local_barrier_hour and weak_four_30min):
        raise ValueError('choose only one explicit single-task mode')
    if local_barrier_hour:
        if skip_first:
            raise ValueError('new hour experiment cannot reuse prefix')
        seconds = 600 if short_600 else 3600
        return [dict(index=0, pair=PAIR, route='pg_weak', strength='weak',
                     hierarchy='four', wall_seconds=seconds, name=f'pg_weak__four__{seconds}s')]
    if weak_four_30min:
        if skip_first:
            raise ValueError('single 30-minute experiment cannot reuse a completed prefix')
        return [dict(index=0, pair=PAIR, route='pg_weak', strength='weak',
                     hierarchy='four', wall_seconds=1800, name='pg_weak__four__1800s')]
    rows = []
    for seconds in (300, 600):
        for strength in ('strong', 'weak'):
            # Counterbalance execution order between the two time windows.
            for hierarchy in (('single', 'four') if seconds == 300 else ('four', 'single')):
                route = 'pg_'+strength
                rows.append(dict(index=len(rows), pair=PAIR, route=route, strength=strength,
                    hierarchy=hierarchy, wall_seconds=seconds,
                    name=f'{route}__{hierarchy}__{seconds}s'))
    return [dict(t, index=i, original_index=t['index']) for i, t in enumerate(rows[1:])] if skip_first else rows


def completed_prefix(parent):
    """Reuse only the completed first experiment, never a candidate/solver state."""
    parent = Path(parent).resolve()
    schedule = e11.read(parent/'schedule.json')
    assert schedule['tasks'] == tasks(), 'parent must be the original eight-task matrix'
    status = e11.read(parent/'status.json')
    assert status['status'] == 'resource_stopped' and status['worker_pid'] is None
    assert status['completed'] == 1 and status['remaining_not_started'] == 7
    records = e11.read(parent/'runs.json')
    assert len(records) == 1, 'only the completed first task may be reused'
    record = records[0]
    assert all(record[k] == value for k, value in tasks()[0].items())
    assert record['process_status'] == 'completed' and record['returncode'] == 0
    assert record['completion']['complete'] is True
    assert not any((parent/t['name']).exists() for t in tasks()[1:]), 'remaining tasks must be unstarted'
    for name, digest in e11.read(parent/'input_sha256.json').items():
        assert file_sha256(parent/name) == digest, name
    for name in ('pair.npz', 'manifest.json'):
        assert file_sha256(parent/'inputs'/PAIR/name) == file_sha256(SOURCE/name), name
    source = e11.read(parent/'source_sha256.json')
    for path in (ROOT/'jd_curl').rglob('*.py'):
        name = str(path.relative_to(ROOT))
        assert file_sha256(path) == source[name], 'numerical source changed: '+name
    first = parent/record['name']
    for key, name in [('summary','summary.json'), ('labels','labels.json'),
                      ('completion','worker_completion.json'), ('commits','commits.json')]:
        assert record[key] == e11.read(first/name), name
    final = e11.read(first/'final_native.json')
    assert file_sha256(first/final['path']) == final['sha256']
    names = ['schedule.json', 'status.json', 'runs.json', 'input_sha256.json', 'source_sha256.json']
    names += [record['name']+'/'+n for n in ('summary.json', 'labels.json', 'worker_completion.json',
                                           'commits.json', 'final_native.json', final['path'])]
    return dict(parent_run=str(parent), completed_record=record,
        artifact_sha256={name: file_sha256(parent/name) for name in names},
        pending_original_indices=list(range(1, 8)), initialization='each_remaining_task_original_zero')


def progress_costs(attempts, commits):
    """Observed attempts and same-level commit intervals; never imply convergence."""
    levels = []
    for level in range(4):
        rows = [a for a in attempts if a['level']==level]
        accepted = sum(bool(a['accepted']) for a in rows)
        rejects = {}
        for a in rows:
            if not a['accepted']:
                rejects[a['reason']] = rejects.get(a['reason'],0)+1
        times = [c['visible_seconds'] for c in commits if c['level']==level]
        intervals = [b-a for a,b in zip(times,times[1:])]
        levels.append(dict(level=level, attempts=len(rows), inner_accepted=accepted,
            rejected_reasons=rejects, jd_rejected=rejects.get('JACOBIAN_VIOLATION',0),
            jd_uncertain=rejects.get('FEASIBILITY_UNCERTAIN',0), formal_commits=len(times),
            commit_intervals_seconds=intervals,
            mean_commit_interval_seconds=None if not intervals else sum(intervals)/len(intervals)))
    return dict(levels=levels, interval_scope='adjacent same-level saved formal commits; includes IO; excludes first commit and transfers')


def _cgroup_value(name):
    """Actual cgroup v2 limit of this process (recorded, never enforced here)."""
    try:
        relative = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
        return (Path('/sys/fs/cgroup'+relative)/name).read_text().strip()
    except (OSError, IndexError) as error:
        return 'unavailable: '+str(error)


def _jsonl(stream, row):
    stream.write(json.dumps(row, allow_nan=False,
        default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj))+'\n')
    stream.flush()


# T64 (E53, user decision 2026-09-29): weak starts at L1 (4/2/1), strong keeps four levels.
HIERARCHY_POLICIES = {'per_strength_v1': dict(weak=1, strong=0)}


def _plain_array(value):
    """v39: the arrays the deferred writer packs on its own thread (scripts.async_commit_writer._plain)."""
    return type(value) is np.ndarray and not value.dtype.hasobject


def _close_writer_after_failure(writer):
    if writer is not None:
        try:
            writer.close()
        except BaseException:      # never mask the original failure
            pass


def _e53_policy(policy):
    """E53-type PG metric policy names (legacy_q1 and the T62/T69/T73/T70/T74 families)."""
    return policy == 'legacy_q1' or policy.endswith(('_jd_affine_v1', '_jd_affine_v2', '_jd_affine_v3',
                                                     '_jd_affine_triggered_v1', '_jd_affine_always_v1'))


def conservative_pg_route(*, resolution_exit=False, jd_step_global=False, bb2_ggn_guard=False,
                          pg_jd_affine_activation=None, pg_metric_policy=None):
    """True iff the worker builds ConservativePGConfig3D (the E53-type route); the one predicate the worker
    and the launcher share (review E84). With per-level policies pass the first level's policy."""
    return bool(resolution_exit or jd_step_global or bb2_ggn_guard or pg_jd_affine_activation is not None
                or (pg_metric_policy is not None and _e53_policy(pg_metric_policy)))


def validate_inner_steps_levels(levels, first_level=0):
    """E85 T82: four entries, each None (= INNER_STEPS) or an integer in [1, 100] (not bool), at least one set,
    None on every skipped level (a skipped level would silently drop its entry)."""
    import numbers
    if isinstance(levels, (str, bytes)) or not hasattr(levels, '__len__') or len(levels) != 4 or any(
            k is not None and (not isinstance(k, numbers.Integral) or isinstance(k, (bool, np.bool_))) for k in levels):
        raise ValueError('inner_steps_levels must be four entries, each None or a positive integer')
    levels = tuple(None if k is None else int(k) for k in levels)
    if all(k is None for k in levels) or any(k is not None and not 1 <= k <= 100 for k in levels):
        raise ValueError('inner_steps_levels: at least one integer in [1, 100], the others None')
    if any(k is not None for k in levels[:first_level]):
        raise ValueError('inner_steps_levels: a skipped level cannot carry an inner step limit')
    return levels


def validate_first_trial_caps(caps):
    """E84 T81: four entries, each None or a real cap in [1e-6, 1e3] level voxels, at least one set."""
    from jd_curl.three_d.solvers.poisson_gradient import FIRST_TRIAL_CAP_RANGE
    import numbers
    if isinstance(caps, (str, bytes)) or not hasattr(caps, '__len__') or len(caps) != 4 or any(
            c is not None and (not isinstance(c, numbers.Real) or isinstance(c, (bool, np.bool_))) for c in caps):
        raise ValueError('first_trial_cap_levels must be four entries, each None or a real number (not bool)')
    caps = tuple(None if c is None else float(c) for c in caps)
    if all(c is None for c in caps) or not all(
            c is None or (np.isfinite(c) and FIRST_TRIAL_CAP_RANGE[0] <= c <= FIRST_TRIAL_CAP_RANGE[1]) for c in caps):
        raise ValueError('first_trial_cap_levels: at least one cap in [1e-6, 1e3] level voxels, the others None')
    return caps


def validate_momentum_levels(levels):
    """E111 T85: four booleans (one per level, True = the first inner step of each outer follows the last outer
    increment on that level), at least one True."""
    if isinstance(levels, (str, bytes)) or not hasattr(levels, '__len__') or len(levels) != 4 or any(
            not isinstance(x, (bool, np.bool_)) for x in levels):
        raise ValueError('momentum_first_levels must be four booleans')
    levels = tuple(bool(x) for x in levels)
    if not any(levels):
        raise ValueError('momentum_first_levels: at least one level must be on')
    return levels


NATIVE_ABLATION_SWITCHES = ('native_geometry_weights', 'native_reference', 'native_metric_ablation',
                            'native_inner')   # T92 (D2; native_inner = D2-e)
NATIVE_LM_PRECONDITIONER = 'ggn_diagonal_poisson'   # T92 D2-e: the library LM with the F29 GGN-diagonal-equilibrated Poisson PCG
PENALTY_SCALE_RANGE = (1e-6, 1.)        # T92 (D2-c' down-weighted arm): s in [1e-6, 1); x1 = without the option


def parse_native_geometry_weights(value):
    """T92 (D2-a/b): 'mu=M,nu=N' or a dict/pair -> dict(mu=M, nu=N) of finite floats (range checked by the caller)."""
    if isinstance(value, str):
        try:
            parts = dict(item.split('=', 1) for item in value.split(','))
        except ValueError:
            raise ValueError('native_geometry_weights: mu=M,nu=N') from None
        value = parts
    if isinstance(value, dict):
        if set(value) != {'mu', 'nu'}:
            raise ValueError('native_geometry_weights: exactly the keys mu and nu')
        value = (value['mu'], value['nu'])
    if isinstance(value, (str, bytes)) or not hasattr(value, '__len__') or len(value) != 2 \
            or any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError('native_geometry_weights: two real numbers mu, nu')
    try:
        mu, nu = (float(x) for x in value)
    except (TypeError, ValueError):
        raise ValueError('native_geometry_weights: two real numbers mu, nu') from None
    return dict(mu=0. if mu == 0. else mu, nu=0. if nu == 0. else nu)


def validate_native_ablation(*, native_geometry_weights=None, native_reference=None, native_metric_ablation=None,
                             native_penalty_scale=None, pre_optimizer=None, geometry_weight_levels=None, active_set=None,
                             practical_stop=None, momentum_first_levels=None, native_policy=None, first_trial_cap_levels=None,
                             native_inner=None):
    """T92 (D2 ablation switches; user-confirmed 2026-10-19, docs/04 T92): None, or dict(contract='T92', switch, value)
    for the one switch given (plus penalty_scale for the down-weighted D2-c' arm). The native level of the pre-optimizer
    route only; one switch per run."""
    given = [(name, value) for name, value in zip(NATIVE_ABLATION_SWITCHES,
                                                  (native_geometry_weights, native_reference, native_metric_ablation,
                                                   native_inner))
             if value is not None]
    scale = None
    if native_penalty_scale is not None:
        # T92 (D2-c' down-weighted arm; the metric follows)
        if native_reference != 'identity':
            raise ValueError('native_penalty_scale applies to native_reference identity only')
        if geometry_weight_levels is None:
            raise ValueError('native_penalty_scale requires geometry_weight_levels (the native weights it scales)')
        if isinstance(native_penalty_scale, (bool, np.bool_, str, bytes)):
            raise ValueError('native_penalty_scale must be a real number')
        try:
            scale = float(native_penalty_scale)
        except (TypeError, ValueError):
            raise ValueError('native_penalty_scale must be a real number') from None
        if not (np.isfinite(scale) and PENALTY_SCALE_RANGE[0] <= scale < PENALTY_SCALE_RANGE[1]):
            raise ValueError('native_penalty_scale must be in [1e-6, 1) (x1 is the identity arm without the option)')
    if not given:
        return None
    if len(given) > 1:
        raise ValueError('T92: one native ablation switch per run')
    if pre_optimizer is None:
        raise ValueError('T92 native ablation switches require the pre-optimizer route (the native level only)')
    if active_set is not None or practical_stop is not None or (momentum_first_levels is not None and any(momentum_first_levels)):
        raise ValueError('T92 native ablation switches exclude active_set (T89), practical_stop (H17) and momentum (T85)')
    name, value = given[0]
    if name == 'native_geometry_weights':
        value = parse_native_geometry_weights(value)
        if geometry_weight_levels is None:
            raise ValueError('native_geometry_weights requires geometry_weight_levels (the other weight is its native value)')
        mu, nu = value['mu'], value['nu']
        if not (np.isfinite(mu) and np.isfinite(nu) and min(mu, nu) == 0. and max(mu, nu) > 0.):
            raise ValueError('native_geometry_weights: finite, exactly one of mu, nu is 0')
        if max(mu, nu) != float(geometry_weight_levels[3]):
            raise ValueError('native_geometry_weights: the nonzero weight must equal the native geometry weight '
                             '(only one penalty is removed)')
    elif name == 'native_reference':
        if value not in ('fixed', 'identity'):
            raise ValueError("native_reference must be 'fixed' or 'identity'")
    elif name == 'native_inner':
        # T92 D2-e (PATH CHANGE, ablation switch)
        if value != 'lm':
            raise ValueError("native_inner must be 'lm'")
        if first_trial_cap_levels is not None:
            raise ValueError('native_inner lm excludes the T81 first-trial cap option (a PG option; the coarse levels are not solved)')
    else:
        from jd_curl.three_d.solvers.poisson_gradient import METRIC_ABLATION_MODES, T62_POLICIES
        if value not in METRIC_ABLATION_MODES:
            raise ValueError("native_metric_ablation must be 'diagonal' or 'identity'")
        if value == 'diagonal' and native_policy not in T62_POLICIES:
            raise ValueError('native_metric_ablation diagonal keeps the T62-family scale (native policy T62/T69/T73)')
        if value == 'identity' and (first_trial_cap_levels is None or first_trial_cap_levels[3] is None):
            raise ValueError('native_metric_ablation identity needs a T81 first-trial cap on the native level')
    return dict(contract='T92', switch=name, value=value, **({} if scale is None else dict(penalty_scale=scale)))


def _row_energy(row):
    """E-HS1b: the float64 energy terms of an outer candidate from its history row (exact_energy under T90)."""
    from jd_curl.three_d.residuals import EnergyTerms3D
    return EnergyTerms3D(image=row['D_candidate'], proximity=row['penalty_proximity'], jd=row['penalty_jd'],
                         curl=row['penalty_curl'], barrier=row['B_candidate'], curvature=row.get('R_candidate') or 0.)


def _final_energy(state, row):
    """E-HS1b (review): the exported native state is either the last native commit's candidate itself (the run ended before the
    self-reference refresh: GF reached, outer limits, stagnation, practical stop, interruption) -> all six float64 terms of its
    row; or the refreshed state of that candidate (same u) whose proximity/jd/curl are exact zeros -> the outer terms of the row."""
    from dataclasses import asdict
    if getattr(state, 'eval_id', None) == row.get('candidate_eval_id'):
        return asdict(_row_energy(row))
    energy = asdict(state.energy)
    # never expected (refresh_reference builds (image, 0, 0, 0, barrier, curvature)); not raised after e2e: the summary would be
    # lost. The outer terms are still the float64 ones; summary['energy_soft_terms_unverified'] records the case.
    return dict(energy, **_outer_terms(row))


def _outer_terms(row):
    """E-HS1b: the reference-independent outer terms (image, barrier, curvature) of a formal commit in float64; the final
    state after the self-reference refresh has the same u, so these are its outer terms too."""
    return dict(image=row['D_candidate'], barrier=row['B_candidate'], curvature=row.get('R_candidate') or 0.)


def worker(task, out, data, *, device='cuda', task_started=None, low_load=False, host_storage=None, start_memory_gate=True, memory_limits=True, advance_stalled_levels=False, transfer_backtracking=False, local_barrier=False, efficient_cpu_budget=False, gpu_near_boundary=False, reuse_reference=False, device_batch_size=None, selective_kernels=False, compiled_certificate=False, hoisted_certificate=False, certificate_tile_size=None, inner_variant=None, throughput=False, first_level=0, lean_commits=False, pg_metric_policy=None, pg_shift_factor=1., hierarchy_policy=None, resolution_exit=False, pg_jd_affine_activation=None, jd_step_global=False, level_shares=None, barrier_basis='samples', pg_native_metric_policy=None, bb2_ggn_guard=False, curvature_weight=0., geometry_weight=None, geometry_weight_levels=None, pg_metric_policy_levels=None, data_weight=None, first_trial_cap_levels=None, first_trial_scope=None, inner_steps_levels=None, native_start_scale=None,
           transfer_smoothing=None, curvature_weight_levels=None, practical_stop=None, momentum_first_levels=None,
           export_reserve=None, resident=None, pre_optimizer=None, active_set=None, preverified=None, first_use_dry=None,
           native_transfer_alphas=None, ggn_diagonal_reuse=None, native_geometry_weights=None, native_reference=None,
           native_metric_ablation=None, native_penalty_scale=None, native_inner=None):
    """Parent clock includes interpreter/import/SHA checks; no hidden CPU fallback.

    Every level uses the original solver. No observer can accept a candidate.
    IO and labels are inside the parent watchdog, with a 30s export reserve.
    throughput=True (E48) changes execution sizes/threads and diagnostic IO only:
    promoted derivative-ready states keep metadata but no arrays, per-attempt and
    per-outer logs are appended as JSONL and written as full JSON at the end.
    Formal commits, the final export and all numerical decisions are unchanged.
    """
    if geometry_weight is not None:
        # T77 (E79): validated before any output exists.
        if isinstance(geometry_weight, (bool, str)):
            raise ValueError('geometry_weight must be a real number')
        geometry_weight = float(geometry_weight)
        if not np.isfinite(geometry_weight) or geometry_weight <= 0:
            raise ValueError('geometry_weight must be a finite positive float')
    if geometry_weight_levels is not None:
        # T78 (E80): one explicit mu = nu per pyramid level; validated before any output exists.
        if geometry_weight is not None:
            raise ValueError('geometry_weight and geometry_weight_levels are exclusive')
        if isinstance(geometry_weight_levels, (str, bytes)) or len(geometry_weight_levels) != 4 \
                or any(isinstance(w, (bool, str)) for w in geometry_weight_levels):
            raise ValueError('geometry_weight_levels must be four real numbers')
        geometry_weight_levels = tuple(float(w) for w in geometry_weight_levels)
        if not all(np.isfinite(w) and w > 0 for w in geometry_weight_levels):
            raise ValueError('geometry_weight_levels must be finite positive floats')
        if task['hierarchy'] != 'four':
            raise ValueError('geometry_weight_levels requires the four-level route')
    if curvature_weight_levels is not None:
        # G2 (model parameter per level, user-approved 2026-10-12): one curvature weight alpha per pyramid level, as
        # T78 does for mu = nu; validated before any output exists.
        try:
            single = float(curvature_weight)
        except (TypeError, ValueError):
            raise ValueError('curvature_weight must be a real number') from None
        if single:
            raise ValueError('curvature_weight and curvature_weight_levels are exclusive')
        if isinstance(curvature_weight_levels, (str, bytes)) or len(curvature_weight_levels) != 4 \
                or any(isinstance(a, (bool, str, bytes)) for a in curvature_weight_levels):
            raise ValueError('curvature_weight_levels must be four real numbers')
        curvature_weight_levels = tuple(float(a) for a in curvature_weight_levels)
        if not all(np.isfinite(a) and a >= 0 for a in curvature_weight_levels):
            raise ValueError('curvature_weight_levels must be finite nonnegative floats')
        if task['hierarchy'] != 'four':
            raise ValueError('curvature_weight_levels requires the four-level route')
        if first_level != 0:
            raise ValueError('curvature_weight_levels requires first_level 0 (every level solved)')
    if pre_optimizer is not None:
        # T88 (S30-G3, explicit option): the float32 pre-optimizer replaces
        # the coarse levels as an UNCERTIFIED initializer; the native transfer certifies it. Validated before any output.
        from jd_curl.three_d.preopt import PreoptConfig, PREOPT_POLICY
        from jd_curl.validation import InputValidationError as _IVE
        if not isinstance(pre_optimizer, dict) or pre_optimizer.get('policy') != PREOPT_POLICY:
            raise ValueError(f'pre_optimizer: a dict with policy {PREOPT_POLICY}')
        taper = pre_optimizer.get('taper', 4)
        if isinstance(taper, bool) or not isinstance(taper, int) or taper < 1:
            raise ValueError('pre_optimizer: taper must be a positive integer')
        try:
            preopt_config = PreoptConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pre_optimizer.items() if k not in ('policy', 'taper')})
        except (_IVE, TypeError, ValueError) as exc:
            raise ValueError(f'pre_optimizer: {exc}') from None
        if task['hierarchy'] != 'four' or not transfer_backtracking or first_level != 0 or transfer_smoothing is not None:
            raise ValueError('pre_optimizer requires the four-level route with transfer_backtracking, first_level 0 and no transfer_smoothing')
        # C2: T83's native start scale may now be combined
        # with the pre-optimizer route -- the native transfer tries beta * .9^k times the tapered seed (validated below).
    if native_transfer_alphas is not None:
        # AP2 (explicit option, path only): the native
        # transfer of the pre-optimizer seed tries AP1's finer alpha list (1, .99, .98, .96, .94, .92, then .9^k) instead
        # of .9^k; still the first certified scale, whole-cell JD > 0. Validated before any output.
        from jd_curl.three_d.multiresolution import TRANSFER_ALPHA_LISTS
        if not isinstance(native_transfer_alphas, str) or native_transfer_alphas not in TRANSFER_ALPHA_LISTS:
            raise ValueError('native_transfer_alphas must be one of ' + ', '.join(sorted(TRANSFER_ALPHA_LISTS)))
        if pre_optimizer is None:
            raise ValueError('native_transfer_alphas requires the pre-optimizer route')
        if native_start_scale not in (None, 1.):                         # C2: AP2's finer list and beta stay exclusive
            raise ValueError('native_transfer_alphas and native_start_scale are exclusive')
    active_spec = None
    if active_set is not None:
        # T89 (S30-G4, explicit option, path only): the native level's PG direction is
        # p = -(wS) M^-1 (wS) g with the tapered support-distance weights w; F, barrier, certificate, E/F, GF unchanged.
        from jd_curl.three_d.active_set import parse_active_set
        from jd_curl.validation import InputValidationError as _IVE
        try:
            active_spec = parse_active_set(active_set)
        except _IVE as exc:
            raise ValueError(f'active_set: {exc}') from None
        if task['hierarchy'] != 'four':
            raise ValueError('active_set requires the four-level route (the native level only)')
    if practical_stop is not None:
        # H17 (CHANGE OF THE STOPPING RULE as an explicit option, user-approved 2026-10-12): the native level may end
        # early when the projected ratio gain over the remaining budget is <= epsilon; never convergence.
        from jd_curl.three_d.practical_stop import practical_stop_config
        from jd_curl.validation import InputValidationError as _IVE
        try:
            practical_stop = practical_stop_config(dict(practical_stop))
        except (_IVE, TypeError, ValueError) as exc:
            raise ValueError(f'practical_stop: {exc}') from None
        if task['hierarchy'] != 'four':
            raise ValueError('practical_stop requires the four-level route')
    if pg_metric_policy_levels is not None:
        # T78 (E80): one PG metric policy per level (metric only); exclusive with the coarse/native pair.
        if pg_metric_policy is not None or pg_native_metric_policy is not None:
            raise ValueError('pg_metric_policy_levels is exclusive with pg_metric_policy/pg_native_metric_policy')
        if isinstance(pg_metric_policy_levels, (str, bytes)) or len(pg_metric_policy_levels) != 4 \
                or not all(isinstance(p, str) and p for p in pg_metric_policy_levels):
            raise ValueError('pg_metric_policy_levels must be four policy names')
        pg_metric_policy_levels = tuple(pg_metric_policy_levels)
        if task['hierarchy'] != 'four' or not all(_e53_policy(p) for p in pg_metric_policy_levels):
            raise ValueError('pg_metric_policy_levels requires the four-level E53-type PG route (every level an E53 policy)')
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsCheck
        for policy in pg_metric_policy_levels:            # dataclass validation of every name, before any output
            _ConsCheck(metric_policy=policy, shift_factor=pg_shift_factor, resolution_exit=bool(resolution_exit),
                       **(dict(jd_affine_activation=pg_jd_affine_activation) if pg_jd_affine_activation is not None else {}))
        pg_metric_policy = pg_metric_policy_levels[0]      # base inner config; replaced per level below
    if data_weight is not None:
        # T80 (E80, MODEL CHANGE of the data term, user-approved 2026-10-03): validated before any output exists.
        from jd_curl.three_d.data_weight import DATA_WEIGHT_POLICY
        if not isinstance(data_weight, dict) or data_weight.get('policy') != DATA_WEIGHT_POLICY \
                or set(data_weight) != {'policy', 'beta', 'sigma_mm'} \
                or any(isinstance(data_weight[k], (bool, str)) for k in ('beta', 'sigma_mm')):
            raise ValueError('data_weight: dict(policy=brain-shell-v1, beta=..., sigma_mm=...) required')
        data_weight = dict(policy=DATA_WEIGHT_POLICY, beta=float(data_weight['beta']), sigma_mm=float(data_weight['sigma_mm']))
        if not (np.isfinite(data_weight['beta']) and 0. <= data_weight['beta'] <= 1.
                and np.isfinite(data_weight['sigma_mm']) and data_weight['sigma_mm'] > 0.):
            raise ValueError('data_weight: 0 <= beta <= 1 and sigma_mm > 0')
        if task['hierarchy'] != 'four':
            raise ValueError('data_weight requires the four-level route')
    if (geometry_weight_levels is not None or pg_metric_policy_levels is not None) and first_level != 0:
        # review E80: a skipped level would silently drop its entry.
        raise ValueError('per-level geometry weights/metric policies require first_level 0 (every level solved)')
    if first_trial_cap_levels is not None:
        # T81 (E84): explicit first-trial maximum node displacement cap per level (level voxels; None = off on that
        # level). Path only: trial tau <= c/max_i |p_i|_h; F/E, certificates, acceptance, forcing and GF unchanged.
        from jd_curl.three_d.solvers.poisson_gradient import FIRST_TRIAL_SCOPES
        first_trial_cap_levels = validate_first_trial_caps(first_trial_cap_levels)
        if task['hierarchy'] != 'four' or any(c is not None for c in first_trial_cap_levels[:first_level]):
            raise ValueError('first_trial_cap_levels requires the four-level route and no cap on a skipped level')
        if not (local_barrier and inner_variant is None and conservative_pg_route(
                resolution_exit=resolution_exit, jd_step_global=jd_step_global, bb2_ggn_guard=bb2_ggn_guard,
                pg_jd_affine_activation=pg_jd_affine_activation, pg_metric_policy=pg_metric_policy)):
            raise ValueError('first_trial_cap_levels requires the E53-type local-barrier PG route')
        first_trial_scope = 'initial' if first_trial_scope is None else first_trial_scope
        if first_trial_scope not in FIRST_TRIAL_SCOPES:
            raise ValueError('first_trial_scope: initial or all')
    elif first_trial_scope is not None:
        raise ValueError('first_trial_scope requires first_trial_cap_levels')
    if momentum_first_levels is not None:
        # T85 (E111): explicit per-level momentum first direction (path only; F/E, certificates, acceptance, forcing
        # and GF unchanged). Validated before any output, with the T81 route rule.
        momentum_first_levels = validate_momentum_levels(momentum_first_levels)
        if task['hierarchy'] != 'four' or any(momentum_first_levels[:first_level]):
            raise ValueError('momentum_first_levels requires the four-level route and no skipped level switched on')
        if not (local_barrier and inner_variant is None and conservative_pg_route(
                resolution_exit=resolution_exit, jd_step_global=jd_step_global, bb2_ggn_guard=bb2_ggn_guard,
                pg_jd_affine_activation=pg_jd_affine_activation, pg_metric_policy=pg_metric_policy)):
            raise ValueError('momentum_first_levels requires the E53-type local-barrier PG route')
    if native_start_scale is not None:
        # T83 (E90, changes the T38g "try alpha = 1 first" rule on the native transfer; explicit option, user-confirmed
        # 2026-10-05): the native level starts from beta * .9^k * P u_L2. Validated before any output.
        from jd_curl.three_d.multiresolution import validate_start_scale
        from jd_curl.validation import InputValidationError as _IVE
        try:
            native_start_scale = validate_start_scale(native_start_scale)
        except _IVE as exc:
            raise ValueError(f'native_start_scale: {exc}') from None
        if task['hierarchy'] != 'four' or not transfer_backtracking:
            raise ValueError('native_start_scale requires the four-level route with transfer_backtracking')
    if transfer_smoothing is not None:
        # A19 (path only, user-approved 2026-10-12): low-pass S_sigma of the prolonged field before the transfer
        # certification (native transfer only, or every transfer). Validated before any output.
        from jd_curl.three_d.transfer_smoothing import validate_transfer_smoothing
        from jd_curl.validation import InputValidationError as _IVE
        try:
            transfer_smoothing = validate_transfer_smoothing(dict(transfer_smoothing))
        except (_IVE, TypeError, ValueError) as exc:
            raise ValueError(f'transfer_smoothing: {exc}') from None
        if task['hierarchy'] != 'four' or not transfer_backtracking:
            raise ValueError('transfer_smoothing requires the four-level route with transfer_backtracking')
    if inner_steps_levels is not None:
        # T82 (E85, MAINLINE PARAMETER as an explicit option, user-confirmed 2026-10-04): per-level inner step limit;
        # None entries keep INNER_STEPS. Path only: forcing, E/F, certificates and GF are unchanged.
        inner_steps_levels = validate_inner_steps_levels(inner_steps_levels, first_level)
        if task['hierarchy'] != 'four' or inner_variant is not None:
            raise ValueError('inner_steps_levels requires the four-level PG route')
    import torch
    if not isinstance(lean_commits, bool):
        raise ValueError('lean_commits must be boolean')
    if throughput is not False and throughput is not True and throughput not in THROUGHPUT_POLICIES:
        raise ValueError('throughput must be False, True (v1) or an explicit profile version')
    profile = None if throughput is False else THROUGHPUT_POLICIES['v1' if throughput is True else throughput]
    preopt_execution = None
    if pre_optimizer is not None and profile is not None and any(profile.get(k) for k in PREOPT_EXECUTION_SWITCHES):
        # S30 PRE speed: the profile's pre-optimizer execution switches, validated before any output
        from jd_curl.three_d.preopt import PreoptExecution
        from jd_curl.validation import InputValidationError as _IVE
        try:
            preopt_execution = PreoptExecution(lean_sync=bool(profile.get('preopt_lean_sync')), static_grid=bool(profile.get('preopt_static_grid')),
                                               cuda_graph=bool(profile.get('preopt_cuda_graph')),
                                               parallel_setup=bool(profile.get('preopt_parallel_setup')),      # S30 P-12/P-6
                                               analytic_grad=bool(profile.get('preopt_analytic_grad')))
        except _IVE as exc:
            raise ValueError(f'preopt execution: {exc}') from None
        if preopt_execution.cuda_graph and device != 'cuda':
            raise ValueError('preopt execution cuda_graph (throughput-v32) requires the cuda device')
    if profile is not None and profile.get('active_set_skip') and active_spec is None:
        raise ValueError('throughput-v28 (active_set_skip) requires the active_set option')     # T89 (S30-G4)
    ggn_reuse_spec = None
    if ggn_diagonal_reuse is not None:
        # GG2 (explicit option, metric only):
        # the PG metric's GGN diagonal keeps the per-cell locals of cells far from the barrier across outers.
        from jd_curl.three_d.backends.accelerated import parse_ggn_reuse
        from jd_curl.validation import InputValidationError as _IVE
        try:
            ggn_reuse_spec = parse_ggn_reuse(ggn_diagonal_reuse)
        except _IVE as exc:
            raise ValueError(f'ggn_diagonal_reuse: {exc}') from None
        if pre_optimizer is None:
            raise ValueError('ggn_diagonal_reuse requires the pre-optimizer route (the native level is then the only PG level)')
        from jd_curl.three_d.solvers.poisson_gradient import T69_POLICY, T73_POLICY
        native_policy = (pg_native_metric_policy if pg_native_metric_policy is not None else
                         pg_metric_policy_levels[3] if pg_metric_policy_levels is not None else pg_metric_policy)
        if native_policy not in (T69_POLICY, T73_POLICY):
            # review GG2 S1: only T69/T73 hand the build its E94 cell minima; elsewhere the option would be inert
            raise ValueError('ggn_diagonal_reuse requires the T69 or T73 metric on the native level')
        if device != 'cuda' or profile is None or not all(profile.get(k) for k in ('fused_diagonal', 'fused_node_jd', 'estimator_overlap')):
            raise ValueError('ggn_diagonal_reuse requires cuda and a throughput profile with the fused diagonal, the E94 fused '
                             'node JD and the E100 estimator overlap (v18 or later)')
    native_ablation = validate_native_ablation(native_geometry_weights=native_geometry_weights, native_reference=native_reference,
        native_metric_ablation=native_metric_ablation, native_penalty_scale=native_penalty_scale,
        pre_optimizer=pre_optimizer, geometry_weight_levels=geometry_weight_levels,
        active_set=active_spec, practical_stop=practical_stop, momentum_first_levels=momentum_first_levels,
        native_policy=(pg_native_metric_policy if pg_native_metric_policy is not None else
                       pg_metric_policy_levels[3] if pg_metric_policy_levels is not None else pg_metric_policy),
        first_trial_cap_levels=first_trial_cap_levels, native_inner=native_inner)   # T92 (D2), before any output
    if (preverified is not None or first_use_dry is not None) and not (profile is not None and profile.get('first_use_prewarm')):
        raise ValueError('preverified and first_use_dry require a profile with first_use_prewarm (FX1 F2\', v35)')
    if preverified is not None and first_use_dry is not None:
        raise ValueError('preverified and first_use_dry are exclusive')
    if first_use_dry is not None and (pre_optimizer is None or not isinstance(first_use_dry, dict)
                                      or 'seed' not in first_use_dry or not isinstance(first_use_dry.get('registry'), dict)):
        raise ValueError('first_use_dry: a dict with seed and registry, on the pre-optimizer route')
    throughput = profile is not None
    if profile is not None and profile.get('parallel_certificate') and not (compiled_certificate and hoisted_certificate):
        raise ValueError('parallel certificate profile requires the compiled hoisted certificate')
    if inner_variant not in (None,'efficient','coverage','hybrid','safeguarded') or (inner_variant is not None and not local_barrier):
        raise ValueError('inner_variant requires an explicit validated local-barrier branch')
    if not isinstance(compiled_certificate, bool):
        raise ValueError("compiled_certificate must be boolean")
    if not isinstance(hoisted_certificate, bool) or (hoisted_certificate and not compiled_certificate):
        raise ValueError("hoisted_certificate requires boolean and compiled_certificate")
    if compiled_certificate:
        from jd_curl.three_d.compiled_enclosure import runtime_info, HOISTED_KERNEL, LEGACY_KERNEL
        certificate_runtime = runtime_info(HOISTED_KERNEL if hoisted_certificate else LEGACY_KERNEL)
        if throughput and profile.get('parallel_certificate'):
            certificate_runtime = runtime_info('numba-parallel-hoisted-edge-hull-v1')
            if profile.get('compiled_fallback'):
                from jd_curl.three_d.compiled_fallback import FALLBACK_KERNEL
                certificate_runtime = dict(certificate_runtime, fallback_kernel=FALLBACK_KERNEL)
    batch = device_batch_config(device_batch_size, reuse_reference)
    if throughput and batch is not None:
        batch = DeviceBatchConfig(profile['device_tile_size'], batch.screen_batch_tiles)
    if not isinstance(selective_kernels, bool) or (selective_kernels and batch is None):
        raise ValueError("selective_kernels requires boolean and explicit device_batch_size")
    if not isinstance(reuse_reference, bool) or (reuse_reference and not gpu_near_boundary):
        raise ValueError('reuse_reference requires explicit gpu_near_boundary')
    if not isinstance(gpu_near_boundary, bool) or (gpu_near_boundary and not efficient_cpu_budget):
        raise ValueError('gpu_near_boundary requires explicit efficient_cpu_budget')
    if not isinstance(efficient_cpu_budget, bool) or (efficient_cpu_budget and not local_barrier):
        raise ValueError('efficient_cpu_budget requires explicit local_barrier')
    if not isinstance(local_barrier, bool) or (local_barrier and task['hierarchy'] != 'four'):
        raise ValueError('local barrier MRI experiment requires four levels')
    if not isinstance(low_load, bool):
        raise ValueError('low_load must be an explicit boolean')
    if host_storage is not None and not low_load:
        raise ValueError('host storage guard requires low_load')
    if not isinstance(advance_stalled_levels, bool) or (advance_stalled_levels and task['hierarchy'] != 'four'):
        raise ValueError('advance_stalled_levels requires a four-level task and a boolean')
    if not isinstance(transfer_backtracking, bool) or (transfer_backtracking and task['hierarchy'] != 'four'):
        raise ValueError('transfer_backtracking requires a four-level task and a boolean')
    policy = resource_policy(memory_limits=memory_limits, certificate_tile_size=certificate_tile_size)
    if certificate_tile_size is not None and not compiled_certificate:
        raise ValueError('certificate_tile_size requires compiled_certificate')
    guard = None if host_storage is None else StorageGuard(out, host_storage, start_memory_gate=start_memory_gate, memory_limits=memory_limits)
    guard_seconds = [0.]           # S30-G1 (record only): main-thread time inside the storage guard
    if guard is not None:
        tick = perf_counter(); guard.check('worker_preflight', task=task['name']); guard_seconds[0] += perf_counter()-tick
    if low_load:
        torch.set_num_threads(LOW_LOAD_POLICY['torch_threads'])
        if device == 'cuda':
            total = torch.cuda.get_device_properties(0).total_memory
            limit = policy['cuda_allocator_limit_bytes']
            torch.cuda.set_per_process_memory_fraction(1. if limit is None else min(1., limit/total), 0)
    if throughput:
        torch.set_num_threads(profile['torch_threads'])
    started = perf_counter() if task_started is None else task_started
    reserve = EXPORT_RESERVE if export_reserve is None else export_reserve      # S30-G1: per-budget export reserve
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not np.isfinite(reserve) or not float(reserve) > 0.:
        raise ValueError('export_reserve must be a positive finite number')   # >= wall_seconds: an already expired deadline, as before
    reserve = float(reserve)
    if resident is not None:
        # S30-G1: the resident service runs several pairs in one process; the per-process counters restart per pair
        # and every first-use check on real inputs is repeated per pair (the backend is new for every worker call).
        if not isinstance(resident, dict) or resident.get('first_use_checks') != 'per_pair':
            raise ValueError("resident requires a dict with first_use_checks 'per_pair'")
        resident = dict(resident, counters_reset=reset_process_counters())
    deadline = started+task['wall_seconds']-reserve
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    host_memory = None
    if throughput and profile.get('malloc_retention'):
        # E89 (P02): allocator policy only (process-wide, once); recorded in config.execution_profile.
        from jd_curl.three_d.host_memory import enable_malloc_retention
        host_memory = enable_malloc_retention()
    promoted, attempts, commits, history, events = [], [], [], [], []
    active_level = 3 if task['hierarchy'] == 'single' else 0
    io_seconds = 0.
    last_inner, last_outer, last_transfer = {}, {}, {}
    native_anchor = None
    native_baseline = None
    interruption = None
    # E-HS1b: with the T90 float32 screening a state's
    # energy object holds screening values; the recorded final energy uses the float64 terms of the last native formal
    # commit (its history row: D/B/R_candidate and the penalties, computed by exact_energy), and screened records say so.
    screen_records = bool(profile is not None and profile.get('fp32_screen'))
    native_exact_row = None
    work = {}
    backend = None
    costs = progress_costs([],[])
    cpu_cost = dict(finished=0, aborted=0, seconds=0.,
                    scope='fused_cpu_comparison operation wall, including aborted work; nested, do NOT sum with backend timers')
    streams = {}
    saved_maps = {}
    # v38 (dedup_level_files): the maps saved by this level's events (state hash -> file) and the last native commit
    dedup_files = bool(profile is not None and profile.get('dedup_level_files'))
    level_maps, last_native = {}, [None]
    writer = None   # E76: throughput v9 ordered background checkpoint writes
    # v39 (guard_on_writer): the per-checkpoint disk check runs on the writer thread, right before each checkpoint write
    guard_on_writer = bool(throughput and profile.get('guard_on_writer'))
    if guard_on_writer and not (profile.get('async_commit_writes') and profile.get('deferred_checkpoint_hash')):
        raise ValueError('guard_on_writer needs async_commit_writes and deferred_checkpoint_hash (v20 and later)')
    writer_guard_seconds = [0.]

    def writer_guard(path, size, queued_bytes):
        # pending = this checkpoint and every job still queued behind it (each with the save() size estimate)
        tick = perf_counter()
        try:
            guard.check('before_checkpoint', pending_bytes=queued_bytes, task=task['name'], checkpoint=path.name)
        finally:
            writer_guard_seconds[0] += perf_counter()-tick
    if throughput and profile.get('async_commit_writes'):
        from scripts.async_commit_writer import OrderedWriter
        # E102 (v20): np.savez and the SHA-256 move to the writer thread (same bytes, same order)
        writer = OrderedWriter(deferred_hash=bool(profile.get('deferred_checkpoint_hash')),
                               **(dict(before_npz=writer_guard, passthrough=(e11.CampaignResourceError,))
                                  if guard_on_writer and guard is not None else {}))
    last_progress_write = [float('-inf')]
    major_stages = ('load_started', 'solve_started', 'outer_checked', 'transfer_attempt',
                    'level_prepared', 'level_completed', 'transfer_rejected', 'budget_cleanup_started',
                    'native_label_evaluation_started', 'worker_completed', 'sequential_baseline_ready',
                    'work_budget_interrupted')

    def stream(name):
        if name not in streams:
            streams[name] = (out/name).open('a')
        return streams[name]

    def progress(stage, **values):
        if first_use_dry is not None:
            return                       # FX1 F2': the dry pass shows no progress
        if throughput and stage not in major_stages:
            # Throttled live view only; every event stays in work_events.jsonl.
            if perf_counter()-last_progress_write[0] < profile['progress_min_interval_seconds']:
                return
        last_progress_write[0] = perf_counter()
        row = dict(stage=stage, utc=e11.now(), pid=os.getpid(),
             task_elapsed_seconds=perf_counter()-started, level=active_level,
             formal_commits=len(commits), native_commits=sum(c['native'] for c in commits),
             last_inner=last_inner, last_outer=last_outer, last_transfer=last_transfer)
        row.update(values)
        if efficient_cpu_budget:
            row['work'] = work
            row['costs'] = costs
            row['cpu_comparison_cost'] = cpu_cost
            if backend is not None:
                row['comparison_counts'] = dict(backend.comparison_counts)
        dump(out/'progress.json', row)
        if stage in ('load_started', 'solve_started', 'outer_checked', 'transfer_attempt',
                     'level_prepared', 'level_completed', 'transfer_rejected',
                     'native_label_evaluation_started', 'worker_completed'):
            print(json.dumps(row), flush=True)

    def save(name, **arrays):
        nonlocal io_seconds
        if first_use_dry is not None:
            return dict(path=name, sha256=None, first_use_dry_not_written=True)     # FX1 F2': no arrays in the dry pass
        tick = perf_counter(); temp = out/(name+'.partial.npz')
        if guard is not None:
            # Stored NPZ: array payload plus generous NPY/ZIP metadata allowance.
            pending = sum(a.nbytes for a in arrays.values())+65536*len(arrays)+1024**2
            if writer is not None:
                pending += writer.queued_bytes        # review E76: queued checkpoints are not on disk yet
            on_writer = (guard_on_writer and writer is not None and name != 'final_native.npz' and writer.deferred_hash
                         and all(_plain_array(a) for a in arrays.values()))   # review: else npz() writes it unchecked
            if not on_writer:
                tick_guard = perf_counter(); guard.check('before_checkpoint', pending_bytes=pending, task=task['name'])
                guard_seconds[0] += perf_counter()-tick_guard
            # v39: otherwise the writer thread checks right before it writes this checkpoint (writer_guard)
            progress('checkpoint_write_started', checkpoint=name, pending_bytes=pending,
                     formal_commits=len(commits))
        if writer is not None and name != 'final_native.npz' and writer.deferred_hash:
            # E102: a PendingDigest; the writer serializes, hashes and stores these arrays later, in order.
            row = dict(path=name, sha256=writer.npz_deferred(out/name, **arrays))
        elif writer is not None and name != 'final_native.npz':
            # E76: SHA-256 of the exact in-memory bytes; the ordered writer stores them later.
            row = dict(path=name, sha256=writer.npz(out/name, **arrays))
        else:
            if writer is not None:
                writer.flush()
            np.savez(temp, **arrays); temp.replace(out/name)
            row = dict(path=name, sha256=file_sha256(out/name))
        io_seconds += perf_counter()-tick
        return row

    def dump_index(path, value):
        # E76: an index naming checkpoints is written after them (same writer queue, same text).
        if writer is not None:
            writer.json(path, value)
        else:
            dump(path, value)

    def on_work(event):
        nonlocal work, native_baseline
        work = event
        if event.get('phase')=='fused_cpu_comparison' and event['event'] in ('operation_finished','operation_aborted'):
            cpu_cost['finished' if event['event']=='operation_finished' else 'aborted'] += 1
            cpu_cost['seconds'] += event['seconds']
        if event['event'] == 'sequential_baseline_ready' and event['grid_id'] == grid.grid_id:
            if event['baseline_at_zero'] is not True:
                raise RuntimeError('native cleanup requires the original zero baseline')
            native_baseline = dict(event)
            dump(out/'native_baseline.json', native_baseline)
        if event['event'] != 'work_progress':
            row = dict(event, level=active_level, visible_seconds=perf_counter()-started)
            if throughput:
                handle = stream('work_events.jsonl'); handle.write(json.dumps(row)+'\n'); handle.flush()
            else:
                with (out/'work_events.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
        progress(event['event'])

    def on_promote(problem, state):
        if throughput:
            # Derivative-ready diagnostic only: identity metadata, no arrays.
            record = dict(path=None, sha256=None, arrays_saved=False, policy=profile['name'])
        else:
            record = save(f'promoted_{len(promoted)+1:05d}_NONCOMMIT.npz', u=state.u, warped=state.warped)
        record.update(level=active_level, role='derivative_ready_NOT_outer_commit',
            grid_id=problem.grid.grid_id, eval_id=state.eval_id, controls_id=problem.controls.controls_id,
            certificate=asdict(state.certificate), energy=asdict(state.energy),
            visible_seconds=perf_counter()-started)
        if screen_records:
            record['energy_screened'] = True      # E-HS1b: may hold T90 float32 screening values (diagnostic record only)
        promoted.append(record)
        if throughput:
            _jsonl(stream('promoted.jsonl'), record)
        else:
            dump(out/'promoted.json', promoted)
        progress('promoted_noncommit_saved', formal_commits=len(commits), promoted=len(promoted))

    def on_inner(level, row):
        nonlocal last_inner, costs
        last_inner = {k: row.get(k) for k in ('iteration', 'attempt', 'accepted', 'reason',
                     'tau', 'gradient_norm', 'forcing_target', 'jd_prediction', 'jd_witnesses_after_trial',
                     'step_policy','physical_poisson_calls','hybrid_event','hybrid_call_start','hybrid_trigger')}
        attempts.append(dict(level=level, visible_seconds=perf_counter()-started, **row))
        if efficient_cpu_budget:
            costs = progress_costs(attempts,commits)
        if throughput:
            _jsonl(stream('inner_attempts.jsonl'), attempts[-1])
        else:
            dump(out/'inner_attempts.json', attempts)
        progress('inner_attempt', accepted=row['accepted'], reason=row['reason'], formal_commits=len(commits))

    def on_outer(level, problem, start, inner, row, candidate):
        nonlocal last_outer, native_anchor, costs, native_exact_row
        last_outer = {k: row.get(k) for k in ('outer_i', 'outer_accepted', 'reason',
                     'committed_GF', 'GF_threshold')}
        history.append(dict(level=level, **row))
        if screen_records and row['outer_accepted'] and level == 3:
            native_exact_row = row                 # E-HS1b: float64 terms of the latest native formal commit
        if row['outer_accepted']:
            reference = saved_maps.get(start.certificate.state_hash) if lean_commits else None
            if reference is not None:
                # E48 IO-only: the old reference map is the exact array of an earlier saved
                # file (previous commit or certified level initializer); warped is u's sample.
                record = save(f'committed_{len(commits)+1:04d}_L{level}.npz', u=candidate.u)
                record.update(lean_commit=True, reference_map=dict(reference, state_hash=start.certificate.state_hash))
            else:
                record = save(f'committed_{len(commits)+1:04d}_L{level}.npz',
                              before_u=start.u, u=candidate.u, warped=candidate.warped)
            saved_maps[candidate.certificate.state_hash] = dict(path=record['path'], sha256=record['sha256'], array='u')
            if dedup_files and level == 3:          # v38: the native level_completed may refer to this file
                last_native[0] = (candidate.u, dict(path=record['path'], sha256=record['sha256'], array='u',
                                                    state_hash=candidate.certificate.state_hash))
            record.update(level=level, native=level == 3, role='formal_outer_commit',
                ratio=row['committed_mse_ratio'], GF=row['committed_GF'],
                certificate=asdict(candidate.certificate), eval_id=candidate.eval_id,
                solver_commit_seconds=row['elapsed_seconds'], visible_seconds=perf_counter()-started)
            commits.append(record)
        if efficient_cpu_budget:
            costs = progress_costs(attempts,commits)
        if throughput:
            _jsonl(stream('history.jsonl'), history[-1])
        else:
            dump(out/'history.json', history)
        dump_index(out/'commits.json', commits)
        if efficient_cpu_budget and row['outer_accepted'] and level == 3:
            if native_baseline is None or native_baseline['grid_id'] != problem.grid.grid_id:
                raise RuntimeError('missing native baseline for a formal commit')
            # Keep only export arrays, not another full derivative-ready state.
            state = SimpleNamespace(u=candidate.u, warped=candidate.warped,
                certificate=candidate.certificate, energy=candidate.energy if not screen_records else _row_energy(row),
                **({} if not screen_records else dict(eval_id=candidate.eval_id)))   # E-HS1b: identifies the commit's candidate
            native_anchor = SimpleNamespace(final_state=state, status='work_budget_interrupted',
                native_mse=native_baseline['native_mse'], final_mse=mse(candidate, problem.grid, problem),
                mse_ratio=row['committed_mse_ratio'], GF_initial=row['GF_initial'],
                GF_final=row['committed_GF'], GF_threshold=row['GF_threshold'],
                GD_final=row['GD_candidate'], GE_final=row['GE_candidate'], counts=dict(row['counts_after']),
                export_identity=dict(commit_path=record['path'], commit_sha256=record['sha256'],
                    problem_id=problem.problem_id, controls_id=problem.controls.controls_id,
                    eval_id=candidate.eval_id, **(dict(energy_reference='old_reference_at_formal_commit',
                    reference_array='before_u in the committed checkpoint') if not fixed_reference(level) else dict(
                    energy_reference='T92 fixed native reference (never refreshed)',          # review T92 r1 minor 4
                    reference_array='the native level_prepared map' if native_ablation['value'] == 'fixed' else 'u = 0'))))
        progress('outer_checked', formal_commits=len(commits), reason=row['reason'])

    def fixed_reference(level):
        """T92 (D2-c/c'): the native level runs with a fixed reference."""
        return level == 3 and native_ablation is not None and native_ablation['switch'] == 'native_reference'

    def same_level_file(event, field):
        """v38 (dedup_level_files): an earlier file of this run that holds exactly `field`, or None.
        A transfer attempt whose certificate state hash (array_hash(u)) equals the seed of preoptimizer_done or an earlier
        attempt of the level refers to that file; the native level_completed refers to the last native formal commit's file
        when its field is that commit's u (the same array, or the same bytes). level_prepared, commits and final_native are
        always written under their own names; coarse levels' level_completed too (tools read them by name)."""
        if event['event'] == 'transfer_attempt':
            key = (event.get('certificate') or {}).get('state_hash')
            return dict(level_maps[key], reason='same_state_hash') if key and key in level_maps else None
        if event['event'] == 'level_completed' and event['level'] == 3 and last_native[0] is not None:
            u, ref = last_native[0]
            arr = np.asarray(field)
            same_hash = (event.get('final_certificate') or {}).get('state_hash') == ref['state_hash']   # review: also the record
            if same_hash and (arr is u or (arr.shape == u.shape and arr.dtype == u.dtype == np.float64
                                           and np.array_equal(arr.view(np.uint64), np.asarray(u).view(np.uint64)))):
                return dict(ref, reason='last_native_commit_u')
        return None

    def on_level(event, grid, field):
        nonlocal active_level, last_inner, last_outer, last_transfer
        if active_level != event['level']:
            last_inner, last_outer, last_transfer = {}, {}, {}
            level_maps.clear()                   # v38: references stay within a level
        active_level = event['level']
        if event['event'] == 'transfer_attempt':
            last_transfer = {k: event.get(k) for k in ('alpha', 'certificate_passed', 'certificate')}
            if 'barrier_domain_passed' in event:     # T68 only; the samples record is unchanged
                last_transfer.update({k: event[k] for k in ('barrier_domain_passed', 'bernstein_min_coefficient')})
        suffix = f'_a{event["attempt_index"]:02d}' if event["event"] == "transfer_attempt" else ""
        same = same_level_file(event, field) if dedup_files else None
        if same is None:
            record = save(f'level_{active_level}_{event["event"]}{suffix}.npz', u=field)
        else:                                    # v38: the same bytes are already in an earlier file of this run
            record = dict(path=same['path'], sha256=same['sha256'], same_file_as=same)
        if dedup_files:
            key = ((event.get('preoptimizer') or {}).get('seed_hash') if event['event'] == 'preoptimizer_done' else
                   (event.get('certificate') or {}).get('state_hash') if event['event'] == 'transfer_attempt' else None)
            if key and key not in level_maps:
                level_maps[key] = dict(path=record['path'], sha256=record['sha256'], array='u', state_hash=key)
        proof = event.get('transfer_certificate')
        if event['event'] == 'level_prepared' and isinstance(proof, dict) and proof.get('state_hash'):
            saved_maps[proof['state_hash']] = dict(path=record['path'], sha256=record['sha256'], array='u')
        events.append(dict(event, **record, visible_seconds=perf_counter()-started))
        dump_index(out/'levels.json', events)
        progress(event['event'], formal_commits=len(commits))

    progress('load_started')
    with patch.object(e11.run, 'DATA', Path(data)):
        arrays, grid, manifest = e11.run.inputs()
    if np.any(arrays['initial_u']):
        raise ValueError('requires the original zero initial field, not a resumed run')
    mu = 1. if task['strength'] == 'strong' else .001
    if geometry_weight is not None:
        # T77 (E79, MODEL PARAMETER CHANGE, user-approved 2026-10-03): explicit mu = nu instead of the strength
        # default; every other setting of the strength (hierarchy, metric, level shares) is unchanged.
        mu = geometry_weight
    weights_levels = None
    if geometry_weight_levels is not None:
        # T78 (E80, MODEL PARAMETER CHANGE, user-approved 2026-10-03): per-level mu = nu; `weights` is the native level's.
        weights_levels = tuple(Weights3D(manifest['weights']['lam'], w, w, .002) for w in geometry_weight_levels)
        mu = geometry_weight_levels[3]
    weights = Weights3D(manifest['weights']['lam'], mu, mu, .002)
    if native_ablation is not None and native_ablation['switch'] == 'native_geometry_weights':
        # T92 (D2-a/b, MODEL CHANGE, ablation switch): the native level's mu or nu is 0 (validated above); the metric's
        # b = (mu + nu)/2 follows (main ablation (i)); config.weights is the native level's, as for T78
        from jd_curl.three_d.problem import AblationWeights3D
        weights = AblationWeights3D(manifest['weights']['lam'], native_ablation['value']['mu'], native_ablation['value']['nu'], .002)
        weights_levels = weights_levels[:3]+(weights,)
    native_base_weights = None
    if native_ablation is not None and native_ablation.get('penalty_scale') is not None:
        # T92 (D2-c' down-weighted arm, MODEL CHANGE, user 2026-10-19): lam, mu, nu of the native level times s (kappa
        # unchanged); the metric follows (a = s lam + shift, b = s (mu + nu)/2, the GGN diagonal from these weights)
        native_base_weights, s_pen = weights, native_ablation['penalty_scale']
        weights = Weights3D(weights.lam*s_pen, weights.mu*s_pen, weights.nu*s_pen, weights.kappa)
        weights_levels = weights_levels[:3]+(weights,)
    memory = ParallelMemoryConfig(2048, 16384, 1) if low_load else ParallelMemoryConfig(8192, 65536, 4)
    if throughput:
        memory = ParallelMemoryConfig(profile['cell_tile_size'], profile['spline_query_chunk'],
                                      profile['cpu_workers'])
    certificate = replace(e11.CERTIFICATE, tile_size=2048) if low_load else e11.CERTIFICATE
    if certificate_tile_size is not None:
        certificate = replace(certificate, tile_size=certificate_tile_size)
    if throughput and compiled_certificate:
        # Tile size only groups whole cells; per-cell proofs and the min are order-free.
        certificate = replace(certificate, tile_size=profile['certificate_tile_size'])
    if compiled_certificate and not (throughput and profile.get('parallel_certificate')):
        certificate = CompiledEnclosureCertificateConfig(**asdict(certificate), kernel=certificate_runtime['kernel'])
    if throughput and profile.get('parallel_certificate'):
        # Same per-cell hoisted arithmetic, whole-map parallel (E50); legacy classes unchanged.
        certificate = ParallelEnclosureCertificateConfig(**asdict(certificate), fallback_workers=profile['fallback_workers'])
        if profile.get('compiled_fallback'):
            # E54: same decisions/endpoints/counts; only the fallback array execution differs.
            certificate = CompiledFallbackCertificateConfig(**asdict(certificate))
        if profile.get('prescreen_certificate'):
            # E71 (T75): rigorous float Bernstein prescreen before the interval fallback.
            from jd_curl.three_d.feasibility import PrescreenCertificateConfig
            certificate = PrescreenCertificateConfig(**asdict(certificate))
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    backend = RecordingBackend(device, on_promote, reuse_prepared=True,
        **(dict(fused_cpu=True) if efficient_cpu_budget else {}),
        **(dict(near_boundary_cpu=False) if gpu_near_boundary else {}),
        **(dict(reuse_reference=True) if reuse_reference else {}),
        **(dict(device_batch=batch) if batch is not None else {}),
        **(dict(selective_kernels=True) if selective_kernels else {}),
        **(dict(overlap_certificate=True) if throughput and profile.get('overlap_certificate') else {}),
        **(dict(fused_kernels=True) if throughput and profile.get('fused_kernels') else {}),
        **(dict(fused_jd_root=True) if throughput and profile.get('fused_jd_root') else {}),
        **(dict(fused_diagonal=True) if throughput and profile.get('fused_diagonal') else {}),
        **(dict(cpu_gradient_band=profile['cpu_gradient_band']) if throughput and profile.get('cpu_gradient_band') else {}),
        **(dict(fused_hvp=True) if throughput and profile.get('fused_hvp') else {}),
        **(dict(lean_host=True) if throughput and profile.get('lean_host') else {}),
        **(dict(cpu_gradient_margin=profile['cpu_gradient_margin']) if throughput and profile.get('cpu_gradient_margin') else {}),
        **(dict(lean_identity=True) if throughput and profile.get('lean_identity') else {}),
        **(dict(boundary_reuse=True) if throughput and profile.get('boundary_reuse') else {}),
        **(dict(screen_prefilter=True) if throughput and profile.get('screen_prefilter') else {}),
        **(dict(fused_node_jd=True) if throughput and profile.get('fused_node_jd') else {}),
        **(dict(lean_checks=True) if throughput and profile.get('lean_checks') else {}),
        **{name: True for name in BACKEND_SWITCHES_E97_E98 if throughput and profile.get(name)})   # E97, E98
    if throughput and profile.get('gpu_hull_first_use_seed'):
        # v38: the dry pass harvests T84's synthetic first-use keys from this attribute and the run seeds them (registry)
        backend.gpu_hull_first_use_seed = True
    first_use_seeded = None
    if preverified is not None:
        # FX1 F2' (v35): the E93/E94/E100/E99 first-use checks of this grid were done before
        # the task clock, on a synthetic field (launcher first_use_prewarm); their keys seed this new backend.
        from jd_curl.three_d.backends.first_use_registry import seed_backend
        first_use_seeded = seed_backend(backend, preverified)
    if ggn_reuse_spec is not None:
        backend.configure_ggn_reuse(*ggn_reuse_spec)                              # GG2
    if throughput and profile.get('lean_host'):
        # E76: memo only digests/bindings of immutable arrays (same values); process-wide, worker only.
        from jd_curl.three_d.problem import enable_immutable_hash_cache
        from jd_curl.three_d.backends.cpu_comparison import enable_binding_memo
        enable_immutable_hash_cache(); enable_binding_memo()
    if throughput and profile.get('lean_identity'):
        # E89 (P01): facts proven on immutable arrays (certificate digest/predicates, finite theta).
        from jd_curl.three_d.problem import enable_identity_facts
        enable_identity_facts()
    if throughput and profile.get('lean_outer'):
        # E89 (P08): sum q per grid computed once for the reported MSE (the same longdouble sum).
        from jd_curl.three_d.sequential import enable_weight_sum_cache
        enable_weight_sum_cache()
    inner = (JDStepPGConfig3D if local_barrier else PGConfig3D)(max_iterations=INNER_STEPS)
    e53 = conservative_pg_route(resolution_exit=resolution_exit, jd_step_global=jd_step_global, bb2_ggn_guard=bb2_ggn_guard,
                                pg_jd_affine_activation=pg_jd_affine_activation, pg_metric_policy=pg_metric_policy)
    if pg_metric_policy is not None and not e53:
        # T59 (E49/A1): only the SPD PG metric changes; requires the local-barrier JD-step PG route.
        if not local_barrier or inner_variant is not None:
            raise ValueError('pg_metric_policy requires the local-barrier PG route')
        from jd_curl.three_d.solvers.poisson_gradient import CurvaturePGConfig3D
        inner = CurvaturePGConfig3D(max_iterations=INNER_STEPS, metric_policy=pg_metric_policy, shift_factor=pg_shift_factor)
    if throughput and profile.get('fused_jd_root') and not jd_step_global:
        raise ValueError('throughput v5 (fused JD root) is only meaningful with jd_step_global')
    if barrier_basis not in ('samples', 'bernstein') or (barrier_basis != 'samples' and not local_barrier):
        raise ValueError('barrier_basis bernstein requires the closed27 local-barrier route')
    curvature_weight = float(curvature_weight)   # T76 (E78): alpha of the curvature regularizer (0 = absent)
    if not np.isfinite(curvature_weight) or curvature_weight < 0:
        raise ValueError('curvature_weight must be a finite nonnegative float')
    if barrier_basis != 'samples' and not transfer_backtracking:
        # Only the backtracking transfer checks the T68 domain (alpha = .9^k back-off).
        raise ValueError('barrier_basis bernstein requires transfer_backtracking')
    if e53:
        # E53 (T62/T63): explicit conservative-A1 metric and/or backtracking resolution exit.
        if not local_barrier or inner_variant is not None:
            raise ValueError('E53 PG options require the local-barrier PG route')
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D
        inner = ConservativePGConfig3D(max_iterations=INNER_STEPS, metric_policy=pg_metric_policy or 'legacy_q1',
            shift_factor=pg_shift_factor, resolution_exit=bool(resolution_exit), jd_step_global=bool(jd_step_global),
            bb2_ggn_guard=bool(bb2_ggn_guard),
            **(dict(jd_affine_activation=pg_jd_affine_activation) if pg_jd_affine_activation is not None else {}))
    if throughput and profile.get('fast_host'):
        # E76: bit-identical host-side shortcuts of the PG inner solver (explicit config field).
        from dataclasses import replace as _replace_fast
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _Conservative
        if not isinstance(inner, _Conservative) or inner_variant is not None:
            raise ValueError('throughput v9 (fast_host) requires the E53-type PG route')
        inner = _replace_fast(inner, fast_host=True)
    if throughput and profile.get('compiled_extension'):
        # E81: compiled T69/T73 estimator (bit-identical values); the E53-type PG route is required.
        from dataclasses import replace as _replace_compiled
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsCompiled
        if not isinstance(inner, _ConsCompiled) or inner_variant is not None:
            raise ValueError('throughput v10 (compiled_extension) requires the E53-type PG route')
        inner = _replace_compiled(inner, compiled_extension=True)
    if throughput and profile.get('fused_steps'):
        # E89 (P03): compiled fused host step algebra (bit-identical); the E53-type route with fast_host.
        from dataclasses import replace as _replace_fused
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsFused
        if not isinstance(inner, _ConsFused) or inner_variant is not None or not inner.fast_host:
            raise ValueError('throughput v11 (fused_steps) requires the E53-type PG route with fast_host')
        inner = _replace_fused(inner, fused_steps=True, **(dict(metric_host=True) if profile.get('metric_host') else {}),
                               **(dict(lean_checks=True) if profile.get('lean_checks') else {}))
        if profile.get('metric_host'):
            # E89 (P07): the Poisson spectrum per grid and metric is computed once (same function, same values).
            from jd_curl.three_d.poisson import enable_spectrum_cache
            enable_spectrum_cache()
    solver='pg'
    if inner_variant is not None:
        from jd_curl.three_d.solvers.sequential_qp import EfficientQPConfig3D,CoverageQPConfig3D,SafeguardedQPConfig3D
        from jd_curl.three_d.solvers.poisson_gradient import TriggeredQPConfig3D
        inner={'efficient':EfficientQPConfig3D,'coverage':CoverageQPConfig3D,'hybrid':TriggeredQPConfig3D,
               'safeguarded':SafeguardedQPConfig3D}[inner_variant](max_iterations=3)
        solver='pg' if inner_variant=='hybrid' else 'jd_qp'
    inner_configs = (inner,)*4
    if pg_native_metric_policy is not None:
        # T71 (E62): the native level uses its own metric policy; F/E/acceptance/stopping unchanged.
        from dataclasses import replace as _replace
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D, T62_POLICIES, T70_POLICY
        if not isinstance(inner, ConservativePGConfig3D) or inner_variant is not None or task['hierarchy'] != 'four':
            raise ValueError('pg_native_metric_policy requires the E53-type four-level PG route')
        from jd_curl.three_d.solvers.poisson_gradient import T74_POLICY
        activation = inner.jd_affine_activation if pg_native_metric_policy in T62_POLICIES+(T70_POLICY, T74_POLICY) else None
        inner_configs = (inner,)*3+(_replace(inner, metric_policy=pg_native_metric_policy, jd_affine_activation=activation),)
    if pg_metric_policy_levels is not None:
        # T78 (E80): per-level metric policies, same activation rule as T71.
        from dataclasses import replace as _replace_levels
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsL, T62_POLICIES as _T62L, T70_POLICY as _T70L, T74_POLICY as _T74L
        if not isinstance(inner, _ConsL) or inner_variant is not None:
            raise ValueError('pg_metric_policy_levels requires the E53-type four-level PG route')
        inner_configs = tuple(_replace_levels(inner, metric_policy=p,
                                              jd_affine_activation=inner.jd_affine_activation if p in _T62L+(_T70L, _T74L) else None)
                              for p in pg_metric_policy_levels)
    if inner_steps_levels is not None:
        # T82 (E85): replace only the selected levels' accepted-step limit (validated above, before any output).
        from dataclasses import replace as _replace_steps
        inner_configs = tuple(c if k is None else _replace_steps(c, max_iterations=k)
                              for c, k in zip(inner_configs, inner_steps_levels))
    if first_trial_cap_levels is not None:
        # T81 (E84): replace only the capped levels' inner configurations (validated above, before any output).
        from dataclasses import replace as _replace_cap
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsCap
        if not all(isinstance(c, _ConsCap) for c in inner_configs):
            raise ValueError('first_trial_cap_levels requires the E53-type PG route')
        inner_configs = tuple(c if cap is None else _replace_cap(c, first_trial_cap=cap, first_trial_scope=first_trial_scope)
                              for c, cap in zip(inner_configs, first_trial_cap_levels))
    if momentum_first_levels is not None:
        # T85 (E111): switch the momentum first direction on for the selected levels only (validated above).
        from dataclasses import replace as _replace_momentum
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsMomentum
        if not all(isinstance(c, _ConsMomentum) for c in inner_configs):
            raise ValueError('momentum_first_levels requires the E53-type PG route')
        inner_configs = tuple(_replace_momentum(c, momentum_first=True) if on else c
                              for c, on in zip(inner_configs, momentum_first_levels))
    if native_ablation is not None and native_ablation['switch'] == 'native_metric_ablation':
        # T92 (D2-d, PATH CHANGE, ablation switch): the native level's metric inverse replaced (validated above)
        from dataclasses import fields as _fields_abl
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D as _ConsAbl, MetricAblationPGConfig3D
        if type(inner_configs[3]) is not _ConsAbl:
            raise ValueError('native_metric_ablation requires the E53-type PG configuration on the native level')
        inner_configs = inner_configs[:3]+(MetricAblationPGConfig3D(
            **{f.name: getattr(inner_configs[3], f.name) for f in _fields_abl(inner_configs[3])},
            metric_inverse=native_ablation['value']),)
    if native_ablation is not None and native_ablation['switch'] == 'native_inner':
        # T92 D2-e (PATH CHANGE, ablation switch): the native level's inner solver is the library barrier-GGN LM with the
        # same accepted-step limit per outer; levels 0-2 are not solved on the pre-optimizer route
        from jd_curl.three_d.solvers.lm import LMConfig3D
        inner_configs = inner_configs[:3]+(LMConfig3D(max_iterations=inner_configs[3].max_iterations,
                                                      preconditioner=NATIVE_LM_PRECONDITIONER),)
    outer = SequentialReuseConfig3D(max_iterations=OUTER_STEPS, eta0=.01,
                                    **(dict(lean_outer=True) if throughput and profile.get('lean_outer') else {}))
    solve_start = perf_counter()
    remaining = max(0., deadline-solve_start)
    shares = CUMULATIVE_SHARES
    if level_shares is not None:
        # T67 (E56): explicit cumulative level time shares; the default schedule is unchanged.
        shares = tuple(float(x) for x in level_shares)
        if (task['hierarchy'] != 'four' or len(shares) != 4 or shares[-1] != 1.
                or not all(0. < a < b for a, b in zip(shares, shares[1:])) or not shares[0] > 0.):
            raise ValueError('level_shares: four strictly increasing cumulative fractions ending at 1')
    level_deadlines = [solve_start+fraction*remaining for fraction in shares]
    budget = lambda: perf_counter() >= deadline
    if first_use_dry is not None:
        # FX1 F2': the dry pass ends after its first native formal commit (or after max_seconds)
        dry_stop = int(first_use_dry.get('stop_after_native_commits', 1))
        dry_end = started+float(first_use_dry.get('max_seconds', 45.))
        budget = lambda: perf_counter() >= min(deadline, dry_end) or sum(c['native'] for c in commits) >= dry_stop
    outer_gate = None
    if throughput and profile.get('outer_gate') and first_use_dry is None and task['hierarchy'] == 'four':
        # FX1 F4 (v35): the native level does not start an outer that cannot finish before the solve cutoff
        from jd_curl.three_d.outer_gate import OuterGate
        outer_gate = OuterGate(lambda: deadline-perf_counter(), window=int(profile.get('outer_gate_window', 3)))
    config = dict(stage='E19-B', task=task, device=device, weights=asdict(weights),
        memory=asdict(memory), certificate=asdict(certificate), barrier_location='closed27_uniform',
        inner=asdict(inner), outer=asdict(outer), input_sha256=manifest['pair_sha256'],
        **({} if pg_native_metric_policy is None else dict(native_inner=asdict(inner_configs[3]),
            metric_by_level=[getattr(c, 'metric_policy', None) for c in inner_configs])),   # D2-e: an LM level has none
        numerical_policy=backend.evaluation_backend_id, reuse_prepared=True, checkpoint_encoding='stored',
        baseline_policy='native_original_zero_MSE_and_GF; never warm-start denominator',
        termination_policy='original_full_GF_only; time budget is NOT convergence',
        export_reserve_seconds=reserve, solve_cutoff_seconds=deadline-started,
        level_deadline_seconds=[x-started for x in level_deadlines] if task['hierarchy']=='four' else None,
        cumulative_shares=shares, native_grid_id=grid.grid_id,
        labels_policy='native seg4 primary; seg35 audit-only; neither in loss',
        **({} if pre_optimizer is None else dict(preoptimizer=dict(policy=pre_optimizer['policy'], taper=pre_optimizer.get('taper', 4),
                                                             config=asdict(preopt_config), device=device, dtype='float32',
                                                             **({} if preopt_execution is None else dict(execution={
                                                                 k: x for k, x in asdict(preopt_execution).items()
                                                                 if k in ('lean_sync', 'static_grid', 'cuda_graph') or x}))))),   # T88; S30 PRE speed; S30 P-12/P-6: new keys only when set
        threads={k: os.environ.get(k) for k in e11.run.THREADS})
    if native_transfer_alphas is not None:                                       # AP2: a new key only when set
        from jd_curl.three_d.multiresolution import TRANSFER_ALPHA_LISTS
        alpha_policy, alpha_values = TRANSFER_ALPHA_LISTS[native_transfer_alphas]     # review AP2 B1: not `policy` (resource policy)
        config['native_transfer_alphas'] = dict(name=native_transfer_alphas, policy=alpha_policy, alphas=list(alpha_values),
            role='native transfer of the pre-optimizer seed only (AP2, path): the first certified scale of this list, '
                 'each relative to the seed; coarse levels are skipped; F, barrier, certificate, E/F, GF unchanged')
    if inner_variant is not None:config.update(stage='E44-E46-hour',solver=solver,inner_variant=inner_variant)
    if inner_variant=='safeguarded':config['stage']='E47-safeguarded-QP'
    if local_barrier:
        if barrier_basis != 'samples':
            if task['hierarchy'] != 'four' or inner_variant is not None:
                raise ValueError('barrier_basis requires the four-level PG route')
        config['barrier_model'] = dict(policy='localized-c2-v1', activation=.05, delta=0., kappa=.002, basis=barrier_basis,
            location='closed27_uniform', jd_step='historical-cell-closed27-cubic-v1')
    if first_trial_cap_levels is not None:
        from jd_curl.three_d.solvers.poisson_gradient import FIRST_TRIAL_CAP_POLICY
        config['first_trial_cap'] = dict(policy=FIRST_TRIAL_CAP_POLICY, levels=list(first_trial_cap_levels),
            scope=first_trial_scope, unit='level voxels (max over free nodes of sqrt(sum_d (p_d/h_d)^2))',
            role='trial step upper bound only (T81, path); never a certificate')
    if ggn_reuse_spec is not None:                                              # GG2: a new key only when set
        from jd_curl.three_d.backends.accelerated import GGN_REUSE_POLICY
        config['ggn_diagonal_reuse'] = dict(policy=GGN_REUSE_POLICY, spec=ggn_diagonal_reuse, theta=ggn_reuse_spec[0],
            period=ggn_reuse_spec[1], role='PG metric only (GG2): the GGN diagonal of the equilibration scale keeps the per-cell '
            'geometry/barrier locals of cells with E94 minimum J >= theta * activation at this and the previous build; full '
            'build (bit for bit the default) every period-th build; F, barrier, certificate, E/F, GF unchanged')
    if active_spec is not None:
        config['active_set'] = dict(active_spec.record(), role='native level only: node weights of the PG direction p = -(wS) M^-1 (wS) g '
            '(T89, path); nodes with w = 0 never move; F, barrier, certificate, E/F, GF and the ratio denominator unchanged')
    if native_start_scale is not None:
        config['native_start_scale'] = dict(policy='native-start-scale-v1', beta=native_start_scale,
            role=('native transfer tries beta * .9^k * the tapered pre-optimizer seed (T83 on the T88 route, C2; path; '
                  'changes the T38g alpha = 1 first rule)') if pre_optimizer is not None else
                 'native transfer tries beta * .9^k * P u_L2 (T83, path; changes the T38g alpha = 1 first rule)')
    if practical_stop is not None:
        config['practical_stop'] = dict(asdict(practical_stop), role='native level only: stop after a formal native '
            'commit when (r_ref - r_n)/(t_n - t_ref) * T_rem <= epsilon (H17; status native_practical_stop, success '
            'false; the GF threshold and the success definition are unchanged)')
    if transfer_smoothing is not None:
        config['transfer_smoothing'] = dict(transfer_smoothing,
            role='transfers into the native level (scope native) or into every level (scope all) certify alpha_k * '
                 'S_sigma(P u) instead of alpha_k * P u (A19, path only)')
    if inner_steps_levels is not None:
        config['inner_steps'] = dict(policy='explicit-inner-steps-by-level-v1', levels=list(inner_steps_levels), default=INNER_STEPS,
            role='accepted inner PG steps per outer iteration on each level (T82, path; mainline parameter as explicit option)')
    if momentum_first_levels is not None:
        from jd_curl.three_d.solvers.poisson_gradient import MOMENTUM_POLICY
        config['momentum_first'] = dict(policy=MOMENTUM_POLICY, levels=list(momentum_first_levels),
            role='first inner step of each outer along the last accepted outer increment when it descends, proposal '
                 'initial_step; JD prediction, Armijo, backtracking, acceptance unchanged (T85, path only)')
    if first_trial_cap_levels is not None or inner_steps_levels is not None or momentum_first_levels is not None:
        config['inner_by_level'] = [asdict(c) for c in inner_configs]    # review E84: config['inner'] is the level template
    if curvature_weight:
        # T76 (E78, MODEL CHANGE, user-approved 2026-10-02): F = D + R + B with R = alpha/2 sum q |L u|^2.
        config['curvature_model'] = dict(policy='seven-point-laplacian-squared-v1', weight=curvature_weight,
            scope='absolute term of the outer F; residual block after proximity; same alpha and physical spacing on every level')
    if curvature_weight_levels is not None:
        # G2: config.curvature_model.weight is the native level's alpha (as config.weights is for T78)
        config['curvature_model'] = dict(policy='seven-point-laplacian-squared-by-level-v1',
            weight=curvature_weight_levels[3], levels=list(curvature_weight_levels),
            scope='absolute term of the outer F; residual block after proximity; level L uses alpha_L and its own '
                  'physical spacing (G2); weight is the native alpha')
    if geometry_weight is not None:
        config['geometry_weight'] = dict(policy='explicit-mu-nu-v1', value=mu, base_strength=task['strength'],
            scope='T77: mu = nu override of the strength default (strong 1, weak .001); other strength settings unchanged')
    if weights_levels is not None:
        config['weights_by_level'] = [asdict(w) for w in weights_levels]
        config['geometry_weight'] = dict(policy='explicit-mu-nu-by-level-v1', values=list(geometry_weight_levels),
            base_strength=task['strength'], scope='T78: mu = nu per pyramid level; config.weights is the native level')
    if pg_metric_policy_levels is not None:
        config['metric_by_level'] = [getattr(c, 'metric_policy', None) for c in inner_configs]   # D2-e: an LM level has none
        config['native_inner'] = asdict(inner_configs[3])
    if native_ablation is not None:
        config['native_ablation'] = dict(native_ablation, native_inner=asdict(inner_configs[3]),   # T92 (D2)
            **({} if native_base_weights is None else dict(base_weights=asdict(native_base_weights))))
    if data_weight is not None:
        config['data_weight_model'] = dict(data_weight, scope='T80: D = 1/2 sum q m (W-F)^2 with m from the fixed image '
            '(brain support smoothed, beta outside); per level from that level\'s fixed image; proximity/JD/curl/barrier/R keep q; '
            'recorded MSE ratios are UNWEIGHTED (sum q (W-F)^2 / sum q)')
    if efficient_cpu_budget:
        config['execution_policy'] = CPU_BUDGET_POLICY
    if gpu_near_boundary:
        config['gpu_near_boundary'] = True
    if reuse_reference:
        config['reuse_reference'] = True
    if batch is not None:
        config['device_batch'] = asdict(batch)
    if selective_kernels:
        config['selective_kernels'] = True
    if compiled_certificate:
        config['compiled_certificate'] = True
        if hoisted_certificate:
            config['hoisted_certificate'] = True
        config['certificate_runtime'] = certificate_runtime
    if transfer_backtracking:
        config['transfer_backtracking'] = dict(policy=TRANSFER_BACKTRACK_POLICY, alphas=TRANSFER_BACKTRACK_ALPHAS)
    if advance_stalled_levels:
        config['advance_stalled_levels'] = True
    if lean_commits:
        config['commit_policy'] = 'lean-u-with-reference-file-v1 (IO only; before_u = referenced earlier saved u)'
    if first_level != 0:
        if task['hierarchy'] != 'four':
            raise ValueError('first_level requires the four-level hierarchy')
        config['hierarchy_policy'] = dict(first_level=first_level, skipped_levels=list(range(first_level)),
            role='explicit E48 hierarchy probe; default four levels unchanged')
    if level_shares is not None:
        config['level_share_policy'] = dict(cumulative_shares=list(shares), default=list(CUMULATIVE_SHARES),
            role='T67 explicit level time shares (E56); transfer/acceptance/GF unchanged')
    if hierarchy_policy is not None:
        # T64: the name is recorded only when the level actually follows the policy.
        levels = HIERARCHY_POLICIES.get(hierarchy_policy)
        if levels is None or task['hierarchy'] != 'four' or task.get('strength') not in levels \
                or levels[task['strength']] != first_level:
            raise ValueError('hierarchy_policy does not match strength/first_level/four-level task')
        config['hierarchy_policy_name'] = dict(name=hierarchy_policy, first_level=first_level,
            role='T64 per-strength hierarchy (user decision 2026-09-29); solve_four_level default unchanged')
    if low_load:
        config['resource_policy'] = dict(policy, cgroup_owner='external verified systemd user service',
            torch_threads_actual=torch.get_num_threads())
    if throughput:
        config['execution_profile'] = dict(profile, torch_threads_actual=torch.get_num_threads(),
            numba_threads_env=os.environ.get('NUMBA_NUM_THREADS'), cgroup_cpu_max=_cgroup_value('cpu.max'),
            cgroup_memory_max=_cgroup_value('memory.max'),
            **({} if host_memory is None else dict(host_memory=host_memory)))
    if guard is not None:
        config['host_storage_policy'] = storage_policy(host_storage, start_memory_gate=start_memory_gate, memory_limits=memory_limits)
    dump(out/'config.json', config)
    for name in ('commits', 'history', 'promoted', 'inner_attempts', 'levels'):
        dump(out/(name+'.json'), [])
    # E98: the process-level margins kernel follows this run's profile -- row-SoA for v16, libm for every other
    # profile, so that a worker never inherits an earlier run's choice in the same process -- set after every
    # validation and before the first certificate (the solve). The Numba module is imported only when the profile
    # needs it or it is already loaded (the legacy route keeps its imports). The v16 records count this run only.
    certificate_start = None
    if profile is not None and (profile.get('soa_certificate') or profile.get('lean_certificate')):
        from jd_curl.three_d.compiled_enclosure import select_margins_execution, soa_certificate_record
        from jd_curl.three_d.feasibility import lean_certificate_record
        select_margins_execution('soa' if profile.get('soa_certificate') else 'libm')
        certificate_start = dict(soa=soa_certificate_record(), lean=lean_certificate_record())
    elif 'jd_curl.three_d.compiled_enclosure' in sys.modules:
        sys.modules['jd_curl.three_d.compiled_enclosure'].select_margins_execution('libm')
    stopper = None
    if practical_stop is not None:                      # H17: T_rem = the native solve deadline minus now
        from jd_curl.three_d.practical_stop import PracticalStop
        stopper = PracticalStop(practical_stop, lambda: deadline-perf_counter())
    progress('solve_started')
    try:
        with work_session(budget, on_work) if efficient_cpu_budget else nullcontext():
            if task['hierarchy'] == 'four':
                result = solve_four_level(grid=grid, fixed_image=arrays['fixed'], moving_image=arrays['moving'],
                    weights=weights if weights_levels is None else weights_levels, memory=memory, certificate_config=certificate, backend=backend,
                    **(dict(solver=solver) if inner_variant is not None else {}),
                    inner_configs=inner_configs, outer_configs=(outer,)*4,
                    observer=on_outer, inner_observer=on_inner, level_observer=on_level, budget=budget,
                    level_budgets=tuple(lambda t=t: perf_counter() >= t for t in level_deadlines),
                    **(dict(barrier_activation=.05) if local_barrier else {}),
                    **(dict(advance_stalled_levels=True) if advance_stalled_levels else {}),
                    **(dict(transfer_backtracking=True) if transfer_backtracking else {}),
                    **(dict(first_level=first_level) if first_level != 0 else {}),
                    **(dict(barrier_basis=barrier_basis) if barrier_basis != 'samples' else {}),
                    **(dict(curvature_weight=curvature_weight) if curvature_weight else {}),
                    **(dict(curvature_weight=curvature_weight_levels) if curvature_weight_levels is not None else {}),   # G2
                    **(dict(data_weight_policy=data_weight) if data_weight is not None else {}),
                    **(dict(native_start_scale=native_start_scale) if native_start_scale not in (None, 1.) else {}),   # T83
                    **(dict(native_transfer_alphas=native_transfer_alphas) if native_transfer_alphas is not None else {}),   # AP2
                    **(dict(transfer_smoothing=transfer_smoothing) if transfer_smoothing is not None else {}),   # A19
                    **(dict(native_practical_stop=stopper) if stopper is not None else {}),   # H17
                    **({} if outer_gate is None else dict(native_outer_gate=outer_gate)),     # FX1 F4
                    **({} if active_spec is None else dict(active_set=active_spec)),          # T89 (S30-G4)
                    **(dict(native_reference=native_ablation['value']) if native_ablation is not None
                       and native_ablation['switch'] == 'native_reference' else {}),           # T92 (D2-c/c')
                    **(dict(native_solver='lm') if native_ablation is not None
                       and native_ablation['switch'] == 'native_inner' else {}),               # T92 (D2-e)
                    **({} if pre_optimizer is None else dict(                                   # T88 (S30-G3)
                        preoptimizer=(lambda fixed_img, moving_img, native_grid: (first_use_dry['seed'], dict(policy='first-use-dry-synthetic-seed-v1')))   # FX1 F2'
                        if first_use_dry is not None else lambda fixed_img, moving_img, native_grid: __import__('jd_curl.three_d.preopt', fromlist=['preoptimize']).preoptimize(
                            fixed_img, moving_img, native_grid, preopt_config, device=device,
                            **({} if preopt_execution is None else dict(execution=preopt_execution))),   # S30 PRE speed
                        preoptimizer_taper=pre_optimizer.get('taper', 4))))
                native = result.native_result
                dump(out/'level_results.json', result.levels)
            else:
                problem = make_problem(grid=grid, fixed_image=arrays['fixed'], moving_image=arrays['moving'],
                    u_ref=arrays['initial_u'], weights=weights, memory=memory,
                    certificate_config=certificate, barrier_location='closed27_uniform',
                    **(dict(curvature_weight=curvature_weight) if curvature_weight else {}))
                result = native = solve_sequential(problem, restrict_free(arrays['initial_u'], grid),
                    inner_config=inner, config=outer, backend=backend, baseline_at_zero=True, budget=budget,
                    observer=lambda *args: on_outer(3, *args), inner_observer=lambda row: on_inner(3, row))
    except WorkBudgetExceeded as error:
        try:
            if not efficient_cpu_budget:
                _close_writer_after_failure(writer)
                raise
            # The interrupted solver has no complete result. Never manufacture a
            # level result or recover a promoted/uncommitted candidate from disk.
            interruption = dict(phase=error.phase, solver_return_available=False,
                counts_scope='last_formal_native_commit_only', success=False,
                formal_anchor=None if native_anchor is None else native_anchor.export_identity,
                cleanup='outside_work_deadline_inside_parent_watchdog')
            native = native_anchor
            result = SimpleNamespace(status='work_budget_interrupted', success=False,
                reason=str(error)+'; last formal native state only, not convergence', native_result=native)
            dump_index(out/'work_interruption.json', interruption)   # names a checkpoint: after it
            progress('budget_cleanup_started', phase=error.phase)
        except BaseException:
            _close_writer_after_failure(writer)   # review E76 round 2: failures inside the handler
            raise
    except BaseException:
        # Review E76: any other failure (incl. SIGTERM -> KeyboardInterrupt) still writes every
        # accepted checkpoint and its index, in order, before the error propagates.
        _close_writer_after_failure(writer)
        if writer is not None:
            for handle in streams.values():    # v9 only: the append-only logs are complete up to here
                try:
                    handle.close()
                except BaseException:
                    pass
        raise
    if first_use_dry is not None:
        # FX1 F2': the dry pass ends here (no export, labels or summary); its backend's verified keys are its result
        from jd_curl.three_d.backends.first_use_registry import harvest
        _close_writer_after_failure(writer)
        for handle in streams.values():
            handle.close()
        streams.clear()
        backend.device.synchronize()
        first_use_dry['registry'].update(harvest(backend))
        first_use_dry['result'] = dict(status=result.status, native_commits=sum(c['native'] for c in commits),
                                       seconds=perf_counter()-started)
        return result
    fast_export = bool(throughput and profile.get('fast_export'))      # FX1 F3 (v35)
    fast_record = None

    def stage_final(**arrays):
        """FX1 F3: final_native.npz by the same np.savez call, into a private tmpfs directory; hashed there."""
        nonlocal io_seconds
        tick = perf_counter()
        if guard_on_writer and guard is not None and writer is not None:
            writer.raise_if_failed()      # v39 (review): no final export after a failed check on the writer thread
        if guard is not None:
            pending = sum(a.nbytes for a in arrays.values())+65536*len(arrays)+1024**2
            if writer is not None:
                pending += writer.queued_bytes
            tick_guard = perf_counter(); guard.check('before_checkpoint', pending_bytes=pending, task=task['name'])
            guard_seconds[0] += perf_counter()-tick_guard
        stage = Path(profile.get('fast_export_dir', '/dev/shm'))/f'seq_vp_3d_export_{os.getpid()}_{out.parent.name}'
        stage.mkdir(parents=True, exist_ok=False)
        temp = stage/'final_native.partial.npz'
        np.savez(temp, **arrays); temp.replace(stage/'final_native.npz')
        row = dict(path='final_native.npz', sha256=file_sha256(stage/'final_native.npz'))
        io_seconds += perf_counter()-tick
        return row, stage

    def after_e2e(stage, final):
        """FX1 F3 (after e2e): the copy into the run directory (digest checked, then the stage is removed), the queued
        checkpoints in order, the full JSON logs -- the same files as v34, written later."""
        nonlocal io_seconds
        tick = perf_counter(); copy_seconds = None
        if stage is not None:
            tick_copy = perf_counter(); temp = out/'final_native.npz.partial'
            shutil.copyfile(stage/'final_native.npz', temp)
            if file_sha256(temp) != final['sha256']:
                raise RuntimeError('FX1 F3: the copy of final_native.npz differs from the staged file')
            temp.replace(out/'final_native.npz'); shutil.rmtree(stage)
            copy_seconds = perf_counter()-tick_copy
        if writer is not None:
            tick_flush = perf_counter()
            try:
                writer.flush()
            except BaseException:
                _close_writer_after_failure(writer)
                raise
            if writer.deferred_hash:
                from scripts.async_commit_writer import resolve_pending
                for value in (commits, events, saved_maps, interruption,
                              None if native_anchor is None else native_anchor.export_identity):
                    resolve_pending(value)
            io_seconds += perf_counter()-tick_flush
        for handle in streams.values():
            handle.close()
        streams.clear()
        dump(out/'history.json', history); dump(out/'inner_attempts.json', attempts)
        dump(out/'promoted.json', promoted)
        return dict(policy='fast-final-export-v1', staging=None if stage is None else str(stage.parent),
                    copy_seconds=copy_seconds, after_e2e_seconds=perf_counter()-tick,
                    role='e2e ends when final_native.npz is written and hashed on tmpfs; the copy into the run directory '
                         '(digest checked), the queued checkpoints and the full JSON logs follow')
    if writer is not None and not fast_export:
        # Review E76 round 2: flush before the device synchronize (a deferred CUDA error there
        # must not cost queued checkpoints).
        tick = perf_counter()
        try:
            writer.flush()
        except BaseException:
            _close_writer_after_failure(writer)
            raise
        if writer.deferred_hash:
            # E102: every checkpoint is written and hashed; from here on the records hold plain hex strings.
            from scripts.async_commit_writer import resolve_pending
            for value in (commits, events, saved_maps, interruption,
                          None if native_anchor is None else native_anchor.export_identity):
                resolve_pending(value)
        io_seconds += perf_counter()-tick
    if fast_export:
        try:
            backend.device.synchronize()
        except BaseException:
            _close_writer_after_failure(writer)      # FX1 F3: the queued checkpoints are still written, in order
            raise
    else:
        backend.device.synchronize()
    solver_wall = perf_counter()-solve_start
    if throughput and not fast_export:
        # Same full-array schema as the default path, written once, not per event.
        for handle in streams.values():
            handle.close()
        streams.clear()
        dump(out/'history.json', history); dump(out/'inner_attempts.json', attempts)
        dump(out/'promoted.json', promoted)
    state = None if native is None else native.final_state
    native_commits = sum(c['native'] for c in commits)
    role = ('no_native_result' if state is None else 'formal_native_outer_commit' if native_commits else
            'certified_native_initializer_NOT_native_outer_commit')
    e2e_seconds = final_export_seconds = None
    staged = None
    if state is not None:
        tick = perf_counter()
        if fast_export:
            final, staged = stage_final(u=state.u, warped=state.warped)      # FX1 F3: tmpfs
        else:
            final = save('final_native.npz', u=state.u, warped=state.warped)
        final_export_seconds = perf_counter()-tick
        e2e_seconds = perf_counter()-started      # S30-G1: task clock start -> final_native.npz on disk (FX1 F3: on tmpfs)
    if fast_export:
        fast_record = after_e2e(staged, None if state is None else final)
    if state is not None:
        dump(out/'final_native.json', dict(final, role=role, certificate=asdict(state.certificate),
            **(dict(source=interruption['formal_anchor']) if interruption is not None else {})))
    # Preserve solver conclusions BEFORE label evaluation, in case the parent
    # kills an overlong final atomic operation. Completion remains separate.
    if stopper is not None:                             # H17: the stop's time and trigger, for audits and time curves
        hit = stopper.triggered_at
        practical = dict(stopper.record(), triggered=hit is not None, native_commits_seen=len(stopper.commits),
                         stopped_by_rule=result.status == 'native_practical_stop',
                         global_seconds=None if hit is None else hit[0]-started, trigger=None if hit is None else hit[1])
    summary = dict(task=task, status=result.status, success=result.success, reason=result.reason,
        native_status=None if native is None else native.status, metrics_role=role,
        ratio=None if native is None else native.mse_ratio,
        native_mse=None if native is None else native.native_mse,
        final_mse=None if native is None else native.final_mse,
        GF0=None if native is None else native.GF_initial, GF=None if native is None else native.GF_final,
        GF_threshold=None if native is None else native.GF_threshold,
        GD=None if native is None else native.GD_final, GE=None if native is None else native.GE_final,
        certificate=None if state is None else asdict(state.certificate),
        energy=None if state is None else (asdict(state.energy) if native_exact_row is None else _final_energy(state, native_exact_row)),
        **({} if native_exact_row is None or state is None or getattr(state, 'eval_id', None) == native_exact_row.get('candidate_eval_id')
           or (state.energy.proximity, state.energy.jd, state.energy.curl) == (0., 0., 0.) else dict(energy_soft_terms_unverified=True)),
        **({} if native_exact_row is None else dict(energy_source='E-HS1b (T90 screening on): float64 terms of the last native '
                                                   'formal commit (its history row) -- all six when the exported state is that candidate '
                                                   '(no refresh after it), else image/barrier/curvature with the refreshed state\'s exact '
                                                   'zero proximity/jd/curl')),
        counts=None if native is None else native.counts, native_commits=native_commits,
        all_level_commits=len(commits), attempted_inner_candidates=len(attempts),
        solve_call_wall_seconds=solver_wall, stored_array_io_seconds=io_seconds,
        backend_timings=backend.timings, evaluation_counts=backend.evaluation_counts,
        reuse_counts=backend.reuse_counts, comparison_counts=backend.comparison_counts,
        **(dict(costs=costs,cpu_comparison_cost=cpu_cost) if efficient_cpu_budget else {}),
        backend_timing_scope='inclusive nested timers; do NOT sum as total wall',
        peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        device_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        e2e_seconds=e2e_seconds, final_export_seconds=final_export_seconds, storage_guard_seconds=guard_seconds[0],
        **(dict(storage_guard_writer_seconds=writer_guard_seconds[0], storage_guard_policy='writer-thread-before-each-checkpoint-v1')
           if guard_on_writer and guard is not None else {}),                                        # v39
        e2e_scope='S30-G1: task clock start to final_native.npz written; labels, smoothness and audits excluded',
        **({} if resident is None else dict(resident=resident)),
        elapsed_before_labels=perf_counter()-started, independent_audit_pending=True)
    if efficient_cpu_budget:
        summary.update(execution_policy=CPU_BUDGET_POLICY, interruption=interruption)
    if batch is not None:
        summary['fusion_counts'] = backend.fusion_counts
    if getattr(backend, 'boundary_reuse', False):
        # E89 (P05, review): reuse counts and the first-use bitwise verification per level grid.
        summary['boundary_reuse'] = backend.boundary_reuse_record
    if getattr(backend, 'screen_prefilter', False):
        # E93: prefilter calls, tiles skipped, first-use check per level grid.
        summary['screen_prefilter'] = backend.screen_prefilter_record
    if getattr(backend, 'fused_node_jd', False):
        # E94: fused node JD calls, first-use bitwise check per level grid, fallbacks.
        summary['fused_node_jd'] = backend.fused_node_jd_record
    for name in ('step_extrema', 'early_warped', 'device_candidate', 'barrier_skip', 'laplacian_memo',
                 'tuned_launch', 'fused_spline', 'estimator_overlap', 'top4_select', 'outer_reuse', 'fused_laplacian',
                 'metric_prestart', 'device_scalars', 'fused_shift', 'direction_prefetch', 'fp32_poisson',
                 'fused_recheck', 'device_forcing', 'active_set_skip', 'fp32_screen', 'early_trial_hash'):
        if getattr(backend, name, False):                                  # E97, E98, E99, E100, E101, E103, E104, E105 (record only)
            summary[name] = getattr(backend, name+'_record')
    if getattr(backend, 'ggn_reuse', False):
        summary['ggn_reuse'] = backend.ggn_reuse_record                          # GG2 (record only)
    if native_ablation is not None:
        summary['native_ablation'] = dict(native_ablation)                        # T92 (D2)
    if getattr(backend, 'gpu_edge_hull', False):                          # E101 (T84; plan D unless v20, E102)
        summary['gpu_certificate'] = backend.gpu_certificate_record
    if getattr(backend, 'lean_certificate', False):                          # E98: this run's counts
        from jd_curl.three_d.feasibility import lean_certificate_record
        summary['lean_certificate'] = lean_certificate_record(since=(certificate_start or {}).get('lean'))
    if throughput and profile.get('soa_certificate'):
        from jd_curl.three_d.compiled_enclosure import soa_certificate_record
        summary['soa_certificate'] = soa_certificate_record(since=certificate_start['soa'])
    if getattr(backend, 'root_screen', False):
        from jd_curl.three_d.solvers.jd_step import root_screen_record
        summary['root_screen'] = root_screen_record()
    if getattr(backend, 'fused_kernels', False):
        summary['fused_kernel_models'] = dict(backend.fused_status)
        summary['inner_gradient_target_checks'] = dict(cpu=backend.cpu_gradient_checks,
                                                       device=getattr(backend, 'device_gradient_targets', 0),
                                                       **(dict(device_above=backend.device_gradient_above,
                                                               margin=backend.cpu_gradient_margin,
                                                               monitor=dict(backend.gradient_margin_monitor))
                                                          if getattr(backend, 'cpu_gradient_margin', None) else {}))
    if writer is not None:
        summary['async_checkpoint_writes'] = writer.stats()
        summary['stored_array_io_scope'] = ('E76 v9: main-thread serialize+hash and the final flush; '
                                            'the background write time is async_checkpoint_writes.writer_seconds')
        if writer.deferred_hash:
            summary['stored_array_io_scope'] = ('E102 v20: main-thread hand-over (copies of writable arrays only) and '
                                                'the final flush; serialize+hash+write run on the writer thread, see '
                                                'async_checkpoint_writes.deferred_hash and writer_seconds')
    if stopper is not None:
        summary['practical_stop'] = practical
    if first_use_seeded is not None:
        summary['first_use_prewarm'] = first_use_seeded                   # FX1 F2'
    if outer_gate is not None:
        summary['outer_gate'] = outer_gate.record()                       # FX1 F4
    if fast_record is not None:
        summary['fast_export'] = fast_record                              # FX1 F3
        summary['e2e_scope'] = ('FX1 F3 (v35): task clock start to final_native.npz written and hashed on tmpfs; the copy '
                                'into the run directory, the queued checkpoints and the full JSON logs follow (fast_export)')
    if momentum_first_levels is not None:
        # E111 T85 (review S4): per level, first steps with a momentum record, uses, non-descending increments, first
        # momentum trials refused by the certificate, accepted momentum steps and fallbacks to the Poisson direction
        from jd_curl.three_d.solvers.poisson_gradient import MOMENTUM_POLICY, momentum_counts
        summary['momentum'] = dict(policy=MOMENTUM_POLICY, by_level=momentum_counts(attempts))
    dump(out/'summary.json', summary)
    progress('native_label_evaluation_started')
    tick = perf_counter()
    label_sets = manifest.get('label_sets')         # D1: explicit label sets of a non-OASIS pair set; None = seg4/seg35
    labels = None if state is None else e11.label_evaluation(arrays, grid, state.u, label_sets)
    dump(out/'labels.json', dict(role=role, primary=manifest.get('label_primary', 'seg4'),
        audit_only=manifest.get('label_audit_only', 'seg35'), labels_not_in_loss=True,
        baseline=manifest['baseline_label_metrics'], final=labels, seconds=perf_counter()-tick))
    # E78: deformation smoothness of the final native state (diagnostic only, never a rule).
    from jd_curl.three_d.smoothness import smoothness_report
    tick = perf_counter()
    smoothness = None if state is None else smoothness_report(state.u, grid, arrays['fixed_'+manifest.get('brain_mask_set', 'seg4')] > 0)
    dump(out/'smoothness.json', dict(role=role, final=smoothness, seconds=perf_counter()-tick,
        scope='cell-centre J of the final native state; SDlogJ (Learn2Reg convention); not an acceptance or stopping rule'))
    progress('worker_completed', status=result.status, formal_commits=len(commits), native_commits=native_commits)
    if writer is not None:
        writer.close()
    dump(out/'worker_completion.json', dict(complete=True, task_total_seconds=perf_counter()-started,
        native_commits=native_commits, metrics_role=role, e2e_seconds=e2e_seconds, independent_audit_pending=True))
    return result


def prepare(out, *, low_load=False, host_storage=None, start_memory_gate=True, remaining_from=None, memory_limits=True, weak_four_30min=False, advance_stalled_levels=False, transfer_backtracking=False, local_barrier_hour=False, efficient_cpu_budget=False, gpu_near_boundary=False, reuse_reference=False, device_batch_size=None, selective_kernels=False, compiled_certificate=False, hoisted_certificate=False, certificate_tile_size=None, short_600=False):
    if not isinstance(compiled_certificate, bool):
        raise ValueError("compiled_certificate must be boolean")
    if not isinstance(hoisted_certificate, bool) or (hoisted_certificate and not compiled_certificate):
        raise ValueError("hoisted_certificate requires boolean and compiled_certificate")
    if compiled_certificate:
        from jd_curl.three_d.compiled_enclosure import runtime_info, HOISTED_KERNEL, LEGACY_KERNEL
        certificate_runtime = runtime_info(HOISTED_KERNEL if hoisted_certificate else LEGACY_KERNEL)
    batch = device_batch_config(device_batch_size, reuse_reference)
    if not isinstance(selective_kernels, bool) or (selective_kernels and batch is None):
        raise ValueError("selective_kernels requires boolean and explicit device_batch_size")
    if not isinstance(reuse_reference, bool) or (reuse_reference and not gpu_near_boundary):
        raise ValueError('reuse_reference requires explicit gpu_near_boundary')
    if not isinstance(gpu_near_boundary, bool) or (gpu_near_boundary and not efficient_cpu_budget):
        raise ValueError('gpu_near_boundary requires explicit efficient_cpu_budget')
    if not isinstance(efficient_cpu_budget, bool) or (efficient_cpu_budget and not local_barrier_hour):
        raise ValueError('efficient_cpu_budget requires explicit local_barrier_hour')
    if not isinstance(low_load, bool):
        raise ValueError('low_load must be an explicit boolean')
    policy = resource_policy(memory_limits=memory_limits, certificate_tile_size=certificate_tile_size)
    if certificate_tile_size is not None and not compiled_certificate:
        raise ValueError('certificate_tile_size requires compiled_certificate')
    if not memory_limits and (not low_load or host_storage is None):
        raise ValueError('disabling memory limits requires low_load and host storage protection')
    if not start_memory_gate and host_storage is None:
        raise ValueError('disabling the start gate requires explicit host storage protection')
    if host_storage is not None:
        storage_policy(host_storage, start_memory_gate=start_memory_gate, memory_limits=memory_limits)
    if not isinstance(local_barrier_hour, bool):
        raise ValueError('local_barrier_hour must be boolean')
    if local_barrier_hour:
        advance_stalled_levels = transfer_backtracking = True
    if not isinstance(advance_stalled_levels, bool) or (advance_stalled_levels and not (weak_four_30min or local_barrier_hour)):
        raise ValueError('advance_stalled_levels requires explicit weak_four_30min mode and a boolean')
    if not isinstance(transfer_backtracking, bool) or (transfer_backtracking and not (weak_four_30min or local_barrier_hour)):
        raise ValueError('transfer_backtracking requires explicit weak_four_30min mode and a boolean')
    selected = tasks(skip_first=remaining_from is not None, weak_four_30min=weak_four_30min, local_barrier_hour=local_barrier_hour, short_600=short_600)
    continuation = None if remaining_from is None else completed_prefix(remaining_from)
    if host_storage is not None:
        if not low_load:
            raise ValueError('host storage guard requires low_load')
        StorageGuard(out, host_storage).check('prepare_before_copy')
    out.mkdir(parents=True, exist_ok=False)
    data = out/'inputs'/PAIR; data.mkdir(parents=True)
    hashes = {}
    for name in ('pair.npz', 'manifest.json'):
        shutil.copy2(SOURCE/name, data/name)
        assert file_sha256(SOURCE/name) == file_sha256(data/name)
        hashes[str((data/name).relative_to(out))] = file_sha256(data/name)
    manifest = e11.read(data/'manifest.json')
    assert manifest['shape_zyx'] == [192,224,160] and manifest['spacing_xyz'] == [1.,1.,1.]
    assert manifest['unit']=='mm' and manifest['pair']==PAIR
    assert manifest['pair_sha256']==file_sha256(data/'pair.npz')
    if continuation is not None:
        dump(out/'continuation.json', continuation)
        hashes['continuation.json'] = file_sha256(out/'continuation.json')
    dump(out/'schedule.json', dict(tasks=selected, serial=True, device='cuda', inner_steps=INNER_STEPS,
        outer_steps=OUTER_STEPS, cumulative_shares=CUMULATIVE_SHARES, export_reserve_seconds=EXPORT_RESERVE,
        initialization='original_zero_no_resume', created_utc=e11.now(),
        **(dict(local_barrier_hour=True) if local_barrier_hour else {}),
        **(dict(short_600=True) if short_600 else {}),
        **(dict(certificate_tile_size=certificate_tile_size) if certificate_tile_size is not None else {}),
        **(dict(execution_policy=CPU_BUDGET_POLICY) if efficient_cpu_budget else {}),
        **(dict(gpu_near_boundary=True) if gpu_near_boundary else {}),
        **(dict(reuse_reference=True) if reuse_reference else {}),
        **(dict(device_batch=asdict(batch)) if batch is not None else {}),
        **(dict(selective_kernels=True) if selective_kernels else {}),
        **(dict(compiled_certificate=True, certificate_runtime=certificate_runtime) if compiled_certificate else {}),
        **(dict(hoisted_certificate=True) if hoisted_certificate else {}),
        **(dict(weak_four_30min=True) if weak_four_30min else {}),
        **(dict(advance_stalled_levels=True) if advance_stalled_levels else {}),
        **(dict(transfer_backtracking=True) if transfer_backtracking else {}),
        **(dict(resource_policy=policy) if low_load else {}),
        **(dict(host_storage_policy=storage_policy(host_storage, start_memory_gate=start_memory_gate, memory_limits=memory_limits)) if host_storage is not None else {}),
        **(dict(continuation='continuation.json') if continuation is not None else {})))
    hashes['schedule.json'] = file_sha256(out/'schedule.json')
    dump(out/'input_sha256.json', hashes)
    files = [p for d in ('jd_curl','scripts','tests','configs','docs') for p in (ROOT/d).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]
    files += [ROOT/'README.md', ROOT/'pyproject.toml']
    hashes = {}
    for p in files:
        target = out/'snapshot'/p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target); digest = file_sha256(p)
        assert file_sha256(target)==digest
        hashes[str(p.relative_to(ROOT))] = digest
    dump(out/'source_sha256.json', hashes)
    dump(out/'status.json', dict(status='prepared_not_started', scheduled=len(selected), completed=0, utc=e11.now()))


def verify(out):
    import jd_curl
    assert ROOT == (out/'snapshot').resolve() and Path(jd_curl.__file__).resolve().is_relative_to(ROOT)
    schedule = e11.read(out/'schedule.json')
    assert schedule['tasks'] == tasks(skip_first='continuation' in schedule, weak_four_30min=schedule.get('weak_four_30min', False), local_barrier_hour=schedule.get('local_barrier_hour', False), short_600=schedule.get('short_600', False)) and schedule['device']=='cuda' and schedule['serial'] is True
    if 'short_600' in schedule:
        assert schedule['short_600'] is True
    if 'certificate_tile_size' in schedule:
        assert schedule.get('compiled_certificate') is True
        resource_policy(certificate_tile_size=schedule['certificate_tile_size'])
    if schedule.get('local_barrier_hour'):
        assert schedule.get('advance_stalled_levels') is True and schedule.get('transfer_backtracking') is True
    if 'hoisted_certificate' in schedule:
        assert schedule['hoisted_certificate'] is True and schedule.get('compiled_certificate') is True
    if 'compiled_certificate' in schedule:
        assert schedule['compiled_certificate'] is True
        from jd_curl.three_d.compiled_enclosure import runtime_info, HOISTED_KERNEL, LEGACY_KERNEL
        assert schedule['certificate_runtime'] == runtime_info(HOISTED_KERNEL if schedule.get('hoisted_certificate') else LEGACY_KERNEL)
    if 'selective_kernels' in schedule:
        assert schedule['selective_kernels'] is True and 'device_batch' in schedule
    if 'device_batch' in schedule:
        batch = schedule['device_batch']
        expected = device_batch_config(batch['tile_size'], schedule.get('reuse_reference'))
        assert batch == asdict(expected) and type(batch['screen_batch_tiles']) is int
    if 'reuse_reference' in schedule:
        assert schedule['reuse_reference'] is True and schedule.get('gpu_near_boundary') is True
    if 'gpu_near_boundary' in schedule:
        assert schedule['gpu_near_boundary'] is True and schedule.get('execution_policy') == CPU_BUDGET_POLICY
    if 'execution_policy' in schedule:
        assert schedule['execution_policy'] == CPU_BUDGET_POLICY and schedule.get('local_barrier_hour') is True
    if 'transfer_backtracking' in schedule:
        assert schedule['transfer_backtracking'] is True and (schedule.get('weak_four_30min') is True or schedule.get('local_barrier_hour') is True)
    if 'advance_stalled_levels' in schedule:
        assert schedule['advance_stalled_levels'] is True and (schedule.get('weak_four_30min') is True or schedule.get('local_barrier_hour') is True)
    if 'continuation' in schedule:
        assert schedule['continuation'] == 'continuation.json'
        continuation = e11.read(out/'continuation.json')
        assert continuation['pending_original_indices'] == list(range(1, 8))
        assert continuation['completed_record']['name'] == tasks()[0]['name']
    assert (schedule['inner_steps'],schedule['outer_steps'])==(INNER_STEPS,OUTER_STEPS)
    assert schedule['cumulative_shares']==list(CUMULATIVE_SHARES) and schedule['export_reserve_seconds']==EXPORT_RESERVE
    memory_limits = schedule.get('resource_policy', LOW_LOAD_POLICY)['cgroup_memory_max_bytes'] is not None
    if not memory_limits:
        assert 'host_storage_policy' in schedule
    if 'resource_policy' in schedule:
        assert schedule['resource_policy']==resource_policy(memory_limits=memory_limits, certificate_tile_size=schedule.get('certificate_tile_size'))
    if 'host_storage_policy' in schedule:
        assert 'resource_policy' in schedule
        policy = schedule['host_storage_policy']
        assert policy==storage_policy(policy['host_volume'],
            start_memory_gate=policy['host_start_available_bytes']==HOST_START_AVAILABLE_BYTES, memory_limits=memory_limits)
    for name, digest in e11.read(out/'source_sha256.json').items():
        assert file_sha256(ROOT/name)==digest, name
    for name, digest in e11.read(out/'input_sha256.json').items():
        assert file_sha256(out/name)==digest, name
    return schedule['tasks']


def verify_resource_limits(*, memory_limits=True):
    """Fail closed unless this process is protected by the registered limits."""
    relative = Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
    cgroup = Path('/sys/fs/cgroup')/relative
    expected = {'memory.high': str(LOW_LOAD_POLICY['cgroup_memory_high_bytes']),
                'memory.max': str(LOW_LOAD_POLICY['cgroup_memory_max_bytes']),
                'memory.swap.max': '0', 'cpu.max': '100000 100000'}
    resource_policy(memory_limits=memory_limits)
    if not memory_limits:
        expected.update({name: 'max' for name in ('memory.high', 'memory.max', 'memory.swap.max')})
    actual = {name: (cgroup/name).read_text().strip() for name in expected}
    for name, value in expected.items():
        if actual[name] != value:
            raise RuntimeError('low-load cgroup limit mismatch: '+name)
    return dict(cgroup=str(cgroup), limits=actual, verified_utc=e11.now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--low-load', action='store_true', help='prepare a reduced-load run; requires external cgroup limits')
    parser.add_argument('--host-storage', type=Path,
        help='prepare with a verified host volume (e.g. /mnt/c for this WSL VHD); requires --low-load')
    parser.add_argument('--no-start-memory-gate', action='store_true', help='prepare without the extra 4GiB gate; retain the 1GiB floor')
    parser.add_argument('--no-memory-limits', action='store_true', help='remove task RAM/swap/CUDA limits and host RAM gates; retain disk protection')
    parser.add_argument('--transfer-backtracking', action='store_true', help='prepare weak-four-30min with certified transfer displacement scaling')
    parser.add_argument('--advance-stalled-levels', action='store_true', help='prepare weak-four-30min with certified coarse search-stall transfer')
    parser.add_argument('--local-barrier-hour', action='store_true', help='prepare E26 C2 barrier + JD predictor weak four-level 3600s experiment')
    parser.add_argument('--efficient-cpu-budget', action='store_true', help='E27 fused CPU comparison and cooperative deadline/formal export; requires --local-barrier-hour')
    parser.add_argument('--short-600', action='store_true', help='shorten explicit local-barrier-hour task to 600s, including export reserve')
    parser.add_argument('--certificate-tile-size', type=int, choices=(2048,8192), help='explicit CPU certificate tile; independent of GPU tile; requires compiled certificate')
    parser.add_argument('--hoisted-certificate', action='store_true',
        help='E33 monotone edge-hull hoisting; requires --compiled-certificate')
    parser.add_argument('--compiled-certificate', action='store_true',
        help='E32 optional single-thread compiled outward-rounded edge-hull preproof; requires numba')
    parser.add_argument('--selective-kernels', action='store_true',
        help='E31 compute only requested geometry/barrier quantities; requires --device-batch-size')
    parser.add_argument('--device-batch-size', type=int, choices=(8192,32768,65536),
        help='E30 independent GPU tiles, batched witness and fusion; requires --reuse-reference')
    parser.add_argument('--reuse-reference', action='store_true', help='E29 same-map proof/sample/device constant reuse; requires --gpu-near-boundary')
    parser.add_argument('--gpu-near-boundary', action='store_true', help='E28 skip only certificate-near-boundary forced CPU comparison; requires --efficient-cpu-budget')
    parser.add_argument('--weak-four-30min', action='store_true', help='prepare one fresh weak four-level experiment with a 1800s total budget')
    parser.add_argument('--remaining-from', type=Path, help='prepare only the seven unstarted tasks after a verified completed first task')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--campaign', action='store_true')
    action.add_argument('--worker', type=int, choices=range(8))
    args = parser.parse_args(); out = args.output.resolve()
    if (args.short_600 or args.certificate_tile_size is not None or args.low_load or args.host_storage is not None or args.no_start_memory_gate or args.no_memory_limits or args.weak_four_30min or args.local_barrier_hour or args.efficient_cpu_budget or args.gpu_near_boundary or args.reuse_reference or args.device_batch_size is not None or args.selective_kernels or args.compiled_certificate or args.hoisted_certificate or args.advance_stalled_levels or args.transfer_backtracking or args.remaining_from is not None) and not args.prepare:
        parser.error('--low-load/--host-storage are preparation options; campaign uses the frozen schedule policy')
    if args.prepare:
        try:
            return prepare(out, low_load=args.low_load, host_storage=args.host_storage,
                           start_memory_gate=not args.no_start_memory_gate, remaining_from=args.remaining_from, memory_limits=not args.no_memory_limits, weak_four_30min=args.weak_four_30min, advance_stalled_levels=args.advance_stalled_levels, transfer_backtracking=args.transfer_backtracking, local_barrier_hour=args.local_barrier_hour, efficient_cpu_budget=args.efficient_cpu_budget, gpu_near_boundary=args.gpu_near_boundary, reuse_reference=args.reuse_reference, device_batch_size=args.device_batch_size, selective_kernels=args.selective_kernels, compiled_certificate=args.compiled_certificate, hoisted_certificate=args.hoisted_certificate, certificate_tile_size=args.certificate_tile_size, short_600=args.short_600)
        except e11.CampaignResourceError as error:
            print('Resource preflight refused: '+str(error), file=sys.stderr)
            return 75
    schedule = verify(out)
    frozen = e11.read(out/'schedule.json')
    low_load = 'resource_policy' in frozen
    memory_limits = frozen.get('resource_policy', LOW_LOAD_POLICY)['cgroup_memory_max_bytes'] is not None
    host_storage = frozen.get('host_storage_policy', {}).get('host_volume')
    start_memory_gate = frozen.get('host_storage_policy', {}).get('host_start_available_bytes', HOST_START_AVAILABLE_BYTES)==HOST_START_AVAILABLE_BYTES
    if low_load:
        # Check effective kernel limits, not just a configuration claim. Workers
        # inherit the same service cgroup; running this CLI unprotected is an error.
        limits = verify_resource_limits() if memory_limits else verify_resource_limits(memory_limits=False)
        if args.campaign:
            dump(out/'resource_control.json', limits)
    if args.campaign:
        extra = {}
        if host_storage is not None:
            guard = StorageGuard(out, host_storage, cgroup=limits['cgroup'], monitor_host_memory=True, start_memory_gate=start_memory_gate, memory_limits=memory_limits)
            extra['resource_check'] = lambda task, proc: guard.check('supervisor',
                task=task['name'], worker_pid=None if proc is None else proc.pid)
        with patch.object(e11, 'INNER_STEPS', INNER_STEPS), patch.object(e11, 'OUTER_STEPS', OUTER_STEPS):
            return e11.campaign(out, worker_script=Path(__file__).resolve(),
                                task_watchdog=lambda t:t['wall_seconds'], **extra)
    if args.worker >= len(schedule):
        parser.error('worker index is outside this frozen schedule')
    task = schedule[args.worker]
    try:
        worker(task, out/task['name'], out/'inputs'/task['pair'],
               task_started=float(os.environ['JD_CURL_TASK_STARTED']), low_load=low_load,
               host_storage=host_storage, start_memory_gate=start_memory_gate, memory_limits=memory_limits,
               **(dict(advance_stalled_levels=True) if frozen.get('advance_stalled_levels') else {}),
               **(dict(transfer_backtracking=True) if frozen.get('transfer_backtracking') else {}),
               **(dict(efficient_cpu_budget=True) if frozen.get('execution_policy') == CPU_BUDGET_POLICY else {}),
               **(dict(gpu_near_boundary=True) if frozen.get('gpu_near_boundary') else {}),
               **(dict(reuse_reference=True) if frozen.get('reuse_reference') else {}),
               **(dict(device_batch_size=frozen['device_batch']['tile_size']) if 'device_batch' in frozen else {}),
               **(dict(selective_kernels=True) if frozen.get('selective_kernels') else {}),
               **(dict(compiled_certificate=True) if frozen.get('compiled_certificate') else {}),
               **(dict(hoisted_certificate=True) if frozen.get('hoisted_certificate') else {}),
               **(dict(certificate_tile_size=frozen['certificate_tile_size']) if 'certificate_tile_size' in frozen else {}),
               **(dict(local_barrier=True) if frozen.get('local_barrier_hour') else {}))
    except (e11.CampaignResourceError, OSError) as error:
        if host_storage is None or (isinstance(error, OSError) and error.errno not in (errno.ENOSPC, errno.EDQUOT)):
            raise
        folder = out/task['name'];folder.mkdir(exist_ok=True)
        dump(folder/'resource_failure.json', dict(reason=str(error), utc=e11.now(),
            automatic_restart=False, completion=False))
        return 75
    return 0


if __name__ == '__main__':
    sys.exit(main())
