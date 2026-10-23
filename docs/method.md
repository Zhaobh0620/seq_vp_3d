# Method

This document describes the mathematical model and the algorithm implemented in this repository. Section numbers
in brackets, e.g. [T06], refer to the project's internal formula contract and are kept for traceability.

## 1. Problem setting

Let $I_f$ (fixed) and $I_m$ (moving) be 3D images on a rectangular domain $\Omega$ discretised by a regular grid
with spacings $h_x, h_y, h_z$ in millimetres. We seek a displacement $u$ that defines the map

$$\Phi(X) = X + u(X), \qquad I_w(X) = \tilde I_m(\Phi(X)), \qquad u|_{\partial\Omega} = 0 ,$$

i.e. the warped moving image $I_w$ is pulled back onto the fixed grid, and the displacement vanishes on the six
faces of the domain. $\tilde I_m$ is the cubic (not-a-knot) B-spline interpolant of the moving image; the image
gradient $b = \nabla \tilde I_m(\Phi(X))$ used below is the analytic spline gradient at the sampling point.

## 2. Discretisation

**Unknowns.** The displacement is stored at the grid nodes; only interior nodes are free, so the boundary condition
holds by construction. Node weights $q$ are the trapezoidal masses $q = \omega\, t_z \otimes t_y \otimes t_x$ with
$\omega = h_x h_y h_z$ (weight $1/2$ at the two end nodes of each axis).

**Geometry: trilinear (Q1) elements.** On every cell $K$ the displacement is interpolated trilinearly. The
deformation gradient, its determinant (the Jacobian determinant, JD) and the curl are

$$F = I + \nabla u, \qquad J = \det F = \sum_j F_{1j} A_{1j}, \qquad A = \operatorname{cof} F, \qquad
\operatorname{curl} u = (\partial_y u_3 - \partial_z u_2,\; \partial_z u_1 - \partial_x u_3,\; \partial_x u_2 - \partial_y u_1).$$

$J$ is a triquadratic polynomial on each cell (degree at most 2 in each reference variable), and for a direction
$V = \nabla v$ the determinant along a line is the exact cubic

$$\det(F + \tau V) = J + \tau\, A:V + \tau^2\, F:\operatorname{cof} V + \tau^3 \det V . \tag{2.1}$$

**Quadrature.** [T03] The JD and curl change penalties (Section 3) use the 3-point Gauss–Legendre rule per axis
(27 points per cell), which integrates the squared change of the Q1 Jacobian determinant and curl exactly. The barrier
(Section 3) uses the closed 27-point uniform rule $\{0, \tfrac12, 1\}^3$ with equal weights $|K|/27$. Image,
proximity and curvature terms use the nodal weights $q$. Nodal central-difference Jacobians are never used in the
objective or in the certificate.

## 3. Energies

**Data term.**
$$D(u) = \tfrac12 \sum_{\text{nodes}} q\, \big[\tilde I_m(X + u) - I_f\big]^2 .$$

**Barrier.** [T39] A $C^2$ barrier that is active only for small Jacobians: with activation $a = 0.05$,
$t = J/a$ and $r = 1 - t$,

$$b(J) = -r^2 \log t \quad (0 < J < a), \qquad b(J) = 0 \quad (J \ge a),$$

$$B(u) = \kappa \sum_K \frac{|K|}{27} \sum_{\xi \in \{0,\frac12,1\}^3} b\big(J_K(\xi)\big), \qquad \kappa = 0.002 .$$

$b$, $b'$ and $b''$ are continuous at $J = a$ and $b'' > 0$ on the active set. The barrier does not penalise
expansion or mild compression; positivity of $J$ on the whole cell is enforced separately by the certificate
(Section 5).

**Curvature term (optional).** [T76] $R(u) = \tfrac{\alpha}{2} \sum q\, |L u|^2$ with the 7-point Laplacian $L$
in physical spacings and zero boundary values. $\alpha = 0$ on the finest level of the default configuration.

**Outer objective.**
$$F(u) = D(u) + R(u) + B(u).$$

## 4. Algorithm

### 4.1 Self-referenced proximal outer iteration

[T06, T13] At an accepted (committed) state $u_i$ the reference is frozen, $u_{\mathrm{ref}} = u_i$, and the inner
problem minimises

$$E_i(u) = F(u) + \frac{\lambda}{2}\|u - u_i\|^2_{Q} + \frac{\mu}{2}\big\|J_G(u) - J_G(u_i)\big\|^2_{Q_G}
+ \frac{\nu}{2}\big\|C_G(u) - C_G(u_i)\big\|^2_{Q_G},$$

where $J_G$ and $C_G$ are the Jacobian determinant and curl at the 27 Gauss points of every cell. The three change
penalties and their first derivatives vanish at $u_i$, so $E_i(u_i) = F(u_i)$ and $\nabla E_i(u_i) = \nabla F(u_i)$.
Consequently any inner state with $E_i(u_{i+1}) \le E_i(u_i)$ satisfies

$$F(u_{i+1}) + \frac{\lambda}{2}\|u_{i+1} - u_i\|^2 + \frac{\mu}{2}\|\Delta J_G\|^2 + \frac{\nu}{2}\|\Delta C_G\|^2 \le F(u_i).$$

An inner result is committed only if (i) it passes the full-cell certificate (Section 5), and (ii) both $E_i$ and
$F$ decrease strictly by more than a resolution guard $s = 16\,\epsilon_{64} \max(E_{\text{base}}, E_{\text{trial}})$.
Energy differences are evaluated in a numerically stable form (pairwise residual differences, `log1p` for the
barrier, extended-precision accumulation on the CPU). After a commit the reference is refreshed to the committed
state; the reference is never smoothed, projected or reset.

The geometric terms are therefore *path* terms: they shape each step (a proximal metric in terms of volume change
and rotation) but are absent from the outer objective $F$. The iteration traces a regularisation path, and the
stopping time acts as a regularisation parameter (Section 6).

### 4.2 Inner solver: preconditioned gradient with a Poisson-type metric

[T10, T11, T59, T62, T69, T73] Each inner step uses the direction $p = -P^{-1} g$ with $g = \nabla E_i$ and a
symmetric positive definite metric $P$ that is frozen for the whole inner solve.

1. **Q1 Poisson metric.** Per displacement component,
   $M = \omega\,(\bar a\, I + b_P\, K_Q/\omega)$, where $K_Q$ is the consistent Q1 stiffness matrix with zero
   Dirichlet boundary, $b_P = (\mu + \nu)/2$ and $\bar a = \lambda + \bar\alpha$. The shift
   $\bar\alpha = \operatorname{mean}_{\text{interior}} |b|^2 / 3$ is the isotropic average of the Gauss–Newton image
   curvature. $M^{-1}$ is applied exactly with the type-I discrete sine transform.
2. **GGN equilibration.** $S = \operatorname{diag}(\sqrt{m}/\sqrt{d})$, where $d$ is the diagonal of the
   barrier-Gauss–Newton matrix of $E_i$ (image, proximity, JD/curl penalties and barrier curvature) and $m$ is the
   constant diagonal of $M$.
3. **Jacobian affine damping.** $R = \operatorname{diag}(r_a)$, with $r_a = \min\!\big(1, \max(J_a, 10^{-12})/J_{\mathrm{act}}\big)$
   and $J_{\mathrm{act}} = 0.25$. $J_a$ is the smallest estimated cell minimum of $J$ over the cells adjacent to node
   $a$. The cell minimum combines the closed-27 values, the exact minimum of $J$ along every cell edge (a quadratic in
   the edge parameter), and, for candidate cells, a bounded Newton search of the triquadratic $J$ on faces and in the
   interior. The estimates only scale the metric; they are not part of any acceptance decision.

The metric is $P^{-1} = \tilde S M^{-1} \tilde S$ with $\tilde S = S R$. Since $\tilde S$ is positive diagonal, $P$ is
SPD, $g^\top p < 0$ and the set of stationary points is unchanged. Near-folding cells receive steps proportional
to their distance from the feasibility boundary (affine scaling).

**Step size.**
- **BB2:** $\tau = s^\top y / (y^\top P^{-1} y)$, using the identity $P^{-1}y = p_{k-1} - p_k$ for the fixed metric,
  clipped to $[10^{-8}, 10]$. The history is cleared at every reference refresh.
- **Two caps:**
  - 0.9 times the largest step that keeps every sampling point inside the image domain;
  - **JD step prediction** [T39]: at recently violated locations, the first positive root of
    $\det(A + \tau V) - \delta$ (the cubic (2.1)) times 0.9.
- **Acceptance:** the trial must pass the certificate and a stable Armijo test,
  $\mathrm{ared} - s \ge -10^{-4}\,\tau\, g^\top p$ and $\mathrm{ared} - s > 0$.
- **Backtracking:** otherwise the step is halved, at most 25 times, with an early exit once the predicted decrease
  falls below the energy resolution.

Each outer iteration performs at most 5 inner steps. The inner solve also stops when the forcing target
$\eta_i\, G_i$ is met, with $\eta_i = \eta_0 \sqrt{\min(G_i, G_0)/G_0}$ and $\eta_0 = 0.01$.

A barrier-Gauss–Newton Levenberg–Marquardt inner solver (PCG with an equilibrated Poisson preconditioner) is also
implemented [T09, T12]; it is not used in the default configuration.

### 4.3 Multiresolution and initialisation

[T37, T88]
- **Pyramid.** The solver supports a four-level pyramid with shrink factors 8/4/2/1. Coarse images are Gaussian
  pre-smoothed and resampled once from the native image.
- **Level transfer.** The last committed coarse displacement is prolongated by Q1 interpolation. Because the grids
  are not nested, the prolongated field is re-certified on the whole fine grid. On failure the scaled fields
  $\alpha\, u$ with $\alpha = 0.9^k$, $k = 0, \dots, 6$, are tried in turn.

In the default configuration the coarse levels are replaced by a fast **pre-optimiser** that produces an
*uncertified* initial guess on the GPU in single precision:

$$P(u) = \tfrac12\sum \big(I_m(x+u) - I_f(x)\big)^2 + \frac{\mu_\ell}{2} \sum_{\text{cells}} \big[(J_c - 1)^2 + |\operatorname{curl}_c|^2\big]
+ \frac{\alpha_\ell}{2}\sum |Lu|^2 + \kappa_{\mathrm{pre}} \sum_{\text{cells}} \max(0, j_0 - J_c)^2 ,$$

- **Discretisation:** trilinear image sampling, with $J_c$ and $\operatorname{curl}_c$ evaluated at cell centres.
- **Optimiser:** an 8/4/2/1 pyramid with 50/50/100/60 iterations. The gradient and the displacement field are both
  Gaussian smoothed ($\sigma = 1$ voxel), followed by Adam-type normalised updates, each step capped at 0.5 voxel.
- **Weights:** $\mu_\ell = 1/1/0.01/0.003$, $\alpha_\ell = 10^{-3}/10^{-3}/10^{-3}/0$, $\kappa_{\mathrm{pre}} = 10$,
  $j_0 = 0.2$.

The output is tapered linearly to zero over a four-node band at the faces. It is then handed over like a level
transfer: full certification with the $0.9^k$ back-off. The certified field becomes the first reference of the
finest level; it is not counted as a committed outer iterate. All subsequent iterations are the certified FP64
algorithm of Sections 4.1–4.2.

## 5. Full-cell certificate

[T05, T30, T75, T84] A state is feasible if the displacement is zero on the faces, all sampling points lie in the
image domain, and on every cell

$$\inf_{\xi \in [0,1]^3} J_K(\xi) > \delta, \qquad \delta = 0 .$$

The certified object is the trilinear interpolant of the stored binary64 nodal displacements. A cell is processed in
up to three stages:

1. **Edge-hull pre-proof.** Every entry of $\nabla u$ on a cell is a convex combination of the four parallel edge
   differences along that axis. Each entry of $F$ is therefore enclosed in an interval hull, and $\det F$ is enclosed
   by a cofactor expansion in interval arithmetic. A positive lower end proves the cell.
2. **Floating-point Bernstein prescreen.** The Bernstein coefficients of $J$ (Stage 3) are computed in plain
   binary64 together with an a priori rounding-error bound $B$. The cell is proven if $\min_k \beta_k - B > \delta$.
   This stage can only prove a cell, never reject one.
3. **Interval Bernstein fallback with subdivision.** $J$ is evaluated in interval arithmetic at the 27 Lagrange points
   $\{0,\tfrac12,1\}^3$ and converted exactly to the 27 coefficients of its triquadratic Bernstein form.
   - By the convex-hull property, $J \ge \min_k \underline\beta_k$ on the box, so the box is proven if all lower ends
     are positive.
   - The cell is infeasible if the upper end at an evaluated point is $\le \delta$.
   - Otherwise the box is bisected along its longest axis by de Casteljau subdivision, lowest lower bound first, with
     at most depth 48 and 255 boxes per cell.
   - A cell that remains undecided is *uncertain*, and uncertain counts as failure.

All interval operations round outward to the neighbouring binary64 number (`nextafter`). Division by an interval
containing zero, overflow or an unsupported arithmetic mode yields "uncertain", never a proof. Stage 1 may run on the
GPU with a bitwise-identical operation sequence, checked against the CPU on first use; all decisions and records are
made on the CPU.

**Guarantee.** Every committed state, including the exported result, is certified: the Jacobian determinant of the
trilinear map is strictly positive on every closed cell, and the displacement is zero on the boundary. Global
invertibility of $\Phi$ on $\bar\Omega$ follows from these two facts by a degree argument; the formal proof is in
preparation and is not claimed here. Nodal or central-difference Jacobian statistics, including the Learn2Reg
"folding" count, measure different maps and are reported only as diagnostics.

## 6. Stopping and output

- **Convergence.** The run succeeds when the mass-scaled outer gradient satisfies
  $G_F = \|\nabla F\|/\sqrt{\omega} \le 10^{-8} + 10^{-6}\, G_F(0)$, with $G_F(0)$ evaluated at zero displacement on
  the finest grid.
- **Budget termination.** Otherwise the run ends when the wall-clock budget is exhausted. No new outer iteration
  is started if the remaining time is shorter than the shortest recent outer iteration.
- **Output.** The exported displacement is always the last committed, certified state.

Budget termination is reported as such and never as convergence. The MSE ratio is reported against the native
zero-displacement MSE, which is fixed for the whole run.

## 7. Precision and execution

- **Arithmetic.** The model is evaluated in binary64, on the CPU and on the GPU alike. The two exceptions are the
  FP32 pre-optimiser (Section 4.3), whose output is certified before use, and the FP32 energy screen below.
- **CPU and GPU.** The GPU (PyTorch and generated Triton kernels) computes directions, trial energies and the
  edge-hull pre-proof of the certificate. The pre-proof on the GPU follows the CPU operation sequence bit for bit,
  which is checked on first use. Certificate decisions, records and the stopping test are made on the CPU.
- **FP32 energy screen.** In the default execution profile, energy differences are first screened in single
  precision. A screened value is used only outside an error band calibrated against binary64; inside the band the
  comparison is recomputed in binary64. Decisions therefore agree with binary64 whenever the calibrated band bounds
  the single-precision error. The band was calibrated empirically and is not a proven bound. The certificate never
  uses single precision.
  - *Runtime check.* On each grid, the first three screened evaluations and every 50th one after them are also
    evaluated in binary64. Each term is compared against its own band: $64\,\varepsilon_{32}$ relative for the
    geometry terms and $4096\,\varepsilon_{32}$ for the barrier. The energy differences are compared against the
    comparison band. If any of them lies outside its band, that trial uses the binary64 values and screening is
    switched off for the rest of the run. Screened evaluations between checks are not verified.
  - *Observed violation.* In one evaluation on 50 held-out OASIS pairs, 1 of 234 binary64 checks (from 6611
    screened evaluations) found the barrier term outside its band, at 1.6 times the band. This happened on the first
    screened evaluation of the run, which is a checked one, so that run made all its decisions in binary64.
- **Execution profiles.** `--throughput vN` profiles change execution, not the mathematical model. Each profile was
  validated against the reference path, either bit for bit or, for precision options, by checking that all decisions
  agree.

## 8. Default configuration

| Item | Value |
|---|---|
| Strength profile | `weak` |
| Proximity weight $\lambda$ | $0.1 / L_*^2$, with $L_*$ the geometric mean of the field-of-view lengths (mm) |
| Geometry weights $\mu = \nu$ (finest level) | 0.03 |
| Barrier | $\kappa = 0.002$, activation $a = 0.05$, $\delta = 0$ |
| Curvature $\alpha$ (finest level) | 0 |
| Metric (finest level) | Q1 Poisson + image shift + GGN equilibration + Jacobian affine damping ($J_{\mathrm{act}} = 0.25$) |
| Inner steps per outer iteration | 5 |
| Initialisation | FP32 pre-optimiser (Section 4.3) + certified handover |
| Wall-clock budget | 90 s, with 1 s reserved for export |

The full command line is given in [usage.md](usage.md).
