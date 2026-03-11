# TODO

## Metric-Aware Passivity Treatment For Interface Matrices

- Goal: replace or complement the `sigma > 1 -> 1/sigma` hack with a principled passivity projection in the modal power metric.
- Context: interface matching compares two different modal bases (left and right), often non-orthonormal under the chosen overlap product.

### Definitions

- Let left mode count be `N_L`, right mode count be `N_R`.
- Build overlap (Gram) matrices using the same inner product used for interface matching:
- `G_LL in C^{N_L x N_L}` from left-left overlaps.
- `G_RR in C^{N_R x N_R}` from right-right overlaps.
- `G_LR in C^{N_L x N_R}` and `G_RL in C^{N_R x N_L}` for cross overlaps.

### Passivity Condition

- For transmission block `T_LR in C^{N_R x N_L}`, enforce:
- `T_LR^H G_RR T_LR <= G_LL` (Loewner order).
- Similarly for `T_RL`.

### Projection Recipe (Whiten -> Clip -> Unwhiten)

- Factor metrics: `G_LL = C_L^H C_L`, `G_RR = C_R^H C_R` (Cholesky/eigendecomp with regularization).
- Transform to Euclidean coordinates:
- `T_tilde = C_R T_LR C_L^{-1}`.
- Perform SVD in Euclidean space and clip singular values to `<= 1`.
- Transform back:
- `T_proj = C_R^{-1} T_tilde_proj C_L`.

### Practical Notes

- Keep original eigenmodes for propagation; do not replace physical modes by Gram-Schmidt vectors globally.
- Symmetrize and regularize `G` numerically: `G <- (G + G^H)/2`, floor tiny/negative eigenvalues.
- Works with unequal mode counts (`N_L != N_R`) naturally.
- Retain reciprocal-SVD hack as optional fallback/diagnostic when reciprocal branch-flip behavior is observed.

### Validation Plan

- Compare three methods on controlled cases and regression notebooks:
- reciprocal-SVD hack,
- Euclidean clipping,
- metric-aware projection.
- Metrics to track:
- max singular value (Euclidean),
- metric passivity residual `lambda_max(T^H G_out T - G_in)`,
- error against reference (e.g. Lumerical matrices when available).

## Conjugated Metric As Postprocessing

- Goal: keep core EME solve on a single stable matching metric (preferred: unconjugated / biorthogonal-consistent), then provide conjugated quantities for user-facing power interpretation.

### Inputs From Core Solve

- Core solve returns modal amplitudes in the chosen internal basis:
- `a_in` on input port, `a_out = T a_in` on output port.
- Core overlap/metric matrices used by the solver remain unchanged.

### Postprocessing Matrices

- Build conjugated overlap (power-like) Gram matrices on each port:
- `P_in[i,j] = <phi_i, phi_j>_conj`
- `P_out[i,j] = <psi_i, psi_j>_conj`
- Symmetrize numerically: `P <- (P + P^H)/2`.
- Regularize small negative eigenvalues from numerics if needed.

### Reported Power Quantities

- Input power-like scalar:
- `Pin = Re(a_in^H P_in a_in)`.
- Output power-like scalar:
- `Pout = Re(a_out^H P_out a_out)`.
- Optional per-mode "apparent power" (basis-dependent):
- `p_k = Re(conj(a_k) * (P a)_k)`.

### Optional Renormalization For Reporting

- To report modal amplitudes in a power-orthonormalized coordinate system:
- factor `P = C^H C`, define `a_hat = C a`.
- Then `||a_hat||_2^2` corresponds to conjugated power-like quantity.
- Use this only for reporting/plots, not for replacing the propagation eigenbasis.

### Consistency Checks

- Check mismatch between core passivity and conjugated reported powers:
- compare `Pout/Pin` against expected loss/passivity behavior.
- Flag cases where strong non-orthogonality makes per-mode power decomposition unreliable.
- In such cases, prefer reporting total port power over per-mode bars.

### API Idea

- Add a postprocess helper:
- `report_powers(a_in, a_out, modes_in, modes_out, metric='conjugated') -> dict`
- Returns totals, per-mode values, and conditioning diagnostics (`cond(P_in)`, `cond(P_out)`).

## PML-Aware Overlap Stabilization

- Problem: interface matching becomes unstable when many modes are used with PML, likely due to PML-localized / non-physical modes contaminating overlap matrices.

### Proposed Fixes

- Filter out PML-dominated modes before interface solving:
- compute PML energy fraction per mode,
- drop modes above threshold (e.g. `> 0.1` as starting point, tune empirically).
- Exclude (or down-weight) PML regions in overlap integrals:
- compute overlaps on a physical-window mask that removes PML strips.
- Keep an option to combine both approaches:
- `filter_pml_modes=True`, `exclude_pml_in_overlap=True`.

### Diagnostics To Add

- Condition numbers for overlap matrices before/after filtering.
- Number of modes kept/discarded per side.
- Sensitivity sweep over PML threshold.
- Convergence plots vs number of modes (with and without PML-aware handling).

### Validation

- Re-run rib-waveguide interface notebook for:
- no PML filtering/windowing,
- filtering only,
- overlap-window only,
- both.
- Compare against Lumerical reference and monitor blow-up behavior at high mode counts.

## Yee-Grid Co-Location For Overlaps

- Problem: overlap integrals are sensitive if `E` and `H` components are multiplied on mismatched staggered Yee positions.

### Proposed Implementation

- Interpolate fields to common spatial positions before overlap integration (pick one convention and keep it global):
- e.g. all required overlap terms on `Ez` locations or cell centers (`Hz` locations).
- Apply identical co-location pipeline for:
- mode normalization,
- mode-to-mode overlaps (`G_LL`, `G_RR`, `G_LR`, `G_RL`),
- any power reporting overlaps.
- Use area-weighted integration (`dx * dy`) on the co-located grid, including non-uniform mesh support.

### Validation

- Orthogonality sanity checks on symmetric structures before/after co-location.
- Sensitivity test by shifting geometry by one pixel (expect reduced overlap drift).
- Compare overlap matrices against Lumerical/Tidy3D references on same geometry.
