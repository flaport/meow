# Stabilizing Interface S-Matrix Extraction (Pre-Implementation Plan)

This file answers the four questions before any notebook edits.

## 1) What "conditioning" entails

Conditioning is about how sensitive a linear solve is to tiny perturbations in the input matrices.

For your notebook, the key solve is:

- `A = O_LR + O_RL^T` (or `+ O_RL^H` in conjugated case)
- `T = 2 * A^{-1}` (implemented via `solve`)

If `cond(A)` is large, tiny numerical noise in overlaps causes large errors in `T`, and then in `R`.

What this entails in practice:

- Track `cond(A)` for each truncation size `K`.
- Inspect smallest singular values of `A`.
- Reject or downweight mode sets where `A` is near-singular.
- Expect low-order entries (like `T[0,1]`) to become unreliable when `cond(A)` spikes.

Tradeoff:

- Better stability, but may need fewer modes (or different mode subset).

## 2) What "regularized inverse" entails

Instead of solving `A x = b` directly, solve a stabilized variant.

Two common options:

- Tikhonov: solve `(A^H A + lambda^2 I)x = A^H b`
- Truncated SVD pseudoinverse: invert only singular values above threshold

For this notebook:

- Compute SVD `A = U diag(s) V^H`.
- Replace `1/s_i` by:
  - `1/s_i` if `s_i >= s_cut`
  - `0` (TSVD) or `s_i/(s_i^2 + lambda^2)` (Tikhonov) otherwise.
- Build `T` from this regularized inverse.

Tradeoff:

- Greatly reduces blow-ups and spurious cross-coupling.
- Introduces bias (slightly damped amplitudes), so needs a principled choice of `lambda`/`s_cut`.

## 3) What "best-K validation" entails

Do not assume "more modes is better" when conditioning collapses.

Instead:

- Sweep `K` from small to large.
- For each `K`, compute diagnostics:
  - `cond(A_K)`
  - same-medium sanity (`||T_LL - I||`, `||R_LL||`)
  - passivity violation before clipping (`sigma_max(S_K)`)
  - optional mismatch vs reference if available (`||abs(S_K)-abs(S_ref)||`)
- Pick `K*` that minimizes a score, e.g.:
  - low same-medium error
  - low passivity violation
  - acceptable conditioning
  - low reference mismatch (if reference exists)

Tradeoff:

- More robust than fixed `K=100`.
- Requires sweep runtime and a score definition.

## 4) Better way to compute R without subtracting near-equal overlaps?

Short answer: partially yes, but no free lunch.

Current formula:

- `R = 0.5 * (O_RL^T - O_LR) * T`

This is numerically fragile because it subtracts close matrices.

Better formulation (more stable algebraically):

- First solve for `T`.
- Then use continuity form: `R = 0.5 * (O_RL^T * T - 2I)` (equivalently from H-equation)

Why this can help:

- It avoids explicitly forming `O_RL^T - O_LR` first.
- Lets you reuse whichever product (`O_RL^T*T` or `O_LR*T`) is numerically better conditioned.

Important caveat:

- If `T` itself is corrupted by ill-conditioned `A`, `R` is still corrupted.
- So this is an incremental improvement, not a replacement for conditioning/regularization.

Other stronger alternatives:

- Solve a larger constrained linear system for `[R,T]` jointly from E/H continuity (least-squares with regularization).
- Use a flux-orthogonalized basis or biorthogonal left/right modes to improve overlap conditioning.

## Recommended order for implementation in notebook copy

1. Add diagnostics (`cond`, singular spectrum, same-medium checks vs `K`).
2. Add best-K selection logic.
3. Add regularized `T` solve (TSVD or Tikhonov).
4. Compute `R` via continuity-based formula (`0.5*(O_RL^T*T - 2I)` and cross-check equivalent form).
5. Compare raw vs regularized vs passive-clipped against reference.
