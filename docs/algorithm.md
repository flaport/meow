# Interface Algorithm

The interface algorithm mixes exact algebra with engineering intuition. The derivations
of the interface formulas are exact within the stated assumptions. The passivity and
truncation arguments are intentionally more heuristic: they are useful for reasoning
about numerical behavior, but they should not be read as formal proofs.

## Problem Setup

We consider a step discontinuity between two rib waveguides, labeled left (L)
and right (R), with propagation along `z`.

In a `z`-invariant waveguide, the transverse fields are expanded in eigenmodes:

$$
\mathbf{E}_t(x,y,z) =
\sum_p \left[a_p^{+} e^{-j \beta_p z} + a_p^{-} e^{+j \beta_p z}\right]
\mathbf{e}_p(x,y)
$$

$$
\mathbf{H}_t(x,y,z) =
\sum_p \left[a_p^{+} e^{-j \beta_p z} - a_p^{-} e^{+j \beta_p z}\right]
\mathbf{h}_p(x,y)
$$

The sign flip on `H` for backward-propagating modes follows from Maxwell's curl
equation.

At the interface `z = 0`, tangential field continuity gives:

$$
\sum_p (a_p^{+} + a_p^{-}) \mathbf{e}_p^L
=
\sum_q (b_q^{+} + b_q^{-}) \mathbf{e}_q^R
$$

$$
\sum_p (a_p^{+} - a_p^{-}) \mathbf{h}_p^L
=
\sum_q (b_q^{+} - b_q^{-}) \mathbf{h}_q^R
$$

## Inner Product

To convert the vector continuity equations into matrix equations, we project
onto test functions using the asymmetric modal overlap:

$$
\langle \mathbf{e}_a, \mathbf{h}_b \rangle
=
\frac{1}{2}\int_S (\mathbf{e}_a \times \mathbf{h}_b)\cdot\hat{z}\, dA
$$

This overlap is not symmetric across different waveguides.

Define:

- `O_LR[i,j] = <e_i^L, h_j^R>`
- `O_RL[i,j] = <e_i^R, h_j^L>`
- `G_LL[i,j] = <e_i^L, h_j^L>`
- `G_RR[i,j] = <e_i^R, h_j^R>`

For lossless waveguides with the unconjugated product:

- reciprocity within one waveguide gives `G_LL = G_LL^T` and `G_RR = G_RR^T`;
- orthogonal modes with distinct propagation constants have vanishing
  off-diagonal self-overlaps;
- in general `O_LR != O_RL^T`.

If modes are orthonormalized in this same inner product, then `G_LL = G_RR = I`.

## Why The Symmetric Product Fails For Reflection

The symmetric product is

$$
\langle \mathbf{e}_a, \mathbf{h}_b \rangle_{\mathrm{sym}}
=
\frac{1}{4}\int_S
\left[
(\mathbf{e}_a \times \mathbf{h}_b)
+
(\mathbf{e}_b \times \mathbf{h}_a)
\right]\cdot\hat{z}\, dA
$$

By construction it is commutative, so it forces `O_LR = O_RL^T` even for modes
of different waveguides. That makes the reflection term vanish algebraically.

This is the main reason the asymmetric product is used for interface
scattering.

## Interface Formulas

### Left Incidence

For left incidence, set `b^- = 0`. Project E continuity with `h_k^L` and H
continuity with `e_k^L`:

$$
G_{LL}(a^+ + a^-)=O_{RL}^T b^+
$$

$$
G_{LL}(a^+ - a^-)=O_{LR} b^+
$$

Adding:

$$
2 G_{LL} a^+ = (O_{LR} + O_{RL}^T) b^+
$$

$$
T_{LR} = 2 (O_{LR} + O_{RL}^T)^{-1} G_{LL}
$$

Subtracting:

$$
R_{LL} = \tfrac{1}{2} G_{LL}^{-1} (O_{RL}^T - O_{LR}) T_{LR}
$$

For orthonormal modes, `G_LL = I`, so:

$$
T_{LR} = 2 (O_{LR} + O_{RL}^T)^{-1}
$$

$$
R_{LL} = \tfrac{1}{2} (O_{RL}^T - O_{LR}) T_{LR}
$$

Two equivalent continuity forms are also useful:

$$
R_{LL} = O_{RL}^T T_{LR} - I = I - O_{LR} T_{LR}
$$

Averaging those two recovers the standard difference form:

$$
\tfrac{1}{2}
\left[
(O_{RL}^T T_{LR} - I) + (I - O_{LR} T_{LR})
\right]
=
\tfrac{1}{2}(O_{RL}^T - O_{LR}) T_{LR}
$$

In practice, the continuity forms are often numerically preferable because they
avoid explicitly subtracting two nearly equal overlap matrices before
multiplying by `T`.

### Right Incidence

For right incidence, set `a^+ = 0`:

$$
G_{RR}(b^+ + b^-) = O_{LR}^T a^-
$$

$$
G_{RR}(b^+ - b^-) = -O_{RL} a^-
$$

Hence:

$$
T_{RL} = 2 (O_{RL} + O_{LR}^T)^{-1} G_{RR}
$$

$$
R_{RR} = \tfrac{1}{2} G_{RR}^{-1} (O_{LR}^T - O_{RL}) T_{RL}
$$

And for orthonormal modes:

$$
R_{RR} = O_{LR}^T T_{RL} - I = I - O_{RL} T_{RL}
$$

### Full S-Matrix

Assemble the interface two-port as

$$
\begin{pmatrix}
a^- \\
b^+
\end{pmatrix}
=
S
\begin{pmatrix}
a^+ \\
b^-
\end{pmatrix}
$$

with block structure

$$
S=
\begin{pmatrix}
R_{LL} & T_{RL} \\
T_{LR} & R_{RR}
\end{pmatrix}
$$

## The Metric

The self-overlap matrices `G_LL` and `G_RR` are metrics on modal amplitude
space.

If modes are orthonormalized in the same inner product used for the interface
solve, those metrics become identity matrices and the simple formulas above
apply directly.

If the basis is not orthonormal in that metric, then the generalized formulas
must be used. In that setting, passivity is also naturally expressed in the
metric-weighted form

$$
S^\dagger G_{\mathrm{out}} S \le G_{\mathrm{in}}
$$

rather than in plain Euclidean coordinates.

## Analytical Sanity Checks

### Same Medium

If the two sides are identical, then `O_LR = O_RL = G`.

For orthonormal modes, `G = I`, so

$$
T = 2 (I + I)^{-1} = I
$$

$$
R = \tfrac{1}{2}(I - I) I = 0
$$

This is the basic self-consistency check for the interface formulas.

### Fresnel Limit

In a single-mode homogeneous limit, the modal overlap formulas reduce to the
usual normal-incidence Fresnel coefficients.

With normalized overlaps

$$
O_{LR} = \sqrt{n_R / n_L}, \qquad O_{RL} = \sqrt{n_L / n_R}
$$

one obtains

$$
T = \frac{2}{\sqrt{n_R/n_L} + \sqrt{n_L/n_R}}
=
\frac{2 \sqrt{n_L n_R}}{n_L + n_R}
$$

$$
R =
\frac{\sqrt{n_L/n_R} - \sqrt{n_R/n_L}}
{\sqrt{n_R/n_L} + \sqrt{n_L/n_R}}
=
\frac{n_L - n_R}{n_L + n_R}
$$

which matches the Fresnel result in the chosen mode-amplitude convention.

## Practical Numerical Pipeline

The practical pipeline used by the library is:

1. compute modes;
2. post-process them into a filtered, orthonormal basis;
3. build the overlap matrices;
4. solve for the transmission blocks with a TSVD-regularized inverse;
5. reconstruct the reflection blocks from continuity;
6. assemble the interface S-matrix;
7. enforce passivity on the singular values of `S`;
8. optionally enforce reciprocity by symmetrization.

The key sharp edge is that steps 2 and 3 must use compatible inner products.

## TSVD Regularization

The raw interface solve is most sensitive to small singular values of

$$
A_{LR} = O_{LR} + O_{RL}^T
$$

and its right-incidence counterpart.

Rather than using a direct inverse, the library solves with a truncated SVD:

$$
T_{LR} = 2 A_{LR,\mathrm{TSVD}}^{+}
$$

where singular values below

$$
s_{\mathrm{cut}} = rcond \cdot s_{\max}
$$

are discarded.

This is a pragmatic regularization. It is not exact physics; it is a way of
preventing a truncated basis from letting tiny singular directions dominate the
low-order scattering coefficients.

## Passivity And Truncation

With a complete basis, the lossless interface solve is unitary.

With a truncated basis, the computed S-matrix can violate passivity. A common
engineering interpretation is that the missing channels are mostly radiation or
continuum-like modes that should carry power away from the retained guided
subspace.

The following argument is heuristic, not a proof.

In the scalar case, if truncation makes a transmission coefficient appear as

$$
T = 1 + \varepsilon
$$

with `eps > 0`, one may interpret the excess as power that should have gone
into omitted channels. That motivates correcting overly large singular values of
the assembled `S`.

Common corrections are:

- `clip`: `sigma -> min(sigma, 1)`
- `invert`: `sigma -> 1 / sigma` for `sigma > 1`
- `subtract`: `sigma -> max(0, 2 - sigma)` for `sigma > 1`

`clip` is the simplest algebraic projection.

`invert` and `subtract` both map mild gain above one to mild loss below one,
which can be a useful model when a truncated solve is interpreted as missing
radiation loss.

These corrections are practical modeling choices. They should not be confused
with proof of convergence.

## What The Algorithm Does Well

- captures reflection that is destroyed by the symmetric product;
- provides a consistent overlap-based interface operator;
- exposes regularization and passivity controls explicitly;
- works naturally with unequal numbers of modes on each side.

## What Still Requires Care

- reflection is often a small difference of nearly equal overlap terms, so it
  is more sensitive than transmission to mesh, ordering, and truncation;
- passivity correction can improve robustness, but it does not replace basis
  convergence;
- if you override the default inner product, you should usually pass the same
  callable both to mode post-processing and to interface construction.
