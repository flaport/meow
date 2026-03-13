# Ideas

## Periodic boundary conditions + PML for symmetry-classified modes

The tidy3d FDE mode solver uses PEC boundaries by default, which can break the
mirror symmetry of the waveguide cross-section. This causes eigenmodes to be
arbitrary linear combinations within near-degenerate subspaces rather than
parity-classified (even/odd), leading to:

- Small but nonzero cross-couplings (~1e-5) in the S-matrix that should be
  exactly zero by symmetry
- S ≠ S^T and |R_LL| ≠ |R_RR|
- Accumulated numerical errors when cascading many interface S-matrices in EME

**Proposed fix:** Use periodic boundary conditions for the gradient operators
(finite-difference derivative matrices) while keeping PML for radiation
absorption. With periodic derivatives, the discrete curl operator exactly
commutes with mirror reflection (assuming symmetric grid + symmetric PML), so
eigenmodes would naturally be parity-classified.

**Implementation:** Patch tidy3d's derivative matrices
(`tidy3d/components/mode/derivatives.py`) so the forward/backward difference
matrices wrap around (periodic) instead of terminating at PEC. The PML complex
stretching is applied on top as usual.

**Trade-offs:**
- Preserves PML absorption (clean mode spectrum, few modes needed)
- Modes still complex-valued (PML), but parity-classified
- Requires patching tidy3d internals (monkey-patch or fork)
- Need to verify PML + periodic BC interaction is well-posed
