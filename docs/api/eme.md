# API / EME

This page collects the EME functions that carry the most assumptions.

## Recommended Mental Model

`compute_interface_s_matrix` expects a mode basis that is already filtered and
orthonormalized in the same inner product used for the interface overlaps.

The default path in `meow` is internally consistent because:

- the default modal inner product is asymmetric and unconjugated;
- the default mode post-processing uses that same callable;
- the default interface construction also uses that same callable.

If you override the interface inner product, you should usually override the
mode post-processing inner product too.

## EME Interface API

::: meow.eme.compute_interface_s_matrix

::: meow.eme.compute_interface_s_matrices

::: meow.eme.overlap_matrix

::: meow.eme.tsvd_solve

::: meow.eme.enforce_passivity

## Mode Post-Processing API

These functions live in the FDE pipeline, but they are directly relevant to the
validity of the simplified interface formulas.

::: meow.fde.post_process.post_process_modes

::: meow.fde.post_process.orthonormalize_modes

::: meow.fde.post_process.normalize_modes
