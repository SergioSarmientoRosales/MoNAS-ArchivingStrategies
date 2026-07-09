# Evaluation Indicators

All indicators in the cleaned pipeline are computed on normalized minimization objectives.

| Indicator | Direction | Meaning |
| --- | --- | --- |
| `igd_plus` | lower is better | Average IGD+ distance from reference front to approximation. |
| `hypervolume` | higher is better | Dominated area in 2D bounded by the configured reference point. |
| `r2` | lower is better | Preference-space approximation quality using weighted Tchebycheff scalarization. |
| `epsilon` | lower is better | Additive epsilon bound needed for the approximation to cover the reference. |
| `hausdorff` | lower is better | Worst-case symmetric geometric distance between reference and approximation. |

The default hypervolume reference point is `[1.1, 1.1]`, which is worse than the normalized `[0, 1]` objective range.

For more than two objectives, the current implementation falls back to non-hypervolume archivers and indicators that support arbitrary dimensions. Hypervolume and R2 are intentionally limited to 2D in this cleanup because the original research focus is PSNR versus parameter count.
