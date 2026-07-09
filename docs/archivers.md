# Archiving Strategies

The cleaned pipeline currently implements deterministic, lightweight versions of the following offline archivers. All operate on normalized minimization objectives where lower is better.

| Name | Idea |
| --- | --- |
| `pq` | Exact non-dominated archive, truncated with crowding distance when needed. |
| `hv` | Greedy hypervolume-oriented subset selection for 2D normalized objectives. |
| `r2` | Greedy R2-oriented subset selection for 2D normalized objectives. |
| `crowding` | Non-dominated archive truncated by NSGA-II style crowding distance. |
| `grid` | Spatial grid cells; keeps a deterministic representative per occupied cell. |
| `epsilon` | Epsilon-box archive; keeps a deterministic representative per epsilon box. |
| `tight1` | Structure-preserving farthest-point selection along the front. |
| `kmeans` | Seeded k-means selection in objective space, then nearest representative per cluster. |
| `entropy` | Grid-rarity selection that prefers less populated objective-space cells. |

These implementations are intended for transparent offline analysis and small-to-medium reproducibility runs. The original legacy implementations remain in `archivers/` for historical traceability.

Tie-breaking is deterministic through objective sums and architecture identifiers. Randomized methods accept the global `seed` from the YAML config.
