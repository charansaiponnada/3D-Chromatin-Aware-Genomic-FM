# B3 depth gate — K562, pre-registered before the number existed

Written 2026-08-31 **before** `phi_chr14_5000bp.npz` was built for K562, while
the Hi-C band was still streaming. Recorded here so the threshold cannot be
adjusted to the answer. This is a *data-quality* gate I set, not a PI-recorded
experimental endpoint (`architecture_spec.md` §7 is untouched).

## Why a gate exists

`phase1_acquire.py:66-69` states the caveat in the repo's own words: GM12878's
experiment set is 72 experiments merged; K562 is 6, IMR90 is 7. "Confirm this
doesn't starve the balancing weights (the usable-bin fraction) before treating
results as comparable."

## Comparison point — GM12878, measured

| chrom | n_bins | usable | usable_frac | r insulation_100kb vs 4DN | r compartment vs 4DN |
|---|---|---|---|---|---|
| chr9  | 27,679 | 21,519 | 0.7774 | 0.9969 | 0.9759 |
| chr10 | 26,760 | 25,378 | 0.9484 | 0.9969 | 0.9612 |
| chr11 | 27,018 | 25,426 | 0.9411 | 0.9972 | 0.9774 |
| chr12 | 26,656 | 25,688 | 0.9637 | 0.9971 | 0.9585 |
| chr13 | 22,873 | 18,888 | 0.8258 | 0.9968 | 0.9655 |
| **chr14** | **21,409** | **17,211** | **0.8039** | **0.9971** | **0.9711** |
| chr15 | 20,399 | 14,818 | 0.7264 | 0.9969 | 0.9482 |

chr14 is the gate chromosome: a val chromosome in the multichrom split, and
mid-range on usable_frac.

## Thresholds — fixed in advance

K562 chr14 passes only if **all three** hold:

1. `usable_frac >= 0.70`. Below GM12878 chr14's 0.8039 by up to ~0.10
   absolute, and below the weakest GM12878 chromosome (chr15, 0.7264).
   A shallower map is expected to lose some bins; losing more than this means
   the balancing weights are starved and φ is not measuring the same thing.
2. `pearson_insulation_100kb_vs_4DN >= 0.95`, against **K562's own** 4DN
   insulation track (accession 4DNFIXU7QLG6), not GM12878's. GM12878 achieves
   0.9968–0.9972 across seven chromosomes; 0.95 allows real degradation while
   still requiring the feature to be the quantity it claims to be.
3. `pearson_compartment_vs_4DN >= 0.90`, against 4DNFIWUAO2QI. GM12878 spans
   0.9482–0.9774.

## Outcome branches — also fixed in advance

- **All three pass** → acquire K562 for the remaining six chromosomes and treat
  B3 as live.
- **Any one fails** → do **not** silently proceed. K562 is either dropped or
  retried with IMR90 (7 experiments, marginally deeper), and the failure is
  recorded in `project.tex` as a measured negative about map depth, not hidden.
  A cross-cell-line claim built on a starved map would be worse than no
  cross-cell-line claim.
- Partial failure is **not** rescued by lowering a threshold in this file.
