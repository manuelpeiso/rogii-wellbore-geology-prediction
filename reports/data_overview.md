# Data Overview

## Files

| item | count |
| --- | --- |
| train horizontal CSVs | 773 |
| train typewell CSVs | 773 |
| train PNG images | 773 |
| test horizontal CSVs | 3 |
| test typewell CSVs | 3 |
| sample submission rows | 14151 |

## Horizontal Well Columns

- Train: `MD`, `X`, `Y`, `Z`, geology top markers (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`), `TVT`, `GR`, `TVT_input`.
- Test: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`.
- Target: `tvt` in `sample_submission.csv`, keyed by `<well_id>_<row_index>`.

## Horizontal Well Summary

| index | min | mean | 50% | max |
| --- | --- | --- | --- | --- |
| rows | 2058.0 | 6586.95 | 6574.0 | 12141.0 |
| gr_missing_pct | 0.65 | 29.35 | 27.69 | 80.1 |
| tvt_input_missing_pct | 19.78 | 73.32 | 74.0 | 87.52 |
| ps_row | 851.0 | 1692.44 | 1702.5 | 2392.0 |

## Test Wells

| well_id | rows | md_min | md_max | gr_missing | tvt_input_missing | ps_row |
| --- | --- | --- | --- | --- | --- | --- |
| 000d7d20 | 5278 | 11467.0 | 16744.0 | 2258 | 3836 | 1442 |
| 00bbac68 | 7559 | 11578.0 | 19136.0 | 942 | 6014 | 1545 |
| 00e12e8b | 6384 | 10456.0 | 16839.0 | 584 | 4301 | 2083 |

## Typewell Summary

| index | min | mean | 50% | max |
| --- | --- | --- | --- | --- |
| rows | 636.0 | 2026.86 | 1874.5 | 10043.0 |
| tvt_min | 9232.65 | 10829.46 | 10639.19 | 12431.95 |
| tvt_max | 10236.85 | 11713.68 | 11537.41 | 12991.53 |
| gr_missing | 0.0 | 0.0 | 0.0 | 0.0 |
| geology_labels | 0.0 | 6.74 | 6.0 | 21.0 |

## Geology Labels In Train Typewells

| geology | rows |
| --- | --- |
| ANCC | 294268 |
| EGFDL | 205397 |
| ASTNL | 172223 |
| BUDA | 140640 |
| ASTNU | 118025 |
| EGFDU | 70013 |
| OLMOS | 23345 |
| MNSS | 5026 |
| UPSN | 2731 |
| LBHL | 1625 |
| LTHL | 1137 |
| LTGT | 981 |
| Clay Rich Interval | 868 |
| AC_UEF_THL | 840 |
| AC_UEF_TRGT | 840 |
| AC_UEF_BHL | 840 |
| UTGT | 760 |
| UTHL | 589 |
| UEGFD THL | 570 |
| UEGFD TGT | 570 |
| LL_BHL | 444 |
| UEGFD BHL | 323 |
| LL_THL | 276 |
| UL_THL | 258 |
| UL_TGT | 216 |
| UBHL | 190 |
| LL_TGT | 144 |
| UL_BHL | 120 |
| LLEF BHL | 69 |
| LLEF THL | 36 |
| ULEF TGT | 35 |
| EGFD400 | 34 |
| LL THL | 30 |
| LLEF TGT | 26 |
| EGFD300 | 23 |
| EGFD300c | 18 |
| ULEF THL | 16 |
| LL TGT | 10 |
| EGFD100 | 8 |
| EGFD_IPT | 2 |
| ULEF BHL | 2 |
| EGFD300b | 2 |
| EGFD200 | 1 |

## Figures

- `reports/figures/example_well_tvt_gr.png`
- `reports/figures/train_size_and_hidden_pct.png`
