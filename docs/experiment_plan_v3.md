# Experiment Plan V3

V3 no reemplaza V1 ni V2. Es una linea separada para probar features de alineamiento de senales.

Mejor baseline actual:

```text
V1 LightGBM 3M: RMSE 19.6810
```

## Idea

V1 usa buenas features tabulares, pero el problema parece depender mucho de comparar el patron `GR` del pozo horizontal contra el patron `GR` del `typewell`.

V3 parte de las features V1 y agrega features de matching por ventanas:

- `tw_match_tvt_21`
- `tw_match_error_21`
- `tw_match_minus_baseline_21`
- `tw_match_tvt_corrected_21`
- `tw_match_tvt_101`
- `tw_match_error_101`
- `tw_match_minus_baseline_101`
- `tw_match_tvt_corrected_101`
- `tw_anchor_shift_21`
- `tw_anchor_shift_101`

## Como Se Calcula

Para cada pozo:

1. Se calculan descriptores locales de `GR` en el horizontal.
2. Se calculan descriptores equivalentes de `GR` en el typewell.
3. Para cada punto del horizontal, se busca el punto del typewell con patron local mas parecido.
4. Se guarda el `TVT` del match, el error del match y la diferencia contra `baseline_tvt`.
5. En la parte conocida antes del corte, se calcula un offset promedio entre el match por `GR` y el `TVT_input` real. Ese offset se usa para corregir el match en todo el pozo.

## Modelos Iniciales

Probaremos solo los modelos mas prometedores y livianos:

- `lightgbm_v3`
- `catboost_v3`

