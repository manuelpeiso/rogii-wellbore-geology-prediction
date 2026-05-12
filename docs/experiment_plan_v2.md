# Experiment Plan V2

La linea actual queda intacta:

```text
src/make_point_features.py
src/train_model.py
```

El mejor resultado actual de esa linea es:

```text
LightGBM 3M: RMSE 19.6810
```

La nueva linea experimental vive en archivos separados:

```text
src/make_point_features_v2.py
src/train_model_v2.py
```

## Objetivo

Bajar el RMSE sin perder la capacidad de reproducir el baseline fuerte actual.

## Cambios De Features

V2 parte de las features actuales y agrega:

- `baseline_delta_from_last`: cuanto se separa el baseline del ultimo TVT conocido.
- `baseline_slope_from_ps`: pendiente implicita del baseline desde el corte.
- `md_norm`: posicion normalizada dentro del pozo.
- `z_delta_1`, `z_delta_10`: cambio local de trayectoria vertical.
- `gr_roll_mean_101`, `gr_roll_std_101`: contexto GR mas amplio.
- `gr_ewm_mean_20`: promedio exponencial de GR.
- `gr_gradient_md`: cambio de GR por unidad de MD.
- `gr_residual_roll_mean_51`: promedio local de diferencia entre GR horizontal y GR esperado del typewell.
- `nearest_typewell_tvt_minus_baseline`: desacuerdo entre match por GR y baseline lineal.
- `baseline_to_typewell_min`, `baseline_to_typewell_max`: posicion del baseline dentro del rango del typewell.

## Cambios De Modelo

V2 usa features nuevas e hiperparametros nuevos. La V1 queda como comparacion estable.

- `random_forest_v2`
- `extra_trees_v2`
- `hist_gradient_boosting_v2`
- `lightgbm_v2`
- `xgboost_v2`
- `catboost_v2`

Los resultados se guardaran separados:

```text
data/processed_v2/
reports/model_results_v2.csv
models/*_v2_*.joblib
submissions/*_v2_*.csv
```
