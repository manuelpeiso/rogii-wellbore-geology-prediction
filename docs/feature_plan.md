# Feature Plan

Vamos a empezar con **Forma 1**: cada fila que hay que predecir se convierte en un ejemplo de entrenamiento.

```text
un punto del pozo -> features -> TVT
```

## Target

En train:

```text
y = TVT
```

Usaremos principalmente las filas donde `TVT_input` esta vacio, porque esas imitan el caso real de test.

En test:

```text
prediccion = tvt
```

para cada `id` de `sample_submission.csv`.

## Feature Groups

### 1. Identidad y posicion dentro del pozo

Estas features le dicen al modelo donde esta el punto.

- `row_index`
- `n_rows`
- `row_from_ps`: cuantas filas despues del corte estamos
- `frac_from_ps`: posicion relativa despues del corte
- `MD`
- `md_from_ps`: distancia en MD desde el corte

### 2. Coordenadas y trayectoria

Estas features describen hacia donde se mueve el pozo.

- `X`, `Y`, `Z`
- `x_from_ps`
- `y_from_ps`
- `z_from_ps`
- `xy_dist_from_ps`

### 3. Senal GR del horizontal

`GR` es una pista central, pero tiene nulos. Por eso guardamos la senal interpolada y tambien si el dato era originalmente nulo.

- `gr`
- `gr_was_missing`
- `gr_from_ps`
- `gr_roll_mean_11`
- `gr_roll_mean_51`
- `gr_roll_std_51`
- `gr_delta_1`
- `gr_delta_10`

### 4. Contexto de TVT_input

Estas features resumen el comienzo conocido del TVT.

- `last_tvt_input`
- `first_tvt_input`
- `tvt_input_range`
- `tvt_slope_last_25`
- `tvt_slope_last_100`
- `baseline_tvt`

`baseline_tvt` es la continuacion lineal simple de `TVT_input`; el Random Forest puede aprender a corregirla.

### 5. Relacion con el typewell

El typewell es la referencia asociada al pozo. Usamos su curva `TVT` vs `GR` como comparacion.

- `typewell_tvt_min`
- `typewell_tvt_max`
- `typewell_gr_at_baseline_tvt`
- `gr_minus_typewell_gr_at_baseline`
- `nearest_typewell_tvt_by_gr`
- `nearest_typewell_gr_diff`

## Primer modelo

Random Forest inicial:

```text
features -> RandomForestRegressor -> TVT
```

Despues compararemos contra el baseline lineal actual.

