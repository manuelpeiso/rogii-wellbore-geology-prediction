# ROGII Wellbore Geology Prediction

Proyecto para la competencia de Kaggle **ROGII - Wellbore Geology Prediction**.

La tarea inicial es predecir `tvt` para filas ocultas de pozos horizontales. La metrica de evaluacion indicada en la presentacion oficial es RMSE sobre `manualTVT - predictedTVT` para todos los puntos predichos.

Los datos incluyen:

- `train/*__horizontal_well.csv`: trayectoria del pozo, `GR`, `TVT`, `TVT_input` y marcadores geologicos.
- `train/*__typewell.csv`: curva de referencia por `TVT` y `GR`.
- `test/*__horizontal_well.csv`: mismas trayectorias de test, con `TVT_input` parcialmente oculto.
- `test/*__typewell.csv`: curva de referencia de test.
- `sample_submission.csv`: ids con formato `<well_id>_<row_index>` y columna objetivo `tvt`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Datos

Si necesitas volver a descargar:

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
unzip -q data/raw/rogii-wellbore-geology-prediction.zip -d data/raw
```

## Baseline

Genera una submission por interpolacion lineal de `TVT_input` dentro de cada pozo:

```bash
python src/baseline_interpolate.py --raw-dir data/raw --output submissions/baseline_interpolate.csv
```

Tambien imprime una validacion rapida usando las filas de entrenamiento donde `TVT_input` esta oculto.

Resultado inicial local:

```text
Validation RMSE on hidden train TVT_input rows: 105.5294
```

Este baseline solo usa la tendencia de `TVT_input`. El siguiente salto razonable es aprovechar la correlacion de `GR` del pozo horizontal contra el `typewell`, y despues incorporar pozos vecinos para modelar el dip geologico.
