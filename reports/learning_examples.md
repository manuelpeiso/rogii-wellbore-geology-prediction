# Learning Examples

Estos tres pozos de entrenamiento tienen `TVT` real completo, pero simulamos el problema mirando el punto donde `TVT_input` se corta.

En la figura:

- La linea negra es el `TVT` real completo. En test no lo veremos.
- La linea azul es `TVT_input`, la parte que Kaggle si nos deja ver.
- La linea roja vertical es el corte: desde ahi empieza lo que hay que predecir.
- La nube/curva de `GR` es la pista que puede ayudar a saber si el `TVT` sube, baja o queda plano.

## Summary

| well_id | case | total_rows | prediction_start_row | hidden_rows | hidden_tvt_start | hidden_tvt_end | hidden_tvt_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ba48188d | Caso A: el TVT baja despues del corte | 5534 | 1368 | 4166 | 12271.55 | 12167.76 | -103.79 |
| a959858c | Caso B: el TVT sube despues del corte | 5770 | 1366 | 4404 | 12134.38 | 12223.59 | 89.21 |
| 283269ac | Caso C: el TVT queda casi plano | 4567 | 1941 | 2626 | 11451.61 | 11451.39 | -0.22 |

## Figure

`reports/figures/learning_examples.png`
