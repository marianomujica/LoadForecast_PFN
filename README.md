# JIT 2026: demanda eléctrica con TabPFN-3

API y pipeline reproducible para seis experimentos de predicción de potencia activa a 15 minutos usando exclusivamente `TabPFNRegressor` V3 de Prior Labs.

## Entorno inspeccionado

El desarrollo fue verificado con Python 3.14.3, `tabpfn==8.4.0`, PyTorch CPU `2.13.0`, pandas 3.0.5, NumPy 2.5.2 y scikit-learn 1.9.0. La configuración usa CPU, `n_estimators=1`, `fit_mode=fit_preprocessors` y permite controlar explícitamente `ignore_pretraining_limits`. TabPFN 8.4.0 declara 500 features y 10.000 muestras como límites de preentrenamiento; V3 limita por defecto CPU a 5.000 muestras por rendimiento. El override queda registrado y, cuando es `false`, TabPFN rechazará datasets que superen esos límites.

## Configuración

Editar `config/experiment_config.txt`, que es el archivo principal de configuración y usa el formato `nombre: valor`. También se puede indicar otro archivo con `JIT_CONFIG`. Allí se cambian rutas, fechas, ratios, `EXPERIMENT_ID`, parámetros de TabPFN y `MAX_HISTORICAL_ROWS`. Las rutas relativas se resuelven desde la raíz del proyecto. No hay rutas absolutas en los módulos. `DATA_END_DATE` debe ser anterior al `HOLDOUT_DATE`. El 19/04/2026 y cualquier fecha posterior se excluyen del dataset de entrenamiento; el 20/04/2026 queda fuera aunque esté incompleto.

## Instalación y ejecución

Usar el entorno existente de TabPFN y ejecutar desde esta carpeta:

```powershell
C:\Users\Mariano\source\repos\TabPFN-venv\Scripts\python.exe -m pip install -r requirements.txt
C:\Users\Mariano\source\repos\TabPFN-venv\Scripts\python.exe -m unittest discover -s tests -v
C:\Users\Mariano\source\repos\TabPFN-venv\Scripts\python.exe -m scripts.prepare_data
```

Entrenar un experimento independiente:

```powershell
python -m scripts.train_experiment_1
python -m scripts.train_experiment_2
# ... hasta scripts.train_experiment_6
```

Cada experimento produce un único archivo `model_experiment_N.tabpfn_fit` y su metadata. El formato es el mecanismo oficial de persistencia de TabPFN 8.4.0 y requiere un entorno compatible para cargar los pesos base. En esta computadora se recomienda mantener `max_historical_rows: 8000` y `prediction_batch_size: 128`; usar `null` en `max_historical_rows` conserva todas las filas, pero puede superar los límites prácticos de CPU.

## API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Rutas: `GET /health`, `POST /prepare-data`, `POST /train/{experiment_id}`, `POST /train-all`, `POST /predict/{experiment_id}`, `GET /metrics/{experiment_id}`. Swagger queda disponible en `/docs`.

La predicción conserva timestamp separado, no usa timestamp ni target como feature y carga la lista exacta de features desde metadata. La validación externa del 19/04/2026 debe realizarse fuera del entrenamiento, aportando el target real solo para comparar resultados.

## Experimentos

- E1: vector, 11 features base + 96 valores de D-1 = 107.
- E2: estadísticas, 11 features base + 4 estadísticas de D-1 = 15.
- E3: vector, 11 features base + 192 valores de D-1..D-2 = 203.
- E4: estadísticas, 11 features base + 8 estadísticas de D-1..D-2 = 19.
- E5: vector, 11 features base + 288 valores de D-1..D-3 = 299.
- E6: estadísticas, 11 features base + 12 estadísticas de D-1..D-3 = 23.

La meteorología se filtra a `ROSARIO AERO`, se interpola de horario a 15 minutos y se completa solo en extremos con `ffill/bfill`. La potencia activa nunca se interpola. Los días incompletos se registran y se descartan cuando no permiten construir lags completos.

## Resultados y auditoría

Cada carpeta `outputs/experiment_N/` contiene dataset, predicciones TEST con timestamp, métricas TXT, gráficos HTML Plotly, metadata y modelo. Las métricas son R2, MAE, RMSE, MAPE excluyendo reales iguales a cero, MAE_P90 y Bias_P90 sobre `actual >= percentil90(actual)`.
