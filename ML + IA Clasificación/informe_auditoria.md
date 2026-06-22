# Informe Técnico de Auditoría — Proyecto Telecom X (Churn)

**Equipo auditor:** Principal Data Scientist · ML Researcher · Estadístico (inferencia predictiva) · MLOps Engineer · Profesor de ML
**Objeto auditado:** `Telecom_X.ipynb` (80 celdas) + `data_limpia.csv` (7032 × 22) + `modelo_telecom.pkl`
**Fecha:** 2026-06-21
**Naturaleza del informe:** diagnóstico. No se modifica el notebook original; se documenta qué está bien, qué está mal y por qué, con evidencia obtenida directamente del dataset.

---

## 0. Resumen ejecutivo

El notebook original es un trabajo correcto a nivel de "primer proyecto de clasificación": separa train/test, prueba un baseline, entrena dos modelos, hace selección de variables y optimiza hiperparámetros con `GridSearchCV` estratificado. Sin embargo, **contiene al menos tres defectos metodológicos que invalidan parcialmente las métricas reportadas** y una serie de decisiones tomadas por convención y no por evidencia.

| # | Severidad | Hallazgo | Efecto |
|---|-----------|----------|--------|
| L1 | 🔴 Alta | **Preprocesamiento ajustado sobre todo el dataset antes del split** (`one_hot.fit_transform(X)` en celda 20, *antes* del `train_test_split` de la celda 22) | Data leakage de preprocesamiento. El `MinMaxScaler` aprende min/max usando también las filas de test. |
| L2 | 🔴 Alta | **Selección del número de features mirando el conjunto de test** (celdas 48 y 52: el bucle elige `i` variables evaluando `X_test`/`y_test`) | El test deja de ser test. "19 / 17 / 18 variables óptimas" están elegidas sobre datos que debían permanecer ciegos → métricas optimistas. |
| L3 | 🟠 Media | **Métrica de selección = Recall puro** (`refit='recall'` en GridSearchCV) | Recall se maximiza trivialmente prediciendo casi todo como "churn". Sin un control de precisión (F1, PR-AUC, coste), el "ganador" puede ser un modelo casi inútil. |
| C1 | 🟠 Media | **Feature redundante exacta:** `account_charges_day == account_charges_monthly / 30` (verificado, ratio constante = 30.0) | Multicolinealidad perfecta; no aporta información (MI 0.046 vs 0.047). Infla artificialmente el conteo de variables. |
| C2 | 🟡 Baja | `account_charges_total ≈ monthly × tenure` (corr = 0.9996) | Multicolinealidad fuerte; relevante para modelos lineales y para la interpretación de importancias. |
| M1 | 🟠 Media | **No hay pipeline cerrado.** Encoder/scaler viven fuera del flujo de validación | Cualquier CV posterior hereda el leakage L1 y no es reproducible célula a célula. |
| M2 | 🟡 Baja | **Comparación de modelos sobre una sola partición de test** sin desviación estándar ni intervalos | Imposible saber si RF > XGB es señal o ruido de muestreo. |
| M3 | 🟡 Baja | `device='cuda'` hardcodeado en XGBoost | El notebook no corre en máquinas sin GPU NVIDIA (reproducibilidad). |
| M4 | 🟡 Baja | Universo de modelos reducido (Dummy, RF, XGB) | No se prueban LightGBM, CatBoost, HistGB, SVM, KNN, regresión logística como modelo (solo como nada). |
| R1 | 🟢 Info | 22 filas con perfil de features idéntico y, en 18 perfiles, **etiquetas en conflicto** | No es un bug: es el ruido irreducible (techo de Bayes). Conviene documentarlo para fijar expectativas realistas. |

> **Conclusión del resumen:** las métricas del notebook original **no son confiables como estimación de generalización** por L1 y L2. El modelo guardado puede funcionar, pero su rendimiento reportado está sesgado al alza y la elección de "Random Forest ganador por Recall" no está justificada estadísticamente.

---

## 1. Fortalezas (lo que está bien hecho)

1. **Separación train/test con `random_state` fijo.** Hay conciencia de que test debe existir.
2. **Uso de `StratifiedKFold`** para la búsqueda de hiperparámetros — correcto en un problema desbalanceado (26.6 % churn).
3. **Tratamiento del desbalance:** `class_weight='balanced'` en RF y `scale_pos_weight` en XGBoost. Es la decisión adecuada (mejor que sobremuestrear a ciegas).
4. **Baseline explícito** (`DummyClassifier`): buena práctica para tener un piso de comparación.
5. **`drop='if_binary'`** en el OneHotEncoder: evita columnas redundantes en variables binarias.
6. **Dataset ya limpio:** sin valores faltantes, sin `tenure==0`, sin outliers por IQR. El preprocesamiento previo (no auditado aquí) fue cuidadoso.
7. **Exportación del modelo** como paquete `.pkl` con encoder + features + estimador: pensamiento orientado a despliegue.

---

## 2. Debilidades y errores conceptuales (detalle)

### 2.1 🔴 L1 — Leakage de preprocesamiento (celdas 18–22)

```python
# Celda 20  (ANTES del split)
X = one_hot.fit_transform(X)          # fit usa TODAS las filas
# Celda 22
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, ...)
```

El `MinMaxScaler` calcula `min` y `max` de cada variable numérica usando **el dataset completo, incluido el futuro test**. El test deja de ser una muestra "nunca vista": su rango ya influyó en la transformación.

- **Por qué importa:** la estimación de error en test queda sesgada (optimista). En MinMax el efecto suele ser pequeño, pero el **patrón** es el mismo que produce desastres con `StandardScaler`, `PowerTransformer`, imputación por media o, sobre todo, *target encoding*. La regla no admite excepciones: **todo lo que aprende parámetros se ajusta solo con train, dentro de un `Pipeline`.**
- **Corrección:** envolver encoder + scaler + modelo en un `Pipeline`/`ColumnTransformer` y pasar ese pipeline a `cross_validate`/`GridSearchCV`. Así el `fit` del preprocesador se reejecuta dentro de cada fold, solo sobre el train de ese fold.

### 2.2 🔴 L2 — Selección de features sobre el test (celdas 48 y 52)

```python
for i in ct_features:                       # i = 16..21
    selected_features = features_forest_df['Features'].values[:i]
    X_train_selected = X_train[selected_features]
    X_test_selected  = X_test[selected_features]
    modelo.fit(X_train_selected, y_train)
    y_pred = modelo.predict(X_test_selected)
    metricas = calcular_metricas(y_test, y_pred)   # <-- decide mirando TEST
```

Se elige el número de variables que **maximiza la métrica en test**. Eso convierte al test en un conjunto de validación de hiperparámetros. La conclusión "19 óptimas para RF, 17 para XGB, 18 de promedio" está contaminada: cualquier número elegido así rinde de más en ese test concreto y de menos en datos nuevos.

- **Corrección:** la curva "nº de features vs. métrica" debe trazarse con **validación cruzada sobre train** (o en un set de validación interno), nunca sobre el test. El test se toca una sola vez, al final.

### 2.3 🟠 L3 — `refit='recall'` como criterio único (celdas 64 y 67)

Optimizar recall sin restringir precisión es un anti-patrón: el óptimo degenerado es **predecir "churn" para casi todos** (recall → 1, precisión → tasa base). Un negocio que llama a retención a toda su cartera no ahorra nada.

- **Corrección:** seleccionar por una métrica que equilibre ambos errores y sea robusta al desbalance: **F1**, **PR-AUC** (average precision) o **MCC**; idealmente una métrica de coste si existe el coste de un falso negativo vs. el de un falso positivo. El recall se reporta, pero no se optimiza en solitario.

### 2.4 🟠 C1 — `account_charges_day` es redundante exacta

Verificado empíricamente: `monthly / day` es constante e igual a **30.0** en las 7032 filas ⇒ `charges_day = charges_monthly / 30`. Es la **misma variable reescalada**. Su mutual information con churn (0.0461) es prácticamente idéntica a la de `charges_monthly` (0.0472). No aporta nada y, en modelos lineales o en el cálculo de importancias, **reparte el crédito** de una variable entre dos, distorsionando la interpretación.

- **Corrección:** eliminar `account_charges_day` (o `monthly`, da igual: son colineales perfectas).

### 2.5 🟠 M1 — Ausencia de pipeline

Como el encoder y el scaler viven en variables sueltas (`one_hot`), la validación cruzada de la fase de tuning **opera sobre datos ya transformados con leakage** (L1). Además, reproducir el flujo exige ejecutar celdas en un orden frágil (hay dos `train_test_split` distintos, celdas 22 y 62, que reescriben `X_train`/`X_test`).

### 2.6 🟡 M2 — Comparación sin incertidumbre

La tabla final (celda 71) compara RF vs. XGB en **una sola partición**. Una diferencia de 1–2 puntos en F1 sobre 1407 filas de test está dentro del error de muestreo. Declarar un ganador así no es defendible.

- **Corrección:** comparar con **Repeated Stratified K-Fold** y reportar media ± desviación e intervalos de confianza; idealmente un test pareado (p. ej. Wilcoxon sobre los folds).

### 2.7 🟡 M3 / M4 — Reproducibilidad y cobertura de modelos

- `device='cuda'` impide ejecutar en CPU. Debe ser configurable.
- El abanico de modelos es estrecho. Falta, como mínimo: Logistic Regression (como modelo, no solo baseline), HistGradientBoosting, LightGBM, CatBoost, Extra Trees, SVM y KNN, para tener evidencia de que un modelo complejo realmente vale la pena frente a uno simple.

---

## 3. Riesgos metodológicos priorizados

| Riesgo | Probabilidad de afectar la decisión | Acción correctiva en la solución nueva |
|--------|-------------------------------------|----------------------------------------|
| Métricas infladas por L1 + L2 | **Alta** | Pipelines cerrados + test aislado + selección por CV |
| "Ganador" mal elegido por recall puro | **Alta** | Métrica primaria F1/PR-AUC/MCC + criterio de negocio |
| Multicolinealidad (C1, C2) confunde interpretabilidad | Media | Quitar `charges_day`; tratar `total` con cuidado en SHAP |
| Falsos positivos de "mejora" sin significancia (M2) | Media | Repeated CV + intervalos + test estadístico |
| No reproducible en CPU (M3) | Baja | `device` parametrizado / fallback a `hist` |

---

## 4. ¿Hay data leakage de variables (no de proceso)?

Revisado columna a columna: **no existe una variable "tramposa"** del tipo "motivo de baja" o "fecha de cancelación". Las variables de cargos (`monthly`, `total`, `day`) son históricas y **están disponibles en el momento de scorear** a un cliente activo, por lo que no constituyen leakage temporal. `customerid` es un identificador único (7032/7032) sin poder predictivo y debe eliminarse (el notebook lo hace, bien).

> El leakage del proyecto es **de proceso** (L1, L2), no de contenido. Es más sutil y más peligroso, porque no se ve en la lista de columnas.

---

## 5. Métricas potencialmente infladas

Las cifras de F1/Recall reportadas en el notebook deben leerse como **cota superior optimista**, no como estimación de generalización, por la combinación de:

1. Scaler ajustado con test (L1).
2. Nº de features elegido mirando test (L2).
3. Una sola partición sin barras de error (M2).

La solución nueva volverá a medir todo con protocolo limpio; es esperable que las métricas "honestas" sean **ligeramente inferiores** a las del notebook original. Eso no es un retroceso: es la diferencia entre una métrica de marketing y una métrica de ingeniería.

---

## 6. Oportunidades de automatización (lo que la nueva solución incorpora)

- **`Pipeline` + `ColumnTransformer`** únicos por modelo → cero leakage, reproducible.
- **Detección automática** de tipos, cardinalidad, redundancias y candidatos a leakage (FASE 2).
- **Selección de scaler y encoder por evidencia** (CV comparada), no por convención (FASE 4).
- **Optuna** con pruning y early stopping en vez de `GridSearchCV` exhaustivo (FASE 7).
- **Repeated Stratified K-Fold + intervalos de confianza + test pareado** para declarar ganadores (FASE 8).
- **Nested CV** para estimar el error de generalización sin sesgo de tuning (FASE 3).
- **SHAP + permutation importance** para interpretar, no solo `feature_importances_` (FASE 11).

---

## 7. Veredicto

> El notebook original **aprueba como ejercicio académico** pero **no como pieza de ingeniería**. Sus dos defectos de leakage de proceso (L1, L2) y su criterio de selección (L3) hacen que las métricas y la elección del modelo no sean defendibles ante una revisión rigurosa. La reconstrucción en `mejores_practicas.ipynb` corrige cada punto de esta tabla y **demuestra empíricamente** cada decisión, priorizando generalización real por encima de la métrica más vistosa.

El detalle de la reconstrucción, con código, justificación estadística, riesgos, alternativas y ejemplos, está en `documentacion_completa.md`.
