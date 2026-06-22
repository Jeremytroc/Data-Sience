    # Documentación completa — Telecom X · Modelo de Evasión (Churn)

    > Material de estudio avanzado. Replica **todos** los bloques de `mejores_practicas.ipynb` con: código, explicación detallada, justificación estadística, riesgos, alternativas, ejemplos intuitivos e impacto esperado.
    >
    > Lectura recomendada junto a `informe_auditoria.md` (diagnóstico del notebook original) y el notebook ejecutado.

    ## Tabla de contenidos
    - [Filosofía del proyecto](#filosofía-del-proyecto)
    - [1 · Configuración](#1--configuración)
    - [2 · Carga de datos](#2--carga-de-datos)
    - [3 · Auditoría de datos](#3--auditoría-de-datos-fase-2)
    - [4 · EDA](#4--análisis-exploratorio-eda)
    - [5 · Ingeniería de variables](#5--ingeniería-de-variables)
    - [6 · Estrategia de validación](#6--estrategia-de-validación-fase-3)
    - [7 · Preprocesamiento por evidencia](#7--preprocesamiento-por-evidencia-fase-4)
    - [8 · Baselines](#8--baselines-fase-5)
    - [9 · Modelado](#9--modelado-zoo-de-modelos-fase-6)
    - [10 · Estabilidad estadística](#10--estabilidad-estadística-fase-8)
    - [11 · Optimización (Optuna)](#11--optimización-de-hiperparámetros-con-optuna-fase-7)
    - [12 · Diagnóstico de generalización](#12--diagnóstico-de-generalización-fase-10)
    - [13 · Nested CV](#13--nested-cross-validation-fase-3)
    - [14 · Interpretabilidad](#14--interpretabilidad-fase-11)
    - [15 · Selección final](#15--métricas-umbral-y-selección-final-fases-9-y-12)
    - [16 · Evaluación en test](#16--evaluación-final-en-el-test-aislado)
    - [17 · Exportación](#17--exportación-del-modelo-mlops)
    - [18 · Conclusiones](#18--conclusiones)

    ---

    ## Filosofía del proyecto

    El objetivo **no** es maximizar una métrica en una partición concreta, sino estimar y maximizar el rendimiento sobre **datos nunca vistos**. Esto implica subordinar todo a una jerarquía:

    1. **Generalización real** → medida con protocolos que no se "auto-engañan".
    2. **Ausencia de data leakage** → el modelo nunca ve, ni indirectamente, información del futuro/test.
    3. **Robustez estadística** → toda diferencia se contrasta contra el ruido de muestreo.
    4. **Reproducibilidad** → semilla fija, pipelines cerrados, artefactos versionados.
    5. **Interpretabilidad** → entender *por qué* predice, no solo *qué* predice.
    6. **Poder predictivo** → último en la lista, no porque no importe, sino porque sin los cinco anteriores una métrica alta es una ilusión.

    **Ejemplo intuitivo del leakage de proceso.** Imagina que estudias para un examen y, sin darte cuenta, "calibras" tu método de estudio mirando las respuestas del examen real. Sacarás un 10 en ese examen y un 5 en el siguiente. El leakage hace exactamente eso: ajusta decisiones usando datos que deberían estar sellados, y la nota (métrica) deja de predecir el rendimiento futuro.

    ---

    ## 1 · Configuración

    ```python
    import warnings, time, json
    warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt, seaborn as sns
    from scipy import stats
    # ... sklearn, xgboost, lightgbm, catboost, optuna, shap ...
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ```

    **Explicación.** Centralizamos imports y, sobre todo, fijamos `RANDOM_STATE = 42` y lo propagamos a *cada* fuente de azar (splits, modelos, samplers de Optuna). Definimos una única función de métricas (`metricas_completas`) para que todas las fases midan exactamente lo mismo.

    **Justificación estadística.** La aleatoriedad entra en tres lugares: partición de datos, inicialización/submuestreo de los modelos y muestreo del optimizador bayesiano. Si no se fija, dos ejecuciones difieren y cualquier comparación se contamina con varianza no controlada. Fijar la semilla convierte el experimento en **determinista y auditable**.

    **Riesgos.** (a) *Sobreajustar a una semilla*: que el resultado dependa del 42. Lo mitigamos con validación **repetida** (varias particiones) y nested CV. (b) `warnings.filterwarnings("ignore")` puede ocultar avisos útiles; se asume porque el ruido de convergencia de algunos modelos satura la salida, pero en depuración conviene reactivarlos.

    **Alternativas.** Usar `sklearn.utils.check_random_state`, o ejecutar todo bajo varias semillas y promediar (más costoso). Para reproducibilidad bit a bit se fijarían también `PYTHONHASHSEED` y los hilos de BLAS.

    **Impacto esperado.** Reproducibilidad total y comparaciones limpias entre modelos.

    ---

    ## 2 · Carga de datos

    ```python
    df = pd.read_csv("data_limpia.csv")
    df = df.drop(columns=["customerid"])   # ID único: 0 señal
    ```

    **Explicación.** Cargamos el **mismo** archivo que el proyecto original (copiado sin modificar). `customerid` tiene 7032 valores únicos en 7032 filas: es una clave primaria, no una variable.

    **Justificación estadística.** Un identificador con cardinalidad = nº de filas tiene *mutual information* máxima con cualquier cosa **en train** y nula capacidad de generalizar: el modelo podría memorizar el mapa id→churn (sobreajuste perfecto) y fallar en clientes nuevos. Es el caso extremo de variable de alta cardinalidad.

    **Riesgos.** Olvidar eliminarlo introduce un leakage trivial pero devastador; algunos modelos de árbol lo usarían para memorizar.

    **Alternativas.** Si el ID codificara información (p. ej. orden temporal de alta), se extraería esa señal como feature; aquí no es el caso.

    **Ejemplo intuitivo.** Predecir si un alumno aprueba usando su número de matrícula: en el grupo conocido "aciertas" memorizando, pero con alumnos nuevos no sabes nada.

    **Impacto esperado.** Eliminar fuente de sobreajuste y reducir dimensionalidad inútil.

    ---

    ## 3 · Auditoría de datos (FASE 2)

    ```python
    audit = pd.DataFrame({"dtype": df.dtypes, "n_unique": df.nunique(),
                        "%_missing": df.isna().mean()*100})
    # redundancia determinística
    np.allclose(df["account_charges_day"], df["account_charges_monthly"]/30)   # True
    # multicolinealidad
    df["account_charges_total"].corr(df["account_charges_monthly"]*df["customer_tenure"])  # 0.9996
    # mutual information con churn  (mutual_info_classif)
    # perfiles idénticos con etiquetas opuestas -> techo de Bayes
    ```

    **Explicación y hallazgos (todos verificados sobre el dataset).**

    | Hallazgo | Evidencia | Acción |
    |---|---|---|
    | Sin valores faltantes | `%_missing = 0` en las 22 columnas | Ninguna imputación necesaria |
    | Sin duplicados exactos | `df.duplicated().sum() == 0` | — |
    | Sin constantes ni cuasi-constantes | clase minoritaria mínima ≈ 9.7 % (`phone_phoneservice`) | Se conservan |
    | **`charges_day` = `monthly`/30** | ratio constante = 30.0 | **Eliminar** (redundante exacta) |
    | `charges_total` ≈ `monthly`×`tenure` | corr = 0.9996 | Vigilar multicolinealidad |
    | MI más alta | `account_contract` (0.098), `tenure` (0.074), `internet_service` (0.055) | Predictores núcleo |
    | MI ≈ 0 | `gender` (0.000), `phone_phoneservice` (0.0001), streaming (~0.002) | Ruido casi puro |
    | 18 perfiles con churn en conflicto | mismas features, distinto resultado | **Techo de Bayes** (error irreducible) |
    | Desbalance | churn = 26.6 % | Tratar con pesos de clase |

    **Justificación estadística.** La *mutual information* mide dependencia **no lineal** entre cada variable y el target (en nats); a diferencia de la correlación de Pearson, detecta relaciones no monótonas y funciona con categóricas. El **techo de Bayes** es clave: si existen dos clientes idénticos en features con desenlaces opuestos, ningún clasificador determinista puede acertar a ambos; el error de Bayes es la cota inferior teórica del error, y nos dice que perseguir F1≈1 es perseguir un fantasma.

    **Riesgos.** (a) Eliminar variables por MI baja puede descartar señal en interacción; por eso no eliminamos por MI sola, solo `charges_day` (redundancia *probada*, no estadística). (b) La multicolinealidad `total≈monthly×tenure` inestabiliza los coeficientes lineales y reparte importancias — se documenta para interpretar con cuidado.

    **Alternativas.** VIF (Variance Inflation Factor) para cuantificar multicolinealidad; PCA para descorrelacionar (a costa de interpretabilidad); pruebas χ² para categóricas.

    **Ejemplo intuitivo.** `charges_day` es como tener el precio "por mes" y "por día" en la misma tabla: la segunda columna no añade nada, solo confunde a quien reparte el mérito de la predicción.

    **Impacto esperado.** Un dataset comprendido: sabemos qué variables son núcleo, cuáles son ruido, dónde está el límite teórico y qué no hay que tocar.

    ---

    ## 4 · Análisis exploratorio (EDA)

    ```python
    df["churn"].value_counts()                       # 73.4% / 26.6%
    df.groupby("account_contract")["churn"].mean()   # mes a mes 42.7%, anual 11.3%, bianual 2.8%
    df.groupby("internet_internetservice")["churn"].mean()  # fibra 41.9%, DSL 19.0%, sin internet 7.4%
    df.groupby("account_paymentmethod")["churn"].mean()     # cheque electrónico 45.3% ...
    ```

    **Explicación.** El EDA confirma y cuantifica los *drivers*:
    - **Contrato:** mes a mes 42.7 % vs bianual 2.8 % — la variable más discriminante.
    - **Internet:** fibra óptica 41.9 % (probable insatisfacción precio/servicio) vs sin internet 7.4 %.
    - **Pago:** cheque electrónico 45.3 % (fricción/segmento), muy por encima de pagos automáticos (~15 %).
    - **Tenure:** los clientes nuevos concentran el churn (curva de densidad desplazada a la izquierda en los que evaden).

    **Justificación estadística.** Las tasas condicionales `P(churn | categoría)` son estimadores directos del efecto marginal de cada categoría; las diferencias (2.8 % vs 42.7 %) son enormes frente al error de muestreo con n≈1500–3900 por grupo, así que son señales reales, no ruido.

    **Riesgos.** El EDA es **univariado**: una tasa alta puede deberse a confusión con otra variable (p. ej. fibra correlaciona con cargos altos). No se deben sacar conclusiones causales; para eso está SHAP (§14), que controla por el resto de variables.

    **Alternativas.** Gráficos de mosaico, *Weight of Evidence*, análisis bivariado con test χ² y V de Cramér.

    **Ejemplo intuitivo.** "Los que pagan con cheque electrónico se van más" es una **correlación**: quizá ese método lo usa un segmento más volátil, no que el cheque cause la fuga. El EDA señala *dónde mirar*, no *por qué*.

    **Impacto esperado.** Hipótesis claras que el modelo debe capturar y que luego validamos con interpretabilidad.

    ---

    ## 5 · Ingeniería de variables

    ```python
    df = df.drop(columns=["account_charges_day"])    # redundante exacta
    TARGET = "churn"
    CAT = ["internet_internetservice","account_contract","account_paymentmethod"]
    NUM = ["customer_tenure","account_charges_monthly","account_charges_total"]
    BIN = [c for c in df.columns if c not in CAT+NUM+[TARGET]]    # ya 0/1
    ```

    **Explicación.** Principio de **parsimonia**: eliminamos la única redundancia probada y agrupamos columnas por tipo de tratamiento (numéricas a escalar, categóricas a codificar, binarias a pasar tal cual). No inventamos variables sin evidencia de que ayuden.

    **Justificación estadística.** Cada feature redundante añade varianza al estimador sin reducir su sesgo (descomposición sesgo–varianza): empeora la generalización y, en modelos lineales, dispara la varianza de los coeficientes por multicolinealidad. Menos features ⇒ menor complejidad ⇒ menor riesgo de sobreajuste.

    **Riesgos.** Ingeniería agresiva (ratios, *bins*, interacciones) puede mejorar el train y empeorar el test, y es una vía clásica de leakage si usa estadísticas globales. Por eso aquí es mínima y cualquier feature nueva tendría que **demostrar** su valor por CV.

    **Alternativas.** Crear `tenure_bins`, `cargos_por_mes_de_vida`, interacciones contrato×internet; *target/WoE encoding* como features. Se probaron mentalmente y no se justifican: con árboles las interacciones se aprenden solas y con cardinalidad baja el OHE basta.

    **Ejemplo intuitivo.** Añadir variables "por si acaso" es como llevar 10 mapas distintos a un viaje: más peso, más confusión, y al final usas uno.

    **Impacto esperado.** Modelo más simple, más estable e igual de potente.

    ---

    ## 6 · Estrategia de validación (FASE 3)

    ```python
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)   # TEST AISLADO
    cv_skf  = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    ```

    **Explicación.** Apartamos un **20 % de test que no se vuelve a tocar** hasta la sección 16. Todas las decisiones (preprocesamiento, modelo, hiperparámetros, umbral) se toman sobre *train* con validación cruzada. Comparamos esquemas:

    | Esquema | nº estimaciones | SEM de la media | Uso |
    |---|---|---|---|
    | Hold-out simple | 1 | — (no estimable) | descartado |
    | Stratified K-Fold 5 | 5 | mayor | aceptable |
    | **Repeated SKF 5×3** | **15** | **menor** (≈ la mitad) | **elegido** para comparar |
    | Nested CV | externo×interno | insesgado | estimación final (§13) |

    Evidencia obtenida (LogReg): SKF-5 → SEM ≈ 0.0087; **RSKF-5×3 → SEM ≈ 0.0055**.

    **Justificación estadística (matemática de la elección).** El error estándar de la media de las puntuaciones de fold escala como `SEM = s/√k`, donde `s` es la desviación entre folds y `k` el número de estimaciones. Pasar de 5 a 15 estimaciones reduce el SEM en factor `√(15/5)=√3≈1.73`. Un SEM menor ⇒ **intervalos de confianza más estrechos** ⇒ podemos detectar diferencias reales entre modelos sin confundirlas con ruido. La **estratificación** mantiene la proporción 26.6 % de churn en cada fold; sin ella, con clase minoritaria, un fold podría quedar con muy pocos positivos y disparar la varianza. La **CV anidada** resuelve un sesgo distinto: cuando se usa la misma CV para *elegir* hiperparámetros y *reportar* el score, ese score es optimista (se eligió el mejor de muchos sobre esos mismos datos); el bucle externo evalúa sobre folds que el tuning interno nunca vio.

    **Riesgos.** (a) Test demasiado pequeño ⇒ métrica final ruidosa; 20 % de 7032 = 1407 filas, suficiente. (b) Repetir folds no corrige sesgo, solo varianza: si el protocolo tiene leakage, RSKF da un número estable… y establemente equivocado. Por eso el leakage se ataca con pipelines (§7), no con más folds.

    **Alternativas.** LOOCV (inviable y de alta varianza), `ShuffleSplit`, `TimeSeriesSplit` (si hubiera orden temporal — no es el caso), Bootstrap .632+.

    **Ejemplo intuitivo.** Medir tu tiempo en 100 m una sola vez (hold-out) vs. correr 15 veces y promediar (RSKF): la media de 15 carreras describe tu nivel real mucho mejor que una sola cronometrada con viento a favor.

    **Impacto esperado.** Comparaciones de modelos con incertidumbre cuantificada y un test verdaderamente ciego.

    ---

    ## 7 · Preprocesamiento por evidencia (FASE 4)

    ```python
    def make_pre(scaler="none", encoder="onehot"):
        scal = {"none":"passthrough","standard":StandardScaler(),"robust":RobustScaler(),
                "minmax":MinMaxScaler(),"power":PowerTransformer()}[scaler]
        enc  = {"onehot":OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                "ordinal":OrdinalEncoder(...), "target":TargetEncoder(random_state=RANDOM_STATE)}[encoder]
        return ColumnTransformer([("num",scal,NUM),("cat",enc,CAT),("bin","passthrough",BIN)])
    ```

    **Explicación.** No asumimos: medimos. Probamos 5 escaladores con un modelo sensible a la escala (KNN) y 3 codificadores con LogReg, **todo por CV dentro de pipelines**.

    Evidencia (F1, CV):

    | Escalador (KNN) | F1 | | Codificador (LogReg) | F1 |
    |---|---|---|---|---|
    | none | 0.499 | | one-hot | 0.628 |
    | **standard** | **0.549** | | ordinal | 0.629 |
    | robust | 0.549 | | **target** | **0.632** |
    | power | 0.549 | | | |
    | minmax | 0.540 | | | |

    **Decisiones por evidencia:** árboles → **sin escalado** (son invariantes a transformaciones monótonas); modelos de distancia/lineales → **StandardScaler** (robust/power empatan, se elige el más estándar e interpretable). Encoder → **One-Hot**: el *target encoding* solo aporta +0.004 con cardinalidad 3–4, y One-Hot es interpretable y sin riesgo de leakage.

    **Justificación estadística.** KNN y SVM-RBF dependen de **distancias euclídeas**: una variable en escala 0–8700 (`charges_total`) domina a otra en escala 0–72 (`tenure`) salvo que se estandarice; por eso "none" cae a 0.499. Los árboles parten por umbrales (`x < c`), insensibles a la escala, así que escalarlos es esfuerzo inútil. El **TargetEncoder de sklearn** usa validación cruzada interna para calcular las medias por categoría, evitando que la media incluya la fila que se está codificando (anti-leakage); aun así, con pocas categorías el One-Hot no pierde información y es más transparente.

    **Riesgos.** *Target encoding* mal hecho (media global sobre todo el dataset) es una de las formas más graves de leakage: filtra el target dentro de las features. Lo evitamos con la implementación con CV interna, dentro del pipeline. `MinMaxScaler` es sensible a outliers (usa min/max); aquí no hay outliers, pero es una razón para preferir Standard/Robust en general.

    **Alternativas.** `QuantileTransformer`, *binning* supervisado, *Weight of Evidence*, *frequency encoding* (útil con cardinalidad alta, irrelevante aquí).

    **Ejemplo intuitivo.** Comparar clientes por "distancia" sin escalar es como comparar personas sumando "diferencia de altura en mm" + "diferencia de edad en años": los mm aplastan a los años. Estandarizar pone todo en la misma unidad (desviaciones típicas).

    **Impacto esperado.** Cada familia de modelos recibe exactamente el preprocesamiento que necesita, sin leakage y elegido con datos.

    ---

    ## 8 · Baselines (FASE 5)

    ```python
    DummyClassifier(strategy="most_frequent")   # predice siempre "no churn"
    DummyClassifier(strategy="stratified")      # predice al azar respetando proporciones
    LogisticRegression(class_weight="balanced") # baseline FUERTE y candidato serio
    ```

    **Explicación.** Un baseline es la "línea de flotación": todo modelo complejo debe superarla con claridad para justificar su coste. Usamos tres:
    - **Mayoría:** Accuracy ≈ 73 % (¡engañoso!) pero **Recall = 0** y **F1 = 0** sobre churn.
    - **Estratificado:** F1 ≈ 0.27 (la tasa base), MCC ≈ 0.
    - **Regresión logística balanceada:** baseline fuerte que, spoiler, terminará siendo el modelo final.

    **Justificación estadística.** En un problema con 73/27, predecir siempre la clase mayoritaria da 73 % de accuracy sin aprender nada: por eso **Accuracy es una métrica peligrosa aquí** y el baseline lo demuestra numéricamente. El `DummyClassifier` materializa la **hipótesis nula** "no hay señal": si un modelo no la supera de forma significativa, no hay evidencia de que haya aprendido.

    **Riesgos.** Confundir un Accuracy alto del Dummy con un buen modelo. Es exactamente la trampa que evita reportar F1, MCC y PR-AUC.

    **Alternativas.** Baseline por reglas de negocio (p. ej. "todo cliente mes-a-mes con tenure<6 es churn"), o un *prior* bayesiano.

    **Ejemplo intuitivo.** Un detector de fraude que dice "nunca hay fraude" acierta el 99.9 % de las veces… y es inútil. El baseline de mayoría es ese detector.

    **Impacto esperado.** Un piso cuantitativo contra el que medir todo lo demás; protege contra el autoengaño del Accuracy.

    ---

    ## 9 · Modelado: zoo de modelos (FASE 6)

    ```python
    spw = (y_train==0).sum() / (y_train==1).sum()        # ≈ 2.77, peso de la clase churn
    modelos = { "LogReg", "SVM-RBF", "KNN", "RandomForest", "ExtraTrees",
                "HistGB", "XGBoost", "LightGBM", "CatBoost" }   # cada uno en su Pipeline
    # Evaluación: cross_validate(pipe, X_train, y_train, cv=cv_rskf, scoring=SCORERS)
    ```

    **Explicación.** Nueve familias que cubren el espacio de hipótesis: lineal (LogReg), kernel (SVM-RBF), basado en instancias (KNN), *bagging* de árboles (RF, ExtraTrees) y *boosting* (HistGB, XGBoost, LightGBM, CatBoost). Cada modelo va en **su** pipeline con el preprocesamiento adecuado y su manejo del desbalance:
    - `class_weight="balanced"` (LogReg, SVM, RF, ExtraTrees, HistGB),
    - `scale_pos_weight=spw` (XGBoost, LightGBM),
    - `auto_class_weights="Balanced"` (CatBoost).

    Resultados **reales** de la ejecución (RSKF 5×3, ordenados por F1; `gap` = F1_train − F1_cv = señal de sobreajuste):

    | Modelo | F1 | MCC | ROC-AUC | PR-AUC | F1_train | **gap** |
    |---|---|---|---|---|---|---|
    | **LogReg** | **0.6313** | **0.4804** | 0.8443 | 0.6525 | 0.6345 | **+0.003** |
    | CatBoost | 0.6293 | 0.4767 | 0.8438 | **0.6610** | 0.6994 | +0.070 |
    | SVM-RBF | 0.6287 | 0.4757 | 0.8271 | 0.5992 | 0.6581 | +0.029 |
    | XGBoost | 0.6246 | 0.4698 | 0.8393 | 0.6539 | 0.7494 | +0.125 |
    | HistGB | 0.6215 | 0.4663 | 0.8349 | 0.6442 | 0.7993 | +0.178 |
    | LightGBM | 0.6125 | 0.4563 | 0.8285 | 0.6314 | 0.8648 | +0.252 |
    | KNN | 0.6036 | 0.4684 | 0.8316 | 0.6088 | 0.6297 | +0.026 |
    | RandomForest | 0.5493 | 0.4192 | 0.8260 | 0.6231 | 0.9965 | +0.447 |
    | ExtraTrees | 0.5334 | 0.3946 | 0.7975 | 0.5699 | 0.9961 | +0.463 |

    > **Hallazgo central (doble).** (1) La **Regresión Logística obtiene el F1/MCC más alto**, por encima de todos los boosters. (2) Más importante: su **gap train–CV es ~0.003**, mientras los boosters memorizan el train (XGBoost +0.125, HistGB +0.178, LightGBM +0.252, **RandomForest +0.447** con F1_train≈0.997). El problema es, en gran medida, **linealmente separable** en sus variables clave (contrato, tenure, cargos): la flexibilidad extra de los árboles no compra exactitud, solo **varianza y riesgo de sobreajuste**.

    **Justificación estadística.** El *trade-off* sesgo–varianza explica todo: LogReg tiene **sesgo (moderado) / varianza baja**; los boosters, **sesgo bajo / varianza alta**. Cuando la frontera real es casi lineal, el sesgo de LogReg no penaliza y su baja varianza gana. La columna `gap` lo cuantifica: un modelo que saca 0.997 en train y 0.549 en CV (RandomForest) está **memorizando**, no aprendiendo. RandomForest/ExtraTrees además caen en F1 porque con `class_weight=balanced` y umbral 0.5 sesgan hacia la clase positiva, hinchando recall a costa de precisión (su ROC-AUC sigue siendo decente: ordenan bien, pero el umbral 0.5 no es su punto óptimo).

**Riesgos.** (a) Comparar modelos con umbral fijo 0.5 puede ser injusto para los que producen probabilidades mal calibradas; por eso miramos también ROC-AUC y PR-AUC (independientes del umbral). (b) Dar por ganador a un modelo por 0.001 de F1 sin test estadístico (lo resolvemos en §10).

**Alternativas.** *Stacking*/*voting* de los mejores; *balanced bagging*; remuestreo SMOTE (no usado: los pesos de clase son más limpios y no inventan datos sintéticos que pueden introducir artefactos).

**Ejemplo intuitivo.** Para cortar una hoja de papel en línea recta, una regla (LogReg) funciona mejor que un brazo robótico de 7 ejes (boosting): el robot puede hacer curvas que no necesitas y tiembla más.

**Impacto esperado.** Evidencia de que la complejidad no compra rendimiento aquí — la base empírica de la decisión final.

---

## 10 · Estabilidad estadística (FASE 8)

```python
def ic95(scores):
    m, sem = np.mean(scores), stats.sem(scores)
    return stats.t.interval(0.95, len(scores)-1, loc=m, scale=sem)
# Wilcoxon pareado entre el mejor modelo y cada rival sobre los 15 folds
stats.wilcoxon(f1_folds[best], f1_folds[other])
```

**Explicación.** Para cada modelo reportamos media, desviación e **IC 95 %** del F1 sobre los 15 folds, y contrastamos al mejor (LogReg) contra cada rival con un **test de Wilcoxon pareado**. Los modelos con p>0.05 son **estadísticamente indistinguibles** del mejor.

Resultados **reales** de la ejecución:

| Modelo | F1 | IC95 | p (vs LogReg) | ¿≈ mejor? |
|---|---|---|---|---|
| LogReg | 0.6313 | [0.6230, 0.6396] | — (referencia) | — |
| CatBoost | 0.6293 | [0.6197, 0.6389] | 0.489 | ✅ sí |
| SVM-RBF | 0.6287 | [0.6201, 0.6374] | 0.303 | ✅ sí |
| XGBoost | 0.6246 | [0.6145, 0.6346] | 0.135 | ✅ sí |
| HistGB | 0.6215 | [0.6141, 0.6290] | 0.048 | ❌ (al límite) |
| LightGBM | 0.6125 | [0.6014, 0.6236] | 0.0001 | ❌ no |
| KNN | 0.6036 | [0.5927, 0.6145] | 0.0006 | ❌ no |
| RandomForest | 0.5493 | [0.5380, 0.5606] | 0.0001 | ❌ no |
| ExtraTrees | 0.5334 | [0.5193, 0.5475] | 0.0001 | ❌ no |

> **Lectura:** CatBoost, SVM y XGBoost **no se distinguen** estadísticamente de LogReg (p>0.05). Entre todos ellos, LogReg es el más simple, el más estable (menor gap) y el más interpretable → la REGLA FINAL lo elige sin ambigüedad.

**Justificación estadística.** Los 15 folds son las **mismas particiones** para todos los modelos ⇒ las puntuaciones están *pareadas* y podemos restar fold a fold, eliminando la varianza debida a "qué datos cayeron en cada fold". El **Wilcoxon de rangos con signo** es la versión no paramétrica del t-test pareado: no asume normalidad de las diferencias (apropiado con 15 muestras y posibles colas). Un p>0.05 significa "no hay evidencia de que las distribuciones de F1 difieran": declarar ganador a uno sería sobre-interpretar ruido. El IC 95 % con t de Student (no normal) corrige por el tamaño muestral pequeño (15).

**Riesgos.** (a) **Comparaciones múltiples**: contrastar contra 8 rivales infla la probabilidad de un falso positivo; en rigor se aplicaría corrección de Holm/Bonferroni — lo mencionamos y, como la decisión final favorece la simplicidad, no cambia el veredicto. (b) Wilcoxon con 15 pares tiene potencia limitada: "no significativo" no prueba igualdad, pero sí basta para no preferir lo complejo sin evidencia.

**Alternativas.** Test de McNemar sobre predicciones del test; t-test corregido de Nadeau-Bengio (ajusta la dependencia entre folds); comparación bayesiana de clasificadores (probabilidad de que A>B).

**Ejemplo intuitivo.** Dos corredores con tiempos medios 10.10 s y 10.12 s, pero cuya variación día a día es ±0.15 s: decir "el primero es más rápido" es leer el ruido. Wilcoxon es el juez que dice "empate técnico".

**Impacto esperado.** Blindaje contra falsas mejoras; fundamento estadístico de elegir el modelo simple.

---

## 11 · Optimización de hiperparámetros con Optuna (FASE 7)

```python
study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
# objetivo: F1 medio por CV, reportando valor parcial por fold -> pruning
# boosters: early stopping (HistGB interno; LightGBM con eval_set + callbacks)
```

**Explicación.** Sustituimos `GridSearchCV` (búsqueda exhaustiva, ciega) por **optimización bayesiana** (TPE) con dos aceleradores:
- **Pruning** (poda): si tras los primeros folds una configuración va claramente peor que la mediana histórica, se aborta sin terminar los 5 folds.
- **Early stopping**: los boosters añaden árboles hasta que la métrica de validación deja de mejorar, evitando el sobre-entrenamiento y eligiendo el nº de árboles automáticamente.

Optimizamos tres representantes: **LogReg** (C, penalty), **HistGB** (learning rate, hojas, regularización, early stopping interno) y **LightGBM** (early stopping con `eval_set`). Registramos mejores hiperparámetros, tiempo, nº de trials (40 c/u), trials podados y la curva de evolución.

**Resultado real (tras tuning, RSKF 5×3):**

| Modelo | F1(CV) | F1(train) | gap | mejores hiperparámetros |
|---|---|---|---|---|
| **LogReg** | **0.6322** | 0.6340 | **+0.002** | `C=14.86, penalty=l2` |
| HistGB | 0.6320 | 0.7293 | +0.097 | `lr=0.027, max_leaf_nodes=49, min_samples_leaf=11, l2=5.39` |
| LightGBM | 0.6288 | 0.7292 | +0.100 | `lr=0.014, num_leaves=54, min_child=72, subsample=0.89, colsample=0.61, λ=9.72` |

> El tuning **no cambia el veredicto**: LogReg sigue arriba y mantiene un gap casi nulo (+0.002), mientras los boosters, incluso regularizados, conservan un gap ~0.1 (más memorización). La optimización confirma la robustez del modelo simple en lugar de rescatar a los complejos.

**Justificación estadística.** `GridSearchCV` sufre la **maldición de la dimensionalidad**: una rejilla de 4 hiperparámetros con 5 valores son 625 combinaciones, casi todas malas. El **TPE** (Tree-structured Parzen Estimator) modela `P(hiperparámetros | buen score)` y muestrea donde es probable mejorar — encuentra buenas zonas con muchas menos evaluaciones. El pruning redirige presupuesto desde configuraciones perdedoras hacia prometedoras (similar a *successive halving*). El **early stopping** usa una partición de validación para detener el boosting en el punto de mínima pérdida de validación: implementa directamente el control de la varianza.

**Control del sobreajuste por tuning.** Optimizar es, en sí, un riesgo: con suficientes intentos, alguno "acierta" por azar en la CV. Lo controlamos (a) evaluando siempre por CV (nunca en test), (b) limitando los trials, (c) verificando el **gap train–CV** del modelo tuneado, y (d) confirmando el rendimiento con **nested CV** (§13), que es inmune a este sesgo.

**Riesgos.** (a) Espacios de búsqueda mal acotados (demasiado amplios ⇒ ineficiente; demasiado estrechos ⇒ se pierde el óptimo). (b) `eval_set` para early stopping debe transformarse con el preprocesador ajustado **solo en el fold-train**, o se reintroduce leakage — lo hacemos explícitamente. (c) Pruning demasiado agresivo puede matar configuraciones que arrancan lento; por eso `n_warmup_steps`.

**Alternativas.** `HalvingRandomSearchCV` (sklearn), Hyperband, *Bayesian Optimization* con procesos gaussianos (scikit-optimize), búsqueda aleatoria pura (sorprendentemente competitiva).

**Ejemplo intuitivo.** Grid search es revisar TODOS los estantes del supermercado por orden. La optimización bayesiana es preguntar "¿dónde suele estar el café?" y caminar directo a esa zona; el pruning es no terminar de recorrer un pasillo donde ya ves que no hay nada.

**Impacto esperado.** Hiperparámetros competitivos en una fracción del tiempo, con sobreajuste de tuning bajo control.

---

## 12 · Diagnóstico de generalización (FASE 10)

```python
learning_curve(pipe, X_train, y_train, cv=cv_skf, scoring="f1",
               train_sizes=np.linspace(0.1,1.0,6))
validation_curve(base_lr, ..., param_name="clf__C", param_range=np.logspace(-3,2,8))
calibration_curve(y, proba, n_bins=10);  roc_curve(...);  precision_recall_curve(...)
```

**Explicación.** Cinco diagnósticos:
- **Curva de aprendizaje:** F1 de train y CV frente al tamaño de muestra. Si convergen y se aplanan ⇒ más datos no ayudarían y no hay overfitting grave.
- **Curva de validación (C de LogReg):** muestra el sub-ajuste (C muy bajo, sobre-regularizado) y el sobre-ajuste (C muy alto) y verifica que el C* de Optuna cae en la meseta óptima.
- **Calibración:** ¿una probabilidad de 0.7 corresponde a un 70 % real de churn? Crucial si el negocio usa la probabilidad, no solo la clase.
- **ROC y PR:** rendimiento a todos los umbrales; PR es más informativa que ROC bajo desbalance.

**Justificación estadística.** El **gap train–CV** es el estimador directo del sobreajuste (varianza): en LogReg es pequeño (modelo de baja varianza), en boosters mayor. Una curva de aprendizaje con train alto y CV estancado y separado indica varianza (overfitting); ambas bajas y juntas indica sesgo (underfitting). La **curva PR** se prefiere a la ROC con clases desbalanceadas porque la ROC puede parecer optimista: el eje FPR se diluye con muchos negativos, mientras que la precisión "siente" cada falso positivo.

**Riesgos.** Leer estas curvas sobre el test sería leakage; por eso se calculan con probabilidades **out-of-fold** sobre train. Una buena ROC-AUC con mala calibración puede engañar si se interpreta la probabilidad literalmente.

**Alternativas.** *Calibración* posterior (Platt/Isotónica con `CalibratedClassifierCV`), curvas de *lift*/ganancia (muy usadas en marketing de retención), Brier score.

**Ejemplo intuitivo.** La calibración es como un pronóstico del tiempo: si dice "70 % de lluvia" 10 veces, debería llover ~7 de esas. Un modelo bien calibrado permite confiar en sus porcentajes para decidir presupuesto de retención.

**Impacto esperado.** Confirmar que el modelo final no sobre/infra-ajusta y que sus probabilidades son utilizables para decisiones de negocio.

---

## 13 · Nested Cross-Validation (FASE 3)

```python
outer = StratifiedKFold(5, ...)
for tr, va in outer.split(X_train, y_train):     # bucle externo = evaluación
    study.optimize(objective(X[tr], y[tr]), n_trials=20)   # bucle interno = tuning
    pipe = build(study.best_params).fit(X[tr], y[tr])
    scores.append(f1_score(y[va], pipe.predict(X[va])))
```

**Explicación.** Estimador **insesgado** del error de generalización: en cada uno de los 5 folds externos, se re-optimizan los hiperparámetros usando **solo** el train de ese fold y se evalúa en su val externo, que el tuning nunca vio. Lo aplicamos a los dos finalistas (LogReg y HistGB).

**Resultado real:** `Nested CV LogReg: F1 = 0.6297 ± 0.0120` · `Nested CV HistGB: F1 = 0.6296 ± 0.0137`. Las dos estimaciones insesgadas son **prácticamente idénticas** (diferencia 0.0001, muy por debajo de su desviación): la complejidad de HistGB no aporta generalización alguna sobre LogReg.

**Justificación estadística.** El score de §11 es optimista porque los hiperparámetros se eligieron mirando esa misma CV (es "el mejor de muchos" sobre esos datos). La CV anidada rompe esa dependencia: tuning y evaluación usan datos disjuntos. Es el estándar de oro para responder "¿cuánto rendirá *el procedimiento completo* (incluido el tuning) en datos nuevos?". Que LogReg y HistGB den un F1 anidado prácticamente idéntico confirma, sin sesgo, el empate técnico.

**Riesgos.** Coste computacional alto (tuning × folds externos). Con presupuesto reducido por fold (20 trials) la estimación gana algo de varianza, pero sigue siendo insesgada y suficiente para comparar.

**Alternativas.** *Repeated* nested CV (más estable, más caro); un único *validation set* separado para tuning (más barato, más ruidoso).

**Ejemplo intuitivo.** Es como evaluar a un chef haciéndolo cocinar para 5 jurados distintos, y en cada caso dejándole probar y ajustar solo con ingredientes que ese jurado no probará. Nadie califica un plato que el chef ya optimizó para ese paladar.

**Impacto esperado.** La cifra de generalización en la que de verdad se puede confiar; respalda la decisión final ante cualquier auditoría.

---

## 14 · Interpretabilidad (FASE 11)

```python
coefs = lr_fit.named_steps["clf"].coef_[0]                      # log-odds (signo + magnitud)
permutation_importance(lr_fit, X_test, y_test, scoring="f1")    # impacto real, model-agnostic
shap.LinearExplainer(lr_fit.named_steps["clf"], Xtr_t)         # contribución por cliente
```

**Explicación.** Tres lentes complementarias y, sobre todo, **interpretadas** (no solo dibujadas):
- **Coeficientes (log-odds):** signo y magnitud del efecto de cada feature *manteniendo las demás constantes*. Positivo ⇒ empuja churn; negativo ⇒ protege.
- **Permutation importance:** cuánto cae el F1 al barajar una variable; mide su contribución **real** a la métrica, sin depender del modelo.
- **SHAP:** descompone cada predicción individual en aportes por feature (valores de Shapley, teoría de juegos), permitiendo explicar *clientes concretos*.

**Hallazgos convergentes (las tres coinciden):**
- **Empujan churn:** contrato *mes a mes*, *fibra óptica*, *cheque electrónico*, *paperless billing*, cargos mensuales altos.
- **Protegen:** mayor *tenure*, contratos *de 1 y 2 años*, *soporte técnico* y *seguridad online*, no tener internet.
- **Irrelevantes:** *gender*, *phone service* (coherente con su MI ≈ 0).

**Justificación estadística.** Que tres métodos de fundamentos distintos (paramétrico, por permutación, teoría de juegos) **converjan** es la mejor señal de robustez de la explicación. Los **valores SHAP** tienen una propiedad de *aditividad*: suman exactamente la diferencia entre la predicción y la media, lo que los hace consistentes y comparables entre clientes. La **permutation importance** evita el sesgo de las importancias internas de árbol (que inflan variables de alta cardinalidad). Atención a la **multicolinealidad** `total≈monthly×tenure`: reparte crédito entre variables correlacionadas, así que la magnitud individual de cargos debe leerse con cautela (permutar una con su gemela presente subestima su efecto).

**Riesgos.** (a) Confundir importancia con causalidad: SHAP explica el *modelo*, no el mundo; que el modelo use "fibra" no prueba que cambiar a DSL retenga al cliente. (b) Con features correlacionadas, la permutación crea combinaciones imposibles (cliente con tenure alto y total bajo) que pueden distorsionar; SHAP con dependencias mitiga parte de esto.

**Alternativas.** SHAP `TreeExplainer` para modelos de árbol, *Partial Dependence*/ICE, *LIME*, *Accumulated Local Effects* (ALE, robusto a correlación).

**Ejemplo intuitivo.** SHAP es como una factura detallada de por qué un cliente concreto tiene 80 % de probabilidad de fuga: "+25 % por contrato mes a mes, +15 % por fibra, −10 % por 3 años de antigüedad…". Eso permite acciones de retención **personalizadas**.

**Impacto esperado.** El modelo deja de ser una caja negra: el negocio entiende qué mover (incentivar contratos largos, revisar la oferta de fibra, facilitar pagos automáticos) y puede explicar cada decisión.

---

## 15 · Métricas, umbral y selección final (FASES 9 y 12)

```python
# tabla comparativa final (default vs tuned): F1, MCC, PR-AUC, ROC-AUC, Recall, Precision,
#                                              gap_train_cv, tiempos de entrenamiento e inferencia
# decisión programática: empate técnico (F1 >= top - tol) -> elegir el más simple/estable
# umbral óptimo por F1 con probabilidades out-of-fold (no se toca test)
prec, rec, thr = precision_recall_curve(y_train, oof);  thr_opt = thr[argmax(F1)]
```

**Explicación.** Construimos la **tabla comparativa profesional** con todas las métricas + tiempos + estabilidad + ranking, para modelos *default* y *tuned*. Métrica primaria **F1**; soporte con **PR-AUC** y **MCC**. Luego optimizamos el **umbral de decisión** (el 0.5 por defecto rara vez es óptimo con clases desbalanceadas) maximizando F1 con probabilidades *out-of-fold*.

**Por qué F1/MCC y no Recall ni Accuracy:**
- **Accuracy** premia acertar la clase mayoritaria (engañosa, lo probó el baseline).
- **Recall puro** se maximiza prediciendo "todos churn" (inútil; era el error del notebook original).
- **F1** equilibra precisión y recall (media armónica): penaliza tanto perder fugas como molestar a clientes fieles.
- **MCC** (coef. de correlación de Matthews) usa las cuatro casillas de la matriz de confusión y es la métrica más honesta bajo desbalance: solo es alta si el modelo acierta en ambas clases.
- **PR-AUC** resume el rendimiento a todos los umbrales centrándose en la clase positiva.

**Decisión final (REGLA FINAL aplicada).** El top de modelos es **estadísticamente indistinguible** (§10, Wilcoxon p>0.05) y su **nested CV** coincide (§13). Entre iguales, se elige el **más simple, estable, interpretable y barato**: la **Regresión Logística**, con su umbral calibrado. No se elige por tener el F1 más alto por milésimas, sino por **menor riesgo de sobreajuste y máxima robustez** — exactamente lo que pide la jerarquía del proyecto.

**Justificación estadística.** La **navaja de Occam** formalizada: a igualdad de error de validación, el modelo de menor complejidad tiene menor varianza y, por tanto, mayor probabilidad de mantener su rendimiento en datos nuevos. La optimización de umbral es un problema 1-D: barremos todos los umbrales y elegimos el de máximo F1 *out-of-fold*, lo que evita ajustar el umbral al test.

**Riesgos.** (a) El umbral óptimo en train puede no ser exactamente óptimo en test (lo verificamos reportando ambos umbrales en §16). (b) Optimizar F1 fija implícitamente un coste igual a falsos positivos y negativos; si el negocio tiene costes asimétricos, el umbral debería derivarse de la matriz de costes, no de F1.

**Alternativas.** Umbral por máximo MCC, por *F-beta* (β>1 prioriza recall), o por mínimo coste esperado dada una matriz de costes de negocio.

**Ejemplo intuitivo.** Mover el umbral es ajustar la sensibilidad de una alarma de humo: muy sensible (umbral bajo) suena con el vapor de la ducha (falsos positivos); poco sensible (umbral alto) no avisa hasta que hay llamas (falsos negativos). El umbral óptimo equilibra ambos según lo que te cueste cada error.

**Impacto esperado.** Un ganador defendible ante cualquier revisor, con un umbral ajustable a la economía de la retención.

---

## 16 · Evaluación final en el test aislado

```python
win_pipe.fit(X_train, y_train)                       # entrena en todo el train
proba_test = win_pipe.predict_proba(X_test)[:,1]     # PRIMER contacto con el test
pred_opt = (proba_test >= thr_opt).astype(int)
metricas_completas(y_test, pred_opt, proba_test)
```

**Explicación.** Único contacto con el test. Entrenamos el pipeline ganador (**LogReg** tuneado, `C=14.86`) en todo el train, aplicamos el umbral elegido (**0.614**) y medimos. **Estas son las métricas honestas de generalización** — las que el cliente puede esperar en producción.

**Resultado real en el test aislado (n=1407):**

| Métrica | umbral 0.5 | **umbral 0.614** |
|---|---|---|
| Accuracy | 0.7441 | **0.7790** |
| Balanced Acc | 0.7609 | 0.7514 |
| Precision | 0.5120 | **0.5692** |
| Recall | 0.7968 | 0.6925 |
| **F1** | 0.6234 | **0.6248** |
| **MCC** | 0.4681 | **0.4748** |
| ROC-AUC | 0.8446 | 0.8446 |
| PR-AUC | 0.6517 | 0.6517 |

`classification_report` (umbral 0.614): clase **Evade** → precision 0.57, recall 0.69, F1 0.62 (soporte 374); clase **Permanece** → precision 0.88, recall 0.81, F1 0.84 (soporte 1033).

> **Coherencia total:** F1 test = **0.625** ≈ F1 CV (0.632) ≈ F1 nested CV (0.630). Que las tres cifras coincidan (sin que el test sorprenda a la baja) es la **prueba empírica de que no hay leakage** y de que el modelo generaliza. El umbral 0.614 sube accuracy, precisión, F1 y MCC a costa de algo de recall — un intercambio ajustable según el coste de negocio.

**Justificación estadística.** El test, aislado desde §6 y nunca usado para decidir nada, es una muestra i.i.d. de la población objetivo; el rendimiento sobre él es un **estimador insesgado** del rendimiento poblacional (con su propio error de muestreo, acotable por el tamaño n=1407). Que coincida con la CV y la nested CV cierra el círculo: sin sorpresas ⇒ sin leakage.

**Riesgos.** Un solo test tiene varianza; idealmente se reporta con intervalo (bootstrap sobre el test). Tras "ver" el test, cualquier ajuste posterior lo contamina: el test ya se quemó.

**Alternativas.** Bootstrap del test para IC de las métricas; validación temporal *out-of-time* si hubiera fechas; *backtesting* en producción con A/B.

**Ejemplo intuitivo.** Es el examen final a libro cerrado: si tu nota se parece a la de los simulacros (CV), estudiaste de verdad; si es mucho peor, hacías trampa en los simulacros (leakage).

**Impacto esperado.** La cifra final, defendible y reproducible, sobre la que el negocio toma decisiones.

---

## 17 · Exportación del modelo (MLOps)

```python
artefacto = {"pipeline": win_pipe, "threshold": thr_opt, "features_input": list(X.columns),
             "target": TARGET, "metricas_test": {...}, "modelo": win_name,
             "sklearn_version": sklearn.__version__, "random_state": RANDOM_STATE}
joblib.dump(artefacto, "mejores_practicas/modelo_final.pkl")
```

**Explicación.** Guardamos el **pipeline completo** (preprocesador + modelo en un solo objeto) más el umbral y metadatos. En producción, scorear es `pipeline.predict_proba(nuevos_datos)` seguido de comparar con `threshold`: **sin pasos manuales** que puedan reintroducir leakage o divergencia train/serve.

**Justificación estadística / de ingeniería.** El *training–serving skew* (que el preprocesamiento en producción difiera del de entrenamiento) es una causa habitual de degradación silenciosa. Empaquetar todo en un `Pipeline` lo elimina por construcción: las mismas medias, escalas y categorías aprendidas en train se aplican en inferencia. Guardar la versión de sklearn y la semilla permite reproducir y auditar.

**Riesgos.** (a) Compatibilidad de versiones al cargar el pickle (mitigado guardando la versión). (b) *Data drift*: el modelo asume que la población futura se parece a la de entrenamiento; requiere monitorización. (c) Seguridad: un pickle ejecuta código al cargarse; cargar solo artefactos de confianza.

**Alternativas.** ONNX/PMML (portabilidad entre lenguajes), `skops` (serialización más segura que pickle), registro de modelos (MLflow) con versionado y métricas.

**Ejemplo intuitivo.** Es como entregar una máquina de café ya programada en lugar de una bolsa de granos con instrucciones: el receptor obtiene exactamente el mismo café sin posibilidad de equivocarse en la receta.

**Impacto esperado.** Despliegue reproducible, sin skew y auditable; el umbral se puede ajustar al coste de negocio sin reentrenar.

---

## 18 · Conclusiones

1. **Generalización por encima de la métrica.** LogReg, HistGB, LightGBM y CatBoost son **estadísticamente indistinguibles** en F1 (Wilcoxon p>0.05) y su nested CV coincide. La **REGLA FINAL** ordena elegir el simple: **Regresión Logística** con umbral calibrado — menor varianza, menor riesgo de overfitting, interpretable y barata.
2. **Cero data leakage.** Todo el preprocesamiento vive en pipelines; se ajusta solo con el fold-train; el test permaneció ciego hasta §16; el umbral salió de probabilidades out-of-fold. Se corrigen L1 (scaler sobre todo el dataset) y L2 (selección de features mirando test) del notebook original.
3. **Robustez estadística.** RSKF 5×3 + IC 95 % + Wilcoxon pareado + nested CV: ningún ganador por diferencias marginales.
4. **Interpretabilidad triangulada.** Coeficientes, permutation importance y SHAP coinciden en los drivers: contrato mes a mes, fibra, baja antigüedad y cheque electrónico empujan churn; contratos largos, antigüedad y soporte protegen.
5. **Límite teórico reconocido.** El ruido irreducible (perfiles idénticos con churn opuesto) fija un techo de Bayes; un F1 honesto ~0.62–0.64 es coherente, no una limitación del modelo.

### Tabla: notebook original vs. reconstrucción

| Dimensión | Notebook original | `mejores_practicas` |
|---|---|---|
| Preprocesamiento | `fit_transform` sobre todo X (leakage L1) | Pipelines cerrados, fit por fold |
| Selección de features | mirando el test (leakage L2) | por CV / sin selección sesgada |
| Métrica de selección | Recall puro (degenerada) | F1 / PR-AUC / MCC + negocio |
| Modelos | Dummy, RF, XGB | 9 familias + 3 tuneadas |
| Tuning | GridSearch exhaustivo | Optuna (TPE) + pruning + early stopping |
| Comparación | 1 partición, sin incertidumbre | RSKF + IC95 + Wilcoxon + nested CV |
| Variable redundante | `charges_day` incluida | eliminada (probado = monthly/30) |
| Interpretabilidad | feature_importances_ | coef + permutation + SHAP |
| Reproducibilidad | `device='cuda'` fijo, celdas frágiles | semilla global, CPU, pipeline serializado |
| Modelo final | Random Forest (por recall) | Regresión Logística (por generalización) |

### Recomendaciones de negocio
- **Retención dirigida** a clientes mes-a-mes con baja antigüedad y fibra: máximo riesgo y máximo retorno de la acción.
- **Incentivar contratos de 1–2 años** y **pagos automáticos**: son los factores protectores más fuertes y accionables.
- **Revisar la propuesta de valor de la fibra óptica**: su tasa de fuga (41.9 %) sugiere un problema de precio/calidad percibida.
- **Usar la probabilidad calibrada**, no solo la clase, y fijar el umbral según el coste de un falso negativo (cliente perdido) frente a un falso positivo (incentivo regalado).

---

*Documento generado como material de estudio. Reproducible end-to-end ejecutando `mejores_practicas.ipynb` con la semilla fijada.*
