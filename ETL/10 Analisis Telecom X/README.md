# Análisis de Evasión de Clientes - TelecomX

## Descripción del Proyecto

Este proyecto analiza el fenómeno de churn (evasión de clientes) en la empresa de telecomunicaciones TelecomX. El objetivo principal es identificar los factores que influyen en la decisión de los clientes de abandonar el servicio y proporcionar recomendaciones estratégicas basadas en datos.

## Contenido

- `evasion_clientes.ipynb`: Notebook principal con el análisis completo, incluyendo limpieza de datos, análisis exploratorio, pruebas estadísticas y conclusiones.

## Metodología

### 1. Extracción y Limpieza de Datos
- Importación de datos desde archivo JSON
- Normalización de estructura de datos anidados
- Tratamiento de valores nulos y duplicados
- Estandarización de variables categóricas
- Conversión de tipos de datos

### 2. Análisis Exploratorio
- Estadística descriptiva
- Visualizaciones de distribuciones
- Análisis de proporciones
- Matriz de correlación

### 3. Análisis Estadístico Inferencial
- Test de Shapiro-Wilk para verificar normalidad
- Test de Mann-Whitney U para comparación de medianas
- Test de Chi-cuadrado para asociación entre variables categóricas

## Principales Hallazgos

1. **Cargos mensuales**: Los clientes que abandonan pagan significativamente más que los clientes actuales.

2. **Antigüedad**: Los clientes nuevos tienen mayor probabilidad de abandono que los clientes con mayor tiempo en la empresa.

3. **Tipo de contrato**: Existe una fuerte relación entre el tipo de contrato y la tasa de churn:
   - Mes a mes: 42.7%
   - Anual: 11.3%
   - Bianual: 2.8%

## Tecnologías Utilizadas

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- requests

## Estructura del Análisis

El notebook está organizado en las siguientes secciones:

1. Bibliotecas
2. Extracción de datos
3. Transformación de datos
4. Análisis descriptivo
5. Análisis inferencial
6. Informe final con recomendaciones

## Resultados

La tasa de churn actual es del 26.6%. El análisis identifica un perfil de cliente de alto riesgo caracterizado por ser nuevo, tener cargos mensuales elevados y contar con contrato mes a mes. Las recomendaciones se enfocan en mejorar la percepción de valor, fortalecer el onboarding de nuevos clientes e incentivar contratos de largo plazo.

## Autor

Proyecto desarrollado como parte del Postgrado en Ciencia de Datos.