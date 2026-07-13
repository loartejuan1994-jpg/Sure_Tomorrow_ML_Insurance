# Sure Tomorrow — Machine Learning para Aseguradora

>Proyecto de machine learning aplicado a la aseguradora ficticia Sure Tomorrow, que combina técnicas de álgebra lineal y ML para resolver cuatro tareas de >negocio: búsqueda de clientes similares, clasificación, regresión y protección de datos personales mediante ofuscación matricial.
---
# Contenido del proyecto

> Tarea	Descripción	Técnica
Tarea 1	Búsqueda de clientes similares	KNN — `NearestNeighbors`
Tarea 2	Predicción de beneficios de seguro	KNN — `KNeighborsClassifier`
Tarea 3	Predicción del número de prestaciones	Regresión lineal (implementación propia)
Tarea 4	Protección de datos personales	Ofuscación matricial con álgebra lineal
---
> Dataset
Archivo: `insurance_us.csv`
Registros: 5,000 clientes
Características: `gender`, `age`, `income`, `family_members`
Variable objetivo: `insurance_benefits` — número de prestaciones de seguro recibidas en los últimos 5 años
---
> Resultados principales
Tarea 1 — Clientes similares (KNN)
Sin escalado: `income` domina el cálculo de distancia, ignorando las demás variables
Con escalado MaxAbs: vecinos genuinamente similares en todas las dimensiones
Euclidiana vs. Manhattan: impacto mínimo una vez aplicado el escalado
Tarea 2 — Clasificación KNN
Modelo	F1-score
Modelo Dummy (línea base)	0.20
KNN sin escalar (mejor k=1)	0.65
KNN con escalado MaxAbs (k=3 y k=5)	0.94
El escalado es condición necesaria para obtener resultados útiles en KNN.
Tarea 3 — Regresión lineal
Métrica	Datos originales	Datos escalados
RMSE	0.34	0.34
R²	0.43	0.43
Implementación propia de regresión lineal usando la ecuación normal:
$$w = (X^TX)^{-1}X^Ty$$
El escalado no afecta los resultados porque los pesos `w` se ajustan automáticamente para compensar cualquier cambio de escala.
Tarea 4 — Ofuscación de datos
Se demostró analítica y computacionalmente que multiplicar $X$ por una matriz invertible $P$ no altera las predicciones del modelo:
$$\hat{y}_P = (XP)w_P = Xw = \hat{y}$$
	Datos originales	Datos ofuscados (XP)
RMSE	0.34	0.34
R²	0.43	0.43
Diferencia máxima en predicciones	—	7.58 × 10⁻⁸
La diferencia es atribuible únicamente a precisión numérica de punto flotante.
---
# Tecnologías utilizadas

>Python 3
NumPy
Pandas
Scikit-learn (`KNeighborsClassifier`, `NearestNeighbors`, `MaxAbsScaler`)
Seaborn / Matplotlib
Jupyter Notebook
---
# Estructura del repositorio
```
Sure_Tomorrow_ML_Insurance/
│
├── Sure_Tomorrow_ML_Insurance.ipynb   # Notebook principal con todo el análisis
├── insurance_us.csv                   # Dataset (si está disponible localmente)
└── README.md                          # Este archivo
```
---
Cómo ejecutar
Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/Sure_Tomorrow_ML_Insurance.git
```
Instala las dependencias:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```
Abre el notebook:
```bash
jupyter notebook Sure_Tomorrow_ML_Insurance.ipynb
```
Actualiza la ruta del dataset en la celda de carga:
```python
df = pd.read_csv("ruta/a/insurance_us.csv")
```
---
# Conclusiones

> El proyecto demuestra que el escalado previo es indispensable para modelos basados en distancias (KNN), mientras que es irrelevante para la regresión lineal. La ofuscación matricial es un método válido y matemáticamente probado para proteger datos personales sin sacrificar la calidad predictiva del modelo.
---
Autor
Juan Carlos Buri Loarte  

