# Bajo peso al nacer en Colombia 2005-2024 — traducción a Python (v2)

Traducción a Python de los 4 do-files de Stata de la carpeta `Code/stata/v2`
(análisis de bajo peso al nacer, BPN, con microdatos de nacimientos de las
Estadísticas Vitales del DANE). Los scripts espejan la estructura de los
do-files y producen las mismas salidas (xlsx por objetivo, gráficos G1–G14 y
logs), leyendo los datos con la librería `dane-data`.

## Requisitos

- Python del venv `datascience`: `~/.venvs/datascience/bin/python`
  (pandas, numpy, statsmodels, scipy, openpyxl, matplotlib).
- `dane-data` con los archivos de nacimientos 2005-2024 ya descargados
  (`dd.eevv.available()`).
- Insumo `3.0 Datos/altitud_municipios.csv` (altitud por municipio).

## Ejecución

```
cd "…/3. Metodologia/Code/python/v2"
~/.venvs/datascience/bin/python correr.py        # los 4 objetivos
~/.venvs/datascience/bin/python correr.py 2 4    # solo los objetivos 2 y 4
```

`correr.py` carga las celdas una vez y, por objetivo, escribe el log, el xlsx
y los png. La notebook `sintesis.ipynb` recalcula todo con los mismos módulos
y muestra tablas y gráficos junto al texto del análisis.

## Diseño modular

Cada módulo `objN_*.py` expone tres funciones:

- `calcular(celdas) -> dict`: las tablas del objetivo. Cada clave es una hoja
  del xlsx; las claves que empiezan con `_` son intermedias (para gráficos o
  el log) y no se exportan.
- `imprimir(r)`: el resumen que va a la consola y al log.
- `graficos(r) -> dict`: las figuras de matplotlib, sin guardarlas; la clave
  es el nombre del png.

Ningún módulo escribe archivos por su cuenta: eso lo hace `correr.py`.

## Scripts

| Script | Equivalente Stata | Contenido |
|---|---|---|
| `correr.py` | — | corre los objetivos y guarda log, xlsx y png |
| `sintesis.ipynb` | — | síntesis del análisis: recalcula y muestra tablas y gráficos |
| `comun_bpn.py` | Parte A/B de `01_Base_y_Objetivo1.do` | rutas, etiquetas, carga con `dd.eevv`, celdas de conteos, imputación de altitud, caché, helpers (Wilson, APC Poisson, Excel, gráficos, log) |
| `obj1_evolucion.py` | `01_Base_y_Objetivo1.do` C | prevalencia nacional (cruda y estandarizada), APC, departamentos, calidad del registro del peso, sexo, distribución del peso, G1 y G12–G14 (mapa de calor por departamento, % BPN por sexo, distribución del peso) |
| `obj2_desigualdades.py` | `02_Objetivo2_Desigualdades_RT_DT.do` | RP/DP por quinquenio y año frente a la referencia elegible, Theil/MLD, PAR/PAF, extremos, G7/G8 |
| `obj3_educacion.py` | `03_Objetivo3_Educacion_Regimen.do` | prevalencias por educación y régimen, Poisson M0–M5 con errores cluster dpto, RII/SII, G9/G10 |
| `obj4_concentracion.py` | `04_Objetivo4_Indice_Concentracion.do` | estandarización indirecta por altitud y edad, índices C/W/E por regresión conveniente (Kakwani), 10 variantes, bootstrap por departamento, trienios 2008-2010 vs 2022-2024 con bootstrap pareado, curvas G3–G6/G11, G2 |

## Datos y caché

- La lectura de microdatos usa `dd.eevv.load(anio, tipo="nacimientos",
  columns=["PESO_NAC"])` (el default `core` no trae el peso).
- Todo se trabaja sobre **celdas** por (anio, dpto_res, cod_mpio, educ4,
  regimen, edad6) con altitud del municipio: los conteos (nacidos, con peso,
  BPN, BPN a término, régimen, educación) replican exactamente las
  regresiones con datos individuales (nota técnica del do-file 03) y
  aceleran los Poisson ~10 millones de registros.
- La caché de celdas está en `3.0 Datos/python_v2/procesados/celdas/`
  (un parquet + un JSON por año). La huella incluye el stat del archivo del
  DANE, la versión de `dane-data` y los bytes de `comun_bpn.py`: si cambia
  el módulo, se recalculan todas las celdas. El registro de cargas queda en
  `procesados/control_carga.csv`. Con `refrescar=True` se fuerza el recálculo.
- La imputación de altitud (media del departamento ponderada por nacimientos)
  se hace sobre la serie completa, como el `bysort dpto_res: egen mean` del
  do-file, después de unir los años.

## Salidas

`3.0 Datos/python_v2/`:

- `Objetivo1_Evolucion_BPN.xlsx`, `Objetivo2_Desigualdades_RT_DT.xlsx`,
  `Objetivo3_Educacion_Regimen_BPN.xlsx`,
  `Objetivo4_Indice_Concentracion_BPN.xlsx`
- `G1`…`G14` (PNG, 300 dpi) y `log_01…log_04` (log de cada corrida,
  mismo contenido que la consola).

## Notas de replicación

- Las semillas del bootstrap replican las del do-file (anual `20241+anio+100·v`,
  trienios `20241+1000·v+p`, diferencias `20241+5000+v`), pero los replicados
  no coinciden numéricamente con Stata: el generador de números aleatorios es
  distinto (NumPy PCG64). Con las mismas semillas los resultados son
  reproducibles entre corridas de Python.
- La estandarización indirecta por altitud y edad del objetivo 4 **no replica
  el do-file**. El do-file usa la tasa del grupo (año, altitud, edad) incluido
  el propio departamento; como Bogotá es ~2/3 de los nacidos a ≥2.500 m,
  quedaba comparada consigo misma. Python usa un Poisson con altitud, edad,
  año y efectos fijos de departamento (O'Donnell et al. 2008, cap. 5) y predice
  los casos esperados solo con altitud y edad. Los coeficientes van en la hoja
  `Efecto_altitud_edad`.
- El régimen de afiliación solo es comparable desde 2008 (objetivos 3 y 4).
- La validación del ordenamiento contra NBI/IPM (hoja `Validacion_proxy`) se
  omite con un aviso si no existe la plantilla
  `3.0 Datos/nbi_ipm_departamentos.csv`.
- La variante anual "Sin Bogotá" no tiene bootstrap en el gráfico G2 (así lo
  hace el do-file); el bootstrap completo está en los trienios y la
  diferencia pareada.
