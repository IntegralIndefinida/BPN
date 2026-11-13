# Bajo peso al nacer en Colombia 2005-2024 — análisis en Python

Análisis de bajo peso al nacer (BPN) con microdatos de nacimientos de las
Estadísticas Vitales del DANE. Los scripts producen las salidas del análisis
(xlsx por objetivo, gráficos G1–G15 y logs), leyendo los datos con `dane-data`.

## Requisitos

- Python del venv `datascience`: `~/.venvs/datascience/bin/python`
  (pandas, numpy, statsmodels, scipy, openpyxl, matplotlib; las versiones
  exactas están en `requirements.txt`).
- `dane-data` con los archivos de nacimientos 2005-2024 ya descargados
  (`dd.eevv.available()`), instalado editable desde su repositorio.
- Insumo `3.0 Datos/altitud_municipios.csv` (altitud por municipio).

## Ejecución

```
cd "…/3. Metodologia/Code/python/v2"
~/.venvs/datascience/bin/python test_bpn.py      # pruebas de la lógica central
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

| Script | Contenido |
|---|---|---|
| `correr.py` | corre los objetivos y guarda log, xlsx y png |
| `sintesis.ipynb` | síntesis del análisis: recalcula y muestra tablas y gráficos |
| `comun_bpn.py` | rutas, etiquetas, carga con `dd.eevv`, celdas de conteos, imputación de altitud, caché, helpers (Wilson, APC Poisson, Excel, gráficos, log) |
| `obj1_evolucion.py` | prevalencia nacional (cruda y estandarizada), APC (nacional y por subconjunto de registros: con residencia conocida, sin Chocó, La Guajira y Cesar), departamentos, calidad del registro del peso, flujo de registros según la fuente del certificado (hojas `Flujo_registros` y `RNEC_por_ocurrencia`) y origen del faltante (hoja `Faltante_peso_origen`), sexo, distribución del peso, descomposición de la tendencia por semanas de gestación (Kitagawa), corrimiento hacia 28-37 semanas (pretérmino y semana 37 por separado, por tipo de parto, paridad, educación, régimen y departamento, prueba con el peso y defunciones fetales), G1 y G12–G15 |
| `obj2_desigualdades.py` | Theil/MLD, RP/DP frente al promedio nacional y RP/DP/PAF frente a una referencia fija (grupo de cinco departamentos de baja prevalencia, hojas `_fija`; las principales son las ajustadas por altitud y edad, `_aj_fija`), por quinquenio y año, con IC 95 %; la referencia móvil (mínimo elegible de cada periodo) queda como sensibilidad; sensibilidad del PAF a la referencia y validación del grupo (hojas `PAF_referencias` y `Referencia_fija`); G7/G8 |
| `obj3_educacion.py` | prevalencias por educación y régimen, Poisson M0–M5 con errores cluster dpto (principal: M4, con altitud y efectos fijos de departamento; M5 = M4 con BPN a término), RII/SII con ese ajuste por quinquenio y cotas extremas por educación faltante; sensibilidad del gradiente (sin régimen, educación faltante como categoría, paridad, etnia y área), tendencia del RII con bootstrap wild por departamento y controles con la semana 37, % sin dato por año, G9/G10 |
| `obj4_concentracion.py` | estandarización indirecta por altitud y edad, índices C/W/E por regresión conveniente (Kakwani), 11 variantes (una sin Chocó, La Guajira y Cesar), bootstrap por departamento, trienios 2008-2010 vs 2022-2024 y vs 2019-2021 (antes del quiebre de registro de 2022) con bootstrap pareado, curvas G3–G6/G11, G2 |

## Datos y caché

- La lectura de microdatos usa `dd.eevv.load(anio, tipo="nacimientos",
  columns=["PESO_NAC"])` (el default `core` no trae el peso).
- Todo se trabaja sobre **celdas** por (anio, dpto_res, cod_mpio, educ4,
  regimen, edad6, paridad3, etnia3, area2) con altitud del municipio: los
  conteos (nacidos, con peso, BPN, BPN a término, semana 37, régimen,
  educación) replican exactamente las
  regresiones con datos individuales (nota técnica del objetivo 3) y
  aceleran los Poisson ~10 millones de registros.
- La caché de celdas está en `3.0 Datos/python_v2/procesados/celdas/`
  (un parquet + un JSON por año). La huella incluye el stat del archivo del
  DANE, la versión de `dane-data` y los bytes de `comun_bpn.py`: si cambia
  el módulo, se recalculan todas las celdas. El registro de cargas queda en
  `procesados/control_carga.csv`. Con `refrescar=True` se fuerza el recálculo.
- La imputación de altitud (media del departamento ponderada por nacimientos)
  se hace sobre la serie completa, después de unir los años.

## Salidas

`3.0 Datos/python_v2/`:

- `Objetivo1_Evolucion_BPN.xlsx`, `Objetivo2_Desigualdades_RT_DT.xlsx`,
  `Objetivo3_Educacion_Regimen_BPN.xlsx`,
  `Objetivo4_Indice_Concentracion_BPN.xlsx`
- `G1`…`G15` (PNG, 300 dpi) y `log_01…log_04` (log de cada corrida,
  mismo contenido que la consola).

## Notas metodológicas

- Las semillas del bootstrap (anual `20241+anio+100·v`, trienios
  `20241+1000·v+p`, diferencias `20241+5000+v`) hacen que los resultados sean
  reproducibles entre corridas de Python.
- La estandarización indirecta por altitud y edad del objetivo 4 usa un Poisson
  con altitud, edad,
  año y efectos fijos de departamento (O'Donnell et al. 2008, cap. 5) y predice
  los casos esperados solo con altitud y edad. Los coeficientes van en la hoja
  `Efecto_altitud_edad`.
- Los intervalos de confianza y los valores p del bootstrap (objetivo 4) son
  **percentil**: las réplicas son bimodales (C ≈ 0,03 cuando Bogotá no sale en la muestra y
  0,08–0,12 cuando sale), así que estimación ± 1,96·EE no las describe.
- Referencia fija del objetivo 2 (`DPTOS_REF`): Caquetá, Putumayo, Casanare,
  Meta y Huila, los cinco departamentos elegibles en los cuatro quinquenios
  con menor prevalencia ajustada 2005-2024. Se eligió una sola vez con toda
  la serie; con solo 2005-2014 habrían salido cuatro de los cinco (Caldas en
  lugar de Meta). El PAF es lo "mejor alcanzable" entre departamentos, no una
  meta causal.
- El régimen de afiliación solo es comparable desde 2008 (objetivos 3 y 4).
- "A término" significa siempre 38 semanas y más en partos únicos, con
  `T_GES` (los 20 años). Los resultados que usan la semana 37 son controles
  con los años de `T_GES_AGRU_CIE` y no se mezclan con la serie principal.
- Tendencia del RII (hoja `RII_tendencia`): valores p por bootstrap wild de
  los puntajes por departamento con la nula impuesta (Kline y Santos 2012,
  9.999 réplicas, semilla 20241); IC 95 % con t de 32 grados de libertad.
- Etnia: "Sin información" y Rom quedan sin dato; falta en 2005-2007, 2012 y
  2013. Área: centro poblado y rural disperso van juntos.
- Fuente del certificado (`TIPOFORMULARIO`, desde 2021): 1 = certificado
  médico (RUAF-ND), 2 = Formato de Notificación Étnica, 3 = recuperado en el
  cruce con la Registraduría. Los de tipo 3 (2.083 en 2022, 4.086 en 2023 y
  8.303 en 2024) no traen peso, semanas, educación ni residencia, solo dónde
  ocurrieron. La base analítica es el total menos esos registros, y
  `pct_sin_peso_base` es la serie principal del faltante de peso. Con los
  microdatos públicos no se puede comprobar si duplican certificados médicos.
- Semanas de gestación: `T_GES` agrupa 28-37 semanas. La semana 37 se separa
  del pretérmino (28-36) cruzándola con `T_GES_AGRU_CIE` (`prematuro` en
  `dane-data`), que falta en 2005-2007, 2012, 2013 y 2020.
- La validación del ordenamiento contra NBI/IPM (hoja `Validacion_proxy`) se
  omite con un aviso si no existe la plantilla
  `3.0 Datos/nbi_ipm_departamentos.csv`.
- La variante anual "Sin Bogotá" no tiene bootstrap en el gráfico G2; el
  bootstrap completo está en los trienios y la
  diferencia pareada.
