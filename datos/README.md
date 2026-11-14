# Insumos

Los microdatos de nacimientos del DANE no están aquí: `dane-data` los
descarga del Catálogo Central de Microdatos (ver `v2/README.md`). Estos son
los otros dos insumos del análisis.

| Archivo | Contenido | Fuente | Lo usa |
|---|---|---|---|
| `altitud_municipios.csv` | Altitud (m s. n. m.) de 1.122 municipios, por código DIVIPOLA | GeoNames (CC BY 4.0) emparejado con DIVIPOLA del DANE | `comun_bpn.py` (ajuste por altitud, objetivos 1 a 4) |
| `nbi_ipm_departamentos.csv` | Incidencia de pobreza multidimensional 2018 (%), 33 departamentos | DANE, Incidencia de Pobreza Multidimensional, Departamentos 2018-2025 (columna 2018) | `obj4_concentracion.py` (hoja `Validacion_proxy`) |

## `altitud_municipios.csv`

Columnas: `cod_mpio` (DIVIPOLA, 5 dígitos), `municipio`, `departamento`,
`altitud_m`, `lugar_geonames` (lugar de GeoNames asignado), `distancia_km` y
`tipo_empate`: `nombre` (1.062), `cercano` (33), `lejano` (26) y `asignado`
(1). La mediana de `distancia_km` es 0,21 km y el máximo, 270 km.

Los municipios que no están en el archivo reciben la media de su departamento
ponderada por nacimientos (`_imputar_altitud` en `comun_bpn.py`).

Pendiente: el procedimiento con el que se emparejaron GeoNames y DIVIPOLA
(qué mide `distancia_km` y cómo se decide cada `tipo_empate`) no está en este
repositorio.

Atribución: datos de altitud de GeoNames (https://www.geonames.org),
licencia CC BY 4.0.
