# -*- coding: utf-8 -*-
# =============================================================================
#  TRABAJO FINAL - BAJO PESO AL NACER EN COLOMBIA 2005-2024
#  Fuente: DANE, Estadísticas Vitales (EEVV) - microdatos de nacimientos
#
#  comun_bpn.py  (módulo común de los 4 scripts de Python)
#
#  - Rutas y constantes del proyecto.
#  - Etiquetas de las variables armonizadas.
#  - Lectura de los microdatos con dane-data y agregación a "celdas" por
#    (anio, dpto_res, cod_mpio, educ4, regimen, edad6) con las cuentas
#    (nacidos, con peso, BPN, BPN a término, régimen, educación, altitud).
#    Los 4 objetivos se calculan a partir de estas celdas.
#  - Altitud por municipio: merge con el CSV e imputación con la media del
#    departamento ponderada por nacimientos.
#  - Caché en parquet por año para no releer los archivos del DANE.
# =============================================================================

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import dane_data as dd

# --------------------------------------------------------------------------- #
# 0. RUTAS Y AÑOS
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[3]        # .../3. Metodologia
DATOS = RAIZ / "3.0 Datos"
SALIDAS = DATOS / "python_v2"                     # gráficos, xlsx y logs
PROCESADOS = SALIDAS / "procesados"               # celdas intermedias (parquet)
ALTITUD_CSV = DATOS / "altitud_municipios.csv"    # insumo de altitud

ANIO_INI, ANIO_FIN = 2005, 2024

# --------------------------------------------------------------------------- #
# 1. ETIQUETAS
# --------------------------------------------------------------------------- #
# Departamento de residencia (DIVIPOLA); 0 = sin información / exterior.
# Los códigos válidos son los 33 de dane_data.referencias.DEPARTAMENTOS
# (5..99); 1-4 y 75 son códigos crudos fuera de la tabla y se hacen 0.
DPTO_NOM = {
    0: "Sin información / exterior",
    5: "Antioquia",            8: "Atlántico",          11: "Bogotá D.C.",
    13: "Bolívar",             15: "Boyacá",            17: "Caldas",
    18: "Caquetá",             19: "Cauca",             20: "Cesar",
    23: "Córdoba",             25: "Cundinamarca",      27: "Chocó",
    41: "Huila",               44: "La Guajira",        47: "Magdalena",
    50: "Meta",                52: "Nariño",            54: "Norte de Santander",
    63: "Quindío",             66: "Risaralda",         68: "Santander",
    70: "Sucre",               73: "Tolima",            76: "Valle del Cauca",
    81: "Arauca",              85: "Casanare",          86: "Putumayo",
    88: "San Andrés",          91: "Amazonas",          94: "Guainía",
    95: "Guaviare",            97: "Vaupés",            99: "Vichada",
}
DPTOS_VALIDOS = [c for c in DPTO_NOM if c != 0]

PESO_GRUPO_ETQ = {
    1: "<1.000 g", 2: "1.000-1.499", 3: "1.500-1.999", 4: "2.000-2.499",
    5: "2.500-2.999", 6: "3.000-3.499", 7: "3.500-3.999", 8: "4.000 y más",
}
EDAD6_ETQ = {1: "Menos de 20", 2: "20-24", 3: "25-29", 4: "30-34",
             5: "35-39", 6: "40 y más"}
EDUC4_ETQ = {1: "Ninguno o preescolar", 2: "Primaria",
             3: "Secundaria o media", 4: "Superior",
             9: "Sin información"}
REGIMEN_ETQ = {1: "Contributivo, excepción o especial", 2: "Subsidiado",
               3: "No asegurado", 9: "Sin información"}

# --------------------------------------------------------------------------- #
# 2. LOG: imita el log del análisis (consola + archivo en SALIDAS)
# --------------------------------------------------------------------------- #
class _Tee:
    """Reenvía lo que se imprime a la consola y a un archivo de log."""

    def __init__(self, archivo: Path, consola):
        self.archivo = open(archivo, "w", encoding="utf-8")
        self.consola = consola

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()

    def close(self):
        self.archivo.close()


class log_salida:
    """Context manager: `with log_salida("log_01_..."):` duplica stdout al log."""

    def __init__(self, nombre: str):
        SALIDAS.mkdir(parents=True, exist_ok=True)
        self.ruta = SALIDAS / f"{nombre}.log"
        self._tee = None

    def __enter__(self):
        self._tee = _Tee(self.ruta, sys.stdout)
        sys.stdout = self._tee
        print("=" * 78)
        print(f"Log: {self.ruta}")
        return None

    def __exit__(self, *exc):
        if self._tee is not None:
            sys.stdout = self._tee.consola   # desactiva antes de cerrar
            self._tee.close()


# --------------------------------------------------------------------------- #
# 3. ARMONIZACIÓN DE UN AÑO -> celdas (Parte A/B del 01 análisis, por año)
# --------------------------------------------------------------------------- #
def _peso_grupo(peso: pd.Series) -> pd.Series:
    """PESO_NAC -> grupos 1..8 (el DANE lo publica en grupos en 2005-2024).

    Si algún año trajera gramos, se reparte con los cortes 999, 1499, ...,
    en el rango 200-7.000 g.
    """
    if peso.max() > 100:                                   # rama 'gramos'
        return pd.cut(peso, [199, 999, 1499, 1999, 2499, 2999, 3499, 3999, 7000],
                      labels=range(1, 9)).astype("Int8")
    return peso.where(peso.between(1, 8)).astype("Int8")   # rama 'grupos'


_MAP_EDAD6 = {
    "De 10 a 14 años": 1, "De 15 a 19 años": 1, "De 20 a 24 años": 2,
    "De 25 a 29 años": 3, "De 30 a 34 años": 4, "De 35 a 39 años": 5,
    "De 40 a 44 años": 6, "De 45 a 49 años": 6, "De 50 a 54 años": 6,
}
_MAP_EDUC4 = {
    "Ninguno": 1, "Preescolar": 1, "Primaria": 2,
    "Secundaria y media": 3, "Superior": 4,
}
_MAP_REGIMEN = {
    "Contributivo": 1, "Excepción": 1, "Especial": 1,
    "Subsidiado": 2, "No asegurado": 3,
}

_CUENTAS = ["n", "con_peso", "bpn_n", "n_term", "bpn_term_n", "reg_conocido",
            "n_nocontrib", "n_sub", "edu_conocido", "n_bajaedu"]


def _armonizar(anio: int) -> pd.DataFrame:
    """Lee el año con dane-data y devuelve las celdas de conteos del año.

    Celdas por (anio, dpto_res, cod_mpio, educ4, regimen, edad6); la altitud
    del CSV va por municipio, y su imputación se resuelve después, sobre la
    serie completa, usando la media ponderada por nacimientos.
    """
    d = dd.eevv.load(anio, tipo="nacimientos", columns=["PESO_NAC"])

    peso_grupo = _peso_grupo(pd.to_numeric(d["PESO_NAC"], errors="coerce"))
    bpn = peso_grupo.between(1, 4).astype("boolean").where(peso_grupo.notna())

    # BPN a término (>=38 sem.) en parto único: T_GES 4-5 y MUL_PARTO 1;
    # ignorado (6) y sin información (9) quedan NA.
    termino = d["gestacion_grupo"].astype(object).isin(
        ["De 38 a 41", "De 42 y más"]).astype("boolean")
    unico = d["parto_multiple"].astype("boolean").map(
        {True: False, False: True}).astype("boolean")
    bpn_term = bpn.where(
        termino.fillna(False).to_numpy() & unico.fillna(False).to_numpy())

    edad6 = d["edad_madre"].astype(object).map(_MAP_EDAD6).astype("Int8")
    educ4 = d["nivel_educativo_madre"].astype(object).map(_MAP_EDUC4).astype("Int8")
    if anio >= 2008:      # SEG_SOCIAL solo es comparable desde 2008
        regimen = d["regimen_salud"].astype(object).map(_MAP_REGIMEN).astype("Int8")
    else:
        regimen = pd.Series(pd.NA, index=d.index, dtype="Int8")

    # Departamento: dane-data trae el CODPTORE crudo en dpto_residencia;
    # los códigos fuera de la tabla (1-4, 75) van a 0 como en el análisis.
    bruto = d["dpto_residencia"].fillna(0).astype("int8").to_numpy()
    dpto = pd.Series(np.where(np.isin(bruto, DPTOS_VALIDOS), bruto, 0),
                     index=d.index)

    # Municipio de residencia (DIVIPOLA, 5 dígitos = dpto*1000 + CODMUNRE)
    cod_mpio = pd.to_numeric(d["mpio_residencia"].astype(str),
                             errors="coerce").astype("Int64")

    alt = pd.read_csv(ALTITUD_CSV, usecols=["cod_mpio", "altitud_m"])
    df = pd.DataFrame({
        "anio": anio, "dpto_res": dpto, "cod_mpio": cod_mpio,
        "educ4": educ4, "regimen": regimen, "edad6": edad6,
    }).merge(alt, on="cod_mpio", how="left")

    x = pd.concat([df.reset_index(drop=True),
                   pd.DataFrame({
        "n": np.ones(len(df)),
        "con_peso": bpn.notna().to_numpy(dtype=float),
        "bpn_n": bpn.fillna(False).to_numpy(dtype=float),
        "n_term": bpn_term.notna().to_numpy(dtype=float),
        "bpn_term_n": bpn_term.fillna(False).to_numpy(dtype=float),
        "reg_conocido": df["regimen"].notna().to_numpy(dtype=float),
        "n_nocontrib": df["regimen"].isin([2, 3]).to_numpy(dtype=float),
        "n_sub": df["regimen"].eq(2).to_numpy(dtype=float),
        "edu_conocido": df["educ4"].notna().to_numpy(dtype=float),
        "n_bajaedu": df["educ4"].isin([1, 2]).to_numpy(dtype=float),
    }).reset_index(drop=True)], axis=1)
    celdas = (x.groupby(["anio", "dpto_res", "cod_mpio", "educ4", "regimen",
                         "edad6", "altitud_m"], observed=True, dropna=False)
                .agg({c: "sum" for c in _CUENTAS})
                .reset_index())
    return celdas


def _imputar_altitud(celdas: pd.DataFrame) -> pd.DataFrame:
    """Imputa la altitud con la media del departamento (pooled 2005-2024).

    Imputa los faltantes con la media departamental y crea categorías de
    altitud de 500 metros, con la última categoría abierta.
    La media del dpto pondera por nacimientos: Σ n·alt / Σ n entre las celdas
    del dpto con altitud conocida, sobre toda la serie.
    """
    con_alt = celdas["altitud_m"].notna()
    num = (celdas["n"] * celdas["altitud_m"]).where(con_alt, 0.0)
    den = celdas["n"].where(con_alt, 0.0)
    media = (num.groupby(celdas["dpto_res"]).sum()
             / den.groupby(celdas["dpto_res"]).sum())

    imputada = celdas["altitud_m"].isna() & celdas["dpto_res"].gt(0)
    celdas["altitud_m"] = celdas["altitud_m"].where(
        ~imputada, celdas["dpto_res"].map(media))
    celdas.loc[celdas["dpto_res"].eq(0), "altitud_m"] = np.nan

    cat = np.minimum(1 + np.floor(celdas["altitud_m"].to_numpy(dtype=float) / 500), 6)
    celdas["alt_cat"] = pd.array(cat, dtype="Int8")   # NaN -> <NA>
    return celdas


def _huella(fuente: Path) -> str:
    """Firma del caché: archivo del DANE + versión dane-data + este módulo
    + CSV de altitud (la altitud queda guardada en las celdas)."""
    st = fuente.stat()
    h = hashlib.sha1()
    h.update(str(fuente).encode())
    h.update(f"{st.st_size}-{st.st_mtime_ns}".encode())
    h.update(importlib.metadata.version("dane-data").encode())
    h.update(Path(__file__).read_bytes())
    h.update(ALTITUD_CSV.read_bytes())
    return h.hexdigest()


def cargar_celdas(anios=None, refrescar: bool = False) -> pd.DataFrame:
    """Celdas 2005-2024 (o los años pedidos), con caché parquet por año.

    Devuelve un DataFrame con las llaves (anio, dpto_res, cod_mpio, educ4,
    regimen, edad6, altitud_m, alt_cat) y las cuentas de _CUENTAS.
    """
    if anios is None:
        anios = range(ANIO_INI, ANIO_FIN + 1)
    anios = list(anios)
    SALIDAS.mkdir(parents=True, exist_ok=True)
    carpeta = PROCESADOS / "celdas"
    carpeta.mkdir(parents=True, exist_ok=True)
    control = PROCESADOS / "control_carga.csv"

    partes, registro = [], []
    for anio in anios:
        fuente = dd.eevv.find_files(anio, tipo="nacimientos")[anio]
        h = _huella(fuente)
        destino, meta = carpeta / f"celdas_{anio}.parquet", carpeta / f"celdas_{anio}.json"
        if not refrescar and destino.exists() and json.loads(
                meta.read_text()).get("huella") == h:
            m = json.loads(meta.read_text())
            print(f"  {anio}: celdas en caché ({m['celdas']:,})")
            registro.append({"anio": anio, "archivo": m["archivo"],
                             "registros": m["registros"]})
        else:
            t0 = time.time()
            c = _armonizar(anio)
            c.to_parquet(destino)
            m = {"huella": h, "celdas": len(c), "archivo": fuente.name,
                 "registros": int(c["n"].sum())}
            meta.write_text(json.dumps(m))
            print(f"  {anio}: {len(c):,} celdas en {time.time() - t0:.1f}s "
                  f"({m['registros']:,} registros)")
            registro.append({"anio": anio, "archivo": m["archivo"],
                             "registros": m["registros"]})
        partes.append(pd.read_parquet(destino))

    reg_nuevo = pd.DataFrame(registro).assign(anio=lambda r: r["anio"].astype(int))
    reg_exist = (pd.read_csv(control)
                 if (control.exists() and control.stat().st_size) else None)
    if reg_exist is None or reg_exist.empty:
        reg_final = reg_nuevo
    else:
        viejos = reg_exist[~reg_exist["anio"].astype(int).isin(anios)]
        reg_final = pd.concat([viejos, reg_nuevo], ignore_index=True)
    reg_final.sort_values("anio").to_csv(control, index=False)

    celdas = pd.concat(partes, ignore_index=True)
    # La imputación de altitud depende de TODOS los años a la vez (como el
    # análisis, que apila primero la base completa): se resuelve aquí.
    return _imputar_altitud(celdas)


# --------------------------------------------------------------------------- #
# 4. FRECUENCIAS que no salen de las celdas (dd.eevv.serie_conteos)
# --------------------------------------------------------------------------- #
def conteos(por, anios=None, dropna: bool = True) -> pd.DataFrame:
    """Conteos de nacimientos por año y las columnas `por` (serie_conteos)."""
    return dd.eevv.serie_conteos(anios, tipo="nacimientos", por=por, dropna=dropna)


# --------------------------------------------------------------------------- #
# 5. ESTADÍSTICA COMÚN
# --------------------------------------------------------------------------- #
def wilson(casos, n, alpha: float = 0.05):
    """IC de Wilson de una proporción (li, ls) en porcentaje, como el análisis."""
    from statsmodels.stats.proportion import proportion_confint
    li, ls = proportion_confint(casos, n, alpha=alpha, method="wilson")
    return np.asarray(li) * 100, np.asarray(ls) * 100


def estandarizar_edad(d: pd.DataFrame, por: list[str], pesos: pd.Series,
                      n: str = "n", casos: str = "casos") -> pd.DataFrame:
    """Estandarización directa por edad6 (objetivos 1 y 3).

    `pesos`: w_edad de la población estándar, indexado por edad6.
    Devuelve, por las columnas `por`: n, casos, comp (prevalencia
    estandarizada, en proporción) y v (su varianza binomial)."""
    ae = (d.groupby([*por, "edad6"], observed=True, as_index=False)
          .agg(n=(n, "sum"), casos=(casos, "sum")))
    w = ae["edad6"].map(pesos)
    p = ae["casos"] / ae["n"]
    return (ae.assign(comp=w * p, v=w ** 2 * p * (1 - p) / ae["n"])
            .groupby(por, observed=True, as_index=False)[["n", "casos", "comp", "v"]]
            .sum())


def apc_poisson(df: pd.DataFrame, ini: int = ANIO_INI, fin: int = ANIO_FIN) -> dict:
    """Cambio porcentual anual (APC): Poisson con offset log(con_peso) y errores
    robustos. Requiere columnas anio, bpn_n y con_peso.
    """
    import math
    from scipy.stats import norm
    import statsmodels.api as sm
    d = df[(df["anio"] >= ini) & (df["anio"] <= fin)]
    X = sm.add_constant(d["anio"].to_numpy(float))
    res = sm.GLM(d["bpn_n"].to_numpy(float), X, family=sm.families.Poisson(),
                 offset=np.log(d["con_peso"].to_numpy(float))
                 ).fit(cov_type="HC0")
    # La corrección HC1 multiplica la varianza por n/(n-1); statsmodels no la
    # aplica automáticamente en GLM.
    b = float(res.params[1])
    se = float(res.bse[1]) * math.sqrt(len(d) / (len(d) - 1))
    return {"apc": 100 * (math.exp(b) - 1),
            "li": 100 * (math.exp(b - 1.96 * se) - 1),
            "ls": 100 * (math.exp(b + 1.96 * se) - 1),
            "p": 2 * norm.sf(abs(b / se)), "n_anios": len(d)}


# --------------------------------------------------------------------------- #
# 5. EXPORT Y GRÁFICOS
# --------------------------------------------------------------------------- #
def exportar_excel_nuevo(dfs: dict[str, pd.DataFrame], ruta_xlsx: str) -> None:
    """Escribe un libro completo de una vez (hojas en el orden del dict)."""
    SALIDAS.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(SALIDAS / ruta_xlsx, engine="openpyxl") as xl:
        for hoja, df in dfs.items():
            df.to_excel(xl, sheet_name=hoja, index=False)


def iniciar_estilo() -> None:
    """Estilo de los gráficos (equivale a set scheme s1color)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.grid": True,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 10.5, "axes.titlesize": 12,
        "legend.frameon": True, "legend.edgecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def guardar_grafico(fig, nombre: str) -> None:
    """Guarda el png en SALIDAS (como graph export ... width(2400))."""
    ruta = SALIDAS / nombre if Path(nombre).suffix == ".png" else SALIDAS / f"{nombre}.png"
    fig.savefig(ruta)
    print(f"  Gráfico: {ruta.name}")
