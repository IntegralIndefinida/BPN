# -*- coding: utf-8 -*-
# =============================================================================
#  TRABAJO FINAL - BAJO PESO AL NACER EN COLOMBIA 2005-2024
#  Python: obj3_educacion.py
#  (equivale a 03_Objetivo3_Educacion_Regimen.do, v2)
#
#  Objetivo 3: relación entre el nivel educativo de la madre / régimen de
#  afiliación (proxy socioeconómico, 2008-2024) y la prevalencia de BPN.
#   A. Prevalencia cruda y estandarizada por edad de la madre (directa).
#   B. Razones de prevalencia con Poisson (registro logarítmico) y errores
#      estándar agrupados por departamento: modelos M0 a M5.
#   C. Índice relativo (RII) y de la pendiente (SII) por educación con ridit.
#  No se ajusta por gestación, controles prenatales ni tipo de parto
#  (mediadores, ver nota técnica del do-file).
#  Salidas: Objetivo3_Educacion_Regimen_BPN.xlsx, G9, G10 y log.
# =============================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

import comun_bpn as cb

EXCEL = "Objetivo3_Educacion_Regimen_BPN.xlsx"
Z95 = 1.96


# --------------------------------------------------------------------------- #
# 0. Celdas del objetivo 3
# --------------------------------------------------------------------------- #
def celdas_obj3(celdas: pd.DataFrame) -> pd.DataFrame:
    """Nacidos con dpto válido, peso y edad conocidos, agrupados por
    (anio, dpto_res, educ4, regimen, edad6, alt_cat); los faltantes de
    educ4/régimen/alt_cat pasan a la categoría 9 (se excluyen por modelo).
    A nivel de celda con exposición, los coeficientes de Poisson y sus errores
    agrupados coinciden con los de los datos individuales (nota técnica)."""
    d = celdas
    d = d[d["dpto_res"].gt(0) & d["con_peso"].gt(0) & d["edad6"].notna()].copy()
    d["educ4"] = d["educ4"].fillna(9).astype(int)
    d["regimen"] = d["regimen"].fillna(9).astype(int)
    d["alt_cat"] = d["alt_cat"].fillna(9).astype(int)
    return (d.groupby(["anio", "dpto_res", "educ4", "regimen", "edad6", "alt_cat"],
                      as_index=False, observed=True)
            .agg(n=("con_peso", "sum"), casos=("bpn_n", "sum"),
                 n_term=("n_term", "sum"), casos_term=("bpn_term_n", "sum")))


def poblacion_estandar(celdas: pd.DataFrame) -> pd.Series:
    """Población estándar (w_edad por edad6): todos los nacidos con peso y
    edad conocidos 2005-2024."""
    n = celdas.groupby("edad6", observed=True)["n"].sum()
    return n / n.sum()


# --------------------------------------------------------------------------- #
# A. Prevalencia cruda y estandarizada por edad
# --------------------------------------------------------------------------- #
def prevalencias(celdas: pd.DataFrame, pesos: pd.Series) -> pd.DataFrame:
    filas = []
    for var, etqs, grupo, periodos in [
            ("educ4", cb.EDUC4_ETQ, "Nivel educativo", [(2005, 2024), (2008, 2024)]),
            ("regimen", cb.REGIMEN_ETQ, "Régimen", [(2008, 2024)])]:
        for a0, a1 in periodos:
            d = celdas[celdas[var].ne(9) & celdas["anio"].between(a0, a1)]
            agg = cb.estandarizar_edad(d, [var], pesos)
            for _, f in agg.iterrows():
                filas.append({"grupo": grupo, "categoria": etqs[int(f[var])],
                              "periodo": f"{a0}-{a1}", "n": f.n, "casos": f.casos,
                              "pct_cruda": 100 * f.casos / f.n,
                              "pct_estand": 100 * f.comp,
                              "li_estand": 100 * (f.comp - Z95 * np.sqrt(f.v)),
                              "ls_estand": 100 * (f.comp + Z95 * np.sqrt(f.v))})
    return pd.DataFrame(filas)


def estandarizada_anio(celdas: pd.DataFrame, pesos: pd.Series, var: str,
                       desde: int) -> pd.DataFrame:
    """Prevalencia estandarizada por año según educ4 o regimen (para G9/G10)."""
    d = celdas[celdas[var].ne(9) & celdas["anio"].ge(desde)]
    agg = cb.estandarizar_edad(d, ["anio", var], pesos)
    agg["pct_estand"] = 100 * agg["comp"]
    return agg[["anio", var, "n", "pct_estand"]]


# --------------------------------------------------------------------------- #
# B. Razones de prevalencia: GLM Poisson, exposición n, cluster dpto_res
# --------------------------------------------------------------------------- #
def _dummies(df: pd.DataFrame, col: str, ref: int) -> pd.DataFrame:
    """Indicadoras de `col` (strings) sin la categoría de referencia."""
    d = pd.get_dummies(df[col].astype(int).astype(str), prefix=col, dtype=float)
    return d.drop(columns=f"{col}_{ref}", errors="ignore")


def _ajustar(df: pd.DataFrame, y: str, X: pd.DataFrame, expo: str):
    """GLM Poisson exposición log; EE agrupados por dpto (como vce(cluster)).
    Devuelve (coeficientes, errores estándar, N celdas, #dptos)."""
    X = sm.add_constant(X.reindex(df.index).astype(float), has_constant="add")
    res = sm.GLM(df[y].astype(float), X, family=sm.families.Poisson(),
                 offset=np.log(df[expo].astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": df["dpto_res"].to_numpy(),
                                      "use_correction": True, "df_correction": True})
    return dict(res.params), dict(res.bse), res.nobs, df["dpto_res"].nunique()


def _filas(modelo: str, b: dict, se: dict, etqs: dict, var: str, niveles,
           N: float, G: int) -> list[dict]:
    """guardar_rp del do-file: exp(b), IC y p para los niveles pedidos."""
    fuera = []
    for nivel in niveles:
        nombre = f"{var}_{nivel}"
        if nombre not in b:
            continue
        k, s = b[nombre], se[nombre]
        if s == 0 or not np.isfinite(s):
            continue
        fuera.append({"modelo": modelo, "variable": var,
                      "categoria": etqs[int(nivel)], "rp": np.exp(k),
                      "li": np.exp(k - Z95 * s), "ls": np.exp(k + Z95 * s),
                      "p": 2 * norm.sf(abs(k / s)), "n_obs": N, "n_clust": G})
    return fuera


def razones_prevalencia(celdas: pd.DataFrame) -> pd.DataFrame:
    filas = []

    # M0. Educación, sin ajuste (2005-2024)
    d = celdas[celdas["educ4"] != 9]
    X0 = _dummies(d, "educ4", 4)
    b, se, N, G = _ajustar(d, "casos", X0, "n")
    filas += _filas("M0 Educación (cruda)", b, se, cb.EDUC4_ETQ, "educ4", [1, 2, 3], N, G)

    # M1. Educación ajustada por edad y año (2005-2024)
    X1 = pd.concat([_dummies(d, "educ4", 4), _dummies(d, "edad6", 2),
                    _dummies(d, "anio", 2005)], axis=1)
    b, se, N, G = _ajustar(d, "casos", X1, "n")
    filas += _filas("M1 Educación + edad + año", b, se, cb.EDUC4_ETQ, "educ4",
                     [1, 2, 3], N, G)
    filas += _filas("M1 Educación + edad + año", b, se, cb.EDAD6_ETQ, "edad6",
                     [1, 3, 4, 5, 6], N, G)

    # M2. Régimen ajustado por edad y año (2008-2024)
    d8 = celdas[celdas["anio"] >= 2008]
    dr = d8[d8["regimen"] != 9]
    X2 = pd.concat([_dummies(dr, "regimen", 1), _dummies(dr, "edad6", 2),
                    _dummies(dr, "anio", 2008)], axis=1)
    b, se, N, G = _ajustar(dr, "casos", X2, "n")
    filas += _filas("M2 Régimen + edad + año", b, se, cb.REGIMEN_ETQ, "regimen",
                     [2, 3], N, G)

    # M3. Educación y régimen juntos, ajustados por edad y año [principal]
    dm = d8[(d8["educ4"] != 9) & (d8["regimen"] != 9)]
    X3 = pd.concat([_dummies(dm, "educ4", 4), _dummies(dm, "regimen", 1),
                    _dummies(dm, "edad6", 2), _dummies(dm, "anio", 2008)], axis=1)
    b, se, N, G = _ajustar(dm, "casos", X3, "n")
    filas += _filas("M3 Educación + régimen + edad + año", b, se, cb.EDUC4_ETQ,
                     "educ4", [1, 2, 3], N, G)
    filas += _filas("M3 Educación + régimen + edad + año", b, se, cb.REGIMEN_ETQ,
                     "regimen", [2, 3], N, G)
    filas += _filas("M3 Educación + régimen + edad + año", b, se, cb.EDAD6_ETQ,
                     "edad6", [1, 3, 4, 5, 6], N, G)

    # M4. M3 + altitud del municipio + efectos fijos de departamento
    da = dm[dm["alt_cat"] != 9]
    X4 = pd.concat([_dummies(da, "educ4", 4), _dummies(da, "regimen", 1),
                    _dummies(da, "edad6", 2), _dummies(da, "anio", 2008),
                    _dummies(da, "alt_cat", 1), _dummies(da, "dpto_res", 5)], axis=1)
    b, se, N, G = _ajustar(da, "casos", X4, "n")
    filas += _filas("M4 M3 + altitud + EF departamento", b, se, cb.EDUC4_ETQ,
                     "educ4", [1, 2, 3], N, G)
    filas += _filas("M4 M3 + altitud + EF departamento", b, se, cb.REGIMEN_ETQ,
                     "regimen", [2, 3], N, G)

    # M5. M3 con BPN a término en partos únicos (proxy de restricción del crecimiento)
    dt = dm[dm["n_term"] > 0]
    X5 = pd.concat([_dummies(dt, "educ4", 4), _dummies(dt, "regimen", 1),
                    _dummies(dt, "edad6", 2), _dummies(dt, "anio", 2008)], axis=1)
    b, se, N, G = _ajustar(dt, "casos_term", X5, "n_term")
    filas += _filas("M5 M3 con BPN a término (únicos >=38 sem.)", b, se,
                     cb.EDUC4_ETQ, "educ4", [1, 2, 3], N, G)
    filas += _filas("M5 M3 con BPN a término (únicos >=38 sem.)", b, se,
                     cb.REGIMEN_ETQ, "regimen", [2, 3], N, G)

    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# C. RII y SII por educación (ridit por año: 0 = más educada, 1 = menos)
# --------------------------------------------------------------------------- #
def ridit_educacion(celdas: pd.DataFrame) -> pd.DataFrame:
    d = celdas[celdas["educ4"] != 9]
    tot = (d.groupby(["anio", "educ4"], as_index=False)["n"].sum()
           .sort_values(["anio", "educ4"], ascending=[True, False]))
    tot["acum"] = tot.groupby("anio")["n"].cumsum()
    tot["ridit"] = (tot["acum"] - tot["n"] / 2) / tot.groupby("anio")["n"].transform("sum")
    return tot[["anio", "educ4", "ridit"]]


def rii_sii(celdas: pd.DataFrame, a0: int, a1: int) -> pd.DataFrame:
    """RII: Poisson de casos frente al ridit (exposición n, cluster dpto).
    SII: regresión lineal ponderada por n de p_celda, en puntos porcentuales."""
    d = celdas[celdas["anio"].between(a0, a1)]
    X = sm.add_constant(pd.concat([d["ridit"].rename("ridit"),
                                   _dummies(d, "edad6", 2),
                                   _dummies(d, "anio", a0)], axis=1).astype(float),
                        has_constant="add")
    grupos = {"groups": d["dpto_res"].to_numpy(),
              "use_correction": True, "df_correction": True}
    filas = []
    for titulo, aj in [
        ("RII", sm.GLM(d["casos"].astype(float), X, family=sm.families.Poisson(),
                       offset=np.log(d["n"].astype(float))).fit(cov_type="cluster",
                                                                cov_kwds=grupos)),
        ("SII (pp)", sm.WLS(d["p_celda"].astype(float), X,
                            weights=d["n"].astype(float)).fit(cov_type="cluster",
                                                              cov_kwds=grupos)),
    ]:
        k, s = float(aj.params["ridit"]), float(aj.bse["ridit"])
        p = 2 * norm.sf(abs(k / s))
        if titulo == "RII":
            fuera = {"valor": np.exp(k), "li": np.exp(k - Z95 * s),
                     "ls": np.exp(k + Z95 * s)}
        else:
            fuera = {"valor": 100 * k, "li": 100 * (k - Z95 * s),
                     "ls": 100 * (k + Z95 * s)}
        filas.append({"periodo": f"{a0}-{a1}", "indice": titulo,
                      "p": p, **fuera})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# G9/G10. Prevalencia estandarizada por año según educ4 y según régimen
# --------------------------------------------------------------------------- #
def grafico_estand(agg: pd.DataFrame, var: str, desde: int, titulo: str,
                   colores: dict[int, str], nota: str,
                   trazo3: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for nivel, color in colores.items():
        s = agg[agg[var] == nivel]
        if nivel == 3 and trazo3:
            ax.plot(s["anio"], s["pct_estand"], color=color, ls=trazo3, label=str(nivel))
        else:
            ax.plot(s["anio"], s["pct_estand"], color=color, label=str(nivel))
    etqs = (cb.EDUC4_ETQ if var == "educ4" else cb.REGIMEN_ETQ)
    ax.set_xticks(range(desde, 2025, 2))
    ax.set_ylabel("% BPN estandarizada por edad materna")
    ax.set_title(titulo, fontsize=12)
    ax.legend(labels=[etqs[k] for k in colores], loc="upper left", fontsize=8.5)
    ax.annotate(nota, xy=(1, -0.14), xycoords="axes fraction", ha="right",
                fontsize=7, va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)
# --------------------------------------------------------------------------- #
LOG = "log_03_objetivo3"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 3; las claves con "_" son intermedias (no van al xlsx)."""
    c = celdas_obj3(celdas)
    pesos = poblacion_estandar(c)
    d = c.merge(ridit_educacion(c), on=["anio", "educ4"])
    d["p_celda"] = d["casos"] / d["n"]
    return {
        "Prevalencias": prevalencias(c, pesos),
        "RP_modelos": razones_prevalencia(c),
        "RII_SII_educacion": pd.concat([rii_sii(d, 2005, 2024), rii_sii(d, 2008, 2024)],
                                       ignore_index=True),
        "_est_educ": estandarizada_anio(c, pesos, "educ4", 2005),
        "_est_reg": estandarizada_anio(c, pesos, "regimen", 2008),
        "_n_celdas": len(c),
    }


def imprimir(r: dict) -> None:
    print("\nOBJETIVO 3 - Educación, régimen y BPN")
    print(f"celdas obj3: {r['_n_celdas']:,}")
    print("\n--- A. Prevalencia cruda y estandarizada por edad ---")
    print(r["Prevalencias"][["grupo", "categoria", "periodo", "pct_cruda", "pct_estand",
                             "li_estand", "ls_estand"]].round(2).to_string(index=False))
    print("\n--- B. Razones de prevalencia (M0 a M5), cluster dpto ---")
    print(r["RP_modelos"][["modelo", "variable", "categoria", "rp", "li", "ls"]]
          .round(2).to_string(index=False))
    print("\n--- C. RII y SII por educación ---")
    print(r["RII_SII_educacion"].round(3).to_string(index=False))


def graficos(r: dict) -> dict:
    cb.iniciar_estilo()
    nota = ("Estandarización directa por edad materna (6 grupos).\n"
            "Fuente: DANE, EEVV. Cálculos propios.")
    return {
        "G9_BPN_educacion_estandarizada": grafico_estand(
            r["_est_educ"], "educ4", 2005,
            "BPN según nivel educativo de la madre, 2005-2024",
            {1: "#A50026", 2: "#F46D43", 3: "#74ADD1", 4: "#313695"}, nota),
        "G10_BPN_regimen_estandarizada": grafico_estand(
            r["_est_reg"], "regimen", 2008,
            "BPN según régimen de afiliación de la madre, 2008-2024",
            {1: "#313695", 2: "#A50026", 3: "#888888"}, nota, trazo3="--"),
    }
