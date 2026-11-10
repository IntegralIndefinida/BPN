#  Objetivo 2: desigualdades departamentales en el BPN mediante razones y
#  diferencias de prevalencias (RP/DP) frente a una referencia elegible
#  (menor prevalencia entre departamentos con pesos y cobertura de registro
#  suficientes) y frente a la prevalencia de las 33 unidades, por quinquenio
#  y por año. Medidas resumen: Theil, MLD, PAR/PAF y casos en exceso.
#  Todo se calcula dos veces: con la prevalencia cruda y con la ajustada por
#  altitud y edad materna (hojas "_aj"), porque la referencia cruda cae
#  siempre en tierras bajas. IC 95 %: binomial (delta) para RP y DP;
#  simulación binomial para Theil y PAF.
#  Salidas: Objetivo2_Desigualdades_RT_DT.xlsx, G7 y G8 png, log.

import numpy as np
import pandas as pd

import comun_bpn as cb
from obj4_concentracion import REPS, SEMILLA, estandarizar_indirecta

EXCEL = "Objetivo2_Desigualdades_RT_DT.xlsx"
MIN_QUINQ = 10_000     # nacidos con peso conocido en el quinquenio
MIN_ANUAL = 2_000      # nacidos con peso conocido en el año
MAX_FALT = 5           # % máximo de nacimientos sin información de peso
QUINQ_ETQ = {1: "2005-2009", 2: "2010-2014", 3: "2015-2019", 4: "2020-2024"}
AZUL, NARANJA = "#08519C", "#A63603"   # par verificado para daltonismo


# A. Panel departamento-año

def panel_dpto_anio(celdas: pd.DataFrame) -> pd.DataFrame:
    d = (celdas[celdas["dpto_res"].gt(0)]
         .groupby(["dpto_res", "anio"])
         .agg(nacidos=("n", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum")).reset_index())
    d["departamento"] = d["dpto_res"].map(cb.DPTO_NOM)
    d["quinq"] = 1 + (d["anio"] - 2005) // 5
    return d


def panel_ajustado(celdas: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Panel con los casos de BPN ajustados por altitud y edad materna, para
    repetir todas las medidas sin el efecto de la altitud (la referencia cruda
    cae siempre en tierras bajas).

    Mismos nacidos y con_peso que `panel`; bpn_n = con_peso x prevalencia
    estandarizada del año (observada - esperada + media nacional), con la
    estandarización indirecta del objetivo 4 (un único modelo estimado con
    2005-2024, compartido con el objetivo 4)."""
    d = estandarizar_indirecta(celdas[celdas["dpto_res"].gt(0)])
    s = d.groupby(["dpto_res", "anio"], as_index=False)[["n_s", "casos_s",
                                                         "esperado_s"]].sum()
    g = s.groupby("anio")
    s["p_aj"] = ((s["casos_s"] - s["esperado_s"]) / s["n_s"]
                 + g["casos_s"].transform("sum") / g["n_s"].transform("sum"))
    p = panel.merge(s[["dpto_res", "anio", "p_aj"]], on=["dpto_res", "anio"],
                    validate="one_to_one")
    assert len(p) == len(panel), "dpto-años sin prevalencia ajustada"
    return p.assign(bpn_n=p["con_peso"] * p["p_aj"]).drop(columns="p_aj")


def _prevalencias(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["pct"] = 100 * d["bpn_n"] / d["con_peso"]
    d["pct_falt"] = 100 * (d["nacidos"] - d["con_peso"]) / d["nacidos"]
    return d


def _con_elegible(d: pd.DataFrame, minimo: int) -> pd.DataFrame:
    """Prevalencias + elegibilidad como referencia: >= `minimo` nacidos con
    peso y <= MAX_FALT % sin información de peso."""
    d = _prevalencias(d)
    d["elegible"] = (d["con_peso"] >= minimo) & (d["pct_falt"] <= MAX_FALT)
    return d


def _ic_rt_dt(d: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """IC 95 % de rt y dt de cada departamento frente a la referencia de su
    `grupo` (varianza binomial, método delta). La referencia se toma como
    dada: el intervalo no incluye la incertidumbre de haberla elegido por ser
    la mínima."""
    es_ref = d["elegible"].astype(bool) & d["pct"].eq(d["ref"])
    n_ref = d["con_peso"].where(es_ref).groupby(d[grupo]).transform("max")
    p, p0, n = d["pct"] / 100, d["ref"] / 100, d["con_peso"]
    se_log = np.sqrt((1 - p) / (n * p) + (1 - p0) / (n_ref * p0)).where(~es_ref, 0.0)
    se_dif = 100 * np.sqrt(p * (1 - p) / n + p0 * (1 - p0) / n_ref).where(~es_ref, 0.0)
    return d.assign(rt_li=d["rt"] * np.exp(-1.96 * se_log),
                    rt_ls=d["rt"] * np.exp(1.96 * se_log),
                    dt_li=d["dt"] - 1.96 * se_dif, dt_ls=d["dt"] + 1.96 * se_dif)


def _ic_resumen(n: np.ndarray, p: np.ndarray, elegible: np.ndarray) -> dict:
    """IC 95 % percentil de Theil y PAF por simulación binomial (REPS réplicas):
    en cada una se sortean los casos de cada departamento, Binomial(n, p), y
    se recalculan la media, la referencia elegible y los dos índices."""
    b = np.random.default_rng(SEMILLA).binomial(n.round().astype(int), p,
                                                size=(REPS, len(n)))
    pr, mu = b / n, b.sum(1) / n.sum()
    r = pr / mu[:, None]
    theil = 1000 * (n / n.sum() * np.where(r > 0, r * np.log(np.where(r > 0, r, 1)),
                                           0.0)).sum(1)
    paf = 100 * (mu - np.where(elegible, pr, np.inf).min(1)) / mu
    lim = lambda x: np.percentile(x, [2.5, 97.5]) if np.isfinite(x).all() else (np.nan,) * 2
    (t0, t1), (f0, f1) = lim(theil), lim(paf)
    return {"theil_li": t0, "theil_ls": t1, "paf_li": f0, "paf_ls": f1}


def _quinquenios(panel: pd.DataFrame) -> pd.DataFrame:
    return _con_elegible(panel.groupby(["dpto_res", "departamento", "quinq"],
                                       as_index=False)[["nacidos", "con_peso", "bpn_n"]]
                         .sum(), MIN_QUINQ)


# B. Quinquenios: RP/DP frente a la referencia y a la prevalencia de las 33

def rt_dt_quinquenio(d: pd.DataFrame) -> pd.DataFrame:
    d = _quinquenios(d)
    d["li"], d["ls"] = cb.wilson(d["bpn_n"], d["con_peso"])
    g = d.groupby("quinq")
    d["ref"] = d["pct"].where(d["elegible"]).groupby(d["quinq"]).transform("min")
    d["mu"] = g["bpn_n"].transform("sum") / g["con_peso"].transform("sum") * 100
    d["es_ref"] = (d["elegible"] & d["pct"].eq(d["ref"])).astype(int)
    # Nombre del dpto de referencia: el de mayor código entre los empatados
    # Ordena los departamentos dentro de cada quinquenio por condición de referencia.
    dref = (d[d["es_ref"].eq(1)].sort_values("dpto_res")
            .groupby("quinq")["departamento"].last())
    d["dpto_ref"] = d["quinq"].map(dref)
    d["rt"] = d["pct"] / d["ref"]
    d["dt"] = d["pct"] - d["ref"]
    d["rt_nac"] = d["pct"] / d["mu"]
    d["dt_nac"] = d["pct"] - d["mu"]
    d = _ic_rt_dt(d, "quinq")
    d["elegible"] = d["elegible"].astype(int)
    d["quinq"] = d["quinq"].map(QUINQ_ETQ)
    return d[["quinq", "dpto_res", "departamento", "nacidos", "con_peso",
              "pct_falt", "elegible", "bpn_n", "pct", "li", "ls", "ref",
              "dpto_ref", "rt", "rt_li", "rt_ls", "dt", "dt_li", "dt_ls", "mu",
              "rt_nac", "dt_nac", "es_ref"]]


def rt_dt_ancho(quinq: pd.DataFrame) -> pd.DataFrame:
    d = quinq[["dpto_res", "departamento", "quinq", "rt", "dt"]].copy()
    d["quinq"] = d["quinq"].map({v: k for k, v in QUINQ_ETQ.items()})
    d = d.pivot(index=["dpto_res", "departamento"], columns="quinq",
                values=["rt", "dt"])
    d.columns = [f"{est}{q}" for est, q in d.columns]
    d = d.reset_index()
    d["rt_prom"] = d[["rt1", "rt2", "rt3", "rt4"]].mean(axis=1)
    d["dt_prom"] = d[["dt1", "dt2", "dt3", "dt4"]].mean(axis=1)
    return d.sort_values("rt_prom", ascending=False).reset_index(drop=True)


def rt_dt_anual(d: pd.DataFrame) -> pd.DataFrame:
    d = _con_elegible(d, MIN_ANUAL)
    d["ref"] = d["pct"].where(d["elegible"]).groupby(d["anio"]).transform("min")
    d["rt"] = d["pct"] / d["ref"]
    d["dt"] = d["pct"] - d["ref"]
    d = _ic_rt_dt(d, "anio")
    d["elegible"] = d["elegible"].astype(int)
    return d[["anio", "dpto_res", "departamento", "nacidos", "con_peso",
              "pct_falt", "elegible", "bpn_n", "pct", "ref", "rt", "rt_li", "rt_ls",
              "dt", "dt_li", "dt_ls"]].sort_values(
        ["anio", "dpto_res"]).reset_index(drop=True)


# C. Medidas resumen por periodo (programa resumen_desig del análisis)

def resumen_desig(d: pd.DataFrame, periodo: str, tipo: str) -> dict:
    """Medidas de desigualdad de un periodo: mu, referencia elegible, PAR/PAF,
    exceso, Theil y MLD ponderados por nacidos con peso, y extremos."""
    da = d.sort_values("dpto_res")
    W = da["con_peso"].sum()
    B = da["bpn_n"].sum()
    mu = 100 * B / W
    s = da["con_peso"].to_numpy() / W
    r = da["pct"].to_numpy() / mu
    theil = 1000 * np.sum(np.where(r > 0, s * r * np.log(r), 0.0))
    mld = 1000 * np.sum(np.where(r > 0, s * -np.log(r), 0.0))

    pmax = da["pct"].max()
    pmin = da["pct"].min()
    nmax = da[da["pct"].eq(pmax)]["departamento"].iloc[0]
    nmin = da[da["pct"].eq(pmin)]["departamento"].iloc[0]
    eleg = da[da["elegible"]]
    ref = eleg["pct"].min()
    if np.isnan(ref):
        nref = ""
        rt_el = dt_el = np.nan
    else:
        nref = (eleg[eleg["pct"].eq(ref)].sort_values("dpto_res")
                ["departamento"].iloc[-1])      # último departamento del quinquenio
        rt_el = pmax / ref
        dt_el = pmax - ref
    ic = _ic_resumen(da["con_peso"].to_numpy(float), da["pct"].to_numpy(float) / 100,
                     da["elegible"].to_numpy(bool))
    paf = 100 * (mu - ref) / mu
    assert ic["theil_li"] <= theil <= ic["theil_ls"], f"Theil fuera de su IC ({periodo})"
    assert np.isnan(paf) or ic["paf_li"] <= paf <= ic["paf_ls"], \
        f"PAF fuera de su IC ({periodo})"
    return {"periodo": periodo, "tipo": tipo, "mu": mu, "ref": ref,
            "dpto_ref": nref, "par": mu - ref, "paf": paf, **ic,
            "exceso": (mu - ref) / 100 * W, "theil": theil, "mld": mld,
            "pmax": pmax, "dpto_max": nmax, "pmin": pmin, "dpto_min": nmin,
            "rt_ext": pmax / pmin, "dt_ext": pmax - pmin,
            "rt_ext_el": rt_el, "dt_ext_el": dt_el}


def medidas_resumen(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    quinq = _quinquenios(panel)
    for q, etq in QUINQ_ETQ.items():
        filas.append(resumen_desig(quinq[quinq["quinq"] == q], etq, "Quinquenio"))
    for anio in range(cb.ANIO_INI, cb.ANIO_FIN + 1):
        d_anio = _con_elegible(panel[panel["anio"] == anio], MIN_ANUAL)
        filas.append(resumen_desig(d_anio, str(anio), "Año"))
    return pd.DataFrame(filas)[
        ["periodo", "tipo", "mu", "ref", "dpto_ref", "par", "paf", "paf_li",
         "paf_ls", "exceso", "theil", "theil_li", "theil_ls", "mld", "pmax", "dpto_max", "pmin", "dpto_min", "rt_ext",
         "dt_ext", "rt_ext_el", "dt_ext_el"]]


# G7. Dot plot de RP por departamento, primer y último quinquenio

def grafico_g7(ancho: pd.DataFrame, ancho_aj: pd.DataFrame):
    """Dos paneles con el mismo eje y el mismo orden (el de la RP ajustada del
    último quinquenio): prevalencia cruda y ajustada por altitud y edad."""
    import matplotlib.pyplot as plt
    orden = ancho_aj.sort_values(["rt4", "rt1"], ascending=False)["departamento"]
    y = range(len(orden))
    fig, ejes = plt.subplots(1, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, tabla, titulo in [(ejes[0], ancho, "Cruda"),
                              (ejes[1], ancho_aj, "Ajustada por altitud y edad materna")]:
        d = tabla.set_index("departamento").loc[orden]
        ax.hlines(y, d["rt4"], d["rt1"], color="#DDDDDD", lw=1, zorder=1)
        ax.scatter(d["rt1"], y, facecolor="none", edgecolor="#666666", s=28,
                   label="2005-2009", zorder=2)                    # O hueco
        ax.scatter(d["rt4"], y, marker="D", color=AZUL, s=24,
                   label="2020-2024", zorder=3)
        ax.axvline(1, color="#999999", ls="--", lw=1)
        ax.set_xlabel("Razón de prevalencias frente a la referencia")
        ax.set_title(titulo, fontsize=10.5)
    ejes[0].set_yticks(list(y))
    ejes[0].set_yticklabels(orden, fontsize=6.5)
    ejes[0].invert_yaxis()
    ejes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Razón de prevalencias de BPN por departamento", fontsize=12)
    fig.text(0.01, 0.005,
             "Referencia: menor prevalencia entre departamentos con >=10.000 nacidos "
             "con peso y <=5 % sin información de peso en el quinquenio.\n"
             "Departamentos ordenados por la razón ajustada de 2020-2024. "
             "Fuente: DANE, EEVV. Cálculos propios.", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# G8. Theil y PAF anuales, crudos y ajustados (un panel por medida: sin eje
#     doble, que inventa una relación entre dos escalas arbitrarias)

def grafico_g8(resumen: pd.DataFrame, resumen_aj: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    for ax, col, titulo, eje in [
            (ejes[0], "theil", "Dispersión entre departamentos", "Índice de Theil (x1000)"),
            (ejes[1], "paf", "Fracción atribuible poblacional", "PAF (%)")]:
        for tabla, nombre, color, trazo, marca in [
                (resumen, "Cruda", AZUL, "-", "o"),
                (resumen_aj, "Ajustada por altitud y edad materna", NARANJA, "--", "D")]:
            d = (tabla[tabla["tipo"].eq("Año")]
                 .assign(anio=lambda t: t["periodo"].astype(int)))
            ax.fill_between(d["anio"], d[f"{col}_li"], d[f"{col}_ls"], color=color,
                            alpha=0.18, lw=0)
            ax.plot(d["anio"], d[col], color=color, ls=trazo, marker=marca, ms=4,
                    lw=2, label=nombre)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(2005, 2025, 3))
        ax.set_ylabel(eje)
        ax.set_title(titulo, fontsize=10.5)
    ejes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Desigualdad departamental del BPN, 2005-2024", fontsize=12)
    fig.text(0.01, 0.005,
             "Theil ponderado por nacidos con peso conocido. PAF frente a la referencia "
             "anual elegible. Bandas: IC 95 % por simulación binomial.\n"
             "Fuente: DANE, Estadísticas Vitales. Cálculos propios.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)

LOG = "log_02_objetivo2"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 2; las claves con "_" son intermedias (no van al xlsx)."""
    panel = panel_dpto_anio(celdas)
    r = {}
    # hojas sin sufijo: crudas; "_aj": ajustadas por altitud y edad materna
    for suf, p in [("", panel), ("_aj", panel_ajustado(celdas, panel))]:
        quinq = rt_dt_quinquenio(p)
        r |= {f"RP_DP_quinquenio{suf}": quinq, f"RP_DP_ancho{suf}": rt_dt_ancho(quinq),
              f"RP_DP_anual{suf}": rt_dt_anual(p),
              f"Medidas_resumen{suf}": medidas_resumen(p)}
    return {**r, "_panel": panel}


def imprimir(r: dict) -> None:
    print("\nOBJETIVO 2 - Desigualdades RT/DT (quinquenios y años)")
    panel, quinq = r["_panel"], r["RP_DP_quinquenio"]
    print("\n--- A. Panel departamento-año ---")
    print(f"filas: {len(panel):,}  dptos: {panel.dpto_res.nunique()}")
    print("\n--- B. RP/DP por quinquenio ---")
    no_elegibles = quinq[quinq["elegible"].eq(0)]
    print("NO elegibles por quinquenio:")
    if len(no_elegibles):
        print(no_elegibles[["quinq", "departamento", "con_peso", "pct_falt", "pct"]]
              .to_string(index=False))
    else:
        print("  Ninguno.")
    print("Referencia por quinquenio:")
    print(quinq[quinq["es_ref"].eq(1)][["quinq", "departamento", "ref"]]
          .drop_duplicates().to_string(index=False))
    for titulo, hoja in [("crudas", "Medidas_resumen"),
                         ("ajustadas por altitud y edad", "Medidas_resumen_aj")]:
        print(f"\n--- C. Medidas resumen ({titulo}) ---")
        resumen = r[hoja]
        print(resumen[resumen["tipo"].eq("Quinquenio")]
              [["periodo", "mu", "ref", "dpto_ref", "paf", "paf_li", "paf_ls",
                "theil", "theil_li", "theil_ls", "rt_ext_el"]]
              .round(2).to_string(index=False))
    print("\n--- RP/DP anual ---")
    anual = r["RP_DP_anual"]
    print(anual[anual["anio"].isin([2005, 2024])][["anio", "departamento", "pct", "ref", "rt"]]
          .round(2).to_string(index=False))


def graficos(r: dict) -> dict:
    cb.iniciar_estilo()
    return {"G7_RP_departamentos_quinquenios": grafico_g7(r["RP_DP_ancho"],
                                                           r["RP_DP_ancho_aj"]),
            "G8_Theil_PAF_anual": grafico_g8(r["Medidas_resumen"],
                                             r["Medidas_resumen_aj"])}
