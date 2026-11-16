#  Objetivo 2: desigualdades departamentales en el BPN, por quinquenio y por
#  año. Jerarquía de las medidas:
#    1. Theil y MLD (dispersión entre departamentos; no dependen de ninguna
#       referencia).
#    2. RP y DP frente a la prevalencia de las 33 unidades (rt_nac, dt_nac).
#    3. RP, DP y PAR/PAF frente a una referencia de baja prevalencia: lo
#       "mejor alcanzable" entre departamentos, no una meta causal.
#  La referencia principal es FIJA (hojas "_fija"): el grupo DPTOS_REF, igual
#  en todos los periodos. La referencia móvil (el departamento elegible de
#  menor prevalencia de cada periodo) queda como sensibilidad: cambia de
#  departamento y, por elegirse como mínimo, sesga el PAF hacia arriba.
#  Todo se calcula con la prevalencia cruda y con la ajustada por altitud y
#  edad materna (hojas "_aj"; la principal es la ajustada). IC 95 %: binomial
#  (delta) para RP y DP; simulación binomial para Theil y PAF.
#  Pendiente: descomponer la desigualdad entre departamentos y educación a la
#  vez (el PAF departamental no dice cuánto es de cada una).
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
# Referencia fija: los cinco departamentos, elegibles en los cuatro
# quinquenios, con menor prevalencia ajustada por altitud y edad en 2005-2024
# (Caquetá, Putumayo, Casanare, Meta y Huila). Se eligieron una sola vez con
# toda la serie; el grupo no cambia aunque un miembro deje de ser elegible en
# algún año (la hoja Referencia_fija dice cuándo y valida la elección).
DPTOS_REF = (18, 86, 85, 50, 41)
NOMBRE_REF = "Grupo fijo (5 departamentos)"
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


def _referencia(d: pd.DataFrame, grupo: str, fija: bool) -> pd.DataFrame:
    """Agrega la referencia de cada `grupo` (periodo): su prevalencia (ref),
    sus nacidos con peso (n_ref) y qué filas la forman (es_ref). Fija: la
    prevalencia agrupada de DPTOS_REF. Móvil: el elegible de menor prevalencia."""
    if fija:
        es_ref = d["dpto_res"].isin(DPTOS_REF)
        r = d[es_ref].groupby(grupo)[["bpn_n", "con_peso"]].sum()
        return d.assign(ref=d[grupo].map(100 * r["bpn_n"] / r["con_peso"]),
                        n_ref=d[grupo].map(r["con_peso"]), es_ref=es_ref)
    ref = d["pct"].where(d["elegible"]).groupby(d[grupo]).transform("min")
    es_ref = d["elegible"].astype(bool) & d["pct"].eq(ref)
    return d.assign(ref=ref, es_ref=es_ref,
                    n_ref=d["con_peso"].where(es_ref).groupby(d[grupo]).transform("max"))


def _ic_rt_dt(d: pd.DataFrame, fija: bool) -> pd.DataFrame:
    """IC 95 % de rt y dt de cada departamento frente a la referencia de su
    periodo (varianza binomial, método delta). Con la referencia móvil, se toma
    como dada: el intervalo no incluye la incertidumbre de haberla elegido por
    ser la mínima, y su propia fila no tiene error. Con la fija todas las
    filas lo tienen (en los cinco del grupo se ignora que son parte de él)."""
    p, p0, n, n_ref = d["pct"] / 100, d["ref"] / 100, d["con_peso"], d["n_ref"]
    se_log = np.sqrt((1 - p) / (n * p) + (1 - p0) / (n_ref * p0))
    se_dif = 100 * np.sqrt(p * (1 - p) / n + p0 * (1 - p0) / n_ref)
    if not fija:
        se_log, se_dif = se_log.where(~d["es_ref"], 0.0), se_dif.where(~d["es_ref"], 0.0)
    return d.assign(rt_li=d["rt"] * np.exp(-1.96 * se_log),
                    rt_ls=d["rt"] * np.exp(1.96 * se_log),
                    dt_li=d["dt"] - 1.96 * se_dif, dt_ls=d["dt"] + 1.96 * se_dif)


def _ic_resumen(n: np.ndarray, p: np.ndarray, elegible: np.ndarray,
                es_ref: np.ndarray | None = None) -> dict:
    """IC 95 % percentil de Theil y PAF por simulación binomial (REPS réplicas):
    en cada una se sortean los casos de cada departamento, Binomial(n, p), y
    se recalculan la media, la referencia y los dos índices. La referencia es
    la elegible mínima de la réplica o, con `es_ref`, la del grupo fijo."""
    b = np.random.default_rng(SEMILLA).binomial(n.round().astype(int), p,
                                                size=(REPS, len(n)))
    pr, mu = b / n, b.sum(1) / n.sum()
    r = pr / mu[:, None]
    theil = 1000 * (n / n.sum() * np.where(r > 0, r * np.log(np.where(r > 0, r, 1)),
                                           0.0)).sum(1)
    ref = (np.where(elegible, pr, np.inf).min(1) if es_ref is None
           else b[:, es_ref].sum(1) / n[es_ref].sum())
    paf = 100 * (mu - ref) / mu
    lim = lambda x: np.percentile(x, [2.5, 97.5]) if np.isfinite(x).all() else (np.nan,) * 2
    (t0, t1), (f0, f1) = lim(theil), lim(paf)
    return {"theil_li": t0, "theil_ls": t1, "paf_li": f0, "paf_ls": f1}


def _quinquenios(panel: pd.DataFrame) -> pd.DataFrame:
    return _con_elegible(panel.groupby(["dpto_res", "departamento", "quinq"],
                                       as_index=False)[["nacidos", "con_peso", "bpn_n"]]
                         .sum(), MIN_QUINQ)


# B. Quinquenios: RP/DP frente a la referencia y a la prevalencia de las 33

def rt_dt_quinquenio(d: pd.DataFrame, fija: bool = False) -> pd.DataFrame:
    d = _quinquenios(d)
    d["li"], d["ls"] = cb.wilson(d["bpn_n"], d["con_peso"])
    g = d.groupby("quinq")
    d = _referencia(d, "quinq", fija)
    d["mu"] = g["bpn_n"].transform("sum") / g["con_peso"].transform("sum") * 100
    # Nombre de la referencia móvil: el dpto de mayor código entre los empatados
    dref = (d[d["es_ref"]].sort_values("dpto_res")
            .groupby("quinq")["departamento"].last())
    d["dpto_ref"] = NOMBRE_REF if fija else d["quinq"].map(dref)
    d["rt"] = d["pct"] / d["ref"]
    d["dt"] = d["pct"] - d["ref"]
    d["rt_nac"] = d["pct"] / d["mu"]
    d["dt_nac"] = d["pct"] - d["mu"]
    d = _ic_rt_dt(d, fija)
    d["elegible"] = d["elegible"].astype(int)
    d["es_ref"] = d["es_ref"].astype(int)
    d["quinq"] = d["quinq"].map(QUINQ_ETQ)
    return d[["quinq", "dpto_res", "departamento", "nacidos", "con_peso",
              "pct_falt", "elegible", "bpn_n", "pct", "li", "ls", "ref",
              "dpto_ref", "rt", "rt_li", "rt_ls", "dt", "dt_li", "dt_ls", "mu",
              "rt_nac", "dt_nac", "es_ref", *(["n_ref"] if fija else [])]]


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


def rt_dt_anual(d: pd.DataFrame, fija: bool = False) -> pd.DataFrame:
    d = _referencia(_con_elegible(d, MIN_ANUAL), "anio", fija)
    d["rt"] = d["pct"] / d["ref"]
    d["dt"] = d["pct"] - d["ref"]
    d = _ic_rt_dt(d, fija)
    d["elegible"] = d["elegible"].astype(int)
    return d[["anio", "dpto_res", "departamento", "nacidos", "con_peso",
              "pct_falt", "elegible", "bpn_n", "pct", "ref", "rt", "rt_li", "rt_ls",
              "dt", "dt_li", "dt_ls", *(["n_ref"] if fija else [])]].sort_values(
        ["anio", "dpto_res"]).reset_index(drop=True)


# C. Medidas resumen por periodo (programa resumen_desig del análisis)

def resumen_desig(d: pd.DataFrame, periodo: str, tipo: str,
                  fija: bool = False) -> dict:
    """Medidas de desigualdad de un periodo: mu, referencia (móvil o fija),
    PAR/PAF, exceso, Theil y MLD ponderados por nacidos con peso, y extremos."""
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
    es_ref = da["dpto_res"].isin(DPTOS_REF).to_numpy() if fija else None
    if fija:
        grupo = da[es_ref]
        ref, nref = 100 * grupo["bpn_n"].sum() / grupo["con_peso"].sum(), NOMBRE_REF
    else:
        ref = eleg["pct"].min()
    if np.isnan(ref):
        nref = ""
        rt_el = dt_el = np.nan
    else:
        if not fija:
            nref = (eleg[eleg["pct"].eq(ref)].sort_values("dpto_res")
                    ["departamento"].iloc[-1])      # último departamento del quinquenio
        rt_el = pmax / ref
        dt_el = pmax - ref
    ic = _ic_resumen(da["con_peso"].to_numpy(float), da["pct"].to_numpy(float) / 100,
                     da["elegible"].to_numpy(bool), es_ref)
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


def medidas_resumen(panel: pd.DataFrame, fija: bool = False) -> pd.DataFrame:
    filas = []
    quinq = _quinquenios(panel)
    for q, etq in QUINQ_ETQ.items():
        filas.append(resumen_desig(quinq[quinq["quinq"] == q], etq, "Quinquenio", fija))
    for anio in range(cb.ANIO_INI, cb.ANIO_FIN + 1):
        d_anio = _con_elegible(panel[panel["anio"] == anio], MIN_ANUAL)
        filas.append(resumen_desig(d_anio, str(anio), "Año", fija))
    return pd.DataFrame(filas)[
        ["periodo", "tipo", "mu", "ref", "dpto_ref", "par", "paf", "paf_li",
         "paf_ls", "exceso", "theil", "theil_li", "theil_ls", "mld", "pmax", "dpto_max", "pmin", "dpto_min", "rt_ext",
         "dt_ext", "rt_ext_el", "dt_ext_el"]]


# D. La referencia fija: sensibilidad del PAF y validación de la elección

def paf_referencias(panel: pd.DataFrame, panel_aj: pd.DataFrame) -> pd.DataFrame:
    """PAF de cada quinquenio con cada referencia posible: cada uno de los
    cinco departamentos del grupo, el grupo, el mínimo elegible del periodo
    (móvil) y el promedio nacional (PAF = 0 por definición)."""
    filas = []
    for version, p in [("Ajustada", panel_aj), ("Cruda", panel)]:
        q = _quinquenios(p)
        for k, etq in QUINQ_ETQ.items():
            d = q[q["quinq"] == k]
            prev = lambda g: 100 * g["bpn_n"].sum() / g["con_peso"].sum()
            mu = prev(d)
            refs = {cb.DPTO_NOM[c]: prev(d[d["dpto_res"].eq(c)]) for c in DPTOS_REF}
            refs |= {NOMBRE_REF: prev(d[d["dpto_res"].isin(DPTOS_REF)]),
                     "Mínimo elegible del periodo (móvil)": d["pct"].where(d["elegible"]).min(),
                     "Promedio nacional": mu}
            filas += [{"prevalencia": version, "periodo": etq, "referencia": nombre,
                       "ref": ref, "mu": mu, "paf": 100 * (mu - ref) / mu}
                      for nombre, ref in refs.items()]
    return pd.DataFrame(filas)


def referencia_fija(celdas: pd.DataFrame, panel: pd.DataFrame,
                    panel_aj: pd.DataFrame) -> pd.DataFrame:
    """Validación de DPTOS_REF: prevalencia ajustada de cada miembro en toda
    la serie y en sus dos mitades, con su puesto entre los departamentos
    elegibles en los cuatro quinquenios (¿se habría elegido igual con solo
    2005-2014?); % sin peso total, en cabecera y en el resto (faltante
    selectivo); y años en que el miembro no es elegible como referencia anual."""
    q = _quinquenios(panel)
    siempre = q.groupby("dpto_res")["elegible"].sum().eq(len(QUINQ_ETQ))
    siempre = siempre[siempre].index
    miembros = q[q["dpto_res"].isin(DPTOS_REF)]
    assert len(miembros) == len(DPTOS_REF) * len(QUINQ_ETQ) and miembros["elegible"].all(), \
        "un departamento de DPTOS_REF no es elegible en algún quinquenio"

    def prev(p, a0, a1):
        k = (p[p["anio"].between(a0, a1) & p["dpto_res"].isin(siempre)]
             .groupby("dpto_res")[["bpn_n", "con_peso"]].sum())
        return 100 * k["bpn_n"] / k["con_peso"]

    d = pd.DataFrame({"con_peso": panel.groupby("dpto_res")["con_peso"].sum()
                      .reindex(siempre)})
    for etq, (a0, a1) in {"2005_2024": (2005, 2024), "2005_2014": (2005, 2014),
                          "2015_2024": (2015, 2024)}.items():
        d[f"pct_aj_{etq}"] = prev(panel_aj, a0, a1)
        d[f"puesto_{etq}"] = d[f"pct_aj_{etq}"].rank().astype(int)
    c = celdas[celdas["dpto_res"].isin(siempre)]
    falta = lambda k: 100 * (1 - k.groupby("dpto_res")["con_peso"].sum()
                             / k.groupby("dpto_res")["n"].sum())
    d["pct_sin_peso"] = falta(c)
    d["pct_sin_peso_cabecera"] = falta(c[c["area2"].eq(1)])
    d["pct_sin_peso_resto"] = falta(c[c["area2"].eq(2)])
    anual = _con_elegible(panel, MIN_ANUAL)
    d["anios_no_elegible"] = (anual[~anual["elegible"]].groupby("dpto_res")["anio"]
                              .agg(lambda a: ", ".join(map(str, a)))).reindex(d.index).fillna("")
    d = d.reset_index()
    d.insert(1, "departamento", d["dpto_res"].map(cb.DPTO_NOM))
    d.insert(2, "en_grupo", d["dpto_res"].isin(DPTOS_REF).astype(int))
    return d.sort_values("pct_aj_2005_2024").reset_index(drop=True)


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
             "Referencia fija: prevalencia agrupada de Caquetá, Putumayo, Casanare, "
             "Meta y Huila en cada quinquenio.\n"
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
             "fija (cinco departamentos). Bandas: IC 95 % por simulación binomial.\n"
             "Fuente: DANE, Estadísticas Vitales. Cálculos propios.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)

LOG = "log_02_objetivo2"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 2; las claves con "_" son intermedias (no van al xlsx)."""
    panel = panel_dpto_anio(celdas)
    panel_aj = panel_ajustado(celdas, panel)
    r = {}
    # hojas sin sufijo: crudas; "_aj": ajustadas por altitud y edad materna;
    # "_fija": con la referencia fija (las principales son las "_aj_fija")
    for fija in (False, True):
        for suf, p in [("", panel), ("_aj", panel_aj)]:
            suf += "_fija" if fija else ""
            quinq = rt_dt_quinquenio(p, fija)
            r |= {f"RP_DP_quinquenio{suf}": quinq, f"RP_DP_ancho{suf}": rt_dt_ancho(quinq),
                  f"RP_DP_anual{suf}": rt_dt_anual(p, fija),
                  f"Medidas_resumen{suf}": medidas_resumen(p, fija)}
    return {**r, "PAF_referencias": paf_referencias(panel, panel_aj),
            "Referencia_fija": referencia_fija(celdas, panel, panel_aj),
            "_panel": panel}


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
    print("\n--- Referencia fija: validación ---")
    print(r["Referencia_fija"].head(9).round(2).to_string(index=False))
    paf = r["PAF_referencias"]
    print("\nPAF (%) según la referencia, prevalencia ajustada:")
    print(paf[paf["prevalencia"].eq("Ajustada")]
          .pivot(index="referencia", columns="periodo", values="paf").round(1).to_string())
    for titulo, hoja in [("ajustadas, referencia fija [principal]", "Medidas_resumen_aj_fija"),
                         ("crudas, referencia fija", "Medidas_resumen_fija"),
                         ("crudas, referencia móvil", "Medidas_resumen"),
                         ("ajustadas, referencia móvil", "Medidas_resumen_aj")]:
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
    return {"G7_RP_departamentos_quinquenios": grafico_g7(r["RP_DP_ancho_fija"],
                                                           r["RP_DP_ancho_aj_fija"]),
            "G8_Theil_PAF_anual": grafico_g8(r["Medidas_resumen_fija"],
                                             r["Medidas_resumen_aj_fija"])}
