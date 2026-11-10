# -*- coding: utf-8 -*-
# =============================================================================
#  TRABAJO FINAL - BAJO PESO AL NACER EN COLOMBIA 2005-2024
#  Python: obj4_concentracion.py
#  (equivale a 04_Objetivo4_Indice_Concentracion.do, v2)
#
#  Objetivo 4: desigualdades socioeconómicas del BPN entre departamentos con
#  curvas e índices de concentración (C, Wagstaff W y Erreygers E), ponderados
#  por nacidos. Ordenamiento principal: % de madres NO afiliadas al régimen
#  contributivo (10 variantes). Regresión conveniente de Kakwani et al.
#  (1997). Inferencia: bootstrap por departamento (1.000 réplicas; pareado
#  para la diferencia entre trienios). Estandarización indirecta por altitud
#  del municipio y edad materna.
#
#  Enfoque de superpoblación: los 33 departamentos se tratan como una
#  realización de un proceso; el bootstrap por departamento mide la
#  variabilidad del índice ante esa realización (los nacimientos son censales).
#  Salidas: Objetivo4_Indice_Concentracion_BPN.xlsx, G2-G6 y G11, log.
#  Las semillas replican las del do-file (20241 + ...); los replicados no
#  coinciden con Stata porque el generador de números aleatorios es distinto.
# =============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import comun_bpn as cb

EXCEL = "Objetivo4_Indice_Concentracion_BPN.xlsx"
ANIO_INI, ANIO_FIN = 2008, 2024          # el régimen es comparable desde 2008
REPS = 1_000                             # réplicas bootstrap
SEMILLA = 20241

# Variantes: (nombre, indicador h, ponderación w, orden rango, condición)
variantes = [
    ("Principal", "pct_bpn", "con_peso", "ses_nc", None),
    ("Estandarizado por altitud y edad materna", "pct_is", "n_s", "ses_nc", None),
    ("Sin Bogotá D.C.", "pct_bpn", "con_peso", "ses_nc", "dpto_res != 11"),
    ("Sin departamentos de altiplano", "pct_bpn", "con_peso", "ses_nc",
     "altiplano == 0"),
    ("Sin departamentos con >5 % sin peso", "pct_bpn", "con_peso", "ses_nc",
     "pct_falt <= 5"),
    ("Cota inferior (faltantes sin BPN)", "pct_inf", "nacidos", "ses_nc", None),
    ("Cota superior (faltantes con BPN)", "pct_sup", "nacidos", "ses_nc", None),
    ("BPN a término en partos únicos", "pct_term", "n_term", "ses_nc", None),
    ("Ordenamiento: % subsidiado", "pct_bpn", "con_peso", "ses_sub", None),
    ("Ordenamiento: % baja escolaridad", "pct_bpn", "con_peso", "ses_edu", None),
]
V_PRINCIPAL = 1
V_ESTAND = 2
V_SIN_BOGOTA = 3


# --------------------------------------------------------------------------- #
# A. Panel departamento-año (2008-2024) con estandarización indirecta
# --------------------------------------------------------------------------- #
def estandarizar_indirecta(d: pd.DataFrame) -> pd.DataFrame:
    """Estandarización indirecta por altitud y edad materna, celda a celda.

    Agrega n_s y casos_s (nacidos con peso y casos de BPN que entran al
    cálculo) y esperado_s: los casos esperados si la celda tuviera la tasa
    observada en su grupo (anio, alt_cat, edad6). Celdas sin edad (o sin
    alt_cat) quedan fuera: n_s, casos_s y esperado_s = 0.
    La tasa estandarizada (pct_is) se obtiene después, en calc_ind."""
    g = d.groupby(["anio", "alt_cat", "edad6"])
    p_celda = g["bpn_n"].transform("sum") / g["con_peso"].transform("sum")
    con_edad = d["edad6"].notna()
    return d.assign(n_s=d["con_peso"].where(con_edad, 0.0),
                    casos_s=d["bpn_n"].where(con_edad, 0.0),
                    esperado_s=(p_celda * d["con_peso"]).fillna(0.0))


def panel_obj4(celdas: pd.DataFrame) -> pd.DataFrame:
    d = celdas[celdas["dpto_res"].gt(0) & celdas["anio"].between(ANIO_INI, ANIO_FIN)]
    d = d.rename(columns={"n": "nacidos"})
    d = estandarizar_indirecta(d)
    # Altitud media del dpto-año (ponderada por nacimientos) y nacidos a >=2.000 m
    d = d.assign(n_alt=d["nacidos"] * d["altitud_m"].fillna(0.0).to_numpy(),
                 n_alto=d["nacidos"].where(d["alt_cat"].astype(float) >= 5, 0.0))

    panel = (d.groupby(["anio", "dpto_res"], as_index=False)
             .agg(nacidos=("nacidos", "sum"), con_peso=("con_peso", "sum"),
                  bpn_n=("bpn_n", "sum"), reg_conocido=("reg_conocido", "sum"),
                  n_nocontrib=("n_nocontrib", "sum"), n_sub=("n_sub", "sum"),
                  edu_conocido=("edu_conocido", "sum"), n_bajaedu=("n_bajaedu", "sum"),
                  n_term=("n_term", "sum"), bpn_term_n=("bpn_term_n", "sum"),
                  n_s=("n_s", "sum"), casos_s=("casos_s", "sum"),
                  esperado_s=("esperado_s", "sum"), n_alt=("n_alt", "sum"),
                  n_alto=("n_alto", "sum"), nacidos_alt=("nacidos", "sum")))
    panel = calc_ind(panel, "anio")

    # Altitud media del departamento (pond. por nacimientos) y altiplano:
    # más de la mitad de los nacimientos 2008-2024 en municipios >= 2.000 m
    g = panel.groupby("dpto_res", as_index=False).agg(
        a_alt=("n_alt", "sum"), n_alto=("n_alto", "sum"),
        n_total=("nacidos_alt", "sum"))
    g["alt_dpto"] = g["a_alt"] / g["n_total"]
    g["pct_alto2000"] = 100 * g["n_alto"] / g["n_total"]
    g["altiplano"] = (g["pct_alto2000"] > 50).astype(int)
    panel = panel.merge(g[["dpto_res", "alt_dpto", "pct_alto2000", "altiplano"]],
                        on="dpto_res")
    panel.drop(columns=["n_alt", "n_alto", "nacidos_alt"], inplace=True)
    panel.insert(1, "departamento", panel["dpto_res"].map(cb.DPTO_NOM))
    panel["anio"] = panel["anio"].astype(int)
    return panel.sort_values(["anio", "dpto_res"]).reset_index(drop=True)


def calc_ind(d: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """calc_ind del do-file: prevalencias, estandarización indirecta y SES."""
    d = d.copy()
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    d["pct_falt"] = 100 * (d["nacidos"] - d["con_peso"]) / d["nacidos"]
    d["pct_inf"] = 100 * d["bpn_n"] / d["nacidos"]
    d["pct_sup"] = 100 * (d["bpn_n"] + d["nacidos"] - d["con_peso"]) / d["nacidos"]
    d["pct_term"] = 100 * d["bpn_term_n"] / d["n_term"]
    d["pct_nc"] = 100 * d["n_nocontrib"] / d["reg_conocido"]
    d["pct_sub"] = 100 * d["n_sub"] / d["reg_conocido"]
    d["pct_bajaedu"] = 100 * d["n_bajaedu"] / d["edu_conocido"]
    d["ses_nc"] = 100 - d["pct_nc"]
    d["ses_sub"] = 100 - d["pct_sub"]
    d["ses_edu"] = 100 - d["pct_bajaedu"]
    g = d.groupby(grupo)
    d["mu_s"] = (g["casos_s"].transform("sum")
                 / g["n_s"].transform("sum") * 100)
    d["pct_is"] = 100 * (d["casos_s"] - d["esperado_s"]) / d["n_s"] + d["mu_s"]
    return d


# --------------------------------------------------------------------------- #
# ic_conc: índice de concentración por regresión conveniente (Kakwani 1997)
# --------------------------------------------------------------------------- #
def ic_conc(h: pd.Series, w: pd.Series, rango: pd.Series) -> dict:
    """C por regresión conveniente con errores robustos (vce(robust)); W y E."""
    fin = np.isfinite(h) & np.isfinite(w) & np.isfinite(rango)
    h, w, rango = h[fin].to_numpy(float), w[fin].to_numpy(float), rango[fin].to_numpy(float)
    o = np.argsort(rango, kind="stable")
    h, w = h[o], w[o]
    n = len(h)
    if n < 3:
        return {"C": np.nan, "se_reg": np.nan, "mu": np.nan, "W": np.nan,
                "E": np.nan, "N": n}
    Wt = w.sum()
    mu = (w * h).sum() / Wt
    r = (np.cumsum(w) - w / 2) / Wt
    var_r = (w * (r - 0.5) ** 2).sum() / Wt
    lhs = 2 * var_r * h / mu
    res = sm.WLS(lhs, sm.add_constant(np.column_stack([r])), weights=w).fit(cov_type="HC1")
    C, se = float(res.params[1]), float(res.bse[1])
    return {"C": C, "se_reg": se, "mu": mu, "W": C / (1 - mu / 100),
            "E": 4 * (mu / 100) * C, "N": n}


def ic_vec(h: np.ndarray, w: np.ndarray, rango: np.ndarray | None = None) -> np.ndarray:
    """Fórmula cerrada de la pendiente de la regresión conveniente (base del
    bootstrap). Si `rango` es None, h y w deben venir ya ordenados de menor a
    mayor condición. Devuelve C (escalar o vector de réplicas)."""
    if rango is not None:
        o = np.argsort(rango, axis=-1, kind="stable")
        h, w = np.take_along_axis(h, o, -1), np.take_along_axis(w, o, -1)
    Wt = w.sum(axis=-1, keepdims=True)
    r = (np.cumsum(w, axis=-1) - w / 2) / Wt
    mu = (w * h).sum(axis=-1, keepdims=True) / Wt
    var_r = (w * (r - 0.5) ** 2).sum(axis=-1, keepdims=True) / Wt
    lhs = 2 * var_r * h / mu
    rbar = (w * r).sum(axis=-1, keepdims=True) / Wt
    lbar = (w * lhs).sum(axis=-1, keepdims=True) / Wt
    covn = (w * (r - rbar) * (lhs - lbar)).sum(axis=-1)
    den = (w * (r - rbar) ** 2).sum(axis=-1)
    # sin keepdims: 2-D (réplicas,) o 1-D (escalar), según la entrada
    return covn / den


def bootstrap_ic(h: np.ndarray, w: np.ndarray, rango: np.ndarray, C: float,
                 seed: int, reps: int = REPS) -> dict:
    """Bootstrap por departamento: remuestrea dptos con reemplazo y recalcula
    C, W y E en cada réplica. IC normal centrado en el C observado `C`
    (como el bootstrap de Stata)."""
    n = len(h)
    o = np.argsort(rango, kind="stable")
    h_o, w_o = h[o], w[o]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    idx.sort(axis=1)              # h_o está ordenado por rango: así cada réplica también
    w_b = w_o[idx]
    h_b = h_o[idx]
    C_b = ic_vec(h_b, w_b)
    mu_b = (w_b * h_b).sum(1) / w_b.sum(1)
    se = C_b.std(ddof=1)
    return {"se_C": se, "li_C": C - 1.96 * se, "ls_C": C + 1.96 * se,
            "li_p": np.percentile(C_b, 2.5), "ls_p": np.percentile(C_b, 97.5),
            "se_W": (C_b / (1 - mu_b / 100)).std(ddof=1),
            "se_E": (4 * (mu_b / 100) * C_b).std(ddof=1), "C_b": C_b}


# --------------------------------------------------------------------------- #
# B1. Índices por año, las 10 variantes
# --------------------------------------------------------------------------- #
def C_anual(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for anio in range(ANIO_INI, ANIO_FIN + 1):
        dato = panel[panel["anio"] == anio]
        for v, (nombre, col_h, col_w, col_r, cond) in enumerate(variantes, start=1):
            sub = dato
            if cond:
                sub = dato.query(cond)
            r = ic_conc(sub[col_h], sub[col_w], sub[col_r])
            filas.append(dict(anio=anio, variante=nombre, se_reg=r["se_reg"], **{
                k: r[k] for k in ["C", "W", "E", "mu", "N"]}))
    return pd.DataFrame(filas)[["anio", "variante", "C", "se_reg", "W", "E", "mu", "N"]]


# --------------------------------------------------------------------------- #
# B2. Bootstrap por departamento, anual (principal y estandarizado)
# --------------------------------------------------------------------------- #
def C_anual_bootstrap(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for anio in range(ANIO_INI, ANIO_FIN + 1):
        dato = panel[panel["anio"] == anio]
        for v in (V_PRINCIPAL, V_ESTAND):
            nombre, col_h, col_w, col_r, _ = variantes[v - 1]
            h = dato[col_h].to_numpy(float)
            w = dato[col_w].to_numpy(float)
            rg = dato[col_r].to_numpy(float)
            punto = ic_conc(dato[col_h], dato[col_w], dato[col_r])
            bs = bootstrap_ic(h, w, rg, punto["C"], SEMILLA + anio + 100 * v)
            filas.append({"anio": anio, "variante": nombre, "C": punto["C"],
                          "se_bs": bs["se_C"], "li": bs["li_C"], "ls": bs["ls_C"],
                          "li_p": bs["li_p"], "ls_p": bs["ls_p"],
                          "W": punto["W"], "se_W": bs["se_W"],
                          "E": punto["E"], "se_E": bs["se_E"]})
    d = pd.DataFrame(filas)
    d["p"] = 2 * norm.sf(np.abs(d["C"] / d["se_bs"]))
    return d


# --------------------------------------------------------------------------- #
# C. Trienios 2008-2010 vs 2022-2024
# --------------------------------------------------------------------------- #
def trienio_agregar(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel[panel["anio"].between(2008, 2010) | panel["anio"].between(2022, 2024)]
    d = d.assign(periodo=np.where(d["anio"] <= 2010, 1, 2))
    d = (d.groupby(["periodo", "dpto_res"], as_index=False)
         .agg(nacidos=("nacidos", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum"), reg_conocido=("reg_conocido", "sum"),
              n_nocontrib=("n_nocontrib", "sum"), n_sub=("n_sub", "sum"),
              edu_conocido=("edu_conocido", "sum"), n_bajaedu=("n_bajaedu", "sum"),
              n_term=("n_term", "sum"), bpn_term_n=("bpn_term_n", "sum"),
              n_s=("n_s", "sum"), casos_s=("casos_s", "sum"),
              esperado_s=("esperado_s", "sum"),
              alt_dpto=("alt_dpto", "first"), pct_alto2000=("pct_alto2000", "first"),
              altiplano=("altiplano", "first")))
    d = calc_ind(d, "periodo")
    d.insert(1, "departamento", d["dpto_res"].map(cb.DPTO_NOM))
    return d.sort_values(["periodo", "dpto_res"]).reset_index(drop=True)


def C_trienios(trienios: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for v, (nombre, col_h, col_w, col_r, cond) in enumerate(variantes, start=1):
        for p, etq in [(1, "2008-2010"), (2, "2022-2024")]:
            sub = trienios[trienios["periodo"] == p]
            if cond:
                sub = sub.query(cond)
            punto = ic_conc(sub[col_h], sub[col_w], sub[col_r])
            bs = bootstrap_ic(sub[col_h].to_numpy(float), sub[col_w].to_numpy(float),
                              sub[col_r].to_numpy(float), punto["C"],
                              SEMILLA + 1000 * v + p)
            filas.append({"variante": nombre, "trienio": etq, "C": punto["C"],
                          "se_reg": punto["se_reg"], "se_bs": bs["se_C"],
                          "li": bs["li_C"], "ls": bs["ls_C"],
                          "W": punto["W"], "se_W": bs["se_W"],
                          "E": punto["E"], "se_E": bs["se_E"],
                          "mu": punto["mu"], "N": punto["N"]})
    d = pd.DataFrame(filas)
    d["p"] = 2 * norm.sf(np.abs(d["C"] / d["se_bs"]))
    return d[["variante", "trienio", "N", "mu", "C", "se_reg", "se_bs", "li",
              "ls", "p", "W", "se_W", "E", "se_E"]]


def diferencia_trienios(trienios: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap PAREADO por departamento: se remuestrean dptos con sus dos
    periodos, porque son los mismos 33 departamentos."""
    filas = []
    for v, (nombre, col_h, col_w, col_r, cond) in enumerate(variantes, start=1):
        sub = trienios
        if cond:
            sub = trienios.query(cond)
        t1 = sub[sub["periodo"] == 1].sort_values("dpto_res")
        t2 = sub[sub["periodo"] == 2].sort_values("dpto_res")
        comunes = sorted(set(t1["dpto_res"]) & set(t2["dpto_res"]))
        t1 = t1[t1["dpto_res"].isin(comunes)]
        t2 = t2[t2["dpto_res"].isin(comunes)]
        h1, w1 = t1[col_h].to_numpy(float), t1[col_w].to_numpy(float)
        h2, w2 = t2[col_h].to_numpy(float), t2[col_w].to_numpy(float)
        r1 = t1[col_r].to_numpy(float)
        r2 = t2[col_r].to_numpy(float)
        c1 = ic_conc(t1[col_h], t1[col_w], t1[col_r])["C"]
        c2 = ic_conc(t2[col_h], t2[col_w], t2[col_r])["C"]

        # t1 y t2 alineados por dpto_res: el mismo idx toma los mismos dptos
        # en ambos periodos, y cada réplica se reordena por su propio rango.
        rng = np.random.default_rng(SEMILLA + 5000 + v)
        G = len(comunes)
        idx = rng.integers(0, G, size=(REPS, G))
        C1_b = ic_vec(h1[idx], w1[idx], r1[idx])
        C2_b = ic_vec(h2[idx], w2[idx], r2[idx])
        dif_b = C2_b - C1_b
        d_obs = c2 - c1
        se = dif_b.std(ddof=1)
        filas.append({"variante": nombre, "C1": c1, "C2": c2, "dif": d_obs,
                      "se_bs": se, "li": d_obs - 1.96 * se,
                      "ls": d_obs + 1.96 * se, "p": 2 * norm.sf(abs(d_obs / se))})
    return pd.DataFrame(filas)


def orden_dptos(trienios: pd.DataFrame) -> pd.DataFrame:
    d = []
    for p in (1, 2):
        t = trienios[trienios["periodo"] == p].copy()
        t = t.sort_values("pct_nc").reset_index(drop=True)
        t["orden"] = len(t) - np.arange(len(t))          # 1 = más pobre (mayor nc)
        t["trienio"] = "2008-2010" if p == 1 else "2022-2024"
        d.append(t)
    d = pd.concat(d)
    return d[["trienio", "orden", "departamento", "pct_nc", "pct_sub",
              "pct_bajaedu", "pct_bpn", "pct_is", "pct_falt", "alt_dpto",
              "altiplano", "nacidos", "con_peso"]]


# --------------------------------------------------------------------------- #
# D. Curvas de concentración (trienios; 3 muestras)
# --------------------------------------------------------------------------- #
def curva(trienios: pd.DataFrame, muestra: int, periodo: int) -> pd.DataFrame:
    """muestra (numeración del do-file, no la de `variantes`):
    1 = principal, 2 = sin Bogotá, 3 = estandarizada."""
    t = trienios[trienios["periodo"] == periodo].sort_values("ses_nc")
    if muestra == 2:
        t = t[t["dpto_res"] != 11]
    w = t["con_peso"] if muestra != 3 else t["n_s"]
    b = t["bpn_n"] if muestra != 3 else t["pct_is"] / 100 * t["n_s"]
    Wt, Bt = w.sum(), b.sum()
    return pd.DataFrame({
        "muestra": muestra, "periodo": periodo, "dpto_res": t["dpto_res"].to_numpy(),
        "x": 100 * np.cumsum(w) / Wt, "y": 100 * np.cumsum(b) / Bt,
        "rotulo": np.where(t["dpto_res"].eq(11) & (muestra == 1), "Bogotá", ""),
    })


def curvas_datos(trienios: pd.DataFrame) -> pd.DataFrame:
    partes = [curva(trienios, m, p) for m in (1, 2, 3) for p in (1, 2)]
    d = pd.concat(partes, ignore_index=True)
    extras = pd.DataFrame([{"muestra": m, "periodo": p, "dpto_res": np.nan,
                            "x": 0.0, "y": 0.0, "rotulo": ""}
                           for m in (1, 2, 3) for p in (1, 2)])
    return pd.concat([d, extras], ignore_index=True).sort_values(
        ["muestra", "periodo", "x"]).reset_index(drop=True)


def _C_trienio(C_trienios_df: pd.DataFrame, m: int, p: int) -> str:
    noms = [variantes[V_PRINCIPAL - 1][0], variantes[V_SIN_BOGOTA - 1][0],
            variantes[V_ESTAND - 1][0]]
    etq = "2008-2010" if p == 1 else "2022-2024"
    v = C_trienios_df[(C_trienios_df["variante"] == noms[m - 1])
                      & (C_trienios_df["trienio"] == etq)]["C"].iloc[0]
    return f"{v:.3f}".replace(".", ",", 1)


def grafico_curvas(d: pd.DataFrame, t: pd.DataFrame) -> dict:
    """G3-G6 y G11: devuelve {nombre: figura}."""
    import matplotlib.pyplot as plt
    def axial(ax, m, p, color, titulo, texto_C, puntos=False, rotul=False):
        d_c = d[(d["muestra"] == m) & (d["periodo"] == p)]
        x, y = d_c["x"].to_numpy(), d_c["y"].to_numpy()
        ax.fill_between(x, y, x, color=color["area"], lw=0)
        ax.plot([0, 100], [0, 100], color="black", lw=1.6, label="Igualdad")
        ax.plot(x, y, color=color["linea"], lw=2.4)
        if puntos:
            # puntos observados = todos los departamentos de la muestra;
            # Bogotá en rojo solo cuando se rotula (G3)
            obs = d_c[d_c["dpto_res"].notna()]
            ax.scatter(obs["x"], obs["y"], s=14, color=color["linea"])
            if rotul:
                bog = d_c[d_c["dpto_res"] == 11]
                if len(bog):
                    bx, by = float(bog["x"].iloc[0]), float(bog["y"].iloc[0])
                    ax.scatter([bx], [by], s=28, color="#C00000", zorder=5)
                    ax.annotate("Bogotá", (bx, by), xytext=(6, 8),
                                textcoords="offset points", fontsize=8,
                                color="#C00000")
        ax.text(0.07, 0.88, texto_C, fontsize=10, transform=ax.transAxes)
        axial_estilado(ax, titulo)

    periodos = [(1, "2008-2010"), (2, "2022-2024")]
    figs = {}

    # G3. Principal, los dos trienios con área de concentración
    fig, ejes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, (p, etq) in zip(ejes, periodos):
        axial(ax, 1, p, {"area": "#FFE699", "linea": "#08306B"}, etq,
              f"C = {_C_trienio(t, 1, p)}", puntos=True, rotul=True)
    fig.suptitle("Curvas de concentración del bajo peso al nacer en Colombia",
                 fontsize=12)
    fig.text(0.01, 0.005, "Ordenamiento: % de madres no afiliadas al régimen\n"
             "contributivo (mayor % = más pobre). Ponderación: nacidos vivos.\n"
             "Curva bajo la diagonal: el BPN se concentra en los departamentos\n"
             "menos pobres. Fuente: DANE, Estadísticas Vitales. Cálculos propios.",
             fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figs["G3_curvas_concentracion_trienios"] = fig

    # G4. Superpuestas
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for p, etq, c in [(1, "2008-2010", "0.60"), (2, "2022-2024", "#08306B")]:
        d_c = d[(d["muestra"] == 1) & (d["periodo"] == p)]
        ax.plot(d_c["x"], d_c["y"], color=c, ls="--" if p == 1 else "-",
                lw=2.2,
                label=f"{etq} (C = {_C_trienio(t, 1, p)})")
    ax.plot([0, 100], [0, 100], color="black", lw=1.4, label="Igualdad")
    axial_estilado(ax, "Curvas de concentración del bajo peso al nacer")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    figs["G4_curvas_superpuestas"] = fig

    # G5. Sin Bogotá
    fig, ejes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, (p, etq) in zip(ejes, periodos):
        axial(ax, 2, p, {"area": "#FDD07A", "linea": "#A63603"}, etq,
              f"C = {_C_trienio(t, 2, p)}", puntos=True)
    fig.suptitle("Curvas de concentración del bajo peso al nacer, sin Bogotá D.C.",
                 fontsize=12)
    fig.text(0.01, 0.005, "32 departamentos. Ordenamiento: % de madres no afiliadas\n"
             "al régimen contributivo. Ponderación: nacidos vivos. Fuente: DANE,\n"
             "Estadísticas Vitales. Cálculos propios.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    figs["G5_curvas_sin_Bogota_trienios"] = fig

    # G6. Con vs. sin Bogotá, por trienio
    fig, ejes = plt.subplots(1, 2, figsize=(10, 5.6))
    for ax, (p, etq) in zip(ejes, periodos):
        axial(ax, 1, p, {"area": "#FFE699", "linea": "#08306B"}, etq, "",
              puntos=True)
        d2 = d[(d["muestra"] == 2) & (d["periodo"] == p)]
        ax.plot(d2["x"], d2["y"], color="#A63603", ls="--", lw=2.2)
        C1, C2 = _C_trienio(t, 1, p), _C_trienio(t, 2, p)
        ax.legend(handles=[
            plt.Line2D([], [], color="#08306B", lw=2.4, label=f"Con Bogotá (C = {C1})"),
            plt.Line2D([], [], color="#A63603", ls="--", lw=2.2,
                       label=f"Sin Bogotá (C = {C2})"),
            plt.Line2D([], [], color="black", lw=1.6, label="Igualdad")],
            loc="lower right", fontsize=7.5)
    fig.suptitle("Curvas de concentración del BPN con y sin Bogotá D.C.", fontsize=12)
    fig.text(0.01, 0.005, "En la curva sin Bogotá, el eje X acumula los nacidos de\n"
             "los otros 32 departamentos. Fuente: DANE, Estadísticas Vitales.\n"
             "Cálculos propios.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    figs["G6_con_sin_Bogota_trienios"] = fig

    # G11. Observada vs. estandarizada, por trienio
    fig, ejes = plt.subplots(1, 2, figsize=(10, 5.6))
    for ax, (p, etq) in zip(ejes, periodos):
        axial(ax, 1, p, {"area": "#FFE699", "linea": "#08306B"}, etq, "", puntos=True)
        d3 = d[(d["muestra"] == 3) & (d["periodo"] == p)]
        ax.plot(d3["x"], d3["y"], color="#006D2C", ls="--", lw=2.2)
        C1, C3 = _C_trienio(t, 1, p), _C_trienio(t, 3, p)
        ax.legend(handles=[
            plt.Line2D([], [], color="#08306B", lw=2.4, label=f"Observada (C = {C1})"),
            plt.Line2D([], [], color="#006D2C", ls="--", lw=2.2,
                       label=f"Estandarizada (C = {C3})"),
            plt.Line2D([], [], color="black", lw=1.6, label="Igualdad")],
            loc="lower right", fontsize=7.5)
    fig.suptitle("Curvas de concentración del BPN observada y estandarizada "
                 "por altitud y edad", fontsize=12)
    fig.text(0.01, 0.005, "Estandarización indirecta por altitud del municipio de\n"
             "residencia (6 grupos) y edad materna (6 grupos). Fuente: DANE,\n"
             "Estadísticas Vitales. Cálculos propios.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    figs["G11_curvas_estandarizadas_trienios"] = fig
    return figs


def axial_estilado(ax, titulo: str) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel("% acumulado de nacidos vivos\n"
                  "(departamentos del más pobre al menos pobre)", fontsize=8.5)
    ax.set_ylabel("% acumulado de casos\nde bajo peso al nacer", fontsize=8.5)
    ax.set_title(titulo, fontsize=11)


# --------------------------------------------------------------------------- #
# G2. C anual: principal (IC bootstrap) y estandarizado y sin Bogotá
# --------------------------------------------------------------------------- #
def grafico_g2(boot: pd.DataFrame, anual: pd.DataFrame):
    import matplotlib.pyplot as plt
    pr = boot[boot["variante"] == variantes[V_PRINCIPAL - 1][0]]
    anual_std = anual[anual["variante"] == variantes[V_ESTAND - 1][0]]
    anual_sb = anual[anual["variante"] == variantes[V_SIN_BOGOTA - 1][0]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(pr["anio"], pr["ls"], pr["li"], color="#C6DBEF", lw=0)
    ax.plot(pr["anio"], pr["C"], color="#08306B", marker="o", ms=4, lw=1.8,
            label="Principal")
    ax.plot(anual_std["anio"], anual_std["C"], color="#A63603", ls="--", marker="D",
            ms=4, label="Estandarizado por altitud y edad")
    ax.plot(anual_sb["anio"], anual_sb["C"], color="#888888", ls="-.", marker="^",
            ms=4, label="Sin Bogotá")
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.set_xticks(range(ANIO_INI, ANIO_FIN + 1, 2))
    ax.set_ylabel("Índice de concentración (C)")
    ax.set_xlabel("Año")
    ax.set_title("Índice de concentración del bajo peso al nacer, 2008-2024\n"
                 "Departamentos ordenados por % de madres no afiliadas al "
                 "régimen contributivo", fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5, ncols=1)
    ax.annotate("C > 0: el BPN se concentra en los departamentos menos pobres;\n"
                "C < 0: en los más pobres. IC 95 % bootstrap para Principal.\n"
                "Fuente: DANE, Estadísticas Vitales. Cálculos propios.",
                xy=(1, -0.16), xycoords="axes fraction", ha="right", fontsize=7,
                va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# E. Validación del ordenamiento contra NBI/IPM (si existe la plantilla)
# --------------------------------------------------------------------------- #
def validacion_proxy(panel: pd.DataFrame) -> pd.DataFrame | None:
    ruta = cb.DATOS / "nbi_ipm_departamentos.csv"
    if not ruta.exists():
        print("No se encontró la plantilla NBI/IPM "
              "(3.0 Datos/nbi_ipm_departamentos.csv); se omite la validación.")
        return None
    val = pd.read_csv(ruta)
    val["dpto_res"] = pd.to_numeric(val["dpto_res"], errors="coerce").astype("Int64")
    d = panel[panel["anio"].isin([2008, 2018])][["dpto_res", "anio", "pct_nc", "pct_sub"]]
    d = d.pivot(index="dpto_res", columns="anio", values=["pct_nc", "pct_sub"])
    d.columns = [f"{e}{a}" for e, a in d.columns]
    d = d.reset_index().merge(val, on="dpto_res", how="left")
    from scipy.stats import spearmanr
    filas = []
    for a, b in [("pct_nc2008", "nbi_2005"), ("pct_nc2018", "nbi_2018"),
                 ("pct_nc2018", "ipm_2018"), ("pct_sub2008", "nbi_2005"),
                 ("pct_sub2018", "nbi_2018"), ("pct_sub2018", "ipm_2018")]:
        if b not in d or d[[a, b]].dropna().shape[0] < 5:
            print(f"  {b}: sin datos suficientes en la plantilla; se omite.")
            continue
        par = d[[a, b]].dropna()
        rho, pv = spearmanr(par[a], par[b])
        filas.append({"indicador": a, "referencia": b, "rho": rho, "p": pv,
                      "N": len(par)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)
# --------------------------------------------------------------------------- #
LOG = "log_04_objetivo4"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 4; las claves con "_" son intermedias (no van al xlsx)."""
    panel = panel_obj4(celdas)
    trienios = trienio_agregar(panel)
    r = {
        "Panel_dpto_anio": panel,
        "C_anual_variantes": C_anual(panel),
        "C_anual_bootstrap": C_anual_bootstrap(panel),
        "C_trienios": C_trienios(trienios),
        "Diferencia_trienios": diferencia_trienios(trienios),
        "Orden_dptos_trienios": orden_dptos(trienios),
        "_curvas": curvas_datos(trienios),
    }
    valid = validacion_proxy(panel)
    if valid is not None:
        r["Validacion_proxy"] = valid
    return r


def imprimir(r: dict) -> None:
    print("\nOBJETIVO 4 - Índice de concentración del BPN (2008-2024)")
    panel = r["Panel_dpto_anio"]
    print("\n--- A. Panel departamento-año ---")
    alt = panel[panel["anio"].eq(ANIO_FIN) & panel["altiplano"].eq(1)]
    print("Departamentos de altiplano (>50 % de nacidos a >=2.000 m):")
    print(alt[["departamento", "alt_dpto", "pct_alto2000", "pct_nc", "pct_bpn"]]
          .round(2).to_string(index=False))
    print("\n--- B2. Bootstrap anual (principal y estandarizado) ---")
    for f in r["C_anual_bootstrap"].itertuples():
        print(f"  {f.anio} - {f.variante}: C = {f.C:.4f}  EE bootstrap = {f.se_bs:.4f}")
    print("\n--- C. Trienios 2008-2010 vs 2022-2024 ---")
    for f in r["Diferencia_trienios"].itertuples():
        print(f"  {f.variante}: cambio en C = {f.dif:.4f}  EE = {f.se_bs:.4f}")


def graficos(r: dict) -> dict:
    cb.iniciar_estilo()
    return {"G2_indice_concentracion_anual": grafico_g2(r["C_anual_bootstrap"],
                                                        r["C_anual_variantes"]),
            **grafico_curvas(r["_curvas"], r["C_trienios"])}
