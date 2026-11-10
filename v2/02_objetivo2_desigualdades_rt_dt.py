# -*- coding: utf-8 -*-
# =============================================================================
#  TRABAJO FINAL - BAJO PESO AL NACER EN COLOMBIA 2005-2024
#  Python: 02_objetivo2_desigualdades_rt_dt.py
#  (equivale a 02_Objetivo2_Desigualdades_RT_DT.do, v2)
#
#  Objetivo 2: desigualdades departamentales en el BPN mediante razones y
#  diferencias de prevalencias (RP/DP) frente a una referencia elegible
#  (menor prevalencia entre departamentos con pesos y cobertura de registro
#  suficientes) y frente a la prevalencia de las 33 unidades, por quinquenio
#  y por año. Medidas resumen: Theil, MLD, PAR/PAF y casos en exceso.
#  Salidas: Objetivo2_Desigualdades_RT_DT.xlsx, G7 y G8 png, log.
# =============================================================================
import numpy as np
import pandas as pd

import comun_bpn as cb

EXCEL = "Objetivo2_Desigualdades_RT_DT.xlsx"
MIN_QUINQ = 10_000     # nacidos con peso conocido en el quinquenio
MIN_ANUAL = 2_000      # nacidos con peso conocido en el año
MAX_FALT = 5           # % máximo de nacimientos sin información de peso
QUINQ_ETQ = {1: "2005-2009", 2: "2010-2014", 3: "2015-2019", 4: "2020-2024"}


# --------------------------------------------------------------------------- #
# A. Panel departamento-año
# --------------------------------------------------------------------------- #
def panel_dpto_anio(celdas: pd.DataFrame) -> pd.DataFrame:
    d = (celdas[celdas["dpto_res"].gt(0)]
         .groupby(["dpto_res", "anio"])
         .agg(nacidos=("n", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum")).reset_index())
    d["departamento"] = d["dpto_res"].map(cb.DPTO_NOM)
    d["quinq"] = 1 + (d["anio"] - 2005) // 5
    return d


def _prevalencias(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["pct"] = 100 * d["bpn_n"] / d["con_peso"]
    d["pct_falt"] = 100 * (d["nacidos"] - d["con_peso"]) / d["nacidos"]
    return d


# --------------------------------------------------------------------------- #
# B. Quinquenios: RP/DP frente a la referencia y a la prevalencia de las 33
# --------------------------------------------------------------------------- #
def rt_dt_quinquenio(d: pd.DataFrame) -> pd.DataFrame:
    d = _prevalencias(d.groupby(["dpto_res", "departamento", "quinq"],
                                as_index=False)[["nacidos", "con_peso", "bpn_n"]]
                      .sum())
    d["elegible"] = (d["con_peso"] >= MIN_QUINQ) & (d["pct_falt"] <= MAX_FALT)
    d["li"], d["ls"] = cb.wilson(d["bpn_n"], d["con_peso"])
    g = d.groupby("quinq")
    d["ref"] = d["pct"].where(d["elegible"]).groupby(d["quinq"]).transform("min")
    d["mu"] = g["bpn_n"].transform("sum") / g["con_peso"].transform("sum") * 100
    d["es_ref"] = (d["elegible"] & d["pct"].eq(d["ref"])).astype(int)
    # Nombre del dpto de referencia: el de mayor código entre los empatados
    # (replica el bysort quinq (es_ref) + [_N] del do-file)
    dref = (d[d["es_ref"].eq(1)].sort_values("dpto_res")
            .groupby("quinq")["departamento"].last())
    d["dpto_ref"] = d["quinq"].map(dref)
    d["rt"] = d["pct"] / d["ref"]
    d["dt"] = d["pct"] - d["ref"]
    d["rt_nac"] = d["pct"] / d["mu"]
    d["dt_nac"] = d["pct"] - d["mu"]
    d["elegible"] = d["elegible"].astype(int)
    d["quinq"] = d["quinq"].map(QUINQ_ETQ)
    return d[["quinq", "dpto_res", "departamento", "nacidos", "con_peso",
              "pct_falt", "elegible", "bpn_n", "pct", "li", "ls", "ref",
              "dpto_ref", "rt", "dt", "mu", "rt_nac", "dt_nac", "es_ref"]]


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
    d = _prevalencias(d)
    d["elegible"] = (d["con_peso"] >= MIN_ANUAL) & (d["pct_falt"] <= MAX_FALT)
    g = d.groupby("anio")
    d["ref"] = d["pct"].where(d["elegible"]).groupby(d["anio"]).transform("min")
    d["rt"] = d["pct"] / d["ref"]
    d["dt"] = d["pct"] - d["ref"]
    d["elegible"] = d["elegible"].astype(int)
    return d[["anio", "dpto_res", "departamento", "nacidos", "con_peso",
              "pct_falt", "elegible", "bpn_n", "pct", "ref", "rt", "dt"]].sort_values(
        ["anio", "dpto_res"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# C. Medidas resumen por periodo (programa resumen_desig del do-file)
# --------------------------------------------------------------------------- #
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
                ["departamento"].iloc[-1])      # replicates bysort quinq (es_ref)
        rt_el = pmax / ref
        dt_el = pmax - ref
    return {"periodo": periodo, "tipo": tipo, "mu": mu, "ref": ref,
            "dpto_ref": nref, "par": mu - ref, "paf": 100 * (mu - ref) / mu,
            "exceso": (mu - ref) / 100 * W, "theil": theil, "mld": mld,
            "pmax": pmax, "dpto_max": nmax, "pmin": pmin, "dpto_min": nmin,
            "rt_ext": pmax / pmin, "dt_ext": pmax - pmin,
            "rt_ext_el": rt_el, "dt_ext_el": dt_el}


def medidas_resumen(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    panel = _prevalencias(panel)
    for q, etq in QUINQ_ETQ.items():
        d_quinq = (panel[panel["quinq"] == q]
                   .groupby(["dpto_res", "departamento"], as_index=False)
                   [["nacidos", "con_peso", "bpn_n"]].sum())
        d_quinq = _prevalencias(d_quinq)
        d_quinq["elegible"] = ((d_quinq["con_peso"] >= MIN_QUINQ)
                               & (d_quinq["pct_falt"] <= MAX_FALT))
        filas.append(resumen_desig(d_quinq, etq, "Quinquenio"))
    for anio in range(cb.ANIO_INI, cb.ANIO_FIN + 1):
        d_anio = _prevalencias(panel[panel["anio"] == anio])
        d_anio["elegible"] = ((d_anio["con_peso"] >= MIN_ANUAL)
                              & (d_anio["pct_falt"] <= MAX_FALT))
        filas.append(resumen_desig(d_anio, str(anio), "Año"))
    return pd.DataFrame(filas)[
        ["periodo", "tipo", "mu", "ref", "dpto_ref", "par", "paf", "exceso",
         "theil", "mld", "pmax", "dpto_max", "pmin", "dpto_min", "rt_ext",
         "dt_ext", "rt_ext_el", "dt_ext_el"]]


# --------------------------------------------------------------------------- #
# G7. Dot plot de RP por departamento, primer y último quinquenio
# --------------------------------------------------------------------------- #
def grafico_g7(ancho: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    d = (ancho[["departamento", "rt1", "rt4"]]
         .sort_values(["rt4", "rt1"], ascending=[False, False])
         .reset_index(drop=True))
    y = range(len(d))
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.hlines(y, d["rt4"], d["rt1"], color="#DDDDDD", lw=1, zorder=1)
    ax.scatter(d["rt1"], y, facecolor="none", edgecolor="#666666", s=28,
               label="2005-2009", zorder=2)                    # O hueco
    ax.scatter(d["rt4"], y, marker="D", color="#08306B", s=24,
               label="2020-2024", zorder=3)
    ax.axvline(1, color="#999999", ls="--", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["departamento"], fontsize=6.5)
    ax.set_xlabel("Razón de prevalencias frente a la referencia")
    ax.set_title("Razón de prevalencias de BPN por departamento", fontsize=12)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)
    ax.annotate("Referencia: menor prevalencia entre departamentos con >=10.000\n"
                "nacidos con peso y <=5 % sin información de peso en el quinquenio.\n"
                "Fuente: DANE, EEVV.",
                xy=(0, -0.12), xycoords="axes fraction", ha="left", fontsize=7)
    fig.tight_layout()
    cb.guardar_grafico(fig, "G7_RP_departamentos_quinquenios.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# G8. Theil y PAF anuales (eje dual)
# --------------------------------------------------------------------------- #
def grafico_g8(resumen: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    d = (resumen[resumen["tipo"].eq("Año")]
         .assign(anio=lambda t: t["periodo"].astype(int))
         [["anio", "theil", "paf"]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    ax.plot(d["anio"], d["theil"], color="#08306B", marker="o", ms=4,
            label="Theil (eje izq.)")
    ax2.plot(d["anio"], d["paf"], color="#A63603", ls="--", marker="D", ms=4,
             label="PAF (eje der.)")
    ax.set_xticks(range(2005, 2025, 2))
    ax.set_ylabel("Índice de Theil (x1000)")
    ax2.set_ylabel("PAF (%)")
    ax.set_title("Desigualdad departamental del BPN, 2005-2024", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.annotate("Theil ponderado por nacidos con peso conocido. PAF frente a la "
                "referencia anual elegible.\nFuente: DANE, Estadísticas Vitales. "
                "Cálculos propios.", xy=(1, -0.14), xycoords="axes fraction",
                ha="right", fontsize=7, va="top")
    ax2.grid(False)
    fig.tight_layout()
    cb.guardar_grafico(fig, "G8_Theil_PAF_anual.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    print("\nOBJETIVO 2 - Desigualdades RT/DT (quinquenios y años)")
    celdas = cb.cargar_celdas()
    cb.iniciar_estilo()

    print("\n--- A. Panel departamento-año ---")
    panel = panel_dpto_anio(celdas)
    print(f"filas: {len(panel):,}  dptos: {panel.dpto_res.nunique()}")

    print("\n--- B. RP/DP por quinquenio ---")
    quinq = rt_dt_quinquenio(panel)
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
    ancho = rt_dt_ancho(quinq)

    print("\n--- C. Medidas resumen ---")
    resumen = medidas_resumen(panel)
    print(resumen[resumen["tipo"].eq("Quinquenio")]
          [["periodo", "ref", "dpto_ref", "par", "paf", "theil"]]
          .round(2).to_string(index=False))

    print("\n--- RP/DP anual ---")
    anual = rt_dt_anual(panel)
    muestra = anual[anual["anio"].isin([2005, 2024])]
    print(muestra[["anio", "departamento", "pct", "ref", "rt"]]
          .round(2).to_string(index=False))

    grafico_g7(ancho)
    grafico_g8(resumen)

    cb.exportar_excel_nuevo({
        "RP_DP_quinquenio": quinq, "RP_DP_ancho": ancho, "RP_DP_anual": anual,
        "Medidas_resumen": resumen,
    }, EXCEL)
    print(f"\nListo. Resultados en {cb.SALIDAS / EXCEL}")


if __name__ == "__main__":
    with cb.log_salida("log_02_objetivo2"):
        main()
