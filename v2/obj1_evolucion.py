# -*- coding: utf-8 -*-
# =============================================================================
#  TRABAJO FINAL - BAJO PESO AL NACER EN COLOMBIA 2005-2024
#  Python: obj1_evolucion.py   (equivale a 01_Base_y_Objetivo1.do, v2)
#
#  Parte A/B (base y variables armonizadas) la hace comun_bpn.cargar_celdas.
#  Aquí va la Parte C del do-file: Objetivo 1 - evolución y distribución del
#  BPN: prevalencia nacional (cruda y estandarizada por edad), APC (Poisson
#  robusto), prevalencia por departamento, calidad del registro del peso,
#  por sexo y distribución del peso, con IC 95 % de Wilson. Salidas:
#  Objetivo1_Evolucion_BPN.xlsx, G1_bpn_nacional.png y el log.
# =============================================================================
import numpy as np
import pandas as pd

import comun_bpn as cb

EXCEL = "Objetivo1_Evolucion_BPN.xlsx"
# Etiquetas del peso_reparto de dane-data (loader._PESOS) -> grupos del do-file
_MAP_PESO = {"Menos de 1.000": 1, "1.000 a 1.499": 2, "1.500 a 1.999": 3,
             "2.000 a 2.499": 4, "2.500 a 2.999": 5, "3.000 a 3.499": 6,
             "3.500 a 3.999": 7, "4.000 y más": 8}


# --------------------------------------------------------------------------- #
# C1. Nacional por año: prevalencia cruda con IC 95 % y BPN a término
# --------------------------------------------------------------------------- #
def nacional_anio(celdas: pd.DataFrame) -> pd.DataFrame:
    d = (celdas.groupby("anio")
         .agg(nacidos_vivos=("n", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum"), n_term=("n_term", "sum"),
              bpn_term_n=("bpn_term_n", "sum"))
         .reset_index())
    d["sin_info_peso"] = d["nacidos_vivos"] - d["con_peso"]
    d["pct_sin_peso"] = 100 * d["sin_info_peso"] / d["nacidos_vivos"]
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    d["li_bpn"], d["ls_bpn"] = cb.wilson(d["bpn_n"], d["con_peso"])
    d["pct_bpn_term"] = 100 * d["bpn_term_n"] / d["n_term"]
    return d[["anio", "nacidos_vivos", "con_peso", "sin_info_peso",
              "pct_sin_peso", "bpn_n", "pct_bpn", "li_bpn", "ls_bpn",
              "n_term", "bpn_term_n", "pct_bpn_term"]]


# --------------------------------------------------------------------------- #
# C2. Prevalencia estandarizada por edad de la madre (método directo;
#     población estándar: todos los nacidos vivos 2005-2024 con peso y edad)
# --------------------------------------------------------------------------- #
def nacional_estandarizada(celdas: pd.DataFrame) -> pd.DataFrame:
    base = celdas[celdas["edad6"].notna() & celdas["con_peso"].gt(0)]
    pesos = base.groupby("edad6", observed=True)["con_peso"].sum()
    pesos /= pesos.sum()
    d = cb.estandarizar_edad(base, ["anio"], pesos, n="con_peso", casos="bpn_n")
    d["pct_cruda"] = 100 * d["casos"] / d["n"]
    d["pct_estand"] = 100 * d["comp"]
    d["li_estand"] = 100 * (d["comp"] - 1.96 * np.sqrt(d["v"]))
    d["ls_estand"] = 100 * (d["comp"] + 1.96 * np.sqrt(d["v"]))
    return d[["anio", "n", "casos", "pct_cruda", "pct_estand",
              "li_estand", "ls_estand"]]


# --------------------------------------------------------------------------- #
# C3. Cambio porcentual anual (APC), nacional y por departamento
# --------------------------------------------------------------------------- #
def apc(celdas: pd.DataFrame) -> pd.DataFrame:
    por_anio = (celdas.groupby("anio")
                .agg(con_peso=("con_peso", "sum"), bpn_n=("bpn_n", "sum"))
                .reset_index())
    filas = []
    for periodo, ini, fin in [("2005-2024", 2005, 2024), ("2005-2021", 2005, 2021)]:
        r = cb.apc_poisson(por_anio, ini, fin)
        filas.append({"dpto_res": 0, "departamento": "Colombia",
                      "periodo": periodo, **r})
    dptos = (celdas[celdas["dpto_res"].gt(0)]
             .groupby(["dpto_res", "anio"])
             .agg(con_peso=("con_peso", "sum"), bpn_n=("bpn_n", "sum"))
             .reset_index())
    for dpto in sorted(dptos["dpto_res"].unique()):
        d_dpto = dptos[dptos["dpto_res"] == dpto]
        for periodo, ini, fin in [("2005-2024", 2005, 2024),
                                  ("2005-2021", 2005, 2021)]:
            r = cb.apc_poisson(d_dpto, ini, fin)
            filas.append({"dpto_res": dpto, "departamento": cb.DPTO_NOM[dpto],
                          "periodo": periodo, **r})
    return pd.DataFrame(filas)[
        ["dpto_res", "departamento", "periodo", "apc", "li", "ls", "p", "n_anios"]]


# --------------------------------------------------------------------------- #
# C4. Por departamento de residencia y año, con IC 95 % (Wilson)
# --------------------------------------------------------------------------- #
def departamento_largo(celdas: pd.DataFrame) -> pd.DataFrame:
    d = (celdas[celdas["dpto_res"].gt(0)]
         .groupby(["dpto_res", "anio"])
         .agg(nacidos_vivos=("n", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum"), n_term=("n_term", "sum"),
              bpn_term_n=("bpn_term_n", "sum"))
         .reset_index())
    d["pct_sin_peso"] = 100 * (d["nacidos_vivos"] - d["con_peso"]) / d["nacidos_vivos"]
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    d["li_bpn"], d["ls_bpn"] = cb.wilson(d["bpn_n"], d["con_peso"])
    d["pct_bpn_term"] = 100 * d["bpn_term_n"] / d["n_term"]
    d.insert(1, "departamento", d["dpto_res"].map(cb.DPTO_NOM))
    return d[["dpto_res", "departamento", "anio", "nacidos_vivos", "con_peso",
              "pct_sin_peso", "bpn_n", "pct_bpn", "li_bpn", "ls_bpn",
              "n_term", "bpn_term_n", "pct_bpn_term"]]


def departamento_ancho(largo: pd.DataFrame) -> pd.DataFrame:
    d = largo.pivot(index=["dpto_res", "departamento"], columns="anio",
                    values="pct_bpn")
    d.columns = [str(c) for c in d.columns]
    return d.reset_index()


# --------------------------------------------------------------------------- #
# C5. Calidad del registro del peso por departamento y quinquenio
# --------------------------------------------------------------------------- #
def calidad_peso(celdas: pd.DataFrame) -> pd.DataFrame:
    d = celdas[celdas["dpto_res"].gt(0)].copy()
    d["quinq"] = 1 + (d["anio"] - 2005) // 5
    d = (d.groupby(["dpto_res", "quinq"])
         .agg(nacidos_vivos=("n", "sum"), con_peso=("con_peso", "sum")).reset_index())
    d["pct_sin_peso"] = 100 * (d["nacidos_vivos"] - d["con_peso"]) / d["nacidos_vivos"]
    d = d.pivot(index="dpto_res", columns="quinq",
                values="pct_sin_peso").add_prefix("pct_sin_peso").reset_index()
    d.insert(1, "departamento", d["dpto_res"].map(cb.DPTO_NOM))
    return d


# --------------------------------------------------------------------------- #
# C6. Por sexo del nacido vivo y año (se excluye el sexo indeterminado)
# --------------------------------------------------------------------------- #
def por_sexo() -> pd.DataFrame:
    anios = list(range(cb.ANIO_INI, cb.ANIO_FIN + 1))
    tot = cb.conteos(["sexo"], anios)
    tot = tot[tot["sexo"].isin(["Hombre", "Mujer"])]
    bp = cb.conteos(["sexo", "bajo_peso"], anios)
    bp = bp[bp["sexo"].isin(["Hombre", "Mujer"])]
    bpn = (bp[bp["bajo_peso"].eq(True)].groupby(["anio", "sexo"])["n"].sum()
           .rename("bpn_n"))
    con = bp.groupby(["anio", "sexo"])["n"].sum().rename("con_peso")
    nac = tot.groupby(["anio", "sexo"])["n"].sum().rename("nacidos_vivos")
    d = (pd.concat([nac, con, bpn], axis=1).reset_index()
         .rename(columns={"sexo": "sx"}))
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    d = d.pivot(index="anio", columns="sx", values=["nacidos_vivos", "bpn_n", "pct_bpn"])
    d.columns = [f"{est}{sx}" for est, sx in d.columns]
    orden = ["nacidos_vivosHombre", "bpn_nHombre", "pct_bpnHombre",
             "nacidos_vivosMujer", "bpn_nMujer", "pct_bpnMujer"]
    return d.reset_index()[["anio", *orden]]


# --------------------------------------------------------------------------- #
# C7. Distribución del peso al nacer por año (%)
# --------------------------------------------------------------------------- #
def distribucion_peso() -> pd.DataFrame:
    anios = list(range(cb.ANIO_INI, cb.ANIO_FIN + 1))
    d = cb.conteos(["peso_grupo"], anios)
    d["g"] = d["peso_grupo"].map(_MAP_PESO)
    d = d[d["g"].notna()]                     # sin peso no conocido (igual do-file)
    total = d.groupby("anio")["n"].sum()
    d = (d.assign(n=d["n"] * 100 / d["anio"].map(total))
         .pivot(index="anio", columns="g", values="n")
         .rename(columns=cb.PESO_GRUPO_ETQ).reset_index())
    d.columns.name = None
    return d


# --------------------------------------------------------------------------- #
# G1. Cruda con IC 95 % y estandarizada por edad
# --------------------------------------------------------------------------- #
def grafico_g1(nac: pd.DataFrame, estand: pd.DataFrame):
    import matplotlib.pyplot as plt
    d = nac[["anio", "pct_bpn", "li_bpn", "ls_bpn"]].merge(
        estand[["anio", "pct_estand"]], on="anio")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(d.anio, d.ls_bpn, d.li_bpn, color="#C6DBEF",
                    lw=0, label="IC 95 %")
    ax.plot(d.anio, d.pct_bpn, color="#08306B", marker="o", ms=4,
            label="Prevalencia cruda")
    ax.plot(d.anio, d.pct_estand, color="#A63603", ls="--", lw=1.8,
            label="Estandarizada por edad materna")
    ax.set_xticks(range(2005, 2025, 2))
    ax.set_ylabel("% de nacidos vivos con peso < 2.500 g")
    ax.set_title("Bajo peso al nacer en Colombia, 2005-2024", fontsize=12)
    ax.legend(loc="upper left", ncols=1, fontsize=9)
    ax.annotate("Desde 2022 la base incluye registros de la\n"
                "Registraduría sin información de peso.\n"
                "Fuente: DANE, Estadísticas Vitales. Cálculos propios.",
                xy=(1, -0.14), xycoords="axes fraction", ha="right",
                fontsize=7, va="top")
    fig.tight_layout()
    return fig


FUENTE = "Fuente: DANE, Estadísticas Vitales. Cálculos propios."


# --------------------------------------------------------------------------- #
# G12. Mapa de calor: % BPN por departamento y año
# --------------------------------------------------------------------------- #
def grafico_mapa_calor(ancho: pd.DataFrame):
    import matplotlib.pyplot as plt
    anios = [c for c in ancho.columns if c.isdigit()]
    d = (ancho.assign(_orden=ancho[[a for a in anios if int(a) >= 2020]].mean(axis=1))
         .sort_values("_orden", ascending=False))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(d[anios].to_numpy(float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(anios)), anios, rotation=90, fontsize=8)
    ax.set_yticks(range(len(d)), d["departamento"], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% de nacidos vivos con peso < 2.500 g")
    ax.set_title("Bajo peso al nacer por departamento de residencia, 2005-2024",
                 fontsize=12)
    ax.annotate("Departamentos ordenados por el promedio del % BPN 2020-2024.\n"
                + FUENTE, xy=(0, -0.1), xycoords="axes fraction", fontsize=7,
                va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# G13. % BPN por sexo del nacido vivo
# --------------------------------------------------------------------------- #
def grafico_sexo(sexo: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sexo["anio"], sexo["pct_bpnMujer"], color="#A63603", marker="o", ms=4,
            label="Mujeres")
    ax.plot(sexo["anio"], sexo["pct_bpnHombre"], color="#08306B", marker="o", ms=4,
            label="Hombres")
    ax.set_xticks(range(2005, 2025, 2))
    ax.set_ylabel("% de nacidos vivos con peso < 2.500 g")
    ax.set_title("Bajo peso al nacer según sexo del nacido vivo, 2005-2024",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.annotate("Se excluye el sexo indeterminado.\n" + FUENTE, xy=(1, -0.14),
                xycoords="axes fraction", ha="right", fontsize=7, va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# G14. Distribución del peso al nacer por año (áreas apiladas)
# --------------------------------------------------------------------------- #
def grafico_distribucion(dist: pd.DataFrame):
    import matplotlib.pyplot as plt
    grupos = [g for g in cb.PESO_GRUPO_ETQ.values() if g in dist.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(dist["anio"], *[dist[g] for g in grupos], labels=grupos,
                 colors=plt.get_cmap("RdYlBu")(np.linspace(0.05, 0.95, len(grupos))),
                 lw=0)
    ax.set(xlim=(dist["anio"].min(), dist["anio"].max()), ylim=(0, 100),
           ylabel="% de nacidos vivos con peso conocido")
    ax.set_xticks(range(2005, 2025, 2))
    ax.set_title("Distribución del peso al nacer por año, 2005-2024", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Peso al nacer (g)",
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    ax.annotate(FUENTE, xy=(1, -0.1), xycoords="axes fraction", ha="right",
                fontsize=7, va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)
# --------------------------------------------------------------------------- #
LOG = "log_01_base_objetivo1"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 1; cada clave es una hoja del xlsx."""
    largo = departamento_largo(celdas)
    return {
        "Nacional": nacional_anio(celdas),
        "Nacional_estandarizada": nacional_estandarizada(celdas),
        "APC": apc(celdas),
        "Departamento_largo": largo,
        "Departamento_%BPN": departamento_ancho(largo),
        "Calidad_registro_peso": calidad_peso(celdas),
        "Sexo": por_sexo(),
        "Distribucion_peso": distribucion_peso(),
        "Control_carga": pd.read_csv(cb.PROCESADOS / "control_carga.csv"),
    }


def imprimir(r: dict) -> None:
    print("\nPARTE C. OBJETIVO 1 - Evolución y distribución del BPN")
    print("\n--- C1. Nacional por año ---")
    print(r["Nacional"][["anio", "con_peso", "pct_bpn", "li_bpn", "ls_bpn",
                         "pct_bpn_term"]].round(2).to_string(index=False))
    print("\n--- C2. Nacional estandarizada por edad de la madre ---")
    print(r["Nacional_estandarizada"][["anio", "pct_cruda", "pct_estand",
                                       "li_estand", "ls_estand"]]
          .round(2).to_string(index=False))
    print("\n--- C3. APC (Poisson robusto) ---")
    apc_ = r["APC"]
    print(apc_[apc_["dpto_res"].eq(0)].round(2).to_string(index=False))
    print("\n--- C5. Calidad del registro del peso ---")
    print(r["Calidad_registro_peso"].sort_values("pct_sin_peso4", ascending=False)
          [["departamento", "pct_sin_peso1", "pct_sin_peso4"]]
          .head(8).round(2).to_string(index=False))
    print("\n--- C6. Por sexo ---")
    print(r["Sexo"][["anio", "pct_bpnHombre", "pct_bpnMujer"]]
          .round(2).to_string(index=False))
    print("\n--- C7. Distribución del peso ---")
    print(r["Distribucion_peso"].round(2).to_string(index=False))


def graficos(r: dict) -> dict:
    cb.iniciar_estilo()
    return {"G1_bpn_nacional": grafico_g1(r["Nacional"], r["Nacional_estandarizada"]),
            "G12_mapa_calor_departamentos": grafico_mapa_calor(r["Departamento_%BPN"]),
            "G13_bpn_por_sexo": grafico_sexo(r["Sexo"]),
            "G14_distribucion_peso": grafico_distribucion(r["Distribucion_peso"])}
