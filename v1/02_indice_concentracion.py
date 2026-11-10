"""
Desigualdad socioeconómica del bajo peso al nacer (BPN) entre departamentos, 2008-2024.
Traducción de Code/stata/02_Indice_Concentracion.do.

  - Indicador socioeconómico: % de madres en régimen SUBSIDIADO del departamento y año
    (más subsidiado = más pobre). El régimen es comparable desde 2008.
  - Ponderación: nacidos vivos con peso conocido del departamento.

  A. Panel departamento-año con % BPN y % subsidiado
  B. Índice de concentración (IC) por año, con error estándar e IC 95 %, y el de Erreygers
  C. Trienios 2008-2010 vs 2022-2024 y curvas de concentración
  Sensibilidad: (i) pobre = subsidiado + no asegurado; (ii) sin Bogotá (BPN alto por la altitud).

No necesita correr antes 01_resultados_generales.py: dane-data lee los datos directamente.
Resultados en "3.0 Datos/python".
"""
from pathlib import Path

import dane_data as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dane_data.referencias import DEPARTAMENTOS
from scipy.stats import norm

ANIO_INI, ANIO_FIN = 2008, 2024
TRIENIOS = {"2008-2010": range(2008, 2011), "2022-2024": range(2022, 2025)}
BOGOTA = 11
RESULTADOS = Path(__file__).resolve().parents[2] / "3.0 Datos" / "python"
RESULTADOS.mkdir(parents=True, exist_ok=True)
CODIGOS = {nombre: cod for cod, nombre in DEPARTAMENTOS.items()}
AZUL, NARANJA, ROJO = "#08306B", "#A63603", "#C00000"
FUENTE = "Fuente: DANE, Estadísticas Vitales. Cálculos propios."
ORDENAMIENTO = ("Ordenamiento: % de madres en régimen subsidiado (más subsidiado = más pobre). "
                "Ponderación: nacidos vivos.")

ETQ = {
    "anio": "Año", "dpto_res": "Código departamento", "departamento": "Departamento", "nacidos": "Nacidos vivos",
    "con_peso": "Nacidos vivos con peso conocido (ponderación)", "bpn_n": "Nacidos con bajo peso (<2.500 g)",
    "pct_bpn": "% bajo peso al nacer", "reg_conocido": "Nacidos con régimen conocido",
    "n_sub": "Madres en régimen subsidiado", "pct_sub": "% madres en régimen subsidiado",
    "pct_sub_na": "% madres subsidiado o no aseguradas", "ndptos": "Departamentos", "mu": "% BPN (media ponderada)",
    "C": "Índice de concentración", "seC": "Error estándar IC", "li_C": "IC 95 % inferior",
    "ls_C": "IC 95 % superior", "p_C": "Valor p (IC = 0)", "E": "Índice de Erreygers",
    "seE": "Error estándar Erreygers", "C_sens": "IC sensibilidad (subsidiado + no asegurado)",
    "C_sinbog": "IC sensibilidad (sin Bogotá)", "periodo": "Periodo", "orden": "Orden (1 = más subsidiado)",
}


def ic_conc(d, h, w, rango):
    """
    Índice de concentración con datos agrupados (una fila = un departamento).
      h = prevalencia de BPN (%), w = ponderación, rango = indicador de MENOR a MAYOR condición.
    Regresión conveniente (O'Donnell et al., 2008, cap. 8): 2·var(r)·(h/mu) = a + b·r -> b = IC,
    con error estándar robusto (HC1 = vce(robust) de Stata). r = rango fraccional ponderado.
    Erreygers (2009) para una variable acotada entre 0 y 100: E = 4·mu·IC / 100.
    """
    d = d.sort_values(rango, kind="stable")
    W = d[w].sum()
    r = (d[w].cumsum() - d[w] / 2) / W
    mu = np.average(d[h], weights=d[w])
    var_r = (d[w] * (r - 0.5) ** 2).sum() / W
    lhs = 2 * var_r * d[h] / mu
    m = sm.WLS(lhs, sm.add_constant(r), weights=d[w]).fit(cov_type="HC1")
    C, se = m.params.iloc[1], m.bse.iloc[1]
    # La pendiente debe ser la fórmula de covarianza del IC: 2·cov_w(h, r) / mu
    assert np.isclose(C, 2 * np.average((d[h] - mu) * (r - 0.5), weights=d[w]) / mu)
    return {"ndptos": int(m.nobs), "mu": mu, "C": C, "seC": se, "E": 4 * mu / 100 * C, "seE": 4 * mu / 100 * se}


def indices(d):
    """IC principal, sensibilidad con subsidiado + no asegurado, y sin Bogotá."""
    d = d[(d.con_peso > 0) & d.pct_sub.notna()]
    res = ic_conc(d, "pct_bpn", "con_peso", "ses")
    res["C_sens"] = ic_conc(d, "pct_bpn", "con_peso", "ses_na")["C"]
    res["C_sinbog"] = ic_conc(d[d.dpto_res != BOGOTA], "pct_bpn", "con_peso", "ses")["C"]
    return res


SUMAS = ["nacidos", "con_peso", "bpn_n", "reg_conocido", "n_sub", "n_sub_na"]


def agregar(d, por):
    """El collapse (sum) del .do y los porcentajes de cada grupo."""
    r = d.groupby(por, observed=True)[SUMAS].sum().reset_index()
    r["pct_bpn"] = 100 * r.bpn_n / r.con_peso
    r["pct_sub"] = 100 * r.n_sub / r.reg_conocido
    r["pct_sub_na"] = 100 * r.n_sub_na / r.reg_conocido
    r["ses"] = 100 - r.pct_sub        # principal: % NO subsidiado (mayor = mejor condición)
    r["ses_na"] = 100 - r.pct_sub_na  # sensibilidad: % ni subsidiado ni no asegurado
    return r


def coma(x):
    return f"{x:.3f}".replace(".", ",")


# ==============================================================================
# A. PANEL DEPARTAMENTO-AÑO
# ==============================================================================
c = dd.eevv.serie_conteos(range(ANIO_INI, ANIO_FIN + 1), tipo="nacimientos",
                          por=["departamento_residencia", "regimen_salud", "bajo_peso"], dropna=False)
# Fuera: madres residentes en el exterior o sin departamento
c = c[c.departamento_residencia.notna()]

# Régimen (códigos DANE desde 2008): Contributivo, Excepción y Especial juntos; Subsidiado; No asegurado.
# «Sin información» y los códigos fuera del diccionario no cuentan como régimen conocido.
reg = c.regimen_salud.astype(object)
c = c.assign(
    departamento=c.departamento_residencia.astype(object),
    nacidos=c.n,
    con_peso=c.n * c.bajo_peso.notna(),
    bpn_n=c.n * c.bajo_peso.fillna(False).astype(bool),
    reg_conocido=c.n * reg.isin(["Contributivo", "Excepción", "Especial", "Subsidiado", "No asegurado"]),
    n_sub=c.n * reg.eq("Subsidiado"),
    n_sub_na=c.n * reg.isin(["Subsidiado", "No asegurado"]),
)
print(pd.crosstab(c.anio, reg, values=c.n, aggfunc="sum", normalize="index", dropna=False).mul(100).round(1))

panel = agregar(c, ["anio", "departamento"])
panel.insert(1, "dpto_res", panel.departamento.map(CODIGOS))
panel = panel.sort_values(["anio", "dpto_res"], ignore_index=True)

# ==============================================================================
# B. ÍNDICE DE CONCENTRACIÓN POR AÑO
# ==============================================================================
ic = pd.DataFrame([{"anio": a, **indices(g)} for a, g in panel.groupby("anio")])
ic["li_C"] = ic.C - 1.96 * ic.seC
ic["ls_C"] = ic.C + 1.96 * ic.seC
ic["p_C"] = 2 * norm.sf(abs(ic.C / ic.seC))
ic = ic[["anio", "ndptos", "mu", "C", "seC", "li_C", "ls_C", "p_C", "E", "seE", "C_sens", "C_sinbog"]]
print(ic.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(ic.anio, ic.li_C, ic.ls_C, color="#C6DBEF", lw=0, label="Intervalo de confianza 95 %")
ax.plot(ic.anio, ic.C, "o-", color=AZUL, lw=2, ms=4, label="Índice de concentración")
ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_yticks(np.arange(0, 0.21, 0.05))
ax.yaxis.set_major_formatter("{x:.2f}")
ax.set_xticks(range(ANIO_INI, ANIO_FIN + 1, 2))
ax.grid(axis="y", color="#EEEEEE")
ax.set_ylabel("Índice de concentración")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
fig.suptitle(f"Índice de concentración del bajo peso al nacer, {ANIO_INI}-{ANIO_FIN}")
ax.set_title("Departamentos ordenados según % de madres en régimen subsidiado", fontsize=9, color="dimgrey")
fig.text(0.01, -0.04, "Valores positivos: el BPN se concentra en los departamentos menos pobres;\n"
         f"valores negativos: en los más pobres. {FUENTE}", fontsize=7, va="top")
fig.savefig(RESULTADOS / "G2_indice_concentracion_anual.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# C. TRIENIOS 2008-2010 vs 2022-2024
# ==============================================================================
tri = pd.concat([panel[panel.anio.isin(anios)].assign(periodo=p) for p, anios in TRIENIOS.items()])
tri = agregar(tri, ["periodo", "dpto_res", "departamento"])

ict = pd.DataFrame([{"periodo": p, **indices(g)} for p, g in tri.groupby("periodo")])
ict["li_C"] = ict.C - 1.96 * ict.seC
ict["ls_C"] = ict.C + 1.96 * ict.seC
# Prueba de diferencia entre trienios (muestras independientes)
dif = ict.C.iloc[1] - ict.C.iloc[0]
sedif = np.hypot(ict.seC.iloc[0], ict.seC.iloc[1])
print(f"\nCambio en el IC (2022-2024 menos 2008-2010): {dif:.4f}   EE: {sedif:.4f}   "
      f"p = {2 * norm.sf(abs(dif / sedif)):.4f}")
ict = pd.concat([ict, pd.DataFrame([{"periodo": "Diferencia (2022-2024 menos 2008-2010)", "C": dif, "seC": sedif}])],
                ignore_index=True)
ict["p_C"] = 2 * norm.sf(abs(ict.C / ict.seC))
ict = ict[["periodo", "ndptos", "mu", "C", "seC", "li_C", "ls_C", "p_C", "E", "seE", "C_sens", "C_sinbog"]]
print(ict.round(4).to_string(index=False))

# Ordenamiento de departamentos por trienio (para el anexo)
orden = tri.sort_values(["periodo", "pct_sub"], ascending=[True, False])
orden["orden"] = orden.groupby("periodo").cumcount() + 1
orden = orden[["periodo", "orden", "departamento", "pct_sub", "pct_bpn", "nacidos", "con_peso"]]

with pd.ExcelWriter(RESULTADOS / "Indice_concentracion_BPN.xlsx") as xl:
    panel[["anio", "dpto_res", "departamento", "nacidos", "con_peso", "bpn_n", "pct_bpn", "reg_conocido", "n_sub",
           "pct_sub", "pct_sub_na"]].rename(columns=ETQ).to_excel(xl, sheet_name="Dpto_anio", index=False)
    ic.rename(columns=ETQ).to_excel(xl, sheet_name="IC_por_anio", index=False)
    ict.rename(columns={**ETQ, "seC": "Error estándar", "p_C": "Valor p"}).to_excel(xl, sheet_name="IC_trienios", index=False)
    orden.rename(columns={**ETQ, "con_peso": "Nacidos con peso conocido"}).to_excel(
        xl, sheet_name="Orden_dptos_trienios", index=False)

# --- Curvas de concentración ---
# Departamentos del más pobre (más subsidiado) al menos pobre. Eje X: % acumulado de nacidos vivos;
# eje Y: % acumulado de casos de BPN. Bajo la diagonal: el BPN se concentra en los menos pobres;
# encima: en los más pobres. El área entre la curva y la diagonal es el área de concentración (IC = 2 x área).
IC = dict(zip(ict.periodo, ict.C))
IC_SINBOG = dict(zip(ict.periodo, ict.C_sinbog))


def puntos(periodo, sin_bogota=False):
    d = tri[(tri.periodo == periodo) & (tri.con_peso > 0) & tri.pct_sub.notna()]
    if sin_bogota:
        d = d[d.dpto_res != BOGOTA]
    d = d.sort_values("ses", kind="stable")
    x = np.r_[0, 100 * d.con_peso.cumsum() / d.con_peso.sum()]
    y = np.r_[0, 100 * d.bpn_n.cumsum() / d.bpn_n.sum()]
    return x, y, np.r_[0, d.dpto_res]


def ejes(ax, titulo, xlabel="% acumulado de nacidos vivos\n(departamentos del más pobre al menos pobre)"):
    ax.plot([0, 100], [0, 100], color="black", lw=1.5, label="Línea de igualdad")
    ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal", xticks=range(0, 101, 10), yticks=range(0, 101, 10),
           xlabel=xlabel, ylabel="% acumulado de casos de bajo peso al nacer")
    ax.set_title(titulo)
    ax.grid(color="#EEEEEE")


def curva(ax, x, y, color, marcador="o", ms=3, **linea):
    ax.plot(x, y, color=color, lw=2.5, **linea)
    if marcador:
        ax.plot(x[1:], y[1:], marcador, color=color, ms=ms, ls="none")


def g3(ax, p):  # con Bogotá, área de concentración sombreada y Bogotá resaltada
    ejes(ax, p)
    x, y, cod = puntos(p)
    ax.fill_between(x, x, y, color="#FFE699", lw=0)
    curva(ax, x, y, AZUL)
    i = np.flatnonzero(cod == BOGOTA)
    ax.plot(x[i], y[i], "o", color=ROJO, ms=6, clip_on=False, zorder=5)
    for xi, yi in zip(x[i], y[i]):
        ax.annotate("Bogotá", (xi, yi), xytext=(4, -8), textcoords="offset points", color=ROJO, fontsize=8)
    ax.text(6, 88, f"IC = {coma(IC[p])}")


def g5(ax, p):  # sin Bogotá, mismo formato que G3
    ejes(ax, p)
    x, y, _ = puntos(p, sin_bogota=True)
    ax.fill_between(x, x, y, color="#FDD0A2", lw=0)
    curva(ax, x, y, NARANJA)
    ax.text(6, 88, f"IC = {coma(IC_SINBOG[p])}")


def g6(ax, p):  # con vs. sin Bogotá
    ejes(ax, p)
    curva(ax, *puntos(p)[:2], AZUL, ms=2, label=f"Con Bogotá (IC = {coma(IC[p])})")
    curva(ax, *puntos(p, sin_bogota=True)[:2], NARANJA, marcador="D", ms=2, ls="--",
          label=f"Sin Bogotá (IC = {coma(IC_SINBOG[p])})")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[1:] + h[:1], l[1:] + ["Igualdad"], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              frameon=False, fontsize=7)


def figura(dibujar, titulo, nota, archivo, individuales=()):
    """Los dos trienios lado a lado (graph combine) y, si se piden, cada uno en su archivo."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.4), gridspec_kw={"wspace": 0.4})
    for ax, p in zip(axs, TRIENIOS):
        dibujar(ax, p)
    fig.suptitle(titulo)
    fig.text(0.01, -0.02, nota, fontsize=7, va="top")
    fig.savefig(RESULTADOS / archivo, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for p, nombre in zip(TRIENIOS, individuales):
        fig, ax = plt.subplots(figsize=(6, 6))
        dibujar(ax, p)
        fig.savefig(RESULTADOS / nombre, dpi=300, bbox_inches="tight")
        plt.close(fig)


figura(g3, "Curvas de concentración del bajo peso al nacer en Colombia",
       f"{ORDENAMIENTO}\nCurva bajo la diagonal: el BPN se concentra en los departamentos menos pobres.\n{FUENTE}",
       "G3_curvas_concentracion_trienios.png", ["G3_curva_concentracion_1.png", "G3_curva_concentracion_2.png"])

# G4. Con Bogotá: los dos trienios superpuestos
fig, ax = plt.subplots(figsize=(6, 6.5))
ejes(ax, "Curvas de concentración del bajo peso al nacer",
     xlabel="% acumulado de nacidos vivos (del dpto. más pobre al menos pobre)")
(p1, p2) = TRIENIOS
curva(ax, *puntos(p1)[:2], "#8C8C8C", marcador=None, ls="--", label=f"{p1} (IC = {coma(IC[p1])})")
curva(ax, *puntos(p2)[:2], AZUL, marcador=None, label=f"{p2} (IC = {coma(IC[p2])})")
h, l = ax.get_legend_handles_labels()
ax.legend(h[1:] + h[:1], l[1:] + l[:1], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
          fontsize=8)
fig.text(0.01, -0.02, FUENTE, fontsize=7, va="top")
fig.savefig(RESULTADOS / "G4_curvas_superpuestas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

figura(g5, "Curvas de concentración del bajo peso al nacer, sin Bogotá D.C.",
       f"32 departamentos. {ORDENAMIENTO}\n{FUENTE}", "G5_curvas_sin_Bogota_trienios.png")

figura(g6, "Curvas de concentración del bajo peso al nacer con y sin Bogotá D.C.",
       f"{ORDENAMIENTO}\nEn la curva sin Bogotá, el eje X acumula los nacidos vivos de los otros 32 departamentos.\n"
       f"{FUENTE}", "G6_con_sin_Bogota_trienios.png",
       ["G6_con_sin_Bogota_2008_2010.png", "G6_con_sin_Bogota_2022_2024.png"])

print(f"\nListo. Resultados en: {RESULTADOS}")
