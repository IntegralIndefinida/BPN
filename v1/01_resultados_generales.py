"""
Bajo peso al nacer (BPN) en Colombia 2005-2024: resultados generales por año.
Resultados en "3.0 Datos/python".
"""
from pathlib import Path

import dane_data as dd
import matplotlib.pyplot as plt
import pandas as pd
from dane_data.referencias import DEPARTAMENTOS

ANIO_INI, ANIO_FIN = 2005, 2024
RESULTADOS = Path(__file__).resolve().parents[2] / "3.0 Datos" / "python"
RESULTADOS.mkdir(parents=True, exist_ok=True)
CODIGOS = {nombre: cod for cod, nombre in DEPARTAMENTOS.items()}
SIN_DPTO = "Sin información / exterior"

ETQ = {
    "anio": "Año",
    "dpto_res": "Código departamento",
    "departamento": "Departamento",
    "nacidos_vivos": "Nacidos vivos",
    "con_peso": "Nacidos vivos con peso conocido",
    "sin_info_peso": "Sin información de peso",
    "bpn_n": "Nacidos con bajo peso (<2.500 g)",
    "pct_bpn": "% bajo peso al nacer",
    "archivo": "Archivo leído",
    "registros": "Registros",
}

# Nacidos vivos (n) por año y grupo; dropna=False cuenta también los que no tienen el dato.
# bajo_peso: True si PESO_NAC es 1-4 (<2.500 g), False si 5-8, vacío si 9 (sin información).
c = dd.eevv.serie_conteos(range(ANIO_INI, ANIO_FIN + 1), tipo="nacimientos",
                          por=["departamento_residencia", "sexo", "peso_grupo", "bajo_peso"], dropna=False)
c["con_peso"] = c.n * c.bajo_peso.notna()
c["bpn_n"] = c.n * c.bajo_peso.fillna(False).astype(bool)


def resumen(por):
    """El collapse (sum) del .do: nacidos vivos, con peso conocido y con BPN por grupo."""
    r = (c.groupby(por, observed=True, dropna=False)[["n", "con_peso", "bpn_n"]].sum()
         .reset_index().rename(columns={"n": "nacidos_vivos"}))
    r["pct_bpn"] = 100 * r.bpn_n / r.con_peso
    return r


# B1. Nacional por año
nac = resumen(["anio"])
nac.insert(3, "sin_info_peso", nac.nacidos_vivos - nac.con_peso)
print(nac.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(nac.anio, nac.pct_bpn, "o-", color="navy", ms=4)
ax.set(ylabel="% de nacidos vivos con peso < 2.500 g", xticks=range(ANIO_INI, ANIO_FIN + 1, 2),
       title=f"Bajo peso al nacer en Colombia, {ANIO_INI}-{ANIO_FIN}")
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter("{x:.1f}")
fig.text(0.01, -0.02, "Fuente: DANE, Estadísticas Vitales. Cálculos propios.", fontsize=8, va="top")
fig.savefig(RESULTADOS / "G1_bpn_nacional.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# B2. Por departamento de residencia de la madre y año (0 = sin información o exterior)
dep = resumen(["departamento_residencia", "anio"])
dep["departamento"] = dep.pop("departamento_residencia").astype(object).fillna(SIN_DPTO)
dep.insert(0, "dpto_res", dep.departamento.map(CODIGOS).fillna(0).astype(int))
dep = dep.sort_values(["dpto_res", "anio"])[["dpto_res", "departamento", "anio", "nacidos_vivos", "con_peso",
                                             "bpn_n", "pct_bpn"]]
dep_ancho = dep.pivot(index=["dpto_res", "departamento"], columns="anio", values="pct_bpn").reset_index()

# B3. Por sexo del nacido vivo y año
sx = resumen(["anio", "sexo"])
sx = sx[sx.sexo.isin(["Hombre", "Mujer"])].assign(sexo=lambda d: d.sexo.astype(str))
sx = sx.pivot(index="anio", columns="sexo", values=["nacidos_vivos", "bpn_n", "pct_bpn"])
nombres = {"nacidos_vivos": "Nacidos vivos", "bpn_n": "Bajo peso", "pct_bpn": "% BPN"}
plural = {"Hombre": "Hombres", "Mujer": "Mujeres"}
orden = [(v, s) for s in plural for v in nombres]
sx = sx[orden].set_axis([f"{nombres[v]} - {plural[s]}" for v, s in orden], axis=1).reset_index()
print(sx.round(2).to_string(index=False))

# B4. Distribución del peso al nacer por año (%), sin «Sin información»
pg = c[c.peso_grupo.notna() & c.peso_grupo.ne("Sin información")]
dist = pg.groupby(["anio", "peso_grupo"], observed=True).n.sum()
dist = (100 * dist / dist.groupby(level="anio").transform("sum")).unstack().dropna(axis=1, how="all").reset_index()

# Control: qué archivo del DANE se leyó cada año y cuántos registros tiene
ctl = dd.eevv.available().query("tipo == 'nacimientos' and @ANIO_INI <= anio <= @ANIO_FIN")[["anio", "archivo"]]
ctl = ctl.merge(nac[["anio", "nacidos_vivos"]].rename(columns={"nacidos_vivos": "registros"}), on="anio")

with pd.ExcelWriter(RESULTADOS / "Resultados_generales_por_anio.xlsx") as xl:
    for hoja, df in {"Nacional": nac, "Departamento_largo": dep, "Departamento_%BPN": dep_ancho, "Sexo": sx,
                     "Distribucion_peso": dist, "Control_carga": ctl}.items():
        df.rename(columns=ETQ).to_excel(xl, sheet_name=hoja, index=False)

print(f"\nListo. Resultados en: {RESULTADOS}")
