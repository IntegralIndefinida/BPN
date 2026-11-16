#  Parte A/B (base y variables armonizadas) la hace comun_bpn.cargar_celdas.
#  Aquí va la Objetivo 1: Objetivo 1 - evolución y distribución del
#  BPN: prevalencia nacional (cruda y estandarizada por edad), APC (Poisson
#  robusto, también por subconjunto de registros), prevalencia por
#  departamento, calidad del registro del peso, flujo de registros según la
#  fuente del certificado y origen del faltante de peso,
#  por sexo y distribución del peso, con IC 95 % de Wilson, y descomposición
#  de la tendencia por semanas de gestación (Kitagawa). El bloque C9 mira el
#  corrimiento hacia 28-37 semanas: pretérmino y semana 37 por separado, en
#  quién ocurre, prueba con el peso y contraste con las defunciones fetales.
#  El bloque C10 acota el aporte de las madres nacidas en Venezuela.
#  Salidas: Objetivo1_Evolucion_BPN.xlsx, G1 y G12-G15 png y el log.

import numpy as np
import pandas as pd

import comun_bpn as cb

EXCEL = "Objetivo1_Evolucion_BPN.xlsx"
# Etiquetas del peso_reparto de dane-data (loader._PESOS) -> grupos de análisis
_MAP_PESO = {"Menos de 1.000": 1, "1.000 a 1.499": 2, "1.500 a 1.999": 3,
             "2.000 a 2.499": 4, "2.500 a 2.999": 5, "3.000 a 3.499": 6,
             "3.500 a 3.999": 7, "4.000 y más": 8}


# C1. Nacional por año: prevalencia cruda con IC 95 % y BPN a término

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


# C2. Prevalencia estandarizada por edad de la madre (método directo;
#     población estándar: todos los nacidos vivos 2005-2024 con peso y edad)

def nacional_estandarizada(celdas: pd.DataFrame) -> pd.DataFrame:
    base = celdas[celdas["edad6"].notna() & celdas["con_peso"].gt(0)]
    pesos = cb.pesos_edad(base, "con_peso")
    d = cb.estandarizar_edad(base, ["anio"], pesos, n="con_peso", casos="bpn_n")
    d["pct_cruda"] = 100 * d["casos"] / d["n"]
    d["pct_estand"] = 100 * d["comp"]
    d["li_estand"] = 100 * (d["comp"] - 1.96 * np.sqrt(d["v"]))
    d["ls_estand"] = 100 * (d["comp"] + 1.96 * np.sqrt(d["v"]))
    return d[["anio", "n", "casos", "pct_cruda", "pct_estand",
              "li_estand", "ls_estand"]]


# C3. Cambio porcentual anual (APC), nacional y por departamento

# Subconjuntos nacionales de registros: separan el quiebre de registro de la
# tendencia (desde 2022 el faltante de peso está en los recuperados de la
# Registraduría, que no traen departamento de residencia: dpto_res = 0)
# La Guajira, Cesar, Norte de Santander y Arauca. Quitarlos no es una
# sensibilidad sino otro estimando (el resto del país), y en La Guajira mezcla
# migración con composición étnica (wayuu).
DPTOS_FRONTERA = [44, 20, 54, 81]
SUBCONJUNTOS = {
    "Colombia": None,
    "Con residencia conocida": "dpto_res > 0",
    "Con residencia, sin Chocó, La Guajira y Cesar":
        f"dpto_res > 0 and dpto_res not in {cb.DPTOS_FALTANTE_PESO}",
    "Con residencia, sin departamentos de frontera con Venezuela":
        f"dpto_res > 0 and dpto_res not in {DPTOS_FRONTERA}",
}


def apc(celdas: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre, cond in SUBCONJUNTOS.items():
        por_anio = ((celdas.query(cond) if cond else celdas).groupby("anio")
                    .agg(con_peso=("con_peso", "sum"), bpn_n=("bpn_n", "sum"))
                    .reset_index())
        for periodo, ini, fin in [("2005-2024", 2005, 2024),
                                  ("2005-2021", 2005, 2021)]:
            r = cb.apc_poisson(por_anio, ini, fin)
            filas.append({"dpto_res": 0, "departamento": nombre,
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


# C4. Por departamento de residencia y año, con IC 95 % (Wilson)

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


# C5. Calidad del registro del peso por departamento y quinquenio

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


# C5b. Fuente del certificado (TIPOFORMULARIO, desde 2021) y flujo de registros
# 1 = certificado médico (RUAF-ND); 2 = Formato de Notificación Étnica;
# 3 = recuperado en el cruce con la Registraduría (RNEC): nacimientos captados
# por el registro civil y no por el certificado médico. Los de tipo 3 no traen
# peso, semanas, educación ni residencia (sí dónde ocurrieron): no son un
# faltante de peso al azar, son otra clase de registro, y también inflan los
# "sin dato" de gestación y educación de 2022-2024. La base analítica es el
# total menos los de tipo 3. Con los microdatos públicos (sin identificador)
# no se puede comprobar si alguno duplica un certificado médico.

ANIO_FUENTE = 2021
RNEC = "Recuperados de la Registraduría (sin información clínica)"


def fuente_certificado() -> pd.DataFrame:
    """Registros por año, fuente del certificado y departamento de ocurrencia:
    cuántos son y cuántos tienen peso, BPN y departamento de residencia."""
    partes = []
    for anio in range(ANIO_FUENTE, cb.ANIO_FIN + 1):
        d = cb.dd.eevv.load(anio, tipo="nacimientos",
                            columns=["PESO_NAC", "TIPOFORMULARIO"])
        if "TIPOFORMULARIO" not in d:          # el año no trae la variable
            continue
        peso = cb._peso_grupo(pd.to_numeric(d["PESO_NAC"], errors="coerce"))
        g = pd.DataFrame({
            "anio": anio, "fuente": d["TIPOFORMULARIO"].astype("float"),
            "dpto_ocu": d["dpto_ocurrencia"].astype("float"), "n": 1.0,
            "con_peso": peso.notna().to_numpy(float),
            "bpn_n": peso.between(1, 4).fillna(False).to_numpy(float),
            "con_residencia": d["dpto_residencia"].isin(cb.DPTOS_VALIDOS)
                              .fillna(False).to_numpy(float)})
        partes.append(g.groupby(["anio", "fuente", "dpto_ocu"], as_index=False,
                                dropna=False).sum())
    columnas = ["anio", "fuente", "dpto_ocu", "n", "con_peso", "bpn_n", "con_residencia"]
    return pd.concat([pd.DataFrame(columns=columnas, dtype=float), *partes],
                     ignore_index=True)


def flujo_registros(celdas: pd.DataFrame, fuente: pd.DataFrame) -> pd.DataFrame:
    """Flujo de registros por año: totales, recuperados de la Registraduría,
    base analítica (total menos recuperados), sin peso en la base y con peso.
    pct_sin_peso_base es la serie principal del faltante de peso. Los de
    Notificación Étnica siguen en la base (tienen residencia y más de la mitad
    tiene peso): se reporta cuántos son y el % BPN sin ellos."""
    sin_res = celdas["dpto_res"].eq(0)
    d = pd.DataFrame({
        "registros": celdas.groupby("anio")["n"].sum(),
        "con_peso": celdas.groupby("anio")["con_peso"].sum(),
        "bpn_n": celdas.groupby("anio")["bpn_n"].sum(),
        "con_peso_sin_residencia": celdas["con_peso"].where(sin_res, 0.0)
                                   .groupby(celdas["anio"]).sum(),
        "_sin_peso_sin_residencia": (celdas["n"] - celdas["con_peso"]).where(sin_res, 0.0)
                                    .groupby(celdas["anio"]).sum()})
    por = lambda tipo: (fuente[fuente["fuente"].eq(tipo)].groupby("anio")
                        [["n", "con_peso", "bpn_n", "con_residencia"]].sum()
                        .reindex(d.index, fill_value=0.0))
    rnec, etnica = por(3), por(2)
    total = fuente.groupby("anio")["n"].sum()
    assert total.eq(d["registros"].reindex(total.index)).all(), \
        "los registros por fuente no suman los de las celdas"
    assert rnec["con_peso"].eq(0).all() and rnec["con_residencia"].eq(0).all(), \
        "hay recuperados de la Registraduría con peso o con residencia"
    assert (rnec["n"] <= d["_sin_peso_sin_residencia"]).all(), \
        "más recuperados que registros sin peso y sin residencia"
    d.insert(1, "recuperados_rnec", rnec["n"])
    d.insert(2, "base_analitica", d["registros"] - d["recuperados_rnec"])
    d.insert(3, "sin_peso_base", d["base_analitica"] - d["con_peso"])
    d.insert(4, "pct_sin_peso_base", 100 * d["sin_peso_base"] / d["base_analitica"])
    d["pct_sin_peso_total"] = 100 * (d["registros"] - d["con_peso"]) / d["registros"]
    d["notificacion_etnica"] = etnica["n"]
    d["pct_sin_peso_etnica"] = 100 * (1 - etnica["con_peso"] / etnica["n"])
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    d["pct_bpn_sin_etnica"] = (100 * (d["bpn_n"] - etnica["bpn_n"])
                               / (d["con_peso"] - etnica["con_peso"]))
    return d.drop(columns=["bpn_n", "_sin_peso_sin_residencia"]).reset_index()


def rnec_ocurrencia(fuente: pd.DataFrame) -> pd.DataFrame:
    """Recuperados de la Registraduría por departamento de ocurrencia y año,
    frente a los nacimientos con certificado ocurridos allí: qué parte de los
    nacimientos del departamento no tiene información clínica. No entran a los
    análisis por departamento, que usan la residencia."""
    f = fuente.assign(tipo=np.where(fuente["fuente"].eq(3), "recuperados_rnec",
                                    "con_certificado"))
    d = (f.groupby(["anio", "dpto_ocu", "tipo"])["n"].sum().unstack(fill_value=0.0)
         .reset_index())
    d = d[d["recuperados_rnec"] > 0]
    d["pct_rnec"] = 100 * d["recuperados_rnec"] / (d["recuperados_rnec"]
                                                   + d["con_certificado"])
    d["pct_del_total_rnec"] = (100 * d["recuperados_rnec"]
                               / d.groupby("anio")["recuperados_rnec"].transform("sum"))
    d.insert(1, "departamento_ocurrencia", d["dpto_ocu"].map(cb.DPTO_NOM))
    return (d.drop(columns="dpto_ocu").rename_axis(columns=None)
            .sort_values(["anio", "recuperados_rnec"], ascending=[True, False])
            .reset_index(drop=True))


# C5c. Origen del faltante de peso por año, en cuatro grupos disjuntos

def faltante_peso_origen(celdas: pd.DataFrame, flujo: pd.DataFrame) -> pd.DataFrame:
    """Por año y grupo (recuperados de la Registraduría; otros sin departamento
    de residencia; Chocó, La Guajira y Cesar; resto): % sin peso, parte del
    faltante nacional del año que aporta el grupo y % BPN con IC 95 % (Wilson)."""
    dpto = celdas["dpto_res"]
    grupo = np.select([dpto.eq(0), dpto.isin(cb.DPTOS_FALTANTE_PESO)],
                      ["Sin residencia", "Chocó, La Guajira y Cesar"], "Resto")
    d = (celdas.groupby(["anio", grupo])
         .agg(nacidos_vivos=("n", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum"))
         .rename_axis(["anio", "grupo"]).reset_index())
    # los recuperados (sin peso ni residencia) salen de "Sin residencia"
    rnec = flujo[flujo["recuperados_rnec"] > 0].set_index("anio")["recuperados_rnec"]
    sin_res = d["grupo"].eq("Sin residencia")
    d.loc[sin_res, "nacidos_vivos"] -= d.loc[sin_res, "anio"].map(rnec).fillna(0.0)
    d = pd.concat([d, pd.DataFrame({"anio": rnec.index, "grupo": RNEC,
                                    "nacidos_vivos": rnec.to_numpy(),
                                    "con_peso": 0.0, "bpn_n": 0.0})],
                  ignore_index=True).sort_values(["anio", "grupo"], ignore_index=True)
    d["sin_peso"] = d["nacidos_vivos"] - d["con_peso"]
    d["pct_sin_peso"] = 100 * d["sin_peso"] / d["nacidos_vivos"]
    d["pct_del_faltante"] = 100 * d["sin_peso"] / d.groupby("anio")["sin_peso"].transform("sum")
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    con = d["con_peso"] > 0
    d.loc[con, "li_bpn"], d.loc[con, "ls_bpn"] = cb.wilson(d.loc[con, "bpn_n"],
                                                         d.loc[con, "con_peso"])
    assert np.allclose(d.groupby("anio")["pct_del_faltante"].sum(), 100), \
        "las partes del faltante no suman 100"
    return d


# C6. Por sexo del nacido vivo y año (se excluye el sexo indeterminado)

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


# C7. Distribución del peso al nacer por año (%)

def distribucion_peso() -> pd.DataFrame:
    anios = list(range(cb.ANIO_INI, cb.ANIO_FIN + 1))
    d = cb.conteos(["peso_grupo"], anios)
    d["g"] = d["peso_grupo"].map(_MAP_PESO)
    d = d[d["g"].notna()]                     # sin peso no conocido (se excluyen)
    total = d.groupby("anio")["n"].sum()
    d = (d.assign(n=d["n"] * 100 / d["anio"].map(total))
         .pivot(index="anio", columns="g", values="n")
         .rename(columns=cb.PESO_GRUPO_ETQ).reset_index())
    d.columns.name = None
    return d


# C8. Semanas de gestación: qué explica la tendencia del BPN
# Etiquetas de gestacion_grupo de dane-data -> grupos de análisis; el resto
# (ignorado, sin información) va a "Sin dato". T_GES publica 28-37 semanas
# juntas; el bloque C9 las separa en los años con T_GES_AGRU_CIE.

_MAP_GEST = {"Menos de 22": "Menos de 28 sem.", "De 22 a 27": "Menos de 28 sem.",
             "De 28 a 37": "28 a 37 sem.", "De 38 a 41": "38 sem. y más",
             "De 42 y más": "38 sem. y más"}
_ORDEN_GEST = ["Menos de 28 sem.", "28 a 37 sem.", "38 sem. y más", "Sin dato"]


def _partes(d: pd.DataFrame, estrato: str) -> pd.DataFrame:
    """Por año y `estrato`: nacidos con peso, su parte de los nacidos del año,
    % BPN dentro del estrato y aporte al % BPN del año (los aportes suman la
    prevalencia del año)."""
    g = (d.groupby(["anio", estrato], as_index=False, observed=True)
         [["n", "bpn_n"]].sum().rename(columns={"n": "con_peso"}))
    total = g.groupby("anio")["con_peso"].transform("sum")
    g["pct_nacidos"] = 100 * g["con_peso"] / total
    g["pct_bpn"] = 100 * g["bpn_n"] / g["con_peso"]
    g["aporte_pp"] = 100 * g["bpn_n"] / total
    return g


def por_gestacion() -> pd.DataFrame:
    """Por año y grupo de semanas de gestación, entre los nacidos con peso:
    participación del grupo, % BPN dentro del grupo y aporte al % BPN nacional
    (los aportes de un año suman su prevalencia)."""
    d = cb.conteos(["gestacion_grupo", "bajo_peso"],
                   list(range(cb.ANIO_INI, cb.ANIO_FIN + 1)), dropna=False)
    d = d[d["bajo_peso"].notna()]
    d = d.assign(
        gestacion=pd.Categorical(d["gestacion_grupo"].astype(object).map(_MAP_GEST)
                                 .fillna("Sin dato"), _ORDEN_GEST),
        bpn_n=d["n"].where(d["bajo_peso"].astype(bool), 0))
    return _partes(d, "gestacion")


def _kitagawa(g: pd.DataFrame, estrato: str, a0: int, a1: int,
              parte: str = "pct_nacidos", tasa: str = "pct_bpn") -> pd.DataFrame:
    """Descomposición de Kitagawa del cambio, entre a0 y a1, de una tasa global
    (la suma de parte x tasa de los estratos): cuánto se debe a que cambió la
    composición por `estrato` y cuánto a que cambió la tasa dentro de cada
    estrato (en puntos porcentuales)."""
    t = g[g["anio"].isin([a0, a1])].pivot(index=estrato, columns="anio",
                                          values=[parte, tasa])
    s0, s1 = t[parte][a0].fillna(0) / 100, t[parte][a1].fillna(0) / 100
    # un estrato vacío en uno de los dos años toma la tasa del otro
    p0, p1 = t[tasa][a0].fillna(t[tasa][a1]), t[tasa][a1].fillna(t[tasa][a0])
    d = pd.DataFrame({
        f"{parte}_ini": 100 * s0, f"{parte}_fin": 100 * s1,
        f"{tasa}_ini": p0, f"{tasa}_fin": p1,
        "efecto_composicion_pp": (s1 - s0) * (p0 + p1) / 2,
        "efecto_tasas_pp": (p1 - p0) * (s0 + s1) / 2}).reset_index()
    d[estrato] = d[estrato].astype(str)
    total = d.drop(columns=estrato).sum()
    total[f"{tasa}_ini"], total[f"{tasa}_fin"] = (s0 * p0).sum(), (s1 * p1).sum()
    d = pd.concat([d, pd.DataFrame([{estrato: "Total", **total}])],
                  ignore_index=True)
    d["cambio_pp"] = d["efecto_composicion_pp"] + d["efecto_tasas_pp"]
    fin = d.iloc[-1]
    assert np.isclose(fin["cambio_pp"], fin[f"{tasa}_fin"] - fin[f"{tasa}_ini"]), \
        "la descomposición no suma el cambio observado"
    d.insert(0, "periodo", f"{a0} a {a1}")
    return d


def descomposicion_gestacion(g: pd.DataFrame, a0: int = cb.ANIO_INI,
                             a1: int = cb.ANIO_FIN) -> pd.DataFrame:
    """Kitagawa del cambio del % BPN entre a0 y a1 por semanas de gestación."""
    return _kitagawa(g, "gestacion", a0, a1)


# C9. El corrimiento hacia 28-37 semanas: en qué semanas y en quién ocurre
# T_GES agrupa 28-37 semanas. T_GES_AGRU_CIE (`prematuro` en dane-data) separa
# las <37, pero falta en 2005-2007, 2012, 2013 y 2020: en los demás años las dos
# juntas separan el pretérmino (28-36) de la semana 37.
# Base de todo el bloque: nacidos con peso y gestación conocidos.

ANIOS_CIE = [a for a in range(2008, cb.ANIO_FIN + 1) if a not in (2012, 2013, 2020)]
_ORDEN_FINA = ["Menos de 28 sem.", "28 a 36 sem.", "37 sem.", "28 a 37 sem.",
               "38 sem. y más"]
SUBCONJ_PARTO = {
    "Todos": None,
    "Partos únicos": "pluralidad == 'Único'",
    # sin la vía de documentación de las cesáreas
    "Únicos vaginales": "pluralidad == 'Único' and parto == 'Vaginal'",
    "Únicos por cesárea": "pluralidad == 'Único' and parto == 'Cesárea'",
}
FACTORES = {"parto": "Tipo de parto", "paridad": "Paridad", "educacion": "Educación",
            "regimen": "Régimen", "departamento": "Departamento"}
# (resultado, columna de conteo, año inicial): 28-37 existe en toda la serie
RESULTADOS_GEST = [("28 a 37 sem.", "n_2837", cb.ANIO_INI),
                   ("28 a 36 sem.", "n_2836", ANIOS_CIE[0]),
                   ("37 sem.", "n_37", ANIOS_CIE[0])]


def tabla_parto() -> pd.DataFrame:
    """Nacidos con peso y gestación conocidos por año, semanas, grupo de peso
    y los factores del parto y de la madre (una sola tabulación). Los
    faltantes de cada factor quedan como NaN: cada tabla los excluye de forma
    explícita y reporta su porcentaje."""
    d = cb.conteos(["gestacion_grupo", "prematuro", "peso_grupo", "cesarea",
                    "parto_multiple", "hijos_nacidos_vivos", "nivel_educativo_madre",
                    "regimen_salud", "dpto_residencia"],
                   list(range(cb.ANIO_INI, cb.ANIO_FIN + 1)), dropna=False)
    obj = lambda c: d[c].astype(object)
    gest3 = obj("gestacion_grupo").map(_MAP_GEST)
    fina = gest3.mask(gest3.eq("28 a 37 sem.") & obj("prematuro").eq(True), "28 a 36 sem.")
    fina = fina.mask(gest3.eq("28 a 37 sem.") & obj("prematuro").eq(False), "37 sem.")
    dpto = d["dpto_residencia"].astype("float")
    d = pd.DataFrame({
        "anio": d["anio"], "gest3": gest3,
        "gest": pd.Categorical(fina, _ORDEN_FINA),
        "peso": obj("peso_grupo").map(_MAP_PESO),
        "parto": obj("cesarea").map({True: "Cesárea", False: "Vaginal"}),
        "pluralidad": obj("parto_multiple").map({True: "Múltiple", False: "Único"}),
        "paridad": pd.cut(d["hijos_nacidos_vivos"].astype("float"), [0, 1, 2, np.inf],
                          labels=["1", "2", "3 o más"]).astype(object),
        "educacion": obj("nivel_educativo_madre").map(cb._MAP_EDUC4).map(cb.EDUC4_ETQ),
        # el régimen solo es comparable desde 2008
        "regimen": obj("regimen_salud").map(cb._MAP_REGIMEN).map(cb.REGIMEN_ETQ)
                   .where(d["anio"] >= 2008),
        "departamento": dpto.where(dpto.isin(cb.DPTOS_VALIDOS)).map(cb.DPTO_NOM),
        "n": d["n"]})
    d = d[d["peso"].notna() & d["gest3"].notna()]
    llaves = [c for c in d.columns if c != "n"]
    d = d.groupby(llaves, as_index=False, observed=True, dropna=False)["n"].sum()
    en_cie = d["anio"].isin(ANIOS_CIE)
    return d.assign(
        bpn_n=d["n"].where(d["peso"] <= 4, 0),
        n_2837=d["n"].where(d["gest3"].eq("28 a 37 sem."), 0),
        # sin T_GES_AGRU_CIE no se sabe: NaN, no cero
        n_2836=d["n"].where(d["gest"].eq("28 a 36 sem."), 0).where(en_cie),
        n_37=d["n"].where(d["gest"].eq("37 sem."), 0).where(en_cie))


def _subconjuntos(tp: pd.DataFrame, nombres=SUBCONJ_PARTO):
    for nombre in nombres:
        cond = SUBCONJ_PARTO[nombre]
        yield nombre, (tp.query(cond) if cond else tp)


def gestacion_fina(tp: pd.DataFrame) -> pd.DataFrame:
    """Parte de los nacidos, % BPN y aporte al BPN por semanas de gestación
    (28-36 y 37 separadas en los años con T_GES_AGRU_CIE), en todos los
    nacidos y en los subconjuntos de SUBCONJ_PARTO."""
    partes = [_partes(d, "gest").rename(columns={"gest": "gestacion"})
              .assign(subconjunto=nombre) for nombre, d in _subconjuntos(tp)]
    d = pd.concat(partes, ignore_index=True)
    return d[["subconjunto", *[c for c in d.columns if c != "subconjunto"]]]


def descomp_gestacion_conocida(tp: pd.DataFrame, fina: pd.DataFrame) -> pd.DataFrame:
    """Kitagawa del % BPN por semanas de gestación entre los nacidos con
    gestación conocida, por subconjunto: con 28-36 y 37 separadas (periodos
    entre años con T_GES_AGRU_CIE) y con 28-37 juntas desde 2005."""
    partes = []
    for nombre, d in _subconjuntos(tp):
        g = fina[fina["subconjunto"].eq(nombre)]
        g3 = _partes(d, "gest3").rename(columns={"gest3": "gestacion"})
        for tabla, periodos in [(g, [(2008, 2024), (2008, 2021), (2021, 2024)]),
                                (g3, [(2005, 2024), (2005, 2021)])]:
            partes += [_kitagawa(tabla, "gestacion", a0, a1).assign(subconjunto=nombre)
                       for a0, a1 in periodos]
    d = pd.concat(partes, ignore_index=True)
    return d[["subconjunto", *[c for c in d.columns if c != "subconjunto"]]]


# Criterio de decisión (fijado antes de ver los resultados por semana): ¿los
# nacimientos que se añadieron a un grupo de semanas pesan como los de ese
# grupo o como los nacidos a término? similitud = 1: como el grupo; 0: como
# los de 38 semanas y más.
SIMILITUD_GRUPO, SIMILITUD_TERMINO = 0.75, 0.25
CORRIMIENTO_MIN = 1.0      # pp; con menos, el peso de los añadidos es inestable


def prueba_peso(tp: pd.DataFrame) -> pd.DataFrame:
    """% BPN implícito de los nacimientos añadidos a cada grupo de semanas,
    (s1·p1 - s0·p0) / (s1 - s0), y su similitud con el grupo (año inicial)
    frente a los nacidos a término. Si se hubieran reclasificado nacidos a
    término de peso normal, o el grupo solo se hubiera diluido con ellos, los
    añadidos pesarían como los de término (similitud <= 0,25); con similitud
    >= 0,75 los añadidos son tan pequeños como el grupo: nacieron antes o se
    reclasificaron justo los pequeños (el peso solo no separa esas dos)."""
    filas = []
    for nombre, d in _subconjuntos(tp, ["Todos", "Únicos vaginales"]):
        for col, grupo, a0, a1 in [("gest3", "28 a 37 sem.", 2005, 2024),
                                   ("gest", "28 a 36 sem.", 2008, 2024),
                                   ("gest", "37 sem.", 2008, 2024),
                                   ("gest", "28 a 36 sem.", 2021, 2024),
                                   ("gest", "37 sem.", 2021, 2024)]:
            g = _partes(d, col).set_index(["anio", col])
            s0, s1 = g.loc[(a0, grupo), "pct_nacidos"], g.loc[(a1, grupo), "pct_nacidos"]
            p0, p1 = g.loc[(a0, grupo), "pct_bpn"], g.loc[(a1, grupo), "pct_bpn"]
            termino = g.loc[(a0, "38 sem. y más"), "pct_bpn"]
            hay = s1 - s0 >= CORRIMIENTO_MIN
            anadidos = (s1 * p1 - s0 * p0) / (s1 - s0) if hay else np.nan
            similitud = (anadidos - termino) / (p0 - termino)
            # banda de 2.500-2.999 g dentro del grupo: sube si entran pesos normales
            k = d[d[col].eq(grupo)]
            banda = (100 * k[k["peso"].eq(5)].groupby("anio")["n"].sum()
                     / k.groupby("anio")["n"].sum())
            filas.append({
                "subconjunto": nombre, "grupo": grupo, "periodo": f"{a0} a {a1}",
                "pct_nacidos_ini": s0, "pct_nacidos_fin": s1,
                "pct_bpn_ini": p0, "pct_bpn_fin": p1, "pct_bpn_termino_ini": termino,
                "pct_bpn_anadidos": anadidos, "similitud_grupo": similitud,
                "pct_2500_2999_ini": banda[a0], "pct_2500_2999_fin": banda[a1],
                "lectura": ("Sin corrimiento" if not hay else
                            "Tan pequeños como el grupo o más"
                            if similitud >= SIMILITUD_GRUPO else
                            "Como a término" if similitud <= SIMILITUD_TERMINO
                            else "Mixto")})
    return pd.DataFrame(filas)


def validacion_semana37(tp: pd.DataFrame) -> pd.DataFrame:
    """Semana 37 frente a 38 semanas y más, por año con T_GES_AGRU_CIE: parte
    de los nacidos, razón entre las dos y peso dentro de cada una (% BPN, %
    de 2.500-2.999 g y % de 3.500 g o más). Si nacidos de 38 semanas pasaran
    a anotarse en la 37, la razón saltaría y la semana 37 se haría más pesada."""
    d = tp[tp["gest"].isin(["37 sem.", "38 sem. y más"]) & tp["anio"].isin(ANIOS_CIE)]
    d = d.assign(n_2500=d["n"].where(d["peso"].eq(5), 0),
                 n_3500=d["n"].where(d["peso"].ge(7), 0))
    g = (d.groupby(["anio", "gest"], observed=True)[["n", "bpn_n", "n_2500", "n_3500"]]
         .sum())
    total = tp[tp["anio"].isin(ANIOS_CIE)].groupby("anio")["n"].sum()
    partes = {}
    for grupo, suf in [("37 sem.", "37"), ("38 sem. y más", "38mas")]:
        k = g.xs(grupo, level="gest")
        partes[f"pct_nacidos_{suf}"] = 100 * k["n"] / total
        partes[f"pct_bpn_{suf}"] = 100 * k["bpn_n"] / k["n"]
        partes[f"pct_2500_2999_{suf}"] = 100 * k["n_2500"] / k["n"]
        partes[f"pct_3500_mas_{suf}"] = 100 * k["n_3500"] / k["n"]
    v = pd.DataFrame(partes)
    v.insert(2, "razon_37_38mas", v["pct_nacidos_37"] / v["pct_nacidos_38mas"])
    return v.reset_index()


def gestacion_factores(tp: pd.DataFrame) -> pd.DataFrame:
    """Por año y categoría de cada factor: % de nacidos a 28-37, 28-36 y 37
    semanas y % BPN. Los factores se miran en partos únicos (cesárea, parto
    múltiple y pretérmino van acoplados); la pluralidad, en todos. Se excluyen
    los registros sin dato del factor: pct_sin_dato dice cuántos son."""
    unicos = tp.query(SUBCONJ_PARTO["Partos únicos"])
    cuentas = ["n", "bpn_n", "n_2837", "n_2836", "n_37"]
    partes = []
    for col, etq, base in [("pluralidad", "Pluralidad", tp),
                           *[(c, e, unicos) for c, e in FACTORES.items()]]:
        g = (base[base[col].notna()].groupby(["anio", col], observed=True)[cuentas]
             .sum(min_count=1).rename_axis(["anio", "categoria"]).reset_index())
        sin = 100 * (1 - g.groupby("anio")["n"].sum() / base.groupby("anio")["n"].sum())
        g.insert(0, "factor", etq)
        g["pct_sin_dato"] = g["anio"].map(sin)
        partes.append(g)
    d = pd.concat(partes, ignore_index=True)
    for cuenta, pct in [("n_2837", "pct_2837"), ("n_2836", "pct_2836"),
                        ("n_37", "pct_37"), ("bpn_n", "pct_bpn")]:
        d[pct] = 100 * d[cuenta] / d["n"]
    return d.drop(columns=["n_2837", "n_2836", "n_37"])


def descomp_corrimiento(tp: pd.DataFrame) -> pd.DataFrame:
    """Kitagawa del cambio en la parte de partos únicos a 28-37, 28-36 y 37
    semanas, por cada factor: cuánto se debe a que cambió el peso de sus
    categorías (composición) y cuánto a que la parte subió dentro de ellas.
    Las descomposiciones por factor no se suman entre sí (los factores se
    traslapan); "Conjunto" cruza tipo de parto, paridad y educación."""
    unicos = tp.query(SUBCONJ_PARTO["Partos únicos"])
    # NaN si falta alguno de los tres
    unicos = unicos.assign(conjunto=unicos["parto"] + " | " + unicos["paridad"]
                           + " | " + unicos["educacion"])
    filas = []
    for col, etq in [*FACTORES.items(), ("conjunto", "Conjunto")]:
        con = unicos[unicos[col].notna()]
        g = con.groupby(["anio", col], as_index=False, observed=True)[
            ["n", "n_2837", "n_2836", "n_37"]].sum(min_count=1)
        g["pct_nacidos"] = 100 * g["n"] / g.groupby("anio")["n"].transform("sum")
        for resultado, cuenta, a0 in RESULTADOS_GEST:
            a0 = max(a0, int(con["anio"].min()))        # régimen: desde 2008
            t = _kitagawa(g.assign(tasa=100 * g[cuenta] / g["n"]), col, a0,
                          cb.ANIO_FIN, tasa="tasa").iloc[-1]
            filas.append({
                "resultado": resultado, "factor": etq, "periodo": t["periodo"],
                "pct_ini": t["tasa_ini"], "pct_fin": t["tasa_fin"],
                "cambio_pp": t["cambio_pp"],
                "efecto_composicion_pp": t["efecto_composicion_pp"],
                "efecto_dentro_pp": t["efecto_tasas_pp"],
                "pct_por_composicion": 100 * t["efecto_composicion_pp"] / t["cambio_pp"]})
    return pd.DataFrame(filas)


def fetales_gestacion(tp: pd.DataFrame) -> pd.DataFrame:
    """Defunciones fetales de 28 semanas y más por año, frente a los nacidos
    vivos de esas semanas: si el aumento de nacidos vivos antes de término
    viniera de registrar como vivos los que antes eran muertes fetales, las
    fetales caerían en una cantidad parecida. Antes de 2008 las fetales no
    separan 28-37 de 38 y más."""
    f = cb.dd.eevv.serie_conteos(list(range(cb.ANIO_INI, cb.ANIO_FIN + 1)),
                                 tipo="fetales", por=["gestacion_grupo"], dropna=False)
    g = f["gestacion_grupo"].astype(object)
    grupo = np.select([g.eq("De 28 a 37"), g.isin(["De 38 a 41", "De 42 y más"]),
                       g.eq("De 28 y más"), g.isin(["Menos de 22", "De 22 a 27"])],
                      ["fetales_28_37", "fetales_38_mas", "fetales_28_mas_sin_separar",
                       "fetales_menos_28"], "fetales_sin_dato")
    d = (f.groupby(["anio", grupo])["n"].sum().unstack(fill_value=0)
         .rename_axis(columns=None))
    d["fetales_28_mas"] = d[["fetales_28_37", "fetales_38_mas",
                             "fetales_28_mas_sin_separar"]].sum(axis=1)
    vivos = tp[tp["gest3"].ne("Menos de 28 sem.")].groupby("anio")[["n", "n_2837"]].sum()
    d["vivos_28_mas"], d["vivos_28_37"] = vivos["n"], vivos["n_2837"]
    d["fetales_28_mas_por_mil"] = 1000 * d["fetales_28_mas"] / (d["fetales_28_mas"]
                                                                + d["vivos_28_mas"])
    d["fetales_28_37_por_mil"] = (1000 * d["fetales_28_37"]
                                  / (d["fetales_28_37"] + d["vivos_28_37"])
                                  ).where(d["fetales_28_mas_sin_separar"].eq(0))
    return d.reset_index()

# G1. Cruda con IC 95 % y estandarizada por edad

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
    ax.annotate("Nacidos con peso conocido. Los recuperados del cruce con la\n"
                "Registraduría (2022-2024) no traen peso y no entran.\n"
                "Fuente: DANE, Estadísticas Vitales. Cálculos propios.",
                xy=(1, -0.14), xycoords="axes fraction", ha="right",
                fontsize=7, va="top")
    fig.tight_layout()
    return fig


FUENTE = "Fuente: DANE, Estadísticas Vitales. Cálculos propios."

# G12. Mapa de calor: % BPN por departamento y año

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


# G13. % BPN por sexo del nacido vivo

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


# G14. Distribución del peso al nacer por año (áreas apiladas)

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


# C10. Migración: madres nacidas en Venezuela
# CODPAISNACMAD (país de nacimiento de la madre, códigos M49) existe desde
# 2024; el archivo de 2025 es preliminar y solo se usa aquí, como réplica. Es
# país de nacimiento, no nacionalidad (los microdatos no la traen): incluye a
# colombianas retornadas nacidas en Venezuela. Los recuperados de la
# Registraduría no traen país: quedan en "Sin dato" y fuera de las tasas.

ANIOS_PAIS = (2024, 2025)
_PAISES = ["Colombia", "Venezuela", "Otro país", "Sin dato"]


def _registros_pais(anio: int) -> pd.DataFrame:
    """Un registro por nacido del año, con el país de nacimiento de la madre
    y las variables de la comparación (mismas definiciones del pipeline)."""
    d = cb.dd.eevv.load(anio, tipo="nacimientos",
                        columns=["PESO_NAC", "CODPAISNACMAD", "TIPOFORMULARIO"])
    pais = pd.to_numeric(d["CODPAISNACMAD"], errors="coerce")
    peso = cb._peso_grupo(pd.to_numeric(d["PESO_NAC"], errors="coerce"))
    obj = lambda c: d[c].astype(object)
    dpto = d["dpto_residencia"].astype("float")
    return pd.DataFrame({
        "anio": anio,
        "pais": np.select([pais.eq(170), pais.eq(862), pais.notna()], _PAISES[:3], _PAISES[3]),
        "rnec": d["TIPOFORMULARIO"].eq(3).to_numpy(float),
        "con_peso": peso.notna().to_numpy(float),
        "bpn": peso.between(1, 4).fillna(False).to_numpy(float),
        "g2837": obj("gestacion_grupo").eq("De 28 a 37").to_numpy(float),
        "pretermino": obj("prematuro").eq(True).to_numpy(float),
        "cesarea": obj("cesarea").eq(True).to_numpy(float),
        "no_asegurada": obj("regimen_salud").eq("No asegurado").to_numpy(float),
        "baja_educ": obj("nivel_educativo_madre").map(cb._MAP_EDUC4).isin([1, 2]).to_numpy(float),
        "dpto_res": dpto.where(dpto.isin(cb.DPTOS_VALIDOS)).to_numpy(),
        "edad6": obj("edad_madre").map(cb._MAP_EDAD6).to_numpy(float),
        "educ4": obj("nivel_educativo_madre").map(cb._MAP_EDUC4).to_numpy(float)})


def migracion_pais(reg: pd.DataFrame) -> pd.DataFrame:
    """Por año y país de nacimiento de la madre: nacidos, % sin peso, % BPN
    con IC 95 % (Wilson), diferencia frente a las nacidas en Colombia con IC y
    valor p (dos proporciones), y composición entre los nacidos con peso."""
    from scipy.stats import norm
    con = reg[reg["con_peso"].eq(1)]
    g = reg.groupby(["anio", "pais"]).agg(nacidos=("con_peso", "size"),
                                          recuperados_rnec=("rnec", "sum"),
                                          con_peso=("con_peso", "sum"), bpn_n=("bpn", "sum"))
    for col in ["g2837", "pretermino", "cesarea", "no_asegurada", "baja_educ"]:
        g[f"pct_{col}"] = 100 * con.groupby(["anio", "pais"])[col].mean()
    g = g.reset_index()
    g.insert(3, "pct_nacidos", 100 * g["nacidos"] / g.groupby("anio")["nacidos"].transform("sum"))
    g["pct_sin_peso"] = 100 * (1 - g["con_peso"] / g["nacidos"])
    g["pct_bpn"] = 100 * g["bpn_n"] / g["con_peso"]
    g["li_bpn"], g["ls_bpn"] = cb.wilson(g["bpn_n"], g["con_peso"])
    col = g[g["pais"].eq("Colombia")].set_index("anio")
    p, n = g["bpn_n"] / g["con_peso"], g["con_peso"]
    p0, n0 = g["anio"].map(col["bpn_n"] / col["con_peso"]), g["anio"].map(col["con_peso"])
    ee = np.sqrt(p * (1 - p) / n + p0 * (1 - p0) / n0)
    g["dif_vs_colombia_pp"] = (100 * (p - p0)).where(g["pais"].ne("Colombia"))
    g["dif_li"], g["dif_ls"] = g["dif_vs_colombia_pp"] - 196 * ee, g["dif_vs_colombia_pp"] + 196 * ee
    g["p_dif"] = pd.Series(2 * norm.sf(np.abs(p - p0) / ee),
                           index=g.index).where(g["pais"].ne("Colombia"))
    orden = pd.Categorical(g["pais"], _PAISES)
    return g.assign(_o=orden).sort_values(["anio", "_o"]).drop(columns="_o").reset_index(drop=True)


def migracion_cota(reg: pd.DataFrame, nacional: pd.DataFrame) -> pd.DataFrame:
    """Cuánto del BPN nacional puede deberse a las madres nacidas en
    Venezuela, por año: BPN con y sin ellas (aporte al nivel) frente al alza
    desde 2005; qué tan bien las aproxima el régimen "no asegurado"
    (sensibilidad y valor predictivo); y razón de prevalencias frente a las
    nacidas en Colombia, cruda y ajustada por edad, educación y departamento
    (Poisson, errores agrupados por departamento). La ajustada no es causal:
    la educación puede estar en el camino entre el origen y el BPN."""
    from obj3_educacion import _ajustar, _dummies
    bpn_2005 = float(nacional.loc[nacional["anio"].eq(cb.ANIO_INI), "pct_bpn"].iloc[0])
    filas = []
    for anio, d in reg[reg["con_peso"].eq(1)].groupby("anio"):
        ven = d["pais"].eq("Venezuela")
        todos, sin_ven = 100 * d["bpn"].mean(), 100 * d.loc[~ven, "bpn"].mean()
        brecha = 100 * (d.loc[ven, "bpn"].mean() - d.loc[d["pais"].eq("Colombia"), "bpn"].mean())
        fila = {"anio": anio, "pct_nacidas_venezuela": 100 * ven.mean(),
                "brecha_bpn_pp": brecha, "bpn_todos": todos,
                "bpn_sin_nacidas_venezuela": sin_ven, "aporte_al_nivel_pp": todos - sin_ven,
                "alza_desde_2005_pp": todos - bpn_2005,
                "aporte_pct_del_alza": 100 * (todos - sin_ven) / (todos - bpn_2005),
                # el régimen "no asegurado" como aproximación de nacida en Venezuela
                "no_aseg_valor_predictivo": 100 * ven[d["no_asegurada"].eq(1)].mean(),
                "no_aseg_sensibilidad": 100 * d.loc[ven, "no_asegurada"].mean()}
        k = d[d["pais"].isin(["Colombia", "Venezuela"]) & d["dpto_res"].notna()
              & d["edad6"].notna() & d["educ4"].notna()]
        k = (k.assign(ven=k["pais"].eq("Venezuela").astype(int))
             .groupby(["dpto_res", "edad6", "educ4", "ven"], as_index=False)
             .agg(n=("bpn", "size"), casos=("bpn", "sum")))
        for etq, X in [("cruda", k[["ven"]]),
                       ("ajustada", pd.concat([k["ven"], _dummies(k, "edad6", 2),
                                               _dummies(k, "educ4", 4),
                                               _dummies(k, "dpto_res", 5)], axis=1))]:
            b, se, _, _ = _ajustar(k, "casos", X, "n")
            fila |= {f"rp_{etq}": np.exp(b["ven"]),
                     f"rp_{etq}_li": np.exp(b["ven"] - 1.96 * se["ven"]),
                     f"rp_{etq}_ls": np.exp(b["ven"] + 1.96 * se["ven"])}
        filas.append(fila)
    return pd.DataFrame(filas)


def migracion_dptos(reg: pd.DataFrame) -> pd.DataFrame:
    """Por año y departamento de residencia: % de nacidos (con peso) de madres
    nacidas en Venezuela y % BPN según el país de nacimiento de la madre."""
    d = reg[reg["con_peso"].eq(1) & reg["dpto_res"].notna()
            & reg["pais"].isin(["Colombia", "Venezuela"])]
    g = (d.groupby(["anio", "dpto_res", "pais"]).agg(n=("bpn", "size"), bpn=("bpn", "sum"))
         .unstack("pais", fill_value=0.0))
    t = pd.DataFrame({
        "con_peso": g["n"].sum(axis=1), "nacidas_venezuela": g["n"]["Venezuela"],
        "pct_nacidas_venezuela": 100 * g["n"]["Venezuela"] / g["n"].sum(axis=1),
        "pct_bpn_colombia": 100 * g["bpn"]["Colombia"] / g["n"]["Colombia"],
        "pct_bpn_venezuela": 100 * g["bpn"]["Venezuela"] / g["n"]["Venezuela"]}).reset_index()
    t.insert(2, "departamento", t["dpto_res"].map(cb.DPTO_NOM))
    return t.sort_values(["anio", "pct_nacidas_venezuela"], ascending=[True, False],
                         ignore_index=True)


def no_aseguradas_anual(celdas: pd.DataFrame) -> pd.DataFrame:
    """Aproximación administrativa a la migración para la serie, 2018-2024:
    nacimientos de madres no aseguradas. No es la parte migrante de cada año
    (antes de 2018 es afiliación de colombianas; desde 2021 cae por el
    Estatuto Temporal de Protección, no porque haya menos madres migrantes).
    Sirve para una cosa: ver cuánto se mueve el BPN al quitarlas."""
    d = celdas[celdas["anio"].ge(2018) & celdas["regimen"].notna()]
    no = d["regimen"].eq(3)
    g = lambda col, m: d[col].where(m, 0.0).groupby(d["anio"]).sum()
    t = pd.DataFrame({
        "pct_no_aseguradas": 100 * g("n", no) / d.groupby("anio")["n"].sum(),
        "bpn_no_aseguradas": 100 * g("bpn_n", no) / g("con_peso", no),
        "bpn_aseguradas": 100 * g("bpn_n", ~no) / g("con_peso", ~no),
        "bpn_todas": 100 * d.groupby("anio")["bpn_n"].sum() / d.groupby("anio")["con_peso"].sum()})
    t["diferencia_pp"] = t["bpn_todas"] - t["bpn_aseguradas"]
    return t.reset_index()


def alza_por_periodo(celdas: pd.DataFrame) -> pd.DataFrame:
    """% BPN en 2005, 2015 y 2024 y su cambio antes y después de 2015 (la
    migración desde Venezuela crece desde 2016-2017), en todo el país y en
    los subconjuntos de SUBCONJUNTOS."""
    filas = []
    for nombre, cond in SUBCONJUNTOS.items():
        d = celdas.query(cond) if cond else celdas
        p = 100 * d.groupby("anio")["bpn_n"].sum() / d.groupby("anio")["con_peso"].sum()
        filas.append({"subconjunto": nombre, "bpn_2005": p[2005], "bpn_2015": p[2015],
                      "bpn_2024": p[2024], "cambio_2005_2015_pp": p[2015] - p[2005],
                      "cambio_2015_2024_pp": p[2024] - p[2015]})
    return pd.DataFrame(filas)


NOTAS_MIGRACION = pd.DataFrame([
    ("Definición", "«Madres nacidas en Venezuela»: país de nacimiento (CODPAISNACMAD = 862, "
     "códigos M49), no nacionalidad; los microdatos no traen nacionalidad, así que incluye a "
     "colombianas retornadas nacidas allá."),
    ("Definición", "La variable existe desde 2024. 2025 es preliminar: se usa como réplica y "
     "no entra a ninguna serie."),
    ("Denominadores", "Los recuperados del cruce con la Registraduría (RNEC) no traen país ni "
     "peso: están en «Sin dato» y fuera de todas las tasas. «Sin dato» = RNEC + certificados "
     "sin país."),
    ("Denominadores", "Las tasas usan todos los nacidos con peso (únicos y múltiples), como "
     "el resto del objetivo 1. Que los totales cuadren con la hoja Nacional es consistencia "
     "interna, no validación externa."),
    ("Límite", "Sesgo salmón: si los embarazos complicados regresan a Venezuela antes del "
     "parto, el BPN de las que paren en Colombia queda subestimado."),
    ("Límite", "Solo se ven nacidos vivos: no hay defunciones fetales por país de la madre."),
    ("Límite", "«No asegurada» mide política de afiliación (pandemia, Estatuto Temporal de "
     "Protección): no sirve para estimar cuántas madres migrantes hay cada año."),
    ("Límite", "Sin departamentos de frontera es otro estimando (el resto del país), y en La "
     "Guajira confunde migración con composición étnica (wayuu)."),
    ("Límite", "La razón de prevalencias ajustada no es causal: la educación puede estar en "
     "el camino entre el origen y el BPN."),
    ("Glosario", "BPN: bajo peso al nacer (< 2.500 g). RP: razón de prevalencias. APC: cambio "
     "porcentual anual. RII: índice relativo de desigualdad. PAF: fracción atribuible "
     "poblacional. RNEC: Registraduría Nacional del Estado Civil. Pretérmino: menos de 37 "
     "semanas de gestación."),
], columns=["tema", "nota"])


# G15. Nacidos antes de las 38 semanas: pretérmino (28-36) y semana 37

def grafico_semanas(fina: pd.DataFrame):
    import matplotlib.pyplot as plt
    anios = range(cb.ANIO_INI, cb.ANIO_FIN + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

    def serie(sub, grupos):                  # años sin dato: la línea se corta
        d = fina[fina["subconjunto"].eq(sub) & fina["gestacion"].isin(grupos)]
        return d.groupby("anio")["pct_nacidos"].sum().reindex(anios)

    todas = serie("Todos", ["28 a 36 sem.", "37 sem.", "28 a 37 sem."])
    ejes[0].plot(todas.index, todas, color="#666666", ls="--", lw=1.8,
                 label="28 a 37 semanas")
    for grupo, color in [("37 sem.", "#A63603"), ("28 a 36 sem.", "#08306B")]:
        d = serie("Todos", [grupo])
        ejes[0].plot(d.index, d, color=color, marker="o", ms=4, lw=2,
                     label="Semana 37" if grupo == "37 sem." else "28 a 36 semanas")
    ejes[0].set_title("Todos los nacidos", fontsize=10.5)
    ejes[0].set_ylabel("% de nacidos vivos")
    for sub, color, trazo in [("Únicos por cesárea", "#A63603", "-"),
                              ("Únicos vaginales", "#08306B", "--")]:
        d = serie(sub, ["37 sem."])
        ejes[1].plot(d.index, d, color=color, ls=trazo, marker="o", ms=4, lw=2,
                     label=sub)
    ejes[1].set_title("Semana 37 en partos únicos, según tipo de parto", fontsize=10.5)
    for ax in ejes:
        ax.set_xticks(range(2005, 2025, 3))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Nacimientos antes de las 38 semanas de gestación, 2005-2024",
                 fontsize=12)
    fig.text(0.01, 0.005,
             "Nacidos con peso y gestación conocidos. La semana 37 solo se separa en "
             "los años con T_GES_AGRU_CIE (faltan 2005-2007, 2012, 2013 y 2020).\n"
             + FUENTE, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)

LOG = "log_01_base_objetivo1"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 1; cada clave es una hoja del xlsx."""
    largo = departamento_largo(celdas)
    gest = por_gestacion()
    tp = tabla_parto()
    fina = gestacion_fina(tp)
    flujo = flujo_registros(celdas, fuente := fuente_certificado())
    nacional = nacional_anio(celdas)
    reg = pd.concat([_registros_pais(a) for a in ANIOS_PAIS], ignore_index=True)
    return {
        # la base analítica (sin los recuperados de la Registraduría) es la
        # serie principal del faltante de peso: pct_sin_peso_base
        "Nacional": nacional.merge(
            flujo[["anio", "recuperados_rnec", "base_analitica", "pct_sin_peso_base"]],
            on="anio", validate="one_to_one"),
        "Flujo_registros": flujo,
        "RNEC_por_ocurrencia": rnec_ocurrencia(fuente),
        "Nacional_estandarizada": nacional_estandarizada(celdas),
        "APC": apc(celdas),
        "Departamento_largo": largo,
        "Departamento_%BPN": departamento_ancho(largo),
        "Calidad_registro_peso": calidad_peso(celdas),
        "Faltante_peso_origen": faltante_peso_origen(celdas, flujo),
        "Sexo": por_sexo(),
        "Distribucion_peso": distribucion_peso(),
        "Gestacion": gest,
        "Descomposicion_gestacion": descomposicion_gestacion(gest),
        "Gestacion_fina": fina,
        "Descomp_gestacion_conocida": descomp_gestacion_conocida(tp, fina),
        "Prueba_peso_gestacion": prueba_peso(tp),
        "Semana37_validacion": validacion_semana37(tp),
        "Gestacion_factores": gestacion_factores(tp),
        "Descomp_corrimiento": descomp_corrimiento(tp),
        "Fetales_gestacion": fetales_gestacion(tp),
        "Migracion_pais": migracion_pais(reg),
        "Migracion_cota": migracion_cota(reg, nacional),
        "Migracion_dptos": migracion_dptos(reg),
        "No_aseguradas_anual": no_aseguradas_anual(celdas),
        "Alza_por_periodo": alza_por_periodo(celdas),
        "Migracion_notas": NOTAS_MIGRACION,
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
    print("\n--- C5b. Flujo de registros (base analítica = total - recuperados RNEC) ---")
    print(r["Flujo_registros"].round(2).to_string(index=False))
    ro = r["RNEC_por_ocurrencia"]
    print("\nRecuperados de la Registraduría por departamento de ocurrencia (5 mayores por año):")
    print(ro.groupby("anio").head(5).round(1).to_string(index=False))
    print("\n--- C5c. Origen del faltante de peso (% del faltante nacional) ---")
    print(r["Faltante_peso_origen"].pivot(index="anio", columns="grupo",
                                          values="pct_del_faltante")
          .round(1).to_string())
    print("\n--- C6. Por sexo ---")
    print(r["Sexo"][["anio", "pct_bpnHombre", "pct_bpnMujer"]]
          .round(2).to_string(index=False))
    print("\n--- C7. Distribución del peso ---")
    print(r["Distribucion_peso"].round(2).to_string(index=False))
    print("\n--- C8. Descomposición del cambio del BPN por semanas de gestación ---")
    print(r["Descomposicion_gestacion"].round(2).to_string(index=False))
    print("\n--- C9. Corrimiento hacia 28-37 semanas (peso y gestación conocidos) ---")
    fina = r["Gestacion_fina"]
    print("% de nacidos por semanas (todos):")
    print(fina[fina["subconjunto"].eq("Todos")]
          .pivot(index="anio", columns="gestacion", values="pct_nacidos")
          .round(1).to_string())
    dg = r["Descomp_gestacion_conocida"]
    print("\nKitagawa del % BPN por semanas (totales):")
    print(dg[dg["gestacion"].eq("Total")][
        ["subconjunto", "periodo", "pct_bpn_ini", "pct_bpn_fin", "cambio_pp",
         "efecto_composicion_pp", "efecto_tasas_pp"]].round(2).to_string(index=False))
    print("\nPrueba del peso de los nacimientos añadidos a cada grupo:")
    print(r["Prueba_peso_gestacion"].round(2).to_string(index=False))
    print("\nSemana 37 frente a 38 semanas y más, por año:")
    print(r["Semana37_validacion"].round(2).to_string(index=False))
    print("\nKitagawa del corrimiento por factor (partos únicos):")
    print(r["Descomp_corrimiento"].round(2).to_string(index=False))
    print("\nDefunciones fetales de 28 semanas y más:")
    print(r["Fetales_gestacion"].round(2).to_string(index=False))
    print("\n--- C10. Migración: madres nacidas en Venezuela (2025 es preliminar) ---")
    print(r["Migracion_pais"].round(2).to_string(index=False))
    print("\nCota del aporte al BPN nacional y razón de prevalencias:")
    print(r["Migracion_cota"].round(3).T.to_string(header=False))
    print("\nDepartamentos con más nacidos de madres nacidas en Venezuela (%):")
    print(r["Migracion_dptos"].groupby("anio").head(6).round(2).to_string(index=False))
    print("\nMadres no aseguradas, 2018-2024 (aproximación administrativa):")
    print(r["No_aseguradas_anual"].round(2).to_string(index=False))
    print("\nAlza del BPN antes y después de 2015:")
    print(r["Alza_por_periodo"].round(2).to_string(index=False))


def graficos(r: dict) -> dict:
    cb.iniciar_estilo()
    return {"G1_bpn_nacional": grafico_g1(r["Nacional"], r["Nacional_estandarizada"]),
            "G12_mapa_calor_departamentos": grafico_mapa_calor(r["Departamento_%BPN"]),
            "G13_bpn_por_sexo": grafico_sexo(r["Sexo"]),
            "G14_distribucion_peso": grafico_distribucion(r["Distribucion_peso"]),
            "G15_semanas_gestacion": grafico_semanas(r["Gestacion_fina"])}
