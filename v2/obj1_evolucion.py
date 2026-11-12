#  Parte A/B (base y variables armonizadas) la hace comun_bpn.cargar_celdas.
#  Aquí va la Objetivo 1: Objetivo 1 - evolución y distribución del
#  BPN: prevalencia nacional (cruda y estandarizada por edad), APC (Poisson
#  robusto, también por subconjunto de registros), prevalencia por
#  departamento, calidad del registro del peso y origen del faltante de peso,
#  por sexo y distribución del peso, con IC 95 % de Wilson, y descomposición
#  de la tendencia por semanas de gestación (Kitagawa). El bloque C9 mira el
#  corrimiento hacia 28-37 semanas: pretérmino y semana 37 por separado, en
#  quién ocurre, prueba con el peso y contraste con las defunciones fetales.
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
# tendencia (desde 2022 el faltante de peso está en los registros sin
# departamento de residencia, dpto_res = 0)
SUBCONJUNTOS = {
    "Colombia": None,
    "Con residencia conocida": "dpto_res > 0",
    "Con residencia, sin Chocó, La Guajira y Cesar":
        f"dpto_res > 0 and dpto_res not in {cb.DPTOS_FALTANTE_PESO}",
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


# C5b. Origen del faltante de peso por año, en tres grupos disjuntos

def faltante_peso_origen(celdas: pd.DataFrame) -> pd.DataFrame:
    """Por año y grupo (sin departamento de residencia; Chocó, La Guajira y
    Cesar; resto): % sin peso, parte del faltante nacional del año que aporta
    el grupo y % BPN con IC 95 % (Wilson)."""
    dpto = celdas["dpto_res"]
    grupo = np.select([dpto.eq(0), dpto.isin(cb.DPTOS_FALTANTE_PESO)],
                      ["Sin residencia", "Chocó, La Guajira y Cesar"], "Resto")
    d = (celdas.groupby(["anio", grupo])
         .agg(nacidos_vivos=("n", "sum"), con_peso=("con_peso", "sum"),
              bpn_n=("bpn_n", "sum"))
         .rename_axis(["anio", "grupo"]).reset_index())
    d["sin_peso"] = d["nacidos_vivos"] - d["con_peso"]
    d["pct_sin_peso"] = 100 * d["sin_peso"] / d["nacidos_vivos"]
    d["pct_del_faltante"] = 100 * d["sin_peso"] / d.groupby("anio")["sin_peso"].transform("sum")
    d["pct_bpn"] = 100 * d["bpn_n"] / d["con_peso"]
    d["li_bpn"], d["ls_bpn"] = cb.wilson(d["bpn_n"], d["con_peso"])
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
    ax.annotate("Desde 2022 la base incluye registros de la\n"
                "Registraduría sin información de peso.\n"
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
    return {
        "Nacional": nacional_anio(celdas),
        "Nacional_estandarizada": nacional_estandarizada(celdas),
        "APC": apc(celdas),
        "Departamento_largo": largo,
        "Departamento_%BPN": departamento_ancho(largo),
        "Calidad_registro_peso": calidad_peso(celdas),
        "Faltante_peso_origen": faltante_peso_origen(celdas),
        "Sexo": por_sexo(),
        "Distribucion_peso": distribucion_peso(),
        "Gestacion": gest,
        "Descomposicion_gestacion": descomposicion_gestacion(gest),
        "Gestacion_fina": fina,
        "Descomp_gestacion_conocida": descomp_gestacion_conocida(tp, fina),
        "Prueba_peso_gestacion": prueba_peso(tp),
        "Gestacion_factores": gestacion_factores(tp),
        "Descomp_corrimiento": descomp_corrimiento(tp),
        "Fetales_gestacion": fetales_gestacion(tp),
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
    print("\n--- C5b. Origen del faltante de peso (% del faltante nacional) ---")
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
    print("\nKitagawa del corrimiento por factor (partos únicos):")
    print(r["Descomp_corrimiento"].round(2).to_string(index=False))
    print("\nDefunciones fetales de 28 semanas y más:")
    print(r["Fetales_gestacion"].round(2).to_string(index=False))


def graficos(r: dict) -> dict:
    cb.iniciar_estilo()
    return {"G1_bpn_nacional": grafico_g1(r["Nacional"], r["Nacional_estandarizada"]),
            "G12_mapa_calor_departamentos": grafico_mapa_calor(r["Departamento_%BPN"]),
            "G13_bpn_por_sexo": grafico_sexo(r["Sexo"]),
            "G14_distribucion_peso": grafico_distribucion(r["Distribucion_peso"]),
            "G15_semanas_gestacion": grafico_semanas(r["Gestacion_fina"])}
