#  Objetivo 3: relación entre el nivel educativo de la madre / régimen de
#  afiliación (proxy socioeconómico, 2008-2024) y la prevalencia de BPN.
#   A. Prevalencia cruda y estandarizada por edad de la madre (directa).
#   B. Razones de prevalencia con Poisson (registro logarítmico) y errores
#      estándar agrupados por departamento: modelos M0 a M5. El principal es
#      M4 (edad, año, altitud y efectos fijos de departamento); M5 repite ese
#      ajuste con el BPN a término en partos únicos (resultado secundario).
#   C. Índice relativo (RII) y de la pendiente (SII) por educación con ridit,
#      con el ajuste de M4, para los dos resultados y por quinquenio; con
#      cotas extremas para la educación faltante (todas al nivel más bajo o
#      al más alto) y el % faltante por departamento.
#   D. Sensibilidad del gradiente: sin ajuste por régimen, con la educación
#      faltante como categoría y con paridad, etnia y área de residencia.
#   E. Tendencia del RII (interacción ridit x quinquenio y por década) con
#      bootstrap wild por departamento, y controles con la semana 37.
#  No se ajusta por gestación, controles prenatales ni tipo de parto
#  (mediadores, ver nota técnica del análisis).
#
#  "A término" significa siempre 38 semanas y más en partos únicos, con T_GES
#  (los 20 años). La semana 37 solo se distingue en los años con
#  T_GES_AGRU_CIE; los resultados que la usan son controles con esos años y
#  no se mezclan con la serie principal.
#  Salidas: Objetivo3_Educacion_Regimen_BPN.xlsx, G9, G10 y log.

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

import comun_bpn as cb

EXCEL = "Objetivo3_Educacion_Regimen_BPN.xlsx"
Z95 = 1.96
REPS_WILD = 9_999            # réplicas del bootstrap wild
SEMILLA = 20241


# 0. Celdas del objetivo 3

def celdas_obj3(celdas: pd.DataFrame, extra: tuple = ()) -> pd.DataFrame:
    """Nacidos con dpto válido, peso y edad conocidos, agrupados por
    (anio, dpto_res, educ4, regimen, edad6, alt_cat) y las llaves `extra`; los
    faltantes de educ4/régimen/alt_cat/extra pasan a la categoría 9 (se
    excluyen por modelo).
    A nivel de celda con exposición, los coeficientes de Poisson y sus errores
    agrupados coinciden con los de los datos individuales (nota técnica)."""
    d = celdas
    d = d[d["dpto_res"].gt(0) & d["con_peso"].gt(0) & d["edad6"].notna()].copy()
    for col in ["educ4", "regimen", "alt_cat", *extra]:
        d[col] = d[col].fillna(9).astype(int)
    c = (d.groupby(["anio", "dpto_res", "educ4", "regimen", "edad6", "alt_cat", *extra],
                   as_index=False, observed=True)
         .agg(n=("con_peso", "sum"), casos=("bpn_n", "sum"),
              n_term=("n_term", "sum"), casos_term=("bpn_term_n", "sum"),
              n_gest=("n_gest_u", "sum"), n_sem37=("n_sem37", "sum"),
              casos_sem37=("bpn_sem37", "sum")))
    # únicos de 37 semanas y más: solo en los años con T_GES_AGRU_CIE; para
    # comparar, también los de 38 y más en esos mismos años (_cie)
    con_cie = c.groupby("anio")["n_gest"].transform("sum") > 0
    return c.assign(n_term37=(c["n_sem37"] + c["n_term"]).where(con_cie, 0.0),
                    casos_term37=(c["casos_sem37"] + c["casos_term"]).where(con_cie, 0.0),
                    n_term_cie=c["n_term"].where(con_cie, 0.0),
                    casos_term_cie=c["casos_term"].where(con_cie, 0.0))


# A. Prevalencia cruda y estandarizada por edad

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


# B. Razones de prevalencia: GLM Poisson, exposición n, cluster dpto_res

def _dummies(df: pd.DataFrame, col: str, ref: int) -> pd.DataFrame:
    """Indicadoras de `col` (strings) sin la categoría de referencia."""
    d = pd.get_dummies(df[col].astype(int).astype(str), prefix=col, dtype=float)
    return d.drop(columns=f"{col}_{ref}", errors="ignore")


def _ajustar(df: pd.DataFrame, y: str, X: pd.DataFrame, expo: str):
    """GLM Poisson con exposición log; errores agrupados por departamento.
    Devuelve (coeficientes, errores estándar, N celdas, #dptos)."""
    X = sm.add_constant(X.reindex(df.index).astype(float), has_constant="add")
    res = sm.GLM(df[y].astype(float), X, family=sm.families.Poisson(),
                 offset=np.log(df[expo].astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": df["dpto_res"].to_numpy(),
                                      "use_correction": True, "df_correction": True})
    return dict(res.params), dict(res.bse), res.nobs, df["dpto_res"].nunique()


def _filas(modelo: str, b: dict, se: dict, etqs: dict, var: str, niveles,
           N: float, G: int) -> list[dict]:
    """guardar_rp del análisis: exp(b), IC y p para los niveles pedidos."""
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

    # M3. Educación y régimen juntos, ajustados por edad y año
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

    # M4. M3 + altitud del municipio + efectos fijos de departamento [principal]
    # Sin altitud ni departamento, el gradiente educativo sale confundido: las
    # madres más educadas viven más en Bogotá y el altiplano, donde hay más BPN.
    da = dm[dm["alt_cat"] != 9]
    X4 = pd.concat([_dummies(da, "educ4", 4), _dummies(da, "regimen", 1),
                    _dummies(da, "edad6", 2), _dummies(da, "anio", 2008),
                    _dummies(da, "alt_cat", 1), _dummies(da, "dpto_res", 5)], axis=1)
    b, se, N, G = _ajustar(da, "casos", X4, "n")
    filas += _filas("M4 M3 + altitud + EF departamento", b, se, cb.EDUC4_ETQ,
                     "educ4", [1, 2, 3], N, G)
    filas += _filas("M4 M3 + altitud + EF departamento", b, se, cb.REGIMEN_ETQ,
                     "regimen", [2, 3], N, G)

    # M5. M4 con BPN a término en partos únicos (proxy de restricción del
    # crecimiento): mismo ajuste que el principal, resultado secundario
    dt = da[da["n_term"] > 0]
    b, se, N, G = _ajustar(dt, "casos_term", X4, "n_term")
    filas += _filas("M5 M4 con BPN a término (únicos >=38 sem.)", b, se,
                     cb.EDUC4_ETQ, "educ4", [1, 2, 3], N, G)
    filas += _filas("M5 M4 con BPN a término (únicos >=38 sem.)", b, se,
                     cb.REGIMEN_ETQ, "regimen", [2, 3], N, G)

    return pd.DataFrame(filas)


# C. RII y SII por educación (ridit por año: 0 = más educada, 1 = menos)

def ridit_educacion(celdas: pd.DataFrame) -> pd.DataFrame:
    d = celdas[celdas["educ4"] != 9]
    tot = (d.groupby(["anio", "educ4"], as_index=False)["n"].sum()
           .sort_values(["anio", "educ4"], ascending=[True, False]))
    tot["acum"] = tot.groupby("anio")["n"].cumsum()
    tot["ridit"] = (tot["acum"] - tot["n"] / 2) / tot.groupby("anio")["n"].transform("sum")
    return tot[["anio", "educ4", "ridit"]]


# "A término": 38 semanas y más en partos únicos, con T_GES (los 20 años)
RESULTADOS = {"BPN": ("casos", "n"),
              "BPN a término (únicos >=38 sem.)": ("casos_term", "n_term")}
PERIODOS = [(2005, 2024), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]


def rii_sii(celdas: pd.DataFrame, a0: int, a1: int, resultado: str) -> pd.DataFrame:
    """RII: Poisson de casos frente al ridit (exposición n, cluster dpto).
    SII: regresión lineal ponderada por n de la prevalencia de la celda, en
    puntos porcentuales. Mismo ajuste que el modelo principal (M4): edad, año,
    altitud del municipio y efectos fijos de departamento."""
    y, n = RESULTADOS[resultado]
    d = celdas[celdas["anio"].between(a0, a1) & (celdas["alt_cat"] != 9)
               & (celdas[n] > 0)]
    X = sm.add_constant(pd.concat([d["ridit"].rename("ridit"),
                                   _dummies(d, "edad6", 2), _dummies(d, "anio", a0),
                                   _dummies(d, "alt_cat", 1),
                                   _dummies(d, "dpto_res", 5)], axis=1).astype(float),
                        has_constant="add")
    grupos = {"groups": d["dpto_res"].to_numpy(),
              "use_correction": True, "df_correction": True}
    filas = []
    for titulo, aj in [
        ("RII", sm.GLM(d[y].astype(float), X, family=sm.families.Poisson(),
                       offset=np.log(d[n].astype(float))).fit(cov_type="cluster",
                                                              cov_kwds=grupos)),
        ("SII (pp)", sm.WLS((d[y] / d[n]).astype(float), X,
                            weights=d[n].astype(float)).fit(cov_type="cluster",
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
        filas.append({"resultado": resultado, "periodo": f"{a0}-{a1}",
                      "indice": titulo, **fuera, "p": p})
    return pd.DataFrame(filas)


# La educación falta más en los departamentos más pobres (no al azar). Cotas
# extremas: se asigna a todas las madres sin dato el nivel más bajo o el más alto.

FALTANTES = {"Excluidos (principal)": None, "Como nivel más bajo": 1,
             "Como nivel más alto": 4}


def rii_sii_cotas(c: pd.DataFrame) -> pd.DataFrame:
    """RII y SII por resultado y periodo, con las tres formas de tratar la
    educación faltante de FALTANTES (el ridit se recalcula en cada una)."""
    partes = []
    for etq, nivel in FALTANTES.items():
        k = c if nivel is None else c.assign(educ4=c["educ4"].replace(9, nivel))
        d = k.merge(ridit_educacion(k), on=["anio", "educ4"])
        t = pd.concat([rii_sii(d, a0, a1, res) for res in RESULTADOS
                       for a0, a1 in PERIODOS], ignore_index=True)
        t.insert(0, "faltantes", etq)
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def educacion_faltante(c: pd.DataFrame) -> pd.DataFrame:
    """% de nacidos (con peso y edad conocidos) sin dato de educación de la
    madre, por departamento y periodo; la primera fila es el total nacional."""
    falt = c["n"].where(c["educ4"].eq(9), 0)
    cols, nac = {}, {}
    for a0, a1 in PERIODOS:
        m = c["anio"].between(a0, a1)
        cols[f"{a0}-{a1}"] = (100 * falt[m].groupby(c["dpto_res"]).sum()
                              / c["n"][m].groupby(c["dpto_res"]).sum())
        nac[f"{a0}-{a1}"] = 100 * falt[m].sum() / c["n"][m].sum()
    d = pd.DataFrame(cols).sort_values("2020-2024", ascending=False).reset_index()
    d.insert(1, "departamento", d["dpto_res"].map(cb.DPTO_NOM))
    return pd.concat([pd.DataFrame([{"dpto_res": 0, "departamento": "Colombia", **nac}]),
                      d], ignore_index=True)


# D. Sensibilidad del gradiente educativo

TERMINO = "BPN a término (únicos >=38 sem.)"
EXTRA = {"paridad3": (cb.PARIDAD3_ETQ, 1), "etnia3": (cb.ETNIA3_ETQ, 3),
         "area2": (cb.AREA2_ETQ, 1)}          # variable: (etiquetas, referencia)


def rp_sensibilidad(celdas: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Razones de prevalencia por educación con el ajuste de M4 (edad, año,
    altitud y efectos fijos de departamento), 2008-2024, para BPN y BPN a
    término:
      M4s  M4 sin régimen, en la misma muestra que M4. El régimen puede estar
           en el camino entre la educación y el BPN: que sea mediador es un
           supuesto, no algo que estos modelos midan.
      M4f  M4s con la educación faltante como una categoría más.
      M6a  M4s en la muestra con paridad, etnia y área conocidas (la etnia
           falta en 2005-2007, 2012 y 2013).
      M6b  M6a + paridad + etnia + área."""
    def ajuste(d):
        return [_dummies(d, "edad6", 2), _dummies(d, "anio", int(d["anio"].min())),
                _dummies(d, "alt_cat", 1), _dummies(d, "dpto_res", 5)]

    filas = []

    def modelo(nombre, d, niveles=(1, 2, 3), extra=()):
        X = pd.concat([_dummies(d, "educ4", 4), *ajuste(d),
                       *[_dummies(d, v, EXTRA[v][1]) for v in extra]], axis=1)
        for resultado, (y, n) in RESULTADOS.items():
            k = d[d[n] > 0]
            b, se, N, G = _ajustar(k, y, X, n)
            nuevas = _filas(nombre, b, se, cb.EDUC4_ETQ, "educ4", niveles, N, G)
            for v in extra:
                etqs = EXTRA[v][0]
                nuevas += _filas(nombre, b, se, etqs, v,
                                 [j for j in etqs if j not in (EXTRA[v][1], 9)], N, G)
            filas.extend({"resultado": resultado, **f} for f in nuevas)

    d = c[(c["anio"] >= 2008) & (c["regimen"] != 9) & (c["alt_cat"] != 9)]
    modelo("M4s M4 sin régimen", d[d["educ4"] != 9])
    modelo("M4f M4s con educación faltante como categoría", d, niveles=(1, 2, 3, 9))
    cx = celdas_obj3(celdas, tuple(EXTRA))
    dx = cx[(cx["anio"] >= 2008) & (cx["regimen"] != 9) & (cx["alt_cat"] != 9)
            & (cx["educ4"] != 9) & cx[list(EXTRA)].ne(9).all(axis=1)]
    modelo("M6a M4s en la muestra con paridad, etnia y área", dx)
    modelo("M6b M6a + paridad + etnia + área", dx, extra=tuple(EXTRA))
    return pd.DataFrame(filas)


def completitud_anual(celdas: pd.DataFrame) -> pd.DataFrame:
    """% de nacidos (dpto válido y con peso) sin dato de cada variable, por año."""
    d = celdas[celdas["dpto_res"].gt(0) & celdas["con_peso"].gt(0)]
    total = d.groupby("anio")["con_peso"].sum()
    return pd.DataFrame({
        f"pct_sin_{nombre}": 100 * d["con_peso"].where(d[col].isna(), 0.0)
                             .groupby(d["anio"]).sum() / total
        for col, nombre in [("edad6", "edad"), ("educ4", "educacion"),
                            ("regimen", "regimen"), ("paridad3", "paridad"),
                            ("etnia3", "etnia"), ("area2", "area")]}).reset_index()


# E. Tendencia del RII

def _p_wild(d: pd.DataFrame, y: str, n: str, X: pd.DataFrame, nulos: list[str],
            reps: int = REPS_WILD) -> float:
    """Valor p de H0: los coeficientes `nulos` son cero, por bootstrap wild de
    los puntajes agrupados por departamento con la nula impuesta (Kline y
    Santos 2012): se ajusta el modelo sin esos términos y, en cada réplica, se
    cambia al azar el signo del puntaje de cada departamento (Rademacher) y se
    recalcula el estadístico de puntaje. Con 33 departamentos, la prueba de
    Wald con la covarianza agrupada rechaza de más."""
    X = sm.add_constant(X.astype(float), has_constant="add")
    yv, off = d[y].to_numpy(float), np.log(d[n].to_numpy(float))
    mu = sm.GLM(yv, X.drop(columns=nulos), family=sm.families.Poisson(),
                offset=off).fit().mu
    Xa = X.to_numpy()
    puntaje = pd.DataFrame(Xa * (yv - mu)[:, None]).groupby(
        d["dpto_res"].to_numpy()).sum().to_numpy()             # G x k
    B = np.linalg.inv(Xa.T @ (Xa * mu[:, None]))[[X.columns.get_loc(v) for v in nulos]]
    Vinv = np.linalg.inv(B @ puntaje.T @ puntaje @ B.T)

    def estadistico(total):                                    # (réplicas, k)
        b = total @ B.T
        return np.einsum("ri,ij,rj->r", b, Vinv, b)

    signos = np.random.default_rng(SEMILLA).choice([-1.0, 1.0],
                                                   size=(reps, len(puntaje)))
    obs = estadistico(puntaje.sum(0)[None, :])[0]
    return (1 + (estadistico(signos @ puntaje) >= obs).sum()) / (reps + 1)


# Los dos resultados del RII y cuatro controles con la semana 37 (solo años
# con T_GES_AGRU_CIE): si la ampliación a término se debiera a quién sale del
# denominador de 38 semanas y más, no aparecería al contar desde la semana 37.
RESULTADOS_TEND = {
    "BPN": ("casos", "n", "Peso < 2.500 g entre los nacidos con peso; 2005-2024"),
    TERMINO: ("casos_term", "n_term",
              "Peso < 2.500 g en partos únicos de 38 semanas y más (T_GES); 2005-2024"),
    "BPN a término, años con T_GES_AGRU_CIE": (
        "casos_term_cie", "n_term_cie",
        "Igual que el anterior, solo en los años en que se distingue la semana 37"),
    "Nacer en la semana 37 (únicos)": (
        "n_sem37", "n_gest", "Semana 37 entre los partos únicos con T_GES_AGRU_CIE"),
    "BPN dentro de la semana 37 (únicos)": (
        "casos_sem37", "n_sem37", "Peso < 2.500 g en partos únicos de la semana 37"),
    "BPN en únicos de 37 semanas y más": (
        "casos_term37", "n_term37",
        "Peso < 2.500 g en partos únicos de 37 semanas y más (T_GES_AGRU_CIE)"),
}


def rii_tendencia(c: pd.DataFrame) -> pd.DataFrame:
    """¿Se amplía el gradiente educativo? Por resultado, dos Poisson con el
    ajuste del RII (edad, año, altitud, efectos fijos de departamento):
    ridit x quinquenio (RII del primer quinquenio y razón de cada uno frente a
    él, con prueba conjunta) y ridit x década (cuánto se multiplica el RII
    cada diez años). IC 95 % con la covarianza agrupada y t de G-1 grados de
    libertad; valores p por bootstrap wild (_p_wild)."""
    from scipy.stats import t as t_student
    d0 = c[c["educ4"] != 9].merge(ridit_educacion(c), on=["anio", "educ4"])
    filas = []
    for resultado, (y, n, definicion) in RESULTADOS_TEND.items():
        d = d0[(d0["alt_cat"] != 9) & (d0[n] > 0)]
        a0 = int(d["anio"].min())
        base = [_dummies(d, "edad6", 2), _dummies(d, "anio", a0),
                _dummies(d, "alt_cat", 1), _dummies(d, "dpto_res", 5)]
        quinq = 1 + (d["anio"] - 2005) // 5
        q0 = int(quinq.min())
        inter = pd.DataFrame({f"ridit_q{q}": d["ridit"] * (quinq == q)
                              for q in sorted(quinq.unique()) if q != q0})
        decada = (d["ridit"] * (d["anio"] - a0) / 10).rename("ridit_decada")
        tc = t_student.ppf(0.975, d["dpto_res"].nunique() - 1)
        anios = f"{a0}-{int(d['anio'].max())} ({d['anio'].nunique()} años)"

        def fila(termino, b, se, p):
            filas.append({"resultado": resultado, "definicion": definicion,
                          "anios": anios, "termino": termino, "valor": np.exp(b),
                          "li": np.exp(b - tc * se), "ls": np.exp(b + tc * se),
                          "p_wild": p})

        Xq = pd.concat([d["ridit"], inter, *base], axis=1)
        b, se, _, _ = _ajustar(d, y, Xq, n)
        etq = {q: f"{2000 + 5 * q}-{2004 + 5 * q}" for q in range(1, 5)}
        fila(f"RII {etq[q0]}", b["ridit"], se["ridit"], np.nan)
        p_conj = _p_wild(d, y, n, Xq, list(inter.columns))
        for col in inter.columns:
            fila(f"RII {etq[int(col[-1])]} / RII {etq[q0]}", b[col], se[col], p_conj)
        Xd = pd.concat([d["ridit"], decada, *base], axis=1)
        b, se, _, _ = _ajustar(d, y, Xd, n)
        fila("Cambio del RII por década (razón)", b["ridit_decada"], se["ridit_decada"],
             _p_wild(d, y, n, Xd, ["ridit_decada"]))
    return pd.DataFrame(filas)


# G9/G10. Prevalencia estandarizada por año según educ4 y según régimen

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


# Interfaz del módulo: calcular -> imprimir / graficos (ver correr.py)

LOG = "log_03_objetivo3"


def calcular(celdas: pd.DataFrame) -> dict:
    """Tablas del objetivo 3; las claves con "_" son intermedias (no van al xlsx)."""
    c = celdas_obj3(celdas)
    pesos = cb.pesos_edad(c)   # población estándar: con peso y edad conocidos
    return {
        "Prevalencias": prevalencias(c, pesos),
        "RP_modelos": razones_prevalencia(c),
        "RII_SII_educacion": rii_sii_cotas(c),
        "Educacion_faltante": educacion_faltante(c),
        "RP_sensibilidad": rp_sensibilidad(celdas, c),
        "RII_tendencia": rii_tendencia(c),
        "Completitud_anual": completitud_anual(celdas),
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
    print("\n--- C. RII y SII por educación (y cotas por educación faltante) ---")
    print(r["RII_SII_educacion"].round(3).to_string(index=False))
    print("\n--- D. % sin dato de educación (Colombia y 8 dptos con más faltante) ---")
    print(r["Educacion_faltante"].head(9).round(1).to_string(index=False))
    print("\n--- E. Sensibilidad del gradiente (sin régimen, faltantes, covariables) ---")
    print(r["RP_sensibilidad"][["resultado", "modelo", "variable", "categoria", "rp",
                                "li", "ls", "n_obs"]].round(3).to_string(index=False))
    print("\n--- F. Tendencia del RII (p por bootstrap wild; conjunta en las razones) ---")
    print(r["RII_tendencia"].drop(columns="definicion").round(4).to_string(index=False))
    print("\n--- G. % sin dato por año ---")
    print(r["Completitud_anual"].round(2).to_string(index=False))


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
