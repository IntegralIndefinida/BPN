#  Pruebas rápidas de la lógica central, sin pytest ni datos reales:
#
#    ~/.venvs/datascience/bin/python test_bpn.py
#
#  1. ic_vec (fórmula cerrada del bootstrap) == ic_conc (regresión conveniente)
#  2. _peso_grupo: rama de grupos y rama de gramos
#  3. _armonizar: definición de BPN y BPN a término sobre datos sintéticos
#  4. _p_wild: el bootstrap wild distingue un efecto real de ninguno

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import comun_bpn as cb
from obj4_concentracion import ic_conc, ic_vec


def test_ic_vec_igual_ic_conc():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = rng.integers(5, 40)
        h = pd.Series(rng.uniform(1, 30, n))
        w = pd.Series(rng.uniform(1, 100, n))
        rango = pd.Series(rng.uniform(0, 100, n))
        c = ic_conc(h, w, rango)["C"]
        o = np.argsort(rango.to_numpy(), kind="stable")
        assert np.isclose(c, ic_vec(h.to_numpy(float), w.to_numpy(float),
                                    rango.to_numpy(float))), \
            "ic_vec != ic_conc con rango"
        assert np.isclose(c, ic_vec(h.to_numpy(float)[o], w.to_numpy(float)[o])), \
            "ic_vec != ic_conc preordenado (como bootstrap_ic)"


def test_peso_grupo():
    grupos = pd.Series([1, 4, 5, 8, 9, np.nan], dtype="float")
    g = cb._peso_grupo(grupos)
    assert list(g) == [1, 4, 5, 8, pd.NA, pd.NA]
    gramos = pd.Series([1500, 2500, 150, 7500, np.nan])
    g = cb._peso_grupo(gramos)
    assert list(g) == [3, 5, pd.NA, pd.NA, pd.NA]


def test_armonizar():
    # 6 nacimientos sintéticos; anio=2008 para que el régimen se mapee
    d = pd.DataFrame({
        "PESO_NAC": [1, 1, 1, 5, 9, 8],
        "gestacion_grupo": ["De 38 a 41", "De 38 a 41", "De 28 a 37",
                            "De 38 a 41", "Sin información", "Sin información"],
        "parto_multiple": [False, True, False, False, pd.NA, True],
        "edad_madre": ["De 20 a 24 años", "De 20 a 24 años", "De 25 a 29 años",
                       "De 30 a 34 años", "De 15 a 19 años", "De 40 a 44 años"],
        "nivel_educativo_madre": ["Primaria", "Primaria", "Primaria",
                                  "Secundaria y media", "Ninguno", "Superior"],
        "regimen_salud": ["Contributivo", "Contributivo", "Subsidiado",
                          "Contributivo", "Sin información", "No asegurado"],
        "dpto_residencia": [5, 5, 5, 5, 5, 5],
        "mpio_residencia": [5001, 5001, 5001, 5001, 5001, 5001],
        # semana 37 = T_GES 28-37 que T_GES_AGRU_CIE no marca como <37 (fila 3)
        "prematuro": pd.array([False, False, False, False, pd.NA, pd.NA],
                              dtype="boolean"),
        "hijos_nacidos_vivos": [1, 1, 2, 5, np.nan, 1],
        "etnia": ["Ninguna", "Ninguna", "Indígena", "Raizal", "Rom", "Sin información"],
        "area_residencia": ["Cabecera", "Cabecera", "Rural disperso",
                            "Centro poblado", "Cabecera", "Cabecera"],
    })
    alt = "cod_mpio,altitud_m\n5001,1500\n"
    with TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "alt.csv"
        ruta.write_text(alt)
        original_load, original_alt = cb.dd.eevv.load, cb.ALTITUD_CSV
        cb.dd.eevv.load = lambda anio, **k: d
        cb.ALTITUD_CSV = ruta
        try:
            celdas = cb._armonizar(2008)
        finally:
            cb.dd.eevv.load = original_load
            cb.ALTITUD_CSV = original_alt
    #  las celdas se parten por (educ4, regimen, edad6, paridad3, etnia3,
    #  area2): las filas 1 y 2 coinciden en todo, 5 combinaciones
    assert len(celdas) == 5
    assert celdas["n"].sum() == 6                    # todos los nacidos
    assert celdas["con_peso"].sum() == 5             # fuera el peso 9
    assert celdas["bpn_n"].sum() == 3                # grupos 1-4
    #  término = 38+ semanas y parto único, con peso: filas 1 y 4
    assert celdas["n_term"].sum() == 2
    assert celdas["bpn_term_n"].sum() == 1
    #  únicos con peso y T_GES_AGRU_CIE conocida: filas 1, 3 y 4; semana 37: la 3
    assert celdas["n_gest_u"].sum() == 3
    assert celdas["n_sem37"].sum() == 1 and celdas["bpn_sem37"].sum() == 1
    assert celdas["paridad3"].eq(3).sum() == 1               # 5 hijos -> 3 o más
    assert celdas["etnia3"].isna().sum() == 2                # Rom y sin información
    assert sorted(celdas["area2"].dropna().unique()) == [1, 2]
    assert celdas["reg_conocido"].sum() == 5         # fuera "Sin información"
    assert celdas["n_nocontrib"].sum() == 2          # subsidiado + no asegurado
    assert celdas["n_sub"].sum() == 1
    assert celdas["altitud_m"].eq(1500).all()


def test_p_wild():
    """Con un efecto real el valor p es chico; sin efecto, no."""
    from obj3_educacion import _p_wild
    rng = np.random.default_rng(3)
    G, m = 30, 60
    d = pd.DataFrame({"dpto_res": np.repeat(np.arange(G), m),
                      "x": rng.normal(size=G * m), "z": rng.normal(size=G * m),
                      "n": 200.0})
    efecto_dpto = np.repeat(rng.normal(0, 0.2, G), m)
    for beta, esperado in [(0.0, lambda p: p > 0.05), (0.3, lambda p: p < 0.01)]:
        mu = d["n"] * np.exp(-2 + 0.2 * d["x"] + beta * d["z"] + efecto_dpto)
        d["y"] = rng.poisson(mu)
        p = _p_wild(d, "y", "n", d[["x", "z"]], ["z"], reps=999)
        assert esperado(p), f"p wild = {p} con beta = {beta}"


if __name__ == "__main__":
    test_ic_vec_igual_ic_conc()
    print("ok: ic_vec == ic_conc")
    test_peso_grupo()
    print("ok: _peso_grupo (grupos y gramos)")
    test_armonizar()
    print("ok: _armonizar (BPN, término, semana 37, régimen, altitud, llaves)")
    test_p_wild()
    print("ok: _p_wild (bootstrap wild por departamento)")
    print("Todas las pruebas pasaron.")
