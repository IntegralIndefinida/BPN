#  Pruebas rápidas de la lógica central, sin pytest ni datos reales:
#
#    ~/.venvs/datascience/bin/python test_bpn.py
#
#  1. ic_vec (fórmula cerrada del bootstrap) == ic_conc (regresión conveniente)
#  2. _peso_grupo: rama de grupos y rama de gramos
#  3. _armonizar: definición de BPN y BPN a término sobre datos sintéticos

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
    #  las celdas se parten por (educ4, regimen, edad6): 5 combinaciones
    assert len(celdas) == 5
    assert celdas["n"].sum() == 6                    # todos los nacidos
    assert celdas["con_peso"].sum() == 5             # fuera el peso 9
    assert celdas["bpn_n"].sum() == 3                # grupos 1-4
    #  término = 38+ semanas y parto único, con peso: filas 1 y 4
    assert celdas["n_term"].sum() == 2
    assert celdas["bpn_term_n"].sum() == 1
    assert celdas["reg_conocido"].sum() == 5         # fuera "Sin información"
    assert celdas["n_nocontrib"].sum() == 2          # subsidiado + no asegurado
    assert celdas["n_sub"].sum() == 1
    assert celdas["altitud_m"].eq(1500).all()


if __name__ == "__main__":
    test_ic_vec_igual_ic_conc()
    print("ok: ic_vec == ic_conc")
    test_peso_grupo()
    print("ok: _peso_grupo (grupos y gramos)")
    test_armonizar()
    print("ok: _armonizar (BPN, término, régimen, altitud)")
    print("Todas las pruebas pasaron.")
