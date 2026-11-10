"""Corre los objetivos y guarda sus salidas (log, xlsx y png) en cb.SALIDAS.

    python correr.py          # los 4 objetivos, en orden
    python correr.py 2 4      # solo esos objetivos

Cada módulo objN_* expone calcular(celdas) -> dict de tablas, imprimir(r)
y graficos(r) -> dict de figuras; la notebook sintesis.ipynb usa lo mismo.
"""
import sys

import matplotlib.pyplot as plt

import comun_bpn as cb
import obj1_evolucion
import obj2_desigualdades
import obj3_educacion
import obj4_concentracion

OBJETIVOS = {1: obj1_evolucion, 2: obj2_desigualdades,
             3: obj3_educacion, 4: obj4_concentracion}


def correr(numeros) -> None:
    celdas = None
    for k in numeros:
        mod = OBJETIVOS[k]
        with cb.log_salida(mod.LOG):
            if celdas is None:           # se cargan una vez para todos
                celdas = cb.cargar_celdas()
            r = mod.calcular(celdas)
            mod.imprimir(r)
            for nombre, fig in mod.graficos(r).items():
                cb.guardar_grafico(fig, nombre)
                plt.close(fig)
            # las claves con "_" son intermedias (para gráficos o el log)
            cb.exportar_excel_nuevo({h: t for h, t in r.items()
                                     if not h.startswith("_")}, mod.EXCEL)
            print(f"\nListo. Resultados en {cb.SALIDAS / mod.EXCEL}")


if __name__ == "__main__":
    correr([int(a) for a in sys.argv[1:]] or sorted(OBJETIVOS))
