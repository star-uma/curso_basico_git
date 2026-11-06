"""
Prueba 01 - Trayectoria de un robot diferencial
================================================

El robot recibe una lista de órdenes (velocidad lineal, velocidad angular,
duración) y se dibuja el camino que recorre. Con el programa por defecto
dibuja una estrella (¡STAR!).

Cómo ejecutarlo:
    - En VS Code: abre este archivo y pulsa el botón ▶ (Run Python File).
    - O desde una terminal en la carpeta del repositorio:  py pruebas/01_trayectoria/trayectoria.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # para importar 'star'

# Importar 'star' primero: comprueba que numpy y matplotlib están instalados
from star import config, estilo_star, guardar_figura  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

# =====================================================================
#  PARÁMETROS  ->  ¡Modifica esto!
# =====================================================================

# Cada orden es: (velocidad lineal [m/s], velocidad angular [rad/s], duración [s])
#   - w = 0          -> avanza en línea recta
#   - v = 0          -> gira sobre sí mismo
#   - v != 0, w != 0 -> describe un arco
LADO = 2.0                   # longitud de cada punta de la estrella [m]
GIRO_ESTRELLA = 0.8 * np.pi  # 144 grados

PROGRAMA = [
    (1.0, 0.0, LADO), (0.0, np.pi, GIRO_ESTRELLA / np.pi),
    (1.0, 0.0, LADO), (0.0, np.pi, GIRO_ESTRELLA / np.pi),
    (1.0, 0.0, LADO), (0.0, np.pi, GIRO_ESTRELLA / np.pi)
]

POSICION_INICIAL = (0.0, 0.0, 0.0)  # x [m], y [m], orientación [rad]
DT = 0.02                           # paso de simulación [s]

# =====================================================================


def simular(programa, inicio, dt):
    """Integra el modelo cinemático del robot diferencial (uniciclo)."""
    x, y, th = inicio
    xs, ys, ts, ths = [x], [y], [0.0], [th]
    t = 0.0
    cambios = [(x, y)]  # puntos donde empieza cada orden

    for v, w, duracion in programa:
        pasos = max(1, int(round(duracion / dt)))
        for _ in range(pasos):
            x += v * np.cos(th) * dt
            y += v * np.sin(th) * dt
            th += w * dt
            t += dt
            xs.append(x); ys.append(y); ts.append(t); ths.append(th)
        cambios.append((x, y))

    return np.array(xs), np.array(ys), np.array(ts), np.array(ths), cambios


def dibujar_robot(ax, x, y, th, tam=0.35, color=None):
    """Dibuja el robot como un triángulo que apunta hacia donde mira."""
    color = color or config.COLOR_SECUNDARIO
    puntos = np.array([[tam, 0], [-tam * 0.6, tam * 0.5], [-tam * 0.6, -tam * 0.5]])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    puntos = puntos @ rot.T + [x, y]
    ax.add_patch(plt.Polygon(puntos, color=color, zorder=5, ec="black", lw=1))


def main():
    estilo_star()
    xs, ys, ts, ths, cambios = simular(PROGRAMA, POSICION_INICIAL, DT)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Trayectoria coloreada según el tiempo
    segmentos = np.stack([np.column_stack([xs[:-1], ys[:-1]]),
                          np.column_stack([xs[1:], ys[1:]])], axis=1)
    lineas = LineCollection(segmentos, cmap="viridis", linewidths=3)
    lineas.set_array(ts[:-1])
    ax.add_collection(lineas)
    barra = fig.colorbar(lineas, ax=ax, shrink=0.8)
    barra.set_label("tiempo [s]")

    # Puntos donde cambia la orden
    cx, cy = zip(*cambios)
    ax.scatter(cx, cy, s=25, color="white", ec="black", zorder=4, label="cambio de orden")

    ax.scatter([xs[0]], [ys[0]], s=120, marker="o", color=config.COLOR_ACENTO,
               ec="black", zorder=6, label="inicio")
    dibujar_robot(ax, xs[-1], ys[-1], ths[-1])

    distancia = np.sum(np.hypot(np.diff(xs), np.diff(ys)))
    ax.set_title(f"Trayectoria de {config.NOMBRE_ROBOT}\n"
                 f"{len(PROGRAMA)} órdenes · {distancia:.1f} m · {ts[-1]:.1f} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.12)
    ax.legend(loc="upper right")

    guardar_figura(fig, "01_trayectoria")


if __name__ == "__main__":
    main()
