# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:32:51 2024
Finalized on Tuesday Oct 22 19:33:40 2024
Este programa calcula los coeficientes de rigidez de una trampa óptica por el método
de Boltzmann o del Análisis del Potencial (https://doi.org/10.1364/AOP.394888)
Se recibe un connjunto de archivos de texto en formato *.txt con dos columnas, una para x y otra para y
Calcula la distribución con una nueva particiónn de n-bins y de aquí se obtiene el potencial U(x)
Si se grafica U(x)vs x, tiene una forma parabólica y siguiendo la relación
U(x)=1/2*k*x^2, podemos realizar un ajuste polinomial para obtener el polinomio:
    U(x)=\alpha*x^2+\beta*x+\gamma
de manera que se puede obtener:
    k=2*\alpha
Este programa también calcula los coeficientes de rigidez de la trampa por el método de Equipartición, donde:
        
            k_\alpha=(k_B*T)/var(\alpha)
            
por lo tanto, obtenemos: "k_x" y "k_y" y guarda todos estos datos en un vector de k's'
@author: Josué Hernández Torres
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tkinter import Tk, filedialog
from pathlib import Path

# Desactivar el modo interactivo de matplotlib
plt.ioff()

# Crear una ventana de tkinter para seleccionar la carpeta de datos
root = Tk()
root.withdraw()  # Ocultar la ventana principal
ruta_datos = filedialog.askdirectory(title="Selecciona la carpeta de datos")

# Verificar que se seleccionó una carpeta
if not ruta_datos:
    print("No se seleccionó ninguna carpeta.")
    exit()

# Convertir la ruta a un objeto Path
ruta_datos = Path(ruta_datos)

# Crear carpetas de salida dentro de la carpeta de datos seleccionada
carpeta_posiciones = ruta_datos / "Posiciones"
carpeta_distribuciones = ruta_datos / "Distribuciones"
carpeta_potenciales = ruta_datos / "Potenciales"
for carpeta in [carpeta_posiciones, carpeta_distribuciones, carpeta_potenciales]:
    carpeta.mkdir(exist_ok=True)

# Constantes físicas y número de bins:
k_B = 1.380649e-23
T = 295.15
C = k_B * T
bins = 30
N = 4  # Número de experimentos para las k's promedio

# Vectores para almacenar resultados
kx_Boltzmann = []
ky_Boltzmann = []
kx_Equiparticion = []
ky_Equiparticion = []
Fx_B = []
Fy_B = []
Fx_Eq = []
Fy_Eq = []

# Funciones auxiliares
def graficar_potencial(px, py, x, y, Particion_X, Particion_Y, U_x, U_y, ruta_guardado=None):
    x_min = -px[1] / (2 * px[0])
    y_min = -py[1] / (2 * py[0])
    # Se modifica la Partición con los nuevos valores
    Particion_X = [x - x_min for x in Particion_X]
    Particion_Y = [y - y_min for y in Particion_Y]
    # Se vuelven a ajustar las parábolas para estos nuevos valores
    px = np.polyfit(Particion_X, U_x, 2)
    py = np.polyfit(Particion_Y, U_y, 2)
    potx = [px[0] * x**2 + px[1] * x + px[2] for x in Particion_X]
    poty = [py[0] * y**2 + py[1] * y + py[2] for y in Particion_Y]
    # Conversión para gráfica
    U_x = np.array(U_x) / 1e-20
    potx = np.array(potx) / 1e-20
    U_y = np.array(U_y) / 1e-20
    poty = np.array(poty) / 1e-20
    Particion_X = np.array(Particion_X) / 1e-8
    Particion_Y = np.array(Particion_Y) / 1e-8
    # Se grafican 2 subgráficas, izquierda para U(x)vsx y derecha para U(y)vsy
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Potenciales para $x$ y $y$')
    ax1.plot(Particion_X, U_x, '.', color='black')
    ax1.plot(Particion_X, potx, '-.', linewidth=0.5, label='kx={}'.format(2 * px[0]))
    ax2.plot(Particion_Y, U_y, '.', color='red')
    ax2.plot(Particion_Y, poty, '-.', linewidth=0.5, label='ky={}'.format(2 * py[0]))
    ax1.set_title('Potencial $U_x$')
    ax2.set_title('Potencial $U_y$')
    ax1.set(xlabel='Posición ($10^{-8}m$)', ylabel='$U_x(k_BT)$')
    ax2.set(xlabel='Posición ($10^{-8}m$)', ylabel='$U_y(k_BT)$')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    plt.tight_layout()

    # Guardar la figura si se proporciona una ruta de guardado
    if ruta_guardado:
        plt.savefig(ruta_guardado)
        plt.close()  # Cerrar la figura para liberar memoria
    else:
        plt.show()  # Mostrar la figura si no se proporciona una ruta de guardado

def graficar_posiciones(x, y, ruta_guardado=None):
    tx = np.linspace(0, len(x), len(x))
    ty = np.linspace(0, len(y), len(y))
    fig, (g1, g2) = plt.subplots(2, 1)
    fig.suptitle('Posiciones en el tiempo para $x$ y $y$')
    g1.vlines(tx, 0, x, color='r', alpha=0.6, label='X')  # Rojo para X
    g2.vlines(ty, 0, y, color='k', alpha=0.6, label='Y')  # Negro para Y
    g1.set_title('Posiciones $x$')
    g2.set_title('Posiciones $y$')
    g1.set(xlabel='$t$ (a.u.)', ylabel='$x$')
    g2.set(xlabel='$t$', ylabel='$y$')
    g1.legend(loc='best')
    g2.legend(loc='best')
    g1.grid(True, linestyle="--", alpha=0.5)
    g2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Guardar la figura si se proporciona una ruta de guardado
    if ruta_guardado:
        plt.savefig(ruta_guardado)
        plt.close()  # Cerrar la figura para liberar memoria
    else:
        plt.show()  # Mostrar la figura si no se proporciona una ruta de guardado

def graficar_distribucion(Particion_X, X_Frec, Particion_Y, Y_Frec, ruta_guardado=None):
    # Crear la figura y las subgráficas
    fig, (g1, g2) = plt.subplots(2, 1)
    fig.suptitle('Distribución de posiciones $x$ y $y$')

    # Gráfica para X
    g1.plot(Particion_X, X_Frec, marker='o', linestyle='-', color='r', alpha=0.6, label='X')  # Rojo para X
    g1.set_title('Posiciones $x$')
    g1.set(xlabel='$x$', ylabel='$count$')
    g1.legend(loc='best')
    g1.grid(True, linestyle="--", alpha=0.5)

    # Gráfica para Y
    g2.plot(Particion_Y, Y_Frec, marker='o', linestyle='-', color='k', alpha=0.6, label='Y')  # Negro para Y
    g2.set_title('Posiciones $y$')
    g2.set(xlabel='$y$', ylabel='$count$')
    g2.legend(loc='best')
    g2.grid(True, linestyle="--", alpha=0.5)

    # Ajustar el layout para que no se solapen las subgráficas
    plt.tight_layout()

    # Guardar la figura si se proporciona una ruta de guardado
    if ruta_guardado:
        plt.savefig(ruta_guardado)  # Guardar la figura completa
        plt.close()  # Cerrar la figura para liberar memoria
    else:
        plt.show()  # Mostrar la figura si no se proporciona una ruta de guardado

def filtro(Frecuencias, Particion):
    a = [i for i, f in enumerate(Frecuencias) if f == 0.0]
    for i in reversed(a):
        Frecuencias.pop(i)
        Particion.pop(i)
    return Frecuencias, Particion

def fuerzas(k, alpha):
    # Siguiendo la relación F_{\alpha}=-k_{\alpha}*\alpha, tomaremos \alpha=mean(\alpha)
    F = -k * np.mean(alpha)
    return F

# Obtener la lista de archivos en la carpeta seleccionada
lista_archivos = list(ruta_datos.glob("*.txt"))

# Procesar cada archivo
for na in lista_archivos:
    # Extraer el nombre del archivo sin la ruta ni la extensión
    nombre_archivo = na.stem

    # Leer el archivo *.txt
    try:
        x, y = np.loadtxt(na, usecols=(1, 3), unpack=True, skiprows=1)
    except Exception as e:
        print(f"Error al leer el archivo {na}: {e}")
        continue

    # Convertir a metros
    x = x * 1e-6
    y = y * 1e-6

    # Calcular varianzas y rigideces por el método de Equipartición
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    kx_Equiparticion.append(C / sigma_x)
    ky_Equiparticion.append(C / sigma_y)
    Fx_Eq.append(fuerzas(C / sigma_x, x))
    Fy_Eq.append(fuerzas(C / sigma_y, y))

    # Graficar y guardar posiciones
    ruta_posiciones = carpeta_posiciones / f"posiciones_{nombre_archivo}.jpg"
    graficar_posiciones(x, y, ruta_posiciones)

    # Ordenar los datos
    x.sort()
    y.sort()

    # Crear la partición para los histogramas
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    dx = (xmax - xmin) / bins
    dy = (ymax - ymin) / bins
    Particion_X = [xmin + i * dx for i in range(bins + 1)]
    Particion_Y = [ymin + i * dy for i in range(bins + 1)]

    # Calcular frecuencias
    X_Frec = [sum((Particion_X[i] - dx / 2 < x) & (x <= Particion_X[i] + dx / 2)) for i in range(bins + 1)]
    Y_Frec = [sum((Particion_Y[i] - dy / 2 < y) & (y <= Particion_Y[i] + dy / 2)) for i in range(bins + 1)]

    # Graficar y guardar distribución de posiciones
    ruta_distribuciones = carpeta_distribuciones / f"distribucion_posiciones_{nombre_archivo}.jpg"
    graficar_distribucion(Particion_X, X_Frec, Particion_Y, Y_Frec, ruta_distribuciones)

    # Filtrar frecuencias iguales a 0
    X_Frec, Particion_X = filtro(X_Frec, Particion_X)
    Y_Frec, Particion_Y = filtro(Y_Frec, Particion_Y)

    # Calcular potenciales
    U_x = [-k_B * T * np.log(f) for f in X_Frec]
    U_y = [-k_B * T * np.log(f) for f in Y_Frec]

    # Ajustar parábolas y calcular rigideces por el método de Boltzmann
    px = np.polyfit(Particion_X, U_x, 2)
    py = np.polyfit(Particion_Y, U_y, 2)
    kx_Boltzmann.append(2 * px[0])
    ky_Boltzmann.append(2 * py[0])
    Fx_B.append(fuerzas(2 * px[0], x))
    Fy_B.append(fuerzas(2 * py[0], y))

    # Graficar y guardar potenciales
    ruta_potenciales = carpeta_potenciales / f"potenciales_{nombre_archivo}.jpg"
    graficar_potencial(px, py, x, y, Particion_X, Particion_Y, U_x, U_y, ruta_potenciales)

# Calcular promedios de las k's cada N experimentos
kx_B_mean = [sum(kx_Boltzmann[i:i + N]) / N for i in range(0, len(kx_Boltzmann), N)]
ky_B_mean = [sum(ky_Boltzmann[i:i + N]) / N for i in range(0, len(ky_Boltzmann), N)]
kx_Eq_mean = [sum(kx_Equiparticion[i:i + N]) / N for i in range(0, len(kx_Equiparticion), N)]
ky_Eq_mean = [sum(ky_Equiparticion[i:i + N]) / N for i in range(0, len(ky_Equiparticion), N)]

# Guardar resultados en un archivo Excel
ruta_excel = ruta_datos / "Stiffness_Analysis.xlsx"
with pd.ExcelWriter(ruta_excel) as writer:
    df = pd.DataFrame({"kx_B": kx_Boltzmann, "ky_B": ky_Boltzmann, "kx_Eq": kx_Equiparticion, "ky_Eq": ky_Equiparticion})
    df_2 = pd.DataFrame({"kx_B_Mean": kx_B_mean, "ky_B_Mean": ky_B_mean, "kx_Eq_Mean": kx_Eq_mean, "ky_Eq_Mean": ky_Eq_mean})
    df.to_excel(writer, sheet_name="Stiffness_Completas", index=False)
    df_2.to_excel(writer, sheet_name="Stiffness_Promedios", index=False)

print("¡Todas las imágenes han sido generadas y guardadas con éxito!")
