# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:14:17 2024

@author: PC
"""

import pandas as pd
import matplotlib.pyplot as plt

Datos=pd.read_excel("Stiffness1.xlsx","Hoja 2",index_col=None, header=None) #Con esto importamos la tabla de datos del documento "Stiffness.xlsx" en su sheet "Hoja 2", se le dice que no hay columna de índices y que no hay encabezado, si la tabla de datos es diferente, hay que cambiar estos parámetros
kx_Eq_T = list(Datos[0])  #Coeficiente de rigidez para x; Equipartición; TweezPal
kx_Eq_J = list(Datos[1])  #Coeficiente de rigidez para x; Equipartición; Josué
ky_Eq_T = list(Datos[2])  #Coeficiente de rigidez para y; Equipartición; TweezPal
ky_Eq_J = list(Datos[3])  #Coeficiente de rigidez para y; Equipartición; Josué
kx_B_T  = list(Datos[4])  #Coeficiente de rigidez para x; Boltzmann; TweezPal
kx_B_J  = list(Datos[5])  #Coeficiente de rigidez para x; Boltzmann; Josué
ky_B_T  = list(Datos[6])  #Coeficiente de rigidez para y; Boltzmann; TweezPal
ky_B_J  = list(Datos[7])  #Coeficiente de rigidez para y; Boltzmann; Josué
t       = list(range(34)) #Se genera un arreglo de datos simples sólo para el número de filas, en este caso, de archivos que se está trabajando, es sólo para visualizar la evolución de las k's

#Creamos una figura con dos subplots, el subplot a la izquierda 'ax' para x y el subplot a la derecha 'ay' para y, (1,2)=1 fila dos columnas
fig, (ax, ay) = plt.subplots(1, 2)
fig.suptitle('Método de Equipartición') #Título para las dos gráficas, es decir, "en común"
#Ploteamos las dos para x y las dos para y
ax.plot(t,kx_Eq_T, '.',linestyle='--',color='red'   , label='kx-TweezPal')
ax.plot(t,kx_Eq_J, '+',linestyle='-',color='black'  , label='kx-Josué'   )
ay.plot(t,ky_Eq_T, '.',linestyle='--',color='red'   , label='ky-TweezPal')
ay.plot(t,ky_Eq_J, '+',linestyle='-',color='black'  , label='ky-Josué'   )
ax.set_title('Coeficientes para X')
ay.set_title('Coeficientes para Y')
ax.set(xlabel='secuencia', ylabel='$kx$')
ay.set(xlabel='secuencia', ylabel='$ky$')
ax.legend(loc='best')
ay.legend(loc='best')

#Se crea otra figura para imprimir, igual a una fila y dos columnas, para el otro método
fig, (ax, ay) = plt.subplots(1, 2)
fig.suptitle('Método de Boltzmann')

ax.plot(t,kx_B_T, '.',linestyle='--',color='red'   , label='kx-TweezPal')
ax.plot(t,kx_B_J, '+',linestyle='-',color='black'  , label='kx-Josué'   )
ay.plot(t,ky_B_T, '.',linestyle='--',color='red'   , label='ky-TweezPal')
ay.plot(t,ky_B_J, '+',linestyle='-',color='black'  , label='ky-Josué'   )
ax.set_title('Coeficientes para X')
ay.set_title('Coeficientes para Y')
ax.set(xlabel='secuencia', ylabel='$kx$')
ay.set(xlabel='secuencia', ylabel='$ky$')
ax.legend(loc='best')
ay.legend(loc='best')