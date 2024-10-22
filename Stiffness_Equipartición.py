# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:31:47 2024
Este programa se coloca en una carpeta con archivos de formato *.txt, recorre todos los archivos uno por uno, extrae las columnas de posiciones x y y
Las asigna a listas y de ahí se calculan los coeficientes de rigidez de la trampa por el método de equipartición, donde:
    
            k_\alpha=(k_B*T)/var(\alpha)
por lo tanto, obtenemos: "k_x" y "k_y" y guarda todos estos datos en un vector de k's'
@author: Josué Hernández Torres
"""
import numpy as np
import glob
k_B=1.380649e-23        #Constante de Boltzmann
T=295.15                #K =22°C
C=k_B*T
k_x=[]                  #Se crean los vectores para la rigidez en x y y
k_y=[]
lista_archivos=glob.glob('*.txt')  #Cargamos todos los archivos *.txt a la variable
for na in lista_archivos:       #Recorremos archivo por archivo con el nombre 'na' (nombre de archivo)
    c1,x,c3,y,c5,c6=np.loadtxt(na, unpack=True)     #Se asignan las columnas 0-->x y 1-->y
    x=list(x);y=list(y)
    x.pop(0);y.pop(0)
    x=np.array(x);y=np.array(y)
    x*=1e-6                             #Los datos están escritos en \mu m, así que hay que convertir a m
    y*=1e-6
    sigma_x=np.var(x)                   #Se calculan las varianzas de x y y
    sigma_y=np.var(y)
    # print('k_x=',k_B*T/sigma_x)
    # print('k_y=',k_B*T/sigma_y)
    k_x.append(C/sigma_x)               #Se calculan las rigideces por el método de Equipartición
    k_y.append(C/sigma_y)