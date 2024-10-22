# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:32:51 2024
Limpieza de columnas de tiempo y primera fila de 0's

Este programa elimina las columnas de tiempos (1ra, 3ra, 5ta y 6ta) de un archivo proveniente del guardado de datos de posición de las pinzas,
además elimina la primera fila, compuesta por 0's
@author: Josué Hernández Torres
"""
import glob   #Se importa la librería glob, para poder importar los archivos *.txt
lista_archivos=glob.glob('*.txt')  #Cargamos todos los archivos *.txt a la variable
for na in lista_archivos:       #Recorremos archivo por archivo con el nombre 'na' (nombre de archivo)
    cadena=''                   #La cadena es un espacio, para la reescritura
    aux=[]
    for renglon in open(na):    #Recorremos cada renglón de cada archivo na
        partes=renglon.split()  #Separamos las columnas
        del partes[5]           #Eliminamos las columnas 6 [5]
        del partes[4]           #Se elimina la 5  [4]
        del partes[2]           #Se elimina la 3  [2]
        del partes[0]           #Finalmente, se elimina la 1 [0]
        aux.append(partes)      #Las incluimos en el auxiliar
    del aux[0]                  #Borramos la primera fila que no nos sirve
    for i in range(len(aux)):   #Recorremos toda la matriz
        for j in range(2):
            cadena+=str(aux[i][j])+" "#A cada número le asignamos un espacio al final para separar valores
        cadena+='\n'           #Hacemos un cambio de línea
    print(cadena)               #Imprimimos en la terminal
    a=open(na,'w')              #Abrimos el archivo .txt
    a.write(cadena)             #Reescribimos
    a.close()                   #Cerramos




