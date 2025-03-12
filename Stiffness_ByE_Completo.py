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
#Constantes físicas y número de bins:
k_B=1.380649e-23
T=295.15
C=k_B*T
bins=30
N=4             #Número de experimentos para las k's promedio
#Se crean los vectores que van a contenter a los n valores de stiffness en x e y
kx_Boltzmann=[]
ky_Boltzmann=[]
kx_Equipartición=[]                  
ky_Equipartición=[]
Fx_B=[]
Fy_B=[]
Fx_Eq=[]
Fy_Eq=[]

#Definimos una función que grafique, por si se necesita en algún momento, sólo descomentar la línea 163
def graficar_potencial(px,py,x,y,Particion_X,Particion_Y,U_x,U_y):
    x_min = -px[1] / (2 * px[0])
    y_min = -py[1] / (2 * py[0])
    #Se modifica la Partición con los nuevos valores
    Particion_X = [x - x_min for x in Particion_X]
    Particion_Y = [y - y_min for y in Particion_Y]
    #Se vuelven a ajustar las parábolas para estos nuevos valores
    px=np.polyfit(Particion_X,U_x,2)
    py=np.polyfit(Particion_Y,U_y,2)
    potx = [px[0] * x**2 + px[1] * x + px[2] for x in Particion_X]
    poty = [py[0] * y**2 + py[1] * y + py[2] for y in Particion_Y]
    #Conversión para gráfica
    U_x=np.array(U_x)/1e-20
    potx=np.array(potx)/1e-20
    U_y=np.array(U_y)/1e-20
    poty=np.array(poty)/1e-20
    Particion_X=np.array(Particion_X)/1e-8
    Particion_Y=np.array(Particion_Y)/1e-8
    #Se grafican 2 subgráficas, izquierda para U(x)vsx y derecha para U(y)vsy
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Potenciales para $x$ y $y$')
    ax1.plot(Particion_X, U_x, '.', color='black')
    ax1.plot(Particion_X, potx, '-.', linewidth=0.5, label='kx={}'.format(2*px[0]))
    ax2.plot(Particion_Y, U_y, '.', color='red')
    ax2.plot(Particion_Y, poty, '-.', linewidth=0.5, label='ky={}'.format(2*py[0]))
    ax1.set_title('Potencial $U_x$')
    ax2.set_title('Potencial $U_y$')
    ax1.set(xlabel='Posición ($10^{-8}m$)', ylabel='$U_x(k_BT)$')
    ax2.set(xlabel='Posición ($10^{-8}m$)', ylabel='$U_y(k_BT)$')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
def graficar_posiciones(x,y):
    tx=np.linspace(0, len(x),len(x))
    ty=np.linspace(0, len(y),len(y))
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
def graficar_distribucion(Particion_X, X_Frec, Particion_Y, Y_Frec):
    fig, (g1, g2) = plt.subplots(2, 1)
    fig.suptitle('Distribución de posiciones $x$ y $y$')
    g1.plot(Particion_X , X_Frec,marker='o', linestyle='-',color='r', alpha=0.6, label='X')  # Rojo para X
    g2.plot(Particion_Y , Y_Frec,marker='o', linestyle='-', color='k', alpha=0.6, label='Y')  # Negro para Y
    g1.set_title('Posiciones $x$')
    g2.set_title('Posiciones $y$')
    g1.set(xlabel='$x$', ylabel='$count$')
    g2.set(xlabel='$y$', ylabel='$count$')
    g1.legend(loc='best')
    g2.legend(loc='best')
    g1.grid(True, linestyle="--", alpha=0.5) 
    g2.grid(True, linestyle="--", alpha=0.5)
def filtro(Frecuencias, Particion):
    a=[]
    for i in range(len(Frecuencias)):
        if Frecuencias[i]==0.0:
                a.append(i)
    while True:
        try:
            Frecuencias.remove(0)
        except ValueError:
            break
    a.sort(reverse=True)
    for i in range(len(a)):
        Particion.pop(a[i])
    return 0;
def fuerzas(k,alpha):
    #Siguiendo la relación 
    #F_{\alpha}=-k_{\alpha}*\alpha, tomaremos \alpha=mean(\alpha)
    F=-k*np.mean(alpha)
    return F

import glob   #Se importa la librería glob, para poder importar los archivos *.txt
lista_archivos=glob.glob('C:/Users/PC/OneDrive/Desktop/Datos 11032025/20pwr/*.txt')  #Cargamos todos los archivos *.txt a la variable "lista_archivos", modificar el path si es necesario
#El path debe estar escrito con '/' y no con '\', si se copia del directorio, la copia sale como '\'
for na in lista_archivos:       #Recorremos archivo por archivo con el nombre 'na' (nombre de archivo)
    #Se lee el archivo *.txt y splitea
    x,y=np.loadtxt(na,usecols=(1,3),unpack=True, skiprows=1)    #El archivo de datos contiene 6 columnas, las columnas 2 y 4 son las de x e y, respectivamente, aquí la cuenta se inicia en 0, modificar 'usecols' si es necesario
    x=list(x);y=list(y)
    x=np.array(x);y=np.array(y)
    #El factor de conversión por trabajar en micras:
    x*=1e-6
    y*=1e-6
    sigma_x=np.var(x)                   #Se calculan las varianzas de x y y
    sigma_y=np.var(y)
    kx_Equipartición.append(C/sigma_x)               #Se calculan las rigideces por el método de Equipartición
    ky_Equipartición.append(C/sigma_y)
    Fx_Eq.append(fuerzas(C/sigma_x,x))
    Fy_Eq.append(fuerzas(C/sigma_y,y))
    #Se convierten en listas:
    x=list(x)
    y=list(y)
    """
    #############################################Graficar posiciones vs t
    """
    #graficar_posiciones(x, y)
    
    #Se ordenan:
    x.sort()
    y.sort() 
    
    
    
    #Se obtienen los valores máximos y mínimos
    xmax=max(x);xmin=min(x);
    ymax=max(y);ymin=min(y);
    #Se crean los dx y dy           Desde aquí debería ser la condición
    dx=(xmax-xmin)/bins
    dy=(ymax-ymin)/bins
    
    
    
    #Se crean los vectores para los valores de la nueva partición
    Particion_X = [xmin + i * dx for i in range(bins + 1)]
    Particion_Y = [ymin + i * dy for i in range(bins+1)]
    
    
    
    #Estos son los valores (sin repetir) de x y y, nos van a servir para el conteo de frecuencias
    xrepres=list(set(x))
    xrepres.sort()
    yrepres=list(set(y))
    yrepres.sort()
    #Se crean los vectores para los valores de las frecuencias en la lista original y la de la nueva partición
    xfrec=[]
    yfrec=[]
    X_Frec=[]
    Y_Frec=[]
    
    #Determinamos los valores de x_frecuencias y y_frecuencias de toda la lista
    for i in range(len(xrepres)):
        xfrec.append(x.count(xrepres[i]))
    for i in range(len(yrepres)):
        yfrec.append(y.count(yrepres[i]))
    

    
    
    #Ahora calculamos sus frecuencias con los acumulados anteriores      #Se debe cambiar el intervalo
    for i in range(bins+1):
        suma=0;     #Iniciamos suma
        for j in range(len(xfrec)):
            if Particion_X[i]-dx/2<xrepres[j]<=Particion_X[i]+dx/2: #Imponemos la condición del intervalo
                suma+=xfrec[j]
        X_Frec.append(suma);
    #Para Y
    for i in range(bins+1):
        suma=0;
        for j in range(len(yfrec)):
            if Particion_Y[i]-dy/2<yrepres[j]<=Particion_Y[i]+dy/2:
                suma+=yfrec[j]
        Y_Frec.append(suma);
        
    """
    ############################ Graficar distribución de posiciones
    """
    #graficar_distribucion(Particion_X, X_Frec, Particion_Y, Y_Frec)
        

    # Ahora quitamos todos los valores de frecuencias iguales a 0 con sus respectivos valores en la partición:
    #Para X:
    filtro(X_Frec,Particion_X)
    #Para Y:
    filtro(Y_Frec,Particion_Y)
        
        
    #Se crean los vectores para los valores del potencial en x y en y
    U_x=[]
    U_y=[]
    #Se calculan estos valores de potencial de acuerdo a la ecuación:
    #   U(\alpha)=-k_BTln(\rho(\alpha))     , con \alpha=x,y y \rho= Función de Densidad de Probabilidad (PDF)   
    U_x = [-k_B * T * np.log(f) for f in X_Frec]
    U_y = [-k_B * T * np.log(f) for f in Y_Frec]
    #Se realiza el ajuste a las parábolas abiertas hacia arriba que resulta de U(\alpha)vs\alpha es decir, potencial en función de posición
    px=np.polyfit(Particion_X,U_x,2)    #Aquí se ajusta un polinomio de grado 2, con p=a*x^2+b*x+c, donde p[0]=a; p[1]=b; p[2]=c
    py=np.polyfit(Particion_Y,U_y,2)
    #Se crean los vectores para esos valores del polinomio evaluado en los valores de posición
    kx_Boltzmann.append(2*px[0])
    ky_Boltzmann.append(2*py[0])
    Fx_B.append(fuerzas(2*px[0],x))
    Fy_B.append(fuerzas(2*py[0],y))
    #graficar_potencial(px,py,x,y,Particion_X,Particion_Y,U_x,U_y)
#Sacamos el promedio de las k's cada N experimentos
kx_B_mean=  [sum(kx_Boltzmann[i:i+N])       / N for i in range(0, len(kx_Boltzmann),     N)]
kx_Eq_mean= [sum(kx_Equipartición[i:i+N])   / N for i in range(0, len(kx_Equipartición), N)]
ky_B_mean=  [sum(ky_Boltzmann[i:i+N])       / N for i in range(0, len(ky_Boltzmann),     N)]
ky_Eq_mean= [sum(ky_Equipartición[i:i+N])   / N for i in range(0, len(ky_Equipartición), N)]
#Ahora escribimos un archivo excel con las constantes como columnas y con dos páginas, la primera todas las stiffness y la segunda los promedios
df=pd.DataFrame({"kx_B":kx_Boltzmann,"ky_B":ky_Boltzmann,"kx_Eq":kx_Equipartición,"ky_Eq":ky_Equipartición})
df_2=pd.DataFrame({"kx_B_Mean":kx_B_mean,"ky_B_Mean":ky_B_mean,"kx_Eq_Mean":kx_Eq_mean,"ky_Eq_Mean":ky_Eq_mean})
with pd.ExcelWriter("Stiffness_Analysis.xlsx") as writer:
    df.to_excel(writer, sheet_name="Stiffness_Completas", index=False)  # Primera hoja
    df_2.to_excel(writer, sheet_name="Stiffness_Promedios", index=False)  # Segunda hoja
