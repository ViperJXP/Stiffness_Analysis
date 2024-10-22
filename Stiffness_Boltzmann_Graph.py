# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:16:59 2024
Este programa calcula los coeficientes de rigidez de una trampa óptica por el método
de Boltzmann o del Análisis del Potencial (https://doi.org/10.1364/AOP.394888)
Se recibe un archivo de texto en formato *.txt con dos columnas, una para x y otra para y
Calcula la distribución con una nueva particiónn de n-bins y de aquí se obtiene el potencial U(x)
Si se grafica U(x)vs x, tiene una forma parabólica y siguiendo la relación
U(x)=1/2*k*x^2, podemos realizar un ajuste polinomial para obtener el polinomio:
    U(x)=\alpha*x^2+\beta*x+\gamma
de manera que se puede obtener:
    k=2*\alpha
@author: Josué Hernández Torres
"""
import numpy as np
import matplotlib.pyplot as plt



#Constantes físicas y tamaño de la Partición
k_B=1.380649e-23
T=295.15                #K =22°C
bins=20



#Se lee el archivo *.txt
c1,x,c3,y,c5,c6=np.loadtxt('170359Pos.txt', unpack=True)     #Se asignan las columnas 0-->x y 1-->y
x=list(x);y=list(y)
x.pop(0);y.pop(0)
x=np.array(x);y=np.array(y)
#El factor de conversión por trabajar en micras
x*=1e-6
y*=1e-6
#Se convierten a tipo listas
x=list(x)
y=list(y)
#Se ordenan
x.sort()
y.sort()


 
#Se obtienen los valores máximos y mínimos
xmax=max(x);xmin=min(x);
ymax=max(y);ymin=min(y);
#Se crean los dx y dy
dx=(xmax-xmin)/bins
dy=(ymax-ymin)/bins
#Se crean los vectores para los valores de la nueva partición
Particion_X = [xmin + i * dx for i in range(bins + 1)]
Particion_Y = [ymin + i * dy for i in range(bins + 1)]



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
    
    
    
#Ahora calculamos sus frecuencias con los acumulados anteriores    #######(Se debe recorrer el intervalo de trabajo)
#Para X
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
    
    
    
    
#Normalizamos los valores
for i in range(len(X_Frec)):
    X_Frec[i]/=len(X_Frec)
    Y_Frec[i]/=len(Y_Frec)



# Ahora quitamos todos los valores de frecuencias iguales a 0 con sus respectivos valores en la partición:
#Para X:
a=[]
for i in range(len(X_Frec)):
    if X_Frec[i]==0.0:
        a.append(i)
while True:
    try:
        X_Frec.remove(0)
    except ValueError:
        break
for i in range(len(a)):
    Particion_X.pop(a[i])
#Para Y:    
a=[]
for i in range(len(Y_Frec)):
    if Y_Frec[i]==0.0:
        a.append(i)
while True:
    try:
        Y_Frec.remove(0)
    except ValueError:
        break
a.sort(reverse=True)
for i in range(len(a)):
    Particion_Y.pop(a[i])


"""    
# Si se desea visualizar la gráfica de Densidad de Probabilidad descomente las 3 líneas siguientes:
# fig=plt.figure(1)
# plt.plot(Particion_X,X_Frec,'.',color='blue')
# plt.plot(Particion_Y,Y_Frec,'.',color='red')
"""



#Se crean los vectores para los valores del potencial en x y en y
U_x = [-k_B * T * np.log(f) for f in X_Frec]
U_y = [-k_B * T * np.log(f) for f in Y_Frec]

#Se realiza el ajuste a las parábolas positivas que resultan de U(\alpha)vs\alpha es decir, potencial en función de posición
px=np.polyfit(Particion_X,U_x,2)    #Aquí se ajusta un polinomio de grado 2, con p=a*x^2+b*x+c, donde p[0]=a; p[1]=b; p[2]=c
py=np.polyfit(Particion_Y,U_y,2)

#Se encuentran los valores mínimos de las parábolas, para cambiar de sistema de referencia a uno donde el 0.0 sea el punto de equilibrio:
x_min = -px[1] / (2 * px[0])
y_min = -py[1] / (2 * py[0])
#Se modifica la Partición con los nuevos valores
Particion_X = [x - x_min for x in Particion_X]
Particion_Y = [y - y_min for y in Particion_Y]
#Se vuelven a ajustar las parábolas para estos nuevos valores
px=np.polyfit(Particion_X,U_x,2)
py=np.polyfit(Particion_Y,U_y,2)

#Se crean los vectores para esos valores del polinomio evaluado en los valores de posición
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
legend = ax1.legend(loc='best')
legend=ax2.legend(loc='best')

#Finalmente, se imprimen los valores en la terminal también
print('kx=',2*px[0],'\n')
print('ky=',2*py[0])
