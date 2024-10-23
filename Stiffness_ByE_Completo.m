% Constantes físicas y número de bins:
k_B = 1.380649e-23;
T = 295.15;
C = k_B * T;
bins = 20;

% Se crean los vectores que van a contener a los n valores de stiffness en x e y
kx_Boltzmann = [];
ky_Boltzmann = [];
kx_Equiparticion = [];
ky_Equiparticion = [];

% Definir función de graficación
function graficar(px, py, x, y, Particion_X, Particion_Y, U_x, U_y)
    x_min = -px(2) / (2 * px(1));
    y_min = -py(2) / (2 * py(1));

    % Modificar partición con los nuevos valores
    Particion_X = Particion_X - x_min;
    Particion_Y = Particion_Y - y_min;

    % Ajustar las parábolas para los nuevos valores
    px = polyfit(Particion_X, U_x, 2);
    py = polyfit(Particion_Y, U_y, 2);
    potx = polyval(px, Particion_X);
    poty = polyval(py, Particion_Y);

    % Conversión para la gráfica
    U_x = U_x / 1e-20;
    potx = potx / 1e-20;
    U_y = U_y / 1e-20;
    poty = poty / 1e-20;
    Particion_X = Particion_X / 1e-8;
    Particion_Y = Particion_Y / 1e-8;

    % Graficar
    figure;
    subplot(1, 2, 1);
    plot(Particion_X, U_x, '.', 'Color', 'black');
    hold on;
    plot(Particion_X, potx, '-.', 'LineWidth', 0.5);
    title('Potencial U_x');
    xlabel('Posición (10^{-8} m)');
    ylabel('U_x(k_BT)');
    legend(['kx=', num2str(2*px(1))], 'Location', 'best');
    hold off;

    subplot(1, 2, 2);
    plot(Particion_Y, U_y, '.', 'Color', 'red');
    hold on;
    plot(Particion_Y, poty, '-.', 'LineWidth', 0.5);
    title('Potencial U_y');
    xlabel('Posición (10^{-8} m)');
    ylabel('U_y(k_BT)');
    legend(['ky=', num2str(2*py(1))], 'Location', 'best');
    hold off;
end

% Función filtro (corrige que no devuelva más de un valor)
function [Frecuencias, Particion] = filtro(Frecuencias, Particion)
    a = find(Frecuencias == 0);
    Frecuencias(a) = [];
    Particion(a) = [];
end

% Lectura de archivos *.txt
archivos = dir('C:/Users/PC/Desktop/Optical Tweezers/TweezVip/Datos/*.txt');
for n = 1:length(archivos)
    filename = fullfile(archivos(n).folder, archivos(n).name);
    data = load(filename);
    
    % Leer las columnas x e y
    x = data(:, 2);
    y = data(:, 4);

    % Eliminar primer elemento y convertir a micras
    x(1) = [];
    y(1) = [];
    x = x * 1e-6;
    y = y * 1e-6;

    % Calcular varianzas
    sigma_x = var(x);
    sigma_y = var(y);

    % Calcular rigideces por el método de equipartición
    kx_Equiparticion = [kx_Equiparticion, C / sigma_x];
    ky_Equiparticion = [ky_Equiparticion, C / sigma_y];

    % Ordenar los valores de x y y
    x = sort(x);
    y = sort(y);

    % Obtener máximos y mínimos
    xmin = min(x); xmax = max(x);
    ymin = min(y); ymax = max(y);

    % Crear particiones
    dx = (xmax - xmin) / bins;
    dy = (ymax - ymin) / bins;
    Particion_X = xmin + (0:bins) * dx;
    Particion_Y = ymin + (0:bins) * dy;

    % Representación de valores únicos
    xrepres = unique(x);
    yrepres = unique(y);

    % Contar frecuencias
    xfrec = histcounts(x, [-Inf, xrepres', Inf]);
    yfrec = histcounts(y, [-Inf, yrepres', Inf]);

    % Calcular frecuencias para cada partición
    X_Frec = zeros(1, bins+1);
    Y_Frec = zeros(1, bins+1);
    
    for i = 1:bins+1
        X_Frec(i) = sum(xrepres >= Particion_X(i) - dx/2 & xrepres < Particion_X(i) + dx/2);
        Y_Frec(i) = sum(yrepres >= Particion_Y(i) - dy/2 & yrepres < Particion_Y(i) + dy/2);
    end

    % Filtrar frecuencias iguales a 0
    [X_Frec, Particion_X] = filtro(X_Frec, Particion_X);
    [Y_Frec, Particion_Y] = filtro(Y_Frec, Particion_Y);

    % Calcular el potencial
    U_x = -k_B * T * log(X_Frec);
    U_y = -k_B * T * log(Y_Frec);

    % Ajuste de polinomio cuadrático
    px = polyfit(Particion_X, U_x, 2);
    py = polyfit(Particion_Y, U_y, 2);

    % Calcular valores del polinomio
    kx_Boltzmann = [kx_Boltzmann, 2 * px(1)];
    ky_Boltzmann = [ky_Boltzmann, 2 * py(1)];

    % Graficar (descomentar si es necesario)
    graficar(px, py, x, y, Particion_X, Particion_Y, U_x, U_y);
end

% % Mostrar resultados
% disp('kx_Boltzmann:');
% disp(kx_Boltzmann);
% 
% disp('ky_Boltzmann:');
% disp(ky_Boltzmann);
% 
% disp('kx_Equiparticion:');
% disp(kx_Equiparticion);
% 
% disp('ky_Equiparticion:');
% disp(ky_Equiparticion);
