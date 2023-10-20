clear;
clc;
% Создайте сигнал
k = 1:1024;
signal = sin(2*pi/10*(k-1)) + sin(2*pi/100*(k-1));

% Размер окна для скользящего среднего
window_size = 30;

% Инициализируйте массив для сглаженных данных
smoothed_signal = zeros(size(signal));

% Примените скользящее среднее во временной области
for i = 1:length(signal)
    % Рассчитайте границы окна
    window_start = max(1, i - floor(window_size/2));
    window_end = min(length(signal), i + floor(window_size/2));
    
    % Вычислите среднее внутри окна
    smoothed_signal(i) = mean(signal(window_start:window_end));
end

% Выведите результат
figure;
plot(k, signal, 'b');
hold on;
% plot(k, smoothed_signal, 'r');
plot(k(window_size/2:1024-window_size/2), smoothed_signal(window_size/2:1024-window_size/2), 'r');
legend('Исходный сигнал', 'Сглаженный сигнал');
xlabel('Время (k)');
ylabel('Амплитуда');
title('Скользящее среднее во временной области');
hold off;