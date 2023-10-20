clear;
clc;
% Создайте сигнал
k = 1:1024;
signal = sin(2*pi/10*(k-1)) + sin(2*pi/100*(k-1));

% Размер окна для скользящего среднего
window_size = 30;

% Выполните скользящее среднее в частотной области
fft_signal = fft(signal);

% Создайте окно для фильтрации частот
fft_window = zeros(size(fft_signal));
half_window_size = floor(window_size / 2);
fft_window(1:half_window_size) = 1/window_size;
fft_window(end-half_window_size+1:end) = 1/window_size;
% plot(fft_window)

% Примените окно к спектру
fft_window = fft(fft_window);
smoothed_signal = ifft(fft_signal .* fft_window);

% Выведите результат
figure;
plot(k, signal, 'b');
hold on;
plot(k, real(smoothed_signal), 'r');
legend('Исходный сигнал', 'Сглаженный сигнал');
xlabel('Время (k)');
ylabel('Амплитуда');
title('Скользящее среднее в частотной области');
hold off;

% Очистите память от ненужных переменных
% clear k signal window_size fft_signal fft_window smoothed_signal;