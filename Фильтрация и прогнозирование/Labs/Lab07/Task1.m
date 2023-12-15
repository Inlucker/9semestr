clear;
clc;
close all;

[signal, N, years] = GenMySignal();

times = 1:1:N;
sampling_interval = 1/30/4;
t_sampled = 1:sampling_interval:N;
start = 24005;
years_sampled = start/12:1/12*sampling_interval:(start+N-1)/12;
N_sampl = size(years_sampled, 2);

signal = interp1(years, signal, years_sampled, 'linear');

figure
plot(years_sampled, signal)
title('Исходный сигнал');

% Добавляем белый шум
SNR_dB = 10; % отношение сигнал-шум в децибелах
signal_power = rms(signal)^2; % Calculate signal power
noise_power = signal_power / (10^(SNR_dB/10)); % Calculate noise power
noise = sqrt(noise_power) * randn(size(years_sampled)); % Generate white noise with the desired power

noisy_signal = signal + noise;

% Отображаем исходный сигнал и шум
figure;
plot(years_sampled, noise);
title('Белый шум');

% Отображаем исходный сигнал и сигнал с шумом
figure;
plot(years_sampled, noisy_signal);
title('Сигнал с добавленным белым шумом');

% Построить СПМ (Спектр)
[spectr, freq] = spect_fftn(years_sampled, signal + noise);
figure
plot(freq(N_sampl*299/600:N_sampl*301/600), abs(spectr(N_sampl*299/600:N_sampl*301/600))*2)

% вейвлет-скейлограмму
a_max=128;
figure
c = cwt(noisy_signal,[1:a_max],'morl','plot'); %старая версия cwt