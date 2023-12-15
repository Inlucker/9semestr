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

% signal = interp1(times, signal, t_sampled, 'linear');
% signal = interp1(years, signal, years_sampled, 'linear');

figure
plot(years, signal)

noise = 20*randn(size(signal)); % 10*randn(size(signal));
figure
plot(years, noise)

input = signal + noise;
figure
plot(years, signal + noise)

% Построить СПМ
[spectr, freq] = spect_fftn(years, signal + noise);
figure
plot(freq(2:N), abs(spectr(2:N))*2)
% plot(freq(N_sampl*299/600:N_sampl*301/600), abs(spectr(N_sampl*299/600:N_sampl*301/600))*2)

% вейвлет-скейлограмму
a_max=128;
figure
c = cwt(input,[1:a_max],'morl','plot'); %старая версия cwt
