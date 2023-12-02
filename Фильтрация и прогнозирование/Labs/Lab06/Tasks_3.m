clear;
clc;
close all;

Q = 100;
f_c = 0.843; % 0.843; % 365/433;
alpha = 2*pi*f_c;
betta = pi*f_c/Q;

G = [[-betta -alpha]; [alpha -betta]];
F = -G;

C = [[1 0]; [0 1]];

sys = ss(G, F, C, 0);
figure
step(sys)

figure
impulse(sys)

null(G)

det(G)

p = charpoly(G)
r = roots(p)

syms t;
expm(t*G)

[signal, N, years] = GenMySignal();

figure
plot(years, signal)
times = 1:1:N;
sampling_interval = 1/30/4;
t_sampled = 1:sampling_interval:N;
start = 24005;
years_sampled = start/12:1/12*sampling_interval:(start+N-1)/12;
N_sampl = size(years_sampled, 2);

% signal = interp1(times, signal, t_sampled, 'linear');
signal = interp1(years, signal, years_sampled, 'linear');

figure
plot(years_sampled, signal)

noise = 20*randn(size(signal)); % 10*randn(size(signal));
figure
plot(years_sampled, noise)

input = signal + noise;
figure
plot(years_sampled, signal + noise)

% Построить СПМ
[spectr, freq] = spect_fftn(years_sampled, signal + noise);
figure
plot(freq(N_sampl*299/600:N_sampl*301/600), abs(spectr(N_sampl*299/600:N_sampl*301/600))*2)

% times = 1:1:N;
x0 = [0; 0];

dots_N = 30000;
% With lsim
% input_matrix = repmat(input, 2, 1);
input_matrix = vertcat(signal+noise, signal+noise);
% figure
% lsim(sys, input_matrix, years_sampled, x0)
[y, t_sim, x] = lsim(sys, input_matrix(:,1:dots_N), years_sampled(1:dots_N), x0);

signal_noise = signal+noise;
% Построение графика угла поворота от времени
% Получить текущий цветовой цикл
colorOrder = get(gca, 'ColorOrder');
% Получить цвет первого графика
firstColor = colorOrder(1, :);
figure
subplot(2, 1, 1);
plot(t_sim, signal_noise(1:dots_N), 'Color', [0.7 0.7 0.7]);
hold on
plot(t_sim, y(:,1), 'Color', firstColor);
hold off

% figure
subplot(2, 1, 2);
plot(t_sim, signal(1:dots_N), t_sim, noise(1:dots_N), 'Color', [0.7 0.7 0.7]);
hold on
plot(t_sim, y(:,2), 'Color', firstColor);
hold off
