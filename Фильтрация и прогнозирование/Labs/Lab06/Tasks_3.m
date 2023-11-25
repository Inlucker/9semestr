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

[signal, N] = GenMySignal();

figure
plot(signal)
times = 1:1:N;
sampling_interval = 1/30/4;
t_sampled = 1:sampling_interval:N;

signal = interp1(times, signal, t_sampled, 'linear');

figure
plot(signal)

noise = 20*randn(size(signal)); % 10*randn(size(signal));
figure
plot(noise)

input = signal + noise;
figure
plot(signal + noise)

% times = 1:1:N;
x0 = [0; 0];

% With lsim
% input_matrix = repmat(input, 2, 1);
input_matrix = vertcat(signal, noise);
figure
% lsim(sys, input_matrix, times, x0)
lsim(sys, input_matrix, t_sampled, x0)

