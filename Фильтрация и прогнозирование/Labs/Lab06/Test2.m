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

[u,t] = gensig('square', 5, 300, 0.05);
figure
plot(t, u)

noise = 0.1*randn(size(u));
figure
plot(t, noise)

input = u + noise;
figure
plot(t, input)

input = [u noise];
% input = [noise noise];
lsim(sys, input, t)