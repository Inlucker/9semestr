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
% figure('Position', get(0,'Screensize'));
figure
step(sys)

% figure('Position', get(0,'Screensize'));
figure
impulse(sys)

null(G)

det(G)

p = charpoly(G)
r = roots(p)

syms t;
expm(t*G)