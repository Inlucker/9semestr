%wavelet-transformation of the signal
% program is written 27.01.2009 by L.V. Zotov
clear;
clc;
close all;

% MY Signal
[garm, N_signal] = GenMySignal();

figure;
plot(garm);

% ARMA process generating
ar(1)=0.5*randn(1);
ar(2)=-0.2*ar(1)+0.5*randn(1);

for (i=3:1:N_signal)
    ar(i)=0.9*ar(i-1)-0.7*ar(i-2)+0.5*randn(1);
end;

for (i=1:1:N_signal)
    ar(i)=ar(i)*20;
end;
    
figure;
plot(ar);

%making a sum
signal=garm+ar;

% signal(100) = signal(100) + 100; % task6

figure;
plot(signal);

savefile = 'signal.mat';
save(savefile,'signal');