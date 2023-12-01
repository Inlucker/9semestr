clc;
clear;
close all;
%1 задание

N_signal=1024;
k=1:1:N_signal;
signal=sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));

figure;
hold on;
plot(k , signal); 
title('Исходный сигнал из семинара');
grid on;
hold off;

eps=0.2*randn(1,N_signal);

histogram(eps,20);
plot(signal+eps);

% ARMA process generating
ar(1)=eps(1);
ar(2)=-0.7*ar(1)+eps(2);

for i=3:1:N_signal
    ar(i)=-0.7*ar(i-1)+0.2*ar(i-2)+eps(i);
end

noisy_signal = signal+ar;

figure;
hold on;
plot(k , ar); 
title('Шум');
grid on;
hold off;

figure;
hold on;
plot(k , noisy_signal); 
title('Зашумленный сигнал из семинара');
grid on;
hold off;

% %fast Fourier tr
% spectr=fft(noisy_signal);
% 
% figure;
% hold on;
% plot(abs(spectr));
% title('amplitude spectrum - module of Fourier-transformation ');
% xlabel('frequency, cycles per year')
% grid on;
% hold off;


signal_centered=noisy_signal-mean(noisy_signal);
acf_biased = zeros(1,N_signal);
acf_unbiased = zeros(1,N_signal);

for tau=1:1:N_signal
    acf_biased(tau)=0;
    for j=1:1:N_signal-tau
        acf_biased(tau)=acf_biased(tau)+signal_centered(j)*signal_centered(j+tau-1);
    end
    acf_biased(tau)= acf_biased(tau)/(N_signal);
end

for tau = 1:1:N_signal
    acf_unbiased(tau) = 0;
    for j = 1:1:N_signal - tau
        acf_unbiased(tau) = acf_unbiased(tau) + signal_centered(j) * signal_centered(j + tau - 1);
    end
    acf_unbiased(tau) = acf_unbiased(tau) / (N_signal - tau);
end


figure;
hold on;
plot(acf_biased); 
title('АКФ смещенная');
grid on;
hold off;

figure;
hold on;
plot(acf_unbiased); 
title('АКФ несмещенная');
grid on;
hold off;

spectr_dens=fft(acf_biased);

figure;
hold on;
plot(abs(spectr_dens));
title('СПМ');
grid on;
hold off;