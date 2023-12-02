clc;
clear;
close all;

[signal, N_signal, years] = GenMySignal();
figure
plot(years, signal)

% try to add noise with Coef*randn([1,N_signal])
eps=9*randn(1,N_signal);

% figure
% hist(eps,20)

% figure
% plot(years, signal+eps);

% ARMA process generating
% 1) Добавить коррелированный шум
ar(1)=eps(1);
ar(2)=-0.7*ar(1)+eps(2);

for (i=3:1:N_signal)
    ar(i)=-0.7*ar(i-1)+0.2*ar(i-2)+eps(i);
end;

figure
plot(years, ar)

figure
plot(years, signal+ar);

%fast Fourier tr (NO NEED?)
% spectr=fft(signal+ar);
% figure
% plot(abs(fftshift(spectr)))

% ACF calculation
% 2) Вычислить АКФ 
signal_centered=signal+ar-mean(signal+ar);

for(tau=1:1:N_signal)
 acf(tau)=0;
 for(j=1:1:N_signal-tau)
    acf(tau)=acf(tau)+signal_centered(j)*signal_centered(j+tau-1);
 end;
  acf(tau)= acf(tau)/(N_signal);
end;
figure
plot(years, acf);

for(tau=1:1:N_signal)
 acf_unbiased(tau)=0;
 for(j=1:1:N_signal-tau)
    acf_unbiased(tau)=acf_unbiased(tau)+signal_centered(j)*signal_centered(j+tau-1);
 end;
  acf_unbiased(tau)= acf_unbiased(tau)/(N_signal - tau);
end;
acf_unbiased(N_signal) = 0;
figure
plot(years, acf_unbiased);

% 3) Построить СПМ 
% spectr_dens=fft(acf);
% figure
% plot(abs(fftshift(spectr_dens)))
% 
% spectr_dens2=fft(acf_unbiased);
% figure
% plot(abs(fftshift(spectr_dens2)))


[spectr, freq] = spect_fftn(years, acf);
figure
plot(freq(2:N_signal), abs(spectr(2:N_signal))) % линейная частота (циклов в год)
xlabel('Cycles per year');

[spectr_unbiased, freq_unbiased] = spect_fftn(years, acf_unbiased);
figure
plot(freq_unbiased(2:N_signal), abs(spectr_unbiased(2:N_signal))) % линейная частота (циклов в год)
xlabel('Cycles per year');