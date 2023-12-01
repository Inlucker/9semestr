clc;
clear;
close all;

N_signal=1024;
% generating two-sin signal
%for (k=1:1:N_signal)
%    signal(k)=sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
%end;

k=1:1:N_signal;
signal=sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
figure
plot(signal);

% try to add noise with Coef*randn([1,N_signal])
eps=0.2*randn(1,N_signal);

figure
hist(eps,20)

figure
plot(signal+eps);


% ARMA process generating
ar(1)=eps(1);
ar(2)=-0.7*ar(1)+eps(2);

for (i=3:1:N_signal)
    ar(i)=-0.7*ar(i-1)+0.2*ar(i-2)+eps(i);
end;

figure
plot(ar)

figure
plot(signal+ar);

%fast Fourier tr


spectr=fft(signal+ar);
figure
plot(abs(spectr))


% ACF calculation
signal_centered=signal+ar-mean(signal+ar);


for(tau=1:1:N_signal)
 acf(tau)=0;
 for(j=1:1:N_signal-tau)
    acf(tau)=acf(tau)+signal_centered(j)*signal_centered(j+tau-1);
 end;
  acf(tau)= acf(tau)/(N_signal);
end;
figure
plot(acf);



spectr_dens=fft(acf);



figure
plot(abs(fftshift(spectr_dens)))

