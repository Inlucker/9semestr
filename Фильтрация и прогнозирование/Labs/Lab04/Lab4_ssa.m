% program is written 27.01.2009 by L.V. Zotov
clear;

N_signal=1024;
% generating two-sin signal
garm=zeros(1,N_signal);
ar=zeros(1,N_signal);
dates=zeros(1,N_signal);

dt=0.05
P1=10/dt;
P2=1/dt;
for (k=1:1:N_signal)
    garm1(k)=0.1*k*sin(2*pi/P1*(k-1));
    garm2(k)=10*cos(2*pi/P2*(k-1));
    trend(k)=0.1*k;
    dates(k)=2000+dt*(k-1);
end;

plot(garm);

% ARMA process generating
noise=2*randn(1,N_signal);

%making a sum
signal=garm1+garm2+trend+noise;

plot(dates,garm1,dates,garm2,dates,trend,dates,noise,dates,signal,'black');
legend('harmonic 1', 'harmonic 2','trend','noise','signal')



pathout='./';
addpath('./functions/');

[ spectr, freq] = spect_fftn(dates,signal);

plot(freq', abs(spectr)')   
title('amplitude spectrum - module of Fourier-transformation ')
xlabel('frequency, cycles per year')

cwt(signal,years(dt),'amor');

 L=300;
 N_loc=1;
 N_ev=7;
 coef=1;
 dir_add='S_'
 p_group=[1 0;2 3 ;4 5]
 Mssa(dates, signal, N_loc,N_signal,L, N_ev,coef,dir_add,pathout,p_group)
 