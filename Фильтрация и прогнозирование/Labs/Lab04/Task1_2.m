% program is written 27.01.2009 by L.V. Zotov
clear;
clc;
close all;

% 90*12 = 1080 месяцев
N = 1080;
N_signal = N;
% День рождения 05.2000 => 2000*12+5=24005
start = 24005;
k=start:1:start+N-1;
% период 1 год
omega1 = 2*pi/12;
% период 8.86 год
omega2 = 2*pi/12/8.86;
% период 18.6 год
omega3 = 2*pi/12/18.6;
%фаза начало года
ph1 = 0;
%фаза начало 2024 года  
ph2 = 1.12*pi;
%фаза начало 2006 года
ph3 = 0.31*pi;
% Амплитуда, ФИО - ПАС 16 0 18
a1 = 36;
a2 = 20;
a3 = 38;
% период 1 год
f1=a1*cos(omega1*k + ph1);
% период 8.86 год
f2=a2*sin(omega2*k + ph2);
% период 18.6 год
f3=a3*sin(omega3*k + ph3);

garm = f1+f2+f3;

% N_signal=1024;
% generating two-sin signal
% garm=zeros(1,N_signal);
ar=zeros(1,N_signal);
dates=zeros(1,N_signal);

% dt=0.05
% P1=10/dt;
% P2=1/dt;
dt=1;
garm1 = f1;
garm2 = f2;
garm3 = f3;
for (k=1:1:N_signal)
%     garm1(k)=0.1*k*sin(2*pi/P1*(k-1));
%     garm2(k)=10*cos(2*pi/P2*(k-1));
    trend(k)=0.1*k;
    dates(k)=start+(k-1);
end;

% garm = garm1+garm2;

figure;
plot(garm);

% ARMA process generating
noise=2*randn(1,N_signal);

%making a sum
signal=f1+f2+f3+trend+noise;

figure;
plot(dates,garm1,dates,garm2,dates,garm3,dates,trend,dates,noise,dates,signal,'black');
legend('harmonic 1', 'harmonic 2', 'harmonic 3','trend','noise','signal')



pathout='./';
addpath('./functions/');

[ spectr, freq] = spect_fftn(dates,signal);

figure;
plot(freq', abs(spectr)')   
title('amplitude spectrum - module of Fourier-transformation ')
xlabel('frequency, cycles per year')

figure;
cwt(signal,years(dt),'amor');

 L=350;
 N_loc=1;
 N_ev=7;
 coef=1;
 dir_add='S_'
 p_group=[1 0;2 5 ; 6 7 ;3 4;]
 figure;
 Mssa(dates, signal, N_loc,N_signal,L, N_ev,coef,dir_add,pathout,p_group)

 cd("./..");