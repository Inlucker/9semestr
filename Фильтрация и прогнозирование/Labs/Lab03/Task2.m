%wavelet-transformation of the signal
% program is written 27.01.2009 by L.V. Zotov
clear;
clc;
close all;

% MY Signal
[garm, N_signal] = GenMySignal();

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
    
plot(ar);

%making a sum
signal=garm+ar;

% signal(100) = signal(100) + 100; % task6

plot(signal);

%------------------------------------------------
a_max=128;

c = cwt(signal,[1:a_max],'morl','plot'); %старая версия cwt

% clf;

%3d-plot
[A,B] = meshgrid([1:100],[1:a_max]);
for(i=1:1:a_max)
  %morl
    scale_ampl=sqrt(2*pi);
  %mexh
  %  scale_ampl=sqrt(2*pi)*i;
 for(j=1:1:100)
   Z(i,j)=c(i,j*10)/scale_ampl;
 end;
end;

%morl
scale_period=2*pi/5;
%mexh
%scale_period=pi*sqrt(2);

% figure;
% surface(B*scale_period,A,abs(Z))