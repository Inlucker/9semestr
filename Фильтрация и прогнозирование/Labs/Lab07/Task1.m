clear;
clc;
close all;

filename=sprintf('./lr6_output_file.txt');
fin=fopen(filename,'rt');
A=fscanf(fin,'%f',[5 inf]);% A - array of data
fclose(fin);

N=size(A,2);
N_max=30000;
N_max=N;
year=A(1,1:N_max);
x_in=A(2,1:N_max);
y_in=A(3,1:N_max);
m_1=A(4,1:N_max);
m_2=A(5,1:N_max);

dt=year(2)-year(1);

inp=x_in+i*y_in;
m=m_1+i*m_2;




Q = 100;
FC=365/433;
f_c = 0.0;

%filter center frequency year^-1
 f_om=10 %filter parameter year^-1
 
 inv=1
 outfilename=sprintf('CW_TRF_%3d_%2d_1.dat',round(1/f_c),round(1/f_om));
  
 noise=(randn(N_max,1)+i*randn(N_max,1))';

 mn=m+noise;
 figure
 plot(year, real(m),year, real(mn))

 % Построить СПМ (Спектр)
[spectr, freq] = spect_fftn(year, mn);
figure
plot(freq(N_max*299/600:N_max*301/600), abs(spectr(N_max*299/600:N_max*301/600))*2)

% вейвлет-скейлограмму
% dt=1/30/4/12;
figure;
cwt(real(mn), years(dt),'amor');
% cwt(abs(mn), years(1/30/4/12),'amor');


[ filtered_signal ]=ChandPantFreqFilter(year,mn,f_om,f_c,dt,FC,Q,inv,outfilename)


figure
plot(year, real(mn),year, real(inp),year,real(filtered_signal),'green')


[ spectr_in, freq_in] = spect_fftn(year,inp);
 [ spectr, freq] = spect_fftn(year, mn);
  [ spectr_fltr, freq_fltr] = spect_fftn(year, filtered_signal );
  figure
%   plot(freq_fltr,abs(spectr_fltr),freq,abs(spectr));
plot(freq_fltr(N_max*299/600:N_max*301/600), abs(spectr_fltr(N_max*299/600:N_max*301/600)), ...
    freq(N_max*299/600:N_max*301/600), abs(spectr(N_max*299/600:N_max*301/600)))
  