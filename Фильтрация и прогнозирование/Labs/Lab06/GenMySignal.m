function [ signal, N, years] = GenMySignal(years_num)

% Проверяем, был ли предоставлен аргумент years_num
    if nargin < 1 || isempty(years_num)
        years_num = 90;
    end

% 90*12 = 1080 месяцев
% N = 1080;
N = 12 * years_num;
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

signal = f1+f2+f3;

years = start/12:1/12:(start+N-1)/12;