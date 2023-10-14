clear;
clc;

% 90*12 = 1080 месяцев
N = 1080;
% N = 12; % 1*12
% N = 108; % 9*12
% N = 228; % 19*12

% День рождения 05.2000 => 2000*12+5=24005
start = 24005;
% start=0;
% start = 24288; % 2024*12 = 24288
% start = 24072; % 2006*12 = 24072
k=start:1:start+N;

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

f = f1+f2+f3;

% plot(k,f1)
% plot(k,f2)
% plot(k,f3)
plot(k,f)

[apl_spectr, omega] = ampl_fft(f);

% plot(apl_spectr(2:N/2)*2) % график амплитудного спектра (неправильно без частот omega?)
plot(omega(2:N/2), apl_spectr(2:N/2)*2) % циклическая частота
% plot(omega, apl_spectr*2) % циклическая частота (неправильно начитнать с 1ого элемента?)
% plot(omega/2/pi, apl_spectr) % линейная частота, спектрограмма
% plot(1./(omega/2/pi), apl_spectr) % период, периодограмма


% фазовый спектр
% hold on
% f_tr=fft(f);
% phase_spectr = angle(f_tr);
% % phase_spectr = imag(f_tr)/N;
% plot(omega, phase_spectr)
% hold off