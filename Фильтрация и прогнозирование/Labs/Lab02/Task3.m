clear;
clc;

% 90*12 = 1080 месяцев
N = 1080;
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

% MY Signal
signal = f1+f2+f3;

% N=1024;
% My Signal
% for (k = 1:1:N)
%     signal(k) = sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
% end

% Not My Signal
% k=1:1:N;
% omega=2*pi/100;
% a=0.5;
% signal=2*a*cos(omega*k+pi/3)+a*cos(10*omega*k);

% Скользящее среднее
% 1) свертка в цикле
% 2) свертка векторно-матрично (HERE)
% 3) через частотную область, домножением спектра

N_Avg = 20; % 10 12 20 30 50

for (j = 1:1:N_Avg)
    h(j) =  1/N_Avg;
end

X=zeros(N_Avg, N-N_Avg);

% for (j = 1:1:N-N_Avg)
%     for (k = 1:1:N_Avg)
%         X(k, j) = signal(1, k+j-1);
%     end
% end

for (j = 1:1:N-N_Avg)
    X(:,j) = transpose(signal(j:j+N_Avg-1));
end

res = h*X;

figure;
% сглаживание
plot(signal)
hold on
plot(N_Avg/2:1:N-N_Avg/2-1, res);
hold off

% усиливание высоких частот
% test = signal(N_Avg/2:N-N_Avg/2)-res;
% plot(N_Avg/2:1:N-N_Avg/2, test);
% hold off