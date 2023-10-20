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
% Signal from lecture
% for (k = 1:1:N)
%     signal(k) = sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
% end

% Not My Signal
% Coef=0.2;
% k=1:1:N;
% signal=.5*sin(2*pi/10*(k-1))+3*sin(2*pi/300*(k-1))+2*sin(2*pi/100*(k-1))+Coef*randn([1,N]);

% Скользящее среднее
% 1) свертка в цикле
% 2) свертка векторно-матрично
% 3) через частотную область, домножением спектра (HERE)

N_Avg = 10; % 10 12 30 60

h = zeros(1,N);
for (j = 1:1:N)
    if (or(j <= N_Avg/2, j > N-N_Avg/2))
        h(j) =  1/N_Avg;
    else
        h(j) = 0;
    end
end

% plot(h);

h_f_tr = fft(h);
s_f_tr = fft(signal);

svertka_fft = h_f_tr.*s_f_tr;
svertka = abs(svertka_fft);

% plot(-N/2:0,svertka(N/2:N),1:N/2,svertka(1:N/2));

res = ifft(svertka_fft);

figure;
plot(signal)
hold on
%С краевым эфектом
plot(res);
hold off