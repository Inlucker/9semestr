clear;
clc;

N=1024;
for (k = 1:1:N)
    signal(k) = sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
end

% Скользящее среднее
% 1) свертка в цикле
% 2) свертка векторно-матрично
% 3) через частотную область, домножением спектра (HERE)

N_Avg = 30; % 12 30

h = zeros(1,N);
for (j = 1:1:N)
    if (or(j <= N_Avg/2, j > N-N_Avg/2))
        h(j) =  1/N_Avg;
    else
        h(j) = 0;
    end
end

% plot(h);

[h_apl_spectr, h_omega] = ampl_fft(h);
[s_apl_spectr, s_omega] = ampl_fft(signal);
h_f_tr = fft(h);
s_f_tr = fft(signal);

% plot(h_apl_spectr);
% plot(s_apl_spectr);
% plot(h_f_tr);
% plot(s_f_tr);

% svertka = h_apl_spectr.*s_apl_spectr;
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