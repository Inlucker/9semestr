clear;
clc;

N=1024;
for (k = 1:1:N)
    signal(k) = sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
end

% Скользящее среднее
% 1) свертка в цикле (HERE домножать в цикле)
% 2) свертка векторно-матрично
% 3) через спектральную область, домножением спектра

N_Avg = 30;

for (j = 1:1:N_Avg)
    h(j) =  1/N_Avg;
end

y = zeros(1, N);

for (j = N_Avg/2:N-N_Avg/2)
    for (k = 1:N_Avg-1)
        id = j-k+(N_Avg/2);
        % y(j) = y(j)+signal(j-k+(N_Avg/2-1))*h(k);
        y(j) = y(j)+signal(id)*h(k);
    end
end

plot(y);

plot(signal)
hold on
plot(N_Avg/2:1:N-N_Avg/2, y(N_Avg/2:1:N-N_Avg/2));
hold off