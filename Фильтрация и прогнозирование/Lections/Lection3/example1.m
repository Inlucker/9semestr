clear;
clc;

N=1024;
for (k = 1:1:N)
    signal(k) = sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
end

% Скользящее среднее
% 1) свертка в цикле
% 2) свертка векторно-матрично (HERE?)
% 3) через спектральную область, домножением спектра

N_Avg = 12; % 12 30

for (j = 1:1:N_Avg)
    h(j) =  1/N_Avg;
end

X=zeros(N_Avg, N-N_Avg+1);

% for (j = 1:1:N-N_Avg+1)
%     for (k = 1:1:N_Avg)
%         X(k, j) = signal(1, k+j-1);
%     end
% end

for (j = 1:1:N-N_Avg+1)
    X(:,j) = transpose(signal(j:j+N_Avg-1));
end

res = h*X;

% сглаживание
plot(signal)
hold on
plot(N_Avg/2:1:N-N_Avg/2, res);
% hold off

% усиливание высоких частот
test = signal(N_Avg/2:N-N_Avg/2)-res;
plot(N_Avg/2:1:N-N_Avg/2, test);
hold off