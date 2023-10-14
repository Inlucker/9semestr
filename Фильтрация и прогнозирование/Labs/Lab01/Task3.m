clear;
clc;

% Читаем файл
filename = 'eopc01.iau2000.1900-now.dat';
fin = fopen(filename);
fgetl(fin);
A=fscanf(fin,'%f',[11,inf]);
fclose(fin);

l = size(A);
N = l(2);
    
x=A(2,1:N);
y=A(4,1:N);

% Строим график
% plot(x, y)

% Формируем комплексный временной ряд
f = complex(x, y);
% Спектральный анализ 
[apl_spectr, omega] = ampl_fft(f);
% res = cat(2, apl_spectr(N/2:N), apl_spectr(2:N/2));

hold on
% plot(apl_spectr(2:N)*2) % график амплитудного спектра
% plot(apl_spectr(2:N/2)*2) % график амплитудного спектра
% plot(apl_spectr1(2:N/2)*2) % график амплитудного спектра
% plot(apl_spectr2(2:N/2)*2) % график амплитудного спектра
hold off

% plot(-(N/2)+1:1:(N/2)-1, res*2);

% plot(apl_spectr(2:N)*2) % график амплитудного спектра
% plot(fftshift(apl_spectr)) % график амплитудного спектра
% plot(omega(1:N), apl_spectr(1:N)*2) % циклическая частота
plot(omega(2:N), apl_spectr(2:N)*2) % циклическая частота
% plot(omega(2:N), res) % циклическая частота
% plot(omega/2/pi, apl_spectr) % линейная частота, спектрограмма
% plot(1./(omega/2/pi), apl_spectr) % период, периодограмма