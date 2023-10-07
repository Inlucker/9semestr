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
[apl_spectr1, omega1] = ampl_fft(x);
[apl_spectr2, omega2] = ampl_fft(y);

hold on
plot(apl_spectr(2:N/2)*2) % график амплитудного спектра
% plot(apl_spectr1(2:N/2)*2) % график амплитудного спектра
% plot(apl_spectr2(2:N/2)*2) % график амплитудного спектра
hold off


% plot(omega, apl_spectr) % циклическая частота
% plot(omega/2/pi, apl_spectr) % линейная частота, спектрограмма
% plot(1./(omega/2/pi), apl_spectr) % период, периодограмма