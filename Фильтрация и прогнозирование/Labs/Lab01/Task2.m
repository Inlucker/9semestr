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
    
YEARS=A(1,1:N);
UTTAI=A(6,1:N);

dT=0.05;

% Строим график
% plot(YEARS,UTTAI)

% Дифференцируем данные
LOD=zeros(1,N);
for(k=2:1:N-1)
 LOD(k)=-(UTTAI(k+1)-UTTAI(k-1))/2/dT;
end

dates=YEARS(1242:N-1);
lod=LOD(1242:N-1);

% Строим график минус производной
% plot(dates,lod)

% Спектральный анализ lod
f=lod;
[apl_spectr, omega] = ampl_fft(f);

NN = N-1242;

% plot(apl_spectr) % график амплитудного спектра
% plot(apl_spectr(2:NN/2)*2) % график амплитудного спектра правильный
plot(omega(2:NN/2), apl_spectr(2:NN/2)*2) %циклическая частота
% plot(omega(2:NN), apl_spectr(2:NN)*2) %циклическая частота
% plot(fftshift(apl_spectr(2:N))) %циклическая частота (Тоже с правильной omega)
% plot(omega/2/pi, apl_spectr) %линейная частота, спектрограмма
% plot(omega/2/pi/dT, apl_spectr) %линейная частота, спектрограмма
% plot(1./(omega/2/pi), apl_spectr) %период, периодограмма