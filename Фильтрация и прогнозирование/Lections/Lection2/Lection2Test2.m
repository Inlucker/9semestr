clear;
clc;
N = 1024;

k = 1:1:N;
omega=2*pi/100;
a = 0.5;
% f=a*cos(omega*k);
f=a*cos(omega*k+pi/3); %сдвиг фазы
% f=a*cos(omega*k)+250;

plot(k, f)

filename = 'eopc01.1900-now.dat';

fin = fopen(filename);
fgetl(fin);
A=fscanf(fin,'%f',[11,inf]);
fclose(fin);

l = size(A);
N = l(2);

YEARS=A(1,1:N);
UTTAI=A(6,1:N);

dT=0.05;

plot(YEARS,UTTAI);

LOD=zeros(1,N);

for (k=2:N-1)
    LOD(k-1)=-(UTTAI(k+1)-UTTAI(k-1))/2/dT;
end;

% dates=YEARS(2:N-1);
dates=YEARS(1242:N-1);
lod=LOD(1242:N-1);

% plot(dates,LOD);
plot(dates,lod);

f=lod-mean(lod);
[apl_spectr, omega] = ampl_fft(f);

plot(omega, apl_spectr) %циклическая частота
% plot(omega/2/pi, apl_spectr) %линейная частота, спектрограмма
% plot(1./(omega/2/pi), apl_spectr) %период, периодограмма
hold on
% plot(omega, phase_spectr) %фазовый спектр
