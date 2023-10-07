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

filename = 'filename';

fin = fopen(filename);
fgetl(fin);
A=fscanf(fin,'%f',[11,inf]);
fclose(fin);

[apl_spectr, omega] = ampl_fft(f);

% f_tr = fft(f);
% apl_spectr=abs(f_tr)/N;
% % apl_spectr=real(f_tr)/N;
% phase_spectr = angle(f_tr);
% % phase_spectr = imag(f_tr)/N;
% plot(apl_spectr(2:N/2)*2)
% 
% T = 1;
% 
% for(j=1:1:N)
%       if(j==1)
%         t(j)=0;
%         omega(j)=0;   
% 
%       elseif(j<=N/2+1)
%         t(j)=N*T/(j-1); 
%         omega(j)=2*pi/t(j); 
% 
%       elseif(j>N/2+1)
%         t(j)=N*T/(N-j+1);
%         omega(j)=-2*pi/t(j);
%        
%       end;
%       
%       %spectr(k)=sum(exp(-i*omega(k)*DAT).*signal);
%       
%  end;

plot(omega, apl_spectr) %циклическая частота
% plot(omega/2/pi, apl_spectr) %линейная частота, спектрограмма
% plot(1./(omega/2/pi), apl_spectr) %период, периодограмма
hold on
% plot(omega, phase_spectr) %фазовый спектр
