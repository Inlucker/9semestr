function [ par, resX , D] = HarmFitHomo(Dat,signal,T,date_st)
%МНК-подбор гармоник
%returns a and b amplitudes of cos and sin, model and cov matrix
%in form [a_cos b_sin]
%or for complex form [a_cos_x+ia_cos_y,b_sin_x+ib_sin_y]
%Dat -  dates 
%signal to be alalized
%T array of periods
%date_st starting date


%размер входного массива
N=size(signal,1);
%размер массива периодов (число гармоник в модели)
M=size(T,2);

%удаление среднего (гармоники с бесконечном периодом)
X=signal-mean(signal);
%создание матрицы модели 
A=ones(N,M);
for(i=1:1:N)
 for(j=1:1:M)
    A(i,2*j-1)=cos((Dat(i)-date_st)*2*pi/T(j));
    A(i,2*j)=sin((Dat(i)-date_st)*2*pi/T(j));
 end;
end;

%заполнение ковариационной матрицы
% 


%матрица весов


%нормальная матрица системы
NOM=transpose(A)*A;

%изучение обусловленности
[U,S,V]=svd(NOM);
Obusl=S(1,1)/S(2*M,2*M);

% обращение нормальной матрицы
F=inv(NOM);

%оценка параметров
par=F*transpose(A)*X;

%оптимальная модель
resX=A*par;

%вычисление суммы квадратов невязко
sum=0;
for(i=1:1:N)
    sum=sum+(resX(i)-X(i))^2;
end;


 plot(Dat,real(signal),Dat,real(resX))
% c=real(par(1))+j*real(par(2));
% Ax=sqrt(real(par(1)).^2+real(par(2)).^2)
% fX=atan(real(par(2))./real(par(1)))
% fX1=angle(real(par(1))+sqrt(-1)*real(par(2)))
% if(real(par(2))<0)
%    fX=fX-pi; 
% end;
% 
% Ay=sqrt(imag(par(1)).^2+imag(par(2)).^2)
% fY=atan(imag(par(2))./imag(par(1)))
% fY1=angle(imag(par(1))+sqrt(-1)*imag(par(2)))
% if(real(par(2))<0)
%    fY=fY-pi; 
% end;
% 
%  for(i=1:1:N)
%   modelX(i)=Ax*cos((Dat(i)-Dat(1))*2*pi/T(1)-fX1);
%   modelY(i)=Ay*cos((Dat(i)-Dat(1))*2*pi/T(1)-fY1);
% end;
% % 
%  plot(Dat,imag(signal),Dat,imag(resX),Dat,modelY)

%оценка дисперсии единицы веса
sigma02=sum/(N-M);

%plot(real(signal),imag(signal),real(resX),imag(resX))

%вычисление ковариационной матрицы параметров
D=sigma02*F;


end

