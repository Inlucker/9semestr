clc;
clear;
close all;

path='./';


% file reading, we will analize El Nino index
% 1)	Считать индекс явления El Nino Southern Oscillation ENSO BEST 
filename=sprintf('%s/ENSO_BEST_index.dat',path);

 fin=fopen(filename,'rt');
 Adat=fscanf(fin,'%f',[3 inf]);
 fclose(fin);

 dates=Adat(1,:); % time column
 x=Adat(2,:);  %data column
 l=size(Adat);

 N=l(2);
 
dt=1/12;%a month step 
figure
plot(dates,x); % to check reading 
title('to check reading');

% 2)	Вычислить оценку АКФ ( смещенную, несмещенную)
m=mean(x);
signal_centered=x-m;
 
for(tau=1:1:N)
 acf(tau)=0;
 for(j=1:1:N-tau)
    acf(tau)=acf(tau)+signal_centered(j)*signal_centered(j+tau-1);
 end;
  acf(tau)= acf(tau)/(N);
end;

for(tau=1:1:N)
 acf_unbiased(tau)=0;
 for(j=1:1:N-tau)
    acf_unbiased(tau)=acf_unbiased(tau)+signal_centered(j)*signal_centered(j+tau-1);
 end;
  acf_unbiased(tau)= acf_unbiased(tau)/(N - tau);
end;
acf_unbiased(N) = 0;

figure
plot(0:dt:(N-1)*dt,acf); %autocovariance function over tau plot
title('ACF over tau plot');
figure
plot(0:dt:(N-1)*dt,acf_unbiased); %autocovariance function unbiased over tau plot
title('ACF unbiased over tau plot');


% from Lab 1 - caluclation of frequencyes
for(j=1:1:N)
      if(j==1)
        t(j)=0;
        omega(j)=0;   

      elseif(j<=N/2+1)
        t(j)=N*dt/(j-1); 
        omega(j)=2*pi/t(j); 

      elseif(j>N/2+1)
        t(j)=N*dt/(N-j+1);
        omega(j)=-2*pi/t(j);
       
      end;
      
      
 end;
spectr_dens=fft(acf);
figure
plot(t(2:N),abs(spectr_dens(2:N))) %Power Spectral Dencity over periods (not frequencies)
title('Power Spectral Dencity over periods (not frequencies)');

% 3)	Построить спектральную плотность
[spectr, freq] = spect_fftn(dates, acf);
figure
plot(freq(2:N), abs(spectr(2:N))) % линейная частота (циклов в год)
title('Power Spectral Dencity over frequencies');


%array of dates to predict
 dates_to_predict=dates(N):dt:dates(N)+0.8
 
% 4)	Подобрать полиномиальную модель (можно поменять порядок)
 % remove polynomial trend
poly=1;
deg=3;
if(poly)
figure
 [x_without_trend,poly_pred]=predict_poly(dates,x,dates_to_predict, deg);
 title('predict\_poly()');
else
    x_without_trend=x;
end;
 
%trend plotting
figure
plot(dates,x,dates,x-x_without_trend,dates_to_predict,poly_pred)
title('trend plotting');

%here we check PSD after trend was removed
 [ spectr, freq] = spect_fftn(dates, x_without_trend)
figure
 plot(freq , abs(spectr))
 title('here we check PSD after trend was removed');

%harmonic trend
% 5)	Подобрать гармоники (периоды выбрать самим)
  Periods=[2.2 2.8 3 6 8]; %ENSO contains many frequencyes
figure
  [Xsinh, harm_pred] = predict_harm(dates,x_without_trend,Periods,dates_to_predict);
 title('predict\_harm()');
figure
    plot(dates,x,dates_to_predict,harm_pred+poly_pred,'red',dates,x-Xsinh+m); 
 title('harmonic trend'); 
    
% data after harmonics have been remooved    
figure
plot(dates,Xsinh)
title('data after harmonics have been removed'); 
     
  %autoregression
%   6)	Подобрать авторегрессию – поменять порядок
   ar_order=25;
figure
  [XAR_pred,cf] = predict_ar(dates,Xsinh,dates_to_predict,ar_order);
 title('predict\_ar()');
figure
  plot(dates, Xsinh,dates_to_predict,XAR_pred)
title('autoregression'); 
 
  
 %add all prediction together
%  7)	Построить графики моделей и прогноза
 dates2=dates_to_predict;
 z=harm_pred+poly_pred+XAR_pred;
 l=size(z);

 N2=l(2);
 
 %plot and output to file
figure
 plot(dates,x,dates_to_predict,z);  
title('add all prediction together - Построить графики моделей и прогноза'); 

  foutname= sprintf('%s/ENSO_prediction.dat',path);
  fout=fopen(foutname, 'wt');
  for (j=1:1:N2)
    fprintf(fout,'%6.2f %10.8e\n',dates_to_predict(j),z(j));
  end; 
  fclose(fout);
  