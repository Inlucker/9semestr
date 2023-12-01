clear;
path='./';


% file reading, we will analize El Nino index
filename=sprintf('%s/ENSO_BEST_index.dat',path);

 fin=fopen(filename,'rt');
 Adat=fscanf(fin,'%f',[3 inf]);
 fclose(fin);

 dates=Adat(1,:); % time column
 x=Adat(2,:);  %data column
 l=size(Adat);

 N=l(2);
 
dt=1/12;%a month step 
plot(dates,x); % to check reading 

m=mean(x);
signal_centered=x-m;
 
for(tau=1:1:N)
 acf(tau)=0;
 for(j=1:1:N-tau)
    acf(tau)=acf(tau)+signal_centered(j)*signal_centered(j+tau-1);
 end;
  acf(tau)= acf(tau)/(N);
end;

plot(0:dt:(N-1)*dt,acf); %autocovariance function over tau plot


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
plot(t,abs((spectr_dens))) %Power Spectral Dencity over periods (not frequencies)



%array of dates to predict
 dates_to_predict=dates(N):dt:dates(N)+0.8
 
 % remove polynomial trend
poly=1;
deg=3;
if(poly)
 [x_without_trend,poly_pred]=predict_poly(dates,x,dates_to_predict, deg);
else
    x_without_trend=x;
end;
 
%trend plotting
plot(dates,x,dates,x-x_without_trend,dates_to_predict,poly_pred)

%here we check PSD after trend was removed
 [ spectr, freq] = spect_fftn(dates, x_without_trend)
 plot(freq , abs(spectr))

%harmonic trend
  Periods=[2.2 2.8 3 6 8]; %ENSO contains many frequencyes
  [Xsinh, harm_pred] = predict_harm(dates,x_without_trend,Periods,dates_to_predict);
    plot(dates,x,dates_to_predict,harm_pred+poly_pred,'red',dates,x-Xsinh+m);  
    
% data after harmonics have been remooved    
plot(dates,Xsinh)
     
  %autoregression
   ar_order=25;
  [XAR_pred,cf] = predict_ar(dates,Xsinh,dates_to_predict,ar_order);
  plot(dates, Xsinh,dates_to_predict,XAR_pred)
 
  
 %add all prediction together
 dates2=dates_to_predict;
 z=harm_pred+poly_pred+XAR_pred;
 l=size(z);

 N2=l(2);
 
 %plot and output to file
 plot(dates,x,dates_to_predict,z);  

  foutname= sprintf('%s/ENSO_prediction.dat',path);
  fout=fopen(foutname, 'wt');
  for (j=1:1:N2)
    fprintf(fout,'%6.2f %10.8e\n',dates_to_predict(j),z(j));
  end; 
  fclose(fout);
  