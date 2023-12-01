function [XARpr,cf] = predict_ar(MJDsc,Xin,MJD_pred,ar_order)
%this function models auto regression and predicst next values
%INPUT
%MJDsc time steps
%Xin data
%MJD_pred dates to predict
%ar_order  order of autoregression
%OUTPUT
%XARpr - prediction
%cf - coefficients

 N_p=size(MJD_pred,2);
 
 mn=mean(Xin,2);
 Xsinh=Xin-mn;

 %autocovariance function modelling
 [cf,lags]=xcorr( Xsinh, Xsinh);
 
% plot(lags,cf)
 
 %[ sp, Freq] = spect_hand(lags, cf);
 % plot(1./(Freq), abs(sp))

 N1=size(MJDsc,2);

 % ar parameters determination
 [arm,ref1] = ar(Xsinh,ar_order,'burg','now') ;
 %bode(arm)
 std(Xsinh)

 INP=fliplr(Xsinh(N1-ar_order+1:N1))'
 whitenoise=sqrt(arm.NoiseVariance).*randn(1,N_p);
 %plot(whitenoise)

 % N_p ponts prediction
 XAR_pred=zeros(1,N_p);

 for(j=1:1:N_p)
  XAR_pred(j)=-arm.a(2:ar_order+1)*INP+whitenoise(j);
  INP=circshift(INP,1);
  INP(1)=XAR_pred(j);
 end;
  
  XARpr=XAR_pred+mn;
 
  plot(MJDsc,Xin,MJD_pred,XARpr,'red')
end

