function [Xsinh, harm_pred] = predict_harm(MJDsc,Xsint,Periods,MJD_pred)
%LS-fit and prediction of harmonics
%INPUT
%MJDsc initial dates
%Xsint  initial signal
%Periods  array of preiods to use for LS
%MJD_pred  dates to predict
%OUTPUT
%Xsinh  time series after harmonic model remooved
%harm_pred - predictions according to harmonic model

[par, resX, D] = HarmFitHomo(MJDsc,Xsint',Periods,MJDsc(1))
 Xsinh=Xsint-resX';
 
 N_p=size(MJD_pred,2);
 harm_pred=zeros(1,N_p);
 harm_pred1=zeros(1,N_p);
 
 N_periods=size(Periods,2);
for(k=1:1:N_periods)
  for(j=1:1:N_p)
    harm_pred1(j)=par(2*k-1)*cos((MJD_pred(j)-MJDsc(1))*2*pi/Periods(k))+par(2*k)*sin((MJD_pred(j)-MJDsc(1))*2*pi/Periods(k));
   end;
    harm_pred=harm_pred+harm_pred1;
end;

 plot(MJDsc,Xsint, MJDsc, resX, MJDsc, Xsinh,  MJD_pred,harm_pred)
end

