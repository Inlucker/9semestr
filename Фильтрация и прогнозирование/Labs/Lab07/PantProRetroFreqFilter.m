% filtering in the chandler frequency band with Panteleev filter with or
% without inversion in prograde, retrograde, or both bands
%in the frequency domain written by L Zotov 2016
function [ out_signal ]=PantProRetroFreqFilter(year,s,f_om,fc,dt,pro,outfilename)
if (nargin==6)
    outfilename=0;
end;
% year - array of dates - could be any, but expected time steps in MJD
% s - complex array of signal to be smoothed
% f_om parameter of the filter
%fc defince center frequency of Panteleev filter
%half_width halfwidth of the filter
%dt - time step
%FC, Q - defince frequency and quality of resonance
%inv - parameter for inversion
%pro - filter in prograde 1, retrograde -1, or both bands 0
N=size(s,2);

om=2*pi*f_om;

omc=2*pi*fc;


sF=fft(s);


 for(k=1:1:N)
      if (k==1)
          t(k)=0;
          omega(k)=1; 
          PF(k)=0;
          PFr(k)=0;
           
      elseif(k<=N/2+1)
          t(k)=N*dt/(k-1);
          omega(k)=2*pi/t(k); 
        
          PF(k)=om^4/((omega(k)-omc)^4+om^4);
          PFr(k)=om^4/((omega(k)+omc)^4+om^4);
          
      elseif(k>N/2+1)
          t(k)=N*dt/(N-k+1);
          omega(k)=-2*pi/t(k);  
       
          
          PF(k)=om^4/((omega(k)-omc)^4+om^4);
          PFr(k)=om^4/((omega(k)+omc)^4+om^4);
      end;
 end;
 
 %prograde, retrograde and both filter transfer functions
if(pro==0)
 TRF=PF+PFr;
elseif(pro==1)
 TRF=PF;
elseif(pro==-1)
 TRF=PFr;
else
        print 'wrong pro'
        return;
end;



plot(omega/2/pi,abs(TRF),omega/2/pi,abs(sF)/N);
if(outfilename)
 fout=fopen(outfilename,'wt');
 for(j=1:1:N)
   fprintf(fout,'%10.8f ',1/(omega(j)/2/pi));
   fprintf(fout,'%10.8e ', abs(TRF(j))); 
   fprintf(fout,'%10.8e ', abs(sF(j))/N); 
   fprintf(fout,'\n');
 end;
 fclose(fout);
end;

resSym=sF.*TRF;

out_signal_fft=resSym;


out_signal=ifft(out_signal_fft);


 
