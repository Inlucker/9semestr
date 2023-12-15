%filtering in the chandler frequency band with Panteleev filter with or without inversion
%in the frequency domain written by L. Zotov 2016-2018
function [ out_signal ]=ChandPantFreqFilter(year,s,f_om,fc,dt,FC,Q,inv,outfilename)
% INPUT
% year - array of dates - could be any, but expected time steps in MJD
% s - complex array of signal to be smoothed
% f_om parameter of the filter
% fc defince center frequency of Panteleev filter
% half_width halfwidth of the filter
% dt - time step
% FC, Q - defince frequency and quality of resonance
% inv - parameter for inversion
% OUTPUT
% out_signal


N=size(s,2);

om=2*pi*f_om;

omc=2*pi*fc;

sigmaC=2*pi*FC*(1+i/2/Q); 
tau=i/sigmaC;

sF=fft(s);


 for(k=1:1:N)
      if (k==1)
          t(k)=0;
          omega(k)=1; 
          if (omc==0)
           TRF(k)=1;    
          else
           TRF(k)=0;
          end;
          Sym(k)=0;
          if(inv==-1)
          Sym(k)=1;
          end;
          
      elseif(k<=N/2+1)
          t(k)=N*dt/(k-1);
          omega(k)=2*pi/t(k); 
          Sym(k)=(1+tau*i*omega(k));
         
          TRF(k)=om^4/((omega(k)-omc)^4+om^4);
          
      elseif(k>N/2+1)
          t(k)=N*dt/(N-k+1);
          omega(k)=-2*pi/t(k);  
          Sym(k)=(1+tau*i*omega(k));
          TRF(k)=om^4/((omega(k)-omc)^4+om^4);
          
      end;
 end;
 
if(nargin==9)
 fout=fopen(outfilename,'wt');
 for(j=1:1:N)
   fprintf(fout,'%10.8f ',(omega(j)/2/pi));
   fprintf(fout,'%10.8e ', abs(TRF(j))); 
   fprintf(fout,'%10.8e ', abs(Sym(j))); 
   fprintf(fout,'%10.8e ', abs(sF(j))/N); 
   fprintf(fout,'\n');
 end;
 fclose(fout);
end;


resSym=sF.*TRF;

if(inv==1)
    out_signal_fft=resSym.*Sym;
elseif(inv==-1)
    out_signal_fft=resSym./Sym;
else  
    out_signal_fft=resSym;
end

out_signal=ifft(out_signal_fft);


 