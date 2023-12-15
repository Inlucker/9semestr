function [ spectr, freq] = spect_fftn(DAT, signal)
% performes spectrum analysys of signal
%DAT - equaly-sampled dateus
%signal - input (complex) signal
%   output
%spectr - array of spectrum values
%freq - corresponding frequencies


%program written by Leonid Zotov  27.04.15
N=size(signal,2);

m=signal;
tic
spectr=fftn(m);

T=(DAT(2)-DAT(1));


%spectr=zeros(1,N);

for(k=1:1:N)
      if(k==1)
        t(k)=0;
        omega(k)=0;   

      elseif(k<=N/2+1)
        t(k)=N*T/(k-1); 
        omega(k)=2*pi/t(k); 

      elseif(k>N/2+1)
        t(k)=N*T/(N-k+1);
        omega(k)=-2*pi/t(k);
       
      end;
      
      %spectr(k)=sum(exp(-i*omega(k)*DAT).*signal);
      
 end;
toc
ind=[2,floor(N/2)];
freq=circshift(omega/2/pi,ind);
spectr=circshift(spectr,ind)/N;


 