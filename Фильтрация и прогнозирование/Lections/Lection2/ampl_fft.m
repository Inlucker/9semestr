function [apl_spectr, omega] = ampl_fft(f)
% f - inout signal
% output - ampl spectrum

l = size(f);
N = l(2);

f_tr = fft(f);
apl_spectr=abs(f_tr)/N;

T = 1;

for(j=1:1:N)
  if(j==1)
    t(j)=0;
    omega(j)=0;   

  elseif(j<=N/2+1)
    t(j)=N*T/(j-1); 
    omega(j)=2*pi/t(j); 

  elseif(j>N/2+1)
    t(j)=N*T/(N-j+1);
    omega(j)=-2*pi/t(j);
   
  end;
  %spectr(k)=sum(exp(-i*omega(k)*DAT).*signal);
end;

end