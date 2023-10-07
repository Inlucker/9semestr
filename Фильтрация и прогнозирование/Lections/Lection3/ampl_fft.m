function [apl_spectr, omega] = ampl_fft(f)
% f - input signal
% output - ampl spectrum

    l=size(f);
    N=l(2);
    
    fourier_transform=fft(f);
    apl_spectr=abs(fourier_transform)/N;
    
    T=1;
    
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
          end
    end
end

