clear;
N=1024;


k=1:1:N;
omega=2*pi/100;
a=0.5
f=a*cos(omega*k+pi/3);

plot(k,f)

 [apl_spectr, omega] = ampl_fft(f)

 plot(omega/2/pi,apl_spectr)
 
 