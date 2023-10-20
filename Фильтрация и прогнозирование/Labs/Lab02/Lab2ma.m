% program is improved 14.02.2011 by L.V. Zotov
clear;


%--------- Part 1---------------------
% fft of artificial signal

N_signal=1024;
% generating two-sin signal
%for (k=1:1:N_signal)
%    signal(k)=sin(2*pi/10*(k-1))+sin(2*pi/100*(k-1));
%end;

Coef=0.2;

k=1:1:N_signal;
signal=.5*sin(2*pi/10*(k-1))+3*sin(2*pi/300*(k-1))+2*sin(2*pi/100*(k-1))+Coef*randn([1,N_signal]);

% try to add noise with Coef*randn([1,N_signal])

plot(signal);

%fast Fourier transform calculation

Ftrns_signal=fft(signal);

%amplitude spectrum calculation
apl_spectrum=abs(Ftrns_signal);
plot(-N_signal/2:0,apl_spectrum(N_signal/2:N_signal),1:N_signal/2,apl_spectrum(1:N_signal/2))


width=10;
h=zeros(1,N_signal);


h(N_signal/2-width:N_signal/2+width)=1/(width*2+1);
plot(h(N_signal/2-2*width:N_signal/2+2*width))

h=zeros(1,N_signal);
h(1:width)=1/(width*2+1);
h(N_signal:-1:N_signal-width)=1/(width*2+1);

Ftrns_h=fft(h);

%amplitude spectrum calculation
apl_spectrum_h=abs(Ftrns_h);

plot(apl_spectrum_h);
plot(-N_signal/2:0,apl_spectrum_h(N_signal/2:N_signal),1:N_signal/2,apl_spectrum_h(1:N_signal/2))


conv_fft=Ftrns_h.*Ftrns_signal;

conv_spectrum=abs(conv_fft);

plot(-N_signal/2:0,conv_spectrum(N_signal/2:N_signal),1:N_signal/2,conv_spectrum(1:N_signal/2))


result=ifft(conv_fft);

plot(k,signal,'--',k,result,'red')