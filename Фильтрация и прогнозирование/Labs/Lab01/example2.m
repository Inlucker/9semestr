clear;

filename='eopc01.iau2000.1900-now.dat';

fin=fopen(filename,'rt');
fgetl(fin);
A=fscanf(fin,'%f',[11 inf]);% A - array of data
fclose(fin);

l=size(A);
N=l(2);

YEARS=A(1,1:N);
UTTAI=A(6,1:N);

dT=0.05

plot(YEARS,UTTAI)

LOD=zeros(1,N);


for(k=2:1:N-1)
 LOD(k)=-(UTTAI(k+1)-UTTAI(k-1))/2/dT;
end;

dates=YEARS(1242:N-1);
lod=LOD(1242:N-1)

plot(dates,lod)

f=lod-mean(lod);
 [apl_spectr, omega] = ampl_fft(f)

 plot(omega/2/pi,apl_spectr)
 
 