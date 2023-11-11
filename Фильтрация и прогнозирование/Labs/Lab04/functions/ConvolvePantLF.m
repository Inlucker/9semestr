function [ Res, Start, Finish] = ConvolvePantLF(Ma,omw,hw,dt)
%Function performs low-frequency Panteleev filtering of a matrix
%written by L.V. Zotov in 2012, checked in 2018
%INPUT
%Ma - matrix 
%omw - panteleev filter width parameter 
%hw - half width of the window 
%dt - step between points 

%hw should be in the same units as dt


l=size(Ma);
M=l(1);
N=l(2);



a=omw/sqrt(2);


W2=round(hw/dt);%half width of window in points
W=W2*2+1;

if(N<W)
    'something wrong with signal length'
    'try transpose'
end;

t=zeros(1,W);
h=zeros(1,W);
e=zeros(1,W);
p=zeros(1,W);

for(k=1:1:W)
     t(k)=(k-W2-1)*dt;
     h(k)=a/2*exp(-a*abs(t(k)))*(cos(a*t(k))+sin(a*abs(t(k))));
end;
 h=h./sum(h);%normalization
 
 
% %output of the filter window to file
% fout=fopen('pantLFTrFunc.dat','wt');
% for(j=1:1:W)
%    fprintf(fout,'%10.8e ',t(j)/365.0);
%    fprintf(fout,'%10.8e ',real(h(j))); 
%  fprintf(fout,'\n');
% end;
% fclose(fout);
%  
 
plot(t,real(h))

 j=1:1:W;
 Res=zeros(M,N-W);
 %convolution exchanged by scalar product
 for(k=1:1:N-W)
    Res(:,k)=Ma(:,k+W-j)*(h');
 end;
 
 Start=W2+1;
 Finish=N-W+W2;
 
end

