function [ Res, Start, Finish] = MoovingAverageFilter(signal,hw,dt)
%Function performs mooving average of a matrix
%written by L.V. Zotov in 2012
%signal - matrix with signal
%hw - half width of the window
%dt - step between points 
%hw should be in the same units as dt


l=size(signal);
M=l(1);
N=l(2);

%a=om/sqrt(2);


W2=hw;%half width of window in points
W=W2*2;

t=zeros(1,W);
h=zeros(1,W);
e=zeros(1,W);
p=zeros(1,W);

for(k=1:1:W)
     t(k)=(k-W2-1)*dt;
     h(k)=1/W;
     %convolution exchanged by scalar product, sign of exp changed
     %filter transfer function is centered at W(omega-omc)
     
end;
 h=h./sum(h);%normalization
 
 
% %output of the filter window to file
% fout=fopen('rectangleFLT.dat','wt');
% for(j=1:1:W)
%    fprintf(fout,'%10.8e ',t(j)/365.0);
%    fprintf(fout,'%10.8e ',real(h(j))); 
%  fprintf(fout,'\n');
% end;
% fclose(fout);
%  
 
%plot(t,real(h),'-+')

 j=1:1:W;
 Res=zeros(M,N-W+1);
 %convolution exchanged by scalar product
 for(k=1:1:N-W+1)
    Res(:,k)=signal(:,k+W-j)*h';
 end;
 
 Start=W2;
 Finish=N-W2;
 
end

