%hankelization Ver 1.1 for M singular spectral analysis
% program is written 17.02.2009 by L.V. Zotov
% error remooved 01.10.2010

function X=Hank(G1,L,N,L_min, K_max)
if L~=L_min X=0; return;
end; 

for ii=1:1:L_min  
  X1(ii)=0;
  X1(N-ii+1)=0;
  for j=0:1:ii-1
   X1(ii)=X1(ii)+G1(ii-j,j+1);
   X1(N-ii+1)=X1(N-ii+1)+G1(L_min-(ii-j)+1,K_max-j);
  end;
  X1(ii)=X1(ii)/ii;
  X1(N-ii+1)=X1(N-ii+1)/ii;
end;

for ii=(L_min+1):1:K_max
  X1(ii)=0;
  for j=0:1:L_min-1
   %if(L_min==L)
    X1(ii)=X1(ii)+G1(L_min-j,(ii-L_min)+j+1);
   %else
   % X1(ii)=X1(ii)+G1(ii-j,j+1);
   %end;
  end;
  X1(ii)=X1(ii)/L_min;
end;



X=X1;