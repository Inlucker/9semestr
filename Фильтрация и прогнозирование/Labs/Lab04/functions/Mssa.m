function [ components ] = Mssa(Date, SIGNAL, N_loc,N,L, N_ev,coef,p_add,pathout,group_seq)
% this program performes mssa analysis of complex vectorial time series
%Zotov 14 October 2018
%date - array of dates
%SIGNAL - matrix of input values
%N_loc - vector signal dimentionality (number of components)
%N- size (length) of arrays
%L- lag parameter
%N_ev number of eigenvalues to calculate
%coef - complex vector of coefficients of size [N_loc] to multiply at output
%p_add - string to add to the output folder name
%pathout - folder, where to create the output
%group_seq - array which shows what SN to group into PC, if components are
%zeros - skipped

%trajectory matrix construction
K=N-L+1;
A=zeros(L*N_loc,K);
  for ii=1:1:L    
   for j=1:1:K
    for k=1:1:N_loc
     A((k-1)*L+ii,j)=SIGNAL(k,(ii+j-1));
    end;
   end;
  end;

size(A)



% ssa decomposition
[U,S,V] = svd(A);
K_max=max(L,K)
L_min=min(L,K)




% Hankelization of components
RX=zeros(N_ev,N_loc,N);
for(ii=1:1:N_ev)
 clear SX;
 clear G;
 
 G=U(:,ii)*S(ii,ii)*V(:,ii)';
 for(k=1:1:N_loc)
     Block=G((k-1)*L+1:k*L,1:K);
     RX(ii,k,:)=Hank(Block,L,N,L_min,K_max);
 end;
 
end;
 
%Hankelization of the sum of all N_sn components 
Rall=zeros(N_loc,N);
%SX=zeros(L*N_loc,K);% for svd
SX=zeros(L*N_loc,K);% for svd econom
  for(ii=1:1:N_ev)
   SX(ii,ii)=(S(ii,ii));
  end;
   G=U*SX*V';
   for(k=1:1:N_loc)
    Block=G((k-1)*L+1:k*L,1:K);
    Rall(k,:)=Hank(Block,L,N,L_min,K_max);
   end;
%sum of N_ev components

    

% creating the directory
st=sprintf('%s/ssa%s%d',pathout,p_add,L);
if (not (isdir(st)))
    mkdir(st)
end;
cd (st);

cla;  
% output of components
for (l=1:1:N_loc)
for (k=1:1:N_ev)
 fout=fopen(sprintf('ssa%02d_%02d.dat',l,k), 'wt');
 for (j=1:1:N)
  fprintf(fout,'%6f ',Date(j));  
   %fprintf(fout,'%6.2f ',MJD(j));
   fprintf(fout,'%10.8e ', real(RX(k,l,j))*coef(l)); 
   fprintf(fout,'%10.8e ', imag(RX(k,l,j))*coef(l)); 
   fprintf(fout,'\n'); 
   
 end;
   fclose(fout);
 plot_index=l+(k-1)*N_loc;
 subplot(N_ev,N_loc,plot_index);plot(Date,real(squeeze(RX(k,l,:)))*coef(l));
end;

end;
% output of grouped components
N_gr=size(group_seq,1);
N_comp=size(group_seq,2);

 components=zeros(N_loc,N_gr,N);

% grouping according to group_seq, ploting and output
clf; 
for (l=1:1:N_loc)

for (m=1:1:N_gr)
   
    label='';
  for (k=1:1:N_comp)
    ind=group_seq(m,k);
    if(ind~=0)
     components(l,m,:)=components(l,m,:)+RX(ind,l,:);
     label=strcat(label,num2str(ind));
    end;
  end;

     fout=fopen(sprintf('ssa%02d_%s.dat',l,label), 'wt');
     for (j=1:1:N)
      fprintf(fout,'%6f ',Date(j));  
      %fprintf(fout,'%6.2f ',MJD(j));
      fprintf(fout,'%10.8e ', real(components(l,m,j))*coef(l)); 
      fprintf(fout,'%10.8e ', imag(components(l,m,j))*coef(l)); 
      fprintf(fout,'\n'); 
     end;
     fclose(fout);
     %N_gr*(l-1)+m
     plot_index=l+(m-1)*N_loc;
     subplot(N_gr,N_loc,plot_index);plot(Date,real(squeeze(components(l,m,:)))*coef(l));
     legend(label);
     
  
end;

%initial signal output
 fout=fopen(sprintf('signal%02d.dat',l), 'wt');
 for (j=1:1:N)
  fprintf(fout,'%6f ',Date(j));  
   %fprintf(fout,'%6.2f ',MJD(j));
   fprintf(fout,'%10.8e ', real(SIGNAL(l,j))*coef(l)); 
   fprintf(fout,'%10.8e ', imag(SIGNAL(l,j))*coef(l)); 
   fprintf(fout,'\n'); 
 end;
   fclose(fout);

%sum of N_ev components signal output
fout=fopen(sprintf('sumall%02d.dat',l), 'wt');
 for (j=1:1:N)
  fprintf(fout,'%6f ',Date(j));  
   %fprintf(fout,'%6.2f ',MJD(j));
   fprintf(fout,'%10.8e ', real(Rall(l,j))*coef(l)); 
   fprintf(fout,'%10.8e ', imag(Rall(l,j))*coef(l)); 
   fprintf(fout,'\n'); 
 end;
   fclose(fout);
end;

    
 %eigvalues output
 fout=fopen('eigval.dat', 'wt');
 alleqigval=trace(abs(S(1:L,1:L)));
 for (j=1:1:L)
   fprintf(fout,'%6f ',j);  
   fprintf(fout,'%10.8e ', abs(S(j,j))); 
   fprintf(fout,'%10.8e ',S(j,j)^2 ); 
   fprintf(fout,'%4.2f ',100*abs(S(j,j))/alleqigval ); 
   fprintf(fout,'\n'); 
 end;
    fclose(fout);
