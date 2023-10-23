clear;
clc;
close all

y = [];
res = [];

for (j = 300:100:4000)
    filename = strcat('output.Nb.T0_', int2str(j), '.txt');

    fin=fopen(filename,'rt');
    fgetl(fin);
    fgetl(fin);
    A=fscanf(fin,'%f',[4 inf]); % A - array of data
    fclose(fin);

    l=size(A);
    N=l(2);
    
    temp=A(2,1:N);

    mean_temp = mean(temp);
    y = [y j];
    res = [res mean_temp];
end

for (j = 4010:10:4090)
    filename = strcat('output.Nb.T0_', int2str(j), '.txt');

    fin=fopen(filename,'rt');
    fgetl(fin);
    fgetl(fin);
    A=fscanf(fin,'%f',[4 inf]); % A - array of data
    fclose(fin);

    l=size(A);
    N=l(2);
    
    temp=A(2,1:N);

    mean_temp = mean(temp);
    min_temp = min(temp);
    y = [y j];
    res = [res mean_temp];
end

for (j = 4100:100:6400)
    filename = strcat('output.Nb.T0_', int2str(j), '.txt');

    fin=fopen(filename,'rt');
    fgetl(fin);
    fgetl(fin);
    A=fscanf(fin,'%f',[4 inf]); % A - array of data
    fclose(fin);

    l=size(A);
    N=l(2);
    
    temp=A(2,1:N);

    mean_temp = mean(temp);
    min_temp = min(temp);
    y = [y j];
    res = [res mean_temp];
end

figure
plot(y,res);