clear;
clc;
close all;

for (j = 4010:10:4050)
    filename = strcat('output.Nb.T0_', int2str(j), '.txt')
    
    fin=fopen(filename,'rt');
    fgetl(fin);
    fgetl(fin);
    A=fscanf(fin,'%f',[4 inf]); % A - array of data
    fclose(fin);
    
    l=size(A);
    N=l(2);
    
    steps=A(1,1:N);
    temp=A(2,1:N);
    
    test_temp = temp(1,301:N);
    mean_temp = mean(test_temp)
    squared_diff = (temp - mean_temp).^2;
    mean_squared_diff = mean(squared_diff);
    standard_deviation = sqrt(mean_squared_diff)

    figure;
    plot(50:50:max(steps),temp);
    title(j)
end