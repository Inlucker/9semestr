clear;
clc;
close all;

% CRYSTAL
filename = 'Nb.crystal.T0_2634.txt';

fin=fopen(filename,'rt');
fgetl(fin);
fgetl(fin);
A=fscanf(fin,'%f',[4 inf]);% A - array of data
fclose(fin);

l=size(A);
N=l(2);

steps=A(1,1:N);
temp=A(2,1:N);
dencity=A(3,1:N);
entalpia=A(4,1:N);

figure;
plot(50:50:max(steps),dencity);
title(strcat('dencity crystal'))
mean_dencity_crystal = mean(dencity)

figure;
plot(50:50:max(steps),entalpia);
title(strcat('entalpia crystal'))
mean_entalpia_crystal = mean(entalpia)

%LIQUID
filename = 'Nb.liquid.T0_2634.txt';

fin=fopen(filename,'rt');
fgetl(fin);
fgetl(fin);
A=fscanf(fin,'%f',[4 inf]);% A - array of data
fclose(fin);

l=size(A);
N=l(2);

steps=A(1,1:N);
temp=A(2,1:N);
dencity=A(3,1:N);
entalpia=A(4,1:N);

figure;
plot(50:50:max(steps),dencity);
title(strcat('dencity liquid'))
mean_dencity_liquid = mean(dencity)

figure;
plot(50:50:max(steps),entalpia);
title(strcat('entalpia liquid'))
mean_entalpia_liquid = mean(entalpia)

dencity_diff = abs(mean_dencity_liquid-mean_dencity_crystal)
entalpia_diff = abs(mean_entalpia_liquid-mean_entalpia_crystal)
teplota_plavleniya = entalpia_diff * 96.485
