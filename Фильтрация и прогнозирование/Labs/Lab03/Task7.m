
clear;
clc;
close all;

coef=1;


fname=sprintf('eopc02_LOD.dat');
fin=fopen(fname,'rt');
fgetl(fin);
A=fscanf(fin,'%f',[3 inf]);% A - array of data
fclose(fin);

%determining the size of the signal
l=size(A);
N=l(2);

%selecting the rows of the Array
 YEAR=A(1,1:N);
 LOD=A(2,1:N)*coef; %initial LOD 

 
 %!!!!!!!!!  What is this rounding for?
 dt=100/365;
 
 plot(YEAR, LOD)
 
 
 
 cwt(LOD,years(dt),'amor'); %новая версия cwt