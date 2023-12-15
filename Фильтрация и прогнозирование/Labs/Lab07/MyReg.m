clear;
clc;
close all;

fin=fopen('eopc01.iau2000.1900-now.dat');
fgetl(fin);
fgetl(fin);
fgetl(fin);
A=fscanf(fin,'%f',[5 inf]);
fclose(fin);

dates=A(1,:);
x=A(3,:);
y=A(4,:);
m_err=A(5,:);

m_init=[x;-y];
N=size(m_init, 2);

Q = 100;
f_c = 0.843;
a = 2*pi*f_c;
b = pi*f_c/Q;

G = [[-b -a]; [a -b]];
F = -G;
C = [[1 0]; [0 1]];
C = [[1 0.9]; [0.8 1]];

svd(C)
m=C*m_init;
plot(dates, m(1,:), dates, m_init(1,:))

E = [[1 0]; [0 1]];
dt = dates(2)-dates(1);

A = ERtrM(a,b,dt);