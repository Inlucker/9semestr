

clear;
N=1024;


k=1:1:N;
omega=2*pi/100;
a=0.5
f=2*a*cos(omega*k+pi/3)+a*cos(10*omega*k);

plot(f)

dt=1
hw=10;%half-width of the window

[ Res, Start, Finish] = MoovingAverageFilter(f,hw,dt);

hold on;
plot(Start:1:Finish,Res)