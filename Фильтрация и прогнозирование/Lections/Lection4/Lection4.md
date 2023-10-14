# Лекция 4 (14.10.2023)

Фурье анализ в скользящем окне

$\Large \int{y f(\omega,t)}=\int_{-\infty}^{+\infty}{f(\tau)g(\tau-t)e^{-i\omega\tau} d\tau}$
оконное преобразование

$\Large =f*g_{\omega,t}$
$\Large g_{\omega,t}=g(\tau-t)e^{i\omega\tau}$



Базис $e_1 e_2 e_3$
$\Large w=xe_2+ye_3$

$\Large v=e_1+e_2+e_3$

$\Large o_1=e_1$

$\Large o_2=-\frac{e_1}{2}+\frac{\sqrt{3}}{2}e_2$

$\Large o_3=-\frac{e_1}{2}-\frac{\sqrt{3}}{2}e_2$

$\Large \sum_{i=1}^{3}{(p_i \cdot \omega_i)}=\frac{3}{2}||\psi||^2$

Вейвлет отображение

$\Large W_\psi f(a,b)=\frac{1}{\sqrt{|a|}}\int_{-\infty}^{+\infty}{f(t)\psi (\frac{t-b}{a}) dt}$

Обратное отображению (Вейвлет отображение) по Вейвлет преобразованию 

$\Large f(t)=\frac{1}{C_\psi}\int_{0}^{a_0}\int_{-\infty}^{+\infty}{\frac{da\cdot db}{a^2}}=W_\psi(a,b)\psi_{a,b}$



$\Large M_{...} M =(x_0+1,y_0-2,z_0+3)$

$\Large f(t)=\frac{1}{C_\psi}\int_{0}^{a_0}{\frac{1}{a^2}}W_\psi f(a,b) \psi_{a,b}da \cdot db + \frac{1}{C_\psi a_0}W_\phi f(a_0,b)*\phi_{a_0,b}$

