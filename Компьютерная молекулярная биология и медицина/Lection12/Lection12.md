# Лекция 12 (06.12.2023)

## Метод молекулярной динамики

$\Large m_i \vec{a}_i=\vec{F}_i=-\frac{\partial V}{\partial \vec{r}}$ - уравнение Ньютона

Задача МД - зная координаты и скорости в начальный момент времени, рассчитать координаты и скорости в любой момент времени

$\Large \begin{aligned} & \Delta t \\ & \vec{r}(t) \rightarrow \vec{r}(t+\Delta t) \\ & \vec{v}(t) \rightarrow \vec{v}(t+\Delta t)\end{aligned}$

$\Large r(t+\Delta t)=r(t)+\frac{d r}{d t} \Delta t+\frac{1}{2} \frac{d^2 r}{d t^2} \Delta t^2+\ldots \theta\left(\Delta t^3\right)$

![](20231206_183237.jpeg)

### Алгоритм Верле

![](Снимок экрана 2023-12-06 184051.png)
![](Снимок экрана 2023-12-06 190335.png)

$\Large r(t+\Delta t)=r(t)+V_{\text {ave }} \cdot \Delta t=T(t)+V\left(t+\frac{\Delta t}{2}\right) \Delta t$

$\Large V\left(t+\frac{\Delta t}{2}\right)=V\left(t-\frac{\Delta t}{2}\right)+a \Delta t$

$\Large a=\frac{F}{m}=-\frac{1}{m} \frac{\partial v}{\partial r}$

$\Large +\left(\begin{array}{l}r(t+\Delta t)=r(t)+\frac{d r}{d t} \Delta t+\frac{1}{2} \frac{d^2 r}{d t^2} \Delta t^2 \\ r(t-\Delta t)=r(t)-\frac{d^2}{d t} \Delta t+\frac{1}{2} \frac{d^2 r}{d t^2} \Delta t^2\end{array}\right.$



![](20231206_184557.jpeg)

![](20231206_192545(1).jpeg)

Какой $\Delta t$ взять? (Порядка $10^{-15}$ с [фента секунд] - тогда корректно учтем самые быстрые изменения, колебания)

![](20231206_190355.jpeg)

Откуда взять начальные скорости? Распределение Максвелла:

![](Снимок экрана 2023-12-06 192402.png)

![](20231206_192545.jpeg)

Отслеживаем с какого момента система выходит на равновесие:

![](20231206_201039.jpeg)

### Алгоритм МД

![](Снимок экрана 2023-12-06 201236.png)

![](20231206_201959.jpeg)

![](20231206_203458.jpeg)

![](Снимок экрана 2023-12-06 205345.png)
