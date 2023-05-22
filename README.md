# possion
这里尝试使用python搭建一个计算二维possion方程的求解器

$-\Delta u=-(\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2})=f(x,y) \forall (x,y) \in\Omega$

$u|_{\partial\Omega}=g(x,y)$

其中

$\Omega =(0,1)\times(0,1)$，

$f(x,y)=1,g(x,y)=0$

为此我们构造其差分格式如下

Step1：使用

$\Delta x=\Delta y=\frac{1}{N}$

的网格线划分

$\Omega$

得到网格

$\mathbb{J}=\{(x_i,y_j)|(x_i,y_j)\in\bar{\Omega}\}$

$\mathbb{J}$上$u,f,g$的取值分别记为$u_{ij},f_{ij},g_{ij}$

Step2:采用中心差分构造差分格式

$\frac{\partial^2 u}{\partial x^2}|_{(x_i,y_j)}=\frac{u_{(i-1,j)}-2u_{i,j}+u_{i+1.j}}{\Delta x^2}$
    
$\frac{\partial^2 u}{\partial y^2}|_{(x_i,y_j)}=\frac{u_{(i,j-1)}-2u_{i,j}+u_{i.j+1}}{\Delta y^2}$
    
由此得到Possion方程的差分格式

$\{(4 u_{(i,j)}-u_{(i+1,j)}-u_{(i-1,j)}-u_{(i,j-1)}-u_{(i,j+1)}\}=f_{i,j}\times\Delta x^2 \forall (x_i,y_i)\in \Omega |\partial\Omega$
    
边界条件：

$u_{ij}=g_{i,j},\forall (x_i,y_i)\in \partial\Omega$

Step3：求解线性方程组     

$\{4u_{i+j\times N}-u_{i+j\times N-1}-u_{(i+j\times N+1)}-u_{(i+j\times N-N)}-u_{(i+j\times N+N)}\}=f_{i+j\times N}\times\Delta x^2 \forall (x_i,y_i)\in \Omega|\partial\Omega$
    
$u_{i+j\times N}=g_{i+j\times N},\forall (x_i,y_i)\in \partial\Omega$
