#这里尝试使用python搭建一个计算二维possion方程的求解器
#$-\Delta u=-(\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2})=f(x,y) \forall (x,y) \in\Omega$
#$u|_{\partial\Omega}=g(x,y)$
#其中$\Omega =(0,1)\times(0,1)$，$f(x,y)=,g(x,y)=$
#为此我们构造其差分格式如下：
#Step1：使用$\Delta x=\Delta y=\frac{1}{N}$的网格线划分$\Omega$得到网格$\mathbb{J}=\{(x_i,y_j)|(x_i,y_j)\in\bar{\Omega}\}$
    #$\mathbb{J}$上$u,f,g$的取值分别记为$u_{ij},f_{ij},g_{ij}$
#Step2:采用中心差分构造差分格式：
    #$\frac{\partial^2 u}{\partial x^2}|_{(x_i,y_j)}=\frac{u_{(i-1,j)}-2u_{i,j}+u_{i+1.j}}{\Delta x^2}$
    #$\frac{\partial^2 u}{\partial y^2}|_{(x_i,y_j)}=\frac{u_{(i,j-1)}-2u_{i,j}+u_{i.j+1}}{\Delta y^2}$
    #由此得到Possion方程的差分格式：$4u_{i,j}-u_{i+1.j}-\frac{u_{(i-1,j)}-u_{(i,j-1)}-u_{(i,j+1)}}{\Delta x^2}=f_{i,j} \forall (x_i,y_i)\in \Omega\backlash\partiall\Omega$
    #边界条件：$u_{ij}=g_{i,j},\forall (x_i,y_i)\in \partial\Omega$
#Step3：求解微分方程： 
    #$4u_{i+j\times N}-u_{i+j\times N-1}-\frac{u_{(i+j\times N+1)}-u_{(i+j\times N-N)}-u_{(i+j\times N+N)}}{\Delta x^2}=f_{i+j\times N} \forall (x_i,y_i)\in \Omega\backlash\partiall\Omega$
    #$u_{i+j\time N}=g_{i+j\times N},\forall (x_i,y_i)\in \partial\Omega$




import numpy as np
import numpy.matlib
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def g(x,y):
    #return cos(pi*x)**2*cos(pi*y)**2
    return 0 
def f(x,y):
    #return sin(pi*x)*sin(pi*y)
    return 1
print('正在计算中')
n=50
N=(n+1)**2
x=np.linspace(0,1,n+1, endpoint=True,dtype=float)
y=np.linspace(0,1,n+1, endpoint=True,dtype=float)
E_n=np.matlib.identity(n+1, dtype=float)
R=-1*E_n
R[0,0]=0
R[n,n]=0
K=4*np.matlib.identity(n+1, dtype=float)-np.matlib.eye(n+1,n+1,1, dtype=float)-np.matlib.eye(n+1,n+1,1, dtype=float).T
K[0,0]=1
K[0,1]=0
K[n,n]=1
K[n,n-1]=0
A=np.matlib.identity(N, dtype=float)
for i in range(0,n-1):
    A[(n+1+i*(n+1)):(2*n+2+i*(n+1)),(i*(n+1)):(n+1+i*(n+1))]=R
    A[(n+1+i*(n+1)):(2*n+2+i*(n+1)),(n+1+i*(n+1)):(2*n+2+i*(n+1))]=K
    A[(n+1+i*(n+1)):(2*n+2+i*(n+1)),(2*n+2+i*(n+1)):(3*n+3+i*(n+1))]=R

B=np.arange(N,dtype=float)
for j in range(0,n+1):
    for i in range(0,n+1):
        if j==0 or j==n or i==0 or i==n:
            B[i+j*(n+1)]=g(x[i],y[j])
        else:
            B[i+j*(n+1)]=f(x[i],y[j])/(n**2)

U=numpy.linalg.solve(A,B)
u=np.arange(N,dtype=float).reshape(n+1,n+1)
for j in range(0,n+1):
    for i in range(0,n+1):
        u[i,j]=U[i+j*(n+1)]

print(u.max())

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap='hot')
plt.show()
print('计算完成')

