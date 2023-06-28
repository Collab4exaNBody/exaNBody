import matplotlib.pyplot as plt
import numpy as np
# fit knots
x=[0,1,2,3,4,5,6,7]
y=[1.5,2,1,0.5,4,3,1,1]
Px=np.concatenate(([0],x,[0]))
Py=np.concatenate(([0],y,[0]))
# interpolation equations
n=len(x)
phi=np.zeros((n+2,n+2))
for i in range(n):
    phi[i+1,i]=1
    phi[i+1,i+1]=4
    phi[i+1,i+2]=1
# end condition constraints 
phi[0,0]=-3
phi[0,2]=3
phi[0,n-1]=3
phi[0,n+1]=-3
phi[n+1,0]=6
phi[n+1,1]=-12
phi[n+1,2]=6
phi[n+1,n-1]=-6
phi[n+1,n]=12
phi[n+1,n+1]=-6
# passage matrix
phi_inv = np.linalg.inv(phi)
# control points
Qx=6*phi_inv.dot(Px)
Qy=6*phi_inv.dot(Py)
# figure plot
plt.figure(figsize=(12, 5))
t=np.linspace(0,1,num=101)
for k in range(0,n-1):
    x_t=1.0/6.0*(((1-t)**3)*Qx[k]+(3*t**3-6*t**2+4)*Qx[k+1]+(-3*t**3+3*t**2+3*t+1)*Qx[k+2]+(t**3)*Qx[k+3])
    y_t=1.0/6.0*(((1-t)**3)*Qy[k]+(3*t**3-6*t**2+4)*Qy[k+1]+(-3*t**3+3*t**2+3*t+1)*Qy[k+2]+(t**3)*Qy[k+3])  
    plt.plot(x_t,y_t,'k',linewidth=2.0,color='orange')
plt.plot(x, y, 'ko', label='fit knots',markersize=15.0)
plt.plot(Qx, Qy, 'o--', label='control points',markersize=15.0)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left', ncol=2)
plt.savefig('cubic_spline_end-to-end_tangent_continuity.png')
plt.show()

