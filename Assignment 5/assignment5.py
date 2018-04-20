'''
Author: Siddharth Nayak
email:ee16b073@smail.iitm.ac.in
Refer the Jupyter Notebook for comments
'''

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3

########### Defining some parameters ###########
scale=1 #cm
Nx=25
Ny=25
radius=0.35 # in cm
Niter=1500
phi=np.zeros((Nx,Ny))
iter_arr=np.linspace(0,Niter,Niter)
limit=scale/2

x=np.linspace(-limit,limit,Nx)
y=np.linspace(-limit,limit,Nx)
Y,X=meshgrid(y,x) # get the co-ordinates of the grid
ii=where(X**2+Y**2<=radius**2)
phi[ii]=1.0  #update the potential

x=arange(0,Nx)   # create x and y axes
y=arange(0,Ny)
plt.contour(phi)
plt.plot(y[ii[0]],x[ii[1]],'ro')
plt.title('Contour plot of the potential')
plt.grid()
plt.show()

error=np.zeros((Niter,1))
for k in range(Niter):
    oldphi=phi.copy()
    phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])#update the potential
    phi[1:-1,0]=phi[1:-1,1] #update left column
    phi[1:-1,-1]=phi[1:-1,-2] #update right column
    phi[0,:]=phi[1,:] #update the topmost row
    phi[ii]=1.0
    error[k]=(abs(oldphi-phi)).max()
    
fig1=figure(4)     # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
x=arange(0,Nx)   # create x and y axes
y=arange(0,Ny)
X,Y=meshgrid(x,y)  # creates arrays out of x and y
title('The 3-D surface plot of the potential')
surf = ax.plot_surface(X, Y, phi.T, rstride=1, cstride=1,cmap='jet')
plt.show()

x=np.linspace(limit,-limit,Nx)
y=np.linspace(limit,-limit,Nx)
Y,X=meshgrid(y,x) # get the co-ordinates of the grid
Cs = plt.contour(X,Y,phi.T)
plt.clabel(Cs, fontsize=10)
plt.plot(y[ii[0]],x[ii[1]],'ro')
plt.title('Contour plot of the potential')
plt.show()

iter_arr=np.linspace(0,Niter,Niter)
semilogy(iter_arr[0:-1:50],error[0:-1:50],'ro',label='error')
title('Error vs number of iterations')
ylabel('$log(error)$')
xlabel('iterations')
grid()
show()

A=np.zeros((int(Niter/30),2))
iter_arr1=np.linspace(1,Niter+1,1500)
A[:,0]=1
A[:,1]=(iter_arr1[0:-1:30])
terms=lstsq(A,np.log(error[::30]))[0]#least squares

logA=terms[0]
b=terms[1]
log_error=logA+b*(iter_arr1)
error1=np.exp(logA)*np.exp(b*(iter_arr1))


plt.semilogy(iter_arr[0:-1:50],error[0:-1:50],'ro',label='error')
plt.semilogy(iter_arr[0:-1:50],error1[0:-1:50],'bx',label='fit 1')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('$log(error)$')
plt.grid()
plt.title('Plot of error vs number of iterations')
plt.show()

x=np.linspace(-limit,limit,Nx)
y=np.linspace(-limit,limit,Nx)
Y,X=meshgrid(y,x)
Jx=np.zeros((Nx,Ny))
Jy=np.zeros((Nx,Ny))
Jx[1:-1,1:-1]=0.5*(phi[1:-1,0:-2]-phi[1:-1,2:])
Jy[1:-1,1:-1]=0.5*(phi[0:-2,1:-1]-phi[2:,1:-1])

    
plt.quiver(y,x,Jx[::-1,:],-Jy[::-1,:],scale=8)
plt.plot(y[ii[0]],x[ii[1]],'ro')
plt.title("Current plot with magnitudes scaled to 1/8 th of it's value")
plt.show()

dx=1
k=1    # thermal conductivity of copper
sigma=1  #electrical conductivity of copper
Jx_sq=Jx[1:-1,1:-1]**2
Jy_sq=Jy[1:-1,1:-1]**2
J_sq=Jx_sq+Jy_sq
constant_term= (J_sq*(dx**2))/(sigma*k) #source term


# initialize the temperature matrix
T=np.zeros((Nx,Ny))
T[ii]=300
T[-1,:]=300


for k in range(Niter):
    T[1:-1,1:-1]=0.25*((T[1:-1,0:-2]+T[1:-1,2:]+T[0:-2,1:-1]+T[2:,1:-1])+constant_term)
    T[1:-1,0]=T[1:-1,1] #update left column
    T[1:-1,-1]=T[1:-1,-2] #update right column
    T[0,:]=T[1,:] #update the topmost row
    T[ii]=300
    
fig1=figure(4)     # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
x=arange(0,Nx)   # create x and y axes
y=arange(0,Ny)
X,Y=meshgrid(x,y)  # creates arrays out of x and y
title('The 3-D surface plot of the Temperature')
surf = ax.plot_surface(X, Y, T.T, rstride=1, cstride=1,cmap='jet')
plt.show()

        