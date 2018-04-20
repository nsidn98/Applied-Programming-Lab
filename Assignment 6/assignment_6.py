from pylab import *
import sys

if len(sys.argv)==6:
    n=sys.argv[0]  # spatial grid size.
    M=sys.argv[1]    # number of electrons injected per turn.
    nk=sys.argv[2] # number of turns to simulate.
    u0=sys.argv[3]  # threshold velocity.
    p=sys.argv[4] # probability that ionization will occur
    Msig=sys.argv[5] #deviation of elctrons injected per turn
    params=[n,M,nk,u0,p,Msig]
else:
    params=[[100,5,500,5,0.25,2],[100,5,500,5,0.50,2],
        [100,5,500,5,1,2],[100,5,500,10,0.25,2],
        [100,5,500,10,0.50,2],[100,5,500,10,1,2]]
        
def loop(params,quadratic):
    n=params[0]
    M=params[1]
    nk=params[2]
    u0=params[3]
    p=params[4]
    Msig=params[5]
    xx=np.zeros((n*M))  # electron position
    u=np.zeros((n*M))   #electron velocity
    dx=np.zeros((n*M))  #displacement in current turn

    I=[]
    V=[]
    X=[]
    
    for i in range(1,nk):
        ii=where(xx>0) #get the indices of positions greater than zero
        dx[ii]=u[ii]+0.5 #increase the displacement
        xx[ii]+=dx[ii] #increase the position
        u[ii]+=1 #increase the velocity
        reached=where(xx[ii]>n)#contains the indices
        #set position,velocities,displacements to zero
        xx[ii[0][reached]]=u[ii[0][reached]]=dx[ii[0][reached]]=0
        kk=where(u>=u0)
        ll=where(rand(len(kk[0]))<=p)
        kl=kk[0][ll]#contains the indices
        #of energetic electrons that suffer collision
        u[kl]=0 # reset the velocity after collision
        if quadratic==False:
            rho=rand(len(kl)) #get random number
        if quadratic==True:
            rho=power(0.5,len(kl)) # a quadratic probability distribution
        xx[kl]=xx[kl]-dx[kl]*rho #get the actual value of x where it collides
        I.extend(xx[kl].tolist())
        m=int(rand()*Msig+M) #get the (random)number of new electrons to be added
        empty=where(xx==0) #get empty spaces where electrons can be injected
        nv=(min(n*M-len(empty),m)) #if no empty spaces are left
        xx[empty[:nv]]=1 #inject the new electrons
        u[empty[0][:nv]]=0 #with velocity zero
        dx[empty[0][:nv]]=0 #and displacement zero
        X.extend(xx.tolist())
        V.extend(u.tolist())
    return X,V,I

def plot_no_of_elec(X,u0,p):
#plot the number of electrons vs x
    figure(1)
    hist(X,bins=np.arange(0,101,0.5),rwidth=0.8,color='g')
    title('Number of Electrons vs $x$ with $u_0=$%f and p=%f'%(u0,p))
    xlabel('$x$')
    ylabel('Number of electrons')
    show()

def plot_intensity_map(I,u0,p):
#plot the intensity map
    histogram_=hist(I,bins=np.arange(0,101,1),rwidth=0.8,color='r')
    x=histogram_[1][1:]
    y=histogram_[0]
    fig, (ax) = plt.subplots(nrows=1, sharex=True)
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    intensity=ax.imshow(y[np.newaxis,:], cmap="gray", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    plt.title('Intensity map with $u_0=$%f and p=%f'%(u0,p))
    plt.xlabel('$x$')
    plt.colorbar(intensity)
    plt.tight_layout()
    show()
    
def plot_intensity(X,V,u0,p):
    #plot the histogram of intensity
    figure(0)
    histogram=hist(I,bins=np.arange(0,101,0.5),rwidth=0.8,color='r')
    title('Intensity histogram with $u_0=$%f and p=%f'%(u0,p))
    xlabel('$x$')
    ylabel('Intensity')
    show()
    return histogram

def plot_phase(X,V,u0,p):
#plot the phase space
    figure(2)
    plt.plot(X,V,'bo')
    title('Electron Phase Space with $u_0=$%f and p=%f'%(u0,p))
    xlabel('$x$')
    ylabel('Velocity-$v$')
    show()
    
def plot_phase_2D(X,V,I,u0,p):
#plot the phase space
    figure(2)
    plt.plot(X,V,'bo')
    title('Electron Phase Space with $u_0=$%f and p=%f'%(u0,p))
    xlabel('$x$')
    ylabel('Velocity-$v$')
    show()
    
param=[100,5,500,5,0.25,2]
X,V,I=loop(param,False)
histo=plot_intensity(X,V,param[3],param[4])
plot_no_of_elec(X,param[3],param[4])
plot_phase(X,V,param[3],param[4])
plot_intensity_map(I,param[3],param[4])

param=[100,5,500,5,0.25,2]
X,V,I=loop(param,True)
plot_intensity(X,V,param[3],param[4])
plot_no_of_elec(X,param[3],param[4])
plot_phase(X,V,param[3],param[4])
plot_intensity_map(I,param[3],param[4])

for param in params:
    X,V,I=loop(param,False)
    plot_intensity(X,V,param[3],param[4])
    plot_no_of_elec(X,param[3],param[4])
    plot_phase(X,V,param[3],param[4])
    plot_intensity_map(I,param[3],param[4])