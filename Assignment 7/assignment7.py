import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
############ Part 1 #################
numerator=[1,0.5]
p=np.poly1d([1,1,2.5]) #denominator of F(s)
q=np.poly1d([1,0,2.25]) #X(s)*(s^2+2.25)
r=np.polymul(p,q)
list(r)
X=sp.lti(numerator,r)
t,x=sp.impulse(X,None,np.linspace(0,50,10001))
plt.plot(t,x)
plt.title('X vs t for input $\cos(1.5t)e^{-0.5t}u_0(t)$')
plt.xlabel('Time')
plt.ylabel('Position of spring')
plt.show()

############ Part 2 #################
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
numerator1=[1,0.05]
r=np.polymul([1,0.1,2.2525],q)
list(r)
X=sp.lti(numerator,r)
t,x=sp.impulse(X,None,np.linspace(0,100,10001))
plt.plot(t,x)
plt.title('X vs t for input $\cos(1.5t)e^{-0.05t}u_0(t)$')
plt.xlabel('Time')
plt.ylabel('Position of spring')
plt.show()

############ Part 3 #################
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import math
def unit_step(x):
    output=[]
    for element in x:
        if(element>0):
            output.append(1)
        elif element<0:
            output.append(0)
        elif element==0:
            output.append(0.5)
    return output
                
H=sp.lti=([1],[1,0,2.25])
vec=np.linspace(1.4,1.6,6)
vec=list(vec)
t=np.linspace(0,50,501)
x=[]
for i in range(1,6):
    w=vec[i-1]
    cos=[math.cos(w*element) for element in t]
    exp=[math.exp(w*element) for element in t]
    u=unit_step(t)
    x=np.multiply(np.multiply(exp,u),cos)
    t,y,svec=sp.lsim(H,x,t)
    plt.subplot(3,2,i)
    #plt.axis('equal')
    plt.plot(t,y)
    
plt.show()





########## Part 4 #############

import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
num_x=np.poly1d([1,2,0,0])
den_x=np.poly1d([1,2,1,2,2])
num_y=np.poly1d([2,0])
den_y=np.poly1d([1,2,1,2,2])
X=sp.lti(num_x,den_x)
Y=sp.lti(num_y,den_y)
wx,mag_x,phi_x=X.bode()
plt.subplot(2,1,1)
plt.semilogx(wx,mag_x)
plt.subplot(2,1,2)
plt.semilogx(wx,phi_x)
plt.show()

t,x=sp.impulse(X,None,np.linspace(0,20,101))
plt.subplot(2,1,1)
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('X,Y as function of time')
t,y=sp.impulse(Y,None,np.linspace(0,20,101))
plt.subplot(2,1,2)
plt.plot(t,y)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()


############ Part 5 ###############
# H(s)=10^12/(s^2+10^8s+10^12)

import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
H=sp.lti([10**12],[1,10**8,10**12])
w,mag,phi=H.bode()
plt.title('Magnitude of transfer function')
plt.xlabel('$\omega$')
plt.ylabel('Magnitude')
plt.semilogx(w,mag)
plt.show()
plt.title('Phase of transfer function')
plt.xlabel('$\phi$')
plt.ylabel('Phase')
plt.semilogx(w,phi)
plt.show()