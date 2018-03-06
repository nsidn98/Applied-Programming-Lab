import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special as sp

x=np.linspace(0.5,20,40)

def method_1(x,x0):
    A_arr=[]
    x_new=x[int(np.where(x==x0)[0]):]
    J=sp.jv(1,x_new)
    for element in x_new:
        cos=math.cos(element)
        sin=math.sin(element)
        A_arr.append([cos,sin])
    return A_arr,J
    
AB1_array=[]  # will contain the A and B values
for x_0 in x:
    if x_0<=18:
        A,J=method_1(x,x_0)
        s=np.linalg.inv(np.dot(np.array(A).T,A))
        cl=(np.dot(s,np.dot(np.array(A).T,J)))
        #cl=np.linalg.lstsq(A,J)[0]
        AB1_array.append(cl)
        
def calculate_v(AB_array):
    v_arr=[]
    for elements in AB_array:
        phi=np.arccos(elements[0]/math.sqrt(elements[0]**2+elements[1]**2))
        v=(phi-np.pi/4)*2/np.pi
        v_arr.append(v)
    return v_arr
    
v_arr1=calculate_v(AB1_array) # has values of v for different x0
x_0_arr=np.linspace(0.5,18,36)

def method_2(x,x0):
    '''
    returns the arrays A and J in the equation Ax=J
    '''
    A_arr=[]
    x_new=x[int(np.where(x==x0)[0]):]
    J=sp.jv(1,x_new)
    for element in x_new:
        cos=math.cos(element)/math.sqrt(element)
        sin=math.sin(element)/math.sqrt(element)
        A_arr.append([cos,sin])
        
    return A_arr,J
    
AB2_array=[]
for x_0 in x:
    if x_0<=18:
        A,J=method_2(x,x_0)
        A=np.array(A)
        s=np.linalg.inv(np.dot(A.T,A))
        cl=(np.dot(s,np.dot(A.T,J)))
        AB2_array.append(cl)
        
A_arr=[]
for element in x:
    cos=math.cos(element)
    sin=math.sin(element)
    A_arr.append([cos,sin])
J=sp.jv(1,x)
cl=np.linalg.lstsq(A_arr,J)[0]
a=cl[0]*np.cos(x)
b=cl[1]*np.sin(x)
c=a+b
plt.plot(x,sp.jv(1,x),'ro',label='Jv')
plt.plot(x,c,'gx',label='lstsq') #obtained function by method 1
plt.title('Plot of function obtained by method 1')
plt.xlabel('$x$')
plt.ylabel('$J_v(x)$')
plt.legend()
plt.grid()
plt.show()

A_arr=[]
for element in x:
    cos=math.cos(element)/math.sqrt(element)
    sin=math.sin(element)/math.sqrt(element)
    A_arr.append([cos,sin])
J=sp.jv(1,x)
cl=np.linalg.lstsq(A_arr,J)[0]
a=cl[0]*np.cos(x)/np.sqrt(x)
b=cl[1]*np.sin(x)/np.sqrt(x)
c=a+b
plt.plot(x,sp.jv(1,x),'ro',label='Jv')
plt.plot(x,c,'gx',label='lstsq') #obtained function by method 2
plt.title('Plot of function obtained by method 2')
plt.xlabel('$x$')
plt.ylabel('$J_v(x)$')
plt.legend()
plt.grid()
plt.show()

v_arr2=calculate_v(AB2_array) # has values of v for different x0

plt.plot(x_0_arr,v_arr2,'bo',label='model 2')
plt.plot(x_0_arr,v_arr1,'go',label='model 1')
plt.title('Plot of $v$ vs $x_0$')
plt.xlabel('$x_0$')
plt.ylabel('$v$')
plt.legend()
plt.grid()
plt.show()

def calcnu(x,x0,eps,model):
    AB_array=[]
    for x_0 in x:
        if x_0<=x0:
            A,J=model(x,x_0)
            noise=eps*np.random.randn(len(J))
            J+=noise #add noise to the function
            A=np.array(A)
            s=np.linalg.inv(np.dot(A.T,A))
            cl=(np.dot(s,np.dot(A.T,J)))
            #cl=np.linalg.lstsq(A,J)[0]  #change to least square
            AB_array.append(cl)
    return AB_array
    
AB3_array=calcnu(x,18,0.01,method_1) # has values of A and B for model 1 with noise
AB4_array=calcnu(x,18,0.01,method_2) # has values of A and B for model 2 with noise
v_arr3=calculate_v(AB3_array)
v_arr4=calculate_v(AB4_array)

AB3_array=calcnu(x,18,0.01,method_1) # has values of A and B for model 1 with noise
AB4_array=calcnu(x,18,0.01,method_2) # has values of A and B for model 2 with noise
v_arr3=calculate_v(AB3_array)
v_arr4=calculate_v(AB4_array)
plt.plot(x_0_arr,v_arr3,'bo',label='method 1 (noise=0.01)')
plt.plot(x_0_arr,v_arr4,'ro',label='method 2 (noise=0.01)')
plt.title('Plot of $v$ vs $x_0$ with noise=0.01')
plt.xlabel('$x_0$')
plt.ylabel('$v$')
plt.legend()
plt.grid()
plt.show()

plt.plot(x_0_arr,v_arr2,'bo',label='model 2')
plt.plot(x_0_arr,v_arr1,'go',label='model 1')
#plt.plot(x_0_arr,v_arr3,'bo',label='method 1 (noise=0.01)')
plt.plot(x_0_arr,v_arr4,'ro',label='method 2 (noise=0.01)')
plt.title('Plot of $v$ vs $x_0$ ')
plt.xlabel('$x_0$')
plt.ylabel('$v$')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#plotting values of v vs x_0 for different values of epsilon
epsilon=[0,0.001,0.01,0.1,1]
def plot(x,x_0,epsilon,method,method_number):
    i=0
    for eps in epsilon:
        AB1_array=calcnu(x,x_0,eps,method) # has values of A and B for model 1 with noise
        v_arr1=calculate_v(AB1_array)
        #plt.subplot(5,2,i+1)
        plt.plot(x_0_arr[:-(int(x_0/10))],v_arr1[:-(int(x_0/10))],'bo',label='method %d (noise=%f)'%(method_number,eps))
        plt.title('Plot of $v$ vs $x_0$ with noise=%f '%eps)
        plt.xlabel('$x_0$')
        plt.ylabel('$v$')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()


plot(x_0_arr,18,epsilon,method_1,1)
plot(x_0_arr,18,epsilon,method_2,2)

sizes=[10,50,100,250,500,1000,10000]
for size in sizes:
    x_array=np.linspace(0.5,20,size)
    AB1_array=calcnu(x_array,20,0.01,method_2) # has values of A and B for model 1 with noise
    v_arr1=calculate_v(AB1_array)
    plt.plot(x_array[:-(int(x_0/10))],v_arr1[:-(int(x_0/10))],'ro',label='method 2')
    plt.title('Plot of $v$ vs $x_0$ with number of measurements=%d'%size)
    plt.xlabel('$x_0$')
    plt.ylabel('$v$')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
                    