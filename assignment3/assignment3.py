'''
Author:Siddharth Nayak
Roll No:EE16B073
email_id:ee16b073@smail.iitm.ac.in
'''
import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt

def exp(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))
    
#get f(x) for each x
lower_limit=-2*math.pi
upper_limit=4*math.pi
x=np.linspace(lower_limit,upper_limit,1000)
y_exp=exp(x)
y_coscos=coscos(x)

plt.semilogy(x,y_exp,label='exp(x)')
plt.title('Plot of $exp(x)$')
plt.xlabel('$x$')
plt.ylabel('$exp(x)$ on a log scale')
plt.grid()
plt.show()

plt.plot(x,y_coscos,label='cos(cos(x))')
plt.title('Plot of $cos(cos(x))$')
plt.xlabel('$x$')
plt.ylabel('$cos(cos(x))$')
plt.grid()
plt.show()

pi=math.pi
cos=False   #make True for cos(cos(x))
lower_limit=0
upper_limit=2*pi

def a0(function):
    '''
        Returns a0 for the Fourier Series
    '''
    y=quad(function,lower_limit,upper_limit)
    return ((1/(2*pi))*y[0])
    
def a_terms(n,function):
    #define f(x)*cos(nx)
    def cos(x,n,function):
        func=function(x)
        return np.cos(n*x)*func
    #pass the above defined function to quad
    integrated=quad(cos,lower_limit,upper_limit,args=(n,function))
    return integrated[0]/pi
    
def b_terms(n,function):
    def sin(x,n,function):
        func=function(x)
        return np.sin(n*x)*func
    #pass the above defined function to quad
    integrated=quad(sin,lower_limit,upper_limit,args=(n,function))
    return integrated[0]/pi
    
    
terms_exp=[]#has terms of exp(x)
terms_cos=[]#has terms of cos(cos(x))
terms_exp.append(a0(exp))
terms_cos.append(a0(coscos))


for i in range(1,26):
    function=exp
    terms_exp.append(a_terms(i,function))
    terms_exp.append(b_terms(i,function))
    function=coscos
    terms_cos.append(a_terms(i,function))
    terms_cos.append(b_terms(i,function))
    
n=[]
n.append(0)
for i in range(1,26):
    n.append(i)
    n.append(i)
    
plt.semilogy(n,np.abs(terms_cos),'ro')
plt.title('Semilog plot of $cos(cos(x))$')
plt.xlabel('n')
plt.ylabel('semilog of coefficients')
plt.show()

plt.loglog(n,np.abs(terms_cos),'ro')
plt.title('Loglog plot of $cos(cos(x))$')
plt.xlabel('$log(n)$')
plt.ylabel('log of coefficients')
plt.show()

plt.semilogy(n,np.abs(terms_exp),'ro')
plt.title('Plot of coefficients of $exp(x)$')
plt.xlabel('n')
plt.ylabel('semilog of coefficients')
plt.show()

plt.loglog(n,np.abs(terms_exp),'ro')
plt.title('Plot of coefficients of $exp(x)$')
plt.xlabel('log(n)')
plt.ylabel('Log of coefficients')
plt.show()

#least squares method for exp(x)
x=np.linspace(0,2*pi,401)
x=x[:-1] # drop last term to have a proper periodic integral b=f(x) # f has been written to take a vector
b=np.exp(x)
A_exp=np.zeros((400,51)) # allocate space for A
A_exp[:,0]=1 #col1isallones
for k in range(1,26):
        A_exp[:,2*k-1]=np.cos(k*x) # cos(kx) column
        A_exp[:,2*k]=np.sin(k*x)   # sin(kx) column
      #endfor
cl_exp=np.linalg.lstsq(A_exp,b)[0]      # the ’[0]’ is to pull out the
      # best fit vector. lstsq returns a list.
      
#least squares method for cos(cos(x))
x=np.linspace(0,2*pi,401)
x=x[:-1] # drop last term to have a proper periodic integral b=f(x) # f has been written to take a vector
b=np.cos(np.cos(x))
A_cos=np.zeros((400,51)) # allocate space for A
A_cos[:,0]=1 #col1isallones
for k in range(1,26):
        A_cos[:,2*k-1]=np.cos(k*x) # cos(kx) column
        A_cos[:,2*k]=np.sin(k*x)   # sin(kx) column
      #endfor
cl_cos=np.linalg.lstsq(A_cos,b)[0]      # the ’[0]’ is to pull out the
      # best fit vector. lstsq returns a list.
      
plt.semilogy(n,np.abs(terms_exp),'ro',label='integration')
plt.semilogy(n,np.abs(cl_exp),'gx',label='least-square')
plt.title('Coefficients of $e^x$(semilog scale)')
plt.xlabel('n')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

plt.loglog(n,np.abs(terms_exp),'ro',label='integration')
plt.loglog(n,np.abs(cl_exp),'gx',label='least-square')
plt.title('Coefficientsof $e^x$ (loglog scale)')
plt.xlabel('$log(n)$')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

plt.semilogy(n,np.abs(terms_cos),'ro',label='integration')
plt.semilogy(n,np.abs(cl_cos),'gx',label='least-square')
plt.title('Coefficients of $cos(cos(x))$ (semilog scale)')
plt.xlabel('$log(n)$')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

plt.loglog(n,np.abs(terms_cos),'ro',label='integration')
plt.loglog(n,np.abs(cl_cos),'gx',label='least-square')
plt.title('Coefficientsof $cos(cos(x))$ (loglog scale)')
plt.xlabel('$log(n)$')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

error_exp_terms=np.abs(cl_exp-terms_exp)
error_cos_terms=np.abs(cl_cos-terms_cos)

plt.loglog(n,error_exp_terms,'ro')
plt.title('Error in the Least Squares Method for $e^x$')
plt.xlabel('n')
plt.ylabel('Error')
plt.grid()
plt.show()
print(np.max(error_exp_terms))

plt.loglog(n,error_cos_terms,'ro')
plt.title('Error in the Least Squares Method for $cos(cos(x))$')
plt.xlabel('n')
plt.ylabel('Error')
plt.grid()
plt.show()
print(np.max(error_cos_terms))

c_exp=terms_exp
c_cos=terms_cos
b_exp=np.dot(A_exp,c_exp)
b_cos=np.dot(A_cos,c_cos)


plt.plot(x,b_exp)
plt.title('Plot of function obtained by summing fourier series for $e^x$')
plt.xlabel('x')
plt.ylabel('$e^x$')
plt.grid()
plt.show()

plt.plot(x,b_cos)
plt.title('Plot of function obtained by summing fourier series for $cos(cos(x))$')
plt.xlabel('x')
plt.ylabel('$cos(cos(x))$')
plt.grid()
plt.show()


                              