'''
Author : Siddharth Nayak EE16B073
Date: 24th January 2018
email: ee16b073@smail.iitm.ac.in
This contains the code for the assignment 2 for the EE2703 assignment
Some parts in the code have been repeated for just reference
eg: some constants have been repeated for clarity in that part of the code

'''
#import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tabulate import tabulate

########## Part 1 ###########
def function(t):
    return(1./(1+np.multiply(t,t))) #element wise multiplication of an array
    
########## Part 2 ###########
vector_x = np.linspace(0,5,51)

########## Part 3 ###########
plt.plot(vector_x,function(vector_x),'ro')
plt.title('Plot of $1/1+x^2$')
plt.xlabel('x')
plt.ylabel('$1/(1+t^{2})$')
plt.show()

########## Part 4 ###########
########## integrating with quad ###########
lower_limit=0
upper=0.001
upper_limit=5
dx=0.01
integrated=[]
x_quad=[]#contains x for Integration
error=[]
while upper<upper_limit:
    x_quad.append(upper)
    y,err=quad(function,lower_limit,upper)
    integrated.append(y)
    error.append(err)
    upper+=dx
    
#error=np.array(np.abs(np.arctan(x)-integrated))
###### tabulate ########
arctan=np.arctan(x_quad)
table=list(zip(integrated,arctan))
print(tabulate(table,tablefmt='fancy_grid'))

########## plot arctan and quad ###########
fig,ax=plt.subplots()
ax.set_title('Integration curves')
ax.set(xlabel='x', ylabel='$\int_{0}^{x} du/(1+u^2)$')
ax.plot(x_quad,integrated,'ro',label='quad')
ax.plot(x_quad,arctan,label='fn')
ax.legend()
plt.show()

########## plot semilog of error ###########
plt.semilogy(x_quad,np.array(integrated)-np.array(np.arctan(x_quad)),'ro')
plt.title('Error in $\int_{0}^{x} dx/(1+x^2)$')
plt.xlabel('x')
plt.ylabel('Error')
plt.show()
    
    
########## Part 5 ###########
########## integrate using trapezoidal method ###########
I=[]
lower=0
I.append(0)
h=0.001
upper=0.001
i=1
summation=[]
x=[]
func_lower=(function(lower))#contains f(0)
#summation is just the summation part of the formula
summation.append(0)
x.append(0)
while upper<upper_limit:
    x.append(upper)
    summation.append(summation[i-1]+function(upper))
    integral_by_h=summation[i]-0.5*(func_lower+function(upper))
    I.append(h*integral_by_h)
    upper+=h
    i+=1
    
####### Vectorized #########
########## integrate using trapezoidal method using cumsum ###########
vector=[]
x=[]
lower=0
func_lower=function(lower)
#vector.append(func_lower)
#x.append(0)
h=0.001
upper_limit=5
upper=0.001
#vector has f(xi)
# generating vector
for i in range(1,int(upper_limit/h)):
    x.append(upper)
    vector.append(function(upper))
    upper+=h
#vectorized integration
summation=np.cumsum(vector)
f_lower=np.ones(int(upper_limit/h)-1)*func_lower# has f(0)
sub=0.5*(f_lower+vector)
I=h*(summation-sub)
###### Plot the functions #########
plt.plot(x_quad,integrated,'ro',label='quad')
plt.plot(x_quad,arctan,label='fn')
plt.plot(x,I,'g-',label='trapezoidal')
plt.title('Integrated Curves')
plt.xlabel('x')
plt.ylabel('$\int_{0}^{x} du/(1+u^2)$')
plt.legend()
plt.show()
########### Error estimation by changing h ##############
# error= (-h^2(b-a)f"(x))/12
#therefore max error is when x=1
# ref:http://homepage.divms.uiowa.edu/~atkinson/ftp/ENA_Materials/Overheads/sec_5-2.pdf
# ref:http://math.cmu.edu/~mittal/Recitation_notes.pdf
# therefore estimated_error(h)=h^2*5/48
est_error=[]
#using the formula
def error_est_h(h):
    return (h**2*(upper_limit-lower_limit))/48
    
h=[]
h.append(0.01)
for i in range(1,10):
    h.append(h[i-1]/2)
    
for i in range(10):
    est_error.append(error_est_h(h[i]))
        

######### Getting the exact error ##################
#get the h array
H=[]
H.append(0.01)
for i in range(1,10):
    H.append(H[i-1]/2)
    
iter_=0
max_error_arr=[] #contains the maximum of the errors for a particular h
dx=1.953125e-05 # base h=2^-10 for getting x
x=[] #contains x for arctan
upper=0
upper_limit=5
print('creating x')
#creating x for arctan
while upper<upper_limit:
    x.append(float(upper))
    upper+=dx
# iterate through H for different h
for h in H:
    integrated=[]
    lower=0
    dx=h
    upper=0
    i=1
    summation=[]
    x_sample=[]
    func_lower=(function(lower))#contains f(0)
    #summation is just the summation part of the formula
    summation.append(0)
    while upper<upper_limit:
        x_sample.append(upper)
        summation.append(summation[i-1]+function(upper))
        integral_by_h=summation[i]-0.5*(func_lower+function(upper))
        integrated.append(dx*integral_by_h)
        upper+=dx
        i+=1
    print('Finding Common')   # get the common x terms
    x_round=np.round(x,10)   #round the values upto 10 decimals
    x_round_sample=np.round(x_sample,10)
    c=np.in1d(x_round,x_round_sample) # get the common x terms
    # the return value is a boolean array
          
            
    j=0
    arctan=[]
    for element in c:
         if element==True:
             arctan.append(np.arctan(x[j]))
         j+=1
    # to check if lenght of both arctan and integrated are same
    if len(integrated)>len(arctan):
        integrated=integrated[:-1]
    if len(integrated)<len(arctan):
        arctan=arctan[:-1]
    #
    error_arr=np.array(np.abs(np.array(integrated)-arctan))
    max_error_arr.append(max((error_arr)))
    print(iter_)
    iter_+=1
      
######### Plotting the exact and estimated error ##################
plt.loglog(H,max_error_arr,'ro',label='exact')
plt.loglog(H,est_error,'gx',label='estimated')
plt.legend()
plt.xlabel('h')
plt.ylabel('Error')
plt.show()

######################################################