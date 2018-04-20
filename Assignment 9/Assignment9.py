import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,2*np.pi,128)
y=np.sin(5*x)
Y=np.fft.fft(y)
plt.plot(abs(Y),lw=2)
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin(5t)$")
plt.grid(True)
plt.show()
plt.plot(np.unwrap(np.angle(Y)),lw=2)
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
plt.show()

def plot_spectrum(y,samples,suppress,w_lim,title):
    Y=np.fft.fftshift(np.fft.fft(y))/samples
    w=np.linspace(-64,63,samples)
    
    #### plot magnitude #####
    plt.plot(w,abs(Y),lw=2,label='magnitude')
    plt.xlim([-w_lim,w_lim-1])
    plt.ylabel(r"$|Y|$",size=16)
    plt.title("Spectrum of" +title)
    plt.grid()
    plt.legend()
    plt.show()
    
    #### plot phase #####
    if suppress==None:
        plt.plot(w,np.angle(Y),'ro',lw=2,label='noise')
        ii=np.where(abs(Y)>1e-3)
        plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2,label='expected_phase')
    else:
        ii=np.where(abs(Y)>suppress)
        plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2,label='expected_phase')
    plt.xlim([-w_lim,w_lim+1])
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$k$",size=16)
    plt.legend()
    plt.grid(True)
    plt.show()
    
x=np.linspace(0,2*np.pi,129)
x=x[:-1] #to account for the overlapping in the sine function
y=np.sin(5*x)
plot_spectrum(y=y,samples=128,suppress=1e-3,w_lim=10,title=' $sin(5t)$')

t=np.linspace(0,2*np.pi,129);t=t[:-1]
y=(1+0.1*np.cos(t))*np.cos(10*t)
plt.plot(t,y)
plt.grid()
plt.title('Amplitude modulation of \n $cos(10t)$ with $(1+0.1cos(t))$')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.show()

plot_spectrum(y=y,samples=128,suppress=None,w_lim=15,title=' $(1+0.1cos(t))(cos(10t))$')

t=np.linspace(-4*np.pi,4*np.pi,513);t=t[:-1]
y=(1+0.1*np.cos(t))*np.cos(10*t)
plt.plot(t,y)
plt.grid()
plt.title('Amplitude modulation of \n $cos(10t)$ with $(1+0.1cos(t))$')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.show()

t=np.linspace(-4*np.pi,4*np.pi,513);t=t[:-1]
y=(1+0.1*np.cos(t))*np.cos(10*t)
plot_spectrum(y=y,samples=512,suppress=None,w_lim=15,title=' $(1+0.1cos(t))(cos(10t))$')

t=np.linspace(0,2*np.pi,129);t=t[:-1]
y=np.sin(t)**3
plot_spectrum(y=y,samples=128,suppress=None,w_lim=15,title=' $sin^3(t)$')



t=np.linspace(0,2*np.pi,129);t=t[:-1]
y=np.cos(t)**3
plot_spectrum(y=y,samples=128,suppress=None,w_lim=15,title=' $cos^3(t)$')


t=np.linspace(-4*np.pi,4*np.pi,513);t=t[:-1]
y=np.cos(20*t+5*np.cos(t))
plot_spectrum(y=y,samples=512,suppress=1e-3,w_lim=40,title=' $cos(20t+5cos(t))$')

import numpy as np
import matplotlib.pyplot as plt
n=w_limit=40
samples=10000
t=np.linspace(-n*np.pi,n*np.pi,samples+1);t=t[:-1]
y=np.exp(-t*t/2)
Y=np.fft.fftshift(np.abs(np.fft.fft(y)))*(2*(n*np.pi))/samples
# magnitude scaling by (2pi/T)
w=np.linspace(-samples/(2*n),samples/(2*n),samples+1);w=w[:-1]
y_exp=np.exp(-w*w/2)*np.sqrt(2*np.pi)
title=' $e^{-x^2/2}$'

#### plot magnitude of obtained graph#####
plt.plot(w,abs(Y),lw=2,label='obtained')
plt.xlim([-10,10])
plt.ylabel(r"$|Y|$",size=16)
plt.title("Spectrum of" +title)


#### plot magnitude of expected graph#####
plt.plot(w,y_exp,label='expected')
plt.xlim([-10,10])
plt.legend()
plt.grid()
plt.show()

plt.semilogy(t,np.abs(abs(Y)-y_exp))
plt.grid()
plt.xlim([-50,50])
plt.title('Plot of the error in the obtained graph')
plt.xlabel('$\omega$')
plt.ylabel('Error(log scale)')
plt.show()

plt.plot(t,np.abs(abs(Y)-y_exp))
plt.grid()
plt.show()
print('Max error is %s'%str(max(np.abs(abs(Y)-y_exp))))
'''
#Bonus
import cv2
img = plt.imread('/Users/siddharthnayak/Downloads/Cristiano-Ronaldo-HD-Portugal-wallpaper-1024x576.jpg',0)
plt.imshow(img)
plt.title('Original Image')
plt.show()

img = cv2.imread('/Users/siddharthnayak/Downloads/Cristiano-Ronaldo-HD-Portugal-wallpaper-1024x576.jpg',0)
# now image is a grayscale version
f = np.fft.fft2(img) #take 2D fft
fshift = np.fft.fftshift(f) # shift
magnitude_spectrum = 20*np.log(np.abs(fshift))
#get the magnitude on dB scale

plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.show()

plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.show()


'''