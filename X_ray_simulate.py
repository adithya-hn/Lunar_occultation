
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

import tqdm


import math as mt
import timeit

startTime = timeit.default_timer()

def circ (r,x):
    y=np.sqrt(r**2-(x-r)**2)
    return y
size_d=10 #0.3 # size in diameter
size=size_d/2
theta=(size/(1000*3600))*np.pi/180
l= 3.84e8 # 1.5e9
r=(l*mt.tan(theta))
print('Disc radius:',r)
#r=74
x=np.arange(0,(2*r)+1,0.2)
y=circ(r,x)
disc_intensity=((y)/(r))
non_l=np.count_nonzero(~np.isnan(disc_intensity))
print(len(disc_intensity),non_l)
disc_intensity=disc_intensity[:non_l]
#print(disc_intensity)

def Simulate(d,lmb):
    def Function (d,lmb,x):
        f1= np.cos((np.pi*x**2)/(d*lmb))
        f2=np.sin((np.pi*x**2)/(d*lmb))
        return f1,f2

    a=-100    
    B=100
    #H=0.02
    H=0.2
    n=int((B-a)/H)
    print('Steps',n)
    func=[]
    B_=[]
    for j in range(n):
        a=-100
        b=a+(j+1)*H
        #h=0.01
        h=0.1
        steps=int((b-a)/h)
        #print(a,b,steps)
        af1,af2= Function(d ,lmb,a)
        bf1,bf2= Function(d ,lmb,b)
        cf1=[]
        cf2=[]
        B_.append(b)
        for i in range(1,steps-2):
            f1,f2= Function(d ,lmb,a+i*h)
            cf1.append(2*f1)
            cf2.append(2*f2)
        trp_meth_f1=(h/2)*(af1+bf1+np.sum(cf1))
        trp_meth_f2=(h/2)*(af2+bf2+np.sum(cf2))
        #print(pow(pow(abs(trp_meth_f1),2)+pow(abs(trp_meth_f2),2),2))
        func.append(pow(pow(abs(trp_meth_f1),2)+pow(abs(trp_meth_f2),2),2))
    return np.array(func),np.array(B_)
func1,B1_=Simulate(3.84e8 ,500e-9)#1.5e9 ,0.309e-9)#/(4000*9) # 0.206 6kev
func,B_=Simulate(3.84e8 ,800e-9)#1.5e9 ,1.24e-9)
func1=func1/(4000*9*.0628*5.29*1.253*1.045)#3.5
func=func/(4000*9*2.9)#3.5
c_f=(func1+func)/(2*0.25*2.1*3)

print(len(func),len(B_))
#convolved_intensity = convolve1d(func, disc_intensity)/(7*80*size_d)#320
convolved_band = convolve1d(c_f, disc_intensity)/(7*size_d)#80
#plt.plot(B_[4500:5500],func[4500:5500],color="red", marker="o", linestyle="-",markersize=1,label='Point source 1.24 nm(1 kev)')
plt.plot(B_,func,color="red", marker="o", linestyle="-",markersize=1,label='Point source 500 nm')
plt.plot(B_,c_f,color="black", marker="o", linestyle="-",markersize=1,label='Point source 500-800 nm')
plt.plot(B_,convolved_band,color="green", marker="o", linestyle="-",markersize=1,label=f'Convolved {size_d} mas object in 500-800 nm ')
#plt.plot(B_[4500:5500],c_f[4500:5500],color="black", marker="o", linestyle="-",markersize=1,label='Point source 1-4 kev)')
#plt.plot(B_[4500:5500],convolved_intensity[4500:5500],color="blue", marker="o", linestyle="-",markersize=1,label=f'Convolved {size_d} milliarcsecond object ')
#plt.plot(B_[4500:5500],convolved_band[4500:5500],color="green", marker="o", linestyle="-",markersize=1,label=f'Convolved {size_d} milliarcsecond object in 1-4 kev ')
plt.grid(True)
plt.legend(loc='upper left',fontsize=4,frameon=True)
plt.xlabel('Distance along ground (in meters) ')
plt.ylabel('Normalized intensity')
plt.title('Simulated diffraction pattern')
plt.savefig('0.3mas_1-4kev_Convolved_plot.jpg',dpi=800)

stopTime = timeit.default_timer()
runtime = (stopTime - startTime)
TotTime=runtime/60 #in Hours
print('Runtime: ',TotTime)
plt.show()
