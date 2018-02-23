from scipy import optimize
import numpy as np
def ufunc(x,alpha,gamma,delta):
    a=-np.log10(10**(alpha*x)+1)
    b=(np.log10(1+np.exp(x)))**gamma
    c=1+np.exp(10**(-x))
    return a+delta*(b/c)

def getlgstellarmass(lghalomass,z=0,use='GK',cluster=False):
    
    a=1.0/(1+z)
    am1=a-1
    
    nu=np.exp(-4*a**2)

    if cluster:
        logep=-1.642
        logm1=11.35
    else:
        logep=-1.777-0.006*am1*nu-.119*am1
        logm1=11.514+(-1.793*am1-.251*z)*nu
    if use=='B':
        alpha=-1.412+(0.731*am1)*nu
    else:
        alpha=-1.92
    if cluster: #Kravtsov 2014
#        delta=4.394
        delta=4.335
#        alpha=-1.779
        alpha=-1.740
#        gamma=0.547
        gamma=0.531
    else:
        delta=3.508+(2.608*am1-0.043*z)*nu
        gamma=0.316+(1.319*am1+.279*z)*nu

    lgmstar=logep+logm1+ufunc(lghalomass-logm1,alpha,gamma,delta)-ufunc(0,alpha,gamma,delta)

    return lgmstar

def minfunc(lghalomass,stellarmass,z,use):
    return getlgstellarmass(lghalomass,z,use)-stellarmass

def gethalomass(stellarmass,z=0,use='GK',**args):
    if stellarmass>10.4:
        lower=10**((stellarmass-6.43)/5.26)
        upper=10**((stellarmass)/5.26)
    else:
        if use=='B':
            lower=(stellarmass+4.8)/1.41
            upper=(stellarmass+8.3)/1.41
        else:
            lower=(stellarmass+10.7)/1.92
            upper=(stellarmass+14.7)/1.92

    return optimize.brentq(minfunc,lower,upper,(stellarmass,z,use),**args)

