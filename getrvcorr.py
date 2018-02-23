import numpy as np
from scipy.optimize import fsolve
import getfe01 as getfe
from scipy.integrate import quad

def inmintegrand(phir,rmax,e,n,m,x,minr=1E-10):
    top=phir(x*rmax)-phir(minr)
    bottom=phir(rmax)-phir(minr)

    if (1-top/bottom)<0:
        return x**(2+m)
    else:
        #return x**(2+m)*(1-(phir(x*rmax)-phir(minr))/(phir(rmax)-phir(minr)))**((1.0+n)/2.0)
        return x**(2+m)*(1-(top/bottom))**((1.0+n)/2.0)
def inm(phir,rmax,e,n,m,minr=1E-10):

    return quad(lambda x: inmintegrand(phir,rmax,e,n,m,x,minr=minr),minr,1)[0]
#    return quad(lambda r: r**(2+m)*(e-phir(r))**((1.0+n)/2.0),0,rmax)[0]

def getrvcorr(phir,e,rmax,phiprime=None,minr=1E-10):

    r0=-4-np.log(e)
    #rmax=fsolve(lambda x: np.log(abs(phir(x)))-np.log(abs(e)),np.exp(r0),full_output=False,fprime=lambda x: 1.0/phir(x)*phiprime(x),factor=1)[0]
    #print phir(r0)
    i00=inm(phir,rmax,e,0,0,minr=minr)
    i22=inm(phir,rmax,e,2,2,minr=minr)
    i20=inm(phir,rmax,e,2,0,minr=minr)
    i02=inm(phir,rmax,e,0,2,minr=minr)
    
    rvcorr=i22*i00/i02/i20-1
    r2=i02/i00
    v2=i20/i00

    return rvcorr,r2,v2
