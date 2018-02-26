import numpy as np
from scipy.optimize import fsolve
import getfe
from scipy.integrate import quad

def inmintegrand(phir,rmax,e,n,m,x,minr=1E-10):
    top=phir(x*rmax)-phir(minr)
    bottom=phir(rmax)-phir(minr)

    if (1-top/bottom)<0:
        return x**(2+m)
    else:
        return x**(2+m)*(1-(top/bottom))**((1.0+n)/2.0)

def phiintegrandtop(phir,rmax,e,x,minr=1E-10):

    return x**2*phir(x)*np.sqrt(e-phir(r))

def phiintegrandbot(phir,rmax,e,x,minr=1E-10):

    return x**2*np.sqrt(e-phir(x))
    
def inm(phir,rmax,e,n,m,minr=1E-10):

    return quad(lambda x: inmintegrand(phir,rmax,e,n,m,x,minr=minr),minr,1)[0]

def getrvcorr(phir,e,rmax,phiprime=None,minr=1E-10):

    i00=inm(phir,rmax,e,0,0,minr=minr)
    i22=inm(phir,rmax,e,2,2,minr=minr)
    i20=inm(phir,rmax,e,2,0,minr=minr)
    i02=inm(phir,rmax,e,0,2,minr=minr)
    
    rvcorr=i22*i00/i02/i20-1
    r2=i02/i00
    v2=i20/i00

    return rvcorr,r2,v2

def getavgphi(phir,e,rmax,phiprime=None,minr=1E-10):

    top=quad(lambda x: phiintegrandtop(phir,rmax,e,x,minr=1E-10))
    bottom=quad(lambda x: phiintegrandbottom(phir,rmax,e,x,minr=1E-10))

    return top/bottom
