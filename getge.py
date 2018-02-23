from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def rsi(sifunc,si,dsidr):

    if si>1:
        return 1E-10
    r0=-1-np.log(si)
    #rsi=fsolve(lambda x: np.log(sifunc(x))-np.log(abs(si)),np.exp(r0),full_output=False,fprime=lambda x: 1.0/sifunc(x)*dsidr(x))[0]
    rsi=fsolve(lambda x: np.log(sifunc(x))-np.log(abs(si)),np.exp(r0),full_output=False)[0]

    print 'rs',rsi,sifunc(rsi),si
    return rsi


def d2rhodsi2(sifunc,drhodr,d2rhodr2,si):

    r0=4-np.log10(si)
    rsi=fsolve(lambda rx: sifunc(rx)-si,10**r0)[0]
    first=drdsi(sifunc,si)**2*d2rhodr2(rsi)
    second=drdsi(sifunc,si)*drhodr(rsi)*derivative(lambda rst: drdsi(sifunc,sifunc(rst)),rsi)
    return first+second

def getge(sifunc,e,dsidr,minr=1E-10):

    rm=rsi(sifunc,e,dsidr)

    integrand=lambda r: np.sqrt(2*(sifunc(r)-e))*r**2

    return quad(integrand,minr,rm)[0]

#def getfe(sifunc,drhodr,d2rhodr2,lowe):
    
#    lowe=np.min(e[np.where(e>0)[0]])
#    rs=np.linspace(0,1.0/np.sqrt(lowe),num=100/np.sqrt(lowe)+1)
#    rs=1.0/np.sqrt(es)
#    sir=lambda r: 1.0/r**2
#    jr=lambda r: d2rhodsi2(sifunc,drhodr,d2rhodr2,sir(r))/r

#    print rs
#    print jr(10)

#    jvalues=np.array([jr(i) for i in rs])
#    jvals=interp1d(rs[1:],jvalues[1:],kind='linear',fill_value='extrapolate')
#    print jvalues
#    rinterp=np.linspace(0,1.0/np.sqrt(lowe),num=1E2/np.sqrt(lowe)+1)
#    print jvals(rinterp)
#    plt.clf()
#    plt.plot(rs,jvalues)
#    plt.plot(rinterp,jvals(rinterp))
#    plt.loglog()
#    plt.savefig('jtst.png')
 #   fe=abel.hansenlaw.hansenlaw_transform(jvalues,direction='forward',dr=rs[1]-rs[0])
#    fe=abel.direct.direct_transform(jvals(rinterp),direction='forward',dr=rinterp[1]-rinterp[0])

#    plt.clf()
#    plt.plot(sir(rinterp),fe*np.sqrt(sir(rinterp)))
#    plt.plot(sir(rinterp),sir(rinterp)**3.5*18.2857)
#    plt.yscale('log')
#    plt.loglog()
#    plt.savefig('atst.png')
#    return interp1d(sir(rinterp),fe*np.sqrt(sir(rinterp)))

sifunc=lambda r: 1.0/np.sqrt(1+r**2)
