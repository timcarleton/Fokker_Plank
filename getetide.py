import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve,minimize_scalar
from scipy.misc import derivative
from scipy import interpolate
from astropy.constants import G
from astropy import units
import getaefromperi
import getrvcorr
reload(getrvcorr)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units
kminkpc=1*units.kpc.to(units.km)

GN=(G.to(units.kpc**3/units.s**2/units.M_sun)).value
GNfront=(G.to(units.km**2*units.kpc/units.s**2/units.M_sun)).value

#params=(pericenter (Mpc),omatpericenter (1/s),rho0host (msun/kpc**3),rshost (kpc),mvirhost (msun))


def getrfrompot(potfunc,e,dsidr,minr=1E-10):

#    print abs(np.log(abs(potfunc(10**-1)))-np.log(abs(2*(e))))
 #   print abs(np.log(abs(potfunc(10**-2)))-np.log(abs(2*(e))))
  #  print abs(np.log(abs(potfunc(10**1)))-np.log(abs(2*(e))))
    if e!=0:
        r0=np.log(1.0/abs(e))+np.log(abs(potfunc(minr)))
    else:
        r0=1.0
    if r0==None:
        for i in np.linspace(0,10):
            r0=1-np.log(abs(e))
            if np.isfinite(np.log(abs(potfunc(np.exp(r0))))):
                break
            else:
                None
    #r0=potfunc(minr)/e
    if abs(e)<.95*abs(potfunc(minr)):
        rsi=np.exp(fsolve(lambda x: np.log(-potfunc(np.exp(x)))-np.log(abs(e)),r0,full_output=False,factor=1)[0])
        if not np.isfinite(rsi):
            rsi=np.exp(fsolve(lambda x: abs(potfunc(np.exp(x))-abs(e)),r0,full_output=False)[0])
    #print si,rsi,sifunc(rsi)
    else:
        #r0=1.0
        rsi=np.exp(fsolve(lambda x: potfunc(np.exp(x))-e,r0,full_output=False)[0])

#    print e
#    rsi=fsolve(lambda x: np.log(abs(potfunc(x)))-np.log(abs(e)),r0,full_output=False,fprime=lambda x: 1.0/potfunc(x)*dsidr(x))[0]
    #rsi=10**(fsolve(lambda x: np.log(abs(potfunc(10**x)))-np.log(abs(e)),np.log10(r0),full_output=False)[0])

    return rsi

# def gethostmass(r,params):
    
#     rho=params[2]
#     rs=params[3]
#     x=r/rs
#     mr=4*np.pi*rho*rs**3*(np.log(1+x)-r/(r+rs))
#     if mr<params[4]:
#         return mr
#     else:
#         return params[4]

# def getr(theta,params):

    
#     a,e=getaefromperi.getaefromperi(params[0],params[1],gethostmass(params[0],params))

#     return a*(1-e**2)/(1+e*np.cos(theta))


# def getmur(theta,params):

#     r=getr(theta,params)*1000 #Mpc to kpc

#     mr=gethostmass(r,params)
#     if mr<params[4]:
#         return mr/params[4]
#     else:
#         return 1.0

# def getmuhat(theta,params):
    
#     r=getr(theta,params)*1000 #Mpc to kpc
#     mr=gethostmass(r*1000,params)
#     if mr>params[4]:
#         return 0

#     rho=params[2]
#     rs=params[3]
#     x=r/rs #unitless
#     bottom=params[4]/(4*np.pi*rho*rs**3) #unitless
#     return x**2/(1+x)**2/bottom


def integrand1(theta,params):

    getmur,getmuhat,getr,a,j=params[0][0],params[0][1],params[0][2],params[0][3],params[0][4]
    try:
        r=getr(theta)
    except ValueError:
        r=np.inf
    return (3*getmur(r)-getmuhat(r))/(r/a)*np.cos(theta)**2/j(theta)

def integrand2(theta,params):

    getmur,getmuhat,getr,a,j=params[0][0],params[0][1],params[0][2],params[0][3],params[0][4]
    try:
        r=getr(theta)
    except ValueError:
        r=np.inf
    return (3*getmur(r)-getmuhat(r))/(r/a)*np.sin(theta)**2/j(theta)

def integrand3(theta,params):

    getmur,getmuhat,getr,a,j=params[0][0],params[0][1],params[0][2],params[0][3],params[0][4]
    try:
        r=getr(theta)
    except ValueError:
        r=np.inf
    return getmur(r)/(r/a)/j(theta)

def getetideorbit(hostmu,hostmuhat,rthetafunc,rs,angmom,hostmass,thetamin=-np.pi,thetamax=np.pi,folder='./'):

    print thetamin,thetamax
    #these are unitless
    b1=quad(integrand1,thetamin,thetamax,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
    b2=quad(integrand2,thetamin,thetamax,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
    b3=quad(integrand3,thetamin,thetamax,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]

    print 'b3',b3
#    j=angmom
    
    first=GN*hostmass/rs**3
#    print 'f',first,hostmass,rs,r,j2,'f'
    top1=(b1-b3)**2
    top2=(b2-b3)**2
    top3=b3**2
    plt.clf()
    plt.plot(np.linspace(-np.pi,np.pi),[hostmu(rthetafunc(i))/rthetafunc(i) for i in np.linspace(-np.pi,np.pi)])
    plt.savefig(folder+'mutheta.png')
#    for i in np.linspace(-np.pi,np.pi):
#        print i,rthetafunc(i),hostmu(rthetafunc(i)),hostmu(rthetafunc(i))/rthetafunc(i)
    print angmom(np.linspace(-np.pi,np.pi))
    b32=top3*np.mean(angmom(np.linspace(-np.pi,np.pi)))**2
    d32=b32*(rthetafunc(0)/rs/hostmu(rthetafunc(0)))**2
    print 'd3',np.sqrt(d32)



 #   bottom=6*j**2

    vp=angmom(0)*np.sqrt(GN*hostmass*rs)/rthetafunc(0)
    print angmom(0)**2
    print rthetafunc(0)**2*vp**2/(GN*hostmass*rs)
    bottom=6
    print 'vp',vp
    print GN,hostmu(rthetafunc(0))*hostmass,rthetafunc(0),vp
    print (GN*hostmu(rthetafunc(0))*hostmass/rthetafunc(0)**2/vp)**2/6
    print hostmu(rthetafunc(0))*hostmass
    ff=vp**2*rthetafunc(0)/GN/(hostmass*hostmu(rthetafunc(0)))
    print 'ff',vp**2*rthetafunc(0)/GN/(hostmass*hostmu(rthetafunc(0)))
    print 'et',top1,top2,top3,bottom,first,rs
    print (GN*hostmass/rs)**2*b32/6.0/rthetafunc(0)**2/vp**2
    print first*(top1+top2+top3)/bottom,GN*hostmass/rs
    print 'rt',rthetafunc(0),angmom(0),angmom(.5)
    print 'j',angmom(2),angmom(1),angmom(-1),angmom(-2),angmom(0),rthetafunc(0)**2*4.540E-16/np.sqrt(GN*hostmass*rs)
    print 'f',GN*hostmass/6/(angmom(0)**2)/rs**3,hostmu(rthetafunc(0))**2,rthetafunc(0)**2,rs**2
    print 'ce',GN*hostmu(rthetafunc(0))*hostmass/6.0/rthetafunc(0)**3*d32/ff
#    print top1,top2,top3

    return first*(top1+top2+top3)/bottom


def getetidenoadiabat(e,rthetafunc,hostmu,hostmuhat,satpotfunc,rs,hostmass,peri,satdphidr,angmom,orbit=None,minr=1E-10,folder='./'):

    lowrs=np.logspace(np.log10(minr),1,num=1000)
    phi=satpotfunc(lowrs)
    lowsatpotfunc=interp1d(lowrs,phi,fill_value='extrapolate')

    if len(np.shape(e))==0:
        if abs(e)<abs(satpotfunc(.01)):
            rmax=getrfrompot(satpotfunc,e,satdphidr,minr=minr)
            vm2=(2*e-2*satpotfunc(minr))
            rvcorr,r2,v2=getrvcorr.getrvcorr(satpotfunc,e,rmax,phiprime=None,minr=minr)
            r2=r2*rmax**2
            v2=v2*vm2
        else:
            rmax=getrfrompot(lowsatpotfunc,e,satdphidr,minr=minr)
            vm2=(2*e-2*satpotfunc(minr))
            rvcorr,r2,v2=getrvcorr.getrvcorr(satpotfunc,e,rmax,phiprime=None,minr=minr)
            r2=r2*rmax**2
            v2=v2*vm2

    else:
        rvcorr=[]
        r2=[]
        v2=[]
        rmax=[]
        vm2=[]

        for i in range(len(e)):

            if abs(e[i])<abs(satpotfunc(.01)):

                rmax.append(getrfrompot(satpotfunc,e[i],satdphidr,minr=minr))
                vm2.append(2*e[i]-2*satpotfunc(minr))
                rvcori,r2i,v2i=getrvcorr.getrvcorr(satpotfunc,e[i],rmax[i],phiprime=None,minr=minr)
                rvcorr.append(rvcori)
                r2.append(r2i)
                v2.append(v2i)
            else:

                rmax.append(getrfrompot(lowsatpotfunc,e[i],satdphidr,minr=minr))
                vm2.append(2*e[i]-2*lowsatpotfunc(minr))
                rvcori,r2i,v2i=getrvcorr.getrvcorr(satpotfunc,e[i],rmax[i],phiprime=None,minr=minr)
                rvcorr.append(rvcori)
                r2.append(r2i)
                v2.append(v2i)

        rmax=np.array(rmax)
        vm2=np.array(vm2)
        r2=np.array(r2)*rmax**2
        v2=np.array(v2)*vm2
        rvcorr=np.array(rvcorr)

    plt.clf()
    #plt.plot(abs(e),v2)
    plt.plot(abs(e),np.sqrt(r2))
    plt.loglog()
    plt.savefig(folder+'r2.png')
    plt.clf()
    plt.plot(abs(e),np.sqrt(v2))
    plt.ylabel('v')
    plt.loglog()
    plt.tight_layout()
    plt.savefig(folder+'v2.png')
    plt.clf()


    if orbit==None:
        thetam=np.pi

    #these are unitless
        b1=quad(integrand1,-thetam,thetam,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
        b2=quad(integrand2,-thetam,thetam,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
        b3=quad(integrand3,-thetam,thetam,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
    
#    print 'bs',b1,b2,b3,peri,rs,(hostpotfunc(peri*(1+ecc)/(1-ecc))-hostpotfunc(peri)),hostmu(peri),ecc,hostmass,hostmu(peri),rs
    #a,ecc=getaefromperi.getaefromperi(params[0],params[1],gethostmass(params[0],params))

    #this is unitless
#    print b1,b2,b3
    #j2=(hostpotfunc(peri*(1+ecc)/(1-ecc))-hostpotfunc(peri))*(peri)**2*(1+ecc)**2/2.0/ecc/GN/hostmass*hostmu(peri)/rs
        j2=angmom**2
    
        j=np.sqrt(j2)
#    print j
    #first has units of kpc^2/s^2
        first=GN*hostmass/rs**3*r2
#    print 'f',first,hostmass,rs,r,j2,'f'
        top1=(b1-b3)**2
        top2=(b2-b3)**2
        top3=b3**2
#        bottom=6*j**2
        bottom=6
#    print top1,top2,top3
        de=first*(top1+top2+top3)/bottom
    else:
        de=orbit*r2

    de2=de*2.0/3*v2*(1+rvcorr)
    plt.clf()
#    print 'f'
#    print de/r2/(GN*9.839E12/278.6)**2*(48.6*1.278E-15)**2*48.6**2
#    print r2/(GN*3.3E10/9)**2/(508/kminkpc)**2
#    print de
#    print e/np.min(e)
    vhalo=np.sqrt(9.14703741927154e-28)
    vperi=48.6*1.278E-15
    rperi=48.6
    print 'x',np.min(e)
    w=np.argmin(abs(e/np.min(e)-.2))
    print w
    print r2[w]
    w=np.argmin(abs(e/np.min(e)-.6))
    print r2[w]
    

    w=np.where((e/np.min(e)>.2) & (e/np.min(e)<.6))[0]
    print np.median(de[w]),np.median(de2[w])
#    print de/(vhalo**4*r2/(vperi*rperi)**2)

    r=np.sqrt(r2)
    omega=np.sqrt(1.0/r*abs(satdphidr(r)))
    
    plt.clf()
    plt.plot(r,omega)
    plt.loglog()
    plt.savefig(folder+'romega.png')
    plt.clf()
    return de,de2,omega

def gete2tidenoadiabat(e,rthetafunc,hostmu,hostmuhat,satpotfunc,rs,hostmass,peri,satdphidr,angmom,orbit=None,minr=1E-10):


    if len(np.shape(e))==0:
        rvcorr,r2,v2=getrvcorr.getrvcorr(satpotfunc,e,getrfrompot(satpotfunc,e,satdphidr,minr=minr),phiprime=satdphidr,minr=minr)
    else:
        rvcorr=[]
        r2=[]
        v2=[]

        for i in range(len(e)):
            rvcori,r2i,v2i=getrvcorr.getrvcorr(satpotfunc,e[i],getrfrompot(satpotfunc,e[i],satdphidr,minr=minr),phiprime=satdphidr,minr=minr)
            rvcorr.append(rvcori)
            r2.append(r2i)
            v2.append(v2i)

        r2=np.array(r2)
        v2=np.array(v2)
        rvcorr=np.array(rvcorr)

    second=2*r2*v2*(1+rvcorr)/3.0


    if orbit==None:
        thetam=np.pi

        b1=quad(integrand1,-thetam,thetam,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
        b2=quad(integrand2,-thetam,thetam,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]
        b3=quad(integrand3,-thetam,thetam,args=[(hostmu,hostmuhat,rthetafunc,rs,angmom)])[0]

    
   # a,ecc=getaefromperi.getaefromperi(params[0],params[1],gethostmass(params[0],params))

#    j2=(hostpotfunc(peri*(1+ecc)/(1-ecc))-hostpotfunc(peri))*(peri)**2*(1+ecc)**2/2.0/ecc/GN/hostmass*hostmu(peri)/rs
        j2=angmom**2
        j=np.sqrt(j2)

            
    #first has units of kpc^2/s^2
        first=GN*hostmass/rs**3
    
        top1=(b1-b3)**2
        top2=(b2-b3)**2
        top3=b3**2
        bottom=6
    #    print 'rew2',hostmass,GN,rs
    #    print first,second,top1,top2,top3,bottom
    
        return first*second*(top1+top2+top3)/bottom
    else:
        return orbit*second

def getadiabatcorr1(e,satpotfunc,tau,omega,dphidr=None,speed='fast',minr=1E-10):

    # lowrs=np.logspace(np.log10(minr),1,num=1000)
    # phi=satpotfunc(lowrs)
    # lowsatpotfunc=interp1d(lowrs,phi,fill_value='extrapolate')

    
    # if len(np.shape(e))==0:
    #     if abs(e)<abs(satpotfunc(.01)):
    #         r=getrfrompot(satpotfunc,e,dphidr,minr=minr)
    #     else:
    #         r=getrfrompot(lowsatpotfunc,e,None,minr=minr)
    # else:
    #     r=[]
    #     for i in range(len(e)):
    #         if abs(e[i])<abs(satpotfunc(.01)):
    #             r.append(getrfrompot(satpotfunc,e[i],dphidr,minr=minr))
    #         else:
    #             r.append(getrfrompot(lowsatpotfunc,e[i],None,minr=minr))

    #     r=np.array(r)
        
    # if dphidr!=None:
    #     omega=np.sqrt(1.0/r*abs(dphidr(r)))
    # else:
    #     omega=np.sqrt(1.0/r*abs(derivative(satpotfunc,r)))
    # plt.clf()
    # plt.plot(abs(e),omega)
    # plt.loglog()
    # plt.savefig('omegaeb.png')

    x=omega*tau

    if speed=='fast':
        a1=(1+x**2)**(-2.5)
    else:
        a1=(1+x**2)**(-1.5)
    return a1
            
def getadiabatcorr2(e,satpotfunc,tau,omega,dphidr=None,speed='fast',minr=1E-10):

    # lowrs=np.logspace(np.log10(minr),1,num=1000)
    # phi=satpotfunc(lowrs)
    # lowsatpotfunc=interp1d(lowrs,phi,fill_value='extrapolate')
    
    # if len(np.shape(e))==0:
    #     if abs(e)<abs(satpotfunc(.01)):
    #         r=getrfrompot(satpotfunc,e,dphidr,minr=minr)
    #     else:
    #         r=getrfrompot(lowsatpotfunc,e,None,minr=minr)
    # else:
    #     r=[]
    #     for i in range(len(e)):
    #         if abs(e[i])<abs(satpotfunc(.01)):
    #             r.append(getrfrompot(satpotfunc,e[i],dphidr,minr=minr))
    #         else:
    #             r.append(getrfrompot(lowsatpotfunc,e[i],None,minr=minr))

    #     r=np.array(r)


    # if dphidr!=None:
    #     omega=np.sqrt(1.0/r*abs(dphidr(r)))
    # else:
    #     omega=np.sqrt(1.0/r*abs(derivative(satpotfunc,r)))
    
    x=omega*tau
    
    if speed=='fast':
        a2=(1+x**2)**(-3)
    else:
        a2=(1+x**2)**(-1.5)
    return a2

#potential functions: give r in kpc, get phi in kpc^2/s^2
#e also in kpc^2/s^2
def getetide(e,rthetafunc,hostmu,hostmuhat,satpotfunc,satdphidr,rs,omatperi,hostmass,peri,angmom,orbit=None,minr=1E-10,taufactor=1.0,tau12=7E15,folder='./'):

    de,de2,omega=getetidenoadiabat(e,rthetafunc,hostmu,hostmuhat,satpotfunc,rs,hostmass,peri,satdphidr,angmom,orbit=orbit,minr=minr,folder=folder)
    tau=2/omatperi/taufactor
    
    if tau>tau12:
        adiabat1=getadiabatcorr1(e,satpotfunc,tau,omega,satdphidr,minr=minr,speed='slow')
        adiabat2=getadiabatcorr2(e,satpotfunc,tau,omega,satdphidr,minr=minr,speed='slow')

    else:
        adiabat1=getadiabatcorr1(e,satpotfunc,tau,omega,satdphidr,minr=minr,speed='fast')
        adiabat2=getadiabatcorr2(e,satpotfunc,tau,omega,satdphidr,minr=minr,speed='fast')
    
    plt.clf()
    plt.plot(abs(e),adiabat1)
    plt.plot(abs(e),adiabat2)
    plt.loglog()
    plt.savefig(folder+'adiabat.png')

#    print 'a3',getadiabatcorr1(e,satpotfunc,tau,satdphidr)
    #print 'a1',getadiabatcorr1(e,satpotfunc,tau,satdphidr)
    #print 'na',getetidenoadiabat(e,rthetafunc,hostmu,hostmuhat,hostpotfunc,satpotfunc,rs,ecc,hostmass,peri)
    
#    adiabat1=getadiabatcorr1(e,satpotfunc,tau,satdphidr,minr=minr)
#    adiabat2=getadiabatcorr2(e,satpotfunc,tau,satdphidr,minr=minr)

    return de*adiabat1,de2*adiabat2

#potential functions: give r in kpc, get phi in kpc^2/s^2
#e also in kpc^2/s^2
def gete2tide(e,rthetafunc,hostmu,hostmuhat,satpotfunc,satdphidr,rs,omatperi,hostmass,peri,angmom,orbit=None,minr=1E-10,taufactor=1.0):

    tau=2/omatperi/taufactor
    #print gete2tidenoadiabat(e,rthetafunc,hostmu,hostmuhat,hostpotfunc,satpotfunc,rs,ecc,hostmass,peri,satdphidr)
    return gete2tidenoadiabat(e,rthetafunc,hostmu,hostmuhat,satpotfunc,rs,hostmass,peri,satdphidr,angmom,orbit=orbit,minr=minr)*getadiabatcorr2(e,satpotfunc,tau,satdphidr,minr=minr)#/(GN*hostmass/rs)**2

def phihernquist(r,m,a):
    return -GN*m/(r+a)

def dphidrhernquist(r,m,a):
    return GN*m/(r+a)**2

def mrhernquist(r,m,a):
    return a**2*np.log((r+a)/a)+0.5*r*(r-2*a)

def phiplummer(r,m,a):
    return -GN*m/np.sqrt(r**2+a**2)

def dphidrplummer(r,m,a):
    return GN*m*r/(r**2+a**2)**1.5

def rthetaint(dx,dy,dz,t):
    xint=interpolate.interp1d(t,dx,fill_value='extrapolate')
    yint=interpolate.interp1d(t,dy,fill_value='extrapolate')
    zint=interpolate.interp1d(t,dz,fill_value='extrapolate')

    mn=minimize_scalar(lambda dt: (xint(dt)**2+yint(dt)**2+zint(dt)**2))

#    print mn
#    print xint(mn['x'])
    dot=(dx*xint(mn['x']))+(dy*yint(mn['x']))+(dz*zint(mn['x']))

#    print dot
    wpre=np.where(t<mn['x'])[0]
    wpost=np.where(t>=mn['x'])[0]
    theta=np.arccos(dot/(np.sqrt(dx**2+dy**2+dz**2)*np.sqrt(mn['fun'])))
    theta[wpost]=-theta[wpost]

#    print theta
    return interpolate.interp1d(theta,np.sqrt(dx**2+dy**2+dz**2))
#mvir, rs, 

test=False
if test:
    hostpotfunc=lambda x: phihernquist(x,3.3E10,0.6)
    hostdphidr=lambda x: dphidrhernquist(x,3.3E10,0.6)
    #hostmu=lambda x: mrhernquist(x,3.3E10,.6)
    hostmu=lambda x: x**2/(x+.6)**2
    hostmuhat=lambda x: 2*.6*x**2/(x+.6)**3
    ecc=0.69
    peri=0.73
    omatperi=1.758E-14
    semi=peri/(1-ecc)
    rtheta=lambda x: semi*(1-ecc**2)/(1+ecc*np.cos(x))
    
    satpotfunc=lambda x: phiplummer(x,2.5E5,.001)
    satdphidr=lambda x: dphidrplummer(x,2.5E5,.001)
    
    tauclust=1.1376E14*2
    
    rstst=np.logspace(-2.5,0)
    estst=satpotfunc(rstst)
    
    corrs=[getadiabatcorr2(i,satpotfunc,tauclust,dphidr=satdphidr) for i in estst]
    des=[gete2tide(i,rtheta,hostmu,hostmuhat,hostpotfunc,satpotfunc,satdphidr,.6,omatperi,3.3E10)[0] for i in estst]
    
    plt.plot(abs(estst),np.array(des)/abs(estst)/abs(estst))
    plt.loglog()
    # #plt.ylim(1E-22,1E-6)
    plt.show()
    
    eccs=np.linspace(.01,.99)
    semis=np.logspace(-2,2)
    dese=[getetide(estst[40],lambda x: i*(1-ecc**2)/(1+ecc*np.cos(x)),hostmu,hostmuhat,hostpotfunc,satpotfunc,satdphidr,.6,omatperi,3.3E10)[0] for i in semis]
    
    #plt.plot(semis,dese)
    #plt.yscale('log')
    #plt.show()
