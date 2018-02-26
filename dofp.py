import linefit
import getge
reload(getge)
import getfe
reload(getfe)
import getetide
reload(getetide)
import numpy as np
from scipy.interpolate import interp1d,spline
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.optimize import minimize
from astropy import units
from astropy.constants import G
from astropy.convolution import convolve, Box1DKernel
import datetime
import matplotlib.pyplot as plt
import potdenfunc
reload(potdenfunc)
import profileclass
GN=(G.to(units.kpc**3/units.s**2/units.M_sun)).value

def chi2(params,rhoint,massint,r):

    rhor=lambda params,r: potdenfunc.zhaorho(r,10**params[0],10**params[1],params[2],params[3],params[4])

    w=np.where(rhoint(r)>100)[0]
    mtot=lambda params: profileclass.getmassfromzhao0(params[2],params[3],params[4],10**params[0],10**params[1],r[w[-1]])


    chi2=0
    npts=0

    d2=(np.log10(rhor(params,r[w]))-np.log10(rhoint(r[w])))**2+(np.log10(mtot(params))-np.log10(massint))**2
    if np.any(~np.isfinite(d2)):
        return np.inf
                                                                                
    return np.nansum(d2)+params[4]**2

def newphi(x,bestfits):
    phi=0
    for i in range(len(bestfits)):
        phi=phi+potdenfunc.zhaophi(x,bestfits[i][0],bestfits[i][1],bestfits[i][2],bestfits[i][3],bestfits[i][4])
    return phi

def newdphidr(x,bestfits):
    dphidr=0
    for i in range(len(bestfits)):
        dphidr=dphidr+potdenfunc.dphidrzhao(x,bestfits[i][0],bestfits[i][1],bestfits[i][2],bestfits[i][3],bestfits[i][4])
    return dphidr
    
def dofp(rthetafunc,hostmu,hostmuhat,satpotfunc,satdphidr,satd2phidr2,drhodr,d2rhodr2,rs,omatperi,hostmass,peri,rpts,gamma,angmom,orbit=None,taufactor=1.0,tau12=7E15,folder='./',numegrid=300,feresolution=1E-3,dodeltae=True):

    #phi(r=0) gives errors, so use phi(1E-10) instead
    minr=10**-10
    satnorm=satpotfunc(minr)
    print 'sn',satnorm

    
    fedist=[getfe.getfe(lambda x: satpotfunc(x)/satnorm,drhodr[i],d2rhodr2[i],lambda x: satdphidr(x)/satnorm,lambda x,dsi: satd2phidr2(x,dsi)/satnorm,feresolution,minr=minr) for i in range(len(drhodr))]

    egrid=np.linspace(-3*feresolution*satnorm,-(1-8*feresolution)*satnorm,num=numegrid)
#    egrid=np.append(np.linspace(-3E-3*satnorm,-.1*satnorm,num=500)[0:-1],np.linspace(-.1*satnorm,-.9*satnorm,num=100)[0:-1],np.linspace(-.9*satnorm,-(1-8E-4)*satnorm,num=500))

#    egrid=np.append(np.linspace(-3E-3*satnorm,-.1*satnorm,num=300)[0:-1],np.append(np.linspace(-.1*satnorm,-.9*satnorm,num=300)[0:-1],np.linspace(-.9*satnorm,-(1-8E-4)*satnorm,num=300)))

    if len(drhodr)>1:
        print '0',fedist[0](egrid[10]),fedist[1](egrid[10])
        plt.clf()
        plt.plot(egrid,fedist[0](-egrid/satnorm))
        plt.plot(egrid,fedist[1](-egrid/satnorm))
        plt.yscale('log')
        plt.savefig(folder+'fe0.png')

    if dodeltae:
        deltae,deltae2=getetide.getetide(-egrid,rthetafunc,hostmu,hostmuhat,satpotfunc,satdphidr,rs,omatperi,hostmass,peri,angmom,orbit=orbit,minr=minr,taufactor=taufactor,folder=folder)

    else:
        deltae=np.zeros(len(egrid))
        deltae2=np.zeros(len(egrid))

    plt.clf()
    plt.plot(abs(-egrid/satnorm),abs(deltae/satnorm),label='de')
    plt.plot(abs(-egrid/satnorm),np.sqrt(abs(deltae2/satnorm/satnorm)),label='de2')
    plt.loglog()
    plt.xlim(.2,1)
    plt.ylim(.001,5)
    plt.legend()
    plt.ylabel(r'$\Delta E/\Phi(0)$')
    plt.xlabel(r'$-E/\Phi(0)$')
    plt.tight_layout()
    plt.savefig(folder+'deltaevse.png')

    nfinal=[np.zeros_like(egrid) for i in range(len(drhodr))]
    firsts=[np.zeros_like(egrid) for i in range(len(drhodr))]
    seconds=[np.zeros_like(egrid) for i in range(len(drhodr))]
    
    ne0=[np.zeros_like(egrid) for i in range(len(drhodr))]

    fes=[np.zeros_like(egrid) for i in range(len(drhodr))]
    ge=np.zeros_like(egrid)

    fes=[fesdist[j](-egrid/satnorm) for j in range(len(drhodr))]
    
    for i in range(len(egrid)):
        ge[i]=getge.getge(lambda x: satpotfunc(x)/satnorm,-egrid[i]/satnorm,lambda x:satdphidr(x)/satnorm,minr=minr)


    deltan=[]
    for j in range(len(drhodr)):
        ne0[j]=ge*fes[j]
        firsts[j]=-deltae*ne0[j]
        seconds[j]=-deltae2*ne0[j]

        dsecond=np.gradient(seconds[j],egrid)
        dsecondsmooth=convolve(dsecond,Box1DKernel(10), boundary='extend')
        dn=-np.gradient(firsts[j],egrid)+.5*np.gradient(dsecondsmooth,egrid)
        deltan.append(dn)


    nfinal=[ne0[j]+deltan[j] for j in range(len(drhodr))]
    fefinal=[np.zeros_like(egrid) for j in range(len(drhodr))]

    for j in range(len(drhodr)):
        w0=np.where((nfinal[j]<0) | (~np.isfinite(nfinal[j])))[0]
        nfinal[j][w0]=0
        fefinal[j]=nfinal[j]/ge
    
    rhor=[np.zeros_like(rpts) for j in range(len(drhodr))]

    newfe=[interp1d(-egrid/satnorm,fefinal[j],fill_value='extrapolate') for j in range(len(drhodr))]


    highfit=[[] for j in range(len(drhodr))]
    nhighfit=np.zeros(len(drhodr)).astype(np.int)+10
    for j in range(len(drhodr)):
        while nhighfit[j]<20:
            try:
                highfit[j]=linefit.linefit(np.log10(1+egrid[-1-nhighfit[j]:]/satnorm),np.log10(fefinal[j][-1-nhighfit[j]:]))
            except:
                nhighfit[j]=nhighfit[j]+1
                continue
            if np.isfinite(highfit[j][0][0]) and np.isfinite(highfit[j][0][1]):
                break
            else:
                nhighfit[j]=nhighfit[j]+1

    plt.clf()
    for j in range(len(drhodr)):
        plt.plot(1+egrid/satnorm,fes[j],'--')
        plt.plot(1+egrid/satnorm,fefinal[j],alpha=.5)
    plt.loglog()
    plt.savefig(folder+'fetest.png')
#    plt.clf()

    for j in range(len(drhodr)):
        for i in range(len(rpts)):
            emax=satpotfunc(rpts[i])/satnorm
            rhor[j][i]=4*np.pi*quad(lambda x: newfe[j](x)*np.sqrt(2*(emax-x)),-min(egrid)/satnorm,emax)[0]


    nadjust=[]
    feadjust=[]
    gadjust=[]
    newphir=[]
    bestfits=[]
    
    for j in range(len(drhodr)):
        rhoint=interp1d(rpts,rhor[j])
        wge0=np.where(rhor[j]>=0)[0]
        mtot=np.trapz((rhor[j][wge0])*rpts[wge0]**2*4*np.pi,x=rpts[wge0])

        print 'mt',mtot
        try:
            mins=minimize(chi2,[np.log10(rhor[j][0]),1,1,3,0],args=(rhoint,mtot,rpts),method='SLSQP',bounds=[(np.log10(rhor[j][0])-3,np.log10(rhor[j][0])+3),(-2,6),(.1,3),(3,50),(0,1)])
            rho0halo,rscale,alpha,beta,gamma=mins['x']
            rho0halo=10**rho0halo
            rscale=10**rscale
            bestfits.append(np.array([rho0halo,rscale,alpha,beta,gamma])) 
        except:
            bestfits.append(np.array([0,1,1,3,0]))

        print bestfits

    newegrid=np.zeros(len(egrid))
    
    try:
        newphir1=lambda x:newphi(x,bestfits)
        newdphidr1=lambda x:newdphidr(x,bestfits)
        phi0=newphir1(minr)
        for i in range(len(newegrid)):
            rmax0=getrsi.getrsi(satpotfunc,egrid[i],satdphidr,minr=minr)
            avgphi0=getrvcorr.getavgphi(satpotfunc,egrid[i],rmax0,phiprime=satdphidr,minr=minr)

            rmax1=getrsi.getrsi(newphir1,egrid[i],satdphidr1,minr=minr)
            avgphi1=getrvcorr.getavgphi(satpotfunc1,egrid[i],rmax0,phiprime=newdphidr1,minr=minr)

            newegrid[i]=egrid[i]+(avgphi0-avgphi1)

        deltae0=np.append(egrid[1:]-egrid[0:-1],egrid[-1]-egrid[-2])
        newdeltae=np.append(newegrid[1:]-newegrid[0:-1],newegrid[-1]-newegrid[-2])

        
        nadjust=nfinal
        for j in range(len(drhodr)):
            for i in range(len(newegrid)):
                nadjust[j][i]=nfinal[j][i]*newdeltae[i]/deltae0[i]
        
        geadjust=[]
        for i in range(len(newegrid)):
            geadjust.append(getge.getge(lambda x: newphir1(x)/phi0,-newegrid[i]/phi0,lambda x:newdphidr1(x)/phi0,minr=minr))

    except:
        return [np.zeros(len(rpts)) for j in range(len(drhodr))]

    geadjust=np.array(geadjust)
    print 'ga',geadjust

    feadjust=[nadjust[j]/ge for j in range(len(drhodr))]
    print 'fa',feadjust

    plt.plot(1+egrid/satnorm,feadjust[0],'--')
    plt.savefig(folder+'fetestb.png')


    newfeadjust=[interp1d(-egrid/satnorm,feadjust[j],fill_value='extrapolate') for j in range(len(drhodr))]
    adjustrhor=[np.zeros_like(rpts) for j in range(len(drhodr))]
    
    for j in range(len(drhodr)):
        for i in range(len(rpts)):
            emax=newphir(rpts[i])/phi0
            print 'r',rpts[i],emax
            print 'em',newfeadjust[j](-min(egrid)/satnorm)
            try:
                adjustrhor[j][i]=4*np.pi*quad(lambda x: newfeadjust[j](x)*np.sqrt(2*(emax-x)),-min(newegrid)/phi0,emax)[0]
            except:
                adjustrhor[j][i]=0
        
            
    return adjustrhor
