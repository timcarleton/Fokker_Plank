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
    plt.savefig(folder+'deltaevseb.png')

    nfinal=[np.zeros_like(egrid) for i in range(len(drhodr))]
    firsts=[np.zeros_like(egrid) for i in range(len(drhodr))]
    seconds=[np.zeros_like(egrid) for i in range(len(drhodr))]
    
    ne0=[np.zeros_like(egrid) for i in range(len(drhodr))]

    fes=[np.zeros_like(egrid) for i in range(len(drhodr))]
    ge=np.zeros_like(egrid)
    
    for i in range(len(egrid)):
        
        ge[i]=getge.getge(lambda x: satpotfunc(x)/satnorm,-egrid[i]/satnorm,lambda x:satdphidr(x)/satnorm,minr=minr)

        highe=.99
        xi=1E-5
        sifromt=lambda t:1-t**2
        tfromu=lambda u:(u+xi)/(1+u*xi)
        sifromu=lambda u: 1-((u+xi)/(1+u*xi))**2
        nhighfit=10
        
        if -egrid[i]/satnorm<highe:
            for j in range(len(drhodr)):
                fes[j][i]=fedist[j](-egrid[i]/satnorm)
                ne0[j][i]=ge[i]*fes[j][i]
            em=-egrid[i]/satnorm
           # print fes[i],quad(lambda x: getfe.d2rhodsi2(lambda x: satpotfunc(x)/satnorm,drhodr,d2rhodr2,lambda x: satdphidr(x)/satnorm,lambda x: satd2phidr2(x)/satnorm,x)/np.sqrt(em-x),0,em)[0]
            ihigh=i
        else:
            for j in range(len(drhodr)):
                if not np.any(np.isfinite(np.log10(fes[j][ihigh-nhighfit+1:ihigh+1]))):
                    fes[j][i]=0
                    ne0[j][i]=0
                else:
                    highfit=linefit.linefit(np.log10(1+egrid[ihigh-nhighfit+1:ihigh+1]/satnorm),np.log10(fes[j][ihigh-nhighfit+1:ihigh+1]))
                    fes[j][i]=10**(highfit[0][0]*np.log10(1+egrid[i]/satnorm)+highfit[0][1])
                    ne0[j][i]=ge[i]*fes[j][i]


            #ne0[i]=ge[i]*getfe.getfehighzhao(highe,fedist(-highe/satnorm),'a',egrid[i],gamma)
            em=-egrid[i]/satnorm
            #fes[i]=quad(lambda t: getfe.d2rhodsi2(lambda x: satpotfunc(x)/satnorm,drhodr,d2rhodr2,lambda x: satdphidr(x)/satnorm,lambda x: satd2phidr2(x)/satnorm,sifromt(t))/np.sqrt(t**2-(1-em**2))*2*t,np.sqrt(1-em**2),1)[0]/np.sqrt(8)/np.pi**2
            #fes[i]=quad(lambda u: getfe.d2rhodsi2(lambda x: satpotfunc(x)/satnorm,drhodr,d2rhodr2,lambda x: satdphidr(x)/satnorm,lambda x: satd2phidr2(x)/satnorm,sifromu(u))/np.sqrt(tfromu(u)**2-(1-em**2))*2*tfromu(u)*(1-xi**2)/(1+u*xi)**2,(np.sqrt(1-em**2)-xi)/(1+xi*np.sqrt(1-em**2)),(1-xi)/(1+xi))[0]/np.sqrt(8)/np.pi**2
#            fes[i]=getfe.getfehighzhao(highe-.02,fedist(highe-.02),'a',-egrid[i]/satnorm,gamma)

            #fes[i]=quad(lambda x: getfe.d2rhodsi2(lambda x: satpotfunc(x)/satnorm,drhodr,d2rhodr2,lambda x: satdphidr(x)/satnorm,lambda x: satd2phidr2(x)/satnorm,x)/np.sqrt(em-x),0,em)[0]/np.sqrt(8)/np.pi**2
            
            
            #ne0[i]=ge[i]*quad(lambda x: getfe.d2rhodsi2(lambda x: satpotfunc(x)/satnorm,drhodr,d2rhodr2,lambda x: satdphidr(x)/satnorm,lambda x: satd2phidr2(x)/satnorm,x)/np.sqrt(em-x),0,em)[0]

        #print 'eg',egrid[i],ge
        #deltae=0
        #deltae2=0
        #print deltae
        #print deltae2

        for j in range(len(drhodr)):
            firsts[j][i]=-deltae[i]*ne0[j][i]
            seconds[j][i]=-deltae2[i]*ne0[j][i]
     #dt=.01
    #deltan=np.zeros_like(egrid)

    #firsts=-deltae*ne0
    #seconds=-deltae2*ne0
    #print 'de',deltae

#    plt.clf()
#    plt.plot(1-egrid/satnorm,abs(firsts[0])[::-1])
#    plt.plot(1-egrid/satnorm,abs(deltae)[::-1])
#    plt.plot(1-egrid/satnorm,abs(seconds[0])[::-1],'--')
#    plt.plot(1-egrid/satnorm,abs(deltae2)[::-1],'--')
#    plt.loglog()
#    plt.savefig(folder+'efirst.png')
    
#    plt.clf()
#    plt.plot(1+egrid/satnorm,np.log10(abs(-np.gradient(firsts[j],egrid))))
#    plt.plot(1+egrid/satnorm,np.log10(abs(0.5*np.gradient(np.gradient(seconds[j],egrid),egrid))))
#    plt.plot(1+egrid/satnorm,np.log10(abs(0.5*np.gradient(1E25*seconds[j],egrid))))
#    plt.xscale('log')



    deltan=[]
    for j in range(len(drhodr)):

        dsecond=np.gradient(seconds[j],egrid)
        dsecondsmooth=convolve(dsecond,Box1DKernel(10), boundary='extend')
        plt.clf()
        plt.plot(egrid,dsecond)
        plt.plot(egrid,abs(dsecondsmooth))
        plt.loglog()
        plt.savefig(folder+'d2smx.png')

        plt.clf()
        plt.plot(egrid,abs(firsts[j]))
        plt.plot(egrid,abs(seconds[j]))
        plt.loglog()
        plt.savefig(folder+'d2smx2.png')

        dn=-np.gradient(firsts[j],egrid)+.5*np.gradient(dsecondsmooth,egrid)
        plt.clf()
        plt.plot(egrid,abs(-np.gradient(firsts[j],egrid)))
        plt.plot(egrid,abs(.5*np.gradient(dsecondsmooth,egrid)))
#        plt.xlim(.9,1)
        plt.loglog()
        plt.savefig(folder+'deltaes.png')
#        w=np.where(dn>np.max(ne0))[0]
#        dn[w]=np.nan
        #dnsmooth=convolve(dn,Box1DKernel(10))
        deltan.append(dn)
#        deltan.append(-np.gradient(firsts[j],egrid)+.5*np.gradient(10**dsecondsmooth,np.linspace(min(egrid),max(egrid),len(dsecondsmooth))))

#        print dsecond,10**dsecond,np.gradient(10**dsecond,egrid)
#        print np.log10(abs(np.gradient(10**dsecond,egrid)))
        plt.clf()
        plt.plot(1+egrid/satnorm,np.gradient(firsts[j],egrid))
        plt.plot(1+np.linspace(min(egrid),max(egrid),abs(len(dsecondsmooth)))/satnorm,abs(np.gradient(dsecondsmooth,np.linspace(min(egrid),max(egrid),len(dsecondsmooth)))))
        plt.loglog()
        #plt.plot(1+egrid/satnorm,np.log10(abs(np.gradient(10**dsecond,egrid))))
#        deltan.append(firsts[j]/egrid*dlogfirst+0.5*(seconds[j]/egrid**2*(dlogsecond**2-dlogsecond+d2logsecond)))
    plt.savefig(folder+'defirst.png')


    nfinal=[ne0[j]+deltan[j] for j in range(len(drhodr))]

    for j in range(len(drhodr)):
        w0=np.where((nfinal[j]<0) | (~np.isfinite(nfinal[j])))[0]
        nfinal[j][w0]=0

    fefinal=[np.zeros_like(egrid) for j in range(len(drhodr))]

    for j in range(len(drhodr)):
        fefinal[j]=nfinal[j]/ge

    #plt.show()
    
    rhor=[np.zeros_like(rpts) for j in range(len(drhodr))]

    #egrid2=np.linspace(3E-3,1-3E-3)
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
        plt.plot(1+egrid/satnorm,24*np.sqrt(2)/7.0/np.pi**3/GN**5/(1E8)**4*(-egrid)**3.5)
#        plt.plot(1+egrid/satnorm,10**(highfit[j][0][1]+highfit[j][0][0]*np.log10(1+(egrid/satnorm))),alpha=.3)
    #plt.xlim(.9,1)
    #plt.xlim(.9*satnorm,1*satnorm)
    plt.yscale('log')
    plt.loglog()
#    plt.ylim(1E4,1E12)
    plt.savefig(folder+'fetest.png')
#    plt.clf()
    #plt.show()


#    print 'nf',deltan,nfinal,fefinal
    
#    print 'ff',fefinal,satpotfunc(rpts)/satnorm

    for j in range(len(drhodr)):
        for i in range(len(rpts)):
            emax=satpotfunc(rpts[i])/satnorm
            print 'r',rpts[i],emax
#            if emax>-egrid[-1-nhighfit[j]]/satnorm:
#                print 'em',highfit,emax,-egrid[-1-nhighfit[j]]/satnorm
#                rhor[j][i]=4*np.pi*quad(lambda x: newfe[j](x)*np.sqrt(2*(emax-x)),-min(egrid)/satnorm,egrid[-1-nhighfit[j]])[0]
#                high=emax
#                low=-egrid[-1-nhighfit[j]]/satnorm
#                if np.isfinite(highfit[j][0][1]) and np.isfinite(highfit[j][0][0]):
#                    rhor[j][i]=rhor[j][i]+10**(highfit[j][0][1])*(highfit[j][0][0]+1)*((1+high/satnorm)**(highfit[j][0][0]+1)-(1+low/satnorm)**(highfit[j][0][0]+1))
#                fefunc=lambda x: 10**(highfit[j][0][1]+highfit[j][0][0]*np.log10(1-x))
#                fefunc=lambda x: np.nan
                #if np.isfinite(highfit[j][0][1]) and np.isfinite(highfit[j][0][0]):
                    #fefunc=lambda x: 10**(highfit[j][0][1]+highfit[j][0][0]*np.log10(1-x))
#                fefunc=newfe[j]
#            else:
#                fefunc=newfe[j]

#            print 'va',emax
#            print [2*(emax-k) for k in np.linspace(-min(egrid)/satnorm,emax)]
#            print [fefunc(k)*np.sqrt(2*(emax-k)) for k in np.linspace(-min(egrid)/satnorm,emax)]
#            print [np.log10(fefunc(k)) for k in np.linspace(-min(egrid)/satnorm,emax)]
#            print [np.log10(1-k) for k in np.linspace(-min(egrid)/satnorm,emax)]
            print 'em',newfe[j](-min(egrid)/satnorm)
            print newfe[j](-min(egrid)/satnorm), newfe[j](.5*emax), newfe[j](emax)
            print quad(lambda x: newfe[j](x)*np.sqrt(2*(emax-x)),-min(egrid)/satnorm,emax)
            rhor[j][i]=4*np.pi*quad(lambda x: newfe[j](x)*np.sqrt(2*(emax-x)),-min(egrid)/satnorm,emax)[0]

    #return rhor

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



#        plt.clf()
        
#        plt.plot(np.logspace(-1,1.5)*rmaxinf[i]/2.163,zhaorho(np.logspace(-1,1.5)*rmaxinf[i]/2.163,rho0inf0[i],rmaxinf[i]/2.163,ainf0[i],3,0),label='before delta e')

#        plt.plot(np.logspace(-1,1.5)*rmaxinf[i]/2.163,zhaorho(np.logspace(-1,1.5)*rmaxinf[i]/2.163,rho0halo,,alpha,beta,gamma),'--',alpha=.2,label='before delta ei 1')

#        plt.loglog()
        
#        plt.plot(np.logspace(-1,1.5)*rmaxinf[i]/2.163,zhaorho(np.logspace(-1,1.5)*rmaxinf[i]/2.163,10**mins['x'][0],10**mins['x'][1],mins['x'][2],mins['x'][3],mins['x'][4]),alpha=.2,label='fit 1')
#        plt.plot(rpts,rhoint(rpts),label='after delta e')
#        plt.legend()
#        plt.ylim(100,1E9)
#        plt.savefig(folder+'recoverytest.png')
        print bestfits
        

    try:
        newphir=lambda x:newphi(x,bestfits)
        phi0=newphir(minr)
        print 'p0',phi0
        print newphir(.01)/phi0
        denew=[]
        for j in range(len(drhodr)):
            denew.append(1-np.gradient(firsts[j]/ne0[j],egrid))
        print 'de',denew
        #nadjust=[nfinal[j]/denew[j] for j in range(len(drhodr))]
        nadjust=[nfinal[j]*satnorm/phi0 for j in range(len(drhodr))]
        #nadjust=nfinal
        print 'na',nadjust
        geadjust=[]
        for i in range(len(egrid)):
            geadjust.append(getge.getge(lambda x: newphir(x)/phi0,-egrid[i]/satnorm,lambda x:newdphidr(x,bestfits)/phi0,minr=minr))
        print geadjust
    except:
        return [np.zeros(len(rpts)) for j in range(len(drhodr))]

    geadjust=np.array(geadjust)
    print 'ga',geadjust

    feadjust=[nadjust[j]/ge for j in range(len(drhodr))]
    print 'fa',feadjust

    plt.plot(1+egrid/satnorm,feadjust[0],'--')

    newfeadjust=[interp1d(-egrid/satnorm,feadjust[j],fill_value='extrapolate') for j in range(len(drhodr))]
    plt.savefig(folder+'fetestb.png')
    adjustrhor=[np.zeros_like(rpts) for j in range(len(drhodr))]
    for j in range(len(drhodr)):
        for i in range(len(rpts)):
            emax=newphir(rpts[i])/phi0
            print 'r',rpts[i],emax
            print 'em',newfeadjust[j](-min(egrid)/satnorm)
            print newfeadjust[j](-min(egrid)/satnorm), newfeadjust[j](.5*emax), newfeadjust[j](emax)
            print quad(lambda x: newfeadjust[j](x)*np.sqrt(2*(emax-x)),-min(egrid)/satnorm,emax)
            try:
                adjustrhor[j][i]=4*np.pi*quad(lambda x: newfeadjust[j](x)*np.sqrt(2*(emax-x)),-min(egrid)/satnorm,emax)[0]
            except:
                adjustrhor[j][i]=0
        
            
    return adjustrhor
