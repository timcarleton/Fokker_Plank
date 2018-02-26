import dofp
reload(dofp)
import numpy as np
from astropy.constants import G
from SIDM_thermal_nob import get_sidm_fit_without_beffect, sidm_setup
import getafromsidm as getafromsidm
import getperi
from astropy import units
import getaefromperi
import getinfallinfo
import gethalomass
import masssize
import profileclass
reload(profileclass)
from scipy import interpolate
from scipy.optimize import minimize_scalar,minimize
from scipy.special import beta as betafunc
from scipy.special import gamma as gammafunc
from scipy.special import hyp2f1 as Hypergeometric2F1
import getrvcorr
import getomega
import getetide
import potdenfunc
import matplotlib.pyplot as plt
import getge
import getetide

kminkpc=1*units.kpc.to(units.km)

GN=(G.to(units.kpc**3/units.s**2/units.M_sun)).value

def munfw(r,rho0,rs,rvir):

    if r>rvir:
        return 1
    
    x=r/rs
    mr=4*np.pi*rho0*rs**3*(np.log(1+x)-r/(r+rs))
    mvir=4*np.pi*rho0*rs**3*(np.log(1+rvir/rs)-rvir/(rvir+rs))
    return mr/mvir

def muhatnfw(r,rho0,rs,rvir):

    if r>rvir:
        return 0

    x=r/rs
    mvir=4*np.pi*rho0*rs**3*(np.log(1+rvir/rs)-rvir/(rvir+rs))
    bottom=mvir/(4*np.pi*rho0*rs**3)
    return x**2/(1+x)**2/bottom


def rthetaint(dx,dy,dz,t,iperi):
    xint=interpolate.interp1d(t,dx,fill_value='extrapolate')
    yint=interpolate.interp1d(t,dy,fill_value='extrapolate')
    zint=interpolate.interp1d(t,dz,fill_value='extrapolate')

    mn=minimize_scalar(lambda dt: (xint(dt)**2+yint(dt)**2+zint(dt)**2),(t[iperi-2],t[iperi],t[iperi+2]))

    dot=(dx*xint(mn['x']))+(dy*yint(mn['x']))+(dz*zint(mn['x']))

    theta=np.arccos(dot/(np.sqrt(dx**2+dy**2+dz**2)*np.sqrt(mn['fun'])))
    wpre=np.arange(iperi,len(dx))

    theta[wpre]=-theta[wpre]
    plt.clf()
    plt.plot(t,np.sqrt(dx**2+dy**2+dz**2))
    plt.savefig('dt.png')
    plt.clf()
    
    dist=np.sqrt(dx**2+dy**2+dz**2)
    thetamin=np.min(theta)
    thetamax=np.max(theta)
    for i in range(iperi+1,len(theta)):
        if theta[i]<theta[i-1] or (dist[i]<dist[i-1] and dist[i-1]<dist[i-2]):
            thetamin=theta[i-1]

    for i in np.arange(iperi-1,0,-1):
        if theta[i]>theta[i+1] or (dist[i]<dist[i+1] and dist[i+1]<dist[i+2]):
            thetamax=theta[i+1]

    plt.clf()
    plt.plot(t,theta)
    plt.savefig('ttheta.png')
    plt.clf()



    return interpolate.interp1d(theta,np.sqrt(dx**2+dy**2+dz**2),fill_value='extrapolate'),thetamin,thetamax

def nfwomegar(r,rs,rho0):
    return np.sqrt(4*np.pi*GN*rho0*rs**3*(np.log(1+r/rs)/r**3-r/(rs+r)/r**3))

def jthetaint(dx,dy,dz,vx,vy,vz,t):
    xint=interpolate.interp1d(t,dx,fill_value='extrapolate')
    yint=interpolate.interp1d(t,dy,fill_value='extrapolate')
    zint=interpolate.interp1d(t,dz,fill_value='extrapolate')

    mn=minimize_scalar(lambda dt: (xint(dt)**2+yint(dt)**2+zint(dt)**2))

    dot=(dx*xint(mn['x']))+(dy*yint(mn['x']))+(dz*zint(mn['x']))

    wpre=np.where(t<mn['x'])[0]
    wpost=np.where(t>=mn['x'])[0]
    theta=np.arccos(dot/(np.sqrt(dx**2+dy**2+dz**2)*np.sqrt(mn['fun'])))
    theta[wpost]=-theta[wpost]

    omega=getomega.getomega_direct(vx,vy,vz,dx,dy,dz)

    return interpolate.interp1d(theta,omega*(dx**2+dy**2+dz**2),fill_value=(omega[0]*(dx[0]**2+dy[0]**2+dz[0]**2),omega[-1]*(dx[-1]**2+dy[-1]**2+dz[-1]**2)),bounds_error=False)


def rhors10(params):
    print 'rho10'
    print params
    print zhaorho(243/2.163*10,params[0],params[1],params[2],params[3],params[4])
    return zhaorho(243/2.163*10,params[0],params[1],params[2],params[3],params[4])
    
def mrs(params):
    print 'ms'
    print params
    print profileclass.getmassfromzhao0(params[2],params[3],params[4],params[0],params[1],243/2.163*2)
    return profileclass.getmassfromzhao0(params[2],params[3],params[4],params[0],params[1],243/2.163*2)


def chi2(params,rhoint,massint,rscale,r,gammapenalty=False):

    rhor=lambda params,r: zhaorho(r,10**params[0],10**params[1],params[2],params[3],params[4])

    w=np.where(rhoint(r)>100)[0]
    mtot=lambda params: profileclass.getmassfromzhao0(params[2],params[3],params[4],10**params[0],10**params[1],r[w[-1]])

    d2=(np.log10(rhor(params,r[w]))-np.log10(rhoint(r[w])))**2+(np.log10(mtot(params))-np.log10(massint))**2
    if np.any(~np.isfinite(d2)):
        return np.inf

    if gammapenalty:
        return np.nansum(d2)+(d2[4]-params[4])**2
    else:
        return np.nansum(d2)
    
def fpwrapper(f,fh,todo=[1548,278,838,1233,786,1082,542,329,1227,885,1076,281,924,961,826,1436,64,1583,999,504,343,1120,1204,1106],taufactor=1.0,getangmom='one',folder=['./'],alphastart=1,betastart=4,gammastart=0):

    if len(folder)==len(todo):
        svfolder=['' for i in range(len(f[1].data))]
    
        for i in range(len(todo)):
            svfolder[todo[i]]=folder[i]
    
    else:
        svfolder=folder
        
    peris=getperi.getperi(f)

    
    iinf,rmaxinf,vmaxinf,mpeak,rvirinf=getinfallinfo.getinfallinfo(f,cross=1,minr=0.1,maxr=200,inrs=False,geta=False,mfield='mvir',alpha=alphastart,beta=betastart)

    print 'a'

    ainf0=np.zeros(len(f[1].data))
    rho0inf0=np.zeros(len(f[1].data))
    rscale0=np.zeros(len(f[1].data))
    alpha0=np.zeros(len(f[1].data))
    beta0=np.zeros(len(f[1].data))
    gamma0=np.zeros(len(f[1].data))
    

    fit0=[]
    
    for i in todo:
        asidm=getafromsidm.getafromsidm(vmaxinf[i]*units.km/units.s,rmaxinf[i]*units.kpc,f[1].data.time[i,iinf[i]]*units.Gyr,rvirinf[i]*units.kpc,gammastart,alpha=alphastart,beta=betastart)
        #ainf0.append(int(asidm[0]))
        
        ainf0[i]=alphastart
        alpha0[i]=alphastart
        beta0[i]=betastart
        gamma0[i]=gammastart
        
        rho0inf0[i]=asidm[1]
        rscale0[i]=rmaxinf[i]/2.163

        fit0.append(np.array([rho0inf0[i],rscale0[i],alpha0[i],beta0[i],gamma0[i]]))

    print 'b'
    bestfits=[]
    ifit=0

    for i in todo:
        print 'i',i
        print peris[2][i]
        for j in range(len(peris[2][i])):

            ih=np.where(f[1].data.hostid0[i]==fh[1].data.id0)[0][0]
            ip=peris[2][i][j]
            
            tophysicalsize=1.0/.7/(1+fh[1].data.redshift[ih][ip])
            tophysicalmass=1.0/.7

            
            hostrho0=profileclass.getnfwrho0(fh[1].data.mvir[ih][ip]*tophysicalmass,fh[1].data.rs[ih][ip]*tophysicalsize,fh[1].data.rvir[ih][ip]*tophysicalsize)
            muhosti=lambda x: munfw(x,hostrho0,fh[1].data.rs[ih][ip]*tophysicalsize,fh[1].data.rvir[ih][ip]*tophysicalsize)
            muhosthati=lambda x: muhatnfw(x,hostrho0,fh[1].data.rs[ih][ip]*tophysicalsize,fh[1].data.rvir[ih][ip]*tophysicalsize)


            whalo=np.where(np.isfinite(f[1].data.dhost[int(i)]))[0]
            nptslow=np.min([200,peris[2][i][j]]-np.min(whalo))
            nptshigh=np.min([200,np.max(whalo)-peris[2][i][j]])

            plt.clf()
            plt.plot(f[1].data.time[i][ip-nptslow:ip+nptshigh],f[1].data.dxhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize)
            plt.plot(f[1].data.time[i][ip-nptslow:ip+nptshigh],f[1].data.dyhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize)
            plt.plot(f[1].data.time[i][ip-nptslow:ip+nptshigh],f[1].data.dzhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize)
            plt.savefig(svfolder[i]+'post.png')

            rthetai,thetamin,thetamax=rthetaint(f[1].data.dxhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize,
                                       f[1].data.dyhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize,
                                       f[1].data.dzhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize,
                                       f[1].data.time[i][ip-nptslow:ip+nptshigh],nptslow)
     
            plt.clf()
            plt.plot(np.linspace(-np.pi,np.pi),rthetai(np.linspace(-np.pi,np.pi)))
            plt.ylim(0,10000)
            plt.savefig(svfolder[i]+'rtheta.png')
            
            angmomtot=lambda x: jthetaint(f[1].data.dxhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize,
                                          f[1].data.dyhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize,
                                          f[1].data.dzhost[i][ip-nptslow:ip+nptshigh]*1000*tophysicalsize,
                                          f[1].data.vxwrthost[i][ip-nptslow:ip+nptshigh]/kminkpc,
                                          f[1].data.vywrthost[i][ip-nptslow:ip+nptshigh]/kminkpc,
                                          f[1].data.vzwrthost[i][ip-nptslow:ip+nptshigh]/kminkpc,
                                          f[1].data.time[i][ip-nptslow:ip+nptshigh])(x)
            
            print 'a'
            print peris[1][i][j],rthetai(0)
            hostphi=lambda r:4*np.pi*hostrho0*GN*(fh[1].data.rs[ih][ip]*tophysicalsize)**2/r*np.log(1+r/(fh[1].data.rs[ih][ip]*tophysicalsize))

            if getangmom=='avg':
                sate=0.5/kminkpc**2*(f[1].data.vxwrthost[i][ip-5:ip+5]**2+f[1].data.vywrthost[i][ip-5:ip+5]**2+f[1].data.vzwrthost[i][ip-5:ip+5]**2)-hostphi(f[1].data.dhost[i][ip-5:ip+5]*tophysicalsize)
                vp=np.sqrt(2*(np.mean(sate)-hostphi(peris[1][i][j]*1000*tophysicalsize)))
                angmom=lambda x:(peris[1][i][j]*1000*tophysicalsize*vp)/np.sqrt(GN*fh[1].data.rs[ih][ip]*tophysicalsize*fh[1].data.mvir[ih,ip]*tophysicalmass)
            else:
                angmom=lambda x:(peris[1][i][j]*1000*tophysicalsize)**2*peris[4][i][j]/np.sqrt(GN*fh[1].data.rs[ih][ip]*tophysicalsize*fh[1].data.mvir[ih,ip]*tophysicalmass)

            eorbit=getetide.getetideorbit(muhosti,muhosthati,rthetai,fh[1].data.rs[ih][ip]*tophysicalsize,angmom,fh[1].data.mvir[ih,ip]*tophysicalmass,thetamin=thetamin,thetamax=thetamax,folder=svfolder[i])
                        
            if j==0:
                alpha=alpha0[i]
                beta=beta0[i]
                gamma=gamma0[i]
                rscale=rscale0[i]
                rho0halo=rho0inf0[i]
                

            satpotfunc=lambda x: potdenfunc.zhaophi(x,rho0halo,rscale,alpha,beta,gamma)
            satdphidr=lambda x: potdenfunc.dphidrzhao(x,rho0halo,rscale,alpha,beta,gamma)
            satd2phidr2=lambda x: potdenfunc.dphi2dr2zhao(x,rho0halo,rscale,alpha,beta,gamma)
            satdrhodr=lambda x: potdenfunc.drhodrzhao(x,rho0halo,rscale,alpha,beta,gamma)
            satd2rhodr2=lambda x: potdenfunc.d2rhodr2zhoa(x,rho0halo,rscale,alpha,beta,gamma)

            #r12halo=minimize_scalar(lambda x: abs(profileclass.getmassfromzhao0(alpha,beta,gamma,rho0halo,rscale,x)-0.5*profileclass.getmassfromzhao0(alpha,beta,gamma,rho0halo,rscale,peris[1][i][j]*1000)),(rscale/10000,rscale,10000*rscale))['x']
            r12halo=rscale*(2**(1.0/3)-1)**-1
            tau12=1.0/np.sqrt(GN*profileclass.getmassfromzhao0(alpha,beta,gamma,rho0halo,rscale,peris[1][i][j]*1000)*3/8.0/np.pi/r12halo**3)

            rpts=np.logspace(-1.2,1,num=30)*rmaxinf[i]/2.163

            #rhof=dofp.dofp(rthetai,muhosti,muhosthati,satpotfunc,satdphidr,satd2phidr2,[satdrhodr],[satd2rhodr2],fh[1].data.rs[ih][ip]*tophysicalsize,peris[4][i][j],f[1].data.mvir[ih,ip]*tophysicalmass,peris[1][i][j]*1000,rpts,gamma,angmom,orbit=eorbit,taufactor=taufactor,folder=svfolder[i],rhoparams=fit0[ifit])[0]
            rhof=dofp.dofp(rthetai,muhosti,muhosthati,satpotfunc,satdphidr,satd2phidr2,[satdrhodr],[satd2rhodr2],fh[1].data.rs[ih][ip]*tophysicalsize,peris[4][i][j],f[1].data.mvir[ih,ip]*tophysicalmass,peris[1][i][j]*1000,rpts,gamma,angmom,orbit=eorbit,taufactor=taufactor,folder=svfolder[i])[0]

            print rpts,rhof
                
            rhoint=interpolate.interp1d(rpts,rhof)
            wge0=np.where(rhof>=0)[0]
            mtot=np.trapz((rhof[wge0])*rpts[wge0]**2*4*np.pi,x=rpts[wge0])

            print 'mt',mtot

            try:
                mins=minimize(chi2,[np.log10(rho0halo)-.5,np.log10(rscale),alpha,beta,gamma],args=(rhoint,mtot,rscale,rpts),method='SLSQP',bounds=[(np.log10(rho0halo)-3,np.log10(rho0halo)+3),(-2,6),(.1,3),(3,50),(0,3)])
            except:
                break

                
            plt.clf()
            plt.plot(np.logspace(-1,1.5)*rmaxinf[i]/2.163,zhaorho(np.logspace(-1,1.5)*rmaxinf[i]/2.163,rho0inf0[i],rmaxinf[i]/2.163,fit0[ifit][2],fit0[ifit][3],fit0[ifit][4]),label='before delta e')
            plt.loglog()
            plt.plot(np.logspace(-1,1.5)*rmaxinf[i]/2.163,zhaorho(np.logspace(-1,1.5)*rmaxinf[i]/2.163,10**mins['x'][0],10**mins['x'][1],mins['x'][2],mins['x'][3],mins['x'][4]),alpha=.2,label='fit 1')
            plt.plot(rpts,rhoint(rpts),label='after delta e')
            plt.legend()
            plt.ylim(100,1E9)
            p=profileclass.Zhao(mins['x'][2],mins['x'][3],mins['x'][4],rho0=10**mins['x'][0]*units.M_sun/units.kpc**3,rs=10**mins['x'][1]*units.kpc,deltavirrhou=30000*units.M_sun/units.kpc**3)
            p0=profileclass.Zhao(fit0[ifit][2],fit0[ifit][3],fit0[ifit][4],rho0=rho0inf0[i]*units.M_sun/units.kpc**3,rs=rmaxinf[i]/2.163*units.kpc,deltavirrhou=30000*units.M_sun/units.kpc**3)
            plt.title(r'$v_{\rm max,f}/v_{\rm max,i}=%1.2f,~r_{\rm max,f}/r_{\rm max,i}=%1.2f$' % ((p.get_vmax()/p0.get_vmax()).value,(p.get_rmax()/p0.get_rmax()).value))
            plt.tight_layout()
            plt.savefig(svfolder[i]+'recoverytest.png')
            
            
            rho0halo,rscale,alpha,beta,gamma=mins['x']
            rho0halo=10**rho0halo
            rscale=10**rscale
                    
            rho0inf0[i]=rho0halo
            rscale0[i]=rscale
            alpha0[i]=alpha
            beta0[i]=beta
            gamma0[i]=gamma

            
        try:
            bestfits.append(np.array([rho0halo,rscale,alpha,beta,gamma]))
        except:
            bestfits.append(np.array([rho0halo0,rscale0,alpha0,beta0,gamma0]))

        ifit=ifit+1
        
    return bestfits,fit0

test=False
if test:
    alpha=1
    beta=4
    gamma=0
    sifunc=lambda r: zhaophi(r,3498407.72641,243.027861857,alpha,beta,gamma)/zhaophi(0,3498407.72641,243.027861857,alpha,beta,gamma)
    drhodr=lambda r: drhodrzhao(r,3498407.72641,243.027861857,alpha,beta,gamma)
    d2rhodr2=lambda r: d2rhodr2zhoa(r,3498407.72641,243.027861857,alpha,beta,gamma)
    dsidr=lambda r: dphidrzhao(r,3498407.72641,243.027861857,alpha,beta,gamma)/zhaophi(0,3498407.72641,243.027861857,alpha,beta,gamma)
    d2sidr2=lambda r: dphi2dr2zhao(r,3498407.72641,243.027861857,alpha,beta,gamma)/zhaophi(0,3498407.72641,243.027861857,alpha,beta,gamma)
    print sifunc(2)
    print d2sidr2(2)
    print getetide.getrfrompot(lambda x: sifunc(x),.01,dsidr)

    fe0=getfe.getfe(sifunc,drhodr,d2rhodr2,dsidr,d2sidr2,1E-2)
    ge0=np.array([getge.getge(sifunc,i,dsidr) for i in np.linspace(.01,1)])
    print ge0
    plt.plot(np.linspace(.01,1),ge0*fe0(np.linspace(.01,1)))
    plt.yscale('log')
 #   plt.ylim(1E10,1E14)
    plt.show()
    
    ## alpha=1
    ## beta=4
    ## gamma=1
    ## sifunc=lambda r: zhaophi(r,3498407.72641,243.027861857,alpha,beta,gamma)/zhaophi(0,3498407.72641,243.027861857,alpha,beta,gamma)
    ## drhodr=lambda r: drhodrzhao(r,3498407.72641,243.027861857,alpha,beta,gamma)
    ## d2rhodr2=lambda r: d2rhodr2zhoa(r,3498407.72641,243.027861857,alpha,beta,gamma)
    ## dsidr=lambda r: dphidrzhao(r,3498407.72641,243.027861857,alpha,beta,gamma)/zhaophi(0,3498407.72641,243.027861857,alpha,beta,gamma)
    ## print dphidrzhao(2,3498407.72641,243.027861857,alpha,beta,gamma)
    ## d2sidr2=lambda r: dphi2dr2zhao(r,3498407.72641,243.027861857,alpha,beta,gamma)/zhaophi(0,3498407.72641,243.027861857,alpha,beta,gamma)
    ## print d2sidr2(2)

    ## fe1=getfe.getfe(sifunc,drhodr,d2rhodr2,dsidr,d2sidr2,1E-3)
    ## psis=np.linspace(.01,.999)
    ## ge0=np.array([getge.getge(sifunc,i) for i in np.linspace(0,1)])
    ## print ge0

    ## plt.plot(np.linspace(0,1),ge0)
    ## plt.plot(psis,fe1(psis)*1E5)
    ## plt.yscale('log')
    ## plt.ylim(1E7,1E12)
    ## plt.show()
   # plt.plot(psis,fe0(psis))
    #plt.plot(psis,fe1(psis*3))
    #plt.yscale('log')
   # plt.show()
