from astropy.io import fits
import numpy as np
#from SIDM_thermal_nob import get_sidm_fit_without_beffect, sidm_setup
#from scipy.interpolate import InterpolatedUnivariateSpline
import getzinfall
import datetime
import minnfw
import get_sidm_da
import getafromsidm
reload(getafromsidm)
from astropy import units

def getinfallinfo(f,cross=1.0,minr=0.1,maxr=200,inrs=True,start='inf',mfield='mvir',rfield='rs_klypin',geta=True,gamma=1,alpha=1,beta=4,h=.7,rstar=1):
    
    profiles=[]
    print len(f[1].data),rstar
    rmax=[]
    vmax=[]
    minf=[]
    rsinf=[]
    rvinf=[]
    iinf=[]
    ainf=[]
    rho0inf=[]

    for i in range(len(f[1].data)):
        if start=='max':
            winf=np.nanargmax(f[1].data.field(mfield)[i])
        elif start=='inf':
            winf=np.where(f[1].data.dhost[i]/f[1].data.rvirhost[i]*1E3<1)[0]
            if len(winf)==0 and start=='inf':
                winf=0
            else:        
                winf=winf[-1]
        elif start=='start':
            winf=np.where(np.isfinite(f[1].data.field(mfield)[i]))[0]
            if len(winf)==0 and start=='inf':
                winf=0
            else:        
                winf=winf[-1]
        else:
            print 'choose other start option'
            return
        wborn=np.where(~np.isfinite(f[1].data.mvir[i]))[0]
        if geta:
            if len(wborn)==0:
                ag=f[1].data.time[i][winf]#-f[1].data.time[i][-1]
#                ag=f[1].data.time[i][0]-f[1].data.time[i][-1]
            else:
                ag=f[1].data.time[i][winf]#-f[1].data.time[i][wborn[0]-1]
#                ag=f[1].data.time[i][0]-f[1].data.time[i][wborn[0]-1]

#        infparams=(f[1].data.vmax[i,winf],2.163*f[1].data.rs[i,winf],cross,ag,minr,maxr)
        
#        iso_params,r,mass,rho,mass_nfw,rho_nfw,res=get_sidm_fit_without_beffect(infparams)
        
        rmax.append(2.163*f[1].data.field(rfield)[i,winf]/(1+f[1].data.redshift[i,winf])/h)
#        rmax.append(2.163*f[1].data.field(rfield)[i,winf])
        vmax.append(f[1].data.vmax[i,winf])
        minf.append(f[1].data.field(mfield)[i,winf]/h)
        rsinf.append(f[1].data.field(rfield)[i,winf]/(1+f[1].data.redshift[i,winf])/h)
        rvinf.append(f[1].data.rvir[i,winf]/(1+f[1].data.redshift[i,winf])/h)
        iinf.append(winf)
        if geta:
            sidminf=getafromsidm.getafromsidm(f[1].data.vmax[i,winf]*units.km/units.s,2.163*rsinf[i]*units.kpc,ag*units.Gyr,f[1].data.rvir[i,winf]/h/(1+f[1].data.redshift[i,winf])*units.kpc,gamma,cross=cross,alpha=alpha,beta=beta,rmatch=rstar[i])
            ainf.append(sidminf[0])
            rho0inf.append(sidminf[1])
            if (i%100000)==10:
                print i,datetime.datetime.now()
    if not inrs:
        if geta:
            return np.array(iinf),np.array(rmax),np.array(vmax),np.array(minf),np.array(rvinf),np.array(ainf),np.array(rho0inf)
        else:
            return np.array(iinf),np.array(rmax),np.array(vmax),np.array(minf),np.array(rvinf)
    else:
        if geta:
            return np.array(iinf),np.array(rmax),np.array(vmax),np.array(vmax)**2*np.array(rmax),np.array(rvinf),np.array(ainf),np.array(rho0inf)#minnfw.minnfw(np.array(minf),np.array(rsinf),np.array(rvinf),np.array(rsinf)),np.array(rvinf)
        else:
            return np.array(iinf),np.array(rmax),np.array(vmax),np.array(vmax)**2*np.array(rmax),np.array(rvinf)#minnfw.minnfw(np.array(minf),np.array(rsinf),np.array(rvinf),np.array(rsinf)),np.array(rvinf)
#    return np.array(rmax),np.array(vmax),np.nanmax(f[1].data.mvir,axis=1)
