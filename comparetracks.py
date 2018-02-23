import domasslossp10coma
import fpwrapersim as fpwraper
reload(fpwraper)
from astropy.io import fits
import profileclass
from astropy import units
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import rubiatracks
import numpy as np
import os


plt.rcParams["figure.figsize"]=[16,12]
plt.rcParams['font.size']=30
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth']=4
plt.rcParams['axes.labelpad']=10
plt.rcParams['xtick.major.size']=9
plt.rcParams['ytick.major.size']=9
plt.rcParams['xtick.major.width']=3
plt.rcParams['ytick.major.width']=3
plt.rcParams['xtick.minor.size']=6
plt.rcParams['ytick.minor.size']=6
plt.rcParams['xtick.minor.width']=1
plt.rcParams['ytick.minor.width']=1
plt.rcParams['ytick.major.pad']=5
plt.rcParams['xtick.major.pad']=10
plt.rcParams['axes.labelsize']=35
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['savefig.bbox']='tight'
plt.rcParams['savefig.dpi']=200
plt.rcParams['lines.linewidth']=4
plt.rcParams['lines.markersize']=10
plt.rcParams['errorbar.capsize']=0
plt.rcParams['hatch.linewidth']=2
fig,ax=plt.subplots(1,1)

f=fits.open('cutsubs_top2mass.fits')
fh=fits.open('cuthosts_top2.fits')

import getperi
peris=getperi.getperi(f)
w1peri=[]
for i in range(len(peris[1])):
    if len(peris[1][i])==1:
        w1peri.append(i)
        

#todo=[1548,1233,1082,834,542,329,885]#1436,1120]
todo=[37,101,3,0,11,1]
#todo=[37,1]
#todo=np.append(w1peri[0:25],101)
print np.array(todo),len(todo)

dirs=['tst'+str(k)+'_coreb' for k in todo]

for k in dirs:
    os.system('mkdir '+k)
alldirs=['tst'+str(i)+'_coreb/' for i in range(len(f[1].data))]
for k in np.array([.5]):
    after,before=fpwraper.fpwrapper(f,fh,todo=todo,taufactor=k,getangmom='b',folder=alldirs)
    
    vmaxfp=[]
    rmaxfp=[]
    
    vmaxp10=[]
    rmaxp10=[]
    
#done=[1548,278,838,888,11,1233,786,1082,834,832,542,329,1227,1568,341,885,1076,281,1372,904,924,961,826,940,1423,1436,64,1583,347,1268,999,504,343,1485,925,1120,1204,1106,377,1249]
#done=[1548,1233,542,885]#1436,1120]
    done=todo

    p0s=[]
    p1s=[]
    for i in range(len(done)):

        vmaxp10.append(domasslossp10coma.rhofcusp[done[i]].get_vmax().value/domasslossp10coma.rhoicusp[done[i]].get_vmax().value)
        rmaxp10.append(domasslossp10coma.rhofcusp[done[i]].get_rmax().value/domasslossp10coma.rhoicusp[done[i]].get_rmax().value)

        p0=profileclass.Zhao(before[i][2],before[i][3],before[i][4],rho0=before[i][0]*units.M_sun/units.kpc**3,rs=before[i][1]*units.kpc,deltavirrhou=30000*units.M_sun/units.kpc**3)
        p0s.append(p0)
        p1=profileclass.Zhao(after[i][2],after[i][3],after[i][4],rho0=after[i][0]*units.M_sun/units.kpc**3,rs=after[i][1]*units.kpc,deltavirrhou=30000*units.M_sun/units.kpc**3)
        p1s.append(p1)
        
        vmaxfp.append(p1.get_vmax().value/p0.get_vmax().value)
        rmaxfp.append(p1.get_rmax().value/p0.get_rmax().value)


        x=[fsolve(lambda x: rubiatracks.getrmaxfinal(x,0)-vmaxp10[j],.1)[0] for j in range(len(vmaxp10))]

    plt.clf()

    fig,ax=plt.subplots(2,1,sharex=True,sharey=False,figsize=(12,16))
    fig.subplots_adjust(hspace=0)
    mloss=np.linspace(.1,1)
    #ax[1].plot(x,vmaxp10,'o',label='P10')
    ax[1].plot(mloss,2**.4*mloss**.37/(1+mloss)**.4,label='Simulations (Penarrubia+10)')
    ax[1].plot(x,vmaxfp,'o',label='Fokker Plank')
    ax[1].set_ylim(.1,1)
    ax[1].set_ylabel(r'$v_{\rm max,f}/v_{\rm max,i}$')

    #ax[0].plot(x,rmaxp10,'o',label='P10')
    ax[0].plot(mloss,2**-1.37*mloss**0.05/(1+mloss)**-1.3,label='Simulations (Penarrubia+10)')
    ax[0].plot(x,rmaxfp,'o',label='Fokker Plank')
    plt.legend(loc='lower right')
    ax[0].set_ylim(.1,1)
    ax[1].set_xlabel(r'$m(<r_{\rm tide})/m_{\rm infall}$')
    ax[0].set_ylabel(r'$r_{\rm max,f}/r_{\rm max,i}$')
    ax[0].set_yticklabels(['','0.2','0.4','0.6','0.8','1.0'])
    plt.savefig('tracks'+str(k)+'core.png')
    plt.clf()
        
        
    plt.clf()
    plt.plot(vmaxp10,vmaxfp,'o')
    plt.xlabel('delta vmax p10')
    plt.ylabel('delta vmax fp')
    plt.plot([.5,2],[.5,2],'k')
    plt.xlim(.5,2)
    plt.ylim(.5,2)
    plt.savefig('deltavmaxtrackcompare'+str(k)+'cusp.png')
        
    plt.clf()
    plt.plot(rmaxp10,rmaxfp,'o')
    plt.xlabel('delta rmax p10')
    plt.ylabel('delta rmax fp')
    plt.plot([.5,2],[.5,2],'k')
    plt.xlim(.5,2)
    plt.ylim(.5,2)
    plt.savefig('deltarmaxtrackcompare'+str(k)+'cusp.png')
        
    plt.clf()
    plt.plot(domasslossp10coma.mwithinrefinalcore[done]/domasslossp10coma.mwithinreinfcore[done],vmaxp10,'o',label='p10')
    plt.plot(domasslossp10coma.mwithinrefinalcore[done]/domasslossp10coma.mwithinreinfcore[done],vmaxfp,'o',label='fp')
    plt.xlabel('mass loss')
    plt.ylabel('delta vmax')
    plt.loglog()
    plt.legend()
    plt.savefig('deltavmaxcomparem'+str(k)+'cusp.png')
        
    plt.clf()
    plt.plot(domasslossp10coma.mwithinrefinalcore[done]/domasslossp10coma.mwithinreinfcore[done],rmaxp10,'o',label='p10')
    plt.plot(domasslossp10coma.mwithinrefinalcore[done]/domasslossp10coma.mwithinreinfcore[done],rmaxfp,'o',label='fp')
    plt.xlabel('mass loss')
    plt.ylabel('delta rmax')
    plt.loglog()
    plt.legend()
    plt.savefig('deltarmaxcomparem'+str(k)+'cusp.png')
