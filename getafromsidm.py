import getrhalfzhao
reload(getrhalfzhao)
import SIDM_thermal_nob
SIDM_thermal_nob.sidm_setup()
from astropy import units
from scipy.optimize import minimize_scalar,minimize
import numpy as np
import profileclass
reload(profileclass)
from scipy import interpolate
import getrhalfnfw
import matplotlib.pyplot as plt
from astropy.constants import G
GN=G.to(units.kpc**3/units.M_sun/(units.s)**2)

def getafromsidm(vmax,rmax,age,rvir,gamma,cross=1.0,minr=.01*units.kpc,rmatch='half',alpha=1,beta=4,mvirmatch=True):

    pnfw=profileclass.NFW(vmax=vmax,rmax=rmax,rvir=rvir)
    delta=(pnfw.mvir/(4.0/3*np.pi*rvir**3)).to(units.M_sun/units.kpc**3)
    if age==0:
        if rmatch=='half':
            r12=getrhalfnfw.getrhalfnfw(pnfw.rs.value,pnfw.rvir.value)
            m12=pnfw.mvir/2.0
#            rho0func=lambda rsc: p.get_mass(rmax).to(units.M_sun).value/getmassfromzhao0
            rho0func=lambda rsc: (pnfw.mvir.value/profileclass.getmassfromzhao0(alpha,beta,gamma,1*units.M_sun/units.kpc**3,rsc*units.kpc,rvir))
            afunc= lambda rsc: np.log10(profileclass.getmassfromzhao0(alpha,beta,gamma,rho0func(rsc)*units.M_sun/units.kpc**3,rsc*units.kpc,r12*units.kpc).value/m12.value)
            try:
                az=minimize_scalar(lambda rs:abs(afunc(rs)),bracket=[1E-6*r12,.1*r12,1E5*r12])['x']
            except:
                return rmax.value/2.163,pnfw.rho0
            return az,pnfw.rho0
        else:
            r12=(rmax/2.163).value
            return pnfw.rs.value,pnfw.rho0.to(units.M_sun/units.kpc**3).value
            rho12=(pnfw.rho0.to(units.M_sun/units.kpc**3)/4.0).value
            m12=pnfw.get_mass(r12*rmax.unit).to(units.M_sun).value
            rho0func=lambda rsc: rho12*(r12/rsc)**(gamma)*(1+r12/rsc)**(1.0*(beta-gamma)/alpha)
            afunc= lambda rsc: np.log10(profileclass.getmassfromzhao0(alpha,beta,gamma,rho0func(rsc)*units.M_sun/units.kpc**3,rsc*units.kpc,r12*units.kpc).value/m12)
            az=minimize_scalar(lambda rs:abs(afunc(rs)),bracket=[1E-5*r12,.1*r12,1E5*r12])['x']

        return rho0func(az),az

    x=SIDM_thermal_nob.get_sidm_fit_without_beffect([vmax.to(units.km/units.s).value,rmax.to(units.kpc).value,cross,age.to(units.Gyr).value,minr.to(units.kpc).value,1.55*rvir.to(units.kpc).value])

    if rmatch=='r1':
        rmatchuse=x[0][-1]

    rs=x[1]
    ms=x[2]
    rhos=x[3]
#    plt.plot(rs,rhos,'g-.',label='SIDM no B',alpha=.2)
#    plt.plot(rs,ms,'g-.',label='SIDM no B',alpha=.2)

#    rhointerpfunc=lambda x: 10**interpolate.interp1d(np.log10(rs),np.log10(rhos))(np.log10(x))
#    massinterpfunc=lambda x: 10**interpolate.interp1d(np.log10(rs),np.log10(ms))(np.log10(x))

    massinterpfunc=interpolate.interp1d(rs,ms)
    rhointerpfunc=interpolate.interp1d(rs,rhos)

    wrhohalf=np.where(rhos<.5*rhos[0])[0]


    rvirsidm=minimize_scalar(lambda rvx: abs(np.log10(massinterpfunc(rvx)/(4.0/3*np.pi*rvx**3)/(delta.value))),bracket=[.5*rvir.value,rvir.value,1.5*rvir.value])['x']

    mvirsidm=massinterpfunc(rvirsidm)

    mvirsidm=pnfw.mvir.to(units.M_sun).value

    if rmatch=='half':
        m12=.5*mvirsidm
        
    else:
        if rmatch!='r1':
            rmatchuse=rmatch
        interpfunc=interpolate.interp1d(rs,ms)
        m12=interpfunc(rmatchuse)
        #interpolate to get m at rmatch, this is the new m12
    w=np.where(ms<m12)[0]

    m1=ms[w[-1]]
    m2=ms[w[-1]+1]
    r1=rs[w[-1]]
    r2=rs[w[-1]+1]

    slope=(np.log10(m2)-np.log10(m1))/(np.log10(r2)-np.log10(r1))

    r12=10**((np.log10(m12)-np.log10(m1))/slope+np.log10(r1))
    rho12=rhointerpfunc(r12)


    rho0=x[0][0]
    if mvirmatch:
#         mr=(massinterpfunc(rmatch)/pnfw.mvir.to(units.M_sun).value)**(1.0/(3.0-gamma))
#         top=mr-1
#         bottom=1./rvir.value-mr/rmatch
#         print top/bottom
#         return top/bottom
        ratio=1.0/4.0
        rx=minimize_scalar(lambda rad:abs(rhointerpfunc(rad)-ratio*rho0),[min(rs),x[0][1],max(rs)])['x']
        print rx

        if gamma==0:
            return rx/((1/ratio)**.25-1),rho0
        else:
            return minimize_scalar(lambda rhoa:abs((rx/rhoa)**gamma*(1+rx/rhoa)**(3-gamma)-ratio))['x'],rho0

    if rmatch=='half':
        return r12*(2**(1.0/(3-gamma))-1)

#    if rmatch=='half':
#        az=minimize_scalar(lambda rsc: abs(np.log10(profileclass.getmassfromzhao0(alpha,beta,gamma,1*units.M_sun/units.kpc**3,rsc*units.kpc,r12*units.kpc)/(profileclass.getmassfromzhao0(alpha,beta,gamma,1*units.M_sun/units.kpc**3,rsc*units.kpc,rs[-1]*units.kpc)/2.0))),bracket=[.01*r12,100*r12])['x'].value

#        a=r12*(2**(1.0/(3-gamma))-1)

 #   else:
        
        
        #rmx=lambda rsc: rsc*profileclass.getxmaxzhao0(alpha,beta,gamma)
        #mrmx= lambda rsc: ((((vmax)**2)*rmx(rsc))/G).to(units.M_sun)
    rho0func=lambda rsc: rho12*(r12/rsc)**(gamma)*(1+r12/rsc)**(1.0*(beta-gamma)/alpha)
#    rho0func=lambda rsc: profileclass.getrho0frommvirzhao(alpha,beta,gamma,mvir,rsc,(mvir/(4.0/3*np.pi*rvir**3)))
    afunc= lambda rsc: np.log10(profileclass.getmassfromzhao0(alpha,beta,gamma,rho0func(rsc)*units.M_sun/units.kpc**3,rsc*units.kpc,r12*units.kpc).value/m12)
        #bfunc=lambda (rh0,rsc): np.log10(profileclass.getmassfromzhao0(alpha,beta,gamma,rh0*units.M_sun/units.kpc**3,rsc*units.kpc,rmx(rsc*units.kpc)).value/(mrmx(rsc*units.kpc)).value)
        #mz=lambda (rh0,rsc): afunc([rh0,rsc])**2+bfunc([rh0,rsc])**2
        #az=minimize(mz,[ms[-1]/r12**3,r12])

    #        a=rmatchuse*((ms[-1]/m12)**(1.0/(3-gamma))-1)
    az=minimize_scalar(lambda rs:abs(afunc(rs)),bracket=[1E-5*r12,.1*r12,1E5*r12])['x']
    
#    rho0=x[0][0]*(minr.value/az)**(gamma)*(1+minr.value/az)**(1.0*(beta-gamma)/alpha)
#    return az['x']
    return rho0func(az),az
