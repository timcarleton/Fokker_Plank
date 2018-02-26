from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import abel
import fpwraper


def d2rhodsi2(sifunc,drhodr,d2rhodr2,dsidr,d2sidr2,si,r0=None):

    rsi=getrsi.getrsi(sifunc,si,dsidr=dsidr,r0=r0)

    
    dsi=dsidr(rsi)

    first=1.0/dsi**2*d2rhodr2(rsi)
    second=1.0/dsi**3*drhodr(rsi)*d2sidr2(rsi,dsi)
    
    if abs((sifunc(rsi)-si)/si)>.1 and si<1:
        print 'bad solve',si,rsi,sifunc(rsi),dsidr(rsi)

#     if not np.isfinite(first-second):
        
#         print 'si',rsi,si,first-second,dsi,drhodr(rsi),d2sidr2(rsi,dsi)
#         print fsolve(lambda x: np.log(sifunc(x))-np.log(abs(si)),np.exp(r0),full_output=True,fprime=lambda x: 1.0/sifunc(x)*dsidr(x))

    return first-second

def getfe(sifunc,drhodr,d2rhodr2,dsidr,d2sidr2,lowe,minr=1E-10):
    

    rs=np.linspace(minr,1.0/np.sqrt(lowe),num=1000/np.sqrt(lowe)+1)

    sir=lambda r: 1/r**2
    
    jr=lambda r: d2rhodsi2(sifunc,drhodr,d2rhodr2,dsidr,d2sidr2,sir(r))/r**3

    jvalues=np.array([jr(i) for i in rs])
    wf=np.where(np.isfinite(jvalues))[0]

    fe=abel.hansenlaw.hansenlaw_transform(jvalues[wf],direction='forward',dr=rs[1]-rs[0])

    fefinal=fe/np.sqrt(8)/np.pi**2/np.sqrt(sir(rs[wf]))


    w=np.where(~(fefinal>0))[0]
    fefinal[w]=0
    
    return interp1d(sir(rs[wf]),fefinal,fill_value='extrapolate')

def getfehighzhao(highe,fehighe,table,e,gamma):

    if gamma==0:
        slope=-1
    elif gamma<2:
        slope=-(6-gamma)/2/(2-gamma)
    elif gamma==2:
        slope=0
    elif gamma<3:
        slope=0
    print gamma,slope,np.log(e),np.log10(highe)

    logfefinal=np.log10(fehighe)+slope*(np.log10(1-e)-np.log10(1-highe))
    return 10**logfefinal

dotest=False
if dotest:
#    sifuncplummer=lambda r: 1.0/np.sqrt(1+r**2)
#    drhodrplummer=lambda r: -5*r/(1+r**2)**3.5
#    d2rhodr2plummer=lambda r: (35*r**2/(1+r**2)**4.5-5.0/(1+r**2)**3.5)
#    dsidrplummer=lambda r: r/(1+r**2)**1.5
#    d2sidr2plummer=lambda r: 1.0/(1+r**2)**1.5-3*r**2/(1+r**2)**2.5

    sifuncplummer=lambda r: fpwraper.zhaophi(r,3.0/4.0/np.pi/fpwraper.GN,1.0,3.0,5.0,0.0)/fpwraper.zhaophi(1E-10,3.0/4.0/np.pi/fpwraper.GN,1.0,3.0,5.0,0.0)/2.0
    drhodrplummer=lambda r: fpwraper.drhodrzhao(r,3.0/4.0/np.pi,1.0,3.0,5.0,0.0)
    d2rhodr2plummer=lambda r: fpwraper.d2rhodr2zhoa(r,3.0/4.0/np.pi,1.0,3.0,5.0,0.0)
    dsidrplummer=lambda r: fpwraper.dphidrzhao(r,3.0/4.0/np.pi/fpwraper.GN,1.0,3.0,5.0,0.0)/fpwraper.zhaophi(1E-10,3.0/4.0/np.pi/fpwraper.GN,1.0,3.0,5.0,0.0)/2.0
    d2sidr2plummer=lambda r,b: fpwraper.dphi2dr2zhao(r,3.0/4.0/np.pi/fpwraper.GN,1.0,3.0,5.0,0.0)/fpwraper.zhaophi(1E-10,3.0/4.0/np.pi/fpwraper.GN,1.0,3.0,5.0,0.0)/2.0
    feanalyticplummer=lambda e: 20*32.0/35*e**3.5/np.sqrt(8)/np.pi**2

    print sifuncplummer(1E-3)

    sifuncdehnen=lambda r: (1-(r/(r+1))**2)/2.0
    drhodrdehnen=lambda r: -3/np.pi/(1+r)**5/2.0
    d2rhodr2dehnen=lambda r: 15/np.pi/(1+r)**6/2.0
    dsidrdehnen=lambda r: -(2*r/(1+r)**3)/2.0
    d2sidr2dehnen=lambda r,b: (-2+4*r)/(1+r)**4/2.0

    #feanalyticdehnen=lambda e: 3/2.0/np.pi**3*(np.sqrt(2*e)*((3-4*e)/(1-2*e))-3*np.arcsinh(np.sqrt(2*e/(1-2*e))))
    from scipy.special import hyp2f1
    feanalyticdehnen=lambda e: 3/2/(2*np.pi**2)**1.5*2*(2*e/(1-2*e)-6*e*hyp2f1(0.5,1,1.5,2*e))/np.sqrt(e)
    d2rhodsi2dehnen=lambda x: (3*x**2*(24*(1 + np.sqrt(1 - x)) - 4*(8 + 5*np.sqrt(1 - x))*x + (9 + 2*np.sqrt(1 - x))*x**2))/(4.*np.pi*(1 + np.sqrt(1 - x))**6*(1 - x)**1.5)
    #feanalyticdehnen=lambda e: 3*(2*np.sqrt(e)*(2*e-3)+6*(e-1)*np.log(1+np.sqrt(e))+3*(e-1)*np.log(1-e))/(4.0*np.pi*(e-1))

    fe=feanalyticplummer
#    fe=getfe(lambda x: sifuncplummer(x)+sifuncdehnen(x),drhodrplummer,d2rhodr2plummer,lambda x:dsidrplummer(x)+dsidrdehnen(x),lambda x,b:d2sidr2plummer(x,b)+d2sidr2dehnen(x,b),1E-2)
    fe=getfe(lambda x: sifuncplummer(x),drhodrplummer,d2rhodr2plummer,lambda x:dsidrplummer(x),lambda x,b:d2sidr2plummer(x,b),1E-2)
    fe2=getfe(lambda x: sifuncplummer(x)+sifuncdehnen(x),drhodrdehnen,d2rhodr2dehnen,lambda x:dsidrplummer(x)+dsidrdehnen(x),lambda x,b:d2sidr2plummer(x,b)+d2sidr2dehnen(x,b),1E-2)
#    fed=getfe(sifuncdehnen,drhodrdehnen,d2rhodr2dehnen,dsidrdehnen,d2sidr2dehnen,1E-4)


    em=.2
#    print quad(lambda x: d2rhodsi2(lambda x: sifuncdehnen(x),drhodrdehnen,d2rhodr2dehnen,lambda x: dsidrdehnen(x),lambda x: d2sidr2dehnen(x),x)/np.sqrt(em-x)0,em)[0]/np.sqrt(8)/np.pi**2
    egrid=np.linspace(.001,1,num=500)
    evalfe=fe(egrid)
    
    rpts=np.logspace(-2,2.5,num=1000)
    rhoplummerfe=np.zeros(len(rpts))
    from scipy.integrate import quad
    newfe=interp1d(egrid,fe(egrid))
    #print newfe(0),newfe(1)
    
#    for i in range(len(rpts)):
#        emax=sifuncplummer(rpts[i])
#        print emax
#        #w=np.where(egrid<emax)[0]
#        rhoplummerfe[i]=4*np.pi*quad(lambda x: newfe(x)*np.sqrt(2*(emax-x)),.001,emax)[0]
        #rhoplummerfe[i]=4*np.pi*np.trapz(fe(egrid[w])*np.sqrt(2*(emax-egrid[w])),egrid[w])


    #plt.clf()
    #plt.plot(rpts,1.0/(1+rpts**2)**2.5)
    #plt.plot(rpts,rhoplummerfe)
    #plt.loglog()
    #plt.show()
    psis=np.linspace(.01,.999)
    psisd=np.linspace(.01,.999)


    plt.clf()
 #   plt.plot(psis,feanalyticplummer(psis),'.-',label='Plummer Analytic')
 #   plt.plot(psisd,feanalyticdehnen(psisd),'.-',label='Dehnen Analytic')
    
    plt.plot(psis,[fe(i) for i in psis],'--',label='Pummer Numerical')
    plt.plot(psis,[fe2(i) for i in psis],'--',label='Pummer Numerical')
 #   plt.plot(psisd,[fed(i) for i in psisd],'--',label='Dehnen Numerical')
    
#    plt.ylim(1E-7,1)
    plt.yscale('log')
    plt.xlabel('E')
    plt.ylabel('f(E)')
    plt.legend()
    plt.savefig('atst.png')
