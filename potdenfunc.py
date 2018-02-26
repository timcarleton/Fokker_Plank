import numpy as np
from astropy import units
from astropy.constants import G
from scipy.special import beta as betafunc
from scipy.special import gamma as gammafunc
from scipy.special import hyp2f1 as Hypergeometric2F1

kminkpc=1*units.kpc.to(units.km)

GN=(G.to(units.kpc**3/units.s**2/units.M_sun)).value

def zhaorho(r,rho0,rs,alpha,beta,gamma):
    return rho0/(r/rs)**gamma/(1+(r/rs)**alpha)**(1.0*(beta-gamma)/alpha)

def plummerrho(r,m,rs):
    return 3*m/4/np.pi/rs**3*(1+(r/rs)**2)**-2.5

def plummerdrhodr(r,m,rs):
    return -(3*m/4/np.pi/rs**3)*5*r/rs**2/(1+(r/rs)**2)**3.5

def plummerd2rhodr2(r,m,rs):
    return (3*m/4/np.pi/rs**3)*((35.*r**2)/(rs**4*(1 + r**2/rs**2)**4.5) - 5./(rs**2*(1 + r**2/rs**2)**3.5))

def plummerphi(r,m,rs):
    return -GN*m/np.sqrt(r**2+rs**2)

def plummerdphidr(r,m,rs):
    return (GN*m*r)/(r**2 + rs**2)**1.5

def plummerd2phidr2(r,m,rs):
    return -(GN*m*((3*r**2)/(r**2 + rs**2)**2.5 - (r**2 + rs**2)**(-1.5)))


def zhaodphidr(r,rho0,rs,alpha,beta,gamma):

    a=1.0*(3-gamma)/alpha
    b=1.0*(beta-gamma)/alpha
    c=1.0*(alpha-gamma+3)/alpha

    x=-(r/rs)**alpha

    f=Hypergeometric2F1(a,b,c,x)

    return GN*4*np.pi*rho0*r**3*(r/rs)**(-gamma)*f/(3-gamma)/r**2


def zhaod2phidr2(r,rho0,rs,alpha,beta,gamma):

    a=float(alpha)
    b=float(beta)
    g=float(gamma)
    x=float(r)
    s=float(rs)

    try:
        d2phidr2=(4*GN*np.pi*rho0*(x/s)**(-b - g)*(-((-3 + g)*(-((1 + (s/x)**a)**(1 + g/a)*(x/s)**g*(1 + (x/s)**a)**(1 + b/a)) + g*((1 + (s/x)**a)**(g/a)*(s/x)**a*(x/s)**g*(1 + (x/s)**a)**(b/a) + (1 + (s/x)**a)**(g/a)*(s/x)**a*(x/s)**(a + g)*(1 + (x/s)**a)**(b/a) - (1 + (s/x)**a)**(b/a)*(x/s)**b*(1 + (x/s)**a)**(g/a) - (1 + (s/x)**a)**(b/a)*(s/x)**a*(x/s)**b*(1 + (x/s)**a)**(g/a)) + b*((1 + (s/x)**a)**(g/a)*(x/s)**g*(1 + (x/s)**a)**(b/a) + (1 + (s/x)**a)**(g/a)*(x/s)**(a + g)*(1 + (x/s)**a)**(b/a) - (1 + (s/x)**a)**(b/a)*(x/s)**(a + b)*(1 + (x/s)**a)**(g/a) - (1 + (s/x)**a)**(b/a)*(s/x)**a*(x/s)**(a + b)*(1 + (x/s)**a)**(g/a)))) + 2*(1 + (s/x)**a)**(1 + b/a)*(x/s)**b*(1 + (x/s)**a)**(1 + b/a)*Hypergeometric2F1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-(x/s)**a)))/((-3 + g)*(1 + (s/x)**a)**((a + b)/a)*(1 + (x/s)**a)**((a + b)/a))
    
    except:
        dx=np.min([r,1E-5])

        dphidrlow=dphidrzhao(r-dx,rho0,rs,alpha,beta,gamma)
        dphidrhigh=dphidrzhao(r+dx,rho0,rs,alpha,beta,gamma)
        
        
        d2phidr2=dphidrlow/dx/2.0+dphidrhigh/dx/2.0


    if not np.isfinite(d2phidr2):

        dx=np.min([r,1E-5])

        dphidrlow=dphidrzhao(r-dx,rho0,rs,alpha,beta,gamma)
        dphidrhigh=dphidrzhao(r+dx,rho0,rs,alpha,beta,gamma)
        
        
        d2phidr2=dphidrlow/dx/2.0+dphidrhigh/dx/2.0

    return d2phidr2


def zhaophi(r,rho0,rs,alpha,beta,gamma):

    a=float(alpha)
    b=float(beta)
    g=float(gamma)
    s=float(rs)

    phir=lambda x:-4*GN*np.pi*rho0*x**2*(Hypergeometric2F1(-2/a + b/a,b/a - g/a,1 - 2/a + b/a,-(s/x)**a)/((-2 + b)*(x/s)**b) + Hypergeometric2F1(3/a - g/a,b/a - g/a,1 + 3/a - g/a,-(x/s)**a)/((3 - g)*(x/s)**g))

    if len(np.shape(r))==0:
        phi=phir(r)
        if not np.isfinite(phi):
            for i in np.linspace(-10,1):
                xpoint=10**i
                dphidrpoint=dphidrzhao(xpoint,rho0,rs,alpha,beta,gamma)
                if np.isfinite(dphidrpoint) and np.isfinite(phir(xpoint)):
                    break
                else:
                    None
            phi=phir(xpoint)+(r-xpoint)*dphidrpoint
        return phi
    else:

        phi=phir(r)
        wnan=np.where(~np.isfinite(phi))[0]

        if len(wnan)>0:
            for i in np.linspace(-10,1):
                xpoint=10**i
                dphidrpoint=dphidrzhao(xpoint,rho0,rs,alpha,beta,gamma)
                print 'dphi',r,dphidrpoint,phir(xpoint)
                if np.isfinite(dphidrpoint) and np.isfinite(phir(xpoint)):
                    break
                else:
                    None
                    
            phi[wnan]=phir(xpoint)+(r[wnan]-xpoint)*dphidrpoint
        return phi

        
def zhaodrhodr(r,rho0,rs,alpha,beta,gamma):

    a=alpha
    b=beta
    g=gamma
    s=rs
    x=r
    
    return -(((b - g)*rho0*(x/s)**(-1 + a - g)*(1 + (x/s)**a)**(-1 - (b - g)/a))/s) - (g*rho0*(x/s)**(-1 - g))/(s*(1 + (x/s)**a)**((b - g)/a))

def zhaod2rhodr2(r,rho0,rs,alpha,beta,gamma):
    
    a=float(alpha)
    b=float(beta)
    g=float(gamma)
    s=float(rs)
    x=float(r)
    

    return (2*(b - g)*g*rho0*(x/s)**(-2 + a - g)*(1 + (x/s)**a)**(-1 - (b - g)/a))/s**2 - ((-1 - g)*g*rho0*(x/s)**(-2 - g))/(s**2*(1 + (x/s)**a)**((b - g)/a)) + (rho0*(-((a*(-1 - (b - g)/a)*(b - g)*(x/s)**(-2 + 2*a)*(1 + (x/s)**a)**(-2 - (b - g)/a))/s**2) - ((-1 + a)*(b - g)*(x/s)**(-2 + a)*(1 + (x/s)**a)**(-1 - (b - g)/a))/s**2))/(x/s)**g
