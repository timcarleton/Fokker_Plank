import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar

class sidm_setup:
    
    def __init__(self):
        self.GNewton = 4.302113488372941e-06  # G in kpc * (km/s)**2 / Msun
        self.fourpi = 4.0 * np.pi
        self.gyr = 3.15576e+16
        self.rate_const = 1.5278827817856099e-26 * self.gyr
        
        self.x_iso_min, self.x_iso_max = 0.1, 6.0
        self._x_iso = np.logspace(np.log10(self.x_iso_min),np.log10(self.x_iso_max),100)
        self._y_iso_0 = np.array([1.0, (4.0 * np.pi / 3.0) * self.x_iso_min ** 3])
        self._y_iso = odeint(self.fsidm_no_b, self._y_iso_0, self._x_iso)
        self.density_iso_no_b = InterpolatedUnivariateSpline(self._x_iso, self._y_iso[:,0])
        self.mass_iso_no_b = InterpolatedUnivariateSpline(self._x_iso, self._y_iso[:,1])
        self._x = np.logspace(-4,4,num=100)
        self._y = list(map((lambda x: self.nfw_m_profile(x)*(1+1/x)**2),self._x))
        self.r1_over_rs = InterpolatedUnivariateSpline(self._y,self._x)
        self._y = self._y_iso[:,1]/(4 * np.pi * self._y_iso[:,0] * self._x_iso ** 3)
        self.r1_over_r0 = InterpolatedUnivariateSpline(self._y,self._x_iso)
    
    def fsidm_no_b(self, y, x): # for scipy.integrate.odeint
        drhodr = - 9.0 * y[1] * y[0] / ( 4 * np.pi * x ** 2 )
        dmassdr = 4 * np.pi * y[0] * x ** 2
        return [drhodr, dmassdr]

    def nfw_m_profile(self, x):
        return np.log(1+x) - x/(1.0 + x)

    def nfw_d_profile(self, x):
        return 1 / ( x * (1+x) ** 2 )


def get_sidm_fit_without_beffect(params):
        
    ss = sidm_setup()
    vmax, rmax, cross, age, min_r, max_r = params
    rs = rmax / 2.163
    rhos = vmax**2 / ( ss.GNewton * rmax **2 * 0.5807)
    mnfw0 = ss.fourpi * rhos * rs ** 3
    r1_slop = 0

    def match_cross(r1, rhos, rs, age): 
        x1 = r1/rs
        rho1 = rhos * ss.nfw_d_profile(x1)
        rho1 *= (1 + r1_slop*(2*np.random.random_sample()-1))
        m1 = mnfw0 * ss.nfw_m_profile(x1)
        m1 *= (1 + r1_slop*(2*np.random.random_sample()-1))
        mr1 = m1 / (ss.fourpi * rho1 * r1 ** 3)
        if ss._y[0] <= mr1 <= ss._y[-1]:
            r0 = r1 / ss.r1_over_r0(mr1)
            rho0 = rho1 / ss.density_iso_no_b(r1/r0)
        else:
            r0 = rs
            rho0 = rhos * 1e-10
        sigma0 = np.sqrt(ss.fourpi * ss.GNewton * rho0) * r0 / 3.0
        cross_out = 1.0/(ss.rate_const * age * sigma0 * rho1)
        return (cross - cross_out)**2
    
    res = minimize_scalar(match_cross, args=(rhos,rs,age), bounds = (rmax*0.01,5.0*rmax), method='bounded', options={'disp': 0, 'xatol': 1e-05})
    r1 = res.x
    x1 = r1 / rs
    m1 = mnfw0 * ss.nfw_m_profile(x1)
    rho1 = rhos * ss.nfw_d_profile(x1)
    mr1 = m1 / (ss.fourpi * rho1  * r1 ** 3)
    r0 = r1 / ss.r1_over_r0(mr1)
    rho0 = rho1 / ss.density_iso_no_b(r1/r0)
    sigma0 = np.sqrt(ss.fourpi * ss.GNewton * rho0) * r0 / 3.0
    miso0 = rho0 * r0 **3
    
    rho = []
    mass = []
    rho_nfw = []
    mass_nfw = []
    n = np.int(np.log10(max_r/min_r)*10)
    ra = np.logspace(np.log10(min_r),np.log10(max_r),n)
    for r in ra:
        mass_cdm = ss.nfw_m_profile(r/rs)*mnfw0
        rho_cdm = ss.nfw_d_profile(r/rs)*rhos
        if r > r1: 
            mass_sidm = mass_cdm
            rho_sidm = rho_cdm
        else:
            x0 = r/r0
            if x0 < ss._x_iso[0]:
                mass_sidm = miso0 * ss.mass_iso_no_b(ss._x_iso[0]) * \
                (x0/ss._x_iso[0])**3
                rho_sidm = rho0 
            elif x0 > ss._x_iso[-1]:
                mass_sidm = miso0 * ss.mass_iso_no_b(ss._x_iso[-1])
                rho_sidm = rho0 * ss.density_iso_no_b(ss._x_iso[-1]) * \
                (ss._x_iso[-1]/x1)**5
            else:
                mass_sidm = miso0 * ss.mass_iso_no_b(x0)
                rho_sidm = rho0 * ss.density_iso_no_b(x0)
            
        mass = np.append(mass, mass_sidm)
        rho = np.append(rho, rho_sidm)
        mass_nfw = np.append(mass_nfw, mass_cdm)
        rho_nfw = np.append(rho_nfw, rho_cdm)
    iso_params = (rho0, r0, sigma0, r1)
    return iso_params, ra, mass, rho, mass_nfw, rho_nfw, res
