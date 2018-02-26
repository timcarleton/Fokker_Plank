from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import getrsi as rsi

def getge(sifunc,e,dsidr,minr=1E-10):

    rm=rsi.getrsi(sifunc,e,dsidr)

    integrand=lambda r: np.sqrt(2*(sifunc(r)-e))*r**2

    return quad(integrand,minr,rm)[0]
