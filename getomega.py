from numpy import sqrt,sin,arccos,array,cross
import profileclass
from numpy import hstack,array,cross,repeat,sum,shape
def getomega(hostprofile,distance):

    return sqrt(profileclass.GN*hostprofile.get_mass(distance)/(distance**3))

def getomega_direct(vx,vy,vz,rx,ry,rz):

    vvec=array([vx,vy,vz]).T
    rvec=array([rx,ry,rz]).T
    
    rmag=sqrt(rx**2+ry**2+rz**2)
    
    omvec=cross(rvec,vvec)

    if len(shape(vx))>0:
        rmag=array([rmag.tolist(),]*3).T
        return sqrt(sum(omvec**2,axis=-1).T)/rmag[:,0]/rmag[:,0]
    else:
        return sqrt(sum(omvec**2))/rmag/rmag
