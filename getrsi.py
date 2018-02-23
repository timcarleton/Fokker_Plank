from scipy import fsolve

def getrsi(sifunc,si,r0=None):
    if r0==None:
        for i in np.linspace(-1,30):
            r0=i-np.log(si)
            if np.isfinite(np.log(sifunc(np.exp(r0)))):
                break
            else:
                None

    if si<1:
        if si<.5:
        #rsi=fsolve(lambda x: np.log(sifunc(x))-np.log(abs(si)),np.exp(r0),full_output=False,fprime=lambda x: 1.0/sifunc(x)*dsidr(x),factor=1)[0]
            rsi=fsolve(lambda x: np.log(sifunc(x))-np.log(abs(si)),np.exp(r0),full_output=False,factor=1)[0]
            if not np.isfinite(rsi):
                fsolve(lambda x: sifunc(x)-(si),np.exp(r0),full_output=False,fprime=dsidr)[0]
        else:
            #rsi=np.exp(fsolve(lambda x: sifunc(np.exp(x))-(si),r0,full_output=False,fprime=dsidr)[0])
            rsi=np.exp(fsolve(lambda x: sifunc(np.exp(x))-(si),r0,full_output=False)[0])
    else:
        rsi=1E-10

    #print 'rsi',r0,fsolve(lambda x: np.log(sifunc(np.exp(x)))-np.log(abs(2*si)),r0,full_output=True,fprime=lambda x: 1.0/sifunc(np.exp(x))*dsidr(np.exp(x))*np.exp(x)),rsi,si,sifunc(rsi)
    if rsi<0 or si>1:
        rsi=1E-10
    dsi=dsidr(rsi)
