#!/usr/bin/env python3
#%%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np
import torch
import matplotlib.pyplot as plt
from   numba import njit

#%%---------------------------------------------------------------------------
#                                TORCH
#-----------------------------------------------------------------------------

def fromTensor(x):
    if(torch.is_tensor(x)):
        return x.data.cpu().numpy()
    return x

def currentDevice():
    
    if(torch.cuda.is_available()):
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
#%%---------------------------------------------------------------------------
#                               SHOELACE AREA
#-----------------------------------------------------------------------------

#%%
@njit()
def shoelaceAreaF(xys):
    
    assert(2==len(xys.shape))
    
    xs = xys[:,0]
    ys = xys[:,1]
    x0 = xs[-1]
    y0 = ys[-1]
    
    area = 0.0
    for i1,x1 in enumerate(xs):
        
        x1    = xs[i1]
        y1    = ys[i1]

        area += x1*y0-x0*y1
        x0    = x1
        y0    = y1
        
    return area/2.0

@njit()
def shoelaceAreaG(xys):
    
    assert(2==len(xys.shape))
    
    g  = np.zeros_like(xys)
    
    xs = xys[:,0]
    ys = xys[:,1]
    x0 = xs[-1]
    y0 = ys[-1]
    nv = xs.shape[0]
    
    
    i0 = nv-1
    #area = 0.0
    for i1,x1 in enumerate(xs):
        x1      = xs[i1]
        y1      = ys[i1]

        g[i0,0] -=  y1 # da/dx0 = -y1 
        g[i1,0] +=  y0 # da/dx1 =  y0
        g[i0,1] +=  x1 # da/dy0 =  x1
        g[i1,1] -=  x0 # da/dy1 = -x0
        
        #area += x0*y1-x1*y0
        x0    = x1
        y0    = y1
        i0    = i1
        
    return g/2.0


def shoelaceArea1(xy):
    
    assert(1==len(xy.size()))
    
    ns    = xy.size(0)
    ids   = np.arange(0,ns,2)
    
    x0 = xy[ids]
    y0 = xy[ids+1]
    
    x1=torch.cat((x0[-1:],x0[0:-1]))
    y1=torch.cat((y0[-1:],y0[0:-1]))
    
    area = torch.sum(y1*x0)-torch.sum(y0*x1)
    return area/2.0

def shoelaceArea2(xy):
    
    assert(2==len(xy.size()) and (2 == xy.size(1)))
    
    x0 = xy[:,0]
    y0 = xy[:,1]
    
    x1=torch.cat((x0[-1:],x0[0:-1]))
    y1=torch.cat((y0[-1:],y0[0:-1]))
    
    area = torch.sum(y1*x0)-torch.sum(y0*x1)
    return area/2.0

#%%---------------------------------------------------------------------------
#                               Affine Transform
#-----------------------------------------------------------------------------

# Affine deformation of a set of points
def affTrf(aff,xy):
    
    if(torch.is_tensor(aff)):
        
        assert(1==len(aff.size())  and 6==aff.size(0))
        assert(torch.is_tensor(xy) and 2==len(xy.size()) and (2 == xy.size(1)))
    
        A = aff[0:4].view((2,2))
        b = aff[4:6]
    
    else:
        
        assert(1==len(aff.shape) and 6==aff.shape[0])
        assert(not(torch.is_tensor(xy)) and 2==len(xy.shape) and (2 == xy.shape[1]))
        
        A = aff[0:4].reshape((2,2))
        b = aff[4:6]
    
    return xy @ A.T + b

#%%---------------------------------------------------------------------------
#                                Display
#-----------------------------------------------------------------------------

def drawAirfoil(xs,color='-b'):
       
    xs=fromTensor(xs)
    xy=xs.reshape((-1,2))    
    plt.plot(xy[:,0],xy[:,1],color)[0]
    plt.gca().axis('equal')


#%%---------------------------------------------------------------------------
#                                Data
#-----------------------------------------------------------------------------

def loadWingProfiles(step=None,trainP=True,targetA=None):
   
    xys = np.load(wingDataName(trainP,targetA,step))
    
    if(targetA is None and step is not None):
        id1 = range(0,301,step)
        id2 = range(301,602,step)
        ids = np.hstack((np.array(id1),np.array(id2),np.array(0)))
        
        ns  = xys.shape[0]
        xys  = xys.reshape((ns,602,2))
        xys  = xys[:,ids,:]
        xys  = xys.reshape((ns,-1))
        
    return np.asarray(xys,dtype=np.float32)

def saveWingProfiles(xys,step=None,trainP=True,targetA=None):
    
    np.save(wingDataName(trainP,targetA,step),xys)

def wingDataName(trainP,targetA,step):
    
    if(trainP):
        bName = 'training'
    else:
        bName = 'testing'
        
    if(targetA):
        fName = 'dat/{}-{}-{}.npy'.format(bName,step,targetA) 
    else:
        fName = 'dat/{}.npy'.format(bName) 
        
    return fName

def netwDataName(zdim,n1,n2,n3,targetA=None):
    
    if(targetA is None):
        fileName = 'dat/airf-{}-{}-{}-{}'.format(zdim,n1,n2,n3)
    else:
        fileName = 'dat/airf-{}-{}-{}-{}-{}'.format(zdim,n1,n2,n3,targetA)
    return(fileName)

