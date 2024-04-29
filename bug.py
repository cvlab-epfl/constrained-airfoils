#!/usr/bin/env python3
#%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import torch

from projarea    import AreaProjector
from airfoildata import loadAirfoilData
from auxfuncs    import netwDataName,currentDevice
#%%---------------------------------------------------------------------------
#                                Load Data
#-----------------------------------------------------------------------------
zdim=8
step=25
n1 = 16
n2 = 32
n3 = 16
targetA= 0.1
dataT = loadAirfoilData(zdim=zdim,batchN=100,trainP=True,step=step,targetA=targetA)
ydim  = dataT.target.size(1)
fName = netwDataName(zdim,n1,n2,n3)
zs,xys0 = dataT.batch(0)
#%%---------------------------------------------------------------------------
#                            Train Autodecoder Alone
#-----------------------------------------------------------------------------
#%%
n1 = 16
n2 = 32
n3 = 16
net  = AreaProjector(n1=n1,n2=n2,n3=n3,nIn=8,nOut=54)
net.toGpu()
#%%
kludgeP=False
def wingLodF(net,z):
    
    gradP = z.requires_grad 
    
    if(gradP and kludgeP):
        # Kludge: If we pass a batch of only one, backward will die.
        zd    = z.size(0)
        z2  = torch.vstack([z,torch.zeros(zd,dtype=torch.float32,device=z.device).requires_grad_(gradP)])
        xy2 = net(z2)
        xy1 = xy2[0:1]
    else:
        xy1 = net(z.contiguous().view((1,-1)))
      
    # cd, cl  = model(xy1)[0]
    # obj     = cd / cl
    
    obj = torch.sum(xy1[0])
        
    return obj

def wingLodG(net,z):
    
    assert(z.requires_grad)
    
    obj = wingLodF(net,z)
    obj.backward()
    grad = z.grad
    
    return grad

#model = loadSurogateModel()
z = torch.tensor([ 0.0149,  0.0232, -0.0196,  0.0356,  0.0021,  0.0252,  0.0184,  0.0410],requires_grad=True,device=currentDevice())
wingLodF(net,z)
wingLodG(net,z)