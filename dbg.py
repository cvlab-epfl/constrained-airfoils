#!/usr/bin/env python3
#%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import torch

from netw.miscfuncs import fromTensor
from netw.netdata import NetData


from airfoildata import loadAirfoilData
from auxfuncs    import drawAirfoil,netwDataName,shoelaceArea1
from projarea   import  AreaProjector
#%%
kludgeP=True
def wingLodDbg(net,z):
    
    gradP = z.requires_grad
    if(gradP):
        net.zero_grad()
      
    if(gradP and kludgeP):       
        # Kludge: If we pass a batch of only one, backward will die.
        zd  = z.size(0)
        z2  = torch.vstack([z,torch.zeros(zd,dtype=torch.float32,device=z.device).requires_grad_(gradP)])
        xy2 = net(z2)
        xy1 = xy2[0:1]
    else:
        xy1 = net(z.contiguous().view((1,-1)))
    
    # Compute Drag over Lift values and associated confidence values.
    obj = torch.norm(xy1)   
   
    return obj

def loadProjector(lambdaN= 0.0,loadP=True,sigN = 2,sigA = 0.1,AoA = None): 
    zdim=8
    step=25
    n1 = 16
    n2 = 32
    n3 = 16
    ns = 1000
    
    targetA= 0.1
    sigT = 0.01
    
    dataT = loadAirfoilData(zdim=zdim,batchN=100,trainP=True,step=step,targetA=targetA,ns=ns,sigN=sigN,AoA=AoA,sigA=sigA,sigT=sigT)
    ydim  = dataT.target.size(1)
    
    net   = AreaProjector(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim)
    fName = netwDataName(zdim,n1,n2,n3,lambdaN=lambdaN,sigN=sigN,targetA=targetA)
        
    if(loadP):
        net.restore(fName)
        dataT.restore(fName)
    else:
        dName = netwDataName(zdim,n1,n2,n3,lambdaN=lambdaN,targetA=None)
        net.percept.restore(dName)
        dataT.restore(dName)
        
    net.toGpu()
    return net,dataT,fName
#%%
lambdaN = 0.01
sigN    = 0
net,dat,fName = loadProjector(lambdaN=lambdaN,sigN=sigN,sigA=0.0,loadP=False,AoA=None)
n     = 512
z     = dat.inputs(torch.tensor([n],device=dat.target.device))[0].detach().requires_grad_(True)
#%%
net = net.percept
#%%
optim = torch.optim.Adam([z],lr=0.01)
optim.zero_grad()
CurrL= wingLodDbg(net,z)
CurrL.backward()
g = fromTensor(z.grad)
eps = 1e-3
for i in range(3):
    zi    = z.clone().detach().requires_grad_(False)
    zi[i]+= eps
    CurrI = wingLodDbg(net,zi)
    print(i,g[i],(CurrI-CurrL).item()/eps)