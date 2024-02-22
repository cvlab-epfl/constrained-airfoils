#!/usr/bin/env python3
#%%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import torch

from netw.netdata import NetData


from airfoildata import loadAirfoilData
from auxfuncs    import drawAirfoil,netwDataName,shoelaceArea1
from projarea   import  AreaProjector

from decoder    import PerceptronDecoder

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
dataV = loadAirfoilData(zdim=zdim,batchN=1,trainP=False,step=step,targetA=targetA)
ydim  = dataT.target.size(1)
drawAirfoil(dataT.target[0])
fName = netwDataName(zdim,n1,n2,n3)

#%%---------------------------------------------------------------------------
#                            Train Autodecoder Alone
#-----------------------------------------------------------------------------
#%% 
loadP = False
net  = PerceptronDecoder(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim,reluP=True)
net.toGpu()
if(loadP):
    # Restore from file
    net.restore(fName)
    dataT.restore(fName)
    dataT.setids(randP=False)
else:
    # Train network
    net.gtrain(dataT,fileName=fName, nIt=100)
#%%---------------------------------------------------------------------------
#                          Train Autodecoder + Projector
#-----------------------------------------------------------------------------
#%%  
loadP = False  
fileName=netwDataName(zdim,n1,n2,n3,targetA) 
dataT.restore(fName)
net  = AreaProjector(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim)
if(loadP):
    net.restore(fileName)
else:
    net.percept.restore(fName)
net.toGpu()
#%%
if(loadP):
    zs,xys0 = dataT.batch(0)
    xys1=net(zs)
    for i in range(dataT.batchL):
        drawAirfoil(xys0[i],'-r')
        drawAirfoil(xys1[i],'-b')
        plt.pause(1.0)
#%% Train the AreaPorjector
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ns  = 1000
    ids = torch.tensor(range(ns),device=dataT.target.device)
    dataZ = NetData(dataT.inputs(ids).detach(),dataT.target[0:ns].detach(),batchN=10)
    net.gtrain(dataZ,fileName=fileName, nIt=2)
#%% Interpolate between two latent vectors
zs,xys0 = dataT.batch(0)
i1  = 0
i2  = 2
z1  = zs[i1].view((1,-1))
z2  = zs[i2].view((1,-1))
for i,l in enumerate(np.linspace(0.0,1.0,10)):
    xy1 = net(l * z1 + (1-l) *z2)[0]
    drawAirfoil(xy1,'-b')
    print(i,shoelaceArea1(xy1).item())
    plt.pause(0.01)
    
# %%
