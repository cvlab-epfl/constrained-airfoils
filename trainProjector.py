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
from projarea  import   AreaProjector
#%%---------------------------------------------------------------------------
#                                
#-----------------------------------------------------------------------------
#%%    
zdim=8
step=25
n1 = 16
n2 = 32
n3 = 16
targetA= 0.1
sigN = 2
sigA = 0.1
sigT = 0.01
drawP = True
loadP = False
dataT = loadAirfoilData(zdim=zdim,batchN=400,trainP=True,step=step,targetA=targetA)
testP = True
ydim  = dataT.target.size(1)
fName = netwDataName(zdim,n1,n2,n3)
if(testP):
    fileName='dat/baz'
else:
    fileName=netwDataName(zdim,n1,n2,n3,targetA) 
    
dataT.restore(fName)
net  = AreaProjector(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim)
net.percept.restore(fName)
net.toGpu()
#%%
if(drawP):
    zs,xys0 = dataT.batch(0)
    xys1=net(zs)
    if drawP:
        for i in range(dataT.batchL):
            drawAirfoil(xys0[i],'-r')
            drawAirfoil(xys1[i],'-b')
            plt.pause(1.0)
#%% Train the AreaPorjector
ns  = 1000
ids = torch.tensor(range(ns),device=dataT.target.device)
dataZ = NetData(dataT.inputs(ids).detach(),dataT.target[0:ns].detach(),batchN=10)
net.gtrain(dataZ,fileName=fileName)

        