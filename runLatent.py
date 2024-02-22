#!/usr/bin/env python3
#%%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from netw.miscfuncs        import makeTensor,fromTensor
from optim       import tstG

from airfoildata import loadAirfoilData
from decoder     import PerceptronDecoder
from auxfuncs    import shoelaceArea2,shoelaceAreaF,drawAirfoil,netwDataName

#from projarea    import prjWing,areaLossF,areaLossG
#%%
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
fileName = netwDataName(zdim,n1,n2,n3)
#%%
for i in range(20):
    xy = dataT.target[i].view((-1,2))
    drawAirfoil(xy)
    print(i,shoelaceArea2(xy).item())
    plt.pause(0.1)
#%% 
loadP = False
net  = PerceptronDecoder(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim,reluP=True)
net.toGpu()
if(loadP):
    # Restore from file
    net.restore(fileName)
    dataT.restore(fileName)
    dataT.setids(randP=False)
else:
    # Train network
    net.gtrain(dataT,fileName=fileName)
#%%
idx = 600
y = dataT.targetV(idx)
z = dataT.latentV(idx)
x = net(z)
plt.clf()
drawAirfoil(y,'-r')
drawAirfoil(x,'-b')
#%%
# from geom  import shoelaceArea1
# from optim import tstG
z = dataT.latentV(10)
x = net(z)
z0 = fromTensor(z)
x0 = fromTensor(x)
drawAirfoil(x0,'-b')
#%%
print(areaLossF(net,z0)+shoelaceAreaF(x0.reshape((-1,2))))
print(areaLossG(net,z0))
#%%
tstG(lambda z : areaLossF(net,z),lambda z : areaLossG(net,z),z0,eps=1e-3)
#%%
targetA = areaLossF(net,z)*2.0
z1=prjWing(net,z0,targetA=targetA)
x1=fromTensor(net(makeTensor(z1)))
drawAirfoil(x0,'-b')
drawAirfoil(x1,'-r')
print(targetA,areaLossF(net,z1),shoelaceArea1(x1),shoelaceAreaF(x1.reshape((-1,2))))

#%%
step = 25
xys0 = loadWingProfiles(step=step)
#%%
areas = np.zeros(xys0.shape[0],dtype=np.float64)
for i,xy in enumerate(xys0): 
    xy = xy.reshape((-1,2))
    areas[i]=shoelaceAreaF(xy)
#%%
targetA = 0.1
xys1 = np.zeros_like(xys0)
for i,xy0 in enumerate(xys0): 
    xy0 = xy0.reshape((-1,2))
    xy1 = affPrjWing(xy0,targetA)
    xys1[i,:]=xy1.flatten()
#%%
saveWingProfiles(xys1,targetA=targetA,step=step)
#%%
xys = loadWingProfiles(step=step,targetA=targetA)
for i in range(10):
    xy = xys[i]
    drawAirfoil(xy,'-r')
    print(i,shoelaceAreaF(xy.reshape((-1,2))))
   