#!/usr/bin/env python3
#%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%
import numpy as np
import scipy.optimize as opt


import torch
import torch.nn as nn

from ddn.pytorch.node import EqConstDeclarativeNode,DeclarativeLayer

from netw.gradnet     import GradNet

from decoder  import PerceptronDecoder
from auxfuncs import fromTensor,currentDevice,shoelaceArea1,shoelaceAreaF,affTrf
from auxfuncs import loadWingProfiles,drawAirfoil
#%%
class AreaProjector(GradNet):
    
    lossFunction = nn.MSELoss()
    
    def __init__(self,n1=16,n2=32,n3=16,nIn=8,nOut=54,targetA=0.1):
        
        super().__init__()
        
        ddn  = constAreaNode(targetA)
        ddn.eps = 1e-3
        
        self.percept  = PerceptronDecoder(n1=n1,n2=n2,n3=n3,nIn=nIn,nOut=nOut,reluP=True)
        self.prjlayer = DeclarativeLayer(ddn)
        
    def toGpu(self):
        
        super().toGpu()
        self.percept.toGpu()
         
    def forward(self,zs):
        
        xys = self.percept(zs)
        xys = self.prjlayer(xys)
        
        return xys
#%% Load wing data
if __name__ == "__main__":
    xys=loadWingProfiles(step=25)
    xyb=torch.tensor(xys[0:3,:],requires_grad=True,device=currentDevice())
    xy0=xyb[0]
    drawAirfoil(xy0)
#%%----------------------------------------------------------------------------
#                              Projection
#------------------------------------------------------------------------------
#%%
def affPrjWing(xy,targetA,drawP=False):
    
    xy0 = xy.copy()
    
    def objF(aff):
        
        xy1 = affTrf(aff,xy0)
        d01 = xy0-xy1
        return 0.5*np.sum(d01*d01)
    
    def cstF(aff):
        
        xy1 = affTrf(aff,xy0)
        return shoelaceAreaF(xy1)-targetA
        
    aff0 = np.array(([1.0,0.0,0.0,1.0,0.0,0.0]),dtype=np.float32)
    cons = ({'type': 'eq', 'fun': cstF})
    bnds = opt.Bounds(-100.0,100.0,keep_feasible=True)
    
    res =  opt.minimize(objF,aff0,method='SLSQP',constraints=cons,bounds=bnds)
    
    if(not(res.success)):
        print('affPrjWing(: Minimization failed')
      
    # Return the deformed contour
    aff1 = res.x
    xy1  = affTrf(aff1,xy0)
        
    if(drawP):
        drawAirfoil(xy0,'-b')
        drawAirfoil(xy1,'-r')
            
    return(xy1)

if __name__ == "__main__": 
    xyv = fromTensor(xy0.view((-1,2)))
    xy1 = affPrjWing(xyv,0.1)
    print(shoelaceAreaF(xyv),shoelaceAreaF(xy1))
#%%   
class constAreaNode(EqConstDeclarativeNode):
    
    def __init__(self,targetA):
        
        super().__init__()
        self.targetA = targetA

    def objective(self,x,y=None):
    
        d1 = x-y
        d2 = d1*d1
        return 0.5*d2.sum(dim=1)
    
    def equality_constraints(self,x,y=None):
 
        targetA = self.targetA
        if(True):
            cs      = []
        else:
            ns      = x.size(0)
            cs      = torch.zeros(ns,dtype=torch.float32,requires_grad=True)
        
        with torch.enable_grad():
            for i, xy in enumerate(y):   
                areaI  = shoelaceArea1(xy)
                if(True):
                    cs.append(areaI-targetA)
                else:
                    cs[i] = areaI-targetA
        
        if(True):
            return torch.vstack(cs)
        else:
            return cs 
        
    def solve(self,xys0):
        
        #xys1    = torch.zeros_like(xys0,requires_grad=False)
        
        xys1    = np.zeros(xys0.size(),dtype=np.float32)
        
        targetA = self.targetA 
                
        for i, xy in enumerate(xys0):
            xy0 = fromTensor(xy).reshape((-1,2))
            xy1 = affPrjWing(xy0,targetA).flatten()
            xys1 [i,:] = xy1
            #xys1 [i,:] = torch.tensor(xy1)

        return torch.tensor(xys1,device=xys0.device,requires_grad=True),None
#%%
if __name__ == "__main__":
    ddn  = constAreaNode(0.1)
    ddn.eps = 1e-3
    decL = DeclarativeLayer(ddn)
    xy1  = decL(xyb)
    eql1 = ddn.equality_constraints(xyb,y=xy1)
    ob1  = ddn.objective(xy0,y=xy1)
    grd  = ddn.gradient(xyb,y=xy1)
#%%
if __name__ == "__main__":
    obj1  = torch.sum(xy1 @ xy1.T)
    obj1.backward()
    grd1  = xyb.grad
#%% Compare analytical gradients to finite difference ones
if __name__ == "__main__":
    eps = 1e-3
    for i in range(0,3):
        for j in range(0,5):
            xyc = xyb.clone().detach()
            xyc[i,j] += eps
            xy2  = decL(xyc)
            obj2 = torch.sum(xy2 @ xy2.T)
            print(i,j,grd1[i,j].item(),(obj2-obj1).item()/eps)
            
    