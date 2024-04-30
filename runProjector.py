#!/usr/bin/env python3
#%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_optimizer as topt

from  util       import fromTensor,makeTensor,setAxes,currentDevice,pytFile
from  optim      import tstG

from airfoildata import loadAirfoilData
from auxfuncs    import drawAirfoil,netwDataName,shoelaceArea1
from decoder     import PerceptronDecoder
from projarea    import AreaProjector
from objective   import loadSurogateModel,wingLodF,predictDraglift

def dispLatentV1(net,model,z,color='-b'):
    cd,cl,xy = predictDraglift(net,model,makeTensor(z))
    cd = cd.item()
    cl = cl.item()
    drawAirfoil(xy,color)
    print('cl {:2.3f}, cd {:2.3f} , lod {:2.3f}'.format(cl,cd,cl/cd))
    
def dispLatentV2(net,model,z1,z2):    
    dispLatentV1(net,model,z1,'-r')
    dispLatentV1(net,model,z2,'-b')
    
def dispLatentV3(net,model,z1,z2,z3):    
    dispLatentV1(net,model,z1,'-r')
    dispLatentV1(net,model,z2,'-g')
    dispLatentV1(net,model,z3,'-b')


zdim=8
step=25
n1 = 16
n2 = 32
n3 = 16
targetA= 0.1
sigN = 2
sigA = 0.1
sigT = 0.01
optP  = True
dataT = loadAirfoilData(zdim=zdim,batchN=400,trainP=True,step=step,targetA=targetA,sigA=sigA,sigN=sigN,sigT=sigT)
#Load data
ydim  = dataT.target.size(1)
dName = netwDataName(zdim,n1,n2,n3)
dataT.restore(dName)
zs,xys0 = dataT.batch(0)
#Load projector
net   = AreaProjector(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim)
pName = netwDataName(zdim,n1,n2,n3,targetA)
net.restore(pName)
net.toGpu()
#Load surrogate
model = loadSurogateModel()
#%%
# n  = 3
# typ = None
# z0  = zs[n].clone().detach().requires_grad_(True)
# xy0 = net(z0.view((1,-1)))[0]
# drawAirfoil(xy0)
#%%----------------------------------------------------------------------------
#                              Lift Over Drag
#------------------------------------------------------------------------------

def getOptimizer(z,lr=0.01,typ=None):
    
    if(typ=='MadGrad'):
        return topt.MADGRAD([z],lr=lr,momentum=0.9,weight_decay=0,eps=1e-6)
    elif(typ=='LFBGS'):
     return torch.optim.LBFGS([z],lr=lr,line_search_fn='strong_wolfe')
 
    return torch.optim.Adam([z],lr=lr)


def torchOtimize(net,z,lambd=0.2,nIt=100,dispP=False,savS=None,typ=None):
    
    #global minLoss
    
    model = loadSurogateModel()
    z     = z.clone().detach().requires_grad_(True)
    z0    = z.clone().detach()
    xy0   = net(z0.view((1,-1)))
    optim = getOptimizer(z,lr=0.01,typ=typ)
    mLoss = [np.inf,z0]
    
    def closure():
        
        optim.zero_grad()
        loss = wingLodF(net,z,model=model,K=None,z0=z0,lambF=lambd)
             
        if(loss<mLoss[0]):
            # Store best current result
            z1       = z.clone().detach()
            mLoss[0] = loss
            mLoss[1] = z1.view(-1)
            if(dispP):
               dispF(z1,savS=savS)
               
        loss.backward()
        return loss
       
    def dispF(z,savS=None):
        
        cd,cl,xy1=predictDraglift(net,model,z)
        ar  = shoelaceArea1(xy1).item()
        label = 'It: {:3d}, loss {:2.3f}, cl: {:2.3f}, cd: {:2.3f} , ar: {:2.3f} , lod: {:2.3f}'.format(it,mLoss[0].item(),cl.item(),cd.item(),ar,(cl/cd).item())
        plt.clf()
        drawAirfoil(xy0,'-r')
        drawAirfoil(xy1,'-b',label=label)
        print(label)
        setAxes(0.0,-0.3,1.0,0.3,legendP=True) 
        plt.pause(0.1)
        if(savS is not None):
            if(it<10):
                plt.savefig('fig/{}00{:d}.png'.format(savS,it))
            elif(it<100):
                plt.savefig('fig/{}0{:d}.png'.format(savS,it))
            else:
                plt.savefig('fig/{}{:d}.png'.format(savS,it))
          
    for it in range(nIt):
        optim.step(closure)
    return mLoss[1]
       
n  = 3
z0  = zs[n].clone().detach().requires_grad_(True)
z1  = torchOtimize(net,z0,lambd=0.1,nIt=30,dispP=True,savS=None)
dispLatentV2(net,model,z0,z1)
