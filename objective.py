#!/usr/bin/env python3
#%%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np

import torch
import scipy.optimize as opt

from  util       import fromTensor,makeTensor,pytFile,currentDevice
from  auxfuncs   import shoelaceArea1

from lod.MLP        import MLP
from lod.GraphSage  import GraphSAGE

#%%---------------------------------------------------------------------------
#                                STIFFNSSS
#-----------------------------------------------------------------------------

def stiffMat (n,allP=False):
    
    
    K  = torch.zeros((n,n),dtype=torch.float32)
    ws = torch.tensor([1,-4,6,-4,1],dtype=torch.float32)
    
    if(allP):
        K[0,0:3]  = ws[2:]
        K[0,-2:]  = ws[0:2]
        K[1,0:4]  = ws[1:]
        K[1,-1]   = ws[0]
    for i in range(2,n-2):
        if(allP or (i < (n//2 - 1)) or (i >= (n//2 + 1))):
            K[i,i-2:i+3]=ws
    if(allP):
        K[-1,-3:] = ws[0:3]
        K[-1,0:2] = ws[3:]
        K[-2,-4:] = ws[0:4]
        K[-2,0]   = ws[4]
    
    return K

def stiffnessF(K,xy1):
    
    if((2==len(xy1.size())) and (2==xy1.size(1))):
        xy2 = xy1
    else:
        xy2 = xy1[0].view((-1,2))
    
    xs  = xy2[:,0]
    ys  = xy2[:,1]
    
    return 0.5 * (xs.T @ K @ xs +  ys.T @ K @ ys)
    
if __name__ == "__main__":
    K = stiffMat (27)
    print(K) 

#%%----------------------------------------------------------------------------
#                                 Lift Over Drag
#%%----------------------------------------------------------------------------

def loadSurogateModel(lodP=False,modelType='mlp'):
    
    device=currentDevice()
    
    model = MLP(predict_lod=lodP).to(device) if modelType == 'mlp' else GraphSAGE(lodP).to(device)
    if(modelType  == 'mlp'):
        if(lodP):
            model.load_state_dict(torch.load(pytFile('misc/wings/2d/lod/mlp_state_dict_lod.pth'),map_location=device))
        else:
            model.load_state_dict(torch.load(pytFile('misc/wings/2d/lod/mlp_state_dict_cdl.pth'),map_location=device))
    else:
        if(lodP):
            model.load_state_dict(torch.load(pytFile('misc/wings/2d/lod/gnn_state_dict_lod.pth'),map_location=device))
        else:
            model.load_state_dict(torch.load(pytFile('misc/wings/2d/lod/gnn_state_dict_cdl.pth'),map_location=device))
    
    return model

kludgeP=True
def wingLodF(net,z,model=None,K=None,targetA=None,z0=None,lambF=0.1,lambA=1.0):
    
    if(model is None):
        model = loadSurogateModel()
        
    gradP = z.requires_grad 
    
    if(gradP and kludgeP):
        # Kludge: If we pass a batch of only one, backward will die.
        zd    = z.size(0)
        z2  = torch.vstack([z,torch.zeros(zd,dtype=torch.float32,device=z.device).requires_grad_(gradP)])
        xy2 = net(z2)
        xy1 = xy2[0:1]
    else:
        xy1 = net(z.contiguous().view((1,-1)))
    
    cd, cl  = model(xy1)[0]
    obj     = cd / cl
    
    if(targetA is not None and lambA > 0.0):
        da = shoelaceArea1(xy1)-targetA
        obj = obj + lambA * da * da
      
    if(z0 is not None and lambF > 0.0):
        #z0     = z0.view((1,-1))
        lossL1 = torch.nn.L1Loss()
        obj    = obj + lambF*lossL1(z, z0)
        
    if(K is not None):
        obj = obj + stiffnessF(K,xy1)
        
    return obj

def wingLodG(net,z,model=None,K=None,z0=None,lambF=0.1):
    
    assert(z.requires_grad)
    
    obj = wingLodF(net,z,model=model,K=K,z0=z0,lambF=0.1)
    obj.backward()
    grad = z.grad
    
    return grad

def predictDraglift(net,model,z):
    
    xy = net(z.view((1,-1)))
    cd, cl = model(xy)[0]
    
    return cd,cl,xy[0]