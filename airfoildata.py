#!/usr/bin/env python3
#%%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np
import random

import torch
from netw.miscfuncs import dumpToFile,fromTensor,floatTensor,makeTensor,loadFromFile
from netw.netdata  import NetData

from auxfuncs import loadWingProfiles,batchAffTrf

#%%---------------------------------------------------------------------------
#                                Latent Vector Data
#-----------------------------------------------------------------------------
#%%        
class LatentData(NetData):
    
    def __init__(self,inp,out,batchN=1,sigN=0,sigA=0.1,sigT=0.02):
        
        super(LatentData, self).__init__(inp,out,batchN=batchN)
        # Not too many noisy samples.
        assert(sigN<self.batchL) 
        
        self.setids(randP=False)
        self.sigN = sigN
        self.sigA = sigA
        self.sigT = sigT
        
        ns,nf       = self.inputs.size()
        latentZs    = torch.nn.Embedding(ns,nf,device=self.inputs.device)
        tensorToEmbedding(self.inputs,latentZs)
        
        self.inputs = latentZs
        
    def latent(self):
        
        ns     = self.batchL*self.batchN
        device = self.target.device
        idx    = torch.tensor(np.arange(0,ns,1,dtype=np.int32),device=device)
        return(self.inputs(idx))
        
    def batch(self,i=0):
        
        i   = i % self.batchN
        ids = self.ids[i]
        
        if(self.inputs is not None):
            idx = torch.tensor(ids,device=self.target.device)
            xv  = self.inputs(idx)
        else:
            xv = None
        
        if((self.target is not None) and ((i+self.batchL)<=self.os)):
            yv=self.yv
            ys=self.target[ids]
            yv.copy_(ys)
        else:
            yv = None   
         
        # Add noise to the first N samples
        sigN = self.sigN 
        if(sigN>0):
            sigA = self.sigA 
            sigT = self.sigT
            xys  = yv[0:sigN,:]
            xys  = batchAffTrf(xys,sigA,sigT)
            yv[0:sigN,:] = xys
            
        return xv,yv
    
    def setids(self,randP=False):
        
        if(randP):
            self.ids=randomBatchIndices(self.batchL*self.batchN,self.batchL)
        else:
            bIds   = np.zeros((self.batchN,self.batchL),np.int32)
            fromI  = 0
            for i in range(self.batchN):
                toI = fromI+self.batchL
                bIds[i,:]=np.arange(fromI,toI,dtype=np.int32)
                fromI = toI
            self.ids = bIds
              
    def save(self,fileName):
        saveEmbed(fileName,self.inputs)
            
    def restore(self,fileName): 
        restoreEmbed(fileName,self.inputs)
        
#%%---------------------------------------------------------------------------
#                                Airfoil Data
#-----------------------------------------------------------------------------
        
class WingData(LatentData):
    
    def latentV(self,index): 
        
        idx = torch.tensor(index,device=self.target.device)
        return self.inputs(idx)
    
    def targetV(self,index):
        return self.target[index]
        
#%%      
def loadAirfoilData(zdim=20,trainP=True,batchN=100,step=None,targetA=None,sigN=0,sigA=0.1,sigT=0.02):
    
    ys = loadWingProfiles(step=step,trainP=trainP,targetA=targetA)
    ns = ys.shape[0]
    xs = floatTensor((ns,zdim))
    torch.nn.init.xavier_uniform_(xs)
    
    if(batchN>1):
        batchL = ns // batchN
        ns     = batchL * batchN
        if(ns < ys.shape[0]):
            xs = xs[0:ns]
            ys = ys[0:ns]

    return WingData(xs,ys,batchN=batchN,sigN=sigN,sigA=sigA,sigT=sigT)

#dat=loadAirfoilData(zdim=10,batchN=100,trainP=True,step=11)
#xs,ys = dat.batch(0)
#%%---------------------------------------------------------------------------
#                                Aux Functions
#-----------------------------------------------------------------------------
        
def arrayFromEmbedding(embed):
    params=[]
    for f in list(embed.parameters()):
        f= fromTensor(f)
        params=np.append(params,f)
    return params

def tensorToEmbedding(params,embed):
    fromI = 0
    for f in list(embed.parameters()):
        if(1==f.dim()):
            x=f.size()
            toI=fromI+x[0]
            p=params[fromI:toI]
            f.data[:]=p
        elif(2==f.dim()):
            x,y=f.size()
            toI=fromI+x*y
            p=params[fromI:toI]
            p=p.reshape(x,y)
            f.data[:,:]=p
        else:
            print('Error: tensorToEmbedding not implemented for layers with more than 2 dimensions')
            return None
        fromI=toI
        
def saveEmbed(fileName,embed):
    
    params=arrayFromEmbedding(embed)
    dumpToFile(fileName+'.lat',params)
    
def restoreEmbed(fileName,embed):
    
    params = loadFromFile(fileName+'.lat')
    params = makeTensor(params)
    tensorToEmbedding(params,embed)
    
def randomBatchIndices(ns,batchL):
    
    assert(0==ns%batchL)
    
    bs = []
    xs = {x for x in range(0,ns)}
    
    while(ns > 0):
        batch = np.array(random.sample(list(xs),batchL),dtype=np.int32)
        bs.append(batch)
        
        xs  = xs.difference(batch)
        ns -= batchL
        
    return np.vstack(bs)
    
#%%---------------------------------------------------------------------------
#                                Test
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    ns   = 6
    zDim = 20
    oDim = 3
    x = np.asarray(np.random.rand(ns,zDim),dtype=np.float32)
    y = np.asarray(np.random.rand(ns,oDim),dtype=np.float32)
    dat = LatentData(x,y,2)
    dat.setids(True)
    print(dat.ids)
    dat.batch(0)
    dat.save('old/foo')