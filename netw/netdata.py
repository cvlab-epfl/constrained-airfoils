#!/usr/bin/env python3
#-----------------------------------------------------------------------------
#                         
#=============================================================================

##%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%
import numpy as np
import torch
import random

from netw.miscfuncs  import dumpToFile,loadFromFile,floatTensor,longTensor,makeTensor
from netw.layer      import variableP
#%%

#%%
class NetData():
    
    def __init__(self,inp,out,ws=None,batchN=1,intP=False,name=None,loadP=False):
        
        if(loadP):
            inp,out,ws=self.load(name)
            
        self.batchN=batchN
        
        # Inputs
        if(inp is None):
            self.inputs=None
        elif(variableP(inp)):
            self.inputs=inp
        else:
            #self.inputs=torch.tensor(inp,dtype=torch.float32,requires_grad=False) 
            self.inputs=makeTensor(inp)
          
        # Outputs
        if(out is None):
            self.target=None
        elif(variableP(out)):
            self.target=out
        elif intP:
            self.target=longTensor(out) 
        else:
            self.target=makeTensor(out)
            
        if(ws is None):
            self.weights=None
        elif(variableP(ws)):
            self.weights=ws
        else:
            self.weights=makeTensor(ws)
            
        if(inp is not None):
            self.ns     = self.inputs.size(0)
            self.batchL = self.ns // batchN
            dims    = list(self.inputs.size())
            dims[0] = self.batchL
            self.xv = floatTensor(*dims)           
        else:
            self.batchL = 0
            self.xv = None
            self.ns = 0
        
        if(out is not None):
            dims    = list(self.target.size())
            if(self.batchL>0):
                dims[0] = self.batchL
            if(intP):
                self.yv  = longTensor(*dims)
            else:
                self.yv  = floatTensor(*dims)
            self.os = self.target.size(0)
        else:
            self.yv = None
            self.os = 0
            
        if(ws is not None):
            dims    = list(self.weights.size())
            if(self.batchL>0):
                dims[0] = self.batchL
            self.wv = floatTensor(*dims)
        else:
            self.wv = None
            
        assert(self.ns==(self.batchL*self.batchN))
            
            
    def shuffle(self):
        
        if(self.batchN>1):
            pns = torch.LongTensor(np.random.permutation(range(self.ns)))
            if(self.inputs is not None):
                self.inputs=self.inputs[pns]
            if(self.target is not None):
                self.target=self.target[pns]
            if(self.weights is not None):
                self.weights=self.weights[pns]
        
    def batch(self,i=0):
        
        i = i % self.batchN
        i = i * self.batchL
        
        if(self.inputs is not None):
            xv=self.xv
            xs=self.inputs[i:i+self.batchL]
            xv.copy_(xs)
        else:
            xv = None
        
        if((self.target is not None) and ((i+self.batchL)<=self.os)):
            yv=self.yv
            ys=self.target[i:i+self.batchL]
            yv.copy_(ys)
        else:
            yv=None
            
        if(self.weights is not None):
            wv=self.wv
            ws=self.weights[i:i+self.batchL]
            wv.copy_(ws)
        else:
            wv = None
            
        return xv,yv,wv
        
    def sample(self,n):
        
        indices=random.sample(range(self.ns),n)
        if(self.inputs is not None):
            xs = self.inputs[indices]
        else:
            xs = None
        if(self.target is not None):
            ys = self.inputs[indices]
        else:
            ys=None
        if(self.weights is not None):
            ws = self.weights[indices]
        else:
            ws=None
        return xs,ys,ws,indices

    def save(self,fileName):  
        if(fileName is not None):
            if(self.inputs is not None):
                dumpToFile(fileName+'.inp',self.inputs.numpy())
            if(self.target is not None):
                dumpToFile(fileName+'.out',self.target.numpy())
            if(self.weights is not None):
                dumpToFile(fileName+'.wgs',self.weights.numpy())
                
    def load(self,name):
         xs = loadFromFile(name+'.inp',verbP=False)
         ys = loadFromFile(name+'.out',verbP=False)
         ws = loadFromFile(name+'.wgs',verbP=False)
         if(xs is not None):
             print('Data loaded from {}'.format(name))
             return xs,ys,ws
         else:
             print('Could not load data from {}'.format(name))
             return None,None,None
#%%
#if __name__ == "__main__":
#    from mnist import selectMNIST 
#    xt,yt=selectMNIST('train')
#    wt = np.random.randn(xt.shape[0])
#    data=NetData(xt,yt,ws=wt,batchN=4)
#    print(data.weights)
#    data.shuffle()
#    xb,yb,wb=data.batch(0)
#    print(xb.size(),yb.size(),wb.size())
#%%
#if __name__ == "__main__":
#    nIn=3
#    nOut=3
#    ns=10
#    xs  = np.array(np.random.randn(ns,nIn),dtype=np.float32)
#    ys  = np.array(np.random.randn(ns,nOut),dtype=np.float32)
#    data=NetData(xs,ys,batchN=3)
#    xb,yb,wb=data.batch(1)
#    print(xb,yb,wb)
#%%
#if __name__ == "__main__":
#    dims=list(data.inputs.size())
#    dims[0]=1000
#    torch.FloatTensor(*dims)
#%%
#if __name__ == "__main__":
#    x=torch.randn(2,3)
#    y=Variable(x)
#    print(isinstance(x,Variable),isinstance(y,Variable))
#    print(x.type,y.type)
