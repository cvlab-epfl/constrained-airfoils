#!/usr/bin/env python3
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd import Variable

# from util import raiseError
def raiseError(msg):
    raise Exception(msg)
#%%----------------------------------------------------------------------------
#                             Variables
#------------------------------------------------------------------------------

#%%
def variableP(x):
    return(isinstance(x,Variable))
    
#%%----------------------------------------------------------------------------
#                           Layers
#------------------------------------------------------------------------------

def initWeights (m,wMin=None,wMax=None,verbP=False):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if(wMin is None):
            wMin=0.0
        if(wMax is None):
            wMax=0.02
        m.weight.data.normal_(wMin,wMax)
        
    elif classname.find('BatchNorm') != -1:
        if(wMin is None):
            wMin=1.0
        if(wMax is None):
            wMax=0.02
        m.weight.data.normal_(wMin,wMax)
        m.bias.data.fill_(0)
        
    elif(verbP):
        print('initWeights: Not implemented.')

def zeroOutLayer(l,sigma=0):
    if(l is not None):
        if(sigma>0.0):
            if(2==len(l.weight.data.size())):
                m,n=l.weight.data.size()
                l.weight.data[:,:]  = sigma*torch.randn(m,n)
                l.bias.data[:]      = sigma*torch.randn(m)
            elif(4==len(l.weight.data.size())):
                m,n,w1,w2=l.weight.data.size()
                l.weight.data[:,:]  = sigma*torch.randn(m,n,w1,w2)
                l.bias.data[:]      = sigma*torch.randn(m)
        else:
            torch.zeros(l.bias.size(),out=l.bias.data)
            torch.zeros(l.weight.size(),out=l.weight.data)
    
def fillLayer(fromL,toL,sigma=1e-3):
    if(toL is not None):
        assert(fromL is not None)
        
        if(2==len(toL.weight.data.size())):
            n1,n2=fromL.weight.data.size()
            m1,m2=toL.weight.data.size()
            
            assert((n1<=m1) and (n2<=m2))
            
            toL.bias.data[0:n1]       = fromL.bias.data
            if(n1<m1):
                toL.bias.data[n1:m1]  = sigma*torch.randn(m1-n1)
                
            toL.weight.data[0:n1,0:n2]= fromL.weight.data
            if(n1<m1):
                toL.weight.data[n1:m1,:]   =sigma*torch.randn(m1-n1,m2)
            if(n2<m2):
                toL.weight.data[0:n1,n2:m2]=sigma*torch.randn(n1,m2-n2)
                
        elif(4==len(toL.weight.data.size())):  
             
             toL.weight.data[:,:,:,:]= fromL.weight.data
             toL.bias.data[:]  = fromL.bias.data
                          
        else:
            raiseError('eyeLayer: Unknown layer type.')
         
def initLayer(l,scale=None,bias=None):
    if(l is not None):
        l.reset_parameters()
        if(scale is not None):
            l.weight.data.mul_(scale)
            l.bias.data.mul_(scale)
        if(bias is not None):
            l.bias.data.add_(bias)

def eyeLayer(l):
    if(l is not None):
        if(2==len(l.weight.data.size())):
            n1,n2=l.weight.data.size()
            assert(n1==n2)
            l.weight.data[:,:]=torch.eye(n1)
            l.bias.data[:]    =torch.zeros(n1)
            
        elif(4==len(l.weight.data.size())):  
            n1,n2,w1,w2=l.weight.data.size()
            assert(w1==w2)
            zeroOutLayer(l)
            l.weight.data[:,:,w1//2,w2//2]=1.0
                         
        else:
            raiseError('eyeLayer: Unknown layer type.')
#%%   
def gridLayer(l,n):
    
    n2=int(n/2)
    dx=2.0/n2
    dy=2.0/n2
    x=-1.0+dx/2
    for i in range(0,n2):
        l.weight.data[i,0]=1.0
        l.weight.data[i,1]=1e-5
        l.bias.data[i]=x
        x+=dx
    y=-1.0+dy/2
    for i in range(n2,n):
        l.weight.data[i,0]=1e-5
        l.weight.data[i,1]=1.0
        l.bias.data[i]=y
        y+=dy
#%%     
def nonLinear(y,reluP):
    
    if(reluP is False):
        return(torch.tanh(y))
    elif (reluP is True):
        return(F.relu(y))      
    else:
        reluP=float(reluP)
        return(nn.LeakyReLU(reluP)(y))