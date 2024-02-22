#!/usr/bin/env python3

#%% ---------------------------------------------------------------------------
#               INTERPOLATING AN IMAGE USING A PERCEPTRON
#% ============================================================================
#%%
import random
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from netw.netdata import NetData
from netw.gradnet import GradNet

from netw.miscfuncs import  pytDir,makeTensor
from netw.layer     import  nonLinear,initLayer,fillLayer,eyeLayer,variableP


#from misc import fillLayer,initLayer,eyeLayer,nonLinear,variableP
#%%----------------------------------------------------------------------------
#                         Basic Multi Layer Perceptron
#------------------------------------------------------------------------------
#%%
class Perceptron(GradNet):
    
    def __init__(self,n1=10,n2=None,n3=None,sparseP=None,reluP=False,nIn=2,nOut=1,lossF=nn.MSELoss()):
        
        super(Perceptron, self).__init__()
            
        self.mY = None
        self.vY = None
        self.sx = None
        self.sy = None
        self.sp = sparseP
        self.rp = reluP
        
        self.setLayers(n1,n2,n3,nIn,nOut)
        self.lossFunction=lossF
        
    def setLayers(self,n1,n2,n3,nIn=2,nOut=1):
        
        self.l1 = nn.Linear(nIn,n1)
        self.n1 = n1
        
        if((n2 is not None) and (n2>0)):
            self.l2=nn.Linear(n1,n2)
            self.n2 = n2
            if((n3 is not None) and (n3>0)):
                self.l3=nn.Linear(n2,n3)
                self.ln=nn.Linear(n3,nOut)
                self.n3 = n3
            else:
                self.l3=None
                self.ln=nn.Linear(n2,nOut)
        else:
            self.l2=None
            self.l3=None
            self.ln=nn.Linear(n1,nOut)
        
    def initLayers(self,scale=None,bias=None):
               
        initLayer(self.l1,scale,bias)
        initLayer(self.l2,scale,bias)
        initLayer(self.l3,scale,bias)
        initLayer(self.ln,scale)
        
    def forward(self,x):
        
        y = self.l1(x)
        
        if (self.l2 is not None):
            y = self.l2(nonLinear(y,self.rp))
        if (self.l3 is not None):
            y = self.l3(nonLinear(y,self.rp))
        
        return(self.ln(nonLinear(y,self.rp)))
        
    def addNodes(self,dn=1,sigma=1e-3):
        
         newNet=Perceptron(self.n+dn,nl=self.nl,nb=0,sp=self.sp,reluP=self.rp)
         
         fillLayer(self.l1,newNet.l1,sigma=sigma)
         if(self.l2 is not None):
             fillLayer(self.l2,newNet.l2,sigma=sigma)
         if(self.l3 is not None):
             fillLayer(self.l3,newNet.l3)
         fillLayer(self.ln,newNet.ln,sigma=sigma)
         
         return(newNet)
        
    def addLayer(self):
        
        newNet=Perceptron(self.n,nl=self.nl+1,nb=0,sp=self.sp,reluP=self.rp)
        
        fillLayer(self.l1,newNet.l1)
        fillLayer(self.ln,newNet.ln)
        
        if(1==self.nl):
            eyeLayer(newNet.l2)
        elif(2==self.nl):
            fillLayer(self.l2,newNet.l2)
            eyeLayer(newNet.l3)
        else:
            print('addLayer: Cannot add a fourth layer')
        
        return(newNet)
    
    def lossFunction(self,output,target):
        
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn (output,target)
        
        if(self.sp is not None):
            addLoss = 0.0
            for f in list(self.parameters()):
                l = torch.norm(f,2)
                addLoss += l*l
            loss += self.sp  * addLoss
        
        return(loss)
    
    def randt(self,data,lr=0.0001,fileName='/tmp/foo',nIt=10000,nRep=10,fileAux='/tmp/bar',sigma=1e-3):
    
         
        bestLoss=self.train(data,nIt=nIt,lr=lr,fileName=fileName)
        for j in range(nRep):
            self.rndState(sigma)
            loss=self.train(data,nIt=nIt,lr=lr,fileName=fileAux)  
            if(loss<bestLoss):
                bestLoss =loss
                self.restore(fileAux)
                self.save(fileName)
        else:
            self.restore(fileName)
        
        self.restore(fileName)
        return(bestLoss)
    
  
    def name (self,datName='foo',datDir=None):
        
        if(datDir is None):
            datDir= pytDir('learn/percept/dat')
        if(self.l3 is not None):
            return '{}/{}.{}.{}.{}'.format(datDir,datName,self.n1,self.n2,self.n3)
        elif(self.l2 is not None): 
            return '{}/{}.{}.{}'.format(datDir,datName,self.n1,self.n2)
        else:
            return '{}/{}.{}'.format(datDir,datName,self.n1)
        
    def rmse(self,data,dispP=False,maxV=0.1):
    
        X=torch.FloatTensor(self.X)
        Y=self(X)
        dY=(self.Y-Y.data.numpy())**2
    
        if(hasattr(data, 'anchors')):
            label='Rms anchors: {:2.3e}, Rms others: {:2.3e}'.format(dY[data.anchors].mean(),dY[data.others].mean())
        else:
            label='Rms all: {:2.3e}'.format(dY.mean())
        
        if(dispP):
            self.disp(Y=self.Y-Y.data.numpy(),maxV=maxV)
            plt.xlabel(label)
            plt.pause(0.01)
        else:
            print(label)
            
    def prnt(self,output,target):
        
        if(variableP(output) and variableP(target)):
            loss=self.lossFunction(output,target).data[0]
        else:
            loss=0.0
        npar=self.numParams()
        if(self.l3 is not None):
            print('3 layers of width {0:d} {1:d} {2:d}, {3:d} weights -> loss {4:e}'.format(self.n1,self.n2,self.n3,npar,loss))
        elif(self.l2 is not None):
            print('2 Layers of width {0:d} {1:d}, {2:d} weights -> loss {3:e}'.format(self.n1,self.n2,npar,loss))
        else:
            print('1 Layer of width {0:d}, {1:d} weights -> loss {2:e}'.format(self.n1,npar,loss))

if __name__ == "__main__":
    nIn=5
    nOut=2
    ns=4
    net = Perceptron(5,None,None,nIn=nIn,nOut=nOut)
    xs  = np.array(np.random.randn(ns,nIn),dtype=np.float32)
    ys  = np.array(np.random.randn(ns,nOut),dtype=np.float32)
    data=NetData(xs,ys,batchN=1)
    xb,yb,_=data.batch(0)
    net.toGpu()
    net.forward(xb)
    net.gtrain(data)
    
#%%----------------------------------------------------------------------------
#                         Classification
#------------------------------------------------------------------------------
#%% Binary classification. The targets are expected to zero or one.
class BinaryPerceptron(Perceptron):
    
     def forward(self,x):
        
         y=super(BinaryPerceptron,self).forward(x)
         return y.view(-1)
     
     def lossFunction(self,output,target):
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn (output,target)
        
        return(loss)
     
     def accuracy(self,xt=None,yt=None):
         
        if(xt is None):
            xt = self.xs
        if(yt is None):
            yt = self.ys
         
        nt = xt.shape[0]
        zt = self.predict(xt,binP=True)
        prec = np.equal(yt,zt).sum()/nt
        
        print('Accuracy: {:f}'.format(100*prec))
        return prec
     
     def predict(self,xs,binP=True):
        
        self.eval()
        
        sig = nn.Sigmoid()
        
        X   = makeTensor(xs)
        
        if(binP):
            # Return only binary value. No need to actually compute the sigmoid.
            P = (self(X)>0)
        else:
            # Return floating point value.
            P   = sig(self(X))
        
        return(np.ravel(P.data.cpu().numpy()))
#%% Multi class classification. The targets are expected to be the desired class.
class LogisticPerceptron(Perceptron):
    
     def lossFunction(self,output,target):
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn (output,target)
        
        return(loss)
     
     def test(self,xs,ys):
         
         ns = xs.shape[0]
         zs = self.predict(xs)
         prec = np.equal(ys,zs).sum()/ns
        
         print('Accuracy: {:f}'.format(100*prec))
         return prec
     
     def predict(self,xs):
        
        self.eval()
        
        X   = makeTensor(xs)
        P   = self(X)
        
        return(np.ravel(P.data.max(1)[1].cpu().numpy()))
#%%       
if __name__ == "__main__":
    nIn=5
    nOut=4
    ns=5
    net = LogisticPerceptron(5,12,7,nIn=nIn,nOut=nOut)
    xs  = np.array(np.random.randn(ns,nIn),dtype=np.float32)
    ys  = np.zeros(ns,dtype=np.long)
    for i in range(ns):
        ys[i] = random.randrange(nOut)
    data= NetData(xs,ys,batchN=2,intP=True)
    print('loss: ',net.computeLoss(data))
    net.gtrain(data,nIt=2000)
    print(net.predict(xs))
    print(ys)
#%%
