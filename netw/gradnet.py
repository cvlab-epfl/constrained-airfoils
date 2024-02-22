#!/usr/bin/env python3
#-----------------------------------------------------------------------------
#                         
#=============================================================================

#-----------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%

import torch,os,gc
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from netw.miscfuncs  import makeTensor,fromTensor,dumpToFile,loadFromFile,cudaOnP,mpsOnP
from netw.miscfuncs  import setLogFile,logf
#%%
class GradNet(nn.Module):
    
    dbgP=True
    
    def getState(self):
        
        params=[]
        for f in list(self.parameters()):
            f=np.ravel(f.data.cpu().numpy())
            params=np.append(params,f)
        return params
    
    def setState(self,params):
        
        fromI=0
        for f in list(self.parameters()):
            if(1==f.dim()):
                x=f.size()
                toI=fromI+x[0]
                p=params[fromI:toI]
                f.data[:]=torch.FloatTensor(p)
            elif(2==f.dim()):
                x,y=f.size()
                toI=fromI+x*y
                p=params[fromI:toI]
                p=p.reshape(x,y)
                f.data[:,:]=torch.FloatTensor(p)
            elif(3==f.dim()):
                x,y,z=f.size()
                toI=fromI+x*y*z
                p=params[fromI:toI]
                p=p.reshape(x,y,z)
                f.data[:,:,:]=torch.FloatTensor(p)
            elif(4==f.dim()):
                w,x,y,z=f.size()
                toI=fromI+w*x*y*z
                p=params[fromI:toI]
                p=p.reshape(w,x,y,z)
                f.data[:,:,:,:]=torch.FloatTensor(p)
            else:
                print('Error: setParams not implemented for layers with more than 4 dimensions')
                break
          
            fromI=toI
            
    def rndState(self,sigma=1e-4):
        
        if(sigma>0.0):
            state=self.getState()
            self.setState(state + sigma*np.random.randn(len(state)))
            
      
    def getGrads(self):
        
        grads=[]
        for f in list(self.parameters()):
            f=np.ravel(fromTensor(f.grad))
            grads=np.append(grads,f)
        return grads
            
    def setGrads(self,grads):
            
            grads=makeTensor(grads)
            fromI = 0
            
            for f in list(self.parameters()):
                g = f.grad.flatten()
                toI   = fromI + g.size(0)
                g[:]  = grads[fromI:toI]
                fromI = toI
                    
            assert(toI==grads.size(0))
  
    def numParams(self):
    
        n=0
        for f in list(self.parameters()):
            f = f.view(-1,1)
            s = f.size()[0]            
            n+=s
        return n
    
    def freeCache(self):
        # By default do nothing
        pass
    
    def toGpu(self):
        
        if(cudaOnP()):
            super(GradNet,self).cuda()
        elif mpsOnP():
            mps_device = torch.device("mps")
            super(GradNet,self).to(mps_device)
    
    def save(self,fileName,epoch=0,loss=0.0,datP=False):
        
        state=self.getState()
        dumpToFile(fileName+'.sta',state)
        if(datP):
            torch.save({
                    'epoch': epoch,
                    'state': self.state_dict(),
                    'bloss': loss
                    }, fileName+'.dat' )
    
    def load(self,fileName):
        
        fileName=fileName+'.dat'
        if os.path.isfile(fileName):
            checkpoint = torch.load(fileName)
            loss  = checkpoint['bloss']
            epoch = checkpoint['epoch']
            
            self.load_state_dict(checkpoint['state'])
            print('=> loaded checkpoint {} (loss {}, epoch {})'.format(fileName,loss,epoch))
            
    def restore(self,fileName,verbP=True):
        
        state=loadFromFile(fileName+'.sta',verbP=verbP)
        if(state is None):
            return False
        
        try:
            self.setState(state)            
        except RuntimeError:
            print('Failed to restore state from file {}.'.format(fileName))
            return False
            
        return True
    
    def predict(self,xs,**kargs):
        
         self.train(False)
           
         with torch.no_grad():
            X   = makeTensor(xs)
            Z   = self(X,**kargs)
            zs  = fromTensor(Z)
            return zs

    
    def cnjF(self,inputs,targets,gradP=True):
    
        outputs = self(inputs)
        loss    = self.lossFunction(outputs,targets)
        if(gradP):
            self.zero_grad()
            loss.backward()

        return loss.item() 
    
    def computeLoss(self,data,optimizer=None,randP=True):
        
        if(isinstance(data,torch.utils.data.dataloader.DataLoader)):
            loss=self.computeLossD(data,optimizer)         
        else:
            loss=self.computeLossB(data,optimizer,randP=randP)
            
        return loss
        
    # Data is a dataloader.
    def computeLossD(self,data,optimizer):
        
        loss = 0.0
        batc = 0
        
        for inputsB,targetB in data:
            inputsB   = makeTensor(inputsB)
            targetB   = makeTensor(targetB)
            outputsB  = self(inputsB)
            lossB     = self.lossFunction(outputsB,targetB)          
            loss     += lossB
            
            if(self.dbgP):
                assert not torch.isnan(inputsB).any(),  "computeLoss: input  contains nan"
                assert not torch.isnan(outputsB).any(), "computeLoss: output contains nan"
                assert not torch.isnan(targetB).any(),  "computeLoss: target contains nan"
                
            if(optimizer is not None):
                optimizer.zero_grad()
                lossB.backward()
                optimizer.step()
                
            # Invoke the garbage collector at regular intervals
            batc+=1
            if(cudaOnP() and (batc%10)):
                gc.collect() 
                
        if(cudaOnP()):
            gc.collect() 
        
        loss /= batc        
        return loss 
    
        
    # Compute the weighted or non weighted loss depending on whether the weights 
    # are set in data. Also takes a step after computing the loss for each batch 
    # if the optimizer is specified. 
    def computeLossB(self,data,optimizer,randP=True):
        
        loss = 0.0
        batc = 0
                
        # Randomize the order.
        if(randP):
            data.shuffle()
            
        for batc in range(data.batchN):
            inputsB,targetB,weightsB=data.batch(batc)
            outputsB  = self(inputsB)
            lossB     = self.weigthedLoss(outputsB,targetB,weightsB)            
            loss     += lossB
            
            if(self.dbgP):
                assert not torch.isnan(inputsB).any(),  "computeLoss: input  contains nan"
                assert not torch.isnan(outputsB).any(), "computeLoss: output contains nan"
                assert not torch.isnan(targetB).any(),  "computeLoss: target contains nan"
                
            if(optimizer is not None):
                optimizer.zero_grad()
                lossB.backward()
                optimizer.step()
                
            # Invoke the garbage collector at regular intervals
            if(cudaOnP() and ((batc+1)%10)):
                gc.collect() 
                
        if(cudaOnP()):
            gc.collect() 
        
        loss = loss/data.batchN
        return(loss)
    
    def weigthedLoss(self,outputsB,targetB,weightsB):
        
        if(weightsB is None):
            return self.lossFunction(outputsB,targetB)
        else:
            ns  = weightsB.size(0)
            ws  = torch.sqrt(weightsB).view(ns,1,1)
            out = torch.bmm(ws,outputsB.view(ns,1,1))
            tar = torch.bmm(ws,targetB.view(ns,1,1))
            mse = self.lossFunction(out.view(-1),tar.view(ns))
            
            return (ns/torch.sum(weightsB))*mse
        
    def checkpoint(self,dataT,dataV,epoch=-1,minLoss=None,fileName=None):
        
        with torch.no_grad():
            if(dataV is None):
                currentLoss = self.computeLoss(dataT,optimizer=None,randP=False)               
            else:
                currentLoss = self.computeLoss(dataV,optimizer=None,randP=False)
        if(minLoss is None):
            minLoss = currentLoss

        logf.info('[%4d] current loss: %.3e min loss: %.3e',epoch+1,currentLoss,minLoss)
            
        if(currentLoss<minLoss):
            minLoss=currentLoss
            if(fileName is not None):
                self.save(fileName,epoch+1,minLoss) 
        return minLoss
        
    def sgd(self,dataT,adamP=True,lr=0.0001,momentum=0.9,wdecay=0,epochN=1000,dataV=None,criterion=nn.MSELoss(),fileName='/tmp/foo',schedP=False,newP=True,verbP=True):
          
        self.train(True)
        if(adamP):
            optimizer = optim.Adam(self.parameters(),lr = lr,weight_decay=wdecay)
        else:
            optimizer = optim.SGD(self.parameters() ,lr=lr,momentum=0.9,weight_decay=wdecay)
            
        if(schedP):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochN)
        else:
            scheduler = None
            
        
        minLoss=self.checkpoint(dataT,dataV,fileName=fileName)
            
        for epoch in range(epochN):
            # Loop over the whole dataset with randomization and take optimization steps. 
            self.computeLoss(dataT,optimizer=optimizer,randP=True)
            # Loop over the whole dataset without randomization to print statistics.
            if (0==((epoch+1)%100)):
                minLoss     = self.checkpoint(dataT,dataV,epoch=epoch,minLoss=minLoss,fileName=fileName)
            if(schedP):
                scheduler.step()
        
        if(not(0==((epoch+1)%100))):
            minLoss     = self.checkpoint(dataT,dataV,epoch=epoch,minLoss=minLoss,fileName=fileName)
            
        return minLoss
    
    def gtrain(self,data,lr=0.0001,fileName='/tmp/foo',nIt=100000,schedP=False,step=1.0,wdecay=0,verbP=True):
        
        setLogFile(fileName+'.log')
       
        loss=self.sgd(data,epochN=nIt,lr=lr,wdecay=wdecay,fileName=fileName,verbP=verbP,schedP=schedP)
            
        logf.info('Finished Training')
        return(loss)
