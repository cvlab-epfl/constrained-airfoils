#%%
import torch
import torch.optim as optim

from netw.gradnet    import GradNet
from netw.perceptron import Perceptron
from netw.miscfuncs  import logf
#%%        
class AutoDecoder (GradNet):
    
    def sgd(self,dataT,lr=0.0001,momentum=0.9,wdecay=0,epochN=1000,dataV=None,fileName='/tmp/foo',schedP=False,randP=True,verbP=True):
          
        self.train(True)
        optimizerW = optim.Adam(self.parameters(),lr = lr,weight_decay=wdecay)
        optimizerL = optim.Adam(dataT.inputs.parameters(),lr = lr,weight_decay=wdecay)
        optimizers = [optimizerW,optimizerL]
            
        if(schedP):
            schedulerW = optim.lr_scheduler.CosineAnnealingLR(optimizerW,epochN)
            schedulerL = optim.lr_scheduler.CosineAnnealingLR(optimizerL,epochN)
            
        
        minLoss=self.checkpoint(dataT,dataV,fileName=fileName)
            
        for epoch in range(epochN):
            # Loop over the whole dataset with randomization and take optimization steps. 
            self.computeLoss(dataT,optimizers=optimizers,randP=randP)
            # Loop over the whole dataset without randomization to print statistics.
            if (0==((epoch+1)%100)):
                minLoss     = self.checkpoint(dataT,dataV,epoch=epoch,minLoss=minLoss,fileName=fileName)
            if(schedP):
                schedulerW.step()
                schedulerL.step()
        
        if(not(0==((epoch+1)%100))):
            minLoss     = self.checkpoint(dataT,dataV,epoch=epoch,minLoss=minLoss,fileName=fileName)
            
        return minLoss
    
    def computeLoss(self,data,optimizers=None,randP=False):
            
            loss = 0.0
            batc = 0
                    
            # Randomize the order.
            if(randP):
                data.setids(True)
                
            for batc in range(data.batchN):
                latentB,targetB=data.batch(batc)
                outputsB  = self(latentB)
                lossB     = self.lossFunction(outputsB,targetB)          
                loss     += lossB
                
                if(self.dbgP):
                    assert not torch.isnan(outputsB).any(), "computeLoss: output contains nan"
                    assert not torch.isnan(targetB).any(),  "computeLoss: target contains nan"
                    
                if(optimizers is not None):
                    for optimizer in optimizers:
                         optimizer.zero_grad()
                    lossB.backward()
                    for optimizer in optimizers:
                        optimizer.step()
            
            loss = loss/data.batchN
            return(loss)

class PerceptronDecoder(AutoDecoder,Perceptron):
    
    def checkpoint(self,dataT,dataV,epoch=-1,minLoss=None,fileName=None):
        
        with torch.no_grad():
            if(dataV is None):
                currentLoss = self.computeLoss(dataT,optimizers=None,randP=False)               
            else:
                currentLoss = self.computeLoss(dataV,optimizers=None,randP=False)
        if(minLoss is None):
            minLoss = currentLoss

        logf.info('[%4d] current loss: %.3e min loss: %.3e',epoch+1,currentLoss,minLoss)
            
        if(currentLoss<minLoss):
            minLoss=currentLoss
            if(fileName is not None):
                self.save(fileName,epoch+1,minLoss) 
                dataT.save(fileName)
        return minLoss

if __name__ == "__main__":
    from util import floatTensor
    from airfoildata import LatentData
    ns    = 10
    xdim  = 3
    ydim  = 7
    net   = PerceptronDecoder(n1=10,n2=10,nIn=xdim,nOut=ydim)
    net.toGpu()
    xs    = floatTensor((ns,xdim))
    ys    = torch.rand ((ns,ydim),dtype=torch.float32,device=xs.device)
    torch.nn.init.xavier_uniform_(xs)
    data  = LatentData(xs,ys,batchN=1)
    #data.shuffle()
    #data.batch(0)
    net.gtrain(data,lr=0.001)



        

    