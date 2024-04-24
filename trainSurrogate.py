# %%
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../')
from data import airfrans_to_deepsdf
import numpy as np
from airfoildata import loadAirfoilData, loadWingProfiles
from auxfuncs    import drawAirfoil,netwDataName,shoelaceArea1
from netw.miscfuncs import *
from projarea   import  AreaProjector
from decoder    import PerceptronDecoder
from MLP import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%---------------------------------------------------------------------------
#                            Load Data
#-----------------------------------------------------------------------------

zdim=8
step=25
n1 = 16
n2 = 32
n3 = 16
targetA= 0.1

dataT = loadAirfoilData(zdim=zdim,batchN=100,trainP=True,step=step,targetA=targetA)
dataV = loadAirfoilData(zdim=zdim,batchN=1,trainP=False,step=step,targetA=targetA)

ydim  = dataT.target.size(1)
drawAirfoil(dataT.target[1])
fName = netwDataName(zdim,n1,n2,n3)

#%%---------------------------------------------------------------------------
#                            Train Autodecoder Alone
#-----------------------------------------------------------------------------
loadP = True
net  = PerceptronDecoder(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim,reluP=True)
net.toGpu()
if(loadP):
    # Restore from file
    net.restore(fName)
    dataT.restore(fName)
    dataT.setids(randP=False)
else:
    # Train network
    net.gtrain(dataT,fileName=fName, nIt=10000)

#%%---------------------------------------------------------------------------
#                            Train MLP Surrogate Alone
#-----------------------------------------------------------------------------
# %%

loadP = True
model = MLP().to(device)

if(loadP):
    model.load_state_dict(torch.load("mlp_state_dict.pth"))
else:
    lr = 1e-4
    NEPOCHS = 5000

    loss_fn = nn.L1Loss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=int(dataT.batchN*0.8), epochs=NEPOCHS)
    best_score = np.inf

    for epoch in range(NEPOCHS):
        model.train()
        running_loss = []
        for bat in range(np.int32(dataT.batchN*0.9)):
            inputT, targetT, cdlT = dataT.batch(bat)
            targetV += (0.00001**0.5)*torch.randn(targetV.shape).to(device)
            y = model(targetT)
            loss = loss_fn(y, cdlT)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            
        scheduler.step()
        
        if (epoch+1)%100==0:        
            model.eval()
            test_loss = []
            with torch.no_grad():
                for bat in range(np.int32(dataT.batchN*0.9),dataT.batchN):
                    inputV, targetV, cdlV = dataT.batch(bat)
                    y = model(targetV)
                    loss = loss_fn(y, cdlV)
                    test_loss.append(loss.item())

                print(epoch, np.mean(running_loss), np.mean(test_loss))
                
            if np.mean(test_loss)<best_score:
                best_score = np.mean(test_loss)
                best_state_dict = model.state_dict()
                
    torch.save(best_state_dict, "mlp_state_dict.pth")

#%%---------------------------------------------------------------------------
#                            Test MLP Grad Smoothness
#-----------------------------------------------------------------------------
zs,xys0,cdl = dataT.batch(0)
z1 = zs.clone().detach().to(device)[0]
zt = makeTensor(z1).view((1,-1)).to(device)
print(z1.shape, zt.shape)
j  = 7
xj = z1[j]
nx = 100

xs = np.zeros(nx)
ys = np.zeros(nx)

for i,dx in enumerate(np.linspace(-0.05,0.05,100)):
    
    zt[0,j] =  xs[i]   = xj + dx
    xy      = net(zt)
    cd, cl  = model(xy)[0]
    lod     = cl / cd
    # lod     = cd
    
    ys[i]   = lod.item()
    
    if(0 == (i%10)):
        drawAirfoil(xy[0],'-b')
        plt.pause(0.2)

plt.plot(xs,ys)    

#%%---------------------------------------------------------------------------
#                                Run Optimization
#-----------------------------------------------------------------------------
z = zs.clone().detach().to(device).requires_grad_(True)
loss_l1 = torch.nn.L1Loss()
optimizer = torch.optim.Adam([z], lr=0.001)
z_init = zs.clone().detach().to(device)

max_epochs = 1000
for e in range(max_epochs):
    xy1 = net(z[0]).to(device)
    cd, cl = model(xy1)
    # cd, cl = cdl[:,0], cdl[:,1]
    if e==0:
        print("Initial cd:", cd.item(), "cl:", cl.item(), "Lift over drag:", (cl/cd).item())
    print(100*e/max_epochs, "%", end='\r')
    lod = cd / cl
    loss = loss_l1(lod, torch.tensor([0], device=device))
    loss_z = loss_l1(z, z_init)
    loss += 1.0*loss_z
    loss.backward()
    # grad = z.grad
    # print(grad)
    optimizer.step()
    optimizer.zero_grad()
    
print("Final cd:", cd.item(), "cl:", cl.item(), "Lift over drag:", (cl/cd).item())

xy1 = net(z_init).view((-1,2)).to(device)
airfoils = xy1.reshape(z.shape[0], xy1.shape[0]//z.shape[0], 2)
drawAirfoil(airfoils[0])
xy1 = net(z).view((-1,2)).to(device)
airfoils = xy1.reshape(z.shape[0], xy1.shape[0]//z.shape[0], 2)
drawAirfoil(airfoils[0],color='-r')

#%%---------------------------------------------------------------------------
#                            Test MLP Grads vs Finite-Diff.
#-----------------------------------------------------------------------------
# 
# dbgP=False
    
def stiffnessF(K,xy1):
    
    if((2==len(xy1.size())) and (2==xy1.size(1))):
        xy2 = xy1
    else:
        xy2 = xy1[0].view((-1,2))
    
    xs  = xy2[:,0]
    ys  = xy2[:,1]
    
    return 0.5 * (xs.T @ K @ xs +  ys.T @ K @ ys)

def objF (z):
    global model
    
    z  = makeTensor(z).to(device)
    return wingLodF(net,z,K=None,model=model).item()

def objG (z):
    global model
    
    z  = makeTensor(z).to(device)
    return fromTensor(wingLodG(net,z,K=None,model=model))

def wingLodF(net,z,model=None,K=None):
        
    #print(z)
    
    z1  = z.view((1,-1))
    xy1 = net(z1)
    
    if(dbgP):
        lod = torch.sum(xy1.view((-1,2))[:,0]*xy1.view((-1,2))[:,1])
    else:
        cdl = model(xy1)
        cd, cl = cdl[:,0], cdl[:,1]
        lod    = cd
        
    #print(cd.size(),cl.size())
    
    obj = - lod
    if(K is not None):
        obj = obj + stiffnessF(K,xy1)
        
    return obj

def wingLodG(net,z,model=None,K=None):
    
    z   = z.clone().detach().requires_grad_(True)
    obj = wingLodF(net,z,model=model,K=K) 
    obj.backward()
    grad = z.grad
    
    return grad

def tstG(objF,objG,xs,eps=1e-8):
    
    xs  = np.asarray(xs,dtype=np.float64)
    g   = objG(xs)

    for i,xi in enumerate(xs):
        xi = xs[i]
        xs[i] = xi + eps
        fp    = objF(xs) 
        xs[i] = xi - eps
        fm    = objF(xs) 
        xs[i] = xi
        df    = (fp-fm)/(2.0*eps)
        print(i,':',g[i],df,'->',g[i]/df)

z0 = fromTensor(zs[0])
objG(z0)

# xs  = np.asarray(z0,dtype=np.float64)
# g   = objF(xs) 
# print(g)

tstG(objF,objG,z0,1e-4)