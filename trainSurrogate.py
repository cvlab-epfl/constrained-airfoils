# %%
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from airfoildata import loadAirfoilData, loadWingProfiles
from auxfuncs    import drawAirfoil,netwDataName,shoelaceArea1
from netw.miscfuncs import *
from projarea   import  AreaProjector
from decoder    import PerceptronDecoder
from surrogate.MLP import MLP
from surrogate.GraphSage import GraphSAGE
from surrogate.MLP_uncertainty import MLP_uncertainty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%---------------------------------------------------------------------------
#                            Load C_dl Data
#-----------------------------------------------------------------------------

zdim=8
step=25
n1 = 16
n2 = 32
n3 = 16
targetA= 0.1

dataT = loadAirfoilData(zdim=zdim,batchN=100,trainP=True,cdl=True)
dataV = loadAirfoilData(zdim=zdim,batchN=11,trainP=False,cdl=True)

ydim  = dataT.target.size(1)
drawAirfoil(dataT.target[1])
fName = netwDataName(zdim,n1,n2,n3)

#%%---------------------------------------------------------------------------
#                            Train/Load MLP/GNN Surrogate
#-----------------------------------------------------------------------------

loadP = True
predict_lod = True  # Predicts directly de lift over drag if true, otherwise predicts both drag and lift coefficients
model = 'mlp' # Surrogate's architecture, should be "mlp" or "gnn"
uncertainty = True

if uncertainty:
    if model == "mlp":
        model = MLP_uncertainty(predict_lod).to(device)
    else:
        raise Exception("Sorry, Uncertainty not available for the GNN")
else:
    model = MLP(predict_lod).to(device) if model == "mlp" else GraphSAGE(predict_lod).to(device)

if loadP:
    model.restore()
else:
    model.gtrain(dataT, dataV)
    
wings = dataV.target
cdl = dataV.target_cdl
for i in range(10):
    out = model(wings[i:i+1])
    if predict_lod:
        print("Lift over Drag prediction:", "{:.3f}".format(out.item()), "GT", "{:.3f}".format(cdl[i:i+1,1].item()/cdl[i:i+1,0].item()))
    else:
        print("Drag", "{:.3f}".format(out[:,0].item()), "GT", "{:.3f}".format(cdl[i:i+1,0].item()), "Lift", "{:.3f}".format(out[:,1].item()), "GT", "{:.3f}".format(cdl[i:i+1,1].item()), "Lift over Drag:", "{:.3f}".format(out[:,1].item()/out[:,0].item()), "GT", "{:.3f}".format(cdl[i:i+1,1].item()/cdl[i:i+1,0].item()))
    
    if uncertainty:
        out2 = model(wings[i:i+1], out)
        print("Uncertainty:", "{:.3f}".format(torch.norm(out2-out).item()))

#%%---------------------------------------------------------------------------
#                            Test on wings.npy
#-----------------------------------------------------------------------------
wings = np.load("dat/wings.npy")
wings = makeTensor(wings.reshape((-1,54)))
for i in range(10):
    if predict_lod:
        print("Lift over Drag:", "{:.3f}".format(model(wings[i:i+1]).item()))
    else:
        out = model(wings[i:i+1])
        print("Drag", "{:.3f}".format(out[:,0].item()), "Lift", "{:.3f}".format(out[:,1].item()), "Lift over Drag:", "{:.3f}".format(out[:,1].item()/out[:,0].item()))

    if uncertainty:
        out2 = model(wings[i:i+1], out)
        print("Uncertainty:", "{:.3f}".format(torch.norm(out2-out).item()))
#%%---------------------------------------------------------------------------
#                            Load AreaProjector and Shape Data
#-----------------------------------------------------------------------------
loadP = True
net  = AreaProjector(n1=n1,n2=n2,n3=n3,nIn=zdim,nOut=ydim)
net.toGpu()
# Restore from file
net.restore(fName)

dataT = loadAirfoilData(zdim=zdim,batchN=100,trainP=True,step=step,targetA=targetA,cdl=False)
dataV = loadAirfoilData(zdim=zdim,batchN=11,trainP=False,step=step,targetA=targetA,cdl=False)
dataT.restore(fName)
dataT.setids(randP=False)

#%%---------------------------------------------------------------------------
#                            Test MLP Grad Smoothness
#-----------------------------------------------------------------------------
zs,xys0,cdl = dataT.batch(0)
z1 = zs.clone().detach().to(device)[0]
zt = makeTensor(z1).view((1,-1)).to(device)
j  = 7
xj = z1[j]
nx = 100

xs = np.zeros(nx)
ys = np.zeros(nx)

for i,dx in enumerate(np.linspace(-0.05,0.05,100)):
    
    zt[0,j] =  xs[i]   = xj + dx
    xy      = net(zt)
    lod  = model(xy)
    if not predict_lod:
        lod = lod[:,1]/lod[:,0]
    # lod     = cl / cd
    # lod     = cd
    ys[i]   = lod.item()
    
    if(0 == (i%10)):
        drawAirfoil(xy[0],'-b')
        plt.pause(0.2)

plt.plot(xs,ys)    


#%%---------------------------------------------------------------------------
#                                Run Optimization
#-----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
for i in range(10):
    zs,xys0,cdl = dataT.batch(i)
    z = zs.clone().detach().to(device).requires_grad_(True)
    loss_l1 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([z], lr=0.001)
    z_init = zs.clone().detach().to(device)
    max_epochs = 1000

    for e in range(max_epochs):
        xy1 = net(z[:1]).to(device)
        lod = model(xy1)
        if not predict_lod:
            lod = lod[:,1]/lod[:,0]
        # cd, cl = cdl[:,0], cdl[:,1]
        if e==0:
            print("Initial Lift over drag:", lod.item())
        print(100*e/max_epochs, "%", end='\r')
        loss = loss_l1(1/lod, torch.zeros_like(lod, device=device))
        loss_z = loss_l1(z, z_init)
        loss += 1.0*loss_z
        if uncertainty:
            out2 = model(xy1, lod)
            if not predict_lod:
                out2 = out2[:,1]/out2[:,0]
            uncert = torch.norm(out2-lod)
            loss_uncert = loss_l1(uncert, torch.zeros_like(lod, device=device))
            loss += 0.001*loss_uncert
        loss.backward()
        # grad = z.grad
        # print(grad)
        optimizer.step()
        optimizer.zero_grad()
        
    print("Final Lift over drag:", lod.item())
    if uncertainty:
        out2 = model(xy1, lod)
        print("Uncertainty:", "{:.3f}".format(torch.norm(out2-lod).item()))

    fig = plt.plot()
    xy1 = net(z_init).view((-1,2)).to(device)
    airfoils = xy1.reshape(z.shape[0], xy1.shape[0]//z.shape[0], 2)
    drawAirfoil(airfoils[0])
    xy1 = net(z).view((-1,2)).to(device)
    airfoils = xy1.reshape(z.shape[0], xy1.shape[0]//z.shape[0], 2)
    drawAirfoil(airfoils[0],color='-r')
    plt.show()

#%%---------------------------------------------------------------------------
#                            Test MLP Grads vs Finite-Diff.
#-----------------------------------------------------------------------------
# 
dbgP=False
    
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
        lod = model(xy1)
        
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
# %%
