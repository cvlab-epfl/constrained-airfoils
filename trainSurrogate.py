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
drawAirfoil(dataT.target[0])
fName = netwDataName(zdim,n1,n2,n3)

#%%---------------------------------------------------------------------------
#                            Train/Load MLP Surrogate
#-----------------------------------------------------------------------------

loadP = False
predict_lod = True  # Predicts directly de lift over drag if true, otherwise predicts both drag and lift coefficients
uncertainty = False

if uncertainty:
    model = MLP_uncertainty(predict_lod).to(device)
else:
    model = MLP(predict_lod).to(device)

if loadP:
    model.restore()
else:
    model.gtrain(dataT, dataV)
    
wings = dataV.target
cdl = dataV.target_cdl
with torch.no_grad():

    for i in range(10):

        out = model(wings[i:i+1])
        if predict_lod:
            print("Lift over Drag prediction:", "{:.3f}".format(out.item()), "GT", "{:.3f}".format(cdl[i:i+1,1].item()/cdl[i:i+1,0].item()))
        else:
            print("Drag", "{:.3f}".format(out[:,0].item()), "GT", "{:.3f}".format(cdl[i:i+1,0].item()), "Lift", "{:.3f}".format(out[:,1].item()), "GT", "{:.3f}".format(cdl[i:i+1,1].item()), "Lift over Drag:", "{:.3f}".format(out[:,1].item()/out[:,0].item()), "GT", "{:.3f}".format(cdl[i:i+1,1].item()/cdl[i:i+1,0].item()))
        
        if uncertainty:
            out2 = model(wings[i:i+1], out)
            print("Uncertainty:", "{:.3f}".format(torch.norm(out2-out).item()))
    
    train_loss_lod = []
    loss_fn = nn.L1Loss() if predict_lod else nn.L1Loss(reduction='none')

    for bat in range(dataT.batchN):
        _, targetT, cdlT = dataV.batch(bat)
        y = model(targetT)
        if model.predict_lod:
            loss_lod = loss_fn(y[:,0], cdlT[:,1]/cdlT[:,0])
        else:
            loss_lod = loss_fn(y[:,1]/y[:,0], cdlT[:,1]/cdlT[:,0]).mean()
        train_loss_lod.append(loss_lod.item())
    print("Train Lift over Drag MAE", np.mean(train_loss_lod))
    
    test_loss_lod = []
    for bat in range(dataV.batchN):
        _, targetV, cdlV = dataV.batch(bat)
        y = model(targetV)
        if model.predict_lod:
            loss_lod = loss_fn(y[:,0], cdlV[:,1]/cdlV[:,0])
        else:
            loss_lod = loss_fn(y[:,1]/y[:,0], cdlV[:,1]/cdlV[:,0]).mean()
        test_loss_lod.append(loss_lod.item())
    print("Test Lift over Drag MAE", np.mean(test_loss_lod))

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
        out = model(xy1)
        if not predict_lod:
            lod = out[:,1]/out[:,0]
        else:
            lod = out

        if e==0:
            print("Initial Lift over drag:", lod.item())
        print(100*e/max_epochs, "%", end='\r')
        loss = loss_l1(1/lod, torch.zeros_like(lod, device=device))
        loss_z = loss_l1(z, z_init)
        loss += 1.0*loss_z
        if uncertainty:
            out2 = model(xy1, out)
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
        print("Uncertainty:", "{:.3f}".format(uncert.item()))

    fig = plt.plot()
    xy1 = net(z_init).view((-1,2)).to(device)
    airfoils = xy1.reshape(z.shape[0], xy1.shape[0]//z.shape[0], 2)
    drawAirfoil(airfoils[0])
    xy1 = net(z).view((-1,2)).to(device)
    airfoils = xy1.reshape(z.shape[0], xy1.shape[0]//z.shape[0], 2)
    drawAirfoil(airfoils[0],color='-r')
    plt.show()
