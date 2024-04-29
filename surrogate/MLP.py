import torch
from torch import nn
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, predict_lod=True):
        super(MLP, self).__init__()
        self.predict_lod = predict_lod
        self.model_filename = "surrogate/mlp_state_dict_lod.pth" if self.predict_lod else "surrogate/mlp_state_dict_cdl.pth"
        out_dim=1 if self.predict_lod else 2
        self.layers = nn.Sequential(
                nn.Linear(54, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, out_dim)
                )

    def forward(self, x):
        out = self.layers(x)
        return out
    
    def gtrain(self, dataT, dataV, model_filename=None, device='cuda'):
                
        if model_filename is not None:
            self.model_filename = model_filename
        
        weights = 1/dataT.target_cdl.std(0)

        lr = 1e-4
        NEPOCHS = 5000
        
        loss_fn = nn.L1Loss() if self.predict_lod else nn.L1Loss(reduction='none')
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=int(dataT.batchN*0.8), epochs=NEPOCHS)
        best_score = np.inf

        for epoch in range(NEPOCHS):
            self.train()
            running_loss = []
            for bat in range(dataT.batchN):
                inputT, targetT, cdlT = dataT.batch(bat)
                targetT += (0.00001**0.5)*torch.randn(targetT.shape).to(device)
                y = self(targetT)
                if self.predict_lod:
                    loss = loss_fn(y[:,0], cdlT[:,1]/cdlT[:,0])
                else:
                    loss = loss_fn(y, cdlT) * weights
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                
            scheduler.step()
            
            if (epoch+1)%100==0:        
                self.eval()
                test_loss = []
                test_loss_lod = []
                with torch.no_grad():
                    for bat in range(dataV.batchN):
                        inputV, targetV, cdlV = dataV.batch(bat)
                        y = self(targetV)
                        if self.predict_lod:
                            loss = loss_fn(y[:,0], cdlV[:,1]/cdlV[:,0])
                            loss_lod = loss
                        else:
                            loss = loss_fn(y, cdlV) * weights
                            loss = loss.mean()
                            loss_lod = loss_fn(y[:,1]/y[:,0], cdlV[:,1]/cdlV[:,0]).mean()
                        test_loss.append(loss.item())
                        test_loss_lod.append(loss_lod.item())
                    print(epoch, "Train Mean Absolute Error", np.mean(running_loss), "Test MAE:", np.mean(test_loss), "Lift over Drag MAE", np.mean(test_loss_lod))

                if np.mean(test_loss)<best_score:
                    best_score = np.mean(test_loss)
                    best_state_dict = self.state_dict()
                    torch.save(best_state_dict, f"{self.model_filename}")
        print("Best score:", best_score)
                    
    def restore(self, model_filename=None):
        
        if model_filename is not None:
            self.model_filename = model_filename

        self.load_state_dict(torch.load(f"{self.model_filename}"))
            