import torch
import torch.nn as nn
import torch_geometric.nn as nng
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity
from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool
import torch.optim as optim
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
    
class MLP(torch.nn.Module):
    r"""A multi-layer perception (MLP) model.

    Args:
        channel_list (List[int]): List of input, intermediate and output
            channels. :obj:`len(channel_list) - 1` denotes the number of layers
            of the MLP.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """
    def __init__(self, channel_list, dropout = 0.,
                 batch_norm = True, relu_first = False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(self.channel_list[:-1], self.channel_list[1:]):
            self.lins.append(Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(self.channel_list[1:-1]):
            self.norms.append(BatchNorm1d(dim, track_running_stats = True) if batch_norm else Identity())     

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = lin.forward(x)
        return x


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'

class GraphSAGE(nn.Module):
    def __init__(self,
                 predict_lod=True,
                 nb_hidden_layers=2, 
                 size_hidden_layers=32, 
                 bn_bool=True,
                 encoder_hidden_size=32, 
                 encoder_hidden_layers=1, 
                 decoder_hidden_size=32, 
                 decoder_hidden_layers=1, 
                 latent_size=4, 
                 encoder_input_size=4, 
                 ):
        super().__init__()
        
        self.predict_lod = predict_lod
        self.model_filename = "surrogate/gnn_state_dict_lod.pth" if self.predict_lod else "surrogate/gnn_state_dict_cdl.pth"
        out_dim = 1 if self.predict_lod else 2
        self.nb_hidden_layers = nb_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.bn_bool = bn_bool
        self.activation = nn.LeakyReLU()

        encoder_params = [encoder_input_size] + [encoder_hidden_size for _ in range(encoder_hidden_layers)] + [latent_size]
        decoder_params = [latent_size] + [decoder_hidden_size for _ in range(decoder_hidden_layers)] + [out_dim]

        self.encoder = MLP(encoder_params, batch_norm = False)
        self.decoder = MLP(decoder_params, batch_norm = False)

        self.in_layer = nng.SAGEConv(
            in_channels = encoder_params[-1],
            out_channels = self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels = self.size_hidden_layers,
                out_channels = self.size_hidden_layers
            ))

        
        self.out_layer = nng.SAGEConv(
                in_channels = self.size_hidden_layers,
                out_channels = decoder_params[0]
            )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats = True))

    def infer(self, data):
        z, edge_index = data.x , data.edge_index.type(torch.int64)
        z = self.encoder(z)
        
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)

        z = self.out_layer(z, edge_index)

        z = self.decoder(z) #.unsqueeze(-1)
        
        z = global_mean_pool(z, data.batch)

        return z
    
    def forward(self, data):
        dat = Batch.from_data_list([airfrans_preprocess(dt.view(dt.shape[0]//2, 2)) for dt in data])
        return self.infer(dat)
    
    def gtrain(self, dataT, dataV, model_filename=None, device='cuda'):
                
        if model_filename is not None:
            self.model_filename = model_filename
        
        weights = 1/dataT.target_cdl.std(0)

        data_list = []
        for i in range(dataT.target.shape[0]):
            data = airfrans_preprocess(dataT.target[i].view(dataT.target[i].shape[0]//2, 2))
            data.y = dataT.target_cdl[i].unsqueeze(0)
            data_list.append(data)
        train_loader = DataLoader(data_list, batch_size=32)
        data_list = []
        for i in range(dataV.target.shape[0]):
            data = airfrans_preprocess(dataV.target[i].view(dataV.target[i].shape[0]//2, 2))
            data.y = dataV.target_cdl[i].unsqueeze(0)
            data_list.append(data)
        test_loader = DataLoader(data_list, batch_size=32)
    
        lr = 1e-4
        NEPOCHS = 5000
        
        loss_fn = nn.L1Loss() if self.predict_lod else nn.L1Loss(reduction='none')
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=int(dataT.batchN*0.8), epochs=NEPOCHS)
        best_score = np.inf

        for epoch in range(NEPOCHS):
            self.train()
            running_loss = []
            for data in train_loader:
                data.x += (0.00001**0.5)*torch.randn(data.x.shape).to(device)
                y = self.infer(data)
                if self.predict_lod:
                    loss = loss_fn(y[:,0], data.y[:,1]/data.y[:,0])
                else:
                    loss = loss_fn(y, data.y) * weights
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
                    for data in test_loader:
                        y = self.infer(data)
                        if self.predict_lod:
                            loss = loss_fn(y[:,0], data.y[:,1]/data.y[:,0])
                            loss_lod = loss
                        else:
                            loss = loss_fn(y, data.y) * weights
                            loss = loss.mean()
                            loss_lod = loss_fn(y[:,1]/y[:,0], data.y[:,1]/data.y[:,0]).mean()
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
        
def calculate_normal(points):
    normals = []
    idxs = []
    num_points = len(points)
    for i in range(num_points):
        prev_point = points[(i-1) % num_points]
        current_point = points[i]
        next_point = points[(i+1) % num_points]
        
        # Calculate vectors from the current point to its neighbors
        vec_prev = prev_point - current_point
        vec_next = next_point - current_point
        
        # Calculate the normal vector
        normal = torch.tensor([-vec_prev[1], vec_prev[0]])  # Perpendicular vector to vec_prev
        normal /= torch.norm(normal)  # Normalize the vector
        
        # Check the direction of the normal vector
        cross_product = vec_prev[0]*vec_next[1] - vec_prev[1]*vec_next[0]
        if cross_product < 0:
            normal = -normal  # Flip the normal if it's pointing inwards
        
        if not torch.isnan(normal).any():
            normals.append(normal.unsqueeze(0))
            idxs.append(i)
    
    return torch.cat(normals), idxs

def airfrans_preprocess(points, device="cuda"):
     
    pos = points.type(torch.float).to(device)
    normal, idxs = calculate_normal(pos)
    pos = pos[idxs]

    normal = normal.to(device)

    # Put everything in tensor
    x = torch.concatenate([pos, normal], dim=-1)

    data_object = torch_geometric.data.Data()
    data_object.x = x
    data_object.edge_index = torch_geometric.nn.radius_graph(x=pos, r=10, loop=False, max_num_neighbors=64).cpu()
    data_object.edge_index = torch_geometric.utils.coalesce(data_object.edge_index.sort(dim=0).values)
    
    data_object.to(device)
    
    return data_object