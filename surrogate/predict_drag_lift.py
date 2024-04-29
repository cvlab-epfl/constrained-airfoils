import torch
import numpy as np
import torch_geometric

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