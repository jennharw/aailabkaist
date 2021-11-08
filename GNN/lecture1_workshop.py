import torch

#Define a graph
edge_list = torch.tensor([
        [0,0,0,1,2,2,3,3], #source node
        [1,2,3,0,0,3,2,0]  #target node
            ], dtype = torch.long ) #undirected graph

node_features = torch.tensor([
        [-8,1,5,8,2,-3],
        [-1,0,2,-3,0,1],
        [1,-1,0,-1,2,1],
        [0,1,4,-2,3,4],
            ], dtype=torch.long)

# Weight for each edge

edge_weight = torch.tensor([
        [35.], #Weight for nodes (0, 1)
        [40.], #(0, 2)  
        [12.],
        [10.],
        [70.],
        [5.],
        [15.],
        [8.],
         ], dtype=torch.long)

from torch_geometric.data import Data

#Make a data object to store graph information
data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_weight)
print("Number of nodes: ", data.num_nodes)
print("Number of edges: ", data.num_edges)
print("Number of features per node", data.num_node_features)
print("Number of Weights per edge (edge-features)", data.num_edge_features) #1

#Plot the graph
from torch_geometric.utils.convert import to_networkx
import networkx as nx
G = to_networkx(data)
nx.draw_networkx(G)
