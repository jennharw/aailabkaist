#https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
#Data, torch_geometric.data
#the adjacency of each node (edge_index)
#the features associated with each node
import torch
from torch_geometric.data import Data

x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float) #feature
y = torch.tensor([0, 1, 0, 1], dtype=torch.float) #label

edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)
print(data)


#DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='', name='Cora')
data = dataset[0]
print(data)
loader = DataLoader([data], batch_size = 32, shuffle = True)
for batch in loader:
    print(batch)
#DataLoader를 정의할 때, train_mask 를 사용해서 구현


#Message Passing - how node embeddings are learned
#x denotes the node embeddings, e denotes the edge featrues, message function, aggregation function, update function 
#propagation (edge_index), Calling this function will consequently call message and update.
#message for each node pair(x_i, x_j)
#update (aggr_out) aggregated message, assigning a new embedding value for each node

#SageConv layer
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class SAGEConv(MessagePassing): 
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr = 'max') #Max aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, out_channels, bias = False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        #X, [N, in_channels], edge_index [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        return self.propagate(edge_index, size=(x.size(0), x.size(0)),x=x)
    
    def message(self, x_j):
        print("x_j: " , x_j.shape)
        #x_j [E, out_channels], 10556, 1433
        x_j = self.lin(x_j) #(10556,1433) (in_channels 1433, out_channels) = (E, out_channels)
        x_j = self.act(x_j) #(E, out_channnels)
        print("x_j: " , x_j.shape)
        return x_j
    
    def update(self, aggr_out, x):
        #aggr_out N, out_channels
        print("aggr_out: ", aggr_out.shape)
        print("x: ", x.shape)
        new_embedding = torch.cat([aggr_out, x], dim = 1)
        print("new_embedding: ", new_embedding.shape)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding) #N, in_channels
        return new_embedding

conv1 = SAGEConv(1433, 50)
print(data.x.size(0)) #2708
print(conv1(data.x, data.edge_index).shape)


#Real-World exmpale, RecSysChallenge2015
#Predict whether there will be a buy event followed by a sequence of clicks
#Predict which item will be bought

from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(128, 128)
        self.pool1 = TopKPooling(128, ratio = 0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio = 0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio = 0.8)
        
        self.item_embedding = torch.nn.Embedding(num_embeddings=1433, embedding_dim=128)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x

