import torch
from torch._C import Node, device
from torch_geometric import datasets
from torch_geometric.datasets import Planetoid #citation network "Cora", "CiteSeer", "PubMed"
from torch_geometric.nn import Node2Vec

dataset = Planetoid("/", "Cora")
data = dataset[0]
print("Cora", data)

#Construct model
N2V_model =  Node2Vec(data.edge_index, embedding_dim=128, walk_length=20, 
context_size=10, walks_per_node=10, 
num_negative_samples=1, p=1, q=1, sparse=True)#.to(device) device = 'cuda'

loader =  N2V_model.loader(batch_size = 128, shuffle = True) # , num_workers = 4)

#train function
optimizer = torch.optim.SparseAdam(list(N2V_model.parameters()), lr=0.01)
def train():
    N2V_model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = N2V_model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

#plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

@torch.no_grad() # Deactivate autograde funtionality
def plot_point(colors):
    N2V_model.eval() #Evaluate the model based on the trained parameters
    z = N2V_model(torch.arange(data.num_nodes)) #Embedding , device=device
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()
    plt.figure()

    for i in range(dataset.num_classes):
        plt.scatter(z[y==i, 0], z[y==i, 1], s = 20, color = colors[i])
    plt.axis('off')
    plt.show()
colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
plot_point(colors)

#classification  - test logisticRegression
def test():
    N2V_model.eval()
    z = N2V_model() #
    acc = N2V_model.test(z[data.train_mask], data.y[data.train_mask], z[data.test_mask], data.y[data.test_mask], max_iter = 150)
    return acc

print(f'test Accuracy: {test()}')

