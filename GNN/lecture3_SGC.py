import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv
import torch.nn.functional as F
dataset = Planetoid('', "Cora")
data = dataset[0]
print("Cora: ", data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#torch_geometric , examples
#Embedding
SGC_model = SGConv(in_channels=data.num_features,
            out_channels=dataset.num_classes,
            K = 1, cached = True)

print("Shape of the original data: ", data.x.shape)
print("Shape of the embedding data: ", SGC_model(data.x, data.edge_index).shape)

#Construct the model for classification
class SGCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(in_channels=data.num_features,
        out_channels=dataset.num_classes,
        K=1, cached = True)
    def forward(self):
        x = self.conv1(data.x, data.edge_index)

        return F.log_softmax(x, dim=1)
    

model, data = SGCNet().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= 0.2, weight_decay=0.005)

for i, parameter in SGC_model.named_parameters():
    print("Parameter {}".format(i))
    print("Shape: ", parameter.shape)

#TRAIN
def train():
    model.train()
    optimizer.zero_grad()
    predicted_y = model()

    true_y = data.y
    losses = F.nll_loss(predicted_y[data.train_mask], true_y[data.train_mask])
    losses.backward()
    optimizer.step()

def test():
    model.eval() #Set the model.training to be False
    logits, accs = model(), [] #Log prob 
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item()  / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc= test_acc = 0
for epoch in range(1,101):
    train()
    train_acc, val_acc, temp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = temp_test_acc
    print(f'Epoch : {epoch}, Train: {train_acc}, Val: {best_val_acc}, Test:{test_acc}')