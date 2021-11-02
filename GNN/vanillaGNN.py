# GNN Structure
# input -> GNN layer (hidden) -> Multilayer Perceptron(MLP)  Prediction layer. employing the encoded graph representation, obtained as output from GNN layer
#
#
# input -> onehot matrix
import numpy as np
import scipy.sparse as sp

X = np.eye(5, 5)
n = X.shape[0]

# to assign initial features to each of these nodes, the input layer applies a linear transformation (projection) - encode
emb = 3
W = np.random.uniform(-np.sqrt(1./emb), np.sqrt(1./emb), (n, emb))
print(W)
L_0 = X.dot(W)
print(L_0)

# the projection step assigns a d-dimensional vector representation to each node in the graph. 5 length one hot vectors representing the nodes are projected into 3-length dense feature vectors
# the goal of the input layer is to embed the input features of nodes (and edges) to a d-dimensional vector of hidden features.
# This new representation is obtained via a simple linear transformation

#Matrix
A = np.random.randint(2, size = (n, n))
print(A)
np.fill_diagonal(A, 1) #self loop
print(A)
A = (A + A.T)
A[A > 1] = 1
print(A)

L_1 = A.dot(L_0)
print(L_1)

print(L_0[0, :] + L_0[1, :] + L_0[2, :] + L_0[4, :])
print(L_1[1, :])

#features of node1, obtained summing the features of local neighbors
# loss 

#GCN
#graph classification
# N * N  adjacency matrix (number of node)
# N * D feature matrix (number of features per node)
# N by E binary label matrix(E is the number of classes) - classification

#data
import pickle as pkl
import sys
import networkx as nx

names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []

for i in range(len(names)):
    with open("ind.citeseer.{}".format(names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

test_idx_reorder = parse_index_file("ind.citeseer.test.index")
test_idx_range = np.sort(test_idx_reorder)


# Fix citeseer dataset (there are some isolated nodes in the graph)
# Find isolated nodes, add them as zero-vecs into the right position
test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
tx_extended[test_idx_range-min(test_idx_range), :] = tx
tx = tx_extended
ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
ty_extended[test_idx_range-min(test_idx_range), :] = ty
ty = ty_extended

#node features
features = sp.vstack((allx, tx)).tolil()
features[test_idx_reorder, :] = features[test_idx_range, :]
print("-----features shape-----")
print(features.shape)

#adjacency matrix
adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#labels
labels = np.vstack((ally, ty))
labels[test_idx_reorder, :] = labels[test_idx_range, :]
print("-----labels shape-----")
print(len(labels))
print(labels.shape)
print(len(labels[10, :]))

idx_test = test_idx_range.tolist()
idx_train = range(len(y))
idx_val = range(len(y), len(y)+500)

import torch

labels = torch.LongTensor(np.where(labels)[1])
print(labels)
idx_train = torch.LongTensor(idx_train)
idx_test = torch.LongTensor(idx_test)
idx_val = torch.LongTensor(idx_val)
features =torch.FloatTensor(np.array(features.todense()))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

adj = sparse_mx_to_torch_sparse_tensor(adj)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

train_mask = sample_mask(idx_train, labels.shape[0]) #3327
y_train = np.zeros(labels.shape) #3327, 6
y_train[train_mask, :] = labels[train_mask, :]

val_mask = sample_mask(idx_val, labels.shape[0])
test_mask = sample_mask(idx_test, labels.shape[0])
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)
y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]

tf.data.Dataset.from_tensor_slices((features, ))

print("-----features shape-----")
print(features.shape)
print("-----labels shape-----")
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)
x_train = x_train/255
x_valid = x_valid/255
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(3703,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(features, y_train, epochs = 5, verbose= 1, validation_data=(features, y_val))

#pytorch - GCN
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

print(torch.cuda.is_available())
model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            dropout=0.5)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)

model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not False:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(10):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()