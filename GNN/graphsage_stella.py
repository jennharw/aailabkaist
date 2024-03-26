import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML

import matplotlib.pyplot as plt
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

dataset = 'Cora'
path = osp.join('data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
data = data
edges = pd.DataFrame(data.edge_index.numpy().T)
edges.columns = ['source', 'target']
node_features = pd.DataFrame(data.x.numpy())
targets = pd.DataFrame(data.y.numpy()).squeeze(1)

G = sg.StellarGraph(node_features, edges)

train_subjects, test_subjects =  model_selection.train_test_split(
    targets, train_size=0.1, test_size=None, stratify= targets
)
#one hot
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)
#one hot
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)
"""
edges_path = 'git_edges.csv'
targets_path = 'git_target.csv'
features_path = 'git.json'

# Read in edges
edges = pd.read_csv(edges_path)
edges.columns = ['source', 'target'] # renaming for StellarGraph compatibility

with open(features_path) as json_data:
    features = json.load(json_data)
    
max_feature = np.max([v for v_list in features.values() for v in v_list])
features_matrix = np.zeros(shape = (len(list(features.keys())), max_feature+1))

i = 0
for k, vs in tqdm(features.items()):
    for v in vs:
        features_matrix[i, v] = 1
    i+=1
    
node_features = pd.DataFrame(features_matrix, index = features.keys())

# Read in targets
targets = pd.read_csv(targets_path)
targets.index = targets.id.astype(str)
targets = targets.loc[features.keys(), :]

# Put the nodes, edges, and features into stellargraph structure
G = sg.StellarGraph(node_features, edges.astype(str))

labels_sampled = targets['ml_target'].sample(frac=0.8, replace=False, random_state=101)
G_sampled = G.subgraph(labels_sampled.index)

print('# nodes in full graph:', len(G.nodes()))
print('# nodes in sampled graph:', len(G_sampled.nodes()))
# 5% train nodes
train_labels, test_labels = model_selection.train_test_split(
    labels_sampled,
    train_size=0.05,
    random_state=42,
)

# 20% of test for validation
val_labels, test_labels = model_selection.train_test_split(
    test_labels, train_size=0.2, random_state=42,
)
"""

# number of nodes per batch
# number of nodes per batch
batch_size = 50

# number of neighbours per layer
num_samples = [10, 5]

# generator
#generator = GraphSAGENodeGenerator(G_sampled, batch_size, num_samples)
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

# Generators for all the data sets
# train_gen = generator.flow(train_labels.index, train_labels, shuffle=True)
# val_gen = generator.flow(val_labels.index, val_labels)
# test_gen = generator.flow(test_labels.index, test_labels)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)

import tensorflow as tf
from tensorflow.keras import Model, optimizers, metrics, losses

# GraphSage tellargraph model
graphsage_model = GraphSAGE(
    layer_sizes=[32, 32], 
    generator=generator,
    #aggregator=MeanPoolingAggregator,
    bias=True, 
    dropout=0.5,
)

# get input and output tensors
x_inp, x_out = graphsage_model.in_out_tensors()
# pass the output tensor through the classification layer
prediction = layers.Dense(7, activation="softmax")(x_out) #1 sigmoid 

# build and compile
from tensorflow.keras import Model

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy, #binary
    metrics=[metrics.AUC(num_thresholds=200, curve='ROC'), 'acc'],
)
model.summary()

test_gen = generator.flow(test_subjects.index, test_targets)

history = model.fit(
    train_gen, 
    epochs=2, 
    validation_data=test_gen, 
    verbose=2, 
    shuffle=False)