#https://github.com/JinheonBaek/RGCN
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_add
import torch.nn.functional as F
from model import RGCN

def load_data(file_path):
    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entities.dict')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    
    with open(os.path.join(file_path, 'relations.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets


def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
    
    return np.array(triplets)

#load data
best_mrr = 0

entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('FB15k-237')
all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

def sample_edge_uniform(n_triples, sample_size):
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace = False)

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    edges = sample_edge_uniform(len(triplets), sample_size)
    #select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeld_edges = np.stack((src, rel, dst)).transpose()

    samples, labels = negative_sampling(relabeld_edges, len(uniq_entity), negative_rate)

    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size), size = split_size, replace=False)
    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    #create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)
    return data

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

train_data = generate_sampled_graph_and_labels(train_triplets, 30000, 0.5, len(entity2id), len(relation2id), 1)
print("TRAIN DATA : ", train_data)


model = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.2)
print(model)

import torch.nn as nn
entity_embedding = nn.Embedding(len(entity2id), 100)
x = entity_embedding(train_data.entity)
relation_embedding = nn.Parameter(torch.Tensor(len(relation2id), 100))

# self.conv1 = RGCNConv(
#             100, 100, num_relations * 2, num_bases=num_bases)
# conv1(x, edge_index, edge_type, edge_norm)

att = nn.Parameter(torch.Tensor(474, 4))
basis = nn.Parameter(torch.Tensor(4, 100, 100))
w = torch.matmul(att, basis.view(4, -1))
print(w.shape)

w = w.view(474, 100, 100)
w = torch.index_select(w, 0, train_data.edge_type)
print(w.shape)
print(train_data.edge_index.unsqueeze(1))

# torch.bmm(train_data.edge_index.unsqueeze(1), w).squeeze(-2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.2)

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)
    return loss

from tqdm import tqdm, trange
for epoch in trange(1,10000, desc = 'Epochs', position = 0):

        model.train()
        optimizer.zero_grad()

        loss = train(train_triplets, model, batch_size=30000, split_size=0.5, 
            negative_sample=1, reg_ratio = 1e-3, num_entities=len(entity2id), num_relations=len(relation2id))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 500 == 0:

            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))