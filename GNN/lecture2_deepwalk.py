import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.construct import random
G = nx.karate_club_graph()
print('Number of nodes: ', len(G.nodes))

nx.draw_networkx(G)

labels = []
for i in G.nodes:
    club_names = G.nodes[i]['club']
    labels.append(1 if club_names == 'Officer' else 0) 

layout_pos = nx.spring_layout(G)
nx.draw_networkx(G,pos = layout_pos ,node_color = labels, cmap = 'coolwarm')

#Node Embedding using Deepwalk
from karateclub import DeepWalk

Deepwalk_model = DeepWalk(walk_number = 10, walk_length = 80, dimensions = 124)
Deepwalk_model.fit(G)
embedding = Deepwalk_model.get_embedding()
print('Embedding' , embedding.shape)

#low dimensional plot
import sklearn

PCA_model = sklearn.decomposition.PCA(n_components = 2)
low_dimension_embedding = PCA_model.fit_transform(embedding)
print('Low dimensional embedding representation', low_dimension_embedding.shape)
plt.scatter(low_dimension_embedding[:,0], low_dimension_embedding[:, 1], c = labels, s = 15, cmap='coolwarm')

#node classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

x_train, x_test, y_train, y_test = train_test_split(embedding, labels, test_size=0.5)
ML_model = LogisticRegression(random_state=0).fit(x_train, y_train)
y_predict = ML_model.predict(x_test)
ML_acc = roc_auc_score(y_test, y_predict)
print("AUC", ML_acc)

##Node2Vec
from karateclub import Node2Vec
#embdding with node2vec
N2V_model = Node2Vec(walk_number = 10, walk_length = 80, p=0.6, q = 0.4, dimensions = 124 )
N2V_model.fit(G)
N2V_embedding = N2V_model.get_embedding()

###https://antonsruberts.github.io/graph/deepwalk/
import numpy as np
print(G.neighbors)
def random_walk(start_node, walk_length):
    walk = [start_node]

    for i in range(walk_length):
        all_neighbours = [n for n in G.neighbors(start_node)]
        next_node = np.random.choice(all_neighbours, 1)[0] #randomly pick 1 neighour
        walk.append(next_node)
        start_node = next_node
    return walk
print(random_walk(6, 10))


def biased_walk(start_node, walk_length, p, q):
    walk = [start_node]
    previous_node = None
    previous_node_neighbours = []
    for _ in range(walk_length-1):
        current_node = walk[-1]
        current_node_neighbors = np.array(list(G.neighbors(current_node))) #neighbors of this node
        probability = np.array([1/q] * len(current_node_neighbors), dtype = float)  #probability by q , very local
        
        probability[current_node_neighbors==previous_node] = 1/p # probability by p
        probability[(np.isin(current_node_neighbors, previous_node_neighbours))] = 1 #weight 1

        norm_probability = probability / sum(probability)
        selected = np.random.choice(current_node_neighbors, 1, p = norm_probability)[0]
        walk.append(selected)
        previous_node_neighbours = current_node_neighbors
        previous_node = current_node
    return walk

print(biased_walk(6, 10, 1, 0.1)) # go outwards
print(biased_walk(6, 10, 1, 10)) # stay local 

#p controls the probability to go back to <t> after visiting <v>, return back to previous node #BFS low value of p 
#q controls the probability to go explore undiscovered parts of the graphs ,  (bfs homophily q = 0.5  , dfs q = 2, structure equivalence )
from karateclub.utils.walker import BiasedRandomWalker
b_walker = BiasedRandomWalker(10, 10, 1, 10)
b_walker = BiasedRandomWalker(10, 10, 1, 0.1)
b_walker = BiasedRandomWalker(20, 10, 0.6, 0.4)
b_walker.do_walks(G)

from gensim.models.word2vec import Word2Vec
node_vec = Word2Vec(b_walker.walks,
                    hs = 1,
                    vector_size = 128,
                    window = 10,
                    min_count = 1
                    )
X_node_vec = []
n_nodes = G.number_of_nodes()
X_node_vec = [node_vec.wv[str(n)] for n in range(n_nodes)]

# X_train, X_test, y_train, y_test = train_test_split(X_node_vec, y, test_size=0.2) # train/test split

# # Train RF
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)

# print(f1_score(y_test, y_pred, average='micro'))
# print(confusion_matrix(y_test, y_pred, normalize='true'))





#word2vec - skip gram
##OneHot Encoding doens't have similarity -> Embedding is dense vector with similarity
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results
corpus = remove_stop_words(corpus)

words = []
for text in corpus:
    for word in text.split():
        words.append(word)
words = set(words)
vocabulary_size = len(words)

word2int = {}
for i, word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence  in corpus:
    sentences.append(sentence.split())

WINDOW_SIZE = 2
data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx+WINDOW_SIZE, len(sentence))]:
            if neighbor != word:
                data.append([word, neighbor])

import pandas as pd

df = pd.DataFrame(data, columns = ['input', 'label'])

import numpy as np

ONE_HOT_DIM = len(words)

# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] # input word
Y = [] # target word

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))

# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)

embedding_dims = 5
import torch
from torch.autograd import Variable
import torch.nn.functional as F
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)

num_epochs = 100
learning_rate = .001


window_size = 2
idx_pairs = []
# for each sentence
for sentence in sentences:
    indices = [word2int[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs)

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1, -1), y_true)
        loss_val += loss.data.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

vector = pd.DataFrame(W1.T.cpu().detach().numpy().tolist(), columns = ['x1', 'x2', 'x3', 'x4', 'x5'])
print(vector)