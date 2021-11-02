#package 말고 코드로 - tensorflow
#https://antonsruberts.github.io/graph/graphsage/

import pandas as pd
import json
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Model, optimizers, metrics, losses
from tensorflow.keras.layers import Dense, Dropout

edges = pd.read_csv('git_edges.csv')
edges.columns = ['source', 'target'] # renaming for StellarGraph compatibility

with open('git.json') as json_data:
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
targets = pd.read_csv('git_target.csv')
targets.index = targets.id.astype(str)
targets = targets.loc[features.keys(), :]

labels_sampled = targets['ml_target'].sample(frac=0.8, replace=False, random_state=101)

# G = sg.StellarGraph(node_features, edges.astype(str))
# G_sampled = G.subgraph(labels_sampled.index)

# print('# nodes in full graph:', len(G.nodes()))
# print('# nodes in sampled graph:', len(G_sampled.nodes()))

from sklearn import model_selection
train_labels, test_labels = model_selection.train_test_split(
    labels_sampled,
    train_size=0.05,
    random_state=42,
)

# 20% of test for validation
val_labels, test_labels = model_selection.train_test_split(
    test_labels, train_size=0.2, random_state=42,
)

def depth_sampler(n, n_sizes):
    node_lists = []
    
    # First layer
    input1 = np.array(node_features.iloc[n]).reshape(-1, 4005)
    # get all neighbours
    neighbours = edges[edges['source'] == n]['target']
  
    # randomly choose neighbours
    if len(neighbours) > 0:
        neighbours_chosen = np.random.choice(neighbours, size=n_sizes[0])
    else:
        neighbours_chosen = np.full(n_sizes[0], -1)
    input2 = np.array(node_features.iloc[neighbours_chosen])
    
    node_lists.append(list(neighbours_chosen))
    
    # Second Layer
    second_layer_list = []
    for node in neighbours_chosen:
        # get all neighbours
        if node != -1:
            neighbours = edges[edges['source'] == node]['target']
        else:
            neighbours = []
        # randomly choose neighbours
        if len(neighbours) > 0:
            neighbours_chosen = list(np.random.choice(neighbours, size=n_sizes[1]))
        else:
            neighbours_chosen = list(np.full(n_sizes[1], -1))
        
        second_layer_list += neighbours_chosen

    input3 = np.array(node_features.iloc[second_layer_list])
    
    node_lists.append(second_layer_list)
    
    return [input1, input2, input3]

class graphSSSS(tf.keras.Model):
  def __init__(self):
    super(graphSSSS, self).__init__()

    self.dense1 = Dense(16,activation='relu') #(1, 16)
    self.dropout1 = Dropout(.2)

    self.neih = Dense(32,  activation='relu') #(10, 32)

    #aggre

    self.output2 = Dense(16, activation='relu') #(1, 16)
    self.dropout2 = Dropout(.2)
    #concat

    #2hop
    self.dense3 = Dense(16, activation='relu')  #(10, 1433) -> (2, 16)
    self.dropout3 = Dropout(.2)
    self.dense4 = Dense(32,input_shape=(-1, 10,4005) ,activation='relu' )         #neinei(10, 100, 1433) -> (10, 100, 32)
    self.dropout4 = Dropout(.2)
    self.dense5 = Dense(16, activation='relu') #2,16

    #
    self.dense6 = Dense(16, activation='relu') #
    self.dense7 = Dense(32, activation='relu')
    self.dropout5= Dropout(.2)
    self.dense8 = Dense(16,activation='relu')
    #
    self.flatten =  tf.keras.layers.Flatten()
    self.outputgra = Dense(1, activation = 'sigmoid') #node subjects

     


  def __call__(self, x ):
    #print("x")
    tar, nei, nienie = x #(1, 4005) (10, 4005) (100, 4005)

    output1 = self.dense1(tar) #(1, 4005) -> (1, 16)
    output1 = self.dropout1(output1)
    #print('output1 shape:', output1.shape)

    neih = self.neih(nei) #(10, 32)
    neih = self.dropout2(neih)
    #print('nieh shape:', neih.shape)


    agg = np.expand_dims(np.mean(neih, axis = 0), axis=0) #(1,32)
    #print('agg shape:', agg.shape)

    output2 = self.output2(agg) #(1, 16)
    #print('output2 shape:', output2.shape)

    #concat
    layer1 = tf.concat([output1, output2], axis = 1) #(1, 16)
    #print('layer1 shape:', layer1.shape) #1, 32

    #hop2
    output3 = self.dense3(nei)
    output3 = self.dropout3(output3)
    #print('output3 shape:', output3.shape) #(10, 4005) -> (10, 16)

    nienie = nienie.reshape(-1, 10, 4005)
    interoutput = self.dense4(nienie)
    interoutput = self.dropout4(interoutput)
    #print('interoutput shape:', interoutput.shape) #(10, 10, 4005) -> (10, 10, 32)

    agg2 = np.mean(interoutput, axis = 1) #
    #print('agg2 shape:', agg2.shape) #(10, 32)

    output4 = self.dense5(agg2)
    #print('output4 shape:', output4.shape) #(10, 16)

    layer2 = tf.concat([output3, output4], axis = 1) 
    #print('layer2 shape:', layer2.shape) #(10, 32)

    #layer1 layer2
    hop1 = self.dense6(layer1) #(1, 32) -> (1,16)
    #print('hop1 shape:', hop1.shape)

    hop2= self.dense7(layer2) #(10, 32)
    #print('hop2 shape:', hop2.shape)

    hop2 = np.expand_dims(np.mean(hop2, axis=0), axis=0) #(1, 32)
    hop2 = self.dropout5(hop2)
    #print('hop2 shape:', hop2.shape)

    hop2 = self.dense8(hop2) #(1, 16)
    #print('hop2 shape:', hop2.shape)

    l2_output =tf.concat([hop1, hop2], axis=1) # (1,16) + (1,16) = ( 1, 32) 
    #print('(Final) output shape:', l2_output.shape)
    return self.outputgra(l2_output) #0, 1

model = graphSSSS()
inputs = depth_sampler(14541, [10, 10])
for i in range(len(inputs)):
  inputs[i]= inputs[i].astype("float32")
tar, nei, nienie = inputs
model(inputs)

model = graphSSSS()

loss_object = tf.keras.losses.binary_crossentropy #SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=0.005)

#train_loss = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')
train_accuracy =tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', name ="train_accuracy")

#@tf.function
def train_step(model, inputs, y, loss_object, optimizers, train_accuracy):
       
        with tf.GradientTape() as tape:
          predictions = model(inputs)
          loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables) #,unconnected_gradients=tf.UnconnectedGradients.ZERO)

        optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, model.trainable_variables)
            if grad is not None)
        #train_loss(loss)
        train_accuracy(y, predictions)

for epoch in range(2):
    for x, y in zip(train_labels.index, train_labels):
       inputs = depth_sampler(int(x), [10, 10])
       y = np.array(y).reshape(-1, 1)
       train_step(model, inputs, y, loss_object, optimizer, train_accuracy)
    print(f'Epoch {epoch + 1}, Accuracy:{train_accuracy.result()*100}')