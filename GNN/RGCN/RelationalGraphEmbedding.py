#https://towardsdatascience.com/graph-neural-networks-for-multi-relational-data-27968a2ed143

### One-hot vector representation of nodes (5,5):
X = np.eye(5, 5)
n = X.shape[0]
np.random.shuffle(X)
print(X)
----- 
[[0. 0. 1. 0. 0.]  # Node 1 
 [0. 1. 0. 0. 0.]  # Node 2
 [0. 0. 0. 0. 1.]  ...
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]] # Node 5
### Weight matrix (5,3)
# Dimension of the hidden features
h = 3 
# Random initialization with Glorot and Bengio
W = np.random.uniform(-np.sqrt(1./h), np.sqrt(1./h),(n,h))
print(W)
-----
[[-0.4294049   0.57624235 -0.3047382 ]
 [-0.11941829 -0.12942953  0.19600584]
 [ 0.5029172   0.3998854  -0.21561317]
 [ 0.02834577 -0.06529497 -0.31225734]
 [ 0.03973776  0.47800217 -0.04941563]]
### Adjacency matrix of an undirect Graph (5,5)
A = np.random.randint(2, size=(n, n))
# Include the self loop
np.fill_diagonal(A, 1)
# Symmetric adjacency matrix (undirected graph)
A_und = (A + A.T)
A_und[A_und > 1] = 1
print(A_und)
-----
[[1 1 1 0 1] # Connections to Node 1
 [1 1 1 1 1]
 [1 1 1 1 0]
 [0 1 1 1 1]
 [1 1 0 1 1]]

 ### Linear transformation
L_0 = X.dot(W)
print(L_0)
-----
[[ 0.5029172   0.3998854  -0.21561317]  # Node 1 (3rd row of W)
 [-0.11941829 -0.12942953  0.19600584]  # Node 2 (2nd row of W)
 [ 0.03973776  0.47800217 -0.04941563]  # Node 3 (5th row of W)
 [-0.4294049   0.57624235 -0.3047382 ]
 [ 0.02834577 -0.06529497 -0.31225734]] # Node 5 (4th row of W)
### GNN - Neighborhood diffusion
ND_GNN = A_und.dot(L_0)
print(ND_GNN)
-----
[[ 0.45158244  0.68316307 -0.3812803 ] # Updated Node 1
 [ 0.02217754  1.25940542 -0.6860185 ]
 [-0.00616823  1.3247004  -0.37376116]
 [-0.48073966  0.85952002 -0.47040533]
 [-0.01756022  0.78140325 -0.63660287]]

### Test on the aggregation
assert(ND_GNN[0,0] == L_0[0,0] + L_0[1,0] + L_0[2,0] + L_0[4,0])

### Degree vector (degree for each node)
D = A_und.sum(axis=1)
print(D)
-----
[4 5 4 4 4] # Degree of Node 1
### Reciprocal of the degree (diagonal matrix)
D_rec = np.diag(np.reciprocal(D.astype(np.float32))) 
print(D_rec)
-----
[[0.25 0.   0.   0.   0.  ] # Reciprocal value of Node 1 degree
 [0.   0.2  0.   0.   0.  ]
 [0.   0.   0.25 0.   0.  ]
 [0.   0.   0.   0.25 0.  ]
 [0.   0.   0.   0.   0.25]]
### GCN - Isotropic average computation
ND_GCN = D_rec.dot(ND_GNN)
print(ND_GCN)
-----
[[ 0.11289561  0.17079077 -0.09532007] # Updated Node 1 (with deg)
 [ 0.00443551  0.25188109 -0.1372037 ]
 [-0.00154206  0.3311751  -0.09344029]
 [-0.12018491  0.21488001 -0.11760133]
 [-0.00439005  0.19535081 -0.15915072]]
### Test on the isotropic average computation:
assert(ND_GCN[0,0] == ND_GNN[0,0] * D_rec[0,0])

### Recall: One-hot vector representation of nodes (n,n)
print(X)
-----
[[0. 0. 1. 0. 0.]  # Node 1
 [0. 1. 0. 0. 0.]  ...
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]]
### Number of relation types (r)
num_rels = 2
print(num_rels)
-----
2
### Weight matrix of relation number 1 (n,n)
## Initialization according to Glorot and Bengio (2010))
W_rel1 = np.random.uniform(-np.sqrt(1./h),np.sqrt(1./h),(n,h))
print(W_rel1)
-----
[[-0.46378913 -0.09109707  0.52872529]
 [ 0.03829597  0.22156061 -0.2130242 ]
 [ 0.21535272  0.38639244 -0.55623279]
 [ 0.28884178  0.56448816  0.28655701]
 [-0.25352144  0.334031   -0.45815514]]
### Weight matrix of relation number 2 (n,h)
## Random initialization with uniform distribution
W_rel2 = np.random.uniform(1/100, 0.5, (n,h))
print(W_rel2)
-----
[[0.22946783 0.4552118  0.15387093]
 [0.15100992 0.073714   0.01948981]
 [0.34262941 0.11369778 0.14011786]
 [0.25087085 0.03614765 0.29131763]
 [0.081897   0.29875971 0.3528816 ]]
### Tensor including both weight matrices (r,n,h)
W_rels = np.concatenate((W_rel1, W_rel2))
W_rels = np.reshape(W_rels,(num_rels, n, h))
print(W_rels)
-----
[[[-0.46378913 -0.09109707  0.52872529] 
  [ 0.03829597  0.22156061 -0.2130242 ]
  [ 0.21535272  0.38639244 -0.55623279]
  [ 0.28884178  0.56448816  0.28655701]
  [-0.25352144  0.334031   -0.45815514]]

 [[ 0.22946783  0.4552118   0.15387093]
  [ 0.15100992  0.073714    0.01948981]
  [ 0.34262941  0.11369778  0.14011786]
  [ 0.25087085  0.03614765  0.29131763]
  [ 0.081897    0.29875971  0.3528816 ]]]
### Linear trasformationwith batch matrix multiplication (r,n,h)
L_0_rels = np.matmul(X, W_rels)
print(L_0_rels)
-----
[[[ 0.21535272  0.38639244 -0.55623279] # Node 1 (3rd row of W_rel1)
  [ 0.03829597  0.22156061 -0.2130242 ]
  [-0.25352144  0.334031   -0.45815514]
  [-0.46378913 -0.09109707  0.52872529]
  [ 0.28884178  0.56448816  0.28655701]]

 [[ 0.34262941  0.11369778  0.14011786] # Node 1 (3rd row of W_rel2)
  [ 0.15100992  0.073714    0.01948981]
  [ 0.081897    0.29875971  0.3528816 ]
  [ 0.22946783  0.4552118   0.15387093]
  [ 0.25087085  0.03614765  0.29131763]]]
### Adjacency matrix of relation number 1 (n,n)
A_rel1 = np.random.randint(2, size=(n, n))
np.fill_diagonal(A, 0)  # No self_loop
print(A_rel1)
-----
[[0 1 1 1 1] # Connections to Node 1 with Rel 1
 [1 1 0 0 1] # Connections to Node 2 with Rel 1
 [1 0 0 1 0]
 [0 0 1 1 1]
 [1 1 0 1 0]]
### Adjacency matrix of relation number 2 (n,n)
A_rel2 = np.random.randint(3,size=(n,n))
np.fill_diagonal(A_rel2, 0)  # No self loop
A_rel2[A_rel2>1] = 0
-----
[[0 0 0 1 0] # Connections to Node 1 with Rel 2
 [1 0 0 0 0] # Connections to Node 2 with Rel 2
 [1 0 0 1 1]
 [0 0 0 0 0]
 [0 1 0 0 0]]
### Tensor including both adjacency matrices (r,n,n)
A_rels = np.concatenate((A_rel1, A_rel2))
A_rels = np.reshape(A_rels, (num_rels, n, n)) 
print(A_rels)
-----
[[[0 1 1 1 1] # Connections to Node 1 with Rel 1
  [1 1 0 0 1]
  [1 0 0 1 0]
  [0 0 1 1 1]
  [1 1 0 1 0]]

 [[0 0 0 1 0] # Connections to Node 2 with Rel 2
  [1 0 0 0 0]
  [1 0 0 1 1]
  [0 0 0 0 0]
  [0 1 0 0 0]]]
### (GCN) Neighborhood diffusion for each typed edge (r,n,h)
ND_GCN = np.matmul(A_rels, L_0_rels)
print(ND_GCN)
-----
[[[-0.39017282  1.0289827   0.14410296] # Updated Node 1 with Rel 1
  [ 0.54249047  1.17244121 -0.48269997]
  [-0.24843641  0.29529538 -0.0275075 ]
  [-0.42846879  0.80742209  0.35712716]
  [-0.21014043  0.51685598 -0.2405317 ]]

 [[ 0.22946783  0.4552118   0.15387093] # Updated Node 1 with Rel 2
  [ 0.34262941  0.11369778  0.14011786]
  [ 0.82296809  0.60505722  0.58530642]
  [ 0.          0.          0.        ]
  [ 0.15100992  0.073714    0.01948981]]]
### (R-GCN) Aggregation of GCN (n,h)
RGCN = np.sum(ND_GCN, axis=0)
print(RGCN)
-----
[[-0.16070499  1.48419449  0.29797389] Updated Node 1(Rel 1 + Rel 2)
 [ 0.88511988  1.28613899 -0.34258211]
 [ 0.57453168  0.9003526   0.55779892]
 [-0.42846879  0.80742209  0.35712716]
 [-0.05913052  0.59056998 -0.22104189]]

### Test of the aggregation
assert(RGCN[0,0] == L_0_rels[0,1,0] + L_0_rels[0,2,0] + L_0_rels[0,3,0] + L_0_rels[0,4,0] + L_0_rels[1,3,0])+