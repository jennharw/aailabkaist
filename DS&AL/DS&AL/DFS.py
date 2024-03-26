#DFS we can augment the DFS algorithm to:
## Compute a graph's minimum spanning ree
## Detect and find cycles in a graph
## Check if a graph is bipartite
## Find strongly connected components
## Topologically sort the node of a graph
## Find bridges and articulation points
## Find augmenting paths in a flow network
## Generate mazes

# Connected components - sometimes a graph is split into multiple components. It's useful to be able to identify and count these components

#Graph DFS, Stacks or Recursion, in graphs only pre-order traverse is used  
#Tree is a directed acyclic graph, graph my not be a DAG, you have to check the repeated visits to avoid falling into a cycle
#BFS  queue, 

#Depth First Search 
import torch
from torch_geometric.data import Data

x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float) #feature
edge_index = torch.tensor([[0, 1, 0, 9, 8, 7, 1, 8, 10, 7, 10, 11, 7, 11, 6, 7, 3, 7, 3, 5, 5, 6, 3, 4, 2, 3],
                           [1, 0, 9, 0, 7, 8, 8, 1, 7, 10, 11, 10, 11, 7, 7, 6, 7, 3, 5, 3, 6, 5, 4, 3, 3, 2]], dtype=torch.long)

data = Data(edge_index=edge_index)
print(data)
from torch_geometric.datasets import Planetoid
#dataset = Planetoid(root='', name='Cora')
# #data = dataset[0]

n = 13 #number of nodes in the graph
g = edge_index #adjacency list representing graph
visited = [False] * n # 

print(edge_index[0]==0)
print(edge_index[1][edge_index[0]==0])

ret = []
def dfs(node = ''): #dfs(0)
    if node =='':
        node = 0
    
    visited[node] = True
    ret.append(node)
    
    neighbors = edge_index[1][edge_index[0]==node]
    for next in neighbors:
        print("next: ", next)
        if visited[next] == False:
            print("next dfs")
            dfs(next)        
    return ret, visited

start_node = 0
#ret, visited = dfs(start_node)
#print(ret, visited)

#Connected components
edge_index = torch.tensor([[0, 4, 8, 4, 8, 0, 0, 14, 0, 13, 13, 14, 7, 6, 6, 11, 11, 7, 3, 9, 9, 15, 15, 2, 15, 10, 1,5, 5, 16, 5, 17],
                           [4, 0, 4, 8, 0, 8, 14, 0, 13, 0, 14, 13, 6, 7, 11, 6, 7, 11, 9, 3, 15, 9, 2, 15, 10, 15, 5, 1, 16, 5, 17, 5]], dtype=torch.long)

n = 18
g = edge_index

components = [0] * n
visited = [False] * n #n size

def findComponents():
    count = 0
    for i in range(n):
        if visited[i] == False:
            count += 1
            dfs(i, count)
    return (count, components)

def dfs(at, count):
    visited[at] = True
    components[at] = count
    #adj for next in g.iloc[at]:
    neighbors = edge_index[1][edge_index[0]==at]
    for next in neighbors:
        if visited[next] == False:
            dfs(next, count) 

count, components = findComponents()
print(count, components)