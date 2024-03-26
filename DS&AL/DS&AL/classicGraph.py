
# import torch
# edge_index = torch.tensor([[0, 9, 9, 9, 9, 8, 1, 10, 1, 8, 8, 12, 12, 2, 2, 3, 3, 4, 3, 7, 0, 7, 0, 11, 7, 6, 6, 5],
#                            [9, 0, 10, 10, 8, 9, 10, 10, 8, 1,12, 8, 2, 12, 3, 2, 4, 3, 7, 3, 7, 0, 11, 0, 6, 7, 5, 6]], dtype=torch.long)

# n = 13 #number of nodes in the graph
# g = edge_index 
def dfs(i, at, visited ,ordering, graph):
    visited[at] = True
    print("visited", visited)
    edges = graph[at]
    print(ordering)
    if (edges is not None):
        for edge in edges:
            print("edge", edge)
            if visited[edge[1]] == False:
                i = dfs(i, edge[1], visited, ordering, graph)


    ordering[i] = at
    print("ordering",ordering)
    return i - 1



def topSort(graph, N):
    
    V = [False] * N #visited
   
    ordering = [0] * N 
    i = N - 1
    for at in range(N):
        if V[at] == False:
            i = dfs(i, at, V, ordering, graph)
    return ordering



N = 7
graph = dict();
for i in range(N):
    graph[i] = list();

graph[0].append([0, 1, 3])
graph[0].append([0, 2, 2])
graph[0].append([0, 5, 3])
graph[1].append([1, 3, 1])
graph[1].append([1, 2, 6])
graph[2].append([2, 3, 1])
graph[2].append([2, 4, 10])
graph[3].append([3, 4, 5])
graph[5].append([5, 4, 7])

#print(topSort(graph, N))


#Diakstra's Shortest 