#sort
#Selection Sort
def performSelectionSort(lst):
    for i in range(len(lst)):
        for j in range(i,len(lst)):
            if lst[i] < lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst
lst = [2,5,0,3,3,3,1,5,4,2]

print(performSelectionSort(lst))

#Quick Sort
def quickSort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    small_lst , equal_lst, bigger_lst = [], [], []

    for i in range(len(lst)):
        if lst[i] < pivot:
            small_lst.append(lst[i])
        elif lst[i] > pivot:
            bigger_lst.append(lst[i])
        else: 
            equal_lst.append(lst[i])
    return quickSort(small_lst) + equal_lst + quickSort(bigger_lst)

print(quickSort(lst))

#Counting sort
def performCountingSort(lst):
    #max 구하기
    max = -9999
    min = 9999
    for i in range(len(lst)):
        if lst[i] > max:
            max = lst[i]
        if lst[i] < min:
            min = lst[i]
    counting = list(range(max - min + 1))
    for i in range(len(counting)):
        counting[i] = 0
    
    #빈도
    for i in range(len(lst)):
        value = lst[i]
        counting[value - min] += 1
    #printing
    cnt = 0
    for i in range(max - min +1):
        for j in range(counting[i]):
            lst[cnt] = i + min
            cnt = cnt + 1
    return lst 
lst = [2,5,0,3,3,3,1,5,4,2]

print(lst)
print("Counting Sort")
print(performCountingSort(lst))

def countingSort(lst, max):
    counting_array = [0] *(max+1)
    #빈도
    for i in lst:
        counting_array[i] += 1
    #업데이트
    for i in range(max):
        counting_array[i + 1] += counting_array[i]

    output_array = [-1] * len(lst)

    for i in lst:
        output_array[counting_array[i] -1 ] = i
        counting_array[i] -= 1

    return output_array
lst = [2,5,0,3,3,3,1,5,4,2]

print(lst)
print(countingSort(lst,5))

#Radix Sort 
from BinarySearchTree import BinarySearchTree
import math
def performRadixSort(lst):
    #find MAX, digit
    max = -99999
    for i in range(len(lst)):
        if lst[i] > max:
            max = lst[i]
    D = int(math.log10(max))

    for i in range(D+1):
        buckets = []
        #빈도
        for j in range(0,10):
            buckets.append([])
        for j in range(len(lst)):
            digit = int(lst[j] / math.pow(10,i)) % 10
            buckets[digit].append(lst[j])
        #printing
        cnt = 0
        for j in range(0, 10):
            for p in range(len(buckets[j])):
                lst[cnt] = buckets[j][p]
                cnt = cnt + 1
    return lst
    
lst = [802,95,10,3,13,3,11,503,4,2,0]
print("Radix Sort")
print(performRadixSort(lst))

print(int(math.log10(802)))
print(int(7 / math.pow(10,1)) % 10)
print(int(802 / math.pow(10,0)) % 10)
#Bubble Sort

#Merge Sort

#Heap Sort

#search 이진 탐색 정렬되어 있을 때
lst = [12,25,31,48,54,66,70,83,95,108]
def BinarySearch(lst, target):
    left = 0
    right = len(lst) - 1
    mid = int((left + right) / 2)
    temp = lst[mid]
    find = False
    while left <= right:
        if target == temp:
            find = True
            break
        elif target < temp:
            right = mid -1
        else:
            left = mid + 1
        mid = int((left + right)/2)
        temp = lst[mid]
    if find == True:
        mid += 1
        print("찾는 값은 ", mid, "번 째 있다")
    else:
        print("없다")
BinarySearch(lst, 83)

#Genetic Algorithm
"""
selection step
crossover 
mutation
substitution 
difficult problem - TSP 
"""



#Graph 
"""
matrix representation for graph 
adjacency list representation for graph 
traverse: DFS BFS traverse on graphs 
shortest path 
find a set of path to control whole vertexes
Dijkstra 's algorithm (DP, DFS, BFS) 
Prim's algorithm 
Apriori algorithm 
"""
#1 Graph matrix 구현
class Graph:
    def __init__(self, size):
        self.size = size 
        self.graph = [[0 for _ in range(size)] for _ in range(size)]

    def DFS_matrix(self, start):
    #stack, recursion
        #preorder
        current = start
        stack = []
        visitedAry = []

        stack.append(current)
        
        while (len(stack) != 0):
            current = stack.pop()
            visitedAry.append(current)
            for vertex in range(self.size -1,-1,-1):
                if self.graph[current][vertex] == 1 :
                    if vertex not in stack and vertex not in visitedAry:
                        stack.append(vertex)          
        return visitedAry
    
    def BFS_matrix(self, start):
        visited = [] 
        q = [start]

        while q:
            vis = q[0]
            visited.append(vis)
            q.pop(0)
            for i in range(self.size):
                if self.graph[vis][i] == 1 :
                    if i not in q and i not in visited:
                        q.append(i)
                    
        return visited

# to avoid falling into a cycle            
        
G1 = Graph(10)
G1.graph[5][3] = 1
G1.graph[3][2] = 1
G1.graph[2][0] = 1
G1.graph[0][1] = 1
G1.graph[1][3] = 1
G1.graph[5][4] = 1
G1.graph[5][7] = 1
G1.graph[4][6] = 1
G1.graph[7][6] = 1
G1.graph[7][9] = 1
G1.graph[9][8] = 1

print(G1.DFS_matrix(5))
print(G1.BFS_matrix(5))



#2 list Graph 구현
graph_list = {1: set([3, 4]),
              2: set([3, 4, 5]),
              3: set([1, 5]),
              4: set([1]),
              5: set([2, 6]),
              6: set([3, 5])}
root_node = 1

from collections import deque
def BFS_adj_list(graph, root):
    visited = []
    queue = deque([root])
    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            queue += graph[n] - set(visited)
    return visited
print(BFS_adj_list(graph_list, root_node))

def DFS_adj_list(graph, root):
    visited = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            stack += graph[n] - set(visited)
    return visited
print(DFS_adj_list(graph_list, root_node))

##양방향
# from collections import deque

# def DFS(graph, root):
#     visited = []
#     stack = [root]

#     while stack:
#         n = stack.pop()
#         if n not in visited:
#             visited.append(n)
#             if n in graph:
#                 temp = list(set(graph[n]) - set(visited))
#                 temp.sort(reverse=True)
#                 stack += temp
#     return " ".join(str(i) for i in visited)

# def BFS(graph, root):
#     visited = []
#     queue = deque([root])

#     while queue:
#         n = queue.popleft()
#         if n not in visited:
#             visited.append(n)
#             if n in graph:
#                 temp = list(set(graph[n]) - set(visited))
#                 temp.sort()
#                 queue += temp
#     return " ".join(str(i) for i in visited)

# graph = {}
# n = input().split(' ')
# node, edge, start = [int(i) for i in n]
# for i in range(edge):
#     edge_info = input().split(' ')
#     n1, n2 = [int(j) for j in edge_info]
#     if n1 not in graph:
#         graph[n1] = [n2]
#     elif n2 not in graph[n1]:
#         graph[n1].append(n2)

#     if n2 not in graph:
#         graph[n2] = [n1]
#     elif n1 not in graph[n2]:
#         graph[n2].append(n1)

# print(DFS(graph, start))
# print(BFS(graph, start))

#Shortest path
#Dijkstra's algorithm
class graphDijk:
    def __init__(self, size):
        self.size = size 
        self.graph = [[0 for _ in range(size)] for _ in range(size)]

    def dijkstra(self, start):
        dist = {} #Memoization Table :start 부터 거리
        for vertex in range(self.size):
            dist[vertex] = 99999
        dist[start] = 0

        V = [_ for _ in range(self.size)]
        
        while len(V) != 0:
            #getVertexWithMinDistance
            min = 999999
            for u in V:
                if dist[u] < min:
                        min = self.graph[start][u]
                        minU = u
            V.remove(minU)
            for i in range(self.size):
                if self.graph[minU][i] != 0:
                    #neighbors 중에 
                    if dist[i] > dist[minU] + self.graph[minU][i]:
                        dist[i] = dist[minU] + self.graph[minU][i]
        
        return dist

    def prim(self, src):
        V = set(_ for _ in range(self.size))
        U = set() # covered vertexes
        E = set() # covered edges
        U.add(src)
        while V != U:
            min = 99999
            for src in U:
                for des in V - U :
                    if self.graph[src][des] != 0 :
                        if self.graph[src][des] < min:
                            min = self.graph[src][des]
                            edge = (src , des)
                            destination = des
                    
            U.add(destination)
            E.add(edge)
        return U, E

#Time Complexity

print("Shortest path")
G2 = graphDijk(5)
G2.graph[0][1] = 3
G2.graph[0][4] = 20
G2.graph[1][2] = 5
G2.graph[1][3] = 6
G2.graph[3][4] = 4
G2.graph[2][4] = 7
G2.graph[1][4] = 15

print(G2.dijkstra(0))

#Minimum Spanning Tree problem
#Prim's Algorithm - Telephone, Electricity grid, TV cable, Computer Road network

G3 = graphDijk(5)
G3.graph[0][1] = 3
G3.graph[0][4] = 20
G3.graph[1][2] = 6
G3.graph[1][3] = 5
G3.graph[2][4] = 4
G3.graph[3][4] = 7
G3.graph[1][4] = 15

U, E = G3.prim(0)
print(U)
print(E)

#미로찾기 Maze
