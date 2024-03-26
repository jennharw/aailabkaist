#Breadth Fisrt Search O(V+E)
#shortest path on unweighted graph
#queue
import torch
edge_index = torch.tensor([[0, 9, 9, 9, 9, 8, 1, 10, 1, 8, 8, 12, 12, 2, 2, 3, 3, 4, 3, 7, 0, 7, 0, 11, 7, 6, 6, 5],
                           [9, 0, 10, 10, 8, 9, 10, 10, 8, 1,12, 8, 2, 12, 3, 2, 4, 3, 7, 3, 7, 0, 11, 0, 6, 7, 5, 6]], dtype=torch.long)

n = 13 #number of nodes in the graph
g = edge_index  #adjacency list representing unweighted graph

import collections
def bfs(s, e): #s = start node, e = end node
    #BFS
    brt, prev = solve(s)
    
    return reconstructPath(s, e, prev)


visited = [False] * n
prev = [None] * n

def solve(s):
    q = collections.deque()
    q.append(s)
    visited[s] = True
    brt = []
    brt.append(s)

    while q:
        node = q.popleft()
        neighbors = g[1][g[0]==node]
        for next in neighbors:
            if visited[next] == False:
                brt.append(next)
                q.append(next)
                visited[next] = True
                prev[next] = node
    return brt, prev
print(prev)
print(solve(0))

def reconstructPath(s, e, prev):
    path = []
    #Reconstruct path going backwards from e 
    at = e
    while at != None:
        path.append(at)
        at = prev[at]
    path.reverse()
    if path[0] == s:
        return path
    return []

print(bfs(0,10))

# q.append("s")
# print(q.popleft())
# if q:
#     print(q)

#graph on grids
#Dungeon Problem
#Direction vectors
#위, 아래, 좌, 우, 대각선 (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)
dr = [-1, 1, 0, 0]
dc = [0, 0, 1, -1]
sr = 0; sc = 0; R = 5; C = 7
#Empty Row queue, Empty Column Queue
rq, cq = collections.deque(), collections.deque()
#q = collections.deque()

import numpy as np
m = np.zeros(shape=(5,7))
m[0,3]= 1
m[1,1]= 1
m[1,5]= 1
m[2,1]= 1
m[3,2]= 1
m[3,3]= 1
m[4,0]= 1
m[4,2]= 1
m[4,5]= 1
m[4,3] = 5 #E
if m[0,3] == 1:
    print(m)

reached_end = False
visited = np.zeros(shape=(5,7), dtype=np.int16)

#shortest 
def solve():
    move_count = 0

    rq.append(sr)
    cq.append(sc)
    visited[sr][sc] = True

    while rq:
        r = rq.popleft()
        c = cq.popleft()
        if m[r][c] == 5:
            print(m[r][c], r, c)
            reached_end = True
            break
        explore_neighbours(r, c)
        move_count += 1        
    if reached_end:
        return move_count
    return -1
        
def explore_neighbours(r, c):
    for i in range(4):
        rr = r + dr[i]
        cc = c + dc[i]
        #Skip invalid cells, Assume R and C for the numboer of rows and columns
        if rr < 0 or cc <0:
            continue
        if  rr >= R or cc >= C:
            continue
        if visited[rr][cc]:
            continue
        if m[rr][cc] == 1:
            continue

        rq.append(rr)
        cq.append(cc)

        visited[rr][cc] = True
        
print(solve())
print(visited)