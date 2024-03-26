from Queue import Queue

class TreeNode:
    nodeLHS = None
    nodeRHS = None
    nodeParent = None
    value = None

    def __init__(self, value, nodeParent):
        self.value = value
        self.nodeParent = nodeParent
    
    def getLHS(self):
        return self.nodeLHS
    def getRHS(self):
        return self.nodeRHS
    def getValue(self):
        return self.value
    def getParent(self):
        return self.nodeParent
    
    def setLHS(self, LHS):
        self.nodeLHS = LHS
    def setRHS(self, RHS):
        self.nodeRHS = RHS
    def setValue(self, value):
        self.value = value
    def setParent(self, nodeParent):
        self.nodeParent = nodeParent

class BinarySearchTree:
    root = None

    def __init__(self):
        pass

    def insert(self, value, node = None):
        if node is None:
            node = self.root

        if self.root is None:
            self.root = TreeNode(value, None)
            return

        if value == node.getValue():
            return
        if value > node.getValue():
            if node.getRHS() is None:
                node.setRHS(TreeNode(value, node))
            else:
                self.insert(value, node.getRHS())
        if value < node.getValue():
            if node.getLHS() is None:
                node.setLHS(TreeNode(value, node))
            else:
                self.insert(value, node.getLHS())
        return

    def search(self, value, node = None):
        if node is None:
            node = self.root
        if value == node.getValue():
            return True
        if value < node.getValue():
            if node.getLHS() is None:
                return False
            else:
                self.search(value, node.getLHS())
        if value > node.getValue():
            if node.getRHS() is None:
                return False
            else:
                self.search(value, node.getRHS())


    def delete(self, value, node = None):
        if node is None:
            node = self.root
        if node.getValue() < value:
            return self.delete(value, node.getRHS())
            
        if node.getValue() > value:
            return self.delete(value, node.getLHS())
        
        if node.getValue() == value:
           
            #childeren 2
            if node.getLHS() is not None and node.getRHS() is not None:
                minNode = self.findMin(node.getRHS())
                node.setValue(minNode.getValue())
                self.delete(minNode.getValue(), node.getRHS())
                return
            
            parent = node.getParent()
    
            if node.getLHS() is not None:
                if node == self.root:
                    self.root == node.getLHS()         
                elif parent.getLHS() == node:
                    parent.setLHS(node.getLHS())
                    node.getLHS().setParent(parent)
                else: 
                    parent.setRHS(node.getLHS())
                    node.getLHS().setParent(parent)
                return
            if node.getRHS() is not None:
                if node == self.root:
                    self.root == node.getRHS()
                elif parent.getLHS() == node:
                    parent.setLHS(node.getRHS())
                    node.getRHS().setParent(parent)
                else: 
                    parent.setRHS(node.getRHS())
                    node.getRHS().setParent(parent)
                return
            if node == self.root:
                self.root = None
            elif parent.getLHS() == node:
                parent.setLHS(None)
            else:
                parent.setRHS(None)
            return
        



        




    def findMax(self, node = None):
        if node is None:
            node = self.root
        if node.getRHS() is None:
            return node
        return self.findMax(node.getRHS())

    def findMin(self, node = None):
        if node is None:
            node = self.root
        if node.getLHS() is None:
            return node
        return self.findMin(node.getLHS())
    
    #BFS queue
    def traverseLevelOrder(self):
        ret = []
        Q = Queue()
        Q.enqueue(self.root)
        while not Q.isEmpty():
            node = Q.dequeue()
            if node is None:
                continue
            ret.append(node.getValue())
            if node.getLHS() is not None:
                Q.enqueue(node.getLHS())
            if node.getRHS() is not None:
                Q.enqueue(node.getRHS())
        return ret

    #DFS stack
    def traverseInOrder(self, node = None):
        ret = []
        if node is None:
            node = self.root

        if node.getLHS() is not None:
            ret = ret + self.traverseInOrder(node.getLHS())
        ret.append(node.getValue())
        if node.getRHS() is not None:
            ret = ret + self.traverseInOrder(node.getRHS())
        return ret
    def traversePreOrder(self, node = None):
        if node is None:
            node = self.root
        ret = []
        ret.append(node.getValue())
        if node.getLHS() is not None:
            ret = ret + self.traversePreOrder(node.getLHS())
        if node.getRHS() is not None:
            ret = ret + self.traversePreOrder(node.getRHS())
        return ret
        

    def traversePostOrder(self, node = None):
        if node is None:
            node = self.root
        ret = []
        if node.getLHS() is not None:
            ret = ret + self.traversePostOrder(node.getLHS())
        if node.getRHS() is not None:
            ret = ret + self.traversePostOrder(node.getRHS())
        ret.append(node.getValue())
        return ret

tree = BinarySearchTree()
tree.insert(3)
tree.insert(2)
tree.insert(0)
tree.insert(5)
tree.insert(7)
tree.insert(4)
tree.insert(6)
tree.insert(1)
tree.insert(9)
tree.insert(8)

print(tree.traverseLevelOrder())
print(tree.traverseInOrder())
print(tree.traversePreOrder())
print(tree.traversePostOrder())
print(tree.findMin().getValue())
tree.delete(3)
print(tree.traverseLevelOrder())

#print(tree.traverseLevelOrder())

dh = [0] *99
dh[0] = "j"
print(dh)

print([_ for _ in range(4)])
V= [_ for _ in range(4)]
V.remove(1)
print(V)

graph = [[0 for _ in range(5)] for _ in range(5)]
graph[0][1] = 3
graph[0][4] = 20
graph[1][2] = 6
graph[1][3] = 5
graph[2][4] = 4
graph[3][4] = 7
graph[1][4] = 15
print(graph)

dist = {} #Memoization Table :start 부터 거리
start = 0
for vertex in range(5):
    dist[vertex] = 99999
dist[start] = 0

V = [_ for _ in range(5)]

print(dist)
#while len(V) !=0:
min = 999999
for u in V:
    if dist[u] < min:
        min = graph[start][u]
        minU = u
V.remove(minU)
for i in range(5):
    if graph[minU][i] != 0:
        if dist[i] > dist[minU] + graph[minU][i]:
            dist[i] = dist[minU] + graph[minU][i]
print(dist)

min = 999999
for u in V:
    if dist[u] < min:
        min = graph[start][u]
        minU = u
V.remove(minU)
for i in range(5):
    if graph[minU][i] != 0:
        if dist[i] > dist[minU] + graph[minU][i]:
            dist[i] = dist[minU] + graph[minU][i]    
print(dist)

print(graph)

src = 0
V = set(_ for _ in range(5))
U = set() # covered vertexes
E = set() # covered edges
U.add(src)
while V != U :
    min = 99999
    for src in U:
        for des in V - U :
            if graph[src][des] != 0 :
                if graph[src][des] < min:
                    min = graph[src][des]
                    edge = (src , des)
                    destination = des
    U.add(destination)
    E.add(edge)
print(U)
print(E)
print("확인")

src = 0
V = set(_ for _ in range(5))
U = set() # covered vertexes
E = set() # covered edges
U.add(src)

min = 99999
for src in U:
    for des in V - U :
        if graph[src][des] != 0 :
            if graph[src][des] < min:
                 min = graph[src][des]
                 edge = (src , des)
                 destination = des
U.add(destination)
E.add(edge)
print(U)
print(E)
# V = [1,2,3,4,5]
# U = [1]
# V= set(_ for _ in range(5))
# print(V)
# V.add(7)
# print(V)
min = 99999
for src in U:
    for des in V - U :
        if graph[src][des] != 0 :
            if graph[src][des] < min:
                 min = graph[src][des]
                 edge = (src , des)
                 destination = des
U.add(destination)
E.add(edge)
print(U)
print(E)
min = 99999
for src in U:
    for des in V - U :
        if graph[src][des] != 0 :
            if graph[src][des] < min:
                 min = graph[src][des]
                 edge = (src , des)
                 destination = des
U.add(destination)
E.add(edge)
print(U)
print(E)

min = 99999
for src in U:
    for des in V - U :
        if graph[src][des] != 0 :
            if graph[src][des] < min:
                 min = graph[src][des]
                 edge = (src , des)
                 destination = des
U.add(destination)
E.add(edge)
print(U)
print(E)
