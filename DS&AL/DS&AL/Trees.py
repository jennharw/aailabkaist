#A tree is an undirected graph with no cycles
#Equivalently, a tree is a connected graph with N nodes and N-1 edges
#Hierarchy

#Edge list, Adjacency list
#root node
#Binary tree, every node has at most two child nodes, Binary Search Trees(BST) left < i < right
#flatten array list, left node : 2*1 +1 right node 2*1+2, parnet node (i-1)/2

#Tree Algorithms, Recursive main motivation for rooting a n undirected tree
#Root node as a starting point



class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = list()
    def getNodeChildren(self):
        return self.children
    def getValue(self):
        return self.value
    
    def setNodeChild(self, node):
        self.children.append(node)
    def setValue(self, value):
        self.value = value

node = TreeNode(5)
node1 = TreeNode(4)
node.setNodeChild(node1)
node2 = TreeNode(1)
node1.setNodeChild(node2)
node3 = TreeNode(2)
node2.setNodeChild(node3)
node4 = TreeNode(9)
node2.setNodeChild(node4)
node5 = TreeNode(-6)
node1.setNodeChild(node5)
node6 = TreeNode(3)
node.setNodeChild(node6)
node7 = TreeNode(0)
node8 = TreeNode(7)
node9 = TreeNode(-4)
node10 = TreeNode(8)
node6.setNodeChild(node7)
node6.setNodeChild(node8)
node6.setNodeChild(node9)
node8.setNodeChild(node10)


#Leaf Sum, What is the sum of all the leaf node values in a tree?
def leafSum(node):
    if node ==  None:
        return -1
    if isLeaf(node):
        return node.getValue()
    total = 0
    for child in node.getNodeChildren():
        total += leafSum(child)
    return total

def isLeaf(node):
    return len(node.getNodeChildren()) == 0

print(leafSum(node))

# Tree Hieght, Height of a binary tree. The height of a tree is the number of edges from the root to the lowest leaf
class TreeNode:
    nodeLHS = None
    nodeRHS = None
    nodeParent = None
    value = None

    def __init__(self, value):
        self.value = value
    
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

def treeHeight(node):
    if node == None:
        return -1
    # if node.getLHS() == None and node.getRHS() == None:
    #     return 0
    return max(treeHeight(node.getLHS()), treeHeight(node.getRHS())) + 1

node = TreeNode("a")
node1 = TreeNode("b")
node2 = TreeNode("c")
node.setLHS(node1)
node.setRHS(node2)
node3 = TreeNode("d")
node1.setLHS(node3)
node4 = TreeNode("e")
node3.setRHS(node4)   

print(treeHeight(node))
print(treeHeight(node1))

#Root a tree
#Sometimes it's useful to root an undirected tree to add structure to the problem you're trying to solve 
# Rooting a tree is easily done depth first

class TreeNode:
    def __init__(self, id, parentNode, children):
        self.id = id
        self.parentNode = parentNode
        self.children = children #list

g = {0: set([2,1,5]),
         1: set([0]),
         2: set([3,0]),
         3: set([2]),
         4: set([5]),
         5: set([4,6,0]),
         6: set([5])}

#Tree 
# g is the graph represented as an adjancecy list with undirected edges
# rootId is the id of the node to root the tree from 
def buildTree(g, node, parent):
    for child in g[node.id]:
        #Avoid adding an edge pointing back to the parent
        if parent != None and child == parent.id:
            continue
        child = TreeNode(child, node, [])
        node.children.append(child)
        buildTree(g, child, node)
    return node

def rootTree(g, rootId = 0):
    root = TreeNode(rootId, None, [])
    return buildTree(g, root, None)


#Build tree recursively depth first
print(rootTree(g, 0).id)
print(rootTree(g, 0).parentNode)
print(rootTree(g, 0).children[1].children[0].id)
print(rootTree(g, 0).children[2].children[0].id)

print(len(g[0]))
# Tree Center, to select a good node to root our tree
def treeCenters(g):
    n = len(g)
    degree = [0] * n
    leaves = set()

    for i in range(n):
        degree[i] = len(g[i])
        if degree[i] == 0 or degree[i] == 1:
            leaves.add(i)
            degree[i] = 0
    count = len(leaves)
    while count < n:
        new_leaves = []
        for node in leaves:
            for neighbor in g[node]:
                degree[neighbor] = degree[neighbor] - 1
                if degree[neighbor] == 1:
                    new_leaves.append(neighbor)
            degree[node] = 0
        count += len(new_leaves)
        leaves = new_leaves

    return new_leaves
print(treeCenters(g))


#Tree Isopmorphism, meaning they are structurally equivalent => a tree into a unique encoding
# Graph Isomorphism, G1(V1, E1) , G2(V2, E2) 
#unique encoding is simply a unique string that represents a tree, 
# Tree Encoding AHU algorithm 

def treesAreIsomorphic(tree1, tree2):
    tree1_centers = treeCenters(tree1)
    tree2_centers = treeCenters(tree2)

    tree1_rooted = rootTree(tree1, tree1_centers[0])
    tree1_encoded = encode(tree1_rooted)

    for center in tree2_centers:
        tree2_rooted = rootTree(tree2, center)
        tree2_encoded = encode(tree2_rooted)
        if tree1_encoded == tree2_encoded:
            return True
    return False


#TreeNode, id, parent, children
def encode(node):
    if node == None:
        return ""
    labels = []
    for child in node.children:
        labels.append(encode(child))
    labels = sorted(labels)
    result = ""
    for label in labels:
        result += label
    return "(" + result + ")"

test1 = {0: set([1]),
         1: set([0, 2,3]),
         2: set([1,4]),
         3: set([1]),
         4: set([2])}

test2 = {0: set([2]),
         1: set([2]),
         2: set([0,1,3]),
         3: set([4]),
         4: set([3])}

#find centers
print(treeCenters(test1))
print(treeCenters(test2))
#build tree
root1 = rootTree(test1, rootId=1) #2
root2 = rootTree(test2, 2)

print(root1.children[0].id)
print(root1.children[1].children[0].id)

print(root2.children[0].id)
print(root2.children[1].id)

#encode
print(encode(root1))
print(encode(root2))

print(treesAreIsomorphic(test1, test2))

graph1 = {0: set([1]),
         1: set([0, 2,4]),
         2: set([1]),
         3: set([4,5]),
         4: set([1,3]),
         5:set([3])}

graph2 = {0: set([1]),
         1: set([0,2]),
         2: set([1,4]),
         3: set([4]),
         4: set([2,3,5]),
         5: set([4])}

print(treeCenters(g))

print(treeCenters(graph1))
print(treeCenters(graph2))

#build tree
root1 = rootTree(graph1, rootId=4) #2
root2 = rootTree(graph2, 2)

print(encode(root1))
print(encode(root2))

print(treesAreIsomorphic(graph1, graph2))


encodingTest = {0: set([2,1,3]),
         1: set([0, 4, 5]),
         2: set([0,6,7]),
         3: set([0,8]),
         4: set([1]),
         5:set([1,9]),
         6 : set([2]),
         7: set([2]),
         8:set([3]),
         9:set([5])}

print(treeCenters(encodingTest))
print(encode(rootTree(encodingTest, treeCenters(encodingTest)[0])))
