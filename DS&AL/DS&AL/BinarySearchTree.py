from TreeNode import TreeNode

from Queue import Queue

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
                    return
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




#priority queue
#heap
#hash

#recursion, memoization
#sort selectiong bubble
#찾기 