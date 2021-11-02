from Node import Node
class SinglyLinkedList:
    nodeHead = ""
    nodeTail = ""
    size = 0

    def __init__(self):
        self.nodeTail = Node(binTail = True)
        self.nodeHead = Node(binHead = True, nodeNext = self.nodeTail)

    def insertAt(self, objInsert, idxInsert):
        nodeNew = Node(objValue = objInsert)
        nodePrev = self.get(idxInsert -1)
        nodeNext = nodePrev.getNext()

        nodePrev.setNext(nodeNew)
        nodeNew.setNext(nodeNext)

        self.size = self.size + 1
    
    def removeAt(self, idxInsert):
        nodePrev = self.get(idxInsert -1)
        nodeRemove = nodePrev.getNext()
        nodeNext = nodeRemove.getNext()

        nodePrev.setNext(nodeNext)

        self.size = self.size -1
        return nodeRemove.getValue()


    def get(self, idx):
        nodeReturn = self.nodeHead
        for i in range(idx + 1):
            nodeReturn = nodeReturn.getNext()
        return nodeReturn
    def getSize(self):
        return self.size

    def printStatus(self):
        nodeCurrent = self.nodeHead
        while nodeCurrent.getNext().isTail() == False:
            nodeCurrent = nodeCurrent.getNext()
            print(nodeCurrent.getValue())
        print(" ")

