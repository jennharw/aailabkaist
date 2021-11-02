from PriorityNode import PriorityNode
from SinglyLinkedList import SinglyLinkedList
class PrioirtyQueue:
    list = ''
    def __init__(self):
        self.list = SinglyLinkedList()

    #sorted implementation
    def enqueue(self,value, prioirty):
        


        
        idxInsert = 0
        
        for i in range(self.list.getSize()):
            node = self.list.get(i)
            if self.list.getSize() == 0:
                idxInsert = i
                break
            ##LinkedListNode <- PrioirtyNode
            
            if node.getValue().getPriority() > prioirty:
                idxInsert = i +1
            else:
                idxInsert = i
                break

        self.list.insertAt(PriorityNode(value, prioirty), idxInsert)


    def dequeue(self):
        return self.list.removeAt(0).getValue()