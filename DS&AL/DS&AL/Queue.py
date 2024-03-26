from SinglyLinkedList import SinglyLinkedList

class Queue:
    lstInstance = SinglyLinkedList()
    def dequeue(self):
        return self.lstInstance.removeAt(0)
    def enqueue(self, value):
        self.lstInstance.insertAt(value, self.lstInstance.getSize())
    def isEmpty(self):
        return self.lstInstance.getSize() == 0
    
