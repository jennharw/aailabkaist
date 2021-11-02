class BinaryHeap:
    def __init__(self):
        self.arrPriority = [0] * 99
        self.arrValue = [0] * 99
        self.size = 0

    #insert - Percolation
    def enqueueWithPrioirty(self, value, prioirty):
        self.arrPriority[self.size] = prioirty
        self.arrValue[self.size] = value
        self.size = self.size + 1
        self.percolateUp(self.size-1)
    
    def percolateUp(self, idxPercolate):
        if idxPercolate == 0:
            return
        parent = int((idxPercolate-1)/2)
        if self.arrPriority[parent] < self.arrPriority[idxPercolate]:
            
            self.arrPriority[parent], self.arrPriority[idxPercolate] = self.arrPriority[idxPercolate] , self.arrPriority[parent]
            self.arrValue[parent] , self.arrValue[idxPercolate] = self.arrValue[idxPercolate] , self.arrValue[parent]
            self.percolateUp(parent)
    
    def dequeueWithPrioirty(self):
        if self.size == 0:
            return
        retPrioirty = self.arrPriority[0]
        retValue = self.arrValue[0]
        self.arrPriority[0] = self.arrPriority[self.size - 1]
        self.arrValue[0]= self.arrValue[self.size -1]
        self.size -= 1
        self.percolateDown(0)
        return retValue

    def percolateDown(self, idxPercolate):
        if 2 * idxPercolate + 1 > self.size:
            return
        else:
            leftChild = idxPercolate*2 +1
            leftPrioirty = self.arrPriority[leftChild]
        if 2 * idxPercolate + 2 > self.size:
            return
        else:
            rightChild = idxPercolate*2 +2
            rightPriority = self.arrPriority[rightChild]
        
        if leftPrioirty > rightPriority:
            biggerChild = leftChild
        else: 
            biggerChild = rightChild
        
        if self.arrPriority[idxPercolate] < self.arrPriority[biggerChild]:
            self.arrPriority[idxPercolate] ,self.arrPriority[biggerChild]= self.arrPriority[biggerChild] , self.arrPriority[idxPercolate]
            self.arrValue[idxPercolate] ,self.arrValue[biggerChild]= self.arrValue[biggerChild],self.arrValue[idxPercolate]
            self.percolateDown(biggerChild)