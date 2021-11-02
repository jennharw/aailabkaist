# def main()
# main()
# Object Orient program ; Class __init__(self), __del__(self), 함수 -> constructor
# Instance
# module import 

# Data Structure
# 1. ArrayList
x = ['a', 'b', 'd', 'e', 'f']

#1_1. insert
idxInsert= 2
valInsert = 'c'
#x[idxInsert] = valInsert

#for i in range(idxInsert, len(x)):
#    temp = x[idxInsert],    x[idxInsert+1] = temp,    idxInsert += 1
y = list(range(6))
for i in range(0,idxInsert):
    y[i] = x[i]
y[idxInsert] = valInsert
for i in range(idxInsert, len(x)):
    y[i+1] = x[i]

#1_2. delete
idxDelete = 3

z = list(range(5))
for i in range(0,idxDelete):
    z[i] = y[i]
for i in range(idxDelete+1, len(y)):
    z[i-1] = y[i]

# 2. LinkedList
from SinglyLinkedList import SinglyLinkedList
list1 = SinglyLinkedList()
list1.insertAt("a",0)
list1.insertAt("b",1)
list1.insertAt("c",2)
list1.insertAt("d",3)
list1.insertAt("e",4)
#list1.printStatus()
list1.insertAt("f",2)
#list1.printStatus()
list1.removeAt(3)
#list1.printStatus()

#3. Stack
from Stack import Stack
stack = Stack()
stack.push("c")
stack.push("e")

#print(stack.pop())

#4 Queue
from Queue import Queue

# queue = Queue()
# queue.enqueue("e")
# queue.enqueue("f")
# queue.enqueue("g")

#print(queue.dequeue())

#5 Tree Binary Search Tree
from BinarySearchTree import BinarySearchTree
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

# print(tree.traverseLevelOrder())
# print(tree.traverseInOrder())
# print(tree.traversePreOrder())
# print(tree.traversePostOrder())
# print(tree.findMin().getValue())
tree.delete(3)
# print(tree.traverseLevelOrder())

#6 Prioirty Queue
from PriorityQueue import PrioirtyQueue

pq = PrioirtyQueue()
pq.enqueue('Tommy',1)
pq.enqueue('Lee', 2)
pq.enqueue('James',3)
pq.enqueue('Peter',99)

print(pq.dequeue())
print(pq.dequeue())
print(pq.dequeue())
print(pq.dequeue())


#7 Binary Heap
"""
PrioirtyQueue implementation
1 Linked List based implementation - Sorted, Unsorted Implementation
2 Tree based implementation - Balanced Tree
Binary Heap is a binary tree with two properties
- shape complete tree
- heap Each node is greater than or equal to each of its children
-- Max heap since we defeind a higher prioirty has a higher value
"""
from BinaryHeap import BinaryHeap
bh = BinaryHeap()
bh.enqueueWithPrioirty('Tommy',1)
bh.enqueueWithPrioirty('Lee', 2)
bh.enqueueWithPrioirty('James',3)
bh.enqueueWithPrioirty('Peter',99)

print(bh.dequeueWithPrioirty())
print(bh.dequeueWithPrioirty())
print(bh.dequeueWithPrioirty())
print(bh.dequeueWithPrioirty())


#8 Hash 

#9 Graph