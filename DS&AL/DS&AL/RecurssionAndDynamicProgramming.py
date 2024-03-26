#Recursion
#Dynamic Programming

#1 
#Fibonacci

def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n-2) + fibonacci(n-1)

for i in range(0,10):
    print(fibonacci(i), end = " ")

print("|n")
#Merge Sort
def MergeSort(lst):
    if len(lst) == 1:
        return lst

    left = []
    right = []
    for i in range(len(lst)):
        if len(lst)/2 > i:
            left.append(lst[i])
        else:
            right.append(lst[i])

    left = MergeSort(left)
    right = MergeSort(right)

    idxCount1 = 0
    idxCount2 = 0
    #left right 비교해서 lst 에 넣기
    for itr in range(len(lst)):
        if idxCount1 == len(left):
            lst[itr] = right[idxCount2]
            idxCount2 += 1
        elif idxCount2 == len(right):
            lst[itr] = left[idxCount1]
            idxCount1 +=1 
        elif left[idxCount1] < right[idxCount2]:
            lst[itr] = left[idxCount1]
            idxCount1 +=1 
        else:
            lst[itr] = right[idxCount2]
            idxCount2 +=1 
    return lst

import random
lstRandom = []
for itr in range(0,10):
    lstRandom.append(random.randrange(0,100))
print(lstRandom)
print(MergeSort(lstRandom))





#2
#Fibonacci
def Fibonacci(n):
    dicFibonacci = {}
    dicFibonacci[0] = 0
    dicFibonacci[1] = 1
    for i in range(2,n+1):
        dicFibonacci[i] = dicFibonacci[i-1] + dicFibonacci[i-2]
    return dicFibonacci[n]
for i in range(0,10):
    print(fibonacci(i), end = " ")

#Assembly Line Scheduling
