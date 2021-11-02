#Inverse Matrix
##Block matrix(partitioned matrix) [A : I]

import numpy as np
def GJE(matrix):
    len = matrix.shape[0]
    bm = np.zeros((len, len+3))
    bm[:len, :len] = matrix
    for i in range(len):
        bm[i, len+i] = 1
    for i in range(len):
        p = bm[i,i]
        for j in range(i+1,len):
            q = bm[j,i]
            if q != 0:
                a = q/p
                bm[j, :] = bm[j, :] - a*bm[i, :]
    #back substitution
    for k in range(len-1, -1,-1):
        p = bm[k,k]
        for j in range(k-1,-1,-1):
            q = bm[j,k]
            if q != 0: #매우 작은 숫자를 0으로 인식해야? 
                a = q/p
                bm[j, :] = bm[j, :] - a*bm[k, :]
    #오류 행교환 해야!!!
    return bm[:, len:]

mat = np.ones((3,3))
mat[0,1] = 2
mat[0,2] = 3
mat[1,1] = 3
mat[1,2] = 5
mat[2,0] = 2
mat[2,1] = 3
mat[2,2] = 5
print(mat)
print(GJE(mat))
