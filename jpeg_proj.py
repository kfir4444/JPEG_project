#from jpeg import *
import numpy as np
def zeros_on_zigzag(f,N):
    i = N-1
    j = N-1
    res = 0
    step = 1
    while i != 0 or j != 0:
        if f[i][j] == 0:
            res += 1
        else:
            break
        if (i == N-1 and step == 1) or (i == 0 and step == -1):
            j -= 1
            step = -step
        elif (j == N-1 and step == -1) or (j == 0 and step == 1):
            i -= 1
            step = -step
        else:
            i+=step
            j-=step
    return res
    
def main(f,N,C,Q):
    #creating f_arr with all possibilities for row and columns shifts
    temp = np.array(f)
    f_arr = []
    for i in range(N):
        f_arr.append([])
        for j in range(N):
            f_arr[i].append(temp)
            temp = np.roll(temp, 1, axis=1) #right
        temp = np.roll(temp, 1, axis=0) #down
    f_arr_jpeg = [[jpeg(img) for img in row] for row in f_arr]
    for i in range(N):
        for j in range(N):
            f_arr_jpeg[i][j] = np.roll(f_arr_jpeg[i][j], -j, axis=1) #j lefts
            f_arr_jpeg[i][j] = np.roll(f_arr_jpeg[i][j], -i, axis=0) #i ups
    
