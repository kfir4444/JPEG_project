from math import cos, pi, floor
import matplotlib.pyplot as plt
import numpy as np

def delta(x):
    """
    return the delta function of the variable x:
    if x = 0, delta(x) = 1
    if x != 0, delta = 1/sqrt(2)
    input: float
    output: float
    """
    if x == 0:
        return 1
    else:
        return 0.5**0.5


def floor_line(L):
    """
    returns an array, with the integer value of the input array
    input: np.array(dtype = "float")
    output: np.array(dtype = "int")
    """
    return map(floor,L)

def q_gen():
    """
    returns the weited matrix of the JPEG
    input: none
    output: an 8x8 array
    """
    q0 = [10,16,22,28,34,40,46,52]
    q = [np.array(q0)]
    for i in range(7):
        q0.append(q0[-1]+6)
        q0 = q0[1:]
        q.append(np.array(q0))
    return np.array(q)

def jpeg(f):
    """
    Return the image f compressed with JPEG algorythem
    input: 8x8 matrix with values from 0 to 255
    output: 8x8 matrix with values from 0 to 255
    C - the cosine transformation matrix-
    [C]_{i,j} = \frac{\delta_i}{\sqrt{N}}\cdot cos(\frac{i(2j+1)\pi}{2N})
    """
    C = np.zeros((8,8))
    q = C
    for k in range(8):
        for h in range(8):
            C[k,h] = delta(k)/(8**0.5)*cos(k*(2*h+1)*pi/(16))
    f = np.array(f)-128
    alpha = np.dot(np.dot(C,f),C.T)
    b = np.zeros((8,8))
    q = q_gen()
    for i in range(8):
        for j in range(8):
            b[i,j] = floor(a[i,j]/q[i,j] + 1/2)*q[i,j]
    ft_jpeg = np.dot(C.T,np.dot(b,C))
    f_jpeg += 128.5
    f_jpeg = map(floor_line,f_jpeg)
    for i in range(8):
        for j in range(8):
            if f_jpeg[i,j] > 255:
                f_jpeg[i,j] = 255
            elif f_jpeg[i,j] < 0:
                f_jpeg[i,j] =0
    return f_jpeg
