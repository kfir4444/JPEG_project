from math import cos, pi, floor, exp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

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

def FLoor(M):
    """
    returns the integer value of a matrix
    """
    M1 = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            M1[i,j] = floor(M[i,j])
    return M1

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

def zeros_on_zigzag(f,N):
    """
    returns the No. of zeros in the end of the zigzag of the metrix
    """
    #print(f)
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
def q_2(beta):
    """
    returns the Q martix, where [q]_{i,j} = exp(\frac{(i+j)}{\alpha})
    input: courently unknown
    output: 8x8 matrix
    """
    q = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            q[i,j] = 10*exp(((i**2+j**2)**0.5)/beta)
    return q
def jpeg(f,i):
    d = {"1":jpeg1,
         "2":jpeg4,
         "3":jpeg2,
         "4":jpeg3}
    i = str(i)
    func = d[i]
    return func(f)
def jpeg1(f):
    """
    Return the image f compressed with JPEG algorythem
    input: 8x8 matrix with values from 0 to 255
    output: 8x8 matrix with values from 0 to 255
    C - the cosine transformation matrix-
    [C]_{i,j} = \frac{\delta_i}{\sqrt{N}}\cdot cos(\frac{i(2j+1)\pi}{2N})
    """
    C = np.array([[1/8**0.5]*8]*8)
    for i in range(1,8):
        for j in range(8):
            C[i,j] = cos((2*j+1)*i*pi/16)/2
    f = np.array(f)-128
    alpha = np.matmul(np.matmul(C,f),C.T)
    b = np.zeros((8,8))
    q = q_gen()
    for i in range(8):
        for j in range(8):
            b[i,j] = floor(alpha[i,j]/q[i,j] + 1/2)*q[i,j]
    z = zeros_on_zigzag(b,8)
    f_jpeg = np.matmul(C.T,np.matmul(b,C))
    f_jpeg += 128.5
    f_jpeg = FLoor(f_jpeg)
    for i in range(8):
        for j in range(8):
            if f_jpeg[i,j] > 255:
                f_jpeg[i,j] = 255
            elif f_jpeg[i,j] < 0:
                f_jpeg[i,j] =0
    return f_jpeg, z

def jpeg2(f):
    """
    Return the image f compressed with JPEG algorythem
    input: 8x8 matrix with values from 0 to 255
    output: 8x8 matrix with values from 0 to 255
    C - the cosine transformation matrix-
    [C]_{i,j} = \frac{\delta_i}{\sqrt{N}}\cdot cos(\frac{i(2j+1)\pi}{2N})
    """
    beta = 4
    C = np.array([[1/8**0.5]*8]*8)
    for i in range(1,8):
        for j in range(8):
            C[i,j] = cos((2*j+1)*i*pi/16)/2
    f = np.array(f)-128
    alpha = np.matmul(np.matmul(C,f),C.T)
    l = np.zeros((8,8))
    q = q_2(beta)
    for i in range(8):
        for j in range(8):
            l[i,j] = floor(alpha[i,j]/q[i,j] + 1/2)
    b = np.multiply(l,q)
    z = zeros_on_zigzag(b,8)
    f_jpeg = np.matmul(C.T,np.matmul(b,C))
    f_jpeg += 128.5
    f_jpeg = FLoor(f_jpeg)
    for i in range(8):
        for j in range(8):
            if f_jpeg[i,j] > 255:
                f_jpeg[i,j] = 255
            elif f_jpeg[i,j] < 0:
                f_jpeg[i,j] =0
    return f_jpeg, z

def remove_zero_coeff(OrigM,M):
    """
    returns the original matrix, with 0 in the positions when M is 0
    """
    for i in range(8):
        for j in range(8):
            if M[i,j] == 0:
                OrigM[i,j] = 0
    return OrigM

def jpeg3(f):
    """
    Return the image f compressed with JPEG algorythem
    input: 8x8 matrix with values from 0 to 255
    output: 8x8 matrix with values from 0 to 255
    C - the cosine transformation matrix-
    [C]_{i,j} = \frac{\delta_i}{\sqrt{N}}\cdot cos(\frac{i(2j+1)\pi}{2N})
    """
    beta = 4
    C = np.array([[1/8**0.5]*8]*8)
    for i in range(1,8):
        for j in range(8):
            C[i,j] = cos((2*j+1)*i*pi/16)/2
    f = np.array(f)-128
    alpha = np.matmul(np.matmul(C,f),C.T)
    l = np.zeros((8,8))
    q = q_2(beta)
    for i in range(8):
        for j in range(8):
            l[i,j] = floor(alpha[i,j]/q[i,j] + 1/2)
    b = remove_zero_coeff(alpha, l)
    z = zeros_on_zigzag(b,8)
    f_jpeg = np.matmul(C.T,np.matmul(b,C))
    f_jpeg += 128.5
    f_jpeg = FLoor(f_jpeg)
    for i in range(8):
        for j in range(8):
            if f_jpeg[i,j] > 255:
                f_jpeg[i,j] = 255
            elif f_jpeg[i,j] < 0:
                f_jpeg[i,j] =0
    return f_jpeg, z

def jpeg4(f):
    """
    Return the image f compressed with JPEG algorythem
    input: 8x8 matrix with values from 0 to 255
    output: 8x8 matrix with values from 0 to 255
    C - the cosine transformation matrix-
    [C]_{i,j} = \frac{\delta_i}{\sqrt{N}}\cdot cos(\frac{i(2j+1)\pi}{2N})
    """
    C = np.array([[1/8**0.5]*8]*8)
    for i in range(1,8):
        for j in range(8):
            C[i,j] = cos((2*j+1)*i*pi/16)/2
    f = np.array(f)-128
    alpha = np.matmul(np.matmul(C,f),C.T)
    l = np.zeros((8,8))
    q = q_gen()
    for i in range(8):
        for j in range(8):
            l[i,j] = floor(alpha[i,j]/q[i,j] + 1/2)
    b = remove_zero_coeff(alpha, l)
    z = zeros_on_zigzag(b,8)
    f_jpeg = np.matmul(C.T,np.matmul(b,C))
    f_jpeg += 128.5
    f_jpeg = FLoor(f_jpeg)
    for i in range(8):
        for j in range(8):
            if f_jpeg[i,j] > 255:
                f_jpeg[i,j] = 255
            elif f_jpeg[i,j] < 0:
                f_jpeg[i,j] =0
    return f_jpeg, z
