import cv2
from JPEG import *
import numpy as np
import matplotlib.pyplot as plt

def frobenius_norm(A1,A2,N):
    summ = 0
    for i in range(N):
        for j in range(N):
            summ += (A1[i][j] - A2[i][j])**2
    return round(summ ** 0.5 , 2)

def main(f,N):
    #creating f_arr with all possibilities for row and columns shifts
    plt.style.use(["grayscale"])
    temp = np.array(f)
    f_arr = np.array([[[[0]*N]*N]*N]*N)
    for i in range(N):
        for j in range(N):
            f_arr[i,j] = np.copy(temp)
            temp = np.roll(temp, 1, axis=1) #right
        temp = np.roll(temp, 1, axis=0) #down
    f_arr_jpeg = np.array([[[[0]*N]*N]*N]*N)
    comp_rates = np.array([[0.0]*N]*N)
    for i in range(N):
        for j in range(N):
            f_arr_jpeg[i,j], comp_rates[i,j] = jpeg(f_arr[i,j])
    #f_arr_jpeg = np.array([[x for x in row] for row in f_arr_jpeg])
    #comp_rates = np.array([[float(x) for x in row] for row in comp_rates])
    #comp_rates = np.array(comp_rates)
    i_comp, j_comp = np.unravel_index(comp_rates.argmax(), comp_rates.shape)
    for i in range(N):
        for j in range(N):
            comp_rates[i,j] = round(float(N**2 - comp_rates[i,j]) / float(N**2) * 100.0 , 2) 
    min_diff = -1
    i_diff = 0
    j_diff = 0
    diffs = np.zeros(np.shape(f))
    for i in range(N):
        for j in range(N):
            f_arr_jpeg[i,j] = np.roll(f_arr_jpeg[i,j], -i, axis=0) #i ups
            f_arr_jpeg[i,j] = np.roll(f_arr_jpeg[i,j], -j, axis=1) #j lefts
            curr_diff = frobenius_norm(f,f_arr_jpeg[i,j],8)
            diffs[i,j] = curr_diff
            if curr_diff < min_diff or min_diff == -1:
                i_diff = i
                j_diff = j
                min_diff = curr_diff
    return (f_arr_jpeg[0,0] , f_arr_jpeg[i_diff, j_diff], f_arr_jpeg[i_comp, j_comp],
            diffs[0,0], diffs[i_diff, j_diff], diffs[i_comp, j_comp],
            comp_rates[0,0], comp_rates[i_diff, j_diff], comp_rates[i_comp, j_comp])
    '''fig, axs = plt.subplots(N,N)
    for i in range(N):
        for j in range(N):
            axs[i,j].imshow(f_arr_jpeg[i,j])
    plt.show()
    
    fig, axs = plt.subplots(2,3)
    for i in range(3): axs[0,i].imshow(f, cmap = 'gray', vmin = 0, vmax = 255)
    axs[0,1].set_title('original')
    axs[1,0].imshow(f_arr_jpeg[0,0], cmap = 'gray', vmin = 0, vmax = 255)
    axs[1,0].set_title('original jpeg\ndistance: ' + str(diffs[0,0])
                       + '\ncomp_rate: ' + str(comp_rates[0,0]))
    axs[1,1].imshow(f_arr_jpeg[i_diff,j_diff], cmap = 'gray', vmin = 0, vmax = 255)
    axs[1,1].set_title('best distance\ndistance: ' + str(diffs[i_diff,j_diff])
                       + '\ncomp_rate: ' + str(comp_rates[i_diff, j_diff]))
    axs[1,2].imshow(f_arr_jpeg[i_comp,j_comp], cmap = 'gray', vmin = 0, vmax = 255)
    axs[1,2].set_title('best compression\ndistance: ' + str(diffs[i_comp,j_comp])
                       + '\ncomp_rate: ' + str(comp_rates[i_comp, j_comp]))
    #print(((i_diff, j_diff),(i_comp,j_comp)))
    #plt.imshow(f_arr_jpeg[0,0])
    #plt.show()'''
def diff(mat):
    arr = []
    for row in mat:
        for x in row:
            arr.append(x)
    return max(arr) - min(arr)
f = cv2.imread("spencer-chow-6W-MS6havNk-unsplash.jpg", 0)
print(f.shape)
f = f[:floor(f.shape[0]/8)*8, :floor(f.shape[1]/8)*8]
f_prev = np.array(f)
f_diff = np.array(f)
f_comp = np.array(f)
prev_diff = []
prev_comp = []
diff_diff = []
diff_comp = []
comp_diff = []
comp_comp = []
for i in range(int(f.shape[0]/8)):
    print(i)
    for j in range(int(f.shape[1]/8)):
        #print(j)
        temp = f[8*i:8*(i+1),8*j:8*(j+1)]
        (f_prev[8*i:8*(i+1),8*j:8*(j+1)],
         f_diff[8*i:8*(i+1),8*j:8*(j+1)],
         f_comp[8*i:8*(i+1),8*j:8*(j+1)],
         t_prev_diff,
         t_diff_diff,
         t_comp_diff,
         t_prev_comp,
         t_diff_comp,
         t_comp_comp)= main(temp,8)
        prev_diff.append(t_prev_diff)
        diff_diff.append(t_diff_diff)
        comp_diff.append(t_comp_diff)
        prev_comp.append(t_prev_comp)
        diff_comp.append(t_diff_comp)
        comp_comp.append(t_comp_comp)
prev_diff = round(sum(prev_diff)/len(prev_diff), 2)
prev_comp = round(sum(prev_comp)/len(prev_comp), 2)
diff_diff = round(sum(diff_diff)/len(diff_diff), 2)
diff_comp = round(sum(diff_comp)/len(diff_comp), 2)
comp_diff = round(sum(comp_diff)/len(comp_diff), 2)
comp_comp = round(sum(comp_comp)/len(comp_comp), 2)
fig, axs = plt.subplots(1,4)
axs[0].imshow(f, cmap = 'gray', vmin = 0, vmax = 255)
axs[0].set_title('original')
axs[1].imshow(f_prev, cmap = 'gray', vmin = 0, vmax = 255)
axs[1].set_title('original jpeg\ndistance: ' + str(prev_diff)
                   + '\ncomp_rate: ' + str(prev_comp))
axs[2].imshow(f_diff, cmap = 'gray', vmin = 0, vmax = 255)
axs[2].set_title('best distance\ndistance: ' + str(diff_diff)
                   + '\ncomp_rate: ' + str(diff_comp))
axs[3].imshow(f_comp, cmap = 'gray', vmin = 0, vmax = 255)
axs[3].set_title('best compression\ndistance: ' + str(comp_diff)
                   + '\ncomp_rate: ' + str(comp_comp))
plt.show()
'''f = np.array([[40,	193,	89,	37,	209,	236,	41,	14],
[102,	165,	36,	150,	247,	104,	7,	19],
[157,	92,	88,	251,	156,	3,	20,	35],
[153,	75,	220,	193,	29,	13,	34,	22],
[116,	173,	240,	54,	11,	38,	20,	19],
[162,	255,	109,	9,	26,	22,	20,	29],
[237,	182,	5,	28,	20,	1,	28,	20],
[222,	33,	8,	23,	24,	29,	23,	23]])
main(f,8)'''
