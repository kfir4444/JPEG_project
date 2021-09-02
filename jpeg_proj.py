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
    f_arr_jpeg = np.array([np.array([[[[0]*N]*N]*N]*N)]*4)
    comp_rates = np.array([np.array([[0.0]*N]*N)]*4)
    i_comp = [0]*4
    j_comp = [0]*4
    for k in range(4):
        for j in range(N):
            for i in range(N):
                f_arr_jpeg[k][i,j], comp_rates[k][i,j] = jpeg(f_arr[i,j],k+1)
        i_comp[k], j_comp[k] = np.unravel_index(comp_rates[k].argmax(), comp_rates[k].shape)
    
    for k in range(4):
        for i in range(N):
            for j in range(N):
                #print(comp_rates[k][i,j])
                comp_rates[k,i,j] = round((N**2 - comp_rates[k,i,j]) / N**2 * 100 , 2)
                #print(comp_rates[k][i,j])
        #print(comp_rates[0])
    min_diff = [-1]*4
    i_diff = [0]*4
    j_diff = [0]*4
    diffs = [np.zeros(np.shape(f))]*4
    for k in range(4):
        for i in range(N):
            for j in range(N):
                f_arr_jpeg[k][i,j] = np.roll(f_arr_jpeg[k][i,j], -i, axis=0) #i ups
                f_arr_jpeg[k][i,j] = np.roll(f_arr_jpeg[k][i,j], -j, axis=1) #j lefts
                curr_diff = frobenius_norm(f,f_arr_jpeg[k][i,j],8)
                diffs[k][i,j] = curr_diff
                if curr_diff < min_diff[k] or min_diff[k] == -1:
                    i_diff[k] = i
                    j_diff[k] = j
                    min_diff[k] = curr_diff
    first = 0
    second = 0
    third = 0
    return (f_arr_jpeg[first][first,first] , f_arr_jpeg[second][i_diff[second], j_diff[second]], f_arr_jpeg[third][i_comp[third], j_comp[third]],
            diffs[first][first,first], diffs[second][i_diff[second], j_diff[second]], diffs[third][i_comp[third], j_comp[third]],
            comp_rates[first][first,first], comp_rates[second][i_diff[second], j_diff[second]], comp_rates[third][i_comp[third], j_comp[third]])
def diff(mat):
    arr = []
    for row in mat:
        for x in row:
            arr.append(x)
    return max(arr) - min(arr)
f = cv2.imread("pexels-8296783.jpg", 0)
print(f.shape)
#f = f[f.shape[0]//2 - 39:f.shape[0]//2 + 40, f.shape[1]//2 - 39:f.shape[1]//2 + 40] #f[:80,:80]#[:floor(f.shape[0]/8)*8, :floor(f.shape[1]/8)*8]
f = f[:(f.shape[0]//8)*8, :(f.shape[1]//8)*8]
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
