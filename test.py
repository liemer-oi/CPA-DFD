import numpy as np

arr = []
for i in range(5): 
    a = []
    for j in range(i+1):
        a.append(j)
    arr.append(a)
arr = np.array(arr)
x = 1