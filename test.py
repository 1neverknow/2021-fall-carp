import numpy as np

arr = np.array([0, 1, 0, 2, 3, 4])
print(np.where(arr))


for i, j in enumerate(np.where(arr)[0]):
    print('i=', i, ' j=', j)
