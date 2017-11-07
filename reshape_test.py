'''
test whether the reshape function in numpy is suited for convolutional operation
'''

import numpy as np
from operator import mul

batch_size, width, height, channel = [64, 28, 28, 1]
kernel_width, kernel_height, kernel_num = [5, 5, 20]
r_width = kernel_width*kernel_height*channel
r_k = (width - kernel_width // 2 * 2) * (height - kernel_height // 2 * 2)
        
matrix = np.random.random((batch_size*r_k,kernel_num))

print matrix.shape

result_shape = [batch_size, (width - kernel_width // 2 * 2), (width - kernel_width // 2 * 2), kernel_num]
print result_shape

reshape_result = np.reshape(matrix, result_shape)
print reshape_result.shape

real_result = np.zeros(result_shape)

for num in range(result_shape[0]):
    for i in range(result_shape[1]):
        for j in range(result_shape[2]):
            for k in range(result_shape[3]):
                real_result[num, i, j, k] = matrix[num*r_k+i*result_shape[2]+j,k]
                
assert (np.sum(real_result == reshape_result) == reduce(mul, result_shape, 1))


