import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')

    output_height = (padded_input.shape[0] - kernel_height) // stride + 1
    output_width = (padded_input.shape[1] - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]
            output_matrix[i, j] = np.sum(region * kernel)
    return output_matrix

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

# this one also has invalid example output on the website
# valid output:
# [[ 1.  1. -4.]
#  [ 9.  7. -4.]
#  [ 0. 14. 16.]]

padding = 1
stride = 2
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
