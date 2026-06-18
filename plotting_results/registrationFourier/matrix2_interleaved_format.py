import numpy as np


def matrix2_interleaved_format(A):
    # Convert matrix to interleaved format (row-major flatten)
    output_vector = A.flatten(order='C')
    return output_vector
