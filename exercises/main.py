import numpy as np
from sklearn.metrics import accuracy_score

def get_sum_metrics(predictions, metrics=None):
    if metrics is None:
        metrics = []

    for i in range(3):
        metrics.append(lambda x, i=i: x + i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics

def main():
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9

if __name__ == "__main__":
    main()



# def randomization(n):
#     """
#     Arg:
#       n - an integer
#     Returns:
#       A - a randomly-generated nx1 Numpy array.
#     """
#     #Your code here
#     A = np.random.random([n,1])
#     return A
#
#
# # print(randomization(5))
#
# def operations(h, w):
#     """
#     Takes two inputs, h and w, and makes two Numpy arrays A and B of size
#     h x w, and returns A, B, and s, the sum of A and B.
#
#     Arg:
#       h - an integer describing the height of A and B
#       w - an integer describing the width of A and B
#     Returns (in this order):
#       A - a randomly-generated h x w Numpy array.
#       B - a randomly-generated h x w Numpy array.
#       s - the sum of A and B.
#     """
#     #Your code here
#
#     A = np.random.random([h,w])
#     B = np.random.random([h, w])
#     s = A + B
#     print(f"THis is A:{A}")
#     print()
#     print(f"THis is B:{B}")
#     print('-----------')
#     print(f'Sum: {s}')
#     return True
#
# #operations(3, 3)
#
#
# def norm(A, B):
#     """
#     Takes two Numpy column arrays, A and B, and returns the L2 norm of their
#     sum.
#
#     Arg:
#       A - a Numpy array
#       B - a Numpy array
#     Returns:
#       s - the L2 norm of A+B.
#     """
#     #Your code here
#     #raise NotImplementedError
#     s = np.linalg.norm(A+B)
#     return A+B, s
#
# # A = np.array([[1,2,3], [4,5,6]])
# # B = np.array([[1,0,4], [3,6,7]])
# # print(norm(A, B))
#
# def neural_network(inputs, weights):
#     """
#      Takes an input vector and runs it through a 1-layer neural network
#      with a given weight matrix and returns the output.
#
#      Arg:
#        inputs - 2 x 1 NumPy array
#        weights - 2 x 1 NumPy array
#      Returns (in this order):
#        out - a 1 x 1 NumPy array, representing the output of the neural network
#     """
#     #Your code here
#     weights_t = np.transpose(weights)
#     z = np.tanh(np.matmul(weights_t, inputs))
#     return z
#     #raise NotImplementedError
#
# weights = np.array([[3],[3]])
# inputs = np.array([[1],[2]])
#
# # print(neural_network(inputs, weights))
#
#
# def scalar_function(x, y):
#     """
#     Returns the f(x,y) defined in the problem statement.
#     """
#     #Your code here
#     #raise NotImplementedError
#     # if x <= y:
#     #     return np.multiply(x,y)
#     # else:
#     #     return np.true_divide(x,y)
#     return np.multiply(x,y) if x <= y else np.true_divide(x,y)
#
# # vfunc = np.vectorize(scalar_function)
# # print(scalar_function(4,6))
# # print(vfunc([4,6],[2,8]))
#
#
# def vector_function(x, y):
#     """
#     Make sure vector_function can deal with vector input x,y
#     """
#     #Your code here
#     #raise NotImplementedError
#     vfunc = np.vectorize(scalar_function)
#     return vfunc(x,y)
#
# print(vector_function([4,6],[2,8]))