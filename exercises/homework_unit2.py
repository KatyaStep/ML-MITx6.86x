import numpy as np
#
# u = np.array([6, 0, 3, 6])
# v = np.array([4, 2, 1])
# y = np.array([[5, 0, 7], [0, 2, 0], [4, 0, 0], [0, 3, 6]])
# x = np.dot(u.reshape(4,1), v.reshape(1,3))
#
# #print(v.reshape(1,3))
#
# #print(u.reshape(4,1))
#
# print('------------')
# print(f'X:{x}')
# print('------------')
# rows = y.shape[0]
# columns = y.shape[1]
# square_error = 0
# sum_of_errors = 0
#
# for row in range(rows):
#     for column in range(columns):
#         if y[row][column] != 0:
#             sum_of_errors += (y[row][column] - x[row][column]) ** 2
#
# square_error = 1/2 * sum_of_errors
#
# regularization_U = 0
# regularization_V = 0
#
# for row in range(u.shape[0]):
#     regularization_U += (u[row] ** 2)
#
# for row in range(v.shape[0]):
#     regularization_V += (v[row] ** 2)
#
#
# print(f'Squared error is: {square_error}')
# print('------------')
# print(f'Regularization of U is: {regularization_U}')
# print(f'Regularization of U is: {regularization_U/2}')
# print('------------')
# print(f'Regularization of V is: {regularization_V}')
# print(f'Regularization of V is: {regularization_V/2}')
# print('------------')
# print(f'Sum of regularization is {regularization_U/2 + regularization_V/2}')


x = np.array([])