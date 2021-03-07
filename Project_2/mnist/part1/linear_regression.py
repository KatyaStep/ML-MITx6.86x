import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
    # raise NotImplementedError
    # print(X)
    # print("-------------")
    #X = np.concatenate((matrix_of_ones, X), 1)
    #print(X)
 
    matrix_of_ones = np.ones(Y.shape)[...,None]
    n_cols = X.shape[1]
    # print(X)
    # print("------------")
    # print(n_cols)
    # print(np.identity(n_cols))
    coeff = np.linalg.inv(X.transpose().dot(X) + lambda_factor * np.identity(n_cols)).dot(X.transpose()).dot(Y)

    return coeff
### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)

X = np.arange(1, 16).reshape(3, 5)
Y = np.arange(1, 4)
lambda_factor = 0.5

exp_res = np.array([-0.03411225,  0.00320187,  0.04051599,  0.07783012,  0.11514424])
print(closed_form(X,Y, lambda_factor))

# print ( exp_res - closed_form(X,Y, lambda_factor))