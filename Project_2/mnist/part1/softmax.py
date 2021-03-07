import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    #raise NotImplementedError
    
    R = (theta.dot(X.T))/temp_parameter
    
    # Compute fixed deduction factor for numerical stability (c is a vector: 1xn)
    c = np.max(R, axis = 0)
    # Compute H matrix
    H = np.exp(R - c)
    
    # Divide H by the normalizing term
    H = H/np.sum(H, axis = 0)
    
    # print(R)
    # print('----------')
    # print(c)
    # print('----------')
    # print(H)
    # print('----------')
    #print(H2)
    return H  

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    #raise NotImplementedError
    reg_term = lambda_factor/2 * (np.linalg.norm(theta)**2)
    H = compute_probabilities(X, theta, temp_parameter)
    
    k = theta.shape[0]
    n = X.shape[0]

    # Create a sparse matrix 
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()

    log_H = np.log(H)

    error = (-1/n) * np.sum(log_H[M==1])

    c = error + reg_term

    return c


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    # raise NotImplementedError
    k = theta.shape[0]
    n = X.shape[0]

    H = compute_probabilities(X, theta, temp_parameter)
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()

    grad_descent = (-1/(temp_parameter*n)) * (np.dot((M - H), X)) + lambda_factor*theta
    theta_final = theta - alpha * grad_descent

    return theta_final


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    # raise NotImplementedError
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)

    #print(train_y_mod3, test_y_mod3)

    return train_y_mod3, test_y_mod3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    #raise NotImplementedError
    test_error = 0
    assigned_labels = get_classification(X, theta, temp_parameter)
    test_error = 1 - np.mean(np.mod(assigned_labels,3) == Y)

    return test_error
    


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)

n, d, k = 3, 5, 7
X = np.arange(0, n * d).reshape(n, d)
Y = np.arange(0, n)
theta = np.arange(0, k * d).reshape(k, d)
zeros = np.zeros((k, d))
alpha = 2
temp = 0.2
lambda_factor = 0.5

run_gradient_descent_iteration(X, Y, theta, alpha, temp, lambda_factor)



# exp_res = np.array([[ -7.14285714,  -5.23809524,  -3.33333333,  -1.42857143, 0.47619048],
# [  9.52380952,  11.42857143,  13.33333333,  15.23809524, 17.14285714],
# [ 26.19047619,  28.0952381 ,  30.        ,  31.9047619 , 33.80952381],
# [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
# [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
# [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
# [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286]
#     ])