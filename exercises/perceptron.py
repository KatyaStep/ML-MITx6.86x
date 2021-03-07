import numpy as np

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        order):

    # # for i in range(len(feature_vector)):
    # #     print(f'current theta: {current_theta}')
    #     if label * (np.dot (current_theta, feature_vector) + current_theta_0) <= 0:
    #         current_theta += label * feature_vector
    #         current_theta_0 += label
    #         mistake += 1
    #         print (f'current theta: {current_theta}, Mistake:{mistake}', end='\n')

    #
    # return current_theta, current_theta_0, mistake

        #print(f'current theta: {current_theta}')

    if label * (np.dot(current_theta, feature_vector)) <= 0:
        current_theta += label * feature_vector
        #current_theta_0 += label
        misclassifier[order] += 1
    return current_theta, misclassifier[order]



# feature_vector = np.array([[-1,-1],[1,0],[-1, 10]])
# label = np.array([1, -1, 1])
# current_theta = [0, 0]
#

feature_vector = np.array([[0,1], [0.9,0]])
label = np.array([1, 1])
current_theta = [0, 0]
#current_theta_0 = 0

# for i in range(10):
#     for i in range (len (feature_vector)):
#         current_theta = perceptron_single_step_update(feature_vector[i], label[i], current_theta)
#         print (current_theta)
misclassifier = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0
}
for i in range(2):
    for i in range (len (feature_vector)):
        order = i
        current_theta, mistake = perceptron_single_step_update (feature_vector[ i ], label[ i ], current_theta, order)
        print (f'theta:{current_theta}, mistake: {mistake}')

#print(current_theta)
#
print('---------------------------------------')

# feature_vector = np.array([[1,0],[-1, 1.5],[-1,1]])
# label = [-1, 1, 1]
# current_theta = np.array([0, 0])
#
# print(perceptron_single_step_update(feature_vector, label, current_theta))



# feature_vector = np.array([1, 2])
# label = 1
# current_theta = np.array([-1, 1])
# current_theta_0 = -1.5
#
# print(perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0))
