import numpy as np

x = np.array([[1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1]])
y = [2, 2.7, -0.7, 2]
theta = np.array([0, 1, 2])
z = 0
loss = 0
for i in range (4):
    z = (y[i] - (np.dot(x[i], theta)))
    if z >= 1:
        loss += 0
    else:
        loss += 1 - z

#print(loss/4)


x1 = np.array([1, 0, 0])
x2 = np. array([0,1,0])

norm1 = (np.linalg.norm(x1 - x2)) ** 2
print(norm1)

kernel = np.exp(-norm1/2)
print(kernel)