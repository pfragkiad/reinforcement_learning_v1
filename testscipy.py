import numpy as np
#A = np.mat('[1 2;3 4]')
#print(A.I)

from scipy import linalg
A = np.array([[1,2],[3,4]])
print(linalg.inv(A))

a = np.array([1,2])
print(a.dot(a))
print(np.matmul(a,a.T))

A = np.array([[1,3,5],[2,5,1],[2,3,8]])
b = np.array([10,8,3])
#print(linalg.inv(A).dot(b))
print(linalg.solve(A,b))
print(linalg.det(A))

import matplotlib.pyplot as plt
rng = np.random.default_rng()
c1, c2 = 5.0,2.0

i = np.r_[1:11]
xi = 0.1*i
yi = c1*np.exp(-xi) + c2*xi
zi = yi + 0.05 * np.max(yi) * rng.standard_normal(len(yi))
A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]]
print(A)
c, resid, rank, sigma = linalg.lstsq(A, zi)
xi2 = np.r_[0.1:1.0:100j]
yi2 = c[0]*np.exp(-xi2) + c[1]*xi2
plt.plot(xi,zi,'x',xi2,yi2)
plt.axis([0,1.1,3.0,5.5])
plt.xlabel('$x_i$')
plt.title('Data fitting with linalg.lstsq')
plt.show()