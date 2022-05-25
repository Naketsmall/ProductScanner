import numpy as np
from neiro import D_relu, D_softmax, Neiro

x1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
#x1 = [[1], [2], [3]]
l1 = D_relu(3, 3)
l2 = D_softmax(3, 5)

N = Neiro([l1, l2])
#print(N.predict(x1))
for i in range(1000):
    N.fit(x1, [0, 0, 4], 0.01)
print(N.predict(x1))

