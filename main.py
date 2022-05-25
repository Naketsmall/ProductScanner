import numpy as np
from neiro import D_relu, D_softmax, Neiro

x1 = [[1, 2, 3, 3], [2, 3, 4, 5], [6, 3, 2, 1]]
#x1 = [[1], [2], [3]]
l1 = D_relu(3, 3)
l2 = D_softmax(3, 5)

N = Neiro([l1, l2])
#print(N.predict(x1))
for i in range(1100):
    N.back_prop(x1, [0, 3, 4, 4], 0.01)
print(N.predict(x1))

