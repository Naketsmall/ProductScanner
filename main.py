import numpy as np
from neiro import D_relu, D_softmax, Neiro

l1 = D_relu(3, 3)
l2 = D_softmax(3, 5)
N = Neiro([l1, l2])

x1 = [[1, 2, 3, 3],
      [2, 3, 4, 5],
      [6, 3, 2, 1]]

y1 = [0, 3, 4, 4]


for i in range(1100):
    N.fit(x1, y1, 10, 2, 0.03)

print(N.predict(x1))

print(N.predict([[1], [2], [6]]))


