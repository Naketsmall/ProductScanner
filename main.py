from neiro import Neiro, D_relu, D_softmax, D_sigm
import sklearn.datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

N = Neiro([D_relu(64, 16), D_softmax(16, 10)])

df = sklearn.datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    df['data'], df['target'], test_size=0.3, shuffle=False
)

N.fit(X_train, np.array([y_train]).T, 1000, 50, 0.0003)
#N.load_pickle('neiro83.pickle')

q = 0
A = {}
for i in range(len(X_test)):
    if not np.argmax(N.predict(X_test[i])) == y_test[i]:
        q += 1
        if (np.argmax(N.predict(X_test[i])), y_test[i]) in A.keys():
            A[(np.argmax(N.predict(X_test[i])), y_test[i])] += 1
        else:
            A[(np.argmax(N.predict(X_test[i])), y_test[i])] = 1

for i in list(sorted(A.keys())):
    print('values:', i, 'mistakes:', A[i] / q * 100)
print('Common: ', q, '/', len(X_test), '(', q * 100 / len(X_test), '%)')

for i in range(5):
    plt.imshow(X_test[i].reshape((8, 8)))
    print('ref / pred:', y_test[i], np.argmax(N.predict(X_test[i])))
    plt.show()

#N.save_pickle('neiro83.pickle')
#
# 278 (10000, 200, 0.0003)  80%
# 250 (10000, 200, 0.0003)  80%
# 272 (10000, 200, 0.0003)  80%
# 212 (10000, 100, 0.0003)  80%
# 181 (10000, 50,  0.0003)  80%
# 192 (10000, 25,  0.0003)  80%
# 193 (20000, 50,  0.0003)  80%
# 202 (20000, 50,  0.00015) 80%
# 9.4 (10000, 50,  0.0003)  80%  Какой негодяй придумал указывать процент тестовой, а не тренировочной выборки..
# 8.5 (10000, 50,  0.0003)  30%
# 10.5(10000, 100, 0.0003)  30% (added extra ReLu)
# 9.25(10000, 50,  0.0003)  30% (sigm)
