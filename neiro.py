from abc import abstractmethod, ABC
import numpy as np
from typing import List


class Dense(ABC):

    def __init__(self, in_n, out_n):
        self.n = out_n
        # self.W = np.random.rand(out_n, in_n)
        # self.b = np.random.rand(out_n, 1)
        self.W = np.random.rand(out_n, in_n)
        self.b = np.random.rand(out_n, 1)

        # print(self.W)
        # print(self.b)
        # print()

    @abstractmethod
    def activate(self, t):
        pass

    def fit(self, dh, in_x, alpha):
        print(type(self))
        print('Fit --- dh:', dh)
        # print('Fit --- deriv:', self.deriv(self.W @ in_x + self.b))
        dt = dh * self.deriv(self.W @ in_x + self.b)
        print('Fit --- dt:', dt)
        print('Fit --- in_x:', in_x)
        dW = dt @ np.array(in_x).T
        db = np.sum(dt, axis=1, keepdims=1)
        d_in_x = self.W.T @ dt
        # print('Fit --- alpha:', alpha)
        print('Fit --- dW:', dW)
        # print('Fit --- W:', self.W)
        print('Fit --- db:', db)
        # print('Fit --- b:', self.b)
        # print('Fit --- d_in_x:', d_in_x)
        # print()
        self.W -= alpha * dW
        self.b -= alpha * db

        return d_in_x

    @abstractmethod
    def deriv(self, h):
        pass

    def do(self, inp):
        return self.activate(self.W @ inp + self.b)


class D_relu(Dense):

    def deriv(self, h):
        return (h >= 0).astype(float)

    def activate(self, t):
        return np.maximum(t, 0)




class D_softmax(Dense):

    def activate(self, t):
        out = np.exp(t)
        return out / np.sum(out, axis=0, keepdims=True)

    def deriv(self, h):
        return 1


class Neiro():

    def __init__(self, layers: List[Dense]):
        self.layers = layers

    def predict(self, x):
        for layer in self.layers:
            x = layer.do(x)
        return x

    @staticmethod
    def sparse_cross_entropy_batch(z, y):
        return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

    @staticmethod
    def to_full_batch(y, num_out_neurons):
        y_full = np.zeros((len(y), num_out_neurons))
        for j, yj in enumerate(y):
            y_full[j, yj] = 1
        return y_full.T

    def back_prop(self, X, y, alpha: int):
        inter_h = [X]
        x = X
        for layer in self.layers:
            x = layer.do(x)
            inter_h.append(x)

        y_full = self.to_full_batch(y, self.layers[-1].n)
        dh = x - y_full
        for i in range(len(inter_h) - 1, 0, -1):
            dh = self.layers[i-1].fit(dh, inter_h[i-1], alpha)
