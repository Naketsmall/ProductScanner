import random
from abc import abstractmethod, ABC
import numpy as np
from typing import List
import pickle


class Dense(ABC):

    def __init__(self, in_n, out_n):
        self.n = out_n
        self.W = np.array(np.random.rand(in_n, out_n), dtype=np.longdouble) / 100
        self.b = np.array(np.random.rand(1, out_n), dtype=np.longdouble) / 100

    @abstractmethod
    def activate(self, t):
        pass

    def fit(self, dh, in_x, alpha):
        dt = dh * self.deriv(in_x @ self.W + self.b)
        dW = np.array(in_x).T @ dt
        db = np.sum(dt, axis=0, keepdims=1)
        d_in_x = dt @ self.W.T

        self.W -= alpha * dW
        self.b -= alpha * db

        return d_in_x

    @abstractmethod
    def deriv(self, h):
        pass

    def do(self, inp):
        return self.activate(inp @ self.W + self.b)


class D_relu(Dense):

    def activate(self, t):
        return np.maximum(t, 0)

    def deriv(self, h):
        return (h >= 0).astype(float)


class D_sigm(Dense):

    def activate(self, t):
        return np.divide(np.ones(t.shape), (np.ones(t.shape) + np.exp(-t)))

    def deriv(self, h):
        return 1


class D_softmax(Dense):

    def activate(self, t):
        out = np.exp(t)
        return out / np.sum(out, axis=1, keepdims=True)

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
            y_full[j, int(yj)] = 1
        return y_full

    def back_prop(self, X, y, alpha: int):
        inter_h = [X]
        x = X
        for layer in self.layers:
            x = layer.do(x)

            inter_h.append(x)

        y_full = self.to_full_batch(y, self.layers[-1].n)
        dh = x - y_full

        for i in range(len(inter_h) - 1, 0, -1):
            dh = self.layers[i - 1].fit(dh, inter_h[i - 1], alpha)

    def fit(self, X, y, epochs, batch_size, alpha):
        for epoch in range(epochs):
            DS = np.concatenate((X, y), axis=1)
            random.shuffle(DS)
            for i in range(len(DS) // batch_size):
                batch = DS[i * batch_size: i * batch_size + batch_size]
                bX, by = batch[:, :-1], batch[:, -1:]
                self.back_prop(bX, by, alpha)

    def save_pickle(self, file_path):
        with open(file_path, "wb") as outfile:
            pickle.dump(self, outfile)


    def load_pickle(self, file_path):
        with open(file_path, "rb") as infile:
            n = pickle.load(infile)
            self.layers = n.layers.copy()

