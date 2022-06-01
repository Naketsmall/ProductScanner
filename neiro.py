import random
from abc import abstractmethod, ABC
import numpy as np
from typing import List
import pickle
import json


class Dense(ABC):

    def __init__(self, in_n, out_n):
        self.n = out_n
        self.W = np.array(np.random.rand(in_n, out_n), dtype=np.longdouble) / 100
        self.b = np.array(np.random.rand(1, out_n), dtype=np.longdouble) / 100
        self.alpha = 0

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


class D_ReLU(Dense):

    def activate(self, t):
        return np.maximum(t, 0)

    def deriv(self, h):
        return (h >= 0).astype(float)


class D_Leaky_ReLU(Dense):

    def activate(self, t):
        return np.maximum(t, 0.1 * t)

    def deriv(self, h):
        return 1 if h >= 0 else 0.1


class D_ELU(Dense):

    def __init__(self, in_n, out_n, alpha):
        super().__init__(in_n, out_n)
        self.alpha = alpha

    def activate(self, t):
        return self.alpha * (np.exp(t) - 1) if t < 0 else t

    def deriv(self, h):
        return 1 if h >= 0 else self.alpha * np.exp(h)


class D_Softmax(Dense):

    def activate(self, t):
        out = np.exp(t)
        return out / np.sum(out, axis=1, keepdims=True)

    def deriv(self, h):
        return 1


class D_sigm(Dense):

    def activate(self, t):
        return np.divide(np.ones(t.shape), (np.ones(t.shape) + np.exp(-t)))

    def deriv(self, h):
        return self.activate(h) * (1 - self.activate(h))


class D_tanh(Dense):

    def activate(self, t):
        return np.tanh(t)

    def deriv(self, h):
        return 1 / np.power(np.cosh(h), 2)


class Neiro:

    def __init__(self, layers: List[Dense]):
        self.layers = layers

    def predict(self, x):
        for layer in self.layers:
            x = layer.do(x)
        return x

    # TODO format y
    @staticmethod
    def __to_full_batch(y, num_out):
        y_full = np.zeros((len(y), num_out))
        for j, yj in enumerate(y):
            y_full[j, int(yj)] = 1
        return y_full

    # TODO refactor back_prop to format y: (key, val)
    def back_prop(self, X, y, alpha: int):
        inter_h = [X]
        x = X
        for layer in self.layers:
            x = layer.do(x)
            inter_h.append(x)

        y_full = self.__to_full_batch(y, self.layers[-1].n)
        dh = x - y_full

        for i in range(len(inter_h) - 1, 0, -1):
            dh = self.layers[i - 1].fit(dh, inter_h[i - 1], alpha)

    # TODO refactor fit
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


# TODO refactor cross-entropy, mse
def cross_entropy(z, y):
    return -np.sum(np.log(np.array([z[j, int(y[j])] for j in range(len(y))])))
    # return - np.sum([y[j] * np.log(z[j, int(y[j])]) + (1-y[j]) for j in range(len(y))])


def mse(z, y):
    return np.power(np.mean(1 - np.array([z[j, int(y[j])] for j in range(len(y))])), 1 / 2)


def serialize(tosd):
    if type(tosd) == type(Neiro([])):
        return '{"type":"' + str(type(tosd)) \
               + '","layers":' + str([serialize(i) for i in tosd.layers]) + '}'
    else:
        return '{ "type":"' + str(type(tosd)) + \
           '","n":' + str(tosd.n) + \
           ',"W":' + str([list(i) for i in tosd.W]) + \
           ',"b":' + str([list(i) for i in tosd.b]) + \
           ',"alpha":' + str(tosd.alpha) + '}'


def deserialize(serialized):
    unserd = json.loads(serialized)
    if unserd['type'] == str(type(Neiro([]))):
        layers = [deserialize(i) for i in unserd['layers']]
        return Neiro(layers)
    else:
        W = np.array(unserd['W'])
        b = np.array(unserd['b'])
        out_n = unserd['n']
        in_n = len(W[0])
        alpha = unserd['alpha']

        if unserd['type'] == "<class 'neiro.D_ReLU'>":
            D = D_ReLU(in_n, out_n)
        elif unserd['type'] == "<class 'neiro.D_ELU'>":
            D = D_ELU(in_n, out_n, alpha)
        elif unserd['type'] == "<class 'neiro.D_sigm'>":
            D = D_sigm(in_n, out_n)
        elif unserd['type'] == "<class 'neiro.D_tanh'>":
            D = D_tanh(in_n, out_n)
        elif unserd['type'] == "<class 'neiro.D_Softmax'>":
            D = D_Softmax(in_n, out_n)

        D.W = W
        D.b = b

    return D

