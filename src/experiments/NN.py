import numpy as np


class Scaler:
    def __init__(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

    def transform(self, X):
        sh = (X.shape[0], 1)
        X = X - np.tile(self.means, sh)
        for i, std in enumerate(self.stds):
            if std != 0.0:
                X[:, i] /= std
        return X


class NN:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return z * (z > 0)

    def forward_activation(self, a_prev, w, b, activation):
        z = np.dot(w, a_prev) + b
        if activation == "sigmoid":
            a = self.sigmoid(z)
        elif activation == "relu":
            a = self.relu(z)
        return a, ((a_prev, w, b), z)

    def backward_activation(self, da, cache, activation):
        (a_prev, w, _), activation_cache = cache
        if activation == "sigmoid":
            dz = da * self.sigmoid(activation_cache) * \
                 (1 - self.sigmoid(activation_cache))
        elif activation == "relu":
            dz = da * (activation_cache > 0)
        sh = a_prev.shape[1]
        return (np.dot(w.T, dz),
                np.dot(dz, a_prev.T) / sh,
                np.sum(dz, axis=1, keepdims=True) / sh)

    def forward(self, X):
        caches, a = [], X
        for i in range(len(self.w) - 1):
            a_prev = a
            a, cache = self.forward_activation(a_prev, self.w[i], self.b[i], "relu")
            caches.append(cache)
        al, cache = self.forward_activation(a, self.w[-1], self.b[-1], "sigmoid")
        caches.append(cache)
        return al, caches

    def train(self, X_train, Y_train, layer_dims=None, learning_rate=1, iterations=750):
        sample_size, height, width = X_train.shape
        X_train = X_train.reshape((sample_size, height * width))
        self.sc = Scaler(X_train)
        X_train = self.sc.transform(X_train).T
        Y_train = Y_train.reshape(Y_train.shape[0], 1).T
        Y_train_ = np.zeros((10, Y_train.shape[1]))
        for i in range(Y_train.shape[1]):
            Y_train_[Y_train[0, i], i] = 1
        Y_train = Y_train_

        np.random.seed(1337)
        if layer_dims is None:
            layer_dims = [X_train.shape[0], _, 10]
            layer_dims[1] = np.round(np.sqrt(layer_dims[0] * layer_dims[2])).astype(int)
        sz = len(layer_dims) - 1
        da, dw, db = [[None] * sz for _ in range(3)]
        self.w = [np.random.randn(layer_dims[i + 1], layer_dims[i]) / 100 for i in range(sz)]
        self.b = [np.zeros((layer_dims[i + 1], 1)) for i in range(sz)]
        for i in range(iterations):
            al, caches = self.forward(X_train)
            da[-1], dw[-1], db[-1] = self.backward_activation(np.divide(1 - Y_train, 1 - al) - np.divide(Y_train, al),
                                                              caches[-1], "sigmoid")
            for j in range(sz - 2, -1, -1):
                da[j], dw[j], db[j] = self.backward_activation(da[j + 1], caches[j], "relu")
            for j in range(len(self.w)):
                self.w[j] -= learning_rate * dw[j]
                self.b[j] -= learning_rate * db[j]
        return self

    def predict(self, X_test):
        sample_size, height, width = X_test.shape
        X_test = X_test.reshape((sample_size, height * width))
        return np.argmax(self.forward(self.sc.transform(X_test).T)[0], axis=0)
