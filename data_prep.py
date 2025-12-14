import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

FASHION_CLASSES = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
    4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

NUM_CLASSES = 10
NUM_FEATURES = 784


def to_categorical(y, num_classes):
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out


def load_and_preprocess_data():
    print("Loading Fashion-MNIST using NumPy only...")

    data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    X /= 255.0
    Y = to_categorical(y, NUM_CLASSES)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train.T, y_train.T, X_val.T, y_val.T, X_test.T, y_test.T


# =========================================================

class NeuralNetwork:

    def __init__(self, layers, hidden_activation='relu', lr=0.05):
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.lr = lr
        self.params = {}
        self.memory = {}
        self.init_weights()

    def init_weights(self):
        np.random.seed(42)
        for i in range(1, len(self.layers)):
            self.params['W'+str(i)] = np.random.randn(
                self.layers[i], self.layers[i-1]
            ) * np.sqrt(2. / self.layers[i-1])
            self.params['b'+str(i)] = np.zeros((self.layers[i], 1))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_backward(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_backward(self, a):
        return a * (1 - a)

    def softmax(self, z):
        z -= np.max(z, axis=0, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=0, keepdims=True)

    def forward_pass(self, X):
        self.memory['A0'] = X
        A = X

        for l in range(1, len(self.layers)-1):
            Z = self.params['W'+str(l)] @ A + self.params['b'+str(l)]
            A = self.relu(Z) if self.hidden_activation == 'relu' else self.sigmoid(Z)
            self.memory['Z'+str(l)] = Z
            self.memory['A'+str(l)] = A

        L = len(self.layers) - 1
        ZL = self.params['W'+str(L)] @ A + self.params['b'+str(L)]
        AL = self.softmax(ZL)

        self.memory['Z'+str(L)] = ZL
        self.memory['A'+str(L)] = AL
        return AL

    def calculate_loss(self, P, Y):
        return -np.mean(np.sum(Y * np.log(P + 1e-9), axis=0))

    def backward_pass(self, Y):
        grads = {}
        m = Y.shape[1]
        L = len(self.layers) - 1

        dZ = self.memory['A'+str(L)] - Y
        grads['dW'+str(L)] = dZ @ self.memory['A'+str(L-1)].T / m
        grads['db'+str(L)] = np.sum(dZ, axis=1, keepdims=True) / m

        for l in reversed(range(1, L)):
            dA = self.params['W'+str(l+1)].T @ dZ
            Z = self.memory['Z'+str(l)]

            if self.hidden_activation == 'relu':
                dZ = dA * self.relu_backward(Z)
            else:
                dZ = dA * self.sigmoid_backward(self.memory['A'+str(l)])

            grads['dW'+str(l)] = dZ @ self.memory['A'+str(l-1)].T / m
            grads['db'+str(l)] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def update_weights(self, grads):
        for l in range(1, len(self.layers)):
            self.params['W'+str(l)] -= self.lr * grads['dW'+str(l)]
            self.params['b'+str(l)] -= self.lr * grads['db'+str(l)]

    def get_accuracy(self, P, Y):
        return np.mean(np.argmax(P, axis=0) == np.argmax(Y, axis=0)) * 100

    def train(self, X, Y, Xv, Yv, epochs=200, batch_size=64):
        n = X.shape[1]

        for e in range(epochs):
            perm = np.random.permutation(n)
            X, Y = X[:, perm], Y[:, perm]

            for i in range(0, n, batch_size):
                xb = X[:, i:i+batch_size]
                yb = Y[:, i:i+batch_size]
                out = self.forward_pass(xb)
                grads = self.backward_pass(yb)
                self.update_weights(grads)

            self.lr *= 0.98

            if (e+1) % 10 == 0:
                val_out = self.forward_pass(Xv)
                print(f"Epoch {e+1}: Val Acc = {self.get_accuracy(val_out, Yv):.2f}%")

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.argmax(self.forward_pass(X), axis=0)

    def test_model(self, X, Y):
        out = self.forward_pass(X)
        print("\nTest Set Results:")
        print(f"Loss: {self.calculate_loss(out, Y):.4f}")
        print(f"Accuracy: {self.get_accuracy(out, Y):.2f}%")


# =========================================================

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    num_hidden = int(input("How many hidden layers? "))
    hidden = [int(input(f"Neurons in layer {i+1}: ")) for i in range(num_hidden)]
    act = input("Activation (relu/sigmoid): ").lower()

    model = NeuralNetwork([784] + hidden + [10], act)
    model.train(X_train, y_train, X_val, y_val)
    model.test_model(X_test, y_test)

    idx = np.random.randint(X_test.shape[1])
    sample = X_test[:, idx]
    true = np.argmax(y_test[:, idx])
    pred = model.predict(sample)[0]
    conf = model.forward_pass(sample.reshape(-1,1))[pred,0] * 100

    print(f"\nActual: {true} ({FASHION_CLASSES[true]})")
    print(f"Predicted: {pred} ({FASHION_CLASSES[pred]})")
    print(f"Confidence: {conf:.2f}%")
