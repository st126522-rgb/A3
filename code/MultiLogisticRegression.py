import numpy as np
import matplotlib.pyplot as plt
import time

class MultinomialLogisticRegression:
    def __init__(self, n_classes, n_features, lr=0.01, max_iter=2000, 
                 method='batch', batch_frac=0.3, l2_lambda=0.0, verbose=False):
        """
        n_classes: number of target classes
        n_features: number of columns in X (including intercept)
        method: 'batch', 'minibatch', 'sto' (stochastic)
        l2_lambda: L2 regularization strength
        """
        self.k = n_classes
        self.n = n_features
        self.lr = lr
        self.max_iter = max_iter
        self.method = method
        self.batch_frac = batch_frac
        self.l2 = l2_lambda
        self.verbose = verbose
        self.W = np.random.randn(self.n, self.k) * 0.01
        self.losses = []


    def softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def h_theta(self, X):
        return self.softmax(X @ self.W)

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X)
        eps = 1e-12
        ce_loss = -np.sum(Y * np.log(h + eps)) / m
        # L2 regularization (ignore intercept)
        W_no_intercept = self.W.copy()
        W_no_intercept[0, :] = 0.0
        l2_loss = 0.5 * (self.l2 / m) * np.sum(W_no_intercept**2)
        total_loss = ce_loss + l2_loss
        grad = (X.T @ (h - Y)) / m
        if self.l2 != 0:
            grad += (self.l2 / m) * W_no_intercept
        return total_loss, grad

   
    def fit(self, X, Y):
        m = X.shape[0]
        start_time = time.time()

        if self.method == "batch":
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.W -= self.lr * grad
                self.losses.append(loss)
                if self.verbose and i % 500 == 0:
                    print(f"[Batch] Iter {i}, Loss: {loss:.6f}")

        elif self.method == "minibatch":
            batch_size = max(1, int(self.batch_frac * m))
            for i in range(self.max_iter):
                ix = np.random.randint(0, m, batch_size)
                Xb, Yb = X[ix], Y[ix]
                loss, grad = self.gradient(Xb, Yb)
                self.W -= self.lr * grad
                self.losses.append(loss)
                if self.verbose and i % 500 == 0:
                    print(f"[MiniBatch] Iter {i}, Loss: {loss:.6f}")

        elif self.method == "sto":
            for i in range(self.max_iter):
                idx = np.random.randint(0, m)
                Xs = X[idx:idx+1]
                Ys = Y[idx:idx+1]
                loss, grad = self.gradient(Xs, Ys)
                self.W -= self.lr * grad
                self.losses.append(loss)
                if self.verbose and i % 1000 == 0:
                    print(f"[Stochastic] Iter {i}, Loss: {loss:.6f}")

        else:
            raise ValueError('Method must be "batch", "minibatch", or "sto".')

        print(f"Training completed in {time.time() - start_time:.2f}s")


    def predict_proba(self, X):
        return self.h_theta(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plot_losses(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Training Loss")
        plt.title("Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def classification_report_custom(y_true, y_pred):
    labels = np.unique(y_true)
    per_class = {}
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    # macro average
    macro = {k: np.mean([v[k] for v in per_class.values()]) for k in ["precision", "recall", "f1"]}

    # weighted average
    weights = [np.sum(y_true == label) for label in labels]
    total = np.sum(weights)
    weighted = {
        k: np.sum([v[k] * w for v, w in zip(per_class.values(), weights)]) / total
        for k in ["precision", "recall", "f1"]
    }

    return per_class, macro, weighted
