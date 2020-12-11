import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from celluloid import Camera
from sklearn.datasets import make_blobs, make_moons


def visualize(X, labels_true, labels_pred, w):
    unique_labels = np.unique(labels_true)
    unique_colors = dict([(l, c) for l, c in zip(unique_labels, [[0.8, 0., 0.], [0., 0., 0.8]])])
    
    if w[1] == 0:
        plt.plot([X[:, 0].min(), X[:, 0].max()], w[0] / w[2])
    elif w[2] == 0:
        plt.plot(w[0] / w[1], [X[:, 1].min(), X[:, 1].max()])  
    else:
        mins, maxs = X.min(axis=0), X.max(axis=0)
        pts = [[mins[0], -mins[0] * w[1] / w[2] - w[0] / w[2]],
               [maxs[0], -maxs[0] * w[1] / w[2] - w[0] / w[2]],
               [-mins[1] * w[2] / w[1] - w[0] / w[1], mins[1]],
               [-maxs[1] * w[2] / w[1] - w[0] / w[1], maxs[1]]]
        pts = [(x, y) for x, y in pts if mins[0] <= x <= maxs[0] and mins[1] <= y <= maxs[1]]
        x, y = list(zip(*pts))
        plt.plot(x, y, c=(0.75, 0.75, 0.75), linestyle="--")
    
    colors_inner = [unique_colors[l] for l in labels_true]
    colors_outer = [unique_colors[l] for l in labels_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors_inner, edgecolors=colors_outer)
    

class Perceptron:
    def __init__(self, iterations=100, alpha=0.5):
        self.iterations = iterations
        self.alpha = alpha
        self.w = None
        
    def _predict(self, X):
        return (X @ self.w > 0).astype(np.int).ravel()
        
    def plot_learning(self, X, y, filename, interval=100):
        self.w = np.zeros((X.shape[1] + 1, 1))
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        
        fig = plt.figure(figsize=(9, 9))
        plt.axis('off')
        camera = Camera(fig)
        for epoch in range(self.iterations):        
            y_pred = self._predict(X_) 
            for i in range(X_.shape[0]):
                if y_pred[i] != y[i]:
                    update_sign = y[i] - y_pred[i]
                    
                    self.w = self.w + self.alpha * (update_sign * X_[i].reshape(-1, 1))
                    
            visualize(X, y, self._predict(X_), self.w)
            camera.snap()
        
        animation = camera.animate(interval=interval, blit=True)
        animation.save(
            f'{filename}.gif',
            dpi=100,
            savefig_kwargs={
                'frameon': False,
                'pad_inches': 'tight'
            }
        )
        
        
if __name__ == "__main__":
    X, y = make_blobs(1000, 2, centers=[[0, 0], [2.5, 2.5]], random_state=16)
    X_moons, y_moons = make_moons(1000, noise=0.075, random_state=16)

    Perceptron(iterations=30).plot_learning(X, y, filename="perceptron_bloobs")
    Perceptron(iterations=15, alpha=0.5).plot_learning(X_moons, y_moons, filename="perceptron_moons", interval=130)
