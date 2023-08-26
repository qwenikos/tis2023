
from sklearn.utils import shuffle
import numpy as np

X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
y = np.array([0, 1, 2, 3, 4])
X, y = shuffle(X, y)
print(X)
print(y)