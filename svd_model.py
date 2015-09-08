import numpy as np
from scipy.sparse.linalg import svds
from scipy import sparse

np.set_printoptions(suppress=True, precision=4)

u2i_dense = np.array([[1,1,1,0,0,0,1,0],
                      [1,1,1,0,0,0,0,0],
                      [1,1,0,0,0,0,0,0],
                      [0,0,0,1,1,1,0,1],
                      [0,0,0,1,1,1,0,0],
                      [0,0,0,0,1,1,0,0]], dtype=np.float64)

user2item = sparse.csr_matrix(u2i_dense)

class SVD_model(object):
    def __init__(self, u2i, n_singular_values):
        self.u2i = u2i
        self.n_singular_values = n_singular_values
    def calc_score(self):
        u, s, vt = svds(self.u2i, k=self.n_singular_values)
        diag_s = np.zeros((self.n_singular_values, self.n_singular_values))
        np.fill_diagonal(diag_s, s)

        return np.dot(u, np.dot(diag_s, vt))

svd_model = SVD_model(user2item, n_singular_values=2)
scores = svd_model.calc_score()

# [[ 1.1579  1.1579  0.8645  0.      0.      0.      0.4615  0.    ]
#  [ 1.0112  1.0112  0.7549 -0.     -0.     -0.      0.403  -0.    ]
#  [ 0.7363  0.7363  0.5497  0.      0.      0.      0.2935  0.    ]
#  [-0.     -0.     -0.      0.8645  1.1579  1.1579  0.      0.4615]
#  [-0.     -0.     -0.      0.7549  1.0112  1.0112  0.      0.403 ]
#  [-0.     -0.     -0.      0.5497  0.7363  0.7363 -0.      0.2935]]
