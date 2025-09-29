import numpy as np

n_rows = 100
n_cols = 200

X = np.random.randn(n_rows, n_cols)

U, D, Vt = np.linalg.svd(X, full_matrices=False)


def getur_data():
    return X

def getur_vars():
    return U, D, Vt
