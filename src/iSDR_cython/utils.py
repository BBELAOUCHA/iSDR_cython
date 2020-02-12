import numpy as np
from scipy.sparse import coo_matrix




def construct_G(X, nbr_samples, model_p):
    n_c, n_s = X.shape
    n_s = n_s//model_p
    data, row, col = [], [], []
    for t in range(nbr_samples):
        for i in range(n_c):
            data.extend(X[i, :])
            row.extend([t*n_c + i]*(n_s*model_p))
            col.extend([t*n_s + j for j in range(n_s*model_p)])
    G = coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(nbr_samples*n_c, n_s*(nbr_samples + model_p - 1)))
    return G




def _constructJt(Jt):
    n_s, m_p = Jt.shape
    jx = np.zeros((n_s, n_s**2*m_p))
    for i in range(n_s):
        jx[i, i*(n_s*m_p):(i+1)*n_s*m_p] = Jt.reshape(-1, order='F')
    return jx
    
def construct_J(G, J, m_p):
    n_s, n_t = J.shape
    n_m, n = G.shape
    if n != n_s:
        print('wrong dimenstion')
    row  = []
    col  = []
    data = []
    data_big = np.zeros(((n_t-m_p+1)*n_m, n_s**2*m_p), dtype=np.float)
    for t in range(n_t-m_p+1):
        data_x = _constructJt(J[:, t:t+m_p])
        data_big[t*n_m:(t+1)*n_m, :] = np.dot(G, data_x)
    return data_big
