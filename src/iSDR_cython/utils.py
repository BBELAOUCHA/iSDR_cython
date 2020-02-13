import numpy as np
from scipy.sparse import coo_matrix

def _constructJt(Jt):
    n_s, m_p = Jt.shape
    jx = np.zeros((n_s, n_s**2*m_p))
    for i in range(n_s):
        jx[i, i*(n_s*m_p):(i+1)*n_s*m_p] = Jt.reshape(-1, order='F')
    return jx
    
def construct_J(G, SC, J, m_p):
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
    SCx = np.zeros((n_s, n_s*m_p))
    for i in range(m_p):
        SCx[:, i*n_s:(i+1)*n_s] = SC
    SCx = SCx.reshape(-1, order='C')
    idx = SCx > 0
    return data_big[:, idx], idx
