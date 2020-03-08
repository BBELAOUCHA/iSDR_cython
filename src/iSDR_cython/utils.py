import numpy as np
import os
import shutil
from scipy.sparse import coo_matrix

def _constructJt(Jt):
    n_s, m_p = Jt.shape
    jx = np.zeros((n_s, n_s**2*m_p))
    for i in range(n_s):
        jx[i, i*(n_s*m_p):(i+1)*n_s*m_p] = Jt.reshape(-1, order='F')
    return jx
    
def construct_J(G, SC, J, m_p, old=False):
    n_s, n_t = J.shape
    n_m, n = G.shape
    if n != n_s:
        print('wrong dimenstion')
    row  = []
    col  = []
    data = []
    SCx = np.zeros((n_s, n_s*m_p))
    for i in range(m_p):
        SCx[:, i*n_s:(i+1)*n_s] = SC
    SCx = SCx.reshape(-1, order='C')
    idx = SCx > 0
    if old:
        n_m = n_s

    data_big = np.zeros(((n_t-m_p+1)*n_m, np.sum(idx)), dtype=np.float)
    for t in range(n_t-m_p+1):
        data_x = _constructJt(J[:, t:t+m_p])[:, idx]
        if not old:
            data_x = np.dot(G, data_x)
        data_big[t*n_m:(t+1)*n_m, :] = data_x
    return data_big, idx


def Compute_alpha_max(Ga, M, model_p):
    n_c, n_s = Ga.shape
    n_s = n_s//model_p
    GM = np.zeros((M.shape[1]*model_p, n_s))
    alpha_max = 0
    for i in range(n_s):
        for j in range(model_p):
            Ax = np.dot(M.T, Ga[:, j*n_s + i])
            GM[j*M.shape[1]:(j+1)*M.shape[1], i] = Ax
    alpha_max = np.sqrt(np.sum(np.power(GM, 2, GM), axis=0))
    return np.max(alpha_max)

def createfolder(filename):
    try:
        os.mkdir(filename)
    except OSError:
        print ("Creation of the directory %s failed" % filename)
    else:
        print ("Successfully created the directory %s " % filename)
    
def deletefolder(filename):
    try:
        shutil.rmtree(filename, ignore_errors=True)
    except OSError:
        print ("Deletion of the directory %s failed" % filename)
    else:
        print ("Successfully deleted the directory %s" % filename) 





def create_bigG(G, A, M):
    GA = np.dot(G, A)
    n_c , n_t  = M.shape
    _ , n_s = G.shape
    m_p = A.shape[1]//n_s
    row  = []
    col  = []
    data = []
    for i in range(n_t):
        for j in range(n_c):
            row.append([j+n_c*i]*n_s*m_p)
            col.append([n_s*i+k for k in range(n_s*m_p)])
            data.append(GA[j, :])
    data = np.array(data).reshape(-1)
    row = np.array(row).reshape(-1)
    col = np.array(col).reshape(-1)

    return coo_matrix((data, (row, col)), shape=(n_t*n_c, n_s*(n_t + m_p - 1)))