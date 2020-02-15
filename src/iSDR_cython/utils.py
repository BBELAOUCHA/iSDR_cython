import numpy as np
import os
import shutil

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
    SCx = np.zeros((n_s, n_s*m_p))
    for i in range(m_p):
        SCx[:, i*n_s:(i+1)*n_s] = SC
    SCx = SCx.reshape(-1, order='C')
    idx = SCx > 0
    data_big = np.zeros(((n_t-m_p+1)*n_m, np.sum(idx)), dtype=np.float)
    for t in range(n_t-m_p+1):
        data_x = _constructJt(J[:, t:t+m_p])[:, idx]
        data_big[t*n_m:(t+1)*n_m, :] = np.dot(G, data_x)
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
        x = np.linalg.norm(GM[:, i])
        if x > alpha_max:
            alpha_max = x
    return alpha_max

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
