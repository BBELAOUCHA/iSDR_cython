from array import array
from random import random
import numpy as np
import iSDR_cython as ciSDR



def test_activeset():
    n_t = 200
    n_c, n_s = 3,3
    np.random.seed(40)
    G = np.abs(np.random.normal(0,1,(n_c, n_s)))
    J = np.zeros((n_s, n_t))
    J[:3, 0] = [10, 0.1, 0]
    A = np.array([[0.9,-0.4,0], [0.25, 0.97,0],[0.5,0,0.5]])
    for i in range(J.shape[-1]-1):
        J[:3, i+1] = np.dot(A, J[:3, i])
    SC = np.array([[1,1,1], [1,1,0],[1,0,1]])
    m_p = 1
    M = np.dot(G, J[:, m_p:])
    cl = ciSDR.linear_model.iSDR(l21_ratio=0.001, la=[1e-2, 0.5], verbose=0, old_version=0)
    cl.solver(G, M, SC, nbr_iter=10, model_p=1, A=np.eye(n_s), S_tol=1e-3)

    activeset = np.sort(cl.active_set[-1])
    t1 = len(activeset) == 3

    t2 = activeset[0]==0
    t3 = activeset[1]==1
    t4 = activeset[2]==2
    t5 = cl.Acoef_.shape[0] == 3
    t6 = cl.Scoef_.shape[0] == 3
    if not t1 or not t2 or not t3 or not t4 or not t5 or not t6:
        return False
    return True

