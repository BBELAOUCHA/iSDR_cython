from array import array
from random import random
import numpy as np
import iSDR_cython as ciSDR
import uuid
from iSDR_cython import linear_model

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

def test_norm():
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
    cl.solver(G, M, SC, nbr_iter=10, model_p=1, A=np.eye(n_s), S_tol=1e-3, normalize=1)

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


def test_createdelete():
    foldername = './tmp/tmp_' + str(uuid.uuid4())
    ciSDR.utils.createfolder(foldername)
    from os import path
    t1 = path.isdir(foldername)
    ciSDR.utils.deletefolder(foldername)
    t2 = path.isdir(foldername)
    if not t1 or t2:
        return False
    return True

def test_create_bigG():
    n_c, n_s, n_t = 2, 3 , 5
    G = np.ones((n_c, n_s))
    A = np.eye(n_s)
    M = np.zeros((n_c, n_t))
    m_p = A.shape[1]//n_s
    Gb = ciSDR.utils.create_bigG(G, A, M).toarray()
    t1 = Gb.shape[0] == n_t * n_c
    t2 = Gb.shape[1] == n_s * (n_t + m_p - 1)
    x = n_c*np.ones(Gb.shape[1])
    y = Gb.sum(axis=0) - n_c
    t3 = np.sum(y) == 0
    if not t1 or not t2 or not t3:
        return False
    return True

def test_getphi():
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
    cl.solver(G, M, SC, nbr_iter=10, model_p=1, A=np.eye(n_s), S_tol=1e-3, normalize=1)
    t1 = not hasattr(cl, 'eigs')
    cl.get_phi()
    t2 = hasattr(cl, 'eigs')
    if t1 and t2:
        return True
    return False

def test_cvfold():
    n_t = 200
    n_c, n_s = 3, 3
    np.random.seed(40)
    G = np.abs(np.random.normal(0, 1, (n_c, n_s)))
    J = np.zeros((n_s, n_t))
    J[:3, 0] = [10, 0.1, 0]
    A = np.array([[0.9, -0.4, 0], [0.25, 0.97, 0], [0.5, 0, 0.5]])
    for i in range(J.shape[-1] - 1):
        J[:3, i + 1] = np.dot(A, J[:3, i])
    SC = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    m_p = 1
    M = np.dot(G, J[:, m_p:])
    from iSDR_cython import linear_model

    clf = linear_model.iSDRcv(l21_values=[10 ** -i for i in range(-1, 3, 1)],
                              la_values=[10 ** -i for i in range(-1, 3, 1)], la_ratio_values=[1],
                              normalize=0,
                              model_p=[1],
                              old_version=False,
                              normalize_Astep=[0],
                              normalize_Sstep=[1],
                              cv=3,
                              parallel=True
                              )
    clf.run(G, M, SC)
    df = clf.results
    t1= np.abs(df.Obj.min() - 29.291233354057265) < 1e-3
    t2 = np.abs(df.rms.min() - 16.45599737341682) < 1e-3
    t3 = np.abs(df.nbr.min() - 1) < 1e-3
    if t1 and t2 and t3:
        return True
    return False

def test_cv():
    n_t = 200
    n_c, n_s = 3, 3
    np.random.seed(40)
    G = np.abs(np.random.normal(0, 1, (n_c, n_s)))
    J = np.zeros((n_s, n_t))
    J[:3, 0] = [10, 0.1, 0]
    A = np.array([[0.9, -0.4, 0], [0.25, 0.97, 0], [0.5, 0, 0.5]])
    for i in range(J.shape[-1] - 1):
        J[:3, i + 1] = np.dot(A, J[:3, i])
    SC = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    m_p = 1
    M = np.dot(G, J[:, m_p:])


    clf = linear_model.iSDRcv(l21_values=[10 ** -i for i in range(-1, 3, 1)],
                              la_values=[10 ** -i for i in range(-1, 3, 1)], la_ratio_values=[1, 0.5],
                              normalize=0,
                              model_p=[1],
                              old_version=False,
                              normalize_Astep=[0],
                              normalize_Sstep=[1],
                              cv=None,
                              parallel=True
                              )
    clf2 = linear_model.iSDRcv(l21_values=[10 ** -i for i in range(-1, 3, 1)],
                              la_values=[10 ** -i for i in range(-1, 3, 1)], la_ratio_values=[1, 0.5],
                              normalize=0,
                              model_p=[1],
                              old_version=False,
                              normalize_Astep=[0],
                              normalize_Sstep=[1],
                              cv=None,
                              parallel=False
                              )
    clf.run(G, M, SC)
    clf2.run(G, M, SC)
    df = clf.results
    df2 = clf2.results
    t1= np.abs(df.Obj.min() - 4.450684173236133) < 1e-3
    t2 = np.abs(df.rms.min() - 0.0005879948232802647) < 1e-3
    t3 = np.abs(df.nbr.min() - 1) < 1e-3
    t1_= np.abs(df2.Obj.min() - 4.450684173236133) < 1e-3
    t2_ = np.abs(df2.rms.min() - 0.0005879948232802647) < 1e-3
    t3_ = np.abs(df2.nbr.min() - 1) < 1e-3
    if t1 and t2 and t3 and t1_ and t2_ and t3_:
        return True
    return False

def test_seqcvfold():
    n_t = 200
    n_c, n_s = 3, 3
    np.random.seed(40)
    G = np.abs(np.random.normal(0, 1, (n_c, n_s)))
    J = np.zeros((n_s, n_t))
    J[:3, 0] = [10, 0.1, 0]
    A = np.array([[0.9, -0.4, 0], [0.25, 0.97, 0], [0.5, 0, 0.5]])
    for i in range(J.shape[-1] - 1):
        J[:3, i + 1] = np.dot(A, J[:3, i])
    SC = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    m_p = 1
    M = np.dot(G, J[:, m_p:])
    from iSDR_cython import linear_model

    clf = linear_model.iSDRcv(l21_values=[10 ** -i for i in range(-1, 3, 1)],
                              la_values=[10 ** -i for i in range(-1, 3, 1)], la_ratio_values=[1],
                              normalize=0,
                              model_p=[1],
                              old_version=False,
                              normalize_Astep=[0],
                              normalize_Sstep=[1],
                              cv=3,
                              parallel=False
                              )
    clf.run(G, M, SC)
    df = clf.results
    t1= np.abs(df.Obj.min() - 29.291233354057265) < 1e-3
    t2 = np.abs(df.rms.min() - 16.45599737341682) < 1e-3
    t3 = np.abs(df.nbr.min() - 1) < 1e-3
    if t1 and t2 and t3:
        return True
    return False

def test_eiSDR():
    n_t = 200
    n_c, n_s = 3, 3
    np.random.seed(40)
    G = np.abs(np.random.normal(0, 1, (n_c, n_s)))
    J = np.zeros((n_s, n_t))
    J[:3, 0] = [10, 0.1, 0]
    A = np.array([[0.9, -0.4, 0], [0.25, 0.97, 0], [0.5, 0, 0.5]])
    for i in range(J.shape[-1] - 1):
        J[:3, i + 1] = np.dot(A, J[:3, i])
    SC = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    m_p = 1
    M = np.dot(G, J[:, m_p:])
    clf = linear_model.eiSDR_cv(l21_values=[10 ** -i for i in range(-1, 3, 1)],
                                la_values=[10 ** -i for i in range(-1, 3, 1)], la_ratio_values=[1],
                                normalize=0,
                                model_p=[1],
                                old_version=False,
                                normalize_Astep=[0],
                                normalize_Sstep=[1],
                                parallel=False
                                )
    clf.get_opt(G, M, SC)
    t1 = np.abs(clf.opt.rms.values[0] - 0.0013515922705655975) < 1e-3
    t2 = np.abs(clf.opt.nbr.values[0] - 3)< 1e-3
    t3 = np.abs(clf.opt.Obj.values[0] -4.480759932047792)< 1e-3
    t4 = np.abs(clf.opt.S_prior.values[0] -117.644404)< 1e-3
    if t1 and t2 and t3 and t4:
        return True
    return False