# Author: Brahim Belaoucha <>
import numpy as np
import os
import shutil
import random
from scipy.sparse import coo_matrix
from joblib import load
from . import linear_model
def _constructJt(Jt):
    n_s, m_p = Jt.shape
    jx = np.zeros((n_s, n_s**2*m_p))
    for i in range(n_s):
        jx[i, i*(n_s*m_p):(i+1)*n_s*m_p] = Jt.reshape(-1, order='F')
    return jx
    
def construct_J(G, SC, J, m_p, old=False):
    """
    This function build the transfer function between the MAR model
    coefficients and the EEG/MEG data (or brain activation at t sample
    when using old version)
    Parameters:
    -----------
    G:  (nbr_channels, nbr_sourcesxm_p)The gain matrix
    SC: (nbr_sources, nbr_sources)The structural connectivity
    J:  (nbr_sources, nbr_samples + m_p -1)The brain activation in a time window
    m_p: The MAR model order
    old: flag to use old iSDR version of the new one
    
    Return:
    -----------
    data_big: matrix that transfer MAR model to M or J(old==True)
    old = False:
        M_v = data_big x A_v
    old = True:
        J_v = data_big x A_v
    idx:  The MAR coefficients that can be non-zero, obtained from SC
    """
    n_s, n_t = J.shape
    n_m, n = G.shape
    if n != n_s:
        print('wrong dimenstion')
    SCx = np.zeros((n_s, n_s*m_p))
    if SC.shape[0] == SC.shape[1]:
        for i in range(m_p):
            SCx[:, i*n_s:(i+1)*n_s] = SC
    else:
        SCx = SC.copy()
    SCx = SCx.reshape(-1, order='C')
    idx = SCx > 0
    if old:
        n_m = n_s
    m = n_t-m_p+1
    data_big = np.zeros((m*n_m, np.sum(idx)), dtype=np.float)
    for t in range(m):
        data_x = _constructJt(J[:, t:t+m_p])[:, idx]
        if not old:
            data_x = np.dot(G, data_x)
        data_big[t*n_m:(t+1)*n_m, :] = data_x
    return data_big, idx


def Compute_alpha_max(Ga, M, model_p):
    """
    This function compute the alpha max which is the smallest regularization
    parameter alpha that results to no active brain region when using
    l21 mixed norm (MxNE)
    
    Parameters:
    ----------
        Ga: GxA where G is the gain matrix and A is the MAR model
        M: The EEG or MEG measurements
        model_p: The order of the MAR model

    Return:
    ---------
    alpha_max: the regularization value that results to no active brain
              region
    """
    _, n_s = Ga.shape
    n_s = n_s//model_p
    GM = np.zeros((M.shape[1]*model_p, n_s))
    alpha_max = 0
    for i in range(n_s):
        for j in range(model_p):
            Ax = np.dot(M.T, Ga[:, j*n_s + i])
            GM[j*M.shape[1]:(j+1)*M.shape[1], i] = Ax
    alpha_max = np.sqrt(np.sum(np.power(GM, 2, GM), axis=0))
    return np.max(alpha_max)

def createfolder(foldername):
    """
    this function creates a folder
    """
    try:
        os.mkdir(foldername)
    except OSError:
        print ("Creation of the directory %s failed" % foldername)
    else:
        print ("Successfully created the directory %s " % foldername)

def deletefolder(foldername):
    """
    this function deletes a folder
    """
    try:
        shutil.rmtree(foldername, ignore_errors=True)
    except OSError:
        print ("Deletion of the directory %s failed" % foldername)
    else:
        print ("Successfully deleted the directory %s" % foldername)

def create_bigG(G, A, M):

    """
    This function is used when running bias correction and build the
    the following matrix:
            ---                                        ---
            | G_p, G_p-1, ...., G_1, 0, ...........   0  |
            |  0 ,  G_p, .......G_2,G_1,0 .......        |
            |  .                                         |
    bigG =  |  .                                         |
            |  .                                         |
            |  .                                         |
            |  0 , ......................., G_p, ..., G_1|
            ---                                        ---
    Where M_v = bigG x J_v
          M_v: is the vec form of EEG/MEG measurement M
          J_v: is the vec form of the reconstructed brain activation J
          G: the gain matrix
          A_i: the ith MAR model
          p: the order of MAR model
          G_i = GxA_i
    """
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

def _run(args):

    """
    The core function used to run the cross validation
    Parameters:
    ----------
    l21_reg:    regularization of l21 norm (Sstep) , 0<values<100
    la:         regularization of l1 and l2 norm (Astep)
    la_ratio:   regularization of l1 and l2 norm (Astep)
    m_p:        MAR model order
    normalize:  Normalize A_i's before Sstep or not
    foldername: Folder where to find:
                    M.dat: EEG/MEG data
                    G.dat: gain matix
                    SC.dat:    structral connectivity
                    A.dat: Initial MAR model, if not None is passed
    o_v:        flag to used old version iSDR or not
    n_Astep:    normalize transfer function in A step
    n_Sstep:    normalize transfer function in S step

    Return:
    -------
    rms: reconstruction error of MEG/EEG measurements
    n: number of active regions/sources
    l21s: l21 norm of the reconstructed sources
    l1a_l1norm: the l1norm of reconstructed MAR model
    l1a_l2norm: the l2norm of the reconstructed MAR model
    cl.l21_ratio: the l21 norm used in the regularization (not in %)
    """
    l21_reg, la, la_ratio, m_p, normalize, foldername, o_v, n_Astep, n_Sstep, includeMNE = args

    G = np.array(load(foldername+'/G.dat', mmap_mode='r'))
    M = np.array(load(foldername+'/M.dat', mmap_mode='r'))
    SC = np.array(load(foldername+'/SC.dat', mmap_mode='r')).astype(int)
    if os.path.isfile(foldername+'/A.dat'):
        A = np.array(load(foldername+'/A.dat', mmap_mode='r'))
    else:
        A = None
    m_p = int(float(m_p))
    if int(o_v):
        if not int(includeMNE):
            cl = linear_model.iSDR(l21_ratio=float(l21_reg), normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))

        else:
            cl = linear_model.iSDRols(l21_ratio=float(l21_reg), normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))


    else:
        if not int(includeMNE):
            cl = linear_model.eiSDR(l21_ratio=float(l21_reg), la=[float(la), float(la_ratio)],
                                normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))
        else:
            cl = linear_model.eiSDRols(l21_ratio=float(l21_reg), la=[float(la), float(la_ratio)],
                                normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))


    cl.solver(G, M, SC, model_p=int(m_p), A=A, normalize=int(float(normalize)))
    R = cl.Scoef_.copy()
    n_c, n_t = M.shape
    rms = np.linalg.norm(M)**2
    n = 0
    l21s = 0
    l1a_l1norm,  l1a_l2norm= 0, 0
    n_a_coef = 0
    stx = 0
    if len(R) > 0 and len(cl.Acoef_) > 0 and len(cl.active_set[-1]) > 0:
        Mx = np.zeros(M.shape)
        n_a_coef = np.sum(np.abs(cl.Acoef_) > 0)
        n = R.shape[0]
        Gx = np.dot(G[:, np.array(cl.active_set[-1])], cl.Acoef_)
        sxy = SC[np.array(cl.active_set[-1]), :]
        stx = np.sum(sxy[:, np.array(cl.active_set[-1])])*m_p
        if not int(includeMNE):
            for j in range(m_p):
                Mx += np.dot(Gx[:, j*n:n*(j+1)], R[:, j:Mx.shape[1]+j])
        else:
            Mx[:,:m_p] = np.dot(G[:, np.array(cl.active_set[-1])], R[:, :m_p])
            for j in range(m_p):
                Mx[:,m_p:] += np.dot(Gx[:, j*n:n*(j+1)], R[:, j:Mx[:,m_p:].shape[1]+j])
                

        x = min(Mx.shape[1], M.shape[1])
        rms = np.linalg.norm(M[:, :x] - Mx[:, :x])**2
        for i in range(n):
            l21s += np.linalg.norm(R[i, :])
        l1a_l1norm = np.sum(np.abs(cl.Acoef_))
        l1a_l2norm = np.linalg.norm(cl.Acoef_)**2

    return rms/(n_t*n_c), n, l21s, l1a_l1norm, l1a_l2norm, cl.l21_ratio, cl.la[0], n_a_coef, stx


def _runCV(args):
    """
    The core function used to run the cross validation
    Parameters:
    ----------
    l21_reg:    regularization of l21 norm (Sstep) , 0<values<100
    la:         regularization of l1 and l2 norm (Astep)
    la_ratio:   regularization of l1 and l2 norm (Astep)
    m_p:        MAR model order
    normalize:  Normalize A_i's before Sstep or not
    foldername: Folder where to find:
                    M.dat: EEG/MEG data
                    G.dat: gain matix
                    SC.dat:    structral connectivity
                    A.dat: Initial MAR model, if not None is passed
    o_v:        flag to used old version iSDR or not
    n_Astep:    normalize transfer function in A step
    n_Sstep:    normalize transfer function in S step

    Return:
    ------
    rms: reconstruction error of MEG/EEG measurements
    n: number of active regions/sources
    l21s: l21 norm of the reconstructed sources
    l1a_l1norm: the l1norm of reconstructed MAR model
    l1a_l2norm: the l2norm of the reconstructed MAR model
    cl.l21_ratio: the l21 norm used in the regularization (not in %)
    """
    l21_reg, la, la_ratio, m_p, normalize, foldername, o_v, n_Astep, n_Sstep, _, seed, includeMNE, test_data, run_ix  = args
    test_data = np.array(test_data)
    G = np.array(load(foldername+'/G.dat', mmap_mode='r'))
    M = np.array(load(foldername+'/M.dat', mmap_mode='r'))
    SC = np.array(load(foldername+'/SC.dat', mmap_mode='r')).astype(int)
    if os.path.isfile(foldername+'/A.dat'):
        A = np.array(load(foldername+'/A.dat', mmap_mode='r'))
    else:
        A = None
    m_p = int(float(m_p))
    n_c, n_t = M.shape
    np.random.seed(int(seed))
    number_list = np.arange(n_c)
    random.shuffle(number_list)
    rms = []
    nbr = 0
    l21s = 0
    l1a_l1norm = 0
    l1a_l2norm = 0
    l21_ratio = 0
    train_data = np.array([j for j in range(n_c) if not j in test_data])
    if int(o_v):
        if not int(includeMNE):
            cl = linear_model.iSDR(l21_ratio=float(l21_reg), normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))

        else:
            cl = linear_model.iSDRols(l21_ratio=float(l21_reg), normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))

    else:
        if not int(includeMNE):
            cl = linear_model.eiSDR(l21_ratio=float(l21_reg), la=[float(la), float(la_ratio)],
                                normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))
        else:
            cl = linear_model.eiSDRols(l21_ratio=float(l21_reg), la=[float(la), float(la_ratio)],
                                normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))
    gtmp = G[train_data, :]
    mtmp = M[train_data, :]
    cl.solver(gtmp, mtmp, SC, model_p=int(m_p), A=A, normalize=int(float(normalize)))
    R = cl.Scoef_.copy()
    l21_ratio = cl.l21_ratio
    n_a_coef = 0
    stx = 0
    if len(R) > 0 and len(cl.Acoef_) > 0 and len(cl.active_set[-1]) > 0:
        sxy = SC[np.array(cl.active_set[-1]), :]
        stx = np.sum(sxy[:, np.array(cl.active_set[-1])])*m_p
        n_a_coef = np.sum(np.abs(cl.Acoef_) > 0)
        n = R.shape[0]
        nbr = n
        if not includeMNE:
            Mx = M[test_data, :].copy()
        else:
            Mx = M[test_data, :-1].copy()
        ns = len(cl.active_set[-1])
        if not int(includeMNE):
            gtmp = G[test_data, :]
            gtmp = gtmp[:, np.array(cl.active_set[-1])]
            Mx[:, :m_p] = np.dot(gtmp, R[:, :m_p])
            for k in range(m_p, Mx.shape[1]):
                Mx[:, k] = 0
                for m in range(m_p):
                    x=np.dot(gtmp, cl.Acoef_[:, m*ns:ns*(m+1)])
                    Mx[:, k] += np.dot(x, R[:, k+m])
        else:
            gtmp = G[test_data, :]
            gtmp = gtmp[:, np.array(cl.active_set[-1])]
            Mx[:, :m_p] = np.dot(gtmp, R[:, :m_p])
            for k in range(m_p, Mx.shape[1]):
                Mx[:, k] = 0
                for m in range(m_p):
                    x=np.dot(gtmp, cl.Acoef_[:, m*ns:ns*(m+1)])
                    Mx[:, k] += np.dot(x, R[:, k+m-m_p])

        x = min(Mx.shape[1], M.shape[1])
        rms = np.linalg.norm(M[test_data, :x] - Mx[:, :x])**2
        l = 0
        for i in range(n):
            l += np.linalg.norm(R[i, :])
        l21s = l
        l1a_l1norm = np.sum(np.abs(cl.Acoef_))
        l1a_l2norm = np.linalg.norm(cl.Acoef_)**2
    else:
        rms = np.linalg.norm(M[test_data, :])**2
        nbr= 0
        l21s = 0
        l1a_l1norm = 0
        l1a_l2norm = 0
    rms = rms/(n_t*len(test_data))
    return rms, nbr, l21s, l1a_l1norm, l1a_l2norm, l21_ratio, cl.la[0], n_a_coef, run_ix, stx

def compute_criterion(M, results, criterion='bic', include_S=1):
    sigma2 = np.var(M)
    n_c, n_t = M.shape
    n_samples = n_c * n_t
    if criterion == 'aic':
        K = 2  # AIC
    elif criterion == 'bic':
        K = np.log(n_samples)  # BIC
    else:
        raise ValueError("Wrong value for criterion: %s" %criterion)
    mean_squared_error = results.rms
    df = results['nbr_coef'] + include_S * results['nbr']
    criterion_ = ((n_samples * mean_squared_error) / sigma2 + K * df)
    results[criterion] = criterion_ / n_samples
    return results
