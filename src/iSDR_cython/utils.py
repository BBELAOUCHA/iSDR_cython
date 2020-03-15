# Author: Brahim Belaoucha <>
import numpy as np
import os
import shutil
from scipy.sparse import coo_matrix
from joblib import dump, load
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
    row  = []
    col  = []
    data = []
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

    data_big = np.zeros(((n_t-m_p+1)*n_m, np.sum(idx)), dtype=np.float)
    for t in range(n_t-m_p+1):
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
    ---------- 
    rms: reconstruction error of MEG/EEG measurements
    n: number of active regions/sources
    l21s: l21 norm of the reconstructed sources
    l1a_l1norm: the l1norm of reconstructed MAR model
    l1a_l2norm: the l2norm of the reconstructed MAR model
    cl.l21_ratio: the l21 norm used in the regularization (not in %)
    """
    l21_reg, la, la_ratio, m_p, normalize, foldername, o_v, n_Astep, n_Sstep = args
    
    G = np.array(load(foldername+'/G.dat', mmap_mode='r'))
    M = np.array(load(foldername+'/M.dat', mmap_mode='r'))
    SC = np.array(load(foldername+'/SC.dat', mmap_mode='r')).astype(int)
    if os.path.isfile(foldername+'/A.dat'):
        A = np.array(load(foldername+'/A.dat', mmap_mode='r'))
    else:
        A = None
    m_p = int(float(m_p))
    cl = linear_model.iSDR(l21_ratio=float(l21_reg), la=[float(la), float(la_ratio)], old_version=int(o_v),
              normalize_Astep=int(n_Astep), normalize_Sstep=int(n_Sstep))
    cl.solver(G, M, SC, model_p=int(m_p), A=A, normalize=int(float(normalize)))
    R = cl.Scoef_.copy()
    n_c, n_t = M.shape
    rms = np.linalg.norm(M)**2
    n = 0
    l21s = 0
    l1a_l1norm,  l1a_l2norm= 0, 0
    if len(R) > 0 and len(cl.Acoef_) > 0 and len(cl.active_set[-1]) > 0:
        n = R.shape[0]
        Mx = np.dot(G[:, cl.active_set[-1]], R[:, m_p:])
        x = min(Mx.shape[1], M.shape[1])
        rms = np.linalg.norm(M[:, :x] - Mx[:, :x])**2
        for i in range(n):
            l21s += np.linalg.norm(R[i, :])
        l1a_l1norm = np.sum(np.abs(cl.Acoef_))
        l1a_l2norm = np.linalg.norm(cl.Acoef_)**2

    return rms/(2*n_t*n_c), n, l21s, l1a_l1norm, l1a_l2norm, cl.l21_ratio
