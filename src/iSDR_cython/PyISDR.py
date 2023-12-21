
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.sparse import csr_matrix, bmat
from scipy.linalg import norm
import sys
class reconstruct_activity():
    def __init__(self, nbr_sources, nbr_sensors, m_p, lambda_v, n_t, nbr_iterations=1000, tol=1e-4):
        self.nbr_sources = nbr_sources
        self.nbr_sensors = nbr_sensors
        self.m_p = m_p
        self.lambda_v = lambda_v
        self.n_t = n_t
        self.nbr_iterations = nbr_iterations
        self.tol = tol
        self.Jsdict = {}
        for s in range(self.nbr_sources):
            self.Jsdict[s] = np.zeros(self.n_t - 1, dtype=np.float64)

    def loop_over_sources(self, Jsdict, R):


        for s in range(self.nbr_sources):
            Jstmp = Jsdict[s]
            Gds = self.Gds_dict[s]
            #print(Jstmp.shape, self.mu_v.shape, Gds.shape, R.shape)
            Js = Jstmp + self.mu_v[s] * Gds.transpose()*R
            n = self.mu_v[s]*self.lambda_v
            if n < np.linalg.norm(Js):
                Js = Js*max(0, 1 - (n)/(max(n, np.linalg.norm(Js))))
            else:
                Js = Js*0
            R = R - Gds * (Js - Jsdict[s])
            Jsdict[s] = Js
        return Jsdict, R
    
    def run(self, M, G, A):
        big_matrix = create_bigG(G, A, M).tocsr()
        self.Gds_dict = {}
        for s in range(self.nbr_sources):
            self.Gds_dict[s] = big_matrix[:, s::self.nbr_sources]
            
        Gblock = np.dot(G, A)
        s = sys.getsizeof(big_matrix)
        Lip = np.zeros(G.shape[1])
        for i in range(self.nbr_sources):
            gi = Gblock[:, i::self.nbr_sources]
            Lip[i] = np.linalg.norm(np.dot(gi.T, gi), ord=2)

        self.mu_v = np.array([1.0/Lip[i] if Lip[i]!=0 else 0 for i in range(len(Lip))])
        R = M.T.reshape(-1)
        mse_list = []
        gap = []
        from tqdm import tqdm
        Mv = M.T.reshape(-1)
        self.tol = self.tol * np.linalg.norm(Mv) ** 2
        for j in tqdm(range(self.nbr_iterations)):
            self.Jsdict, R = self.loop_over_sources(self.Jsdict, R)
            Jreco = np.zeros((self.nbr_sources, self.n_t - 1), dtype=np.float64)

            for k, v in self.Jsdict.items():
                Jreco[k, :] = v
            mse = np.sqrt(np.mean(R**2))
            mse_list.append(mse)
            non_zero_sources = np.linalg.norm(Jreco, axis=1)
            listsources = [i for i in range(len(non_zero_sources)) if non_zero_sources[i] > 0]
            gap.append(self.dual_gap(big_matrix, Mv, Jreco))

            if len(mse_list) > 100:
                if (np.abs(gap[-1] - gap[-2])) <self.tol:
                    break
        return Jreco, R, mse_list, gap
    
    
    def dual_gap(self, G, Mv, J):
        R = (Mv - G*J.T.reshape(-1)).astype(np.float64)
        max_norm = G.transpose()*R
        max_norm = max_norm.reshape((self.nbr_sources, self.n_t - 1)).astype(np.float64)
        # print(np.sum(max_norm ** 2, axis=1).max())
        # dual_norm_XtA = np.max(np.sqrt(np.float64(np.sum(max_norm ** 2, axis=1)))) # inf max(sum(abs(x), axis=1))
        
        max_norm_squared = np.sum(max_norm ** 2, axis=1)
        max_norm_scaled = max_norm_squared / np.max(max_norm_squared)  # Scale values to a smaller range
        dual_norm_XtA = np.max(np.sqrt(np.float64(max_norm_scaled)))

        
        s = 0
        if dual_norm_XtA > 0:
            s = min(1, self.lambda_v/dual_norm_XtA)
        y = R*s
        R_norm = np.linalg.norm(R)
  
        if (dual_norm_XtA > self.lambda_v):
            const =  self.lambda_v / dual_norm_XtA
            A_norm = R_norm * const
            gap = 0.5 * (R_norm ** 2 + A_norm ** 2)
        else:
            const = 1.0
            gap = 0.5*R_norm ** 2
                    
        ry_sum = np.sum(R * Mv)

        # l21_norm = np.sqrt(np.sum(J ** 2, axis=0)).sum()
        
        
        def safe_sqrt(x):
            return np.sqrt(np.clip(x, a_min=0, a_max=None))

        J_squared = J ** 2
        l21_norm = safe_sqrt(np.sum(J_squared, axis=0)).sum()


        gap += self.lambda_v * l21_norm - const * ry_sum

        return gap
    
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


def Astep(G, M, SC, Jt, m_p):
    n_s, n_t = Jt.shape
    n_m, n = G.shape
    SCx = np.tile(SC, (m_p, 1)).T
    Jblock = np.zeros((n_s*m_p, n_t-m_p))
    for i in range(n_t - m_p):
        Jblock[:, i] = Jt[:, i:i+m_p].reshape(-1, order='F')
    block_active_l = []
    Y = np.empty(0)
    nbr_min = 0
    for j in range(n_s):
        ac = np.where(SCx[j] > 0)[0]
        block_active_l.append(coo_matrix(Jblock[ac, :]))
        Y = np.concatenate((Y, Jt[j, m_p:]))
    
    A = np.zeros((n_s, n_s*m_p))
    if n_s>0:
        block_active_matrix = block_diag(block_active_l)
        model = LinearRegression(fit_intercept=False, copy_X=True)
        model.fit(block_active_matrix.T, Y)
        idx = SCx > 0
        A[idx] = model.coef_
    return A


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
