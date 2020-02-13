import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model._base  import LinearModel, _pre_fit, _preprocess_data
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import Lasso, Ridge
from . import cyISDR as cd_fast
from . import utils
class iSDR():
    def __init__(self, l21_ratio=1.0, la=0.0,  copy_X=True,
    max_iter=1000, tol=1e-6, random_state=None, selection='cyclic'):
        """Linear Model trained with the modified L21 prior as regularizer 
           (aka the Mulitasklasso) and ISDR
           this function implements what is called iSDR (S-step) optimization
            ||y - X_A w||^2_2 + l21_ratio * ||w||_21

        Parameters
        ----------
        l21_ratio: scaler, regularization parameter. Has to be > 0
        copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

        max_iter : int, default=1000
        The maximum number of iterations

        tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
        random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

        selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
        y : (n_samples, n_targets) which represents the EEG/MEG data
        """
        self.l21_ratio = l21_ratio
        self.la = la
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.random_state = random_state
        self.selection = selection

    def _fit(self, X, y, model_p=1):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : (n_samples, n_features) which represents the gain matrix

        y : (n_samples, n_targets) which represents the EEG/MEG data
        model_p: integer, the order of the assumed multivariate
                 autoregressive model 

        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples 
        
        Returns
        ----------
        self
        if you wanna get the brain activation please run .reconstruct
        """
        X = check_array(X, dtype=[np.float64, np.float32], order='F',
                        copy=self.copy_X and False)
        y = check_array(y, dtype=X.dtype.type, ensure_2d=False)

        if y.ndim == 1:
            raise ValueError("More than %s is needed" % model_p)

        n_samples, n_features = X.shape
        _, n_tasks = y.shape
        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))
        self.coef_ = np.zeros((n_tasks + model_p - 1) * n_features//model_p,
        dtype=X.dtype.type, order='F')

        self.coef_ = np.asfortranarray(self.coef_)  # coef contiguous in memory

        if self.selection not in ['random', 'cyclic']:
            raise ValueError("selection should be either random or cyclic.")
        random = (self.selection == 'random')
        self.coef_, self.dual_gap_, self.eps_, self.n_iter_ = \
            cd_fast.enet_coordinate_descent_iSDR(
                self.coef_, self.l21_ratio, X, y.reshape(-1, order='F'), model_p, self.max_iter, self.tol,
                check_random_state(self.random_state), random)
        self.coef_ = self.coef_.reshape((n_features//model_p, n_tasks + model_p - 1), order='F')
        return self
    
    def S_step(self, X, y):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : (n_samples, n_features) which represents the gain matrix

        y : (n_samples, n_targets) which represents the EEG/MEG data
        
        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples 
        
        Returns
        ----------
        self.coef_: (n_features, n_targets + model_p - 1)
        """
        self._fit(X, y, self.m_p)
        return self.coef_ 


    def A_step(self, X, y, SC, method):
        nbr_samples = y.shape[1]
        G, idx = utils.construct_J(X, SC, self.coef_[:, 2*self.m_p:-1], self.m_p)
        if method == 'lasso':
            model = Lasso(alpha=self.la, fit_intercept=False, copy_X=True)
        else:
            model = Ridge(alpha=self.la, fit_intercept=False, copy_X=True)
        #if self.m_p == 1:
        #    model.fit(G, y[:, 2*self.m_p:-1].reshape(-1, order='F'))
        #else:
        model.fit(G, y[:, 2*self.m_p+1:].reshape(-1, order='F'))
        A = np.zeros(SC.shape[0]*SC.shape[0]*self.m_p)
        A[idx] = model.coef_
        self.Acoef_ = A.reshape((X.shape[1], X.shape[1]*self.m_p), order='C')
        return self.Acoef_

    def solver(self, G, M, SC, nbr_iter=1, model_p=1, A=None, method='lasso'):
        if model_p < 1:
            raise ValueError("Wrong value for MVAR model =%s should be > 0."%model_p)
        self.n_sensor, self.n_source = G.shape 
        if A is None:
            A = np.random.normal(0, 1, (self.n_source, self.n_source*model_p))
            for i in range(self.n_source):
                A[i, :] /= np.linalg.norm(A[i, :])
        active_regions = np.arange(self.n_source)
        self.active_set = []
        self.dual_gap = []
        self.mxne_iter = []
        nbr_orig = G.shape[1]
        self.m_p = model_p
        for i in range(nbr_iter):
            print("Iteration %s: nbr of active sources %s"%(i, len(active_regions)))
            self.S_step(np.dot(G, A), M)
            idx = np.std(self.coef_, axis=1) > 0
            active_regions = active_regions[idx]
            self.active_set.append(active_regions)
            self.dual_gap.append(self.dual_gap_)
            self.mxne_iter.append(self.n_iter_)
            self.nbr_iter = i
            if len(active_regions) == A.shape[0] or (len(active_regions) == nbr_orig and i > 0):
                self.Acoef_ = A
                break
            else:
                G = G[:, idx]
                SC = SC[idx, ]
                SC = SC[:, idx]
                self.coef_ = self.coef_[idx, :]
            if np.sum(idx) == 0:
                #print("IDX ",np.sum(idx), np.std(self.coef_, axis=1))
                break
            
            A = self.A_step(G, M, SC, method)
            self.Acoef_ = A
            self.n_source = np.sum(idx)
    def reorder_A(self):
        A = self.Acoef_.copy()
        nx, ny = self.Acoef_.shape
        m_p = ny//nx
        for i in range(m_p):
            A[:, i*nx:(i+1)*nx] = self.Acoef_[:, (m_p - i - 1)*nx:(m_p - i)*nx]
        return A
    def get_phi(self):
        A = self.reorder_A()
        nx, ny = A.shape
        self.Phi = np.zeros((ny, ny))
        self.Phi[:nx, :] = A
        for i in range(ny - nx):
            self.Phi[nx+i, i] = 1
        self.eigs = np.linalg.eigvals(self.Phi)
        df = {'real':self.eigs.real, 'imag':np.imag(self.eigs),
        'eig': ['eig_%s'%i for i in range(len(self.eigs))]}
        self.eigs = pd.DataFrame(df).set_index('eig')
    
    def plot_effective(self, fmt='.3f', annot=True, cmap=None):
        A = self.Acoef_
        active = self.active_set[-1]
        ylabel = ['S%s'%s for s in active]
        xlabel = []
        for i in range(self.m_p):
            xlabel.extend(ylabel)
        sns.heatmap(A, annot=annot, fmt=fmt, xticklabels=xlabel,
        yticklabels=ylabel,cmap=cmap)
