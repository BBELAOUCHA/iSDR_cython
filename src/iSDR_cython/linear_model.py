import numpy as np
from sklearn.linear_model._base  import LinearModel, _pre_fit, _preprocess_data
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import Lasso, Ridge
from . import cyISDR as cd_fast
from . import utils
class iSDR():
    def __init__(self, l21_ratio=1.0, la=0.0,  copy_X=True, max_iter=1000, tol=1e-6,
                 random_state=None, selection='cyclic'):
        """Linear Model trained with L21 prior as regularizer (aka the Mulitasklasso)
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
    
    def S_step(self, X, y, model_p=1):
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
        self._fit(X, y, model_p)
        return self.coef_ 


    def A_step(self, X, y, model_p, method):
        nbr_samples = y.shape[1]
        G = utils.construct_J(X, self.coef_, model_p)
        if method == 'lasso':
            model = Lasso(alpha=self.la, fit_intercept=False, copy_X=True)
        else:
            model = Ridge(alpha=self.la, fit_intercept=False, copy_X=True)
        model.fit(G, y.reshape(-1, order='F'))
        
        self.Acoef_ = model.coef_.reshape((X.shape[1], X.shape[1]*model_p), order='C')
        return self.Acoef_

    def solver(self, G, M, nbr_iter=1, model_p=1, A=None, method='lasso'):
        self.n_sensor, self.n_source = G.shape 
        if A is None:
            A = np.random.normal(0, 1, (self.n_source, self.n_source*model_p))
            for i in range(self.n_source):
                A[i, :] /= np.linalg.norm(A[i, :])
        active_regions = np.arange(self.n_source)
        self.active_set = []
        self.dual_gap = []
        self.mxne_iter = []
        for i in range(nbr_iter):
            self.S_step(np.dot(G, A), M, model_p)
            A = self.A_step(G, M, model_p, method)
            idx = np.std(self.coef_, axis=1) > 0
            active_regions = active_regions[idx]
            self.active_set.append(active_regions)
            self.dual_gap.append(self.dual_gap_)
            self.mxne_iter.append(self.n_iter_)
            self.nbr_iter = i
            if len(active_regions) == A.shape[0]:
                break
            else:
                G = G[:, idx]
                A = A[idx, :]
                ix = []
                for x in range(model_p):
                    ix.extend(idx)
                A = A[:, np.array(ix)]

