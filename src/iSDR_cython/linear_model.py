# Author: Brahim Belaoucha <>
import numpy as np
import pandas as pd
import seaborn as sns
import random
from scipy.sparse import linalg
import time

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import dump
import multiprocessing, uuid, warnings, os
from itertools import product
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import ElasticNet, LinearRegression
import traceback
from . import cyISDR as cd_fast
from . import utils
"""
=======================================================================
////===================================================================
///// \author Brahim Belaoucha  <br>
/////         Copyright (c) 2020 <br>
///// If you used this function, please cite one of the following:
//// (1) Brahim Belaoucha, Theodore Papadopoulo. Large brain effective
    network from EEG/MEG data and dMR information. PRNI 2017 - 7th
    International Workshop on Pattern Recognition in NeuroImaging,
    Jun 2017, Toronto, Canada.
//// (2) Brahim Belaoucha, Mouloud Kachouane, Theodore Papadopoulo.
    Multivariate Autoregressive Model Constrained by Anatomical
    Connectivity to Reconstruct Focal Sources. 2016 38th Annual
    International Conference of the IEEE Engineering in Medicine and
    Biology Society (EMBC), Aug 2016, Orlando, United States. 2016.
////
////
////===================================================================
=======================================================================
"""

class iSDR():
    def __init__(self, l21_ratio=1.0, la=[0.0, 1],  copy_X=True,
    max_iter=[10000, 2000], tol=1e-6, random_state=None, selection='cyclic',
    verbose=0, old_version=False, normalize_Sstep=False,
    normalize_Astep=False):
        """
        Linear Model trained with the modified L21 prior as regularizer
           (aka the Mulitasklasso) and iSDR
           this function implements what is called iSDR (S-step)
           optimization
            ||y - G A w||^2_2 + l21_ratio * ||w||_21 +
                       la[0] * la[1] * ||A||_1 +
                       0.5 * la[0] * (1 - la[1]) * ||A||^2_2
        Parameters
        ----------
        l21_ratio: scaler, regularization parameter. Has to be > 0 and
        < 100
        la: list of two elements
                       la[0] * la[1] * ||w||_1 +
                   0.5 * la[0] * (1 - la[1]) * ||w||^2_2
            la[0] = 0  no prior on A
            la[1] = 0 no L1 (lasso) prior only L2 (ridge)
            la[1] = 1 no L2 (ridge) prior only L1 (lasso)
        copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

        max_iter : int, default=[10000, 2000], 10000 The maximum number
        of iterations in the Sstep and 2000 for Astep, when reg
        parameter > 0

        tol : float, default=1e-6
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        random_state : int, RandomState instance, default=None
            The seed of the pseudo random number generator that selects
            a random feature to update. Used when
            ``selection`` == 'random'. Pass an int for reproducible
            output across multiple function calls.
            See :term:`Glossary <random_state>`.

        selection : {'cyclic', 'random'}, default='cyclic'
            If set to 'random', a random coefficient is updated every
            iteration rather than looping over features sequentially by
             default.

        verbose:
            int/bool flag used to give more details about iSDR procedure

        old_version: bool flag to use either eISDR(false) or iSDR (true)
         which can be found in the following papers:
            (1) Brahim Belaoucha, Theodore Papadopoulo. Large brain
            effective network from EEG/MEG data and dMR
            information. PRNI 2017 - 7th International Workshop on
            Pattern Recognition in NeuroImaging,
            Jun 2017, Toronto, Canada.

            (2) Brahim Belaoucha, Mouloud Kachouane,
            Theodore Papadopoulo. Multivariate Autoregressive Model
            Constrained by Anatomical Connectivity to Reconstruct
            Focal Sources. 2016 38th Annual International
            Conference of the IEEE Engineering in Medicine and
            Biology Society (EMBC), Aug 2016, Orlando,
            United States. 2016.

        normalize_Sstep: Normalize transfer function in the Sstep
        normalize_Astep: Normalize transfer function in the Astep

        Attributes
        ----------
        self.Acoef_: (n_active, n_active*model_p) estimated MVAR model
        self.Scoef_: (n_active, n_targets + model_p - 1) estimated brain
                    activity
        self.xscale: list of (n_active) at each iteration, weights that
                    is used to normalize in Sstep
        self.weights: list of weights used to normalize Acoef_ in
                    self.solver()
        self.active_set: list number of active regions/sources at each
                         iteration
        self.dual_gap: list contains the dual gap values of MxNE solver
                       at each iteration
        self.mxne_iter: list containing the number of iteration until
                        convergence for MxNE solver

        n_active == number of active sources/regions
        """
        self.l21_ratio = l21_ratio
        self.la = la
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.random_state = random_state
        self.selection = selection
        self.verbose = verbose
        self.old = old_version
        self.time = None
        self.normalize_Sstep = normalize_Sstep
        self.normalize_Astep = normalize_Astep
        if self.old:
            self.la = [0.0, 0.0]
        self.s_dualgap = []
        self.a_dualgap = []

    def _fit(self, X, y, model_p):
        """Fit model with coordinate descent.
            Sum_t=1^T(||y_t - G sum_i(A_i w_{t-i})||^2_2) +
                      l21_ratio * ||w||_21
        Parameters
        ----------
        X : (n_samples, n_features*model_p) which represents the
        gain matrix muliplied by MAR model of order model_p

        y : (n_samples, n_targets) which represents the EEG/MEG data
        model_p: integer, the order of the assumed multivariate
                 autoregressive model
        model_p: int MVAR model order

        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples

        Returns
        ----------
        self
        if you wanna get the brain activation please run .S_step
        """
        X = check_array(X, dtype=[np.float64, np.float32], order='F',
                        copy=self.copy_X and False)
        y = check_array(y, dtype=X.dtype.type, ensure_2d=False)

        if y.ndim == 1:
            raise ValueError("More than %s is needed" % model_p)

        n_samples, n_features = X.shape
        n_tasks = y.shape[1]
        self.xscale = np.ones((self.n_source, 1))
        if self.normalize_Sstep:
            for i in range(self.n_source):
                v = np.std(X[:, i::self.n_source])
                if v > 0:
                    self.xscale[i] = v
                    X[:, i::self.n_source] /= v

        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))
        self.Scoef_ = np.zeros((n_tasks + model_p - 1) * n_features//model_p,
        dtype=X.dtype.type, order='F')

        self.Scoef_ = np.asfortranarray(self.Scoef_)  # coef contiguous in memory

        if self.selection not in ['random', 'cyclic']:
            raise ValueError("selection should be either random or cyclic.")
        random = (self.selection == 'random')

        self.Scoef_, self.dual_gap_, self.eps_, self.n_iter_ = \
            cd_fast.enet_coordinate_descent_iSDR(
                self.Scoef_, self.l21_ratio, X, y.reshape(-1, order='F'),
                model_p, self.max_iter[0], self.tol,
                check_random_state(self.random_state), random,
                self.verbose)
        n, m = n_features//model_p, n_tasks + model_p - 1
        self.Scoef_ = self.Scoef_.reshape((n, m), order='F')
        if self.normalize_Sstep:
            self.Scoef_ = self.Scoef_/ self.xscale

        self.s_dualgap.append(self.dual_gap_)
        return self

    def S_step(self, X, y):
        """Fit model with coordinate descent.
            Sum_t=1^T(||y_t - G sum_i(A_i w_{t-i})||^2_2) +
                      l21_ratio * ||w||_21
        Parameters
        ----------
        X : (n_samples, n_features*self.m_p) which represents the gain matrix
             = GxA, A[A_self.m_p, .., A_1]
        y : (n_samples, n_targets) which represents the EEG/MEG data

        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples
        self.m_p = MAR model order
        Returns
        ----------
        self.Scoef_: (n_features, n_targets + model_p - 1)
        """
        self._fit(X, y, self.m_p)
        return self.Scoef_ 


    def A_step(self, X, y, SC, normalize):
        """Fit model of MVAR coefficients with either Lasso or Ridge.
        Sum_t=1^T(||y_t - G sum_i(A_i w_{t-i})||^2_2) +
                      la[0] * la[1] * ||A||_1 +
                      0.5 * la[0] * (1 - la[1]) * ||A||_2

        Parameters
        ----------
        X : (n_features, n_samples) which represents the brain
            activation

        y : (n_samples, n_targets) which represents the EEG/MEG data

        SC: (n_features, n_features), structural connectivity between
            brain sources/regions, only coefficients representing
            connected regions will be estimated

        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples

        Returns
        ----------
        self.Acoef_: (n_active, n_active*model_p)
        n_active == number of active sources/regions
        """
        nbr_samples = y.shape[1]
        z = self.Scoef_[:, 2*self.m_p:-self.m_p - 1]
        G, idx = utils.construct_J(X, SC, z, self.m_p, old=self.old)

        if self.old:
            yt = self.Scoef_[:, 3*self.m_p:-self.m_p]
            yt = yt.reshape(-1, order='F')
        else:
            yt = y[:, 2*self.m_p+1:-self.m_p]
            yt = yt.reshape(-1, order='F')
        if len(self.active_set) == 1:
            self.la_max = np.max(np.abs(np.dot(G.T, yt) / (G.shape[0]*self.la[1])))
            self.la[0] *= self.la_max*0.01

        if self.la[0] != 0:
            model = ElasticNet(alpha=self.la[0], l1_ratio=self.la[1],
            fit_intercept=False, copy_X=True,
            normalize=self.normalize_Astep,
            random_state=self.random_state,
            max_iter=self.max_iter[1])
        else:
            model = LinearRegression(fit_intercept=False,
            normalize=self.normalize_Astep, copy_X=True)

        model.fit(G, yt)
        self.GG = G.copy()
        self.YY = yt
        if self.la[0] != 0:
            self.a_dualgap.append(None)
        else:
            self.a_dualgap.append(None)
        A = np.zeros(SC.shape[0]*SC.shape[0]*self.m_p)
        A[idx] = model.coef_
        n = X.shape[1]
        self.Acoef_ = np.array(A.reshape((n, n*self.m_p), order='C'))
        self.weights = np.ones(self.Acoef_.shape[0])
        if normalize:
            for i in range(self.Acoef_.shape[0]):
                self.weights[i] = np.sum(np.abs(self.Acoef_[i, :]))
                if self.weights[i]>0:
                    self.Acoef_[i, :] = self.Acoef_[i, :]/self.weights[i]

        return self.Acoef_, self.weights

    def solver(self, G, M, SC, nbr_iter=50, model_p=1,
               A=None, normalize = False, S_tol=1e-3):
        """ ISDR solver that will iterate between the S-step and A-step
        This code solves the following optimization:

        argmin(w, A)
               ||y - X_A w||^2_2 + l21_ratio * ||w||_21 +
               la[0] * la[1] * ||A||_1 + la[0] * (1-la[1]) * ||A||_2

        S-step:
                argmin(w, A=constant)
               ||y - X_A w||^2_2 + l21_ratio * ||w||_21

        A-step:
                argmin(w=constant, A)
                ||y - X_A w||^2_2  + la[0] * la[1] * ||A||_1 +
                la[0] * (1-la[1]) * ||A||_2
        X_A = G x A, A=[A_p, .., A_1]
        Parameters
        ----------
        G : (n_samples, n_features) which represents the gain matrix

        M : (n_samples, n_targets) which represents the EEG/MEG data

        SC: (n_features, n_features), structural connectivity between
            brain sources/regions
        nbr_iter: int number of iteration between S and A step

        model_p: int MVAR model order (spatial and temporal effective
        connectivity between brain regions/sources)

        A: (n_features, n_features*model_p) an initial MVAR model,
        default None will generate an identity MVAR model at p
        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples

        normalize: bool: normalize A row way (divide by max value)
                   before next S step

        S_tol: tolerance value to stop iterating, small change in
                   the S-step
        Attributes
        ----------
        self.Acoef_: (n_active, n_active*model_p) estimated MVAR model
        self.Scoef_: (n_active, n_targets + model_p - 1) estimated brain
                    activity
        self.Scoef_: (n_active) weights that was used to normalize self.Acoef_
        self.active_set: list number of active regions/sources at each
                         iteration
        self.dual_gap: list containing the dual gap values of MxNE solver
                       at each iteration
        self.mxne_iter: list containing the number of iteration until
                        convergence for MxNE solver

        n_active == number of active sources/regions
        """
        self.time = - time.time()
        self.Morig = M.copy()
        self.Gorig = G.copy()
        Gtmp, Mtmp, SCtmp = G.copy(), M.copy(), SC.astype(int).copy()
        if model_p < 1:
            raise ValueError("Wrong value for MVAR model =%s should be > 0."%model_p)
        self.n_sensor, self.n_source = Gtmp.shape 
        if A is None:
            A = np.zeros((self.n_source, self.n_source*model_p))
            A[:, -self.n_source:] = np.eye(self.n_source)
        self.Acoef_ = A
        alpha_max = utils.Compute_alpha_max(np.dot(Gtmp, A), Mtmp, model_p)
        alpha_max *= 0.01;
        self.l21_ratio *= alpha_max;
        active_regions = np.arange(self.n_source)
        self.active_set = []
        self.dual_gap = []
        self.mxne_iter = []
        nbr_orig = Gtmp.shape[1]
        self.m_p = model_p
        S_tol *= np.linalg.norm(np.dot(np.linalg.pinv(Gtmp), Mtmp))/Mtmp.shape[1]
        previous_j = np.zeros((Gtmp.shape[1], Mtmp.shape[1] + model_p - 1))
        for i in range(nbr_iter):
            GAtmp = np.dot(Gtmp, A)
            self.S_step(GAtmp, Mtmp)
            idx = np.std(self.Scoef_, axis=1) > 0
            active_regions = active_regions[idx]
            self.active_set.append(active_regions)
            self.dual_gap.append(self.dual_gap_)
            self.mxne_iter.append(self.n_iter_)
            self.nbr_iter = i
            t = np.linalg.norm(previous_j - self.Scoef_)/Mtmp.shape[1]
            if self.verbose:
                print("Iteration %s: nbr of active sources %s"%(i+1, len(active_regions)))

            if (len(active_regions) == A.shape[0] and i>0) or (len(active_regions) == nbr_orig and i > 0):
                if self.verbose:
                    print('Stopped at iteration %s : Change in active set tol %.4f > %.4f  '%(i+1, len(active_regions) , A.shape[0]))
                self.time += time.time()
                break
            else:
                Gtmp = Gtmp[:, idx]
                SCtmp = SCtmp[idx, :]
                SCtmp = SCtmp[:, idx]
                self.Scoef_ = self.Scoef_[idx, :]
            if np.sum(idx) == 0:
                self.Acoef_ = []
                self.Scoef_ = []
                self.time += time.time()
                break

            previous_j = self.Scoef_.copy()
            A, weights = self.A_step(Gtmp, Mtmp, SCtmp, normalize=normalize)
            self.Acoef_ = A
            self.n_source = np.sum(idx)
            if t < S_tol:
                if self.verbose:
                    print('Stopped at iteration %s : Change in S-step tol %.4f > %.4f  '%(i+1, S_tol, t))
                self.time += time.time()
                break

    def _reorder_A(self):
        """ this function reorder the MVAR model matrix so that it can
            be used to construct PHI which can be used to compute
            dynamics (eigenvalues)
            
            before:
                 Acoef_ = [A_p, A_p-1, ......, A_1]
            after:
                 Acoef_ = [A_1, A_2, ......, A_p]
            where J_t = sum_i=1^p (A_i x J_{t-i})
            return:
                  A: (n_active, n_active*model_p) reordered MVAR model
        """
        if not hasattr(self, 'Acoef_'):
            if self.verbose:
                print('No MAR model is detected, run "solve" before this function')
            return None
        A = self.Acoef_.copy()
        nx, ny = self.Acoef_.shape
        m_p = ny//nx
        for i in range(m_p):
            A[:, i*nx:(i+1)*nx] = self.Acoef_[:, (m_p - i - 1)*nx:(m_p - i)*nx]
        return A

    def get_phi(self):
        """ this function constructs PHI companion matrix which controls
            the dynamics
            of brain activation
            Phi:
                  A_1  A_2 . . . .   A_p
                   I    0             0
                   0    I             0
                   .    .  . . .. . . .
                   
                   .    .
                   
                   0    0 . . . . 0   0
                   0    0 . . . . I   0
        Attributes
        ----------
           self.Phi: (n_active x model_p, n_active x model_p) model dynamics
           self.eigs: dataframe contains the eigenvalues of self.Phi
        """
        if not hasattr(self, 'Acoef_') or not hasattr(self, 'Scoef_'):
            if self.verbose:
                print('No MAR model is detected, run "solve" before this function')
            return None
        A = self._reorder_A()
        nx, ny = A.shape
        self.Phi = np.zeros((ny, ny))
        self.Phi[:nx, :] = A
        for i in range(ny - nx):
            self.Phi[nx+i, i] = 1
        self.eigs = np.linalg.eigvals(self.Phi)
        df = {'real':self.eigs.real, 'imag':np.imag(self.eigs), 'abs':np.abs(self.eigs),
        'eig': ['eig_%s'%i for i in range(len(self.eigs))]}
        self.eigs = pd.DataFrame(df).set_index('eig')

    def plot_effective(self, fmt='.3f', annot=True, cmap=None,
                       fig_size = 5, mask_flag=True):
        """Plotting function
        Plots the effective connectivity
        """
        if not hasattr(self, 'Acoef_'):
            if self.verbose:
                print('No MAR model is found, run ".solve" before this function')
            return None
        A = self.Acoef_
        active = self.active_set[-1]
        if len(active):
            ylabel = ['S%s'%s for s in active]
            xlabel = []
            for i in range(self.m_p):
                xlabel.extend(ylabel)
            mask = np.zeros_like(A)
            if mask_flag:
                mask[A==0] = True

            plt.figure(figsize=(fig_size*self.m_p, fig_size))
            g = sns.heatmap(A, annot=annot, fmt=fmt, xticklabels=xlabel,
            yticklabels=ylabel,cmap=cmap,mask=mask)
            plt.title('Effective connectivity p=%s'%self.m_p)
            n, m = A.shape
            idx = [i*n for i in range(m//n)]
            for i, k in enumerate(idx):
                r = Rectangle((k, 0), n - 0.01, n - 0.01, fill=False, lw=3)
                g.add_patch(r)
                plt.text(i * n + n // 2, n + 0.5, r'$A_{}$'.format(m // n - i), fontsize=14, weight="bold")

        else:
            if self.verbose:
                print('No active source is detected')


    def bias_correction(self):
        """
        this function is used to corrected the magnitude of the
        reconstructed brain activity by solving the following:
                  min sum_t ||y_t - Gr sum_i(Ar_i Jr_{t-i})||^2_2
        where Jr: is the magnitude of the reduced source space
              Gr: is the gain matrix correspending to the reduced source
                  space
              Ar_i: ith MAR model correspending to the reduced source
                  space
              y_t: EEG/MEG measurement at sample t
        """
        if not hasattr(self, 'Scoef_') or not hasattr(self, 'Acoef_') :
            if self.verbose:
                print('run ".solve" before this function')
            return None
        self.Jbias_corr = []
        active = self.active_set[-1]
        if len(active):
            Gbig = utils.create_bigG(self.Gorig[:, active], self.Acoef_, self.Morig)
            Z = linalg.lsmr(Gbig, self.Morig.reshape(-1, order='F'), atol=1e-12, btol=1e-12)
            self.Jbias_corr = Z[0].reshape((len(active), self.Morig.shape[1] + self.m_p - 1), order='F')






class iSDRcv():
    def __init__(self, model_p=[1], l21_values=[], la_values = [],
                 la_ratio_values=[1], normalize =[0],
                 max_run = None, seed=2020, parallel=True, tmp='/tmp',
                 verbose=False,old_version=False,
                 normalize_Astep=[0],normalize_Sstep=[0], cv=None):
        """
        This function is used to run cross-validation with grid run of
        all combination of parameters and hyper-parameters and return
        the cost function for all of them
        
        Parameters
        ----------
                model_p: list of tried MAR order
                l21_values: list of l21 norm reg parameters for Sstep
                la_values: list of l1 norm reg parameter for Astep
                la_ratio_values: list of l1/2 ratio for Astep
                normalize: can be [0, 1] to normalize or not A before 
                           Step
                max_run: used to limit the number of grid search run
                         default is None== all of grid will be run
                seed: random seed used to randomize the search grid,
                      will be used when max_run is used
                parallel: flag to run cv in parallel or not
                tmp: location to folder used to save intermediate result
                verbose: flag to print intermediate results or not
                old_version: flag to use or not old version of iSDR
                normalize_Astep: list of values to normalize or not the
                               transfer function in Astep
                normalize_Sstep: list of values to normalize or not the
                               transfer function in Sstep
        Attributes
        ----------
        self.results: DataFrame containing the cost function values and
                      parameters used to get it
        """
        foldername = tmp + '/tmp_' + str(uuid.uuid4())
        all_comb = []
        if not hasattr(model_p, "__len__"):
            model_p = [model_p]

        if not hasattr(l21_values, "__len__"):
            l21_values = [l21_values]

        if not hasattr(la_values, "__len__"):
            la_values = [la_values]

        if not hasattr(la_ratio_values, "__len__"):
            la_ratio_values = [la_ratio_values]

        if not hasattr(normalize, "__len__"):
            normalize = [normalize]

        if not hasattr(normalize_Astep, "__len__"):
            normalize_Astep = [normalize_Astep]

        if not hasattr(normalize_Sstep, "__len__"):
            normalize_Sstep = [normalize_Sstep]

        old_version = 1 if old_version else 0
        l21_values = np.unique(l21_values)
        la_values = np.unique(la_values)
        la_ratio_values = np.unique(la_ratio_values)
        model_p = np.unique(model_p)
        normalize = np.unique(normalize)
        if cv is None:
            prod = product(l21_values, la_values,la_ratio_values, model_p,
            normalize, [foldername], [old_version], normalize_Astep,
            normalize_Sstep)
        else:
            prod = product(l21_values, la_values,la_ratio_values, model_p,
            normalize, [foldername], [old_version], normalize_Astep,
            normalize_Sstep, [int(cv)], [seed])
        for i in prod:
            all_comb.append(i)
        all_comb = np.array(all_comb)
        if max_run is None or max_run > len(all_comb):
            max_run = len(all_comb)
        np.random.seed(seed)
        number_list = np.arange(len(all_comb))
        random.shuffle(number_list)
        number_list = number_list[:max_run]
        self.all_comb = all_comb[number_list]
        self.parallel = parallel
        self.foldername = foldername
        self.time = None
        self.cv = cv
        self.seed = seed
    def run(self, G, M, SC, A=None):
        self.time = -time.time()
        if not os.path.exists(self.foldername):
            utils.createfolder(self.foldername)
        dump(G, self.foldername+'/G.dat')
        dump(M, self.foldername+'/M.dat')
        dump(SC, self.foldername+'/SC.dat')
        if not A is None:
            dump(A, self.foldername+'/A.dat')
        #################################
        self.rms, self.nbr, self.l21a, self.l1a_l1norm, self.l1a_l2norm, self.l21_ratio = [], [], [], [], [], []
        df = {}
        test_data = []
        if self.cv is None:
            par_func = utils._run
        else:
            par_func = utils._runCV
            np.random.seed(int(self.seed))
            number_list = np.arange(M.shape[0])
            random.shuffle(number_list)
            combinations = []
            for ii in range(self.all_comb.shape[0]):
                n_block = M.shape[0]//int(self.cv)
                for i in range(int(self.cv)):
                    c = []
                    c.extend(self.all_comb[ii, :])
                    c.append(np.sort(number_list[i*n_block:n_block*(i+1)]))
                    c.extend([ii])
                    combinations.append(c)
            self.all_comb = combinations
        try:
            if self.parallel:
                nbr_cpu = multiprocessing.cpu_count() - 2
                if nbr_cpu < 1:
                    nbr_cpu = 1
                pool = multiprocessing.Pool(nbr_cpu)
                out = list(tqdm(pool.imap(par_func, self.all_comb), total=len(self.all_comb)))
                pool.terminate()
                if self.cv is None:
                    self.rms, self.nbr, self.l21a, self.l1a_l1norm, self.l1a_l2norm, self.l21_ratio = zip(*out)
                    runid = []
                else:
                    self.rms, self.nbr, self.l21a, self.l1a_l1norm, self.l1a_l2norm, self.l21_ratio, runid = zip(*out)
            else:
                runid = []
                for i in tqdm(range(len(self.all_comb))):
                    x = par_func(self.all_comb[i])
                    self.rms.append(x[0])
                    self.nbr.append(x[1])
                    self.l21a.append(x[2])
                    self.l1a_l1norm.append(x[3])
                    self.l1a_l2norm.append(x[4])
                    self.l21_ratio.append(x[5])
                    if not self.cv is None:
                        runid.append(x[6])
                    
            if len(self.rms):
                self.all_comb = np.array(self.all_comb)
                if not len(runid):
                    runid = np.arange(self.all_comb.shape[0])
                df = {
                    'rms':np.array(self.rms),
                    'nbr':np.array(self.nbr),
                    'S_prior':np.array(self.l21a),
                    'A_prior_l1':np.array(self.l1a_l1norm),
                    'A_prior_l2':np.array(self.l1a_l2norm),
                    'ls_reg':self.all_comb[:, 0].astype(float),
                    'la_reg_a':self.all_comb[:, 1].astype(float),
                    'la_reg_r': self.all_comb[:, 2].astype(float),
                    'p':self.all_comb[:, 3].astype(int),
                    'normalize':self.all_comb[:, 4].astype(int),
                    'l21_real':np.array(self.l21_ratio),
                    'normalize_Astep':self.all_comb[:, 7].astype(int),
                    'normalize_Sstep':self.all_comb[:, 8].astype(int),
                    'runidx': runid
                }
                df = pd.DataFrame(df)
                df = df.groupby('runidx').mean()
                df['Obj'] = df.rms + df.S_prior*df.l21_real +\
                df.A_prior_l1*df.la_reg_a*df.la_reg_r +\
                            df.A_prior_l2*df.la_reg_a*(0.5-0.5*df.la_reg_r)
                self.time += time.time()
        except Exception as e:
            print(e)
            print(traceback.print_exc())
            pass
        ###################################
        self._delete()
        self.results = df

    def save_results(self, folder, filename):
        if hasattr(self, 'results'):
            self.results.to_csv(folder+ '/' + filename + '.csv')
        else:
            print('No result found, please use #run# before saving results' )

    def _delete(self):
        utils.deletefolder(self.foldername)



class eiSDR_cv():
    """
    This function run grid search cross validation and return the optimal values
    :return:
    row of the dataframe correspending to the minimum eISDR functional values
    
    
    Parameters:
    -----------
        model_p: list of tried MAR order
        l21_values: list of l21 norm reg parameters for Sstep
        la_values: list of l1 norm reg parameter for Astep
        la_ratio_values: list of l1/2 ratio for Astep
        normalize: can be [0, 1] to normalize or not A before
                           Step
        max_run: used to limit the number of grid search run
                         default is None== all of grid will be run
        seed: random seed used to randomize the search grid,
                      will be used when max_run is used
        parallel: flag to run cv in parallel or not
        tmp: location to folder used to save intermediate result
        verbose: flag to print intermediate results or not
        old_version: flag to use or not old version of iSDR
        normalize_Astep: list of values to normalize or not the
                               transfer function in Astep
        normalize_Sstep: list of values to normalize or not the
                               transfer function in Sstep
    Attributes:
        self.opt = dataframe containing the smallest cost function values
                               
    """
    def __init__(self, l21_values=[1e-3], la_values=[1e-3],
    la_ratio_values=[1], normalize=[0], model_p=[1], verbose=False,
    max_run=None, old_version=False, parallel=True,
    normalize_Astep=[0], normalize_Sstep = [0]):

        if not hasattr(l21_values, "__len__"):
            l21_values = [l21_values]

        if not hasattr(la_values, "__len__"):
            la_values = [la_values]

        if not hasattr(la_ratio_values, "__len__"):
            la_ratio_values = [la_ratio_values]

        if not hasattr(normalize, "__len__"):
            normalize = [normalize]

        if not hasattr(model_p, "__len__"):
            model_p = [model_p]

        self.l21_values = np.unique(l21_values)
        self.la_values = np.unique(la_values)
        self.la_ratio_values = np.unique(la_ratio_values)
        self.normalize = normalize
        self.model_p = model_p
        self.verbose = verbose
        self.max_run = max_run
        self.old_version = 1 if old_version else 0
        self.parallel = parallel
        self.time = None
        self.normalize_Astep = normalize_Astep
        self.normalize_Sstep = normalize_Sstep
    def get_opt(self, G, M, SC):
        self.time = -time.time()
        cv = iSDRcv(l21_values=self.l21_values,
                    la_values=self.la_values,
                    la_ratio_values=self.la_ratio_values,
                    normalize=self.normalize,
                    model_p=self.model_p,
                    verbose=self.verbose,
                    max_run=self.max_run,
                    old_version=self.old_version,
                    parallel=self.parallel,
                    normalize_Astep = self.normalize_Astep,
                    normalize_Sstep = self.normalize_Sstep
                    )
        if self.verbose:
            print('Total number of combination %s'%len(cv.all_comb))

        cv.run(G, M, SC)
        self.time += time.time()
        self.opt = {}
        if hasattr(cv, 'results'):
            self.results = cv.results.copy()
            self.re = self.results[self.results.S_prior > 0]
            if self.re.shape[0] > 0:
                self.opt = self.re[self.re.Obj == self.re.Obj.min()]
                return self.opt
            else:
                if self.verbose:
                    print('No parameter combination resulted in active set')
        else:
            if self.verbose:
                print('can not find results, check iSDRcv')
        return []
        
    def save(self, filename):
        if hasattr(self, 'results'):
            self.results.to_csv(filename)
        else:
            print('can not find results, run get_opt before saving the results')
