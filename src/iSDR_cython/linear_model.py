import numpy as np
import pandas as pd
import seaborn as sns
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import dump, load
import multiprocessing, itertools, uuid, warnings, os
from itertools import product
from sklearn.linear_model._base  import LinearModel, _pre_fit, _preprocess_data
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import traceback
from . import cyISDR as cd_fast
from . import utils
"""
=======================================================================
////===================================================================
///// \author Brahim Belaoucha  <br>
/////         Copyright (c) 2020 <br>
///// If you used this function, please cite one of the following:
//// (1) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective
    network from EEG/MEG data and dMR information. PRNI 2017 – 7th 
    International Workshop on Pattern Recognition in NeuroImaging,
    Jun 2017, Toronto, Canada. 
//// (2) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo.
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
    max_iter=10000, tol=1e-6, random_state=None, selection='cyclic',
    verbose=0, old_version=False):
        """
        Linear Model trained with the modified L21 prior as regularizer 
           (aka the Mulitasklasso) and ISDR
           this function implements what is called iSDR (S-step)
           optimization
            ||y - X_A w||^2_2 + l21_ratio * ||w||_21

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

        max_iter : int, default=10000 The maximum number of iterations

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
         
            (1) Brahim Belaoucha, Théodore Papadopoulo. Large brain
            effective network from EEG/MEG data and dMR
            information. PRNI 2017 – 7th International Workshop on 
            Pattern Recognition in NeuroImaging,
            Jun 2017, Toronto, Canada.

            (2) Brahim Belaoucha, Mouloud Kachouane,
            Théodore Papadopoulo. Multivariate Autoregressive Model
            Constrained by Anatomical Connectivity to Reconstruct
            Focal Sources. 2016 38th Annual International
            Conference of the IEEE Engineering in Medicine and
            Biology Society (EMBC), Aug 2016, Orlando,
            United States. 2016.

        Attributes
        ----------
        self.Acoef_: (n_active, n_active*model_p) estimated MVAR model
        self.coef_: (n_active, n_targets + model_p - 1) estimated brain
                    activity
        self.coef_: (n_active) weights that was used to normalize Acoef_
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
        if self.old:
            self.la = [0.0, 0.0]

    def _fit(self, X, y, model_p):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : (n_samples, n_features) which represents the gain matrix

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
                self.coef_, self.l21_ratio, X, y.reshape(-1, order='F'),
                model_p, self.max_iter, self.tol,
                check_random_state(self.random_state), random,
                self.verbose)
        n, m = n_features//model_p, n_tasks + model_p - 1
        self.coef_ = self.coef_.reshape((n, m), order='F')
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


    def A_step(self, X, y, SC, normalize):
        """Fit model of MVAR coefficients with either Lasso or Ridge.

        Parameters
        ----------
        X : (n_samples, n_features) which represents the gain matrix

        y : (n_samples, n_targets) which represents the EEG/MEG data

        SC: (n_features, n_features), structural connectivity between 
            brain sources/regions
        n_samples == number of EEG/MEG sensors
        n_features == number of brain sources
        n_targets == number of data samples 
        
        Returns
        ----------
        self.Acoef_: (n_active, n_active*model_p)
        n_active == number of active sources/regions
        """
        nbr_samples = y.shape[1]
        z = self.coef_[:, 2*self.m_p:-self.m_p-1]
        G, idx = utils.construct_J(X, SC, z, self.m_p, self.old)
        model = ElasticNet(alpha=self.la[0], l1_ratio=self.la[1],
        fit_intercept=False, copy_X=True,
        random_state=self.random_state)
        #if self.m_p == 1:
        #    model.fit(G, y[:, 2*self.m_p:-1].reshape(-1, order='F'))
        #else:
        if self.old:
            yt = self.coef_[:, 3*self.m_p:-self.m_p].reshape(-1, order='F')
        else:
            yt = y[:, 3*self.m_p:3*self.m_p + z.shape[1] - self.m_p + 1]
            yt = yt.reshape(-1, order='F')

        model.fit(G, yt)
        A = np.zeros(SC.shape[0]*SC.shape[0]*self.m_p)
        A[idx] = model.coef_
        n = X.shape[1]
        self.Acoef_ = np.array(A.reshape((n, n*self.m_p), order='C'))
        self.weights = np.ones(self.Acoef_.shape[0])
        if normalize:
            for i in range(self.Acoef_.shape[0]):
                self.weights[i] = np.max(np.abs(self.Acoef_[i, :]))
                if self.weights[i]>0:
                    self.Acoef_[i, :] = self.Acoef_[i, :]/self.weights[i]
        return self.Acoef_, self.weights

    def solver(self, Gtmp, Mtmp, SCtmp, nbr_iter=1, model_p=1, A=None,
    normalize = False, S_tol=1e-3):
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
        self.coef_: (n_active, n_targets + model_p - 1) estimated brain 
                    activity
        self.coef_: (n_active) weights that was used to normalize self.Acoef_
        self.active_set: list number of active regions/sources at each
                         iteration
        self.dual_gap: list containing the dual gap values of MxNE solver
                       at each iteration
        self.mxne_iter: list containing the number of iteration until
                        convergence for MxNE solver 
        
        n_active == number of active sources/regions
        """
        G, M, SC = Gtmp.copy(), Mtmp.copy(), SCtmp.astype(int).copy()
        if model_p < 1:
            raise ValueError("Wrong value for MVAR model =%s should be > 0."%model_p)
        self.n_sensor, self.n_source = G.shape 
        if A is None:
            A = np.zeros((self.n_source, self.n_source*model_p))
            A[:, -self.n_source:] = np.eye(self.n_source)
        self.Acoef_ = A
        alpha_max = utils.Compute_alpha_max(np.dot(G, A), M, model_p)
        alpha_max *= 0.01;
        self.l21_ratio *= alpha_max;
        active_regions = np.arange(self.n_source)
        self.active_set = []
        self.dual_gap = []
        self.mxne_iter = []
        nbr_orig = G.shape[1]
        self.m_p = model_p
        S_tol *= np.linalg.norm(np.dot(np.linalg.pinv(G), M))/M.shape[1]
        previous_j = np.zeros((G.shape[1], M.shape[1] + model_p - 1))
        for i in range(nbr_iter):
            if self.verbose:
                print("Iteration %s: nbr of active sources %s"%(i, len(active_regions)))
            self.S_step(np.dot(G, A), M)
            idx = np.std(self.coef_, axis=1) > 0
            active_regions = active_regions[idx]
            self.active_set.append(active_regions)
            self.dual_gap.append(self.dual_gap_)
            self.mxne_iter.append(self.n_iter_)
            self.nbr_iter = i
            t = np.linalg.norm(previous_j - self.coef_)/M.shape[1]

            if (len(active_regions) == A.shape[0] and i>0) or (len(active_regions) == nbr_orig and i > 0):
                self.Acoef_ = A
                if self.verbose:
                    print('Stopped at iteration %s : Change in active set tol %.4f > %.4f  '%(i, len(active_regions) , A.shape[0]))
                break
            else:
                G = G[:, idx]
                SC = SC[idx, ]
                SC = SC[:, idx]
                self.coef_ = self.coef_[idx, :]
            if np.sum(idx) == 0:
                self.Acoef_ = []
                self.coef_ = []
                break

            previous_j = self.coef_.copy()
            A, weights = self.A_step(G, M, SC, normalize=normalize)
            self.Acoef_ = A
            self.n_source = np.sum(idx)
            if t < S_tol:
                if self.verbose:
                    print('Stopped at iteration %s : Change in S-step tol %.4f > %.4f  '%(i, S_tol, t))
                break

    def _reorder_A(self):
        """ this function reorder the MVAR model matrix so that it can
            be used to construct PHI which can be used to compute 
            dynamics (eigenvalues)
            
            before:
                 Acoef_ = [A_-p, A_-p+1, ......, A_0]
            after:
                 Acoef_ = [A_0, A_-1, ......, A_-p]
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
                  A_0  A_-1 . . . . A_-p
                   I    0             0
                   0    I             0
                   .    .  . . .. . . .
                   
                   .    .
                   
                   0    0 . . . . 0   0
                   0    0 . . . . I   0
        Attributes
        ----------
           self.Phi: (n_active*model_p, n_active*model_p) model dynamics
           self.eigs: dataframe contains the eigenvalues of self.Phi
        """
        if not hasattr(self, 'Acoef_'):
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
        df = {'real':self.eigs.real, 'imag':np.imag(self.eigs),
        'eig': ['eig_%s'%i for i in range(len(self.eigs))]}
        self.eigs = pd.DataFrame(df).set_index('eig')

    def plot_effective(self, fmt='.3f', annot=True, cmap=None, fig_size = 5):
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
            plt.figure(figsize=(fig_size*self.m_p, fig_size))
            sns.heatmap(A, annot=annot, fmt=fmt, xticklabels=xlabel,
            yticklabels=ylabel,cmap=cmap)
            plt.title('Effective connectivity p=%s'%self.m_p)
        else:
            if self.verbose:
                print('No active source are deteceted')


def _run(args):
    l21_reg, la, la_ratio, m_p, normalize, foldername, old_version = args
    
    G = np.array(load(foldername+'/G.dat', mmap_mode='r'))
    M = np.array(load(foldername+'/M.dat', mmap_mode='r'))
    SC = np.array(load(foldername+'/SC.dat', mmap_mode='r')).astype(int)
    m_p = int(float(m_p))
    cl = iSDR(l21_ratio=float(l21_reg), la=[float(la), float(la_ratio)], old_version=int(old_version))
    cl.solver(G, M, SC, nbr_iter=100, model_p=int(m_p), A=None, normalize=int(float(normalize)))
    R = cl.coef_.copy()
    n_c, n_t = M.shape
    rms = np.linalg.norm(M)**2
    n = 0
    l21s = 0
    l1a_l1norm,  l1a_l2norm= 0, 0
    if len(R) > 0 and len(cl.Acoef_) > 0 and len(cl.active_set[-1]) > 0:
        n = R.shape[0]
        n_c, n_t = R.shape
        for i in range(2*m_p, n_t):
            R[:, i] = 0
            for j in range(m_p):
                R[:, i] += np.dot(cl.Acoef_[:, j*n:(j+1)*n], R[:, i - m_p + j])
        #R = cl.coef_.copy()
        Mx = np.dot(G[:, cl.active_set[-1]], R[:, m_p:])
        x = min(Mx.shape[1], M.shape[1])
        rms = np.linalg.norm(M[:, :x]-Mx[:, :x])**2
        for i in range(n):
            l21s += np.linalg.norm(R[i, :])
        l1a_l1norm = np.sum(np.abs(cl.Acoef_))
        l1a_l2norm = np.linalg.norm(cl.Acoef_)**2

    return rms/(2*n_t*n_c), n, l21s, l1a_l1norm, l1a_l2norm, cl.l21_ratio


class iSDRcv():
    def __init__(self, model_p=[1], l21_values=[], la_values = [], la_ratio_values=[1], normalize =[1],
                 max_run = None, seed=2020, parallel=True, tmp='/tmp', verbose=False, old_version=False):
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
            
        for i in product(np.unique(l21_values), np.unique(la_values),
        np.unique(la_ratio_values), np.unique(model_p),
        np.unique(normalize), [foldername], [old_version]):
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

    def run(self, G, M, SC):
        if not os.path.exists(self.foldername):
            utils.createfolder(self.foldername)
        dump(G, self.foldername+'/G.dat')
        dump(M, self.foldername+'/M.dat')
        dump(SC, self.foldername+'/SC.dat')
        #################################
        self.rms, self.nbr, self.l21a, self.l1a_l1norm, self.l1a_l2norm, self.l21_ratio = [], [], [], [], [], []
        df = {}
        try:
            if self.parallel:
                pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
                out = list(tqdm(pool.imap(_run, self.all_comb), total=len(self.all_comb)))
                pool.terminate()
                self.rms, self.nbr, self.l21a, self.l1a_l1norm, self.l1a_l2norm, self.l21_ratio = zip(*out)
            else:
                for i in tqdm(range(len(self.all_comb))):
                    x = _run(self.all_comb[i])
                    self.rms.append(x[0])
                    self.nbr.append(x[1])
                    self.l21a.append(x[2])
                    self.l1a_l1norm.append(x[3])
                    self.l1a_l2norm.append(x[4])
                    self.l21_ratio.append(x[5])

            if len(self.rms):
                self.all_comb = np.array(self.all_comb)
                df = {'rms':np.array(self.rms), 'nbr':np.array(self.nbr),
                'S_prior':np.array(self.l21a), 'A_prior_l1':np.array(self.l1a_l1norm),
                'A_prior_l2':np.array(self.l1a_l2norm),
                'ls_reg':self.all_comb[:, 0].astype(float),
                'la_reg_a':self.all_comb[:, 1].astype(float),
                'la_reg_r': self.all_comb[:, 2].astype(float),
                'p':self.all_comb[:, 3].astype(int),
                'normalize':self.all_comb[:, 4].astype(int),
                'l21_real':np.array(self.l21_ratio)
                }
                df = pd.DataFrame(df)
                df['Obj'] = df.rms + df.S_prior*df.l21_real +\
                df.A_prior_l1*df.la_reg_a*df.la_reg_r +\
                            df.A_prior_l2*df.la_reg_a*(0.5-0.5*df.la_reg_r)
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
    """
    def __init__(self, l21_values=[1e-3], la_values=[1e-3],
    la_ratio_values=[0,1], normalize=[0], model_p=[1], verbose=False,
    max_run=None, old_version=False, parallel=True):

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
        self.old_version = old_version
        self.parallel = parallel

    def get_opt(self, G, M, SC):
        cv = iSDRcv(l21_values=self.l21_values,
                    la_values=self.la_values,
                    la_ratio_values=self.la_ratio_values,
                    normalize=self.normalize,
                    model_p=self.model_p,
                    verbose=self.verbose,
                    max_run=self.max_run,
                    old_version=self.old_version,
                    parallel=self.parallel)
        if self.verbose:
            print('Total number of combination %s'%len(cv.all_comb))

        cv.run(G, M, SC)
        if hasattr(cv, 'results'):
            self.results = cv.results.copy()
            self.re = self.results[self.results.S_prior > 0]
            if self.re.shape[0] > 0:
                return self.re[self.re.Obj == self.re.Obj.min()]
            else:
                if self.verbose:
                    print('No parameter combination resulted in active set')
        else:
            if self.verbose:
                print('can not find results, check iSDRcv')
        return []
        
    def save(self, filename):
        if hasattr(self, 'results'):
            self.results.to_csv(filename + '')
        else:
            print('can not find results, run get_opt before saving the results')
