# iSDR_cython: Cython implementation of eISDR_p (elasticnet Iterative Source and Dynamics Reconstruction)
[![Build Status](https://travis-ci.com/BBELAOUCHA/iSDR_cython.svg?branch=development)](https://travis-ci.com/BBELAOUCHA/iSDR_cython)
[![codecov](https://codecov.io/gh/BBELAOUCHA/iSDR_cython/branch/development/graph/badge.svg)](https://codecov.io/gh/BBELAOUCHA/iSDR_cython)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/555130d02e134e819bc599b93cfe53c9)](https://www.codacy.com/manual/BBELAOUCHA/iSDR_cython?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BBELAOUCHA/iSDR_cython&amp;utm_campaign=Badge_Grade)

A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) or order p on the source space. In this package, you can find two solvers eISDR and iSDR approaches.


     

iSDR_cython is a Cython package for solving the EEG/MEG inverse problem using structural/functional prior on the causality between brain regions/sources.
It implements the inverse solver iSDR (check Cite section for more details), It worth noting here that this implementation is a bit different from the original 
paper since now we are assuming a prior on the Multivariate Autoregressive Model coefficients

## Examples

### iSDR

<img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20U%28J%29%26%20%3D%5Cleft%20%5C%7CM_v%20-%20G_dJ_v%5C%7C%5Cright_2%5E2%20&plus;%20%5Clambda%20%5Cleft%5C%7CJ%5C%7C%5Cright_%7B21%7D%5E2%5C%5C%20V%28A%29%26%20%3D%5Cleft%20%5C%7CJ_v%20-%20J_JA_v%5C%7C%5Cright_2%5E2%5C%5C%20%5Cend%7Bmatrix%7D%5Cright." title=" \left\{\begin{matrix} U(J)& =\left    \|M_v - G_dJ_v\|\right_2^2 + \lambda \left\|J\|\right_{21}^2\\ V(A)& =\left    \|J_v - J_JA_v\|\right_2^2\\ \end{matrix}\right"/>

```python
from iSDR_cython import iSDR
model = iSDR(l21_reg=lambda_)
model.solver(G, M, SC, model_p=3)

```
### eiSDR

<img src="https://latex.codecogs.com/gif.latex?U%28J%2C%20A%29%20%3D%20%5Cleft%20%5C%7CM_v%20-%20G_dJ_v%5C%7C%5Cright_2%5E2%20&plus;%20%5Clambda%20%5Cleft%20%5C%7CJ%5C%7C%5Cright_%7B21%7D%20&plus;%20%5Calpha%5Cbeta%20%5Cleft%20%5C%7CS_cA_v%5C%7C%5Cright_%7B1%7D%20&plus;%20%5Calpha%281-%5Cbeta%29%20%5Cleft%20%5C%7CS_cA_v%5C%7C%5Cright_%7B2%7D" title=" U(J, A) = \left    \|M_v - G_dJ_v\|\right_2^2 + \lambda \left    \|J\|\right_{21}  + \alpha\beta \left    \|S_cA_v\|\right_{1} + \alpha(1-\beta) \left    \|S_cA_v\|\right_{2}^2"/>

Where: 

     * A_i: i=1,..,p are the matrices of the MVAR model of order p

     * M: EEG or/and MEG measurement

     * G: Lead field matrix which project brain activity into sensor space

     * J: Brain activity (distributed source model with fixed position), J_t at time t

     * lambda: regularization parameter that controls the sparsity of J )0, 100(
     
     * alpha, beta: regularizatiobn parameter that controls the sparsity/minimum norm of A
     
     * alpha values )0, 100(
     
     * beta values [0, 1]
     
     * A_v: vectorial form of A
     
     * S_c: matrix that select only anatomically connected regions/sources (ones and zeros elements).
     Ex if i, j are not connected S_c(i, j) = 0 and 1 otherwise
     
```python
from iSDR_cython import iSDR
model = eiSDR(l21_reg=lambda_, la=[alpha_, beta_])
model.solver(G, M, SC, model_p=3)

```

## Requirements
numpy~=1.6

scipy~=1.4 

scikit-learn~=0.22

joblib>=0.14.1

pandas>=1.0.1

cython>=0.29

seaborn

## Installation

pip3 install .

## Usage

For how to use this package to reconstruct the brain activation from EEG/MEG
check examples/iSDR_example

## Cite

(2) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective network from EEG/MEG data and dMR information. PRNI 2017 – 7th International Workshop on Pattern Recognition in NeuroImaging, Jun 2017, Toronto, Canada.

(3) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo. Multivariate Autoregressive Model Constrained by Anatomical Connectivity to Reconstruct Focal Sources. 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Aug 2016, Orlando, United States. 2016.
