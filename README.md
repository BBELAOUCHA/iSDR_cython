# iSDR_cython: Cython implementation of eISDR_p (elasticnet Iterative Source and Dynamics Reconstruction)
[![Build Status](https://travis-ci.org/BBELAOUCHA/iSDR_cython.svg?branch=master)](https://travis-ci.org/BBELAOUCHA/iSDR_cython)
[![codecov](https://codecov.io/gh/BBELAOUCHA/iSDR_cython/branch/development/graph/badge.svg)](https://codecov.io/gh/BBELAOUCHA/iSDR_cython)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d82ad6541e214a04b3fc5f142cfa9cbf)](https://www.codacy.com/app/BBELAOUCHA/iSDR_cython?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BBELAOUCHA/iSDR_cython&amp;utm_campaign=Badge_Grade)
A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) on the source space. In this package, you can find two solvers eISDR and iSDR approaches. 


<img src="https://latex.codecogs.com/gif.latex?U%28J%2C%20A%29%20%3D%20%5Cleft%20%5C%7CM_v%20-%20G_dJ_v%5C%7C%5Cright_2%5E2%20&plus;%20%5Clambda%20%5Cleft%20%5C%7CJ%5C%7C%5Cright_%7B21%7D%20&plus;%20%5Calpha%20%5Cleft%20%5C%7CS_cA_v%5C%7C%5Cright_%7B1%7D%20&plus;%20%5Cbeta%20%5Cleft%20%5C%7CS_cA_v%5C%7C%5Cright_%7B2%7D" title=" U(J, A) = \left    \|M_v - G_dJ_v\|\right_2^2 + \lambda \left    \|J\|\right_{21}  + \alpha \left    \|S_cA_v\|\right_{1} + \beta \left    \|S_cA_v\|\right_{2}"/>

Where: 

     * A_i: i=1,..,p are the matrices of the MVAR model of order p

     * M: EEG or/and MEG measurement

     * G: Lead field matrix which project brain activity into sensor space

     * J: Brain activity (distributed source model with fixed position), J_t at time t

     * lambda: regularization parameter that controls the sparsity of J )0, 100(
     
     * alpha, beta: regularizatiobn parameter that controls the sparsity/minimum norm or A
     
     * A_v: vectorial form of A
     
     * S_c: matrix that select only anatomically connected regions/sources (ones and zeros elements).
     Ex if i, j are not connected S_c(i, j) = 0 and 1 otherwise
     

iSDR_cython is a Cython package for solving the EEG/MEG inverse problem using structural/functional prior on the causality between brain regions/sources.
It implements the inverse solver iSDR (check Cite section for more details), It worth noting here that this implementation is a bit different from the original 
paper since now we are assuming a prior on the Multivariate Autoregressive Model coefficients

## Requirements
numpy~=1.6

scipy~=1.4 

scikit-learn~=0.22


## Installation

pip3 install .

## Usage

For how to use this package to reconstruct the brain activation from EEG/MEG
check examples/iSDR_example

## Cite

(1) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective network from EEG/MEG data and dMR information. PRNI 2017 – 7th International Workshop on Pattern Recognition in NeuroImaging, Jun 2017, Toronto, Canada.

(2) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo. Multivariate Autoregressive Model Constrained by Anatomical Connectivity to Reconstruct Focal Sources. 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Aug 2016, Orlando, United States. 2016.
