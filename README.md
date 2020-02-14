# iSDR_cython: Cython implementation of iSDR_p (iterative source and dynamics reconstruction)

A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) on the source space


<img src="https://latex.codecogs.com/gif.latex?U%28J%2C%20A%29%20%3D%20%5Csum_t%20%5Cleft%20%5C%7C%20M_t%20-%20G%5Csum_i%20S_cA_iJ_%7B%20t-i%7D%5Cright%20%5C%7C_2%5E2%20&plus;%20%5Clambda%5Cleft%20%5C%7C%20J%20%5Cright%20%5C%7C_%7B%2012%7D%20&plus;%20%5Calpha%5Cleft%20%5C%7C%20S_cA%20%5Cright%20%5C%7C_%7B1/2%7D" title="U(J, A) = \sum_t \left \|  M_t - G\sum_i S_cA_iJ_{ t-i}\right \|_2^2 + \lambda\left \| J \right \|_{ 12} + \alpha\left \| S_cA \right \|_{1/2}
"/>

Where: 

     * A_i: i=1,..,p are the matrices of the MVAR model of order p

     * M: EEG or/and MEG measurement

     * G: Lead field matrix which project brain activity into sensor space

     * J: Brain activity (distributed source model with fixed position), J_t at time t

     * lambda: regularization parameter that controls the sparsity of J
     
     * alpha: regularizatiobn parameter that controls the sparsity/minimum norm or A
     

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
