# iSDR_cython: Cython implementation of iSDR_p (iterative source and dynamics reconstruction)

A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) on the source space

<img src="http://latex.codecogs.com/gif.latex?\Large&space;U(J)&space;=&space;\sum_{t=p}^T||M_t-G\sum_{i=1}^pA_iJ_{t-i}||_{2}^2&plus;\alpha&space;||J||_{21}" title="\Large U(J, A) = \left \|  M - G\sum A_iJ_t\right \|_2^2 + \lambda\left \| J \right \|_{ 12} + \alpha\left \| S_cA \right \|_{1 or 2}" />

<img src="https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20M%20-%20G%5Csum%20A_iJ_t%5Cright%20%5C%7C_2%5E2%20&plus;%20%5Clambda%5Cleft%20%5C%7C%20J%20%5Cright%20%5C%7C_%7B%2012%7D%20&plus;%20%5Calpha%5Cleft%20%5C%7C%20S_cA%20%5Cright%20%5C%7C_%7B1%20or%202%7D"title="\left \|  M - G\sum A_iJ_t\right \|_2^2 + \lambda\left \| J \right \|_{ 12} + \alpha\left \| S_cA \right \|_{1 or 2}"/>

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
