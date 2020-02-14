# iSDR_cython: Cython implementation of iSDR_p (iterative source and dynamics reconstruction)

A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) on the source space


iSDR_cython is a Cython package for solving the EEG/MEG inverse problem using structural/functional prior on the causality between brain regions/sources.
It implements the inverse solver iSDR (check Cite section for more details), It worth noting here that this implementation is a bit different from the original 
paper since now we are assuming a prior on the Multivariate Autoregressive Model coefficients

## Requirements
numpy~=1.6

scipy~=1.4 

scikit-learn~=0.22


## Installation and Usage

pip3 install .

For how to use this package to reconstruct the brain activation from EEG/MEG
check examples/iSDR_example

## Cite

(1) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective network from EEG/MEG data and dMR information. PRNI 2017 – 7th International Workshop on Pattern Recognition in NeuroImaging, Jun 2017, Toronto, Canada.

(2) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo. Multivariate Autoregressive Model Constrained by Anatomical Connectivity to Reconstruct Focal Sources. 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Aug 2016, Orlando, United States. 2016.
