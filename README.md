# Reliable Predictive Inference

An important factor to guarantee a responsible use of data-driven recommendation systems is that we should be able to communicate their uncertainty to decision makers. This can be accomplished by constructing prediction intervals, which provide an intuitive measure of the limits of predictive performance.

This package contains a Python implementation of Orthogonal quantile regression (OQR) [1] methodology for constructing distribution-free prediction intervals. 

# Orthogonal Quantile Regression [1]

OQR is a method that improves the conditional validity of standard quantile regression methods.

[1] Shai Feldman, Stephen Bates, Yaniv Romano, [“Improving Conditional Coverage via Orthogonal Quantile Regression.”](https://arxiv.org/abs/2106.00394) 2021.

## Getting Started

This package is self-contained and implemented in python.

Part of the code is a taken from the calibrated-quantile-uq package available at https://github.com/YoungseogChung/calibrated-quantile-uq. 

### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/shai128/oqr.git
```

## Usage


### OQR

Please refer to [oqr_synthetic_data_example.ipynb](oqr_synthetic_data_example.ipynb) for basic usage. 
Comparisons to competitive methods and can be found in [display_results.ipynb](display_results.ipynb).

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1].

### Publicly Available Datasets

* [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback): BlogFeedback data set.

* [Bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure): Physicochemical  properties  of  protein  tertiary  structure  data  set.

* [Kin8nm](http://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/kin-family/): A variant of Kin family of datasets.

* [Naval](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants): Condition based maintenance of naval propulsion plants data set.

* [Facebook Variant 1 and Variant 2](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset): Facebook  comment  volume  data  set.


### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded using the code in the folder /get_meps_data/ under this repository. It is based on [this explanation](/get_meps_data/README.md) (code provided by [IBM's AIF360](https://github.com/IBM/AIF360)).

* [MEPS_19](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 19.

* [MEPS_20](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 20.

* [MEPS_21](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): Medical expenditure panel survey,  panel 21.



