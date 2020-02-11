# ElasticNetVAR.jl
Selecting time-series hyperparameters with the artificial jackknife: Elastic-net VAR application.

## Description
This repository contains code related to “Selecting time-series hyperparameters with the artificial jackknife”.

This article proposes a generalisation of the delete-*d* jackknife to solve hyperparameter selection problems for time series. This novel technique is compatible with dependent data since it substitutes the jackknife removal step with a fictitious deletion, wherein observed datapoints are replaced with artificial missing values. In order to emphasise this point, I called this methodology artificial delete-*d* jackknife. As an illustration, it is used to regulate vector autoregressions with an elastic-net penalty on the coefficients.

## Citation
If you use this code or build upon it, please use the following (bibtex) citation:
```bibtex
@article{pellegrino2020ajk,
    title={Selecting time-series hyperparameters with the artificial jackknife},
    author={Filippo Pellegrino},
    year={2020},
    eprint={3041007},
    archivePrefix={arXiv},
    primaryClass={stat.ME}
}
```
