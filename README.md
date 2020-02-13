# ElasticNetVAR.jl
Selecting time-series hyperparameters with the artificial jackknife: elastic-net VAR.

## Description
This repository contains code related to “[Selecting time-series hyperparameters with the artificial jackknife](https://arxiv.org/abs/2002.04697)”.

This article proposes a generalisation of the delete-*d* jackknife to solve hyperparameter selection problems for time series. This novel technique is compatible with dependent data since it substitutes the jackknife removal step with a fictitious deletion, wherein observed datapoints are replaced with artificial missing values. In order to emphasise this point, I called this methodology artificial delete-*d* jackknife. As an illustration, it is used to regulate vector autoregressions with an elastic-net penalty on the coefficients.

<img src="./img/heading.svg">

## Citation
If you use this code or build upon it, please use the following (bibtex) citation:
```bibtex
@misc{pellegrino2020ajk,
    title={Selecting time-series hyperparameters with the artificial jackknife},
    author={Filippo Pellegrino},
    year={2020},
    eprint={2002.04697},
    archivePrefix={arXiv},
    primaryClass={stat.ME}
}
```
