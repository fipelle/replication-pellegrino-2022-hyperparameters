## Description
This repository contains code related to “[Selecting time-series hyperparameters with the artificial jackknife](https://arxiv.org/abs/2002.04697)”.

This article proposes a generalisation of the delete-*d* jackknife to solve hyperparameter selection problems for time series. I call it artificial delete-d jackknife to stress that this approach substitutes the classic removal step with a fictitious deletion, wherein observed datapoints are replaced with artificial missing values. Doing so keeps the data order intact and allows plain compatibility with time series. This manuscript shows a simple illustration in which it is applied to regulate high-dimensional elastic-net vector autoregressive moving average (VARMA) models.

<img src="./img/heading.svg">

## Citation
If you use this code or build upon it, please use the following (bibtex) citation:
```bibtex
@misc{pellegrino2021selecting,
      title={Selecting time-series hyperparameters with the artificial jackknife}, 
      author={Filippo Pellegrino},
      year={2021},
      eprint={2002.04697},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
