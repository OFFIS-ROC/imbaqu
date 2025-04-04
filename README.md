# Imbaqu
This package called "imbaqu" (Imbalance Quantification) implements the methods for the quantification of data imbalance as described in the corresponding article "Quantification of Data Imbalance". The article is available [Open Access](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13840).

The goal of imbaqu is to allow the user to assess the data imbalance in a data set. This can be done either univariate or multivariate. To do so, the density of the data is estimated using kernel density estimation. In addition, the relevance of the data is either to be considered uniform, which is the default case, or can be provided by the user in the form of a probability density function. Subsequently, the data imbalance can be assessed using the provided function of the imbaqu package.

* **mean imbalance rate (mIR)**: The mIR assesses how imbalanced the samples are on average. $`\text{mIR}_{N_k}= 1`$ indicates that the probability distribution matches the relevance distribution and the dataset is not imbalanced and the imbalance increases with increasing $`\text{mIR}_{N_k}`$. Thus, $`\text{mIR}_{N_k}= 2`$ denotes, for example, that the frequencies of the samples deviate on average by $`2^{N_k}`$-times from the expected relevant frequencies.
* **imbalanced sample percentage (ISP):** The ISP indicates the percentage of samples that have greater imbalance than a predefined threshold.
 
If used, please cite:

Wibbeke, J., Rohjans, S. and Rauh, A. (2025), Quantification of Data Imbalance. Expert Systems, 42: e13840. https://doi.org/10.1111/exsy.13840
```
@article{wibbeke2024quantification,
author = {Wibbeke, Jelke and Rohjans, Sebastian and Rauh, Andreas},
title = {Quantification of Data Imbalance},
journal = {Expert Systems},
volume = {42},
number = {3},
pages = {e13840},
keywords = {imbalanced regression, machine learning, metric, sample weighting data mining},
doi = {https://doi.org/10.1111/exsy.13840},
year = {2025}
}
```

## Requirements and Installation
Download the package from GitHub, navigate to the `setup.py` and install it using pip:
```
pip install .
```
The package was tested using:
```
python=3.10.13
numpy=1.26.4
pandas=2.1.4
KDEpy=1.1.11
```

Other versions may also work, but have not been tested.

## Usage
```python
# mIR continuous
mir = imbaqu.mean_imbalance_ratio(data['num1'])
print(f'mIR of a continuous variable: {mir:.2f}')

# mIR discrete
mir = imbaqu.mean_imbalance_ratio(data['cat1'], discrete=True)
print(f'mIR of a discrete variable: {mir:.2f}')
```
For further explanation see the [example notebook](example.ipynb).
