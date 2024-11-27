
'''
This script contains everything related to density/probability estimation.
'''
from imbaqu.utils.utils import bisect_kde_score
import numpy as np
import KDEpy
import pandas as pd
from typing import Callable


def get_probabilities(data: pd.Series | pd.DataFrame,
                      relevance_pdf: Callable  | list[Callable] | None = None,
                      discrete: bool | list[bool] = False) -> tuple[pd.Series, pd.Series]:
    if isinstance(data, pd.Series):
        # argument sanity checks
        if isinstance(discrete, list):
            if len(discrete) == 1:
                discrete = discrete[0]
            else:
                raise ValueError(f"Too many values are supplied in 'discrete'. Expected list of length 1, received length {len(discrete)}. {discrete=}")
        if isinstance(relevance_pdf, list):
            if len(relevance_pdf) == 1:
                relevance_pdf = relevance_pdf[0]
            else:
                raise ValueError(f"Too many density functions are supplied in 'relevance_pdf'. Expected list of length 1, received length {len(relevance_pdf)}. {relevance_pdf=}")
        prob_emp = empirical_probability(data, discrete)
        prob_rel = relevance_probability(data, relevance_pdf, discrete)
        if (prob_emp < 0).any():
            print("Found negative values in the empirical probability density. This should not be. Continuing, but results might be corrupted.")
        if (prob_rel < 0).any():
            print("Found negative values in the relevance probability density. This should not be. Continuing, but results might be corrupted.")
    
    if isinstance(data, pd.DataFrame):
        if isinstance(discrete, list):
            if len(discrete) != len(data.columns):
                raise TypeError(f"'discrete' is a list of length {len(discrete)}, but the 'data' pd.DataFrame has {len(data.columns)} columns. " +
                                "They should be equal length")
            if not all(isinstance(item, bool) for item in discrete):
                raise ValueError(f"All values in 'discrete' should be booleans, but they are {discrete=}.")
        elif isinstance(discrete, bool):
            discrete = [discrete for i in data.columns]
        prob_emp = pd.DataFrame(data = None, index= data.index, columns= data.columns)
        prob_rel = pd.DataFrame(data = None, index= data.index, columns= data.columns)
        for i, col in enumerate(data.columns):
            col_data = data[col]
            prob_emp[col] = empirical_probability(col_data, discrete[i])
            if relevance_pdf is None:
                prob_rel[col] = relevance_probability(col_data, relevance_pdf, discrete[i])
            else:
                if isinstance(relevance_pdf, list):
                    prob_rel[col] = relevance_probability(col_data, relevance_pdf[i])
                else:
                    raise TypeError("'relevance_pdf' is not of type list or None. Should be if 'data' is a pd.DataFrame.")
                
        # check for negative values
        for col in prob_emp.columns:
            if (prob_emp[col] < 0).any():
                print(f"Found negative values in the empirical probability density of column {col}. This should not be. Continuing, but results might be corrupted.")
        for col in prob_rel.columns:
            if (prob_rel[col] < 0).any():
                print(f"Found negative values in the relevance probability density of column {col}. This should not be. Continuing, but results might be corrupted.")          

        prob_emp = prob_emp.product(axis = 1)
        prob_rel = prob_rel.product(axis = 1)



    return prob_emp, prob_rel



def continuous_probabilities(x: pd.Series) -> pd.Series:
    # calculates the relative likelihood using a KDE
    silverman_bandwidth = (4*x.std(ddof=1)**5 / 3 / len(x))**(1/5)
    kernel = KDEpy.FFTKDE(bw = silverman_bandwidth, kernel = 'gaussian').fit(x.to_numpy())
    grid_x, grid_y = kernel.evaluate(2**14)
    density = map(lambda y : bisect_kde_score(y, grid_x, grid_y), x)        
    density = np.fromiter(density, dtype=np.float64)
    return pd.Series(density, index= x.index, name= x.name)

def discrete_probabilities(x: pd.Series) -> pd.Series:
    # calculates the probability using the PMF
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    replacement_dict = dict(zip(unique_values, probabilities))
    probs = np.vectorize(replacement_dict.get)(x)
    return pd.Series(probs, index= x.index, name= x.name)



def empirical_probability(data: pd.Series , discrete: bool = False) -> pd.Series:
    if discrete == False:
        # continuous case
        prob_emp = continuous_probabilities(data)
    else:
        prob_emp = discrete_probabilities(data)

    return prob_emp


def relevance_probability(data: pd.Series, relevance_pdf: Callable | None, discrete: bool = False) -> pd.Series:
    """If a probability density function 'relevance_pdf' is supplied, 'discrete' is ignored. If 'relevance_pdf' is None, uniform relevance is assumed."""

    if relevance_pdf is not None:
        prob_rel = data.apply(lambda x: relevance_pdf(x))
    else:
        if discrete == False:
            rel = 1 / (max(data) - min(data))
        else:
            rel = 1 / len(np.unique(data))

        prob_rel = np.full(np.shape(data), rel)

    return pd.Series(prob_rel, index= data.index, name= data.name)