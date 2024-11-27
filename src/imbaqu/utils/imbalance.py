import numpy as np
from typing import Callable
import pandas as pd

from imbaqu.utils.density import get_probabilities
from imbaqu.utils.utils import drop_inf_and_nan

def probability_ratio(data: pd.Series | pd.DataFrame,
                      relevance_pdf: Callable  | list[Callable] | None = None,
                      discrete: bool | list[bool] = False,
                      drop_inf_nan: bool = True) -> pd.Series:

    """
    Computes the probability ratio for given data using empirical and relevance probabilities.

    Parameters:
    - data (pd.Series | pd.DataFrame): Input data for which to compute the probability ratio.
    - relevance_pdf (Callable | list[Callable] | None, optional): Probability density function(s) for relevance probability.
      - If a Series is passed to `data`, `relevance_pdf` should be a single callable function.
      - If a DataFrame is passed to `data`, `relevance_pdf` should be a list of callable functions with one function per column in data.
      - It is assumed that each callable function returns a value of type float.
      - If None, default functions will be used for relevance probability, which assumes uniform distribution of the relevance.
    - discrete (bool | list[bool], optional): Indicates if the data is discrete.
      - If a Series is passed to `data`, `discrete` should be a single boolean value.
      - If a DataFrame is passed to `data`, `discrete` should be a list of boolean values with one value per column.
    - drop_inf_nan (bool, optional): Whether to drop infinite and NaN values from `data` before computation.
      Default is True.

    Returns:
    - pd.Series: The probability ratio for each element in the input data.

    Notes:
    - The function first cleans the data by dropping infinite and NaN values if `drop_inf_nan` is True.
    - It calculates empirical probabilities and relevance probabilities for the given data.
    - The final probability ratio is the division of empirical probabilities by relevance probabilities.
    - For DataFrame input, the product of probabilities across columns is computed and then adjusted by the number of columns.
    """
    # convert/drop inf and nan values.
    data = drop_inf_and_nan(data, drop_inf_nan)
    # calculation for series
    if isinstance(data, pd.Series):
        prob_emp, prob_rel = get_probabilities(data, relevance_pdf, discrete)
        lamb = prob_emp.div(prob_rel)

    else:
        # calculation for Dataframe
        prob_emp, prob_rel = get_probabilities(data, relevance_pdf, discrete)
        lamb = prob_emp.div(prob_rel)
        lamb = lamb.pow(1/len(data.columns))

    return lamb




    


def imbalance_ratio(lamb: pd.Series) -> pd.Series:
    '''Calculates the imbalance ratio (IR) from the probability ratio (Lambda)'''
    ir = (lamb.apply(np.log).abs()).apply(np.exp)
    return ir