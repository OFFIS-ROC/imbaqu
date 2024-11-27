import pandas as pd
import numpy as np
from typing import Callable

from imbaqu.utils.imbalance import imbalance_ratio, probability_ratio


def imbalanced_sample_percentage(
    data: pd.Series | pd.DataFrame,
    relevance_pdf: Callable  | list[Callable] | None = None,
    discrete: bool | list[bool] = False,
    drop_inf_nan: bool = True,  
    ir_bound: None| float = None,
    lower_bound: None| float = None,
    upper_bound: None| float = None) -> float:
    """
    Calculate the percentage of imbalanced samples based on the provided criteria.

    Args:
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
    - ir_bound: Imbalance ratio threshold for considering samples imbalanced. If defined, excludes lower_bound and upper_bound.
    - lower_bound: Lower threshold for probability ratio lamb.
    - upper_bound: Upper threshold for probability ratio lamb.

    Returns:
    - (float): Percentage of imbalanced samples based on the given criteria.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("The argument 'data' is not of type pd.Series or pd.DataFame.")
    if ir_bound is None and lower_bound is None and upper_bound is None:
        raise ValueError("At least one of ir_bound, lower_bound, or upper_bound must be provided.")
    if ir_bound is not None and (lower_bound is not None or upper_bound is not None):
        raise ValueError("lower_bound and upper_bound must be None if ir_bound is defined.")


    lamb = probability_ratio(data, relevance_pdf, discrete, drop_inf_nan)
    if ir_bound is None:
        isp_l = 0.0
        isp_u = 0.0
        if lower_bound is not None:
            isp_l = (lamb < lower_bound).sum() / len(lamb) * 100
        if upper_bound is not None:
            # Perform actions for when t_upper is provided
            isp_u = (lamb > upper_bound).sum() / len(lamb) * 100
        isp = isp_l + isp_u
    
    else:
        ir = imbalance_ratio(lamb)
        isp = (ir > ir_bound).sum() / len(ir) *100

    return isp


def mean_imbalance_ratio(
    data: pd.Series | pd.DataFrame,
    relevance_pdf: Callable  | list[Callable] | None = None,
    discrete: bool | list[bool] = False,
    drop_inf_nan: bool = True) -> float:
    '''
    Computes the mean imbalance ratio (mIR) for given data using empirical and relevance probabilities.
    Args:
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
    - (float): The mean imbalance ratio mIR) for the input data.
    '''
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("The argument 'data' is not of type pd.Series or pd.DataFame.")
    lamb = probability_ratio(data, relevance_pdf, discrete, drop_inf_nan)
    ir = imbalance_ratio(lamb)
    mir = float(np.mean(ir))
    return mir








