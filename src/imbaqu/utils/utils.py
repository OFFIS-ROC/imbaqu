'''
This script contains basic utility and helper functions.
'''

import bisect
import numpy as np
import pandas as pd

def bisect_kde_score(y, grid_x, grid_y) -> np.ndarray:
    idx = bisect.bisect(grid_x, y)
    try:
        dens = grid_y[idx]
    except IndexError:
        if idx <= -1:
            idx = 0
        elif idx >= len(grid_x):
            idx = len(grid_x) - 1
        dens = grid_y[idx]
    return dens



def drop_inf_and_nan(data: pd.Series | pd.DataFrame, drop_inf_nan: bool = True) -> pd.Series | pd.DataFrame:
    # Replace inf values with NaN to handle them together
    cleaned = data.replace([np.inf, -np.inf], np.nan)

    if isinstance(cleaned, pd.Series):
        counter = cleaned.isna().sum()
    else:
        counter = cleaned.isna().sum().sum()
    if counter != 0:
        if drop_inf_nan == True:
            # Drop rows with NaN values
            cleaned = cleaned.dropna(axis = 0)
        else:
            raise ValueError(f"Found {counter} inf/NaN values in the input. Either set 'drop_inf_nan' to True or drop them manually.")
    return cleaned