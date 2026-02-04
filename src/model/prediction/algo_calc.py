import numpy as np
from numba import njit

from utils.logger import logger

_kernel_band_factor = np.sqrt(2)
CONST_PI = np.pi


@njit(fastmath=True, cache=True)
def aakr(input_data, model_data, model_data_for_weight):
    _distance = np.sqrt(np.power((input_data - model_data_for_weight), 2).sum(axis=1))
    # ! _distance_min = np.amin(_distance)
    _distance_min = _distance.min()
    _kb = _distance_min * _kernel_band_factor if _distance_min != 0 else 0.000001
    _weight = (np.exp(-1 * np.power((_distance / _kb), 2) / 2)) * (
        np.sqrt(2 * CONST_PI * _kb) ** -1
    )
    return ((model_data * _weight.reshape(-1, 1)).sum(axis=0)) / _weight.sum(0)


def aakr_no_numba(input_data, model_data, model_data_for_weight):
    _distance = np.sqrt(np.power((input_data - model_data_for_weight), 2).sum(axis=1))
    # ! _distance_min = np.amin(_distance)
    _distance_min = _distance.min()
    kb = _distance_min * _kernel_band_factor if _distance_min != 0 else 0.000001
    weight = (np.exp(-1 * np.power((_distance / kb), 2) / 2)) * (
        np.sqrt(2 * CONST_PI * kb) ** -1
    )
    return ((model_data * weight.reshape(-1, 1)).sum(axis=0)) / weight.sum(0)


""" Numba initialize """

logger.debug(f"{'Initializing':12} | Numba Initializing...")
aakr(
    input_data=np.array([1, 1]),
    model_data=np.array([[1, 1], [1, 1]]),
    model_data_for_weight=np.array([[1, 1], [1, 1]]),
)
