from abc import ABC, abstractclassmethod

import numpy as np


class IndexCalculateError(Exception):
    def __init__(self, message: str = "Can not calculate index"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class _ModelIndexCalculatorBase(ABC):
    """
    Abstract base class for model index calculators.

    Attributes:
        alarm_ref_index (float): The reference index value taken from the setting.

    Methods:
        calc_index(): Abstract method to calculate the index.
    """

    alarm_ref_index: float = 0.7
    # alarm_ref_index: float = settings.ALARM_REF_INDEX

    def __repr__(self):
        return __class__.__name__

    def __str__(self):
        return __class__.__name__

    @abstractclassmethod
    def calc_index(cls):
        """
        Abstract method to calculate the index.

        Returns:
            None
        """
        ...


class _ModelIndexCalcMSE(_ModelIndexCalculatorBase):
    """
    Calculator for index using Mean Squared Error (MSE) based model index.

    Methods:
        calc_index(): Calculate the index using MSE based model index.
    """

    @classmethod
    def calc_index(cls, mse, mse_ref) -> float:
        """
        Calculate the index using Mean Squared Error (MSE) based model index.

        Args:
            mse (float): The Mean Squared Error value.
            mse_ref (float): The reference Mean Squared Error value.

        Returns:
            float: The calculated model index value.
        """
        return 1 - (mse * (1 - cls.alarm_ref_index) / mse_ref)

    def __repr__(self):
        return f"IndexCalculator: {__class__.__name__}"

    def __str__(self):
        return f"IndexCalculator: {__class__.__name__}"


class _ModelIndexCalcTagImportance(_ModelIndexCalculatorBase):
    """
    Calculator for index using Tag Importance based model index.

    Methods:
        calc_index(): Calculate the index using Tag Importance based model index.
    """

    # @classmethod
    # def calc_index(
    #     cls,
    #     index_ary: np.array,
    #     status_ary: np.array,
    #     weights_ary: np.array,
    #     importances_ary: np.array,
    # ) -> float:
    #     """
    #     Calculate the index using Tag Importance based model index.

    #     Args:
    #         index_ary (np.array): Array of index values.
    #         weights_ary (np.array): Array of weights corresponding to index values.
    #         importances_ary (np.array): Array indicating the importance of index values.

    #     Returns:
    #         float: The calculated model index value.
    #     """
    #     _target_idx_not_none = ~np.isnan(index_ary)
    #     _target_idx = np.logical_and(_target_idx_not_none, (status_ary>=192))
    #     if not np.any(_target_idx):
    #         raise IndexCalculateError("Can not calculate model index")

    #     target_index_ary = index_ary[_target_idx]
    #     target_weights_ary = weights_ary[_target_idx]

    #     _index_weighted_ary = target_index_ary * target_weights_ary
    #     _weighted_avg = _index_weighted_ary.sum() / (target_weights_ary.sum())
    #     _weighted_index_ary = np.clip(
    #         1 - (1 - target_index_ary) * target_weights_ary, 0, 1
    #     )
    #     try:
    #         _weighted_min = min(_weighted_index_ary)
    #     except ValueError:
    #         _weighted_min = None
    #     _sub_avg = (_weighted_avg + _weighted_min) / 2

    #     try:
    #         _importance_min = np.min(index_ary[importances_ary])
    #     except ValueError:
    #         _importance_min = 1

    #     index_final = min(_sub_avg, _importance_min)
    #     return index_final


    @classmethod
    def calc_index(
        cls,
        index_ary: np.array,
        status_ary: np.array,
        weights_ary: np.array,
        priority: bool,
    ) -> float:
        """
        Calculate the index using Tag Importance based model index.

        Args:
            index_ary (np.array): Array of index values.
            weights_ary (np.array): Array of weights corresponding to index values.
            importances_ary (np.array): Array indicating the importance of index values.

        Returns:
            float: The calculated model index value.
        """
        _target_idx_not_none = ~np.isnan(index_ary)
        _target_idx = np.logical_and(_target_idx_not_none, (status_ary>=192))
        if not np.any(_target_idx):
            raise IndexCalculateError("Can not calculate model index")
        
        if priority:
            return np.nanmin(index_ary)

        target_index_ary = index_ary[_target_idx]
        target_weights_ary = weights_ary[_target_idx]

        _index_weighted_ary = target_index_ary * target_weights_ary
        _weighted_avg = _index_weighted_ary.sum() / (target_weights_ary.sum())
        _weighted_index_ary = np.clip(
            1 - (1 - target_index_ary) * target_weights_ary, 0, 1
        )
        try:
            _weighted_min = min(_weighted_index_ary)
        except ValueError:
            _weighted_min = None
        _sub_avg = (_weighted_avg + _weighted_min) / 2
        return _sub_avg
