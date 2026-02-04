from abc import ABC, abstractclassmethod

import numpy as np
from utils.logger import logger


class _TagIndexCalculatorBase(ABC):
    """
    Abstract base class for tag index calculators.

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


class _IndexCalcThresholdSingle(_TagIndexCalculatorBase):
    """
    Calculator for index of threshold-based tags.

    Methods:
        calc_index(): Calculate the index for the threshold-based tag.
    """

    @classmethod
    def calc_index(cls, res, threshold):
        """
        Calculate the index for the threshold-based tag.

        Args:
            res (float): The result value used in the index calculation.
            threshold (float): The threshold value.

        Returns:
            float or None: The calculated index value or None if an error occurs.
        """
        try:
            return np.clip(
                1 - (res * (1 - cls.alarm_ref_index) / threshold), a_min=0, a_max=1
            )
        except Exception as e:
            logger.debug(e)
            return np.nan

    def __repr__(self):
        return f"IndexCalculator: {__class__.__name__}"

    def __str__(self):
        return f"IndexCalculator: {__class__.__name__}"


class _IndexCalcSlope(_TagIndexCalculatorBase):
    """
    Calculator for index of slope-based tags.

    Methods:
        calc_index(): Calculate the index for the slope-based tag.
    """

    @classmethod
    def calc_index(cls, slope, normal_slope):
        """
        Calculate the index for the slope-based tag.

        Args:
            slope (float): The slope value.
            normal_slope (float): The normal slope value.

        Returns:
            float or None: The calculated index value or None if an error occurs.
        """
        try:
            return np.clip(
                1 - (slope * (1 - cls.alarm_ref_index) / normal_slope), a_min=0, a_max=1
            )
        except Exception as e:
            logger.debug(e)
            return np.nan

    def __repr__(self):
        return f"IndexCalculator: {__class__.__name__}"

    def __str__(self):
        return f"IndexCalculator: {__class__.__name__}"


class _IndexCalcSPRT(_TagIndexCalculatorBase):
    """
    Calculator for index of SPRT (Sequential Probability Ratio Test) based tags.

    Methods:
        calc_index(): Abstract method to calculate the index for the SPRT-based tag.
        init_llr(): Initialize the log-likelihood ratio for SPRT calculations.
    """

    @classmethod
    def calc_index(cls, res, threshold):
        """
        Abstract method to calculate the index for the SPRT-based tag.

        Returns:
            None
        """
        ...

    def init_llr(self):
        """
        Initialize the log-likelihood ratio for SPRT calculations.
        This method sets llr_m_pos, llr_m_neg, and llr_s to 0.

        Returns:
            None
        """
        self.llr_m_pos, self.llr_m_neg, self.llr_s = 0, 0, 0

    def __repr__(self):
        return f"IndexCalculator: {__class__.__name__}"

    def __str__(self):
        return f"IndexCalculator: {__class__.__name__}"
