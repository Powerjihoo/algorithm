from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from config import settings

from resources.constant import CONST

from ..._data import TagWindowedData
from ._alarm_setting import (
    AlarmSettingBase,
    AlarmSettingSlope,
    AlarmSettingSPRT,
    AlarmSettingThreshold,
)
from ._calculator import (
    _IndexCalcSlope,
    _IndexCalcSPRT,
    _IndexCalcThresholdSingle,
)


@dataclass
class TagIndexCalculatorBase:
    """
    This class will be used to set tags for which alarms are not set.
    The tags set with this class do not calculate an index.

    Attributes:
        tagname (str): Name of the tag.
        data (TagData): Data associated with the tag.
        alarm_setting (AlarmSettingBase): Alarm settings for the tag.

    Methods:
        calc_index(): Abstract method to calculate the index for the tag.
    """

    tagname: str
    data: TagWindowedData
    alarm_setting: AlarmSettingBase

    def update_index_status_code(self) -> None:
        if not np.isnan(self.data.index):
            self.data.status_index = (
                CONST.STATUSCODE_GOOD
                if not self.data.ignore_status
                else CONST.STATUSCODE_IGNORED
            )
        else:
            self.data.status_index = CONST.STATUSCODE_BAD

    @abstractmethod
    def calc_index(self) -> None:
        """
        Abstract method to calculate the index for the tag.

        Returns:
            None
        """
        return None


@dataclass
class TagIndexCalculatorThresholdSingle(TagIndexCalculatorBase):
    """
    Settings for a single threshold-based tag.

    Attributes:
        alarm_setting (AlarmSettingThreshold): Threshold-based alarm settings for the tag.
        index_calculator (IndexCalcThresholdSingle): Index calculator for threshold-based tag.

    Methods:
        calc_index(): Calculate the index for the threshold-based tag.
    """  # noqa: E501

    alarm_setting: AlarmSettingThreshold
    index_calculator = _IndexCalcThresholdSingle

    def calc_index(self) -> None:
        """
        Calculate the index for the threshold-based tag.

        Returns:
            None
        """
        try:
            _index = self.index_calculator.calc_index(
                res=abs(self.data.raw.values - self.data.pred.values),
                threshold=self.alarm_setting.threshold,
            )

            self.data.index.values[-1] = np.nanmean(_index)
            self.data.index.statuscodes[-1] = self.data.pred.statuscodes[-1]
            self.data.index.timestamps[-1] = self.data.pred.timestamps[-1]
        except (AttributeError, TypeError):
            return None


@dataclass
class TagIndexCalculatorThresholdSingle_Ignore100(TagIndexCalculatorBase):
    """
    Settings for a single threshold-based tag.

    Attributes:
        alarm_setting (AlarmSettingThreshold): Threshold-based alarm settings for the tag.
        index_calculator (IndexCalcThresholdSingle): Index calculator for threshold-based tag.

    Methods:
        calc_index(): Calculate the index for the threshold-based tag.
    """  # noqa: E501

    alarm_setting: AlarmSettingThreshold
    index_calculator = _IndexCalcThresholdSingle

    def calc_index(self) -> None:
        """
        Calculate the index for the threshold-based tag.

        Returns:
            None
        """
        try:
            _index = self.index_calculator.calc_index(
                abs(self.data.raw.values - self.data.pred.values),
                self.alarm_setting.threshold,
            )

            for idx, ignore_status in enumerate(self.data.raw.ignore_statuses):
                _index[idx] = 1.0 if ignore_status else _index[idx]

            self.data.index.values[-1] = np.nanmean(_index)
            self.data.index.statuscodes[-1] = self.data.pred.statuscodes[-1]
            self.data.index.timestamps[-1] = self.data.pred.timestamps[-1]
        except (AttributeError, TypeError):
            return None


@dataclass
class TagIndexCalculatorSlope(TagIndexCalculatorBase):
    """
    Settings for a slope-based tag.

    Attributes:
        alarm_setting (AlarmSettingSlope): Slope-based alarm settings for the tag.
        index_calculator (IndexCalcSlope): Index calculator for slope-based tag.

    Methods:
        calc_index(): Calculate the index for the slope-based tag.
    """

    alarm_setting: AlarmSettingSlope
    index_calculator = _IndexCalcSlope

    def calc_index(self) -> None:
        """
        Calculate the index for the slope-based tag.

        Returns:
            None
        """
        try:
            self.data.index = self.index_calculator.calc_index(
                self.data.slope, self.alarm_setting.normal_slope
            )
        except TypeError:
            return None


@dataclass
class TagIndexCalculatorSPRT(TagIndexCalculatorBase):
    """
    Settings for a SPRT (Sequential Probability Ratio Test) based tag.

    Attributes:
        alarm_setting (AlarmSettingSPRT): SPRT-based alarm settings for the tag.
        index_calculator (IndexCalcSPRT): Index calculator for SPRT-based tag.

    Methods:
        calc_index(): Calculate the index for the SPRT-based tag.
    """

    alarm_setting: AlarmSettingSPRT
    index_calculator = _IndexCalcSPRT

    def calc_index(self) -> None:
        """
        Calculate the index for the SPRT-based tag.

        Returns:
            None
        """
        self.data.index = np.nan


class TagIndexCalculatorClasses:
    """
    Class representing the various tag settings.

    Attributes:
        threshold (TagSettingThresholdSingle): Settings for a threshold-based tag.
        sprt (TagSettingSlope): Settings for a slope-based tag.
        slope (TagSettingSPRT): Settings for a SPRT-based tag.

    Methods:
        __getitem__(key): Get the tag settings based on the given key.
    """

    # def __init__(self):
    #     self.threshold = TagSettingThresholdSingle
    #     self.sprt = TagSettingSlope
    #     self.slope = TagSettingSPRT
    if not settings.system.force_ignore_idx_100:
        Threshold = TagIndexCalculatorThresholdSingle
    else:
        Threshold = TagIndexCalculatorThresholdSingle_Ignore100
    SPRT = TagIndexCalculatorSlope
    Slope = TagIndexCalculatorSPRT

    def __getitem__(self, key):
        """
        Get the tag settings based on the given key.

        Args:
            key: The key corresponding to the tag setting.

        Returns:
            TagSettingBase or its subclass: Tag setting class based on the key.
        """
        if key is None:
            return TagIndexCalculatorBase
        return getattr(self, key)
