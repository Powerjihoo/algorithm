from typing import TypeVar

from .model._calculator import _ModelIndexCalcMSE, _ModelIndexCalcTagImportance
from .model._index_setting import ModelIndexSettingMSE, ModelIndexSettingTagImportance
from .model.setting import ModelIndexCalculatorMSE, ModelIndexCalculatorTagImportance
from .tag._alarm_setting import (
    AlarmSettingSlope,
    AlarmSettingSPRT,
    AlarmSettingThreshold,
)
from .tag._calculator import _IndexCalcSlope, _IndexCalcSPRT, _IndexCalcThresholdSingle
from .tag.setting import (
    TagIndexCalculatorSlope,
    TagIndexCalculatorSPRT,
    TagIndexCalculatorThresholdSingle,
)

TModelIndexCalculator = TypeVar(
    "TModelIndexCalculator", bound=_ModelIndexCalcMSE | _ModelIndexCalcTagImportance
)
TModelIndexSetting = TypeVar(
    "TModelIndexSetting", bound=ModelIndexSettingMSE | ModelIndexSettingTagImportance
)
TModelCalcSetting = TypeVar(
    "TModelCalcSetting",
    bound=ModelIndexCalculatorTagImportance | ModelIndexCalculatorMSE,
)
TAlarmSetting = TypeVar(
    "TAlarmSetting",
    bound=AlarmSettingThreshold | AlarmSettingSPRT | AlarmSettingSlope,
)

TTagIndexCalculator = TypeVar(
    "TTagIndexCalculator",
    bound=_IndexCalcThresholdSingle | _IndexCalcSlope | _IndexCalcSPRT,
)
TTagIndexSetting = TypeVar(
    "TTagIndexSetting",
    bound=TagIndexCalculatorThresholdSingle
    | TagIndexCalculatorSlope
    | TagIndexCalculatorSPRT,
)
