from abc import ABC
from dataclasses import dataclass
from enum import Enum

import numpy as np


# * ModelIndexSetting
@dataclass
class ModelIndexSettingBase(ABC):
    ...


@dataclass
class ModelIndexSettingMSE(ModelIndexSettingBase):
    ...


@dataclass
class ModelIndexSettingTagImportance(ModelIndexSettingBase):
    weights_ary: np.array
    importances_ary: np.array


class ModelIndexEnum(Enum):
    TagImportance = 0
    MSE = 1

    @classmethod
    def get_member_by_value(cls, value):
        return next((member for member in cls if member.value == value), None)
