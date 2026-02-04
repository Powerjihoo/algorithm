import inspect
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np

from resources.constant import CONST
from utils.logger import logger

from ._calculator import (
    IndexCalculateError,
    _ModelIndexCalcMSE,
    _ModelIndexCalcTagImportance,
    _ModelIndexCalculatorBase,
)
from ._index_setting import (
    ModelIndexSettingBase,
    ModelIndexSettingMSE,
    ModelIndexSettingTagImportance,
)
from ._model_data import ModelData


# * ModelCalcSetting
@dataclass
class ModelIndexCalculatorBase(ABC):
    data: ModelData = field(default_factory=ModelData)
    setting: ModelIndexSettingBase = None
    calculator: Optional[_ModelIndexCalculatorBase] = None

    @classmethod
    def from_dict(
        cls, index_setting_class: ModelIndexSettingBase, index_setting: dict
    ) -> object:
        return cls(
            setting=index_setting_class(**{
                k: v
                for k, v in index_setting.items()
                if k in inspect.signature(index_setting_class).parameters
            }),
        )

    def update_index_status_code(self, index_statuses: np.array) -> None:
        if np.all(index_statuses == CONST.STATUSCODE_IGNORED):
            self.data.status_index = CONST.STATUSCODE_IGNORED
        elif np.all(index_statuses == CONST.STATUSCODE_BAD):
            self.data.status_index = CONST.STATUSCODE_BAD
        else:
            self.data.status_index = CONST.STATUSCODE_GOOD


@dataclass
class ModelIndexCalculatorMSE(ModelIndexCalculatorBase):
    setting: ModelIndexSettingMSE
    calculator: Type[_ModelIndexCalcMSE] = _ModelIndexCalcMSE

    def calc_index(self, index_ary: np.array) -> float:
        try:
            index = self.calculator.calc_index(
                index_ary, self.setting.weights_ary, self.setting.importances_ary
            )
            self.data.index = index
        except Exception as e:
            logger.error(e)


@dataclass
class ModelIndexCalculatorTagImportance(ModelIndexCalculatorBase):
    setting: ModelIndexSettingTagImportance
    calculator: Type[_ModelIndexCalcTagImportance] = _ModelIndexCalcTagImportance

    def calc_index(self, index_ary: np.array, status_ary: np.array, priority_ary: np.array = None) -> None:
        try:
            index = self.calculator.calc_index(
                index_ary,
                status_ary,
                priority_ary,
                self.setting.weights_ary,
            )
            self.data.index = index
            self.update_index_status_code(status_ary)
        except IndexCalculateError:
            self.data.index = np.nan
            self.data.status_index = 0
        except Exception as e:
            logger.error(e)
