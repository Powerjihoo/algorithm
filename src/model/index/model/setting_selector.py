from typing import Any, Type

from utils.scheme.singleton import SingletonInstance

from .._types import TModelCalcSetting, TModelIndexSetting
from ._index_setting import (
    ModelIndexEnum,
    ModelIndexSettingMSE,
    ModelIndexSettingTagImportance,
)
from .setting import ModelIndexCalculatorMSE, ModelIndexCalculatorTagImportance


# * ModelSettingClasses
class ModelSettingClasses(metaclass=SingletonInstance):
    TagImportance: dict[str, Type] = {
        "calc_setting": ModelIndexCalculatorTagImportance,
        "index_setting": ModelIndexSettingTagImportance,
    }
    MSE: dict[str, Type] = {
        "calc_setting": ModelIndexCalculatorMSE,
        "index_setting": ModelIndexSettingMSE,
    }

    def __getitem__(self, key) -> Type[TModelCalcSetting | TModelIndexSetting]:
        return getattr(self, key)

    def get_calc_setting(
        self, calc_type: ModelIndexEnum, calc_setting: dict[str, Any]
    ) -> TModelCalcSetting:
        if calc_type is None:
            return None
        calc_setting_class: Type[TModelCalcSetting] = self[calc_type.name][
            "calc_setting"
        ]
        index_setting_class: Type[TModelIndexSetting] = self[calc_type.name][
            "index_setting"
        ]
        return calc_setting_class.from_dict(index_setting_class, calc_setting.__dict__)


model_setting_classes = ModelSettingClasses()
