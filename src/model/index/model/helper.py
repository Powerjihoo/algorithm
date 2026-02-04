from dataclasses import dataclass

import numpy as np

from api_server.models.models import ModelSetting
from model.index.model._index_setting import ModelIndexEnum


@dataclass
class IndexCalcMethod:
    enable: bool
    weights_ary: np.array
    importances_ary: np.array

def parse_model_setting(
    model_setting: ModelSetting,
) -> dict[ModelIndexEnum, IndexCalcMethod]:
    index_calc_method: ModelIndexEnum = ModelIndexEnum.get_member_by_value(
        model_setting.indexcalcmethod
    )
    index_calc_enabled = model_setting.indexalarm
    active_tag_settings = [tag_setting for tag_setting in model_setting.tagsettinglist.values() if tag_setting.indexalarm]
    weights_ary = np.array([tag_setting.indexweight for tag_setting in active_tag_settings])
    
    # model_info에서 priority체크해서 True면 전체 tag에 priority 1로 설정
    importances_ary = np.array([tag_setting.indexpriority for tag_setting in active_tag_settings])

    model_calc_setting = {
        index_calc_method: IndexCalcMethod(
            enable=index_calc_enabled,
            weights_ary=weights_ary,
            importances_ary=importances_ary,
        )
    }
    return model_calc_setting

