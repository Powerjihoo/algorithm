from queue import Empty as QueueEmpty
from typing import List, Optional

import numpy as np
import pandas as pd
from dbinfo.model_info import ModelInfoManager
from dbinfo.tag_value import ModelTagValueQueue
from utils.logger import logger, logging_time
from utils.scheme.singleton import SingletonInstance

from model.manager.common import PredModelManagerBase
from model.prediction.annoy import ANNOY
from model.prediction.base_model import ModelArrayData, PredictionModelBase
from api_server.models.models import ModelSetting

model_tag_value_queue = ModelTagValueQueue()


class ANNOYManager(PredModelManagerBase, metaclass=SingletonInstance):
    # Observer
    def __init__(self):
        self.cnt_calc = 0

    def __repr__(self):
        return f"{self.modeltype}Manager (cnt={len(self)})"

    @property
    def modeltype(self):
        return "ANNOY"

    def define_prediction_model(
        self, model_key: str, tagnames: List[str], targettagnames: Optional[List[str]],  custom_base_path: Optional[str] = None, 
        # model_info: Optional[ModelSetting] = None
    ) -> PredictionModelBase:
        model_info = ModelInfoManager.get(model_key)
        return ANNOY(model_info.modelname, tagnames, targettagnames,  custom_base_path=custom_base_path, version = model_info.version)
        # if model_info is None:
        #     model_info = ModelInfoManager.get(model_key)
        # return ANNOY(
        #     modelname=model_info.modelname,
        #     tagnames=tagnames,
        #     targettagnames=targettagnames,
        #     custom_base_path=custom_base_path
        # )

    @logging_time
    def calc_preds(self) -> None:
        self.cnt_calc = 0
        for model_key, model in list(self.items()):
            is_data_updated = False
            pred_model: ANNOY = model.pred

            # * Update model data from model data queue
            try:
                model_tag_values = model_tag_value_queue._pop(model_key)
                is_data_updated = pred_model.update_data_raw_pb2(model_tag_values)
                if np.isnan(pred_model.model_tag_data.array_data_raw.values).any():
                    pred_model.model_tag_data.push_data()
            except (QueueEmpty, KeyError):
                pass
            except Exception as e:
                logger.error(e)
                continue

            # * Skip calculation
            if not is_data_updated:
                continue

            data_raw: ModelArrayData = pred_model.model_tag_data.array_data_raw
            pred_model.update_max_timestamp(data_raw.timestamps.max())

            try:
                # * Pred calculation
                pred_values: np.array = pred_model.predict()
                pred_model.update_last_calc_time()  # FIXME: 제거

                if np.isnan(pred_values).any():
                    continue
                pred_model.update_data_pred(
                    max_timestamp=pred_model.max_timestamp,
                    pred_values=pred_values,
                    status_codes=data_raw.statuscodes[:, -1],
                )
                for (
                    tagname,
                    index_calculator,
                ) in pred_model.index_calculators_tag.items():
                    index_calculator.calc_index()
                    pred_model.update_data_index(
                        tagname=tagname,
                        timestamp=index_calculator.data.index.timestamps[-1],
                        index_value=index_calculator.data.index.values[-1],
                        status_code=index_calculator.data.index.statuscodes[-1],
                    )
                if pred_model.index_calculator_model is not None:
                    model_setting: ModelSetting = pred_model.model_setting  

                    active_tagnames = [tag_setting.tagName for tag_setting in model_setting.tagsettinglist.values() if tag_setting.indexalarm]
                    index_ary = np.array([pred_model.model_tag_data.data[tagname].index.values[-1] for tagname in active_tagnames])
                    status_ary = np.array([pred_model.model_tag_data.data[tagname].index.statuscodes[-1] for tagname in active_tagnames])

                    pred_model.index_calculator_model.calc_index(
                        index_ary=index_ary,
                        status_ary=status_ary,
                    )
                
                pred_model.set_calculated()
                self.cnt_calc += 1
            except TypeError:
                continue
            except Exception as e:
                logger.exception(e)
            finally:
                pred_model.model_tag_data.push_data()
        return self.cnt_calc
