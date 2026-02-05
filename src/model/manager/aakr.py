from queue import Empty as QueueEmpty
from typing import List, Optional

import numpy as np
import pandas as pd

from api_server.models.models import ModelSetting
from dbinfo.model_info import ModelInfoManager
from dbinfo.tag_value import ModelTagValueQueue
from model.manager.common import PredModelManagerBase
from model.prediction import algo_calc
from model.prediction.aakr import AAKR
from model.prediction.base_model import ModelArrayData, PredictionModelBase
from utils.logger import logger, logging_time
from utils.scheme.singleton import SingletonInstance

model_tag_value_queue = ModelTagValueQueue()


class AAKRManager(PredModelManagerBase, metaclass=SingletonInstance):
    # Observer
    def __init__(self):
        self.cnt_calc = 0

    def __repr__(self):
        return f"{self.modeltype}Manager (cnt={len(self)})"

    @property
    def modeltype(self):
        return "AAKR"

    def define_prediction_model(
        self,
        model_key: str,
        tagnames: List[str],
        targettagnames: Optional[List[str]],
        custom_base_path: Optional[str] = None
    ) -> PredictionModelBase:
        model_info = ModelInfoManager.get(model_key)
        return AAKR(model_info.modelname, tagnames, custom_base_path=custom_base_path, version = model_info.version)

    @logging_time
    def calc_preds(self) -> None:
        self.cnt_calc = 0
        for model_key, model in list(self.items()):
            is_data_updated = False
            pred_model: AAKR = model.pred

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
                pred_model.update_targetcols(
                    data_raw.values[-1], data_raw.statuscodes[-1]
                )
                if not any(pred_model.targetcols):
                    logger.debug(f"NO CALC | {model_key} have no data or bad quality")
                    continue
                pred_model.update_input_data_norm(data_raw.values[-1])
                pred_model.update_max_timestamp(pred_model.max_timestamp)

                values_pred = algo_calc.aakr(
                    input_data=pred_model.input_data_norm[
                        pred_model.targetcols == True  # noqa: E712
                    ],
                    model_data=pred_model.model_data_norm,
                    model_data_for_weight=pred_model.model_data_norm[
                        :, pred_model.targetcols == True  # noqa: E712
                    ],
                )
                pred_model.update_last_calc_time()
                values_pred = pred_model.inverse_transform(values_pred)
                pred_model.update_data_pred(
                    max_timestamp=pred_model.max_timestamp,
                    pred_values=values_pred,
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

                    active_tagnames = [tag_setting.tagName for tag_setting in model_setting.tagsettinglist.values() if not tag_setting.noncalc]
                    index_ary = np.array([pred_model.model_tag_data.data[tagname].index.values[-1] for tagname in active_tagnames])
                    status_ary = np.array([pred_model.model_tag_data.data[tagname].index.statuscodes[-1] for tagname in active_tagnames])

                    pred_model.index_calculator_model.calc_index(
                        index_ary=index_ary,
                        status_ary=status_ary,
                        priority=model_setting.indexpriority
                    )

                pred_model.set_calculated()
                self.cnt_calc += 1
            except ZeroDivisionError:
                logger.debug(f"NO CALC | {model_key} have nan data")
            except Exception as e:
                logger.exception(e)
            finally:
                pred_model.model_tag_data.push_data()

        return self.cnt_calc
