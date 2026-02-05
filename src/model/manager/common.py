import os
import time
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import ClassVar, Dict, List, Optional, Union

import numpy as np
import requests
from tqdm import tqdm

from _protobuf.model_data_pb2 import ToIPCM
from api_server.models.models import ModelSetting
from config import settings
from dbinfo import exceptions as ex_db
from dbinfo.model_info import ModelInfoManager
from model import exceptions as ex_model
from model.index._types import TAlarmSetting, TModelCalcSetting, TTagIndexSetting
from model.index.model._index_setting import ModelIndexEnum
from model.index.model.setting_selector import model_setting_classes
from model.index.tag._alarm_setting import AlarmSettingsClasses
from model.index.tag._tag_data import TagData
from model.index.tag.setting import TagIndexCalculatorClasses
from model.prediction.base_model import PredictionModelBase
from types_ import TAlarmSettingDB, TAlarmType
from utils.logger import logger
from utils.scheme.singleton import SingletonInstance
from utils.system import measure_memory

alarm_setting_classes = AlarmSettingsClasses()
tag_setting_classes = TagIndexCalculatorClasses()


@dataclass
class Model:
    num_models: ClassVar[int] = 0
    pred: PredictionModelBase

    def __post_init__(self):
        Model.num_models += 1

    def __repr__(self):
        return f"Model({self.pred.model_key.upper()})"

    def __del__(self):
        Model.num_models -= 1

    def calc_index_tags(self):
        self.index_calculators_tag.calc_index


class PredModelManagerBase(dict):
    """
    This class acts as a OBSERVER in the observer pattern.
    And this class is abstract class of Model Manager

    This class is managing running models as a dictionary
    Example:
    {
        "model_key1":Model1,
        "model_key2":Model2,
        ...
    }
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def modeltype(self):
        raise NotImplementedError

    @abstractmethod
    def define_prediction_model(
        self,
        model_key: str,
        tagnames: List[str],
        targettagnames: Optional[List[str]],
        custom_base_path: Optional[str] = None
        # model_info: Optional[ModelSetting] = None
    ) -> PredictionModelBase:
        raise NotImplementedError

    def get_running_model_list(self) -> List[str]:
        """Return model list"""
        return self.keys()

    def check_model_running(self, model_key: str) -> bool:
        return model_key in self

    def get_model_keys_by_tagname(self, tagname: str) -> List[str]:
        return [
            model_key for model_key in self if tagname in self[model_key].pred.tagnames
        ]

    def create_alarm_setting_data(
        self, alarm_type: TAlarmType, alarm_setting: TAlarmSettingDB
    ) -> "TAlarmSetting":
        return alarm_setting_classes.get_alarm_setting(alarm_type, alarm_setting)

    def update_model_setting(self, model_key: str, model_setting: ModelSetting) -> None:
        if model_key not in self:
            return

        model: Model = self[model_key]
        model.pred.model_setting = model_setting
        index_calculators_tag = self.__create_index_calculators_tag(model_setting)
        index_calculator_model = self.__create_index_calculator_model(model_setting)

        model.pred.attach_index_calculators_tag(index_calculators_tag)
        model.pred.attach_index_calculator_model(index_calculator_model)

    @measure_memory
    def add_model(self, model_key: str, model_setting: ModelSetting,
                  custom_base_path: Optional[str] = None,) -> None:
        tagnames = ModelInfoManager.get_tagnames(model_key)
        target_tagnames = ModelInfoManager.get_targettagnames(model_key)
        prediction_model: PredictionModelBase = self.define_prediction_model(
            model_key=model_key,
            tagnames=tagnames,
            targettagnames=target_tagnames,
        )
        prediction_model.model_setting = model_setting
        if (
            prediction_model.modeltype == "PDMD"
        ):  # if prediction_model.modeltype == "PDMD" or prediction_model.window_size > 1:
            prediction_model.load_initial_history_data(
                time.time(), prediction_model.calc_interval
            )
        index_calculators_tag = self.__create_index_calculators_tag(model_setting)
        index_calculator_model = self.__create_index_calculator_model(model_setting)
        prediction_model.attach_index_calculators_tag(index_calculators_tag)
        prediction_model.attach_index_calculator_model(index_calculator_model)

        self[model_key] = Model(pred=prediction_model)

    @measure_memory
    def __create_index_calculators_tag(
        self, model_setting
    ) -> dict[str, TTagIndexSetting]:
        index_calculators_tag = {}
        for tagname, tag_setting in model_setting.tagsettinglist.items():
            alarm_type, alarm_setting = get_enabled_alarm_type(tag_setting.alarmSetting)
            alarm_setting_data: TAlarmSetting = self.create_alarm_setting_data(
                alarm_type, alarm_setting
            )
            index_calculators_tag[
                tagname
            ]: TTagIndexSetting = self.__create_tag_setting_data(
                tagname, alarm_type, alarm_setting_data
            )

        return index_calculators_tag

    def __create_index_calculator_model(self, model_setting) -> TModelCalcSetting:
        model_index_calc_setting = parse_model_setting(model_setting)

        for _calc_type, _calc_setting in model_index_calc_setting.items():
            index_calculator_model: TModelCalcSetting = (
                model_setting_classes.get_calc_setting(_calc_type, _calc_setting)
            )
            break
        return index_calculator_model

    def __create_tag_setting_data(
        self,
        tagname: str,
        alarm_type: TAlarmType,
        alarm_setting_data: TAlarmSetting,
    ) -> TTagIndexSetting:
        tag_setting_class: type[TTagIndexSetting] = tag_setting_classes[alarm_type]
        return tag_setting_class(tagname, TagData(), alarm_setting_data)

    def remove_model(self, model_key: str) -> None:
        try:
            del self[model_key]
        except IndexError:
            logger.warning(f"Model is not running {model_key}")

    @property
    def num_models(self) -> int:
        return len(self)

    def check_modelfile(self, model_key: str) -> None:
        target_path = os.path.join(settings.data.model_path, model_key)
        if not os.path.isdir(target_path) or not os.listdir(target_path):
            raise ex_model.NotFoundModelFileError(
                model_key, message=f"Can not find model: {target_path}"
            )

    def remove_model_in_main_proc(self, model_key: str) -> None:
        """
        When error occur while initializing the model,
        request to remove model info in APIProcAgent to the main process
        """
        url = f"http://127.0.0.1:{settings.servers['this'].port}/system/{model_key}"
        requests.delete(url)

    def define_models_initial(self, model_info_list: dict) -> None:
        if not model_info_list:
            return None

        num_initialize_model = 0
        pbar = tqdm(total=len(model_info_list))
        for model_key, model_setting in model_info_list.items():
            try:
                pbar.set_description(f"Loading model... {model_key:30} ")
                self.check_modelfile(model_key)
                self.add_model(model_key=model_key, model_setting=model_setting)
                pbar.update(1)
                num_initialize_model += 1
            except (
                ex_model.NotFoundModelFileError,
                ex_model.ModelDataCountNotMatchedError,
                ex_model.InvalidSPRTParameterError,
                ex_model.ModelInitializeError,
                ex_db.ModelInfoNotExistsError,
            ) as err:
                logger.error(err)
                pbar.set_description(f"Failed to Loading model... ({model_key:30}) ")
            except Exception as err:
                pbar.set_description(f"Failed to Loading model... ({model_key:30}) ")
                logger.error(
                    f"{'NOT Loaded':12} | Unexpected error occurs {model_key=}"
                )
                logger.exception(err)
        if model_info_list:
            logger.info(
                f"Model initializing completed ({num_initialize_model}/{len(model_info_list)} models)"
            )

    def create_calc_result_updated_only(self) -> dict:
        model: Model
        pred_model: PredictionModelBase
        response = ToIPCM()
        for model in list(self.values()):
            try:
                pred_model = model.pred
                if pred_model.calc_result_model:
                    model_data = response.model_data.add()
                    model_data.model_key = pred_model.model_key
                    model_data.timestamp = pred_model.max_timestamp
                    model_data.index = pred_model.index_calculator_model.data.index
                    model_data.status_index = (
                        pred_model.index_calculator_model.data.status_index
                    )
                    for (
                        tagname,
                        model_tag_data,
                    ) in pred_model.model_tag_data.data.items():
                        tag_data = model_data.data.add()
                        tag_data.tagname = tagname
                        tag_data.pred = model_tag_data.pred.values[-1]
                        tag_data.index = model_tag_data.index.values[-1]
                        tag_data.status_pred = model_tag_data.pred.statuscodes[-1]
                        tag_data.status_index = model_tag_data.index.statuscodes[-1]
            except Exception as e:
                logger.warning(f"{e}, {model.pred.model_key}")
        return response

    @abstractmethod
    def calc_preds(self) -> None:
        raise NotImplementedError


class ModelAgent(metaclass=SingletonInstance):
    """
    This class acts as a SUBJECT in the observer pattern.
    Each observer(model manager) object is registered in this class, and when a specific event occurs,
    it has each observer to executes a specific method
    """

    def __init__(self) -> None:
        self.observers: Dict[str, PredModelManagerBase] = {}
        
        # 들어오는 데이터에서 _0001 제외하고 들어오는거 반영
        self._alias_map = {}     # { base_key: versioned_key }
        self._lock = RLock()

    def get_observers(self) -> Dict[str, PredModelManagerBase]:
        """
        Return observer instances information

        Returns:
           observer instances dict: Dict[modeltype(str):observer(PredModelManagerBase)]
        """
        return self.observers.items()

    def get_model_managers(self) -> List[Model]:
        return self.observers.values()

    def get_modeltypes(self) -> List[str]:
        return self.observers.keys()

    def get_model(self, model_key: str) -> Union[Model, None]:
        for _observer in self.observers.values():
            if model_key in _observer:
                return _observer[model_key]
        return None

    @measure_memory
    def remove_model(self, model_key: str) -> None:
        for _observer in self.observers.values():
            _observer: PredModelManagerBase
            with suppress(KeyError):
                del _observer[model_key]
                
                # model : model_version dict 제거하는 코드
                base_key = "_".join(model_key.split("_")[:-1])
                self._alias_del_if_match(base_key, model_key)
                
                logger.info(f"Model Deactivated | {model_key=}")
                
                return

        raise ex_model.NotFoundActivatedModelError(model_key)
    
    @staticmethod
    def parse_versioned_model_key(raw_key: str) -> str:
        parts = raw_key.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid model_key format: {raw_key}")
        
        base_key = parts[0]
        version = int(parts[1])
        padded_version = f"{version:04d}"

        return f"{base_key}_{padded_version}"

    @staticmethod
    def check_modelfile(model_key: str) -> None:
        target_path = os.path.join(settings.data.model_path, model_key)
        if not os.path.isdir(target_path) or not os.listdir(target_path):
            raise ex_model.NotFoundModelFileError(model_key)

    def check_model_running(self, model_key: str) -> bool:
        for _observer in self.observers.values():
            _observer: PredModelManagerBase
            if _observer.check_model_running(model_key):
                return True
        return False

    def get_model_list(self) -> list:
        _model_list = []
        for _observer in self.observers.values():
            _observer: PredModelManagerBase
            _models = _observer.get_running_model_list()
            _model_list.extend(_models)

        return _model_list

    def update_model_setting(
        self,
        modeltype: str,
        model_key: str,
    ):
        """Notify to call "update_model_setting" method to target ModelManager object"""
        try:
            _observer: PredModelManagerBase = self.observers[modeltype.upper()]
            model_info = ModelInfoManager.get(model_key)
            _observer.update_model_setting(model_key, model_info)
            logger.info(f"{'Model Updated':12} | {model_key=}\n{' ':>15}")

        except KeyError:
            raise ValueError(f"Not supported model type {modeltype}")
        
    # 기존 코드
    def add_model(self, model_key: str):
        """Add model using only model_key, auto-extract modeltype"""
        try:
            # modeltype 추출
            modeltype = model_key.split("_")[0].upper()

            _observer: PredModelManagerBase = self.observers[modeltype]
            model_info = ModelInfoManager.get(model_key)
            _observer.add_model(model_key, model_info)

            logger.info(f"{'Model Loaded':12} | {model_key=}\n{' ':>15}")
            
            # alias 반영: 'aakr_test111_0002' -> base 'aakr_test111'
            base_key = "_".join(model_key.split("_")[:-1])
            self._alias_set(base_key, model_key)

        except KeyError:
            raise ValueError(f"Not supported model type: {modeltype}")
    
    # ------ 버전 없는 실시간 데이터 -> 버전 모델키  ------- 관리
    def get_versioned_for_base(self, base_key: str) -> Optional[str]:
        """consumer가 큐 넣기 직전에 호출: base_key -> 현재 활성 versioned key"""
        return self._alias_map.get(base_key)
    
    def _alias_set(self, base_key: str, versioned_key: str) -> None:
        """run/add 시 호출: 사본 교체(copy-on-write)로 읽기 경합 최소화"""
        with self._lock:
            m = self._alias_map.copy()
            m[base_key] = versioned_key
            self._alias_map = m

    def _alias_del_if_match(self, base_key: str, versioned_key: str) -> None:
        """stop/remove 시 호출: base가 아직 이 버전을 가리킬 때만 제거"""
        with self._lock:
            cur = self._alias_map.get(base_key)
            if cur != versioned_key:
                return
            m = self._alias_map.copy()
            m.pop(base_key, None)
            self._alias_map = m
            
    def get_alias_map(self) -> dict[str, str]:
        """현재 가지고 있는 base→versioned dict 반환"""
        with self._lock:
            return dict(self._alias_map)

    
    def register(self, observer: PredModelManagerBase) -> None:
        """
        Register ModelManager object in observers dictionary
        self.observers = {
            "modeltypeA":AModelManager,
            "modeltypeB":BModelManager,
            ...
        }
        Args:
            observer (PredModelManagerBase): ModelManager object
                                        (AAKRManager, VAEManager, ...)
        """
        if observer.modeltype in self.observers.keys():
            logger.warning(f"This observer is already registered {observer.modeltype}")
        else:
            self.observers[observer.modeltype] = observer
            logger.debug(f"Success to registering observer {observer.modeltype}")

    def unregister(self, observer: PredModelManagerBase) -> None:
        """
        Unregister ModelManager object from observers dictionary

        Args:
            observer (PredModelManagerBase): ModelManager object
        """
        if observer.modeltype in self.observers.keys():
            del self.observers[observer.modeltype]
            logger.debug(f"Success to unregister observer {observer.modeltype}")
        else:
            logger.warning(
                f"This observer is not registered in agent {observer.modeltype}"
            )

    @abstractmethod
    def calc_preds(self) -> None: ...


@dataclass
class IndexCalcMethod:
    enable: bool
    weights_ary: np.array
    # importances_ary: np.array


def get_enabled_alarm_type(
    alarm_settings: dict[TAlarmType, TAlarmSettingDB],
) -> tuple[TAlarmType, TAlarmSettingDB] | tuple[None, None]:
    for alarm_type, alarm_setting in alarm_settings.__dict__.items():
        if alarm_setting.enable:
            return alarm_type, alarm_setting
    return None, None


def parse_model_setting(
    model_setting: ModelSetting,
) -> dict[ModelIndexEnum, IndexCalcMethod]:
    index_calc_method: ModelIndexEnum = ModelIndexEnum.get_member_by_value(
        model_setting.indexcalcmethod
    )
    index_calc_enabled = not model_setting.noncalc  # noncalc가 False이면 index_calc_enabled는 True
    active_tag_settings = [tag_setting for tag_setting in model_setting.tagsettinglist.values() if not tag_setting.noncalc]
    weights_ary = np.array([tag_setting.indexweight for tag_setting in active_tag_settings])
    # importances_ary = np.array([tag_setting.indexpriority for tag_setting in active_tag_settings])

    model_calc_setting = {
        index_calc_method: IndexCalcMethod(
            enable=index_calc_enabled,
            weights_ary=weights_ary,
            # importances_ary=importances_ary,
        )
    }
    return model_calc_setting


