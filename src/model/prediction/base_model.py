import json
import os
import pickle
import time
import traceback
from abc import abstractmethod
from datetime import datetime
from time import time as time_current
from typing import List, Union, Optional

import numpy as np
import requests
from config import settings
from preprocessing.scaler import MinMaxScaler, StandardScaler
from utils.dataloader import DataLoader
from utils.logger import logger

from model import exceptions as ex_model
from model.base_model import ModelBase
from model.index._types import TModelCalcSetting, TTagIndexSetting

from .._data import ModelArrayData, ModelTagData, TagWindowedData


class PredictionModelBase(ModelBase):
    count = 0
    model_path = settings.data.model_path

    def __init__(
        self,
        modeltype: str,
        modelname: str,
        tagnames: List[str],
        targettagnames: List[str],
        calc_interval: int = 1,
        custom_base_path: Optional[str] = None,
        is_ensemble: bool = False,
        version: int = 0,
    ):
        super().__init__(
            modeltype=modeltype,
            modelname=modelname,
            tagnames=tagnames,
            targettagnames=targettagnames,
            calc_interval=calc_interval,
            version=version
        )

        self.last_calc_time: int = 0
        self.index_calculators_tag: dict[str, TTagIndexSetting] = {}
        self.index_calculator_model: TModelCalcSetting = None
        self.data_raw_old: ModelArrayData = None
        self.model_tag_data: ModelTagData = None
        self.max_timestamp = 0
        self.window_interval = 1
        self.calculated = False
        self.custom_base_path = custom_base_path
        self.is_ensemble = is_ensemble


    def attach_index_calculators_tag(
        self, index_calculators_tag: dict[str, TTagIndexSetting]
    ) -> None:
        self.index_calculators_tag = index_calculators_tag
        for tagname, calculator in self.index_calculators_tag.items():
            calculator.data = self.__get_tag_data(tagname)

    def __get_tag_data(self, tagname: str) -> TagWindowedData:
        return self.model_tag_data.data[tagname]

    def attach_index_calculator_model(
        self, index_calculator_model: TModelCalcSetting
    ) -> None:
        self.index_calculator_model = index_calculator_model

    def create_data_array(self):
        self.data_raw_old = np.full(
            shape=[self.numtag],
            fill_value=np.nan,
            dtype=np.float32,
        )

        data = {}
        for tagname in self.tagnames:
            data[tagname] = TagWindowedData(self.window_size)
        self.model_tag_data = ModelTagData(self.tagnames, data)

    def check_last_calc_time(self) -> bool:
        return time.time() > self.last_calc_time + self.calc_interval

    def update_last_calc_time(self) -> None:
        self.last_calc_time = time.time()

    def set_calculated(self) -> None:
        self.calculated = True

    @property
    def _calc_result_tags(self) -> list[dict]:
        """
        result = []
        for tagname, tag_data in self.model_tag_data.data.items():
            _result_tag = {
                "tagname": tagname,
                "pred": tag_data.pred.values[-1],
                "index": tag_data.index.values[-1],
                "status_pred": tag_data.pred.statuscodes[-1],
                "status_index": tag_data.index.statuscodes[-1],
            }
            result.append(_result_tag)
        Returns:
            list[dict]
        """
        result = [
            {
                "tagname": tagname,
                "pred": tag_data.pred.values[-1],
                "index": tag_data.index.values[-1],
                "status_pred": tag_data.pred.statuscodes[-1],
                "status_index": tag_data.index.statuscodes[-1],
            }
            for tagname, tag_data in self.model_tag_data.data.items()
        ]
        return result

    @property
    def calc_result_model(self) -> dict:
        if self.calculated:
            result = {
                "model_key": self.model_key,
                "timestamp": self.max_timestamp,
                "index": self.index_calculator_model.data.index,
                "status_index": self.index_calculator_model.data.status_index,
                "data": self._calc_result_tags,
            }
            self.calculated = False
            return result
        else:
            return {}

    @property
    def time_now(self):
        return time_current() * 1000

    @property
    def base_path(self) -> str:
        if self.custom_base_path: # esemble path 존재하면 경로 조정
            return os.path.join(self.model_path, self.custom_base_path, self.model_key)
        return os.path.join(PredictionModelBase.model_path, f"{self.model_key}")

    @property
    def save_path_model(self) -> str:
        if isinstance(self.model_file_ext, str):
            return os.path.join(
                self.base_path,
                f"{self.model_key}_model.{self.model_file_ext}",
            )

    @property
    def save_path_json(self) -> str:
        return os.path.join(self.base_path, f"{self.model_key}_info.json")

    @property
    def save_path_scaler(self) -> str:
        return os.path.join(self.base_path, f"{self.model_key}_scaler.pkl")

    @property
    @abstractmethod
    def model_file_ext(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def modeltype(self):
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def load_model_setting(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def setup_model_setting(self):
        raise NotImplementedError

    def load_json(self) -> dict:
        try:
            __data = open(self.save_path_json, "r").read()
            model_setting = json.loads(__data)
            return model_setting
        except FileNotFoundError as err:
            raise ex_model.ModelInitializeError(
                model_key=self.model_key, message="Can not find model info file"
            ) from err

    def load_scaler(self) -> None:
        try:
            with open(self.save_path_scaler, "rb") as f:
                self.scaler: MinMaxScaler | StandardScaler = pickle.load(f)
        except FileNotFoundError as err:
            raise ex_model.ModelInitializeError(
                model_key=self.model_key, message="Can not find model scaler file"
            ) from err

    def load_initial_history_data(self, current_time, interval) -> None:
        """Fill lastvalues by loading historian data from influxdb"""
        # TODO: NEED THREAD LOCK?

        try:
            start_time = current_time - interval * self.window_size
        except AttributeError:
            start_time = current_time - interval

        try:
            loaded_data = DataLoader.load_from_influx(
                tagnames=self.tagnames,
                start=start_time,
                end=current_time,
                method="archive",
            )

            loaded_data_df = DataLoader.convert_dict_to_df_resample(
                data=loaded_data,
                sampling_interval_seconds=interval,
                offset_hour=0,
            )
            df_filter = (
                loaded_data_df.index > datetime.utcfromtimestamp(start_time)
            ) & (loaded_data_df.index < datetime.utcfromtimestamp(current_time))
            loaded_data_df = loaded_data_df[df_filter]
            loaded_data_df = loaded_data_df[self.tagnames]

            if loaded_data_df.size == 0:
                logger.warning("No data to load initialize data from influxdb.")
                return

            self.lastvalues[0] = np.array(loaded_data_df)
            self.timestamps = np.array(
                [loaded_data_df.index.astype(int) / 1_000_000] * self.numtag
            ).T
            self.statuscodes = np.full(
                shape=[self.window_size, self.numtag], fill_value=192, dtype=int
            )

            if self.modeltype == "ARIMA":
                self.model_apply()

        except requests.exceptions.ConnectionError as err:
            logger.error(
                "Connection Error. Can not access to IPCM Server "
                f"to load historian data\n"
                f"{err.args[0]}\n"
            )
        except IndexError:
            logger.error(
                "Historian data length is not matched "
                f"with model window_size model_key={self.model_key}"
            )

    def load_initial_history_data_archive(self, current_time):
        """Fill lastvalues by loading historian data from influxdb"""
        # TODO: NEED THREAD LOCK?

        start_time = current_time - self.window_time

        try:
            loaded_data = DataLoader.load_from_influx(
                tagnames=self.tagnames,
                start=start_time,
                end=current_time,
                method="archive",
            )

            for tag_data in loaded_data:
                idx_col = self.tagnames.index(tag_data["tagName"])

                for idx_row, values in enumerate(tag_data["valueList"]):
                    self.lastvalues[0][idx_row][idx_col] = values["value"]
                    self.timestamps[idx_row][idx_col] = values["unixTimestamp"]
                    self.statuscodes[idx_row][idx_col] = values["statusCodeEnum"]

            logger.debug(f"Historian data loaded model_key={self.model_key}")

        except requests.exceptions.ConnectionError as err:
            logger.error(
                "Connection Error. Can not access to IPCM Server "
                f"to load historian data\n"
                f"{err.args[0]}\n"
            )
        except IndexError:
            logger.error(
                "Historian data length is not matched "
                f"with model window_size model_key={self.model_key}"
            )

    def save_scaler(self) -> None:
        self.scaler.save(self.save_path_scaler)

    def update_max_timestamp(self, max_timestamp) -> None:
        self.max_timestamp = max_timestamp

    def update_data_raw_pb2(self, model_tag_values) -> bool:
        updated = False
        self.model_tag_data.array_data_raw_old = self.model_tag_data.array_data_raw
        for tag_value in model_tag_values:
            try:
                # * 마지막 timestamp 보다 window interval 이후의 데이터가 들어와야 업데이트
                if (
                    tag_value.timestamp
                    < self.max_timestamp + self.window_interval * 1000
                ):
                    continue
                if tag_value.status_code < 192:
                    continue

                self._update_lastdata(
                    tagname=tag_value.tagname,
                    value=tag_value.value,
                    statuscode=tag_value.status_code,
                    timestamp=tag_value.timestamp,
                    ignore_status=tag_value.is_ignored,
                )
            except Exception as e:
                logger.error(e)
        if np.any(
            self.model_tag_data.array_data_raw_old.values
            != self.model_tag_data.array_data_raw.values
        ):
            updated = True
        return updated

    def update_data_raw(self, tag_values: list) -> None:
        for tag_value in tag_values:
            try:
                self._update_lastdata(
                    tagname=tag_value.tagName,
                    value=tag_value.rawValue.value,
                    statuscode=tag_value.rawValue.statusCodeEnum,
                    timestamp=tag_value.rawValue.unixTimestamp,
                    ignore_status=False,  # FIXME
                )
            except Exception as e:
                logger.error(e)

    def _update_lastdata(
        self,
        tagname: str,
        value: Union[int, float],
        statuscode: int,
        timestamp: int,
        ignore_status: bool = False,
    ) -> None:
        """
        Put new data to lastvalue, statuscode, timestamp array
        The order of tag (in lastvalues column) is follow self.tagnames

        Args:
            tagname (str): Any tagname in this model istance
            value (Union[int, float])
            statuscode (int): GOOD: 192, BAD: 4
            timestamp (int): unixtimestamp (ms)
        """

        try:
            model_tag_data_raw: TagWindowedData = self.model_tag_data.data[tagname].raw

            model_tag_data_raw.values[-1] = value
            model_tag_data_raw.statuscodes[-1] = statuscode
            model_tag_data_raw.timestamps[-1] = timestamp
            model_tag_data_raw.ignore_statuses[-1] = ignore_status
        except ValueError:
            logger.debug(f"{tagname=} is not in this model({self.model_key})")
        except IndexError:
            logger.debug(f"Index not matched {self.model_key}, {tagname=}")
        except Exception:
            traceback.print_exc()

    def update_data_pred(
        self, max_timestamp: int, pred_values: np.array, status_codes: np.array
    ) -> None:
        """Put new learned value"""
        for tagname, value, status_code in zip(
            self.tagnames, pred_values, status_codes
        ):
            self.model_tag_data.data[tagname].pred.update_data(
                timestamp=max_timestamp, value=value, statuscode=status_code
            )

    def update_data_index(
        self, tagname: str, timestamp: int, index_value: np.array, status_code: np.array
    ) -> None:
        """Put new learned value"""
        self.model_tag_data.data[tagname].index.update_data(
            timestamp=timestamp, value=index_value, statuscode=status_code
        )

    def preprop_data_shift(self, shift_value):
        """Shift data before update lastvalue"""
        # TODO: get shift_value from model tag settings
        raise NotImplementedError
