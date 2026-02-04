from enum import auto
from typing import Final

import numpy as np
from api_client.apis.tagvalue import tagvalue_api
from api_server.models.pdmd import ModelSettingPDMD
from utils.logger import logger
from utils.scheme.strenum import StrEnum

from model import exceptions as ex_model
from model.prediction.base_model import PredictionModelBase

TIMECONVERSION_MILLISEC2DAY: Final[int] = 86400000


class State(StrEnum):
    NORMAL = auto()
    DROPPED = auto()
    RISING = auto()


class PDMD(PredictionModelBase):
    def __init__(self, modelname: str, tagname: str, calc_interval: int = 1):
        """
        Altough window Size of PDMD model is not "1",
        This model initialize(parent class) with window size = 1.
        Actual window size is applied on other attribute (self.data)
        Because PDMD model does not use raw data, not sampled data
        """

        super().__init__(
            modeltype=self.modeltype,
            modelname=modelname,
            tagnames=tagname,
            targettagnames=tagname,
            calc_interval=calc_interval,
        )
        self.window_size = 1
        self.load_model_setting()
        self.setup_model_setting()
        self.create_data_array()
        self.data = np.zeros(shape=(0, 2))
        self.reactivate_ts = None

    def load_model_setting(self) -> None:
        model_setting = self.load_json()
        self.model_setting = ModelSettingPDMD(**model_setting)

    def __repr__(self):
        return f"type={self.modeltype} name={self.modelname}"

    @property
    def model_file_ext(self):
        return None

    @property
    def modeltype(self):
        return __class__.__name__

    @property
    def oldest_ts(self):
        return self.data[0, 0]

    @property
    def last_ts(self):
        return self.data[-1, 0]

    @property
    def last_prev_ts(self):
        return self.data[-2, 0]

    @property
    def oldest_value(self):
        return self.data[0, 1]

    @property
    def last_value(self):
        return self.data[-1, 1]

    @property
    def last_prev_value(self):
        return self.data[-2, 1]

    @property
    def diff1(self):
        """Return latest diffirence"""
        return self.data[-1][1] - self.data[-2][1]

    @property
    def diff2(self):
        """Return latest 2nd diffirence"""
        return self.data[-1][1] - 2 * self.data[-2][1] + self.data[-3][1]

    @property
    def slope(self):
        def moving_average(values: np.array, window_size: int):
            return np.convolve(values, np.ones(window_size), "valid") / window_size

        X = moving_average(self.data[:, 0], 10)
        Y = moving_average(self.data[:, 1], 10)

        return (
            ((X * Y).mean() - X.mean() * Y.mean())
            / ((X**2).mean() - (X.mean()) ** 2)
            * TIMECONVERSION_MILLISEC2DAY
        )

    def setup_model_setting(self) -> None:
        try:
            __model_params = self.model_setting.model_params
            self.normal_slope = __model_params.normal_slope
            self.diff_threshold_ub = __model_params.diff_threshold_ub
            self.diff_threshold_lb = __model_params.diff_threshold_lb
            self.wait_sec = __model_params.wait_sec
            self.window_time = __model_params.window_time
        except KeyError as err:
            raise ex_model.UndefinedModelSettingError(f"{self.model_key}") from err
        except Exception as err:
            raise ex_model.UndefinedModelSettingError from err

    def load_model(self):
        """PDMD does not need model file"""
        return None

    def push_value(self):
        """Push value so that it has only data as much as window time"""
        try:
            while self.oldest_ts < self.time_now - self.window_time * 1000:
                self.data = np.delete(self.data, 0, axis=0)
            if self.data[0, 0] != self.time_now - self.window_time * 1000:
                self.data = np.insert(
                    self.data,
                    0,
                    [[self.time_now - self.window_time * 1000, self.data[0, 1]]],
                    axis=0,
                )

            new_data = [[self.timestamps[0][-1], self.lastvalues[0][-1][-1]]]
            self.data = np.append(self.data, new_data, axis=0)
        except IndexError:
            """Add initial data from lastvalues if no data in self.data attribute"""
            self.data = np.append(
                self.data,
                [[self.time_now - self.window_time * 1000, self.data[0, 1]]],
                axis=0,
            )
        except Exception as e:
            logger.exception(e)

    def judge_state(self):
        try:
            if self.diff_threshold_lb <= self.diff2 <= self.diff_threshold_ub:
                return State.NORMAL
            elif self.diff2 < self.diff_threshold_lb:
                return State.DROPPED
            else:  # self.diff2 > self.diff_threshold_ub
                return State.RISING
        except IndexError:
            return State.NORMAL

    def load_initial_history_data(self, current_time, interval):
        """The reason why this method overrided -> PDMD use archive data"""
        """Fill lastvalues by loading historian data from influxdb"""
        start_time = current_time - self.window_time
        try:
            __data_influx = tagvalue_api.get_historian_value_archive(
                start=start_time,
                end=current_time,
                tagnames=self.tagnames,
            ).json()

            for tag_datas in __data_influx:
                for data in tag_datas["valueList"]:
                    self.data = np.append(
                        self.data,
                        [
                            [
                                data["unixTimestamp"],
                                data["value"],
                            ]
                        ],
                        axis=0,
                    )

            logger.debug(f"Historian data loaded model_key={self.model_key}")
            # self.learnedvalues = [self.data[-1][1]]
            self.state = self.judge_state()
        except IndexError:
            logger.error(
                f"Historian data length is not matched with model window_time model_key={self.model_key}"
            )

    def predict(self):
        self.push_value()
        self.update_max_timestamp()

        try:
            if self.state == State.NORMAL:
                if self.diff_threshold_lb <= self.diff2 <= self.diff_threshold_ub:
                    return self.calc_ev()
                elif self.diff2 < self.diff_threshold_lb:
                    self.state = State.DROPPED

            elif self.state == State.DROPPED:
                if self.diff2 > self.diff_threshold_ub:
                    self.state = State.RISING
                    self.reactivate_ts = self.last_ts + self.wait_sec
                return self.lastvalues[0][-1]
            else:  # state == State.RISING
                try:
                    if self.last_ts >= self.reactivate_ts:
                        self.state = State.NORMAL
                        self.reactivate_ts = None
                    return self.lastvalues[0][-1]
                except Exception:
                    return
        except IndexError:
            logger.debug(f"Not enough data to predict {self.model_key=}")
            return self.lastvalues[0][-1]
        except AttributeError:
            logger.debug(f"Initial state is not defined yet {self.model_key=}")
            self.state = self.judge_state()

    def calc_ev(self):
        weightedAVG = np.trapz(self.data[:, 1], self.data[:, 0]) / (
            self.data[-1, 0] - self.data[0, 0]
        )
        self.learnedvalues = np.array(
            [
                weightedAVG
                + (self.normal_slope * 0.5)
                * ((self.last_ts - self.oldest_ts) / TIMECONVERSION_MILLISEC2DAY)
            ]
        )

        return self.learnedvalues
