import traceback
from typing import List

import numpy as np
from api_server.models.aakr import ModelSettingAAKR
from numpy import array as nparray
from utils.logger import logger

from model import exceptions as ex_model
from model.exceptions import ModelDataCountNotMatchedError
from model.prediction.base_model import PredictionModelBase
from resources.constant import CONST

CONST_PI = np.pi


class AAKR(PredictionModelBase):
    norm_factor_percent = 0.0
    kernel_band_factor = np.sqrt(2)

    def __init__(
        self,
        modelname: str,
        tagnames: List[str],
        calc_interval: int = 1,
        custom_base_path: str = None,
        is_ensemble: bool = False,
        version : int = 0
    ):
        super().__init__(
            modeltype=self.modeltype,
            modelname=modelname,
            tagnames=tagnames,
            targettagnames=tagnames,
            calc_interval=calc_interval,
            custom_base_path=custom_base_path,
            is_ensemble=is_ensemble,
            version=version
        )
        self.window_size = 1
        self.load_model_setting()
        self.setup_model_setting()
        self.load_model()
        self.create_data_array()
        self.verify_model_with_db()
        self.numdata: int = self.model_data.shape[0]
        self.targetcols: np.array = np.ones(self.numtag, dtype=bool)
        self.norm_factors: np.array = self.calc_norm_factor()
        self.input_data_norm: np.array = np.zeros(self.numtag)
        self.model_data_norm: np.array = self.model_data / self.norm_factors

    def __repr__(self):
        return f"type={self.modeltype} name={self.modelname}"

    @property
    def model_file_ext(self):
        return "pkl"

    @property
    def modeltype(self):
        return __class__.__name__

    def load_model_setting(self) -> None:
        model_setting = self.load_json()
        self.model_setting = ModelSettingAAKR(**model_setting)

    def setup_model_setting(self):
        self.tagnames = self.model_setting.tagnames
        self.targettagnames = self.model_setting.tagnames

    def load_model(self):
        try:
            self.model_data = nparray(
                np.load(self.save_path_model, allow_pickle=True).astype(np.float32)
            )
        except FileNotFoundError as err:
            raise ex_model.NotFoundModelFileError(
                model_key=self.model_key, message="Can not find model file"
            ) from err

    def verify_model_with_db(self) -> bool:
        model_data_col = self.model_data.shape[1]
        if model_data_col != self.numtag:
            raise ModelDataCountNotMatchedError(model_data_col, self.numtag)

    def calc_norm_factor(self):

        if getattr(self, "numdata", None) is None:
            self.numdata = self.model_data.shape[0]

        # 데이터가 1행 이하라면 diff 불가능 → 기본값 1.0 고정
        if self.numdata < 2:
            return [1.0] * self.numtag

        model_data_diff = abs(
            np.subtract(
                self.model_data[: self.numdata - 1], self.model_data[1 : self.numdata]
            )
        )
        try:
            diff_creteria = nparray(
                [
                    np.percentile(model_data_diff[:, i], AAKR.norm_factor_percent)
                    for i in range(self.numtag)
                ]
            )
            model_data_diff_filtered = [
                model_data_diff[:, i][
                    np.logical_or(
                        model_data_diff[:, i] > diff_creteria[i],
                        model_data_diff[:, i] != 0,
                    )
                ]
                for i in range(self.numtag)
            ]
            norm_factors = nparray(
                [np.mean(model_data_diff_filtered[i]) for i in range(self.numtag)]
            )
            norm_factors = norm_factors.tolist()
            norm_factors = np.nan_to_num(norm_factors, nan=1)
            return norm_factors
        except IndexError:
            traceback.print_exc()
            logger.warning("Model Data columns are not matched with tagnames")

    def update_targetcols(self, values, status_codes):
        """
        Treating bad quality data
            - if tag has bad value, the tag should not be calculated with good values
            - The tag which is having nan value or bad quality will be excluded from calculation
        If all tag have bad quality values, current model will not calculate learnd data
        """
        _is_good = status_codes >= CONST.STATUSCODE_GOOD
        _is_not_none = values != None
        self.targetcols = _is_good * _is_not_none == True

    def update_input_data_norm(self, values):
        self.input_data_norm = nparray(
            list(
                map(
                    lambda x, y: x / y if x != None else x,
                    # list(self.lastvalues.values()),
                    values,
                    self.norm_factors,
                )
            ),
            dtype=np.float32,
        )

    def inverse_transform(self, learnedvalues: np.array) -> np.array:
        return learnedvalues * self.norm_factors

    def predict(self):
        self.update_input_data_norm()

        input_data = self.input_data_norm[self.targetcols == True]
        model_data = self.model_data_norm
        model_data_for_weight = self.model_data_norm[:, self.targetcols == True]

        _distance = np.sqrt(
            np.power((input_data - model_data_for_weight), 2).sum(axis=1)
        )
        # ! _distance_min = np.amin(_distance)
        _distance_min = _distance.min()
        kb = _distance_min * AAKR.kernel_band_factor if _distance_min != 0 else 0.000001
        weight = (np.exp(-1 * np.power((_distance / kb), 2) / 2)) * (
            np.sqrt(2 * CONST_PI * kb) ** -1
        )
        learnedvalues = ((model_data * weight.reshape(-1, 1)).sum(axis=0)) / weight.sum(
            0
        )

        return nparray(list(map(lambda x, y: x * y, learnedvalues, self.norm_factors)))
