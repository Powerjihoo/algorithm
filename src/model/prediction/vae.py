from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from api_server.models.vae import ModelSettingVAE
from dbinfo.model_info import ModelInfoManager
from utils.logger import logger

from model import exceptions as ex_model
from model.prediction.base_model import PredictionModelBase

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class VAE(PredictionModelBase):
    preproc_normal_range_factor = 0.1

    def __init__(
        self,
        modelname: str,
        tagnames: list[str],
        targettagnames: list[str],
        calc_interval: int = 5,
        custom_base_path: str = None,
        is_ensemble: bool = False,
        version : int = 0
    ):
        super().__init__(
            modeltype=self.modeltype,
            modelname=modelname,
            tagnames=tagnames,
            targettagnames=targettagnames,
            calc_interval=calc_interval,
            custom_base_path=custom_base_path,
            is_ensemble=is_ensemble,
            version=version
        )
        self.load_model_setting()
        self.setup_model_setting()
        self.create_data_array()
        self.load_model()
        self.load_scaler()
        self._load_tagnames_from_scaler()
        # self.define_prepro_normal_range()

    def __repr__(self):
        return f"type={self.modeltype} name={self.modelname}"

    def get_tagnames(self):
        return self.tagnames

    def get_targettagnames(self):
        return self.targettagnames

    @property
    def model_file_ext(self):
        return "tf"

    @property
    def modeltype(self):
        return __class__.__name__

    def load_model_setting(self) -> None:
        model_setting = self.load_json()
        self.model_setting = ModelSettingVAE(**model_setting)

    def setup_model_setting(self):
        __model_params = self.model_setting.model_params
        self.window_size = __model_params.window_size
        self.window_interval = 5
        self.scaler_type = __model_params.scaler_type
        self.optimizer_type = __model_params.optimizer_type
        self.tagnames = self.model_setting.tagnames
        self.targettagnames = self.model_setting.target_tagnames

    def _load_tagnames_from_scaler(self) -> None:
        __tagname = self.tagnames[0]
        if __tagname not in self.scaler.tagnames_input:
            # prefix 없어
            _prefix = __tagname[: __tagname.find("-") + 1]
            _tagnames_input_new = [
                f"{_prefix}{_tagname}" for _tagname in self.scaler.tagnames_input
            ]

            self.scaler.tagnames_input = _tagnames_input_new
            _tagnames_output_new = [
                f"{_prefix}{_tagname}" for _tagname in self.scaler.tagnames_output
            ]
            self.scaler.tagnames_output = _tagnames_output_new

        if set(self.tagnames) != set(self.scaler.tagnames_input):
            raise ValueError("Not matches tagnames between setting json and scaler")
        if set(self.targettagnames) != set(self.scaler.tagnames_output):
            raise ValueError(
                "Not matches target_tagnames between setting json and scaler"
            )

        self.tagnames = self.scaler.tagnames_input
        self.targettagnames = self.scaler.tagnames_output

    def load_settings(self) -> None:
        try:
            __setting = ModelInfoManager.get_model_setting(self.model_key)
            self.window_size = __setting["window_size"]

        except Exception as err:
            raise ex_model.UndefinedModelSettingError from err

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.save_path_model, compile=False)
            _layer = self.model.layers[0]
            model_input_shape: Tuple[Union[None, int], int, int]
            if hasattr(_layer, "_batch_input_shape"):
                model_input_shape = _layer._batch_input_shape
            elif hasattr(_layer, "_build_input_shape"):
                model_input_shape = tuple(_layer._build_input_shape)
            else:
                raise ValueError(
                    "Can not check the input shape. Check the tensorflow version of model"
                )

            if model_input_shape[1:] != (self.window_size, self.numtag):
                raise ex_model.ModelInitializeError(
                    model_key=self.model_key,
                    message=(
                        f"Model batch input shape is not matched. Model="
                        f"{model_input_shape[1:]}, {self.window_size=}, {self.numtag=}"
                    ),
                )
        except OSError as err:
            raise ex_model.ModelInitializeError(
                model_key=self.model_key,
                message=f"Could not find model file {self.save_path_model}",
            ) from err
        except AttributeError as err:
            logger.exception(err)
            raise ex_model.ModelInitializeError(
                model_key=self.model_key,
                message="Could not load model",
            ) from err

    def define_preproc_normal_range(self):
        """Define normal data range for each tag"""
        # TODO: Need to apply
        for min, max in self.ref_df_min, self.ref_df_max:
            if min > 0:
                self.preprop_normal_range.append(
                    min * (1 - VAE.preproc_normal_range_factor),
                    max * (1 + VAE.preproc_normal_range_factor),
                )
            elif max < 0:
                self.preprop_normal_range.append(
                    min * (1 + VAE.preproc_normal_range_factor),
                    max * (1 - VAE.preproc_normal_range_factor),
                )
            elif max > 0:
                self.preprop_normal_range.append(
                    min * (1 + VAE.preproc_normal_range_factor),
                    max * (1 + VAE.preproc_normal_range_factor),
                )
            else:
                raise ex_model.InvalidRangeError

    def preprop_normal_range(self):
        """Mutate input data when invalid data come
        in order to prevent learned value hunting"""
        # TODO: Need to apply
        ...

    def predict(self) -> np.array:
        """Return prediction result"""
        if np.isnan(lastvalues := self.model_tag_data.array_data_raw.values).any():
            return np.array([np.nan])

        # TODO: preprocessing raw values
        # preprop_normal_range()
        # transform_calc = self.model.predict(self.scaler.transform(self.lastvalues))

        try:
            try:  # StandardScaler
                lastvalues_clipped = np.clip(
                    lastvalues,
                    self.scaler.data_input_mean - self.scaler.data_input_std * 9,
                    self.scaler.data_input_mean + self.scaler.data_input_std * 9,
                )
            except AttributeError:  # MinMaxScaler
                lastvalues_clipped = np.clip(
                    lastvalues,
                    self.scaler.data_input_min,
                    self.scaler.data_input_max,
                )
            values = np.nan_to_num(self.scaler.transform(lastvalues_clipped))
            transform_calc = self.model(
                values.reshape(1, self.window_size, -1), training=False
            )
            calc_inverse = self.scaler.inverse_transform(transform_calc)

            return calc_inverse[0, -1, :].numpy()
        except ValueError as e:
            logger.warning(e)
