import pickle
import warnings

import numpy as np
from api_server.models.ica import ModelSettingICA
from sklearn.decomposition import FastICA
from utils.logger import logger

from model import exceptions as ex_model
from model.prediction.base_model import PredictionModelBase

warnings.filterwarnings("ignore", category=UserWarning)


class ICA(PredictionModelBase):
    preproc_normal_range_factor = 0.1

    def __init__(self, modelname, tagnames, targettagnames,
                custom_base_path: str = None,
                is_ensemble: bool = False,
                version : int = 0):
        super().__init__(
            modeltype=self.modeltype,
            modelname=modelname,
            tagnames=tagnames,
            targettagnames=targettagnames,
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

    def __repr__(self):
        return f"type={self.modeltype} name={self.modelname}"

    def get_tagnames(self):
        return self.tagnames

    def get_targettagnames(self):
        return self.targettagnames

    @property
    def model_file_ext(self):
        return "pkl"

    @property
    def modeltype(self):
        return __class__.__name__

    def load_model_setting(self) -> None:
        model_setting = self.load_json()
        self.model_setting = ModelSettingICA(**model_setting)

    def setup_model_setting(self):
        __model_params = self.model_setting.model_params
        self.tagnames = self.model_setting.tagnames
        self.targettagnames = self.model_setting.tagnames
        self.scaler_type = __model_params.scaler_type
        self.whiten = __model_params.whiten
        self.n_components = __model_params.n_components
        self.window_size = __model_params.window_size

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

    def define_model(self) -> None:
        self.model = FastICA(
            n_components=self.n_components, random_state=0, whiten=self.whiten
        )

    def load_model(self):
        try:
            self.define_model()
            with open(self.save_path_model, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError as err:
            raise ex_model.NotFoundModelFileError(
                model_key=self.model_key, message="Can not find model file"
            ) from err

    def predict(self) -> np.array:
        if np.isnan(lastvalues := self.model_tag_data.array_data_raw.values).any():
            return np.array([np.nan])
        try:
            data_scaled = self.scaler.transform(lastvalues)
            transform_data = self.model.transform(data_scaled)

            result = np.dot(transform_data, self.model.mixing_.T) + self.model.mean_
            pred = self.scaler.inverse_transform(result[-1])

            return pred
        except Exception as e:
            logger.warning(e)
