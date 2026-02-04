import numpy as np
from api_server.models.arima import ModelSettingARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from utils.logger import logger

from model.prediction.base_model import PredictionModelBase

# from model.exceptions import ...


class ARIMA(PredictionModelBase):
    def __init__(self, modelname: str, tagname: str, calc_interval: int = 5):
        super().__init__(
            modeltype=self.modeltype,
            modelname=modelname,
            tagnames=tagname,
            targettagnames=tagname,
            calc_interval=calc_interval,
        )
        self.load_model_setting()
        self.setup_model_setting()
        self.create_data_array()
        self.load_model()
        self.sigma = np.sqrt(self.model.params[-1])

    def __repr__(self):
        return f"type={self.modeltype} name={self.modelname}"

    def setup_model_setting(self):
        self.window_size = self.model_setting.model_params.window_size

    @property
    def model_file_ext(self):
        return "pkl"

    @property
    def modeltype(self):
        return __class__.__name__

    def load_model_setting(self) -> None:
        model_setting = self.load_json()
        self.model_setting = ModelSettingARIMA(**model_setting)

    def load_model(self):
        try:
            self.model = ARIMAResults.load(self.save_path_model)
        except FileNotFoundError:
            logger.warning(f"Could not find model file {self.save_path_model}")

    def model_apply(self, values):
        self.model = self.model.apply(values)

    def predict(self):
        if np.isnan(lastvalues := self.model_tag_data.array_data_raw.values).any():
            return np.array([np.nan])
        self.model_apply(lastvalues)
        return self.model.predict(-1)
