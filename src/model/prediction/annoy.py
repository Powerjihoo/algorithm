import numpy as np
from annoy import AnnoyIndex
from api_server.models.annoy import ModelSettingAnnoy
from utils.logger import logger

from model import exceptions as ex_model
from model.prediction.base_model import PredictionModelBase


class ANNOY(PredictionModelBase):
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
        return "ann"

    @property
    def modeltype(self):
        return __class__.__name__

    def load_model_setting(self) -> None:
        model_setting = self.load_json()
        self.model_setting = ModelSettingAnnoy(**model_setting)

    def setup_model_setting(self):
        __model_params = self.model_setting.model_params
        self.tagnames = self.model_setting.tagnames
        self.targettagnames = self.model_setting.tagnames
        self.window_size = __model_params.window_size
        self.window_interval = 5
        self.scaler_type = __model_params.scaler_type
        self.n_nns = __model_params.n_nns
        self.n_trees = __model_params.n_trees
        self.distance_metric = __model_params.distance_metric

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
        self.model = AnnoyIndex(
            f=self.window_size * len(self.tagnames),
            metric=self.model_setting.model_params.distance_metric.lower(),
        )

    def load_model(self):
        try:
            self.define_model()
            self.model.load(self.save_path_model)
        except (FileNotFoundError, OSError) as err:
            raise ex_model.NotFoundModelFileError(
                model_key=self.model_key, message="Can not find model file"
            ) from err

    def predict(self) -> np.array:
        if np.isnan(lastvalues := self.model_tag_data.array_data_raw.values).any():
            return np.array([np.nan])

        try:
            data_scaled = np.nan_to_num(self.scaler.transform(lastvalues))
            vectorized_data = self._vectorize(data_scaled).flatten()
            vector_idxs, distances = self.model.get_nns_by_vector(
                vector=vectorized_data, n=self.n_nns, include_distances=True
            )

            if 0 in distances:
                matched_index = vector_idxs[distances.index(0)]
                result_vector = np.array(self.model.get_item_vector(matched_index))
            else:
                target_vectors = np.array(
                    [
                        self.model.get_item_vector(vector_idx)
                        for vector_idx in vector_idxs
                    ]
                )
                distances = np.array(distances)
                result_vector = np.sum(
                    target_vectors.T * (1 / distances), axis=1
                ) / np.sum(1 / distances)

            result = result_vector.reshape(self.window_size, -1)[-1]
            pred = self.scaler.inverse_transform(result)

            return pred
        except Exception as e:
            logger.warning(e)

    def _vectorize(self, x: np.array):
        windowed_data = np.lib.stride_tricks.sliding_window_view(
            x,
            self.window_size,
            axis=0,
        )

        reshaped_data = windowed_data.transpose(0, 2, 1).reshape(
            -1, self.window_size * len(self.tagnames)
        )

        return reshaped_data
