from enum import Enum

from pydantic import BaseModel, Field, conlist
from typing_extensions import Annotated

from api_server.models.durations import (
    DataParamsMultipleDuration,
)


class ScalerType(str, Enum):
    MinMaxScaler = "MinMaxScaler"
    StandardScaler = "StandardScaler"


class DistanceMetric(str, Enum):
    euclidean = "Euclidean"
    angular = "Angular"
    manhattan = "Manhattan"
    hamming = "Hamming"
    dot = "Dot"


class ANNOYModelParams(BaseModel):
    window_size: Annotated[int, Field(ge=1, le=512)]
    scaler_type: ScalerType = Field(default=ScalerType.MinMaxScaler)
    distance_metric: DistanceMetric = Field(default=DistanceMetric.euclidean)
    n_nns: Annotated[int, Field(ge=1)]
    n_trees: Annotated[int, Field(ge=2, le=100)]
    n_undersampled_data: int


class ModelSettingAnnoy(BaseModel):
    tagnames: conlist(str, min_length=1, max_length=100)
    data_params: DataParamsMultipleDuration
    model_params: ANNOYModelParams
