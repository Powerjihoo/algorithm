from enum import Enum

from pydantic import BaseModel, Field, conlist
from typing_extensions import Annotated

from api_server.models.durations import DataParamsMultipleDuration


class ScalerType(str, Enum):
    MinMaxScaler = "MinMaxScaler"
    StandardScaler = "StandardScaler"


class ICAWhitenType(str, Enum):
    Unit_Variance = "unit-variance"
    Arbitrary_Variance = "arbitrary-variance"
    False_Value = False


class ICAModelParams(BaseModel):
    scaler_type: ScalerType = Field(default=ScalerType.MinMaxScaler)
    n_components: Annotated[int, Field(ge=0)]
    whiten: ICAWhitenType = Field(default=ICAWhitenType.Unit_Variance)
    window_size: int = 1


class ModelSettingICA(BaseModel):
    tagnames: conlist(str, min_length=1, max_length=100)
    data_params: DataParamsMultipleDuration
    model_params: ICAModelParams


class TestICA(BaseModel):
    tagnames: conlist(str, min_length=1, max_length=100)
    data_params: DataParamsMultipleDuration
