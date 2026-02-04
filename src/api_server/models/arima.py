from pydantic import BaseModel, Field
from typing_extensions import Annotated

from api_server.models.durations import DataParamsSingleDuration


class ARIMAModelParams(BaseModel):
    p: Annotated[int, Field(ge=1, le=5)]
    d: Annotated[int, Field(ge=0, le=5)]
    q: Annotated[int, Field(ge=1, le=5)]
    window_size: Annotated[int, Field(ge=1)]


class ModelSettingARIMA(BaseModel):
    tagname: str
    data_params: DataParamsSingleDuration
    model_params: ARIMAModelParams
