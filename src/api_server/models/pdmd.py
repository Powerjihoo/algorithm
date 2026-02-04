from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from api_server.models.durations import DataParamsDefaultDuration


class PDMDModelParamsTrain(BaseModel):
    wait_sec: Annotated[int, Field(ge=0, le=86400)]
    window_time: Annotated[int, Field(ge=1, le=86400)]
    normal_slope: float
    diff_threshold_ub: float
    diff_threshold_lb: float


class PDMDModelParamsTest(BaseModel):
    wait_sec: Annotated[int, Field(ge=0, le=86400)]
    diff_threshold_lb: float
    diff_threshold_ub: float
    normal_slope: float
    window_time: Annotated[int, Field(ge=30, le=432000)]


class ModelSettingPDMD(BaseModel):
    Optional[str]
    tagname: str
    data_params: DataParamsDefaultDuration
    model_params: PDMDModelParamsTrain
