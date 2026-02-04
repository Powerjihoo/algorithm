from typing import Optional

from pydantic import BaseModel, Field, StringConstraints, conlist
from typing_extensions import Annotated

from api_server.models.durations import DataParamsMultipleDuration


class VAEModelParams(BaseModel):
    # window_size: conint(ge=1, le=512, multiple_of=2) = 4
    window_size: Annotated[int, Field(ge=1, le=512)]
    scaler_type: Annotated[str, StringConstraints(strip_whitespace=True)]
    optimizer_type: Annotated[str, StringConstraints(strip_whitespace=True)]


class VAEHyperParams(BaseModel):
    num_epochs: Annotated[int, Field(ge=1, le=10000)]
    batch_size: Annotated[int, Field(ge=1, le=16384, multiple_of=2)]
    learning_rate: Annotated[float, Field(ge=0.00000000001, le=0.5)]
    early_stop_patience: Optional[Annotated[int, Field(ge=1)]]


class ModelSettingVAE(BaseModel):
    tagnames: conlist(str, min_length=1, max_length=100)
    target_tagnames: conlist(str, min_length=1, max_length=100)
    data_params: DataParamsMultipleDuration
    model_params: VAEModelParams
    hyper_params: VAEHyperParams
