from pydantic import BaseModel, Field, StringConstraints, conlist
from typing_extensions import Annotated
from utils.system import regex_date_validation

DURATION_ERROR_MESSAGE = "duration_end must be greater than duration_start"


class _Duration(BaseModel):
    start: Annotated[str, StringConstraints(pattern=regex_date_validation())]
    end: Annotated[str, StringConstraints(pattern=regex_date_validation())]


class DataParamsMultipleDuration(BaseModel):
    sampling_interval_seconds: Annotated[int, Field(gt=1, le=8640000)]
    durations: conlist(_Duration, min_length=1)


class DataParamsSingleDuration(BaseModel):
    duration_start: Annotated[str, StringConstraints(pattern=regex_date_validation())]
    duration_end: Annotated[str, StringConstraints(pattern=regex_date_validation())]
    sampling_interval_seconds: Annotated[int, Field(gt=1, le=8640000)]


class DataParamsDefaultDuration(BaseModel):
    duration_start: Annotated[str, StringConstraints(pattern=regex_date_validation())]
    duration_end: Annotated[str, StringConstraints(pattern=regex_date_validation())]
