from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr, StringConstraints
from typing_extensions import Annotated


class ModelInfoBase(BaseModel):
    modeltype: Annotated[str, StringConstraints(to_lower=True)]
    modelname: Annotated[str, StringConstraints(to_lower=True)]
    system_idx: int
    _modelkey: str = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self._modelkey = f"{self.modeltype}_{self.modelname}"


class ThresholdSetting(BaseModel):
    enable: bool
    threshold: float
    deadband: float

class SPRTSetting(BaseModel):
    enable: bool
    m0: float
    s0: float
    m: float
    s: float
    alpha: float
    beta: float

class SLOPESetting(BaseModel):
    enable: bool
    normal_slope: float = Field(alias="normalSlope")  # 👈 alias만 추가
    deadband: float

class AlarmSetting(BaseModel):
    Threshold: ThresholdSetting = Field(alias="threshold")
    SPRT: SPRTSetting = Field(alias="sprt")
    Slope: SLOPESetting = Field(alias="slope")

class TagSetting(BaseModel):
    tagName: str = Field(default="", alias="tagName")
    indexalarm: bool = Field(default=False, alias="indexAlarm") #####################보류
    indexweight: float = Field(default=1.0, alias="indexWeight")
    noncalc: bool = Field(default=False, alias="nc")       ############################################### non/calc추가
    alarmSetting: AlarmSetting = Field(alias="alarmSetting")


class DataParams(BaseModel):
    duration_start: str = ""
    duration_end: str = ""
    sampling_interval_seconds: int = 5


class ModelParameter(BaseModel):
    data_params: DataParams = DataParams()
    model_params: dict = {}
    calc_method: Optional[str] = None
    sub_models: Optional[list] = None 

 
class ModelSetting(BaseModel):
    modeltype: str = Field(alias="modelType")
    modelname: str = Field(alias="modelName")
    modelkey: str = Field(alias="modelKey")
    description: str
    runningstatus: str = Field(alias="runningStatus")
    trainingstatus: str = Field(alias="trainingStatus")
    tagsettinglist: dict[str, TagSetting] = Field(default_factory=dict, alias="tagSettingList")
    modelparameter: ModelParameter = Field(default_factory=ModelParameter, alias="modelParameter")
    indexcalcmethod: int = Field(default=0, alias="indexCalcMethod")

    indexpriority: bool = Field(default=False, alias="indexPriority")
    
    indexalarm: bool = Field(default=False, alias="indexAlarm")
    indexweight: float = Field(default=1.0, alias="indexWeight")
    noncalc: bool = Field(default=False, alias="nc")
    systemidx: int = Field(default=-1, alias="systemIdx")
    systemname: str = Field(default="", alias="systemName")
    parentidx: int = Field(default=-1, alias="parentIdx")
    version: Optional[int] = None
    calcMethod: Optional[str] = None
# class ModelSetting(BaseModel):
#     modelType: str
#     modelName: str
#     modelKey: str
#     description: str
#     runningStatus: str
#     trainingStatus: str
#     tagSettingList: dict[str, TagSetting] = Field(default_factory=dict)
#     modelParameter: ModelParameter = ModelParameter()
#     indexCalcMethod: int = 0
#     indexPriority: bool = False
#     indexWeight: float = 1.0
#     indexAlarm: bool = False
#     systemIdx: int = -1
#     systemName: str = ""
#     parentIdx: int = -1
#     version: Optional[int] = None

class ModelSettingAPI(BaseModel):
    modelType: Annotated[str, StringConstraints(to_lower=True)]
    modelName: Annotated[str, StringConstraints(to_lower=True)]
    # modelVersion : Annotated[int, StringConstraints(to_lower=True)]
    modelKey: Annotated[str, StringConstraints(to_lower=True)]
