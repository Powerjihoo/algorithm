from dataclasses import dataclass
from enum import Enum

import orjson
import redis
from api_client.apis.model import model_api
from api_server.models.models import ModelSetting
from config import settings
from dbinfo import exceptions as ex_db
from pathlib import Path
import json

@dataclass
class _ModelTypeSetting:
    modeltype: str

class ModelType(Enum):
    AAKR = _ModelTypeSetting("AAKR")
    VAE = _ModelTypeSetting("VAE")
    ARIMA = _ModelTypeSetting("ARIMA")
    PDMD = _ModelTypeSetting("PDMD")
    ENSEMBLE = _ModelTypeSetting("ENSEMBLE")


class ModelInfoManager:
    redis_client = redis.StrictRedis(
        host=settings.databases["redis"].host,
        port=settings.databases["redis"].port,
        db=settings.databases["redis"].database,
    )
    redis_client_server_map = redis.StrictRedis(
        host=settings.databases["redis_server_map"].host,
        port=settings.databases["redis_server_map"].port,
        db=settings.databases["redis_server_map"].database,
    )
    model_api_client = model_api
    MODEL_INFO_DIR = Path(settings.data.model_path)

    def __init__(self): ...
    
    @staticmethod
    def _to_base_key(model_key: str) -> str:
        """
        aakr_test111_0001  -> aakr_test111
        aakr_test111       -> aakr_test111 (변화 없음)
        """
        parts = model_key.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return model_key

    @classmethod
    def update(cls, model_key: str, model_info: ModelSetting) -> None:
        cls.redis_client.set(model_key, orjson.dumps(model_info.dict()))

    # 원본 get 코드
    # @classmethod
    # def get(cls, model_key: str) -> ModelSetting:
    #     _model_info = cls.redis_client.get(model_key)
    #     if not _model_info:
    #         raise ex_db.ModelInfoNotExistsError(model_key)
    #     _model_info = orjson.loads(_model_info)
    #     ## _model_info["modelParameter"] = orjson.loads(_model_info["modelParameter"])
    #     if isinstance(_model_info["modelParameter"], str):
    #         _model_info["modelParameter"] = orjson.loads(_model_info["modelParameter"])

    #     return ModelSetting(**_model_info)
        
    @classmethod
    def get(cls, model_key: str) -> ModelSetting:
        
        # --- 버전포함된 키를 무버전 키로 정규화하여 redis 호출,  필요할 때 아래 주석 풀고, _model_info 주석처리하여 교체
        base_key = cls._to_base_key(model_key)
        _model_info = cls.redis_client.get(base_key)
        
        # _model_info = cls.redis_client.get(model_key)
        
        if not _model_info:
            raise ex_db.ModelInfoNotExistsError(model_key)
        _model_info = orjson.loads(_model_info)

        if isinstance(_model_info.get("modelParameter"), str):
            _model_info["modelParameter"] = orjson.loads(_model_info["modelParameter"])

        return ModelSetting(**_model_info)

    

    @classmethod
    def get_tagnames(cls, model_key: str) -> list[str]:
        model_info = cls.get(model_key)
        return list(model_info.tagsettinglist)

    @classmethod
    def get_targettagnames(cls, model_key: str) -> list[str]:
        model_info = cls.get(model_key)
        return [
            tagname
            for tagname in model_info.tagsettinglist
            if model_info.tagsettinglist[tagname]#.TagInfo.is_target == True  # noqa: E712
        ]

    @classmethod
    def get_target_activated_model_keys(cls) -> list[str]:
        server_settings = cls.model_api_client.load_modelinfo_for_each_channel()
        server_settings = server_settings.json()
        server_setting = server_settings[settings.server_id]
        activated_model_keys = server_setting["modelList"]
        
        # ensemble_model_keys = server_setting.get("ensemble_model_list", [])
        # return activated_model_keys, ensemble_model_keys
        return activated_model_keys

    @classmethod
    def get_target_model_keys(cls) -> list[str]:
        server_setting = cls.redis_client_server_map.get(settings.server_id)
        server_setting = orjson.loads(server_setting)
        return server_setting["model_list"]

    @classmethod
    def get_model_setting(cls, model_key: str) -> dict:
        """Return model setting"""
        model_info = cls.get(model_key)
        return model_info.modelParameter.model_params

    @classmethod
    def get_tag_alarm_setting(cls, model_key: str, tagname: str) -> dict:
        model_info = cls.get(model_key)
        return model_info.tagsettinglist[tagname].AlarmSetting
