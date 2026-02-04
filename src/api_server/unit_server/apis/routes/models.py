import time
from model.prediction.base_model import PredictionModelBase
import os
from enum import auto

from api_server import exceptions as ex_api
from api_server.unit_server.apis.examples import models as model_examples
from fastapi import APIRouter, Body, status
from fastapi.responses import JSONResponse
from model import exceptions as ex_model
from model.manager.common import Model, ModelAgent
from model.manager.ensemble import EnsembleAgent
from preprocessing.scaler import MinMaxScaler
from pydantic import StringConstraints
from typing_extensions import Annotated
from utils.logger import logger
from utils.scheme.strenum import StrEnum
from api_client.apis.model import model_api

router = APIRouter()
model_agent = ModelAgent()
ensemble_agent = EnsembleAgent()


class ModelRunningStatue(StrEnum):
    RUNNING = auto()
    STOPPED = auto()

def is_ensemble_model(model_key: str) -> bool:
    """Check if the model is an ensemble model based on model_key."""
    modeltype = model_key.split("_")[0].lower()
    return modeltype == "ensemble"

def _split_model_key(model_key: str) -> tuple[str, str]:
    """
    {modeltype}_{modelname}(_{version}) → (modeltype, modelname)
    예: aakr_test_0001 -> ("AAKR", "test")
        aakr_test      -> ("AAKR", "test")
    """
    parts = model_key.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid model_key: {model_key}")

    modeltype = parts[0].upper()
    modelname = "_".join(parts[1:-1]) if parts[-1].isdigit() else "_".join(parts[1:])

    return modeltype, modelname


# 실행 중인 ensemble 모델 리스트 확인
@router.get("/ensemble_list", summary="Get ensemble model list which are running on algorithm server")
async def get_ensemble_model_list():
    ensemble_model_list = ensemble_agent.get_model_keys()
    return {
        "message": "Requested ensemble model list is sent",
        "detail": {"model_list": ensemble_model_list},
    }


@router.get("/alias", summary="Get current base→versioned alias map")
async def get_alias_map():
    """
    현재 알고리즘 서버에 등록된 모델 기준,
    base_key를 versioned_key로 라우팅하는 alias dict 조회.
    예) {"aakr_test111": "aakr_test111_0002", ...}
    """
    aliases = model_agent.get_alias_map()
    return {
        "message": "Requested alias map is sent",
        "detail": {"aliases": aliases, "count": len(aliases)},
    }

@router.get("", summary="Get model list which are running on algorithm server")
async def get_model_list():
    model_list = model_agent.get_model_list()

    logger.debug(f"Requested model list is sent | Number of models: {len(model_list)}")
    return {
        "message": "Requested model list is sent",
        "detail": {"model_list": model_list},
    }

@router.post("/{model_key}")
async def activate_model(
    model_key: Annotated[str, StringConstraints(to_lower=True)],
):
    start_time = time.time()
    try:
        # Ensemble 모델 체크
        if is_ensemble_model(model_key):
            # Ensemble 모델은 버전이 없으므로 parse하지 않음
            if ensemble_agent.check_model_running(model_key):
                return JSONResponse(
                    content={
                        "message": "Requested ensemble model is already activated",
                        "detail": {
                            "status": str(ModelRunningStatue.RUNNING),
                            "model_key": model_key,
                        },
                    },
                    status_code=status.HTTP_200_OK,
                )
            
            # Ensemble 모델 파일 체크 (간단한 파일 시스템 체크)
            ensemble_path = os.path.join(PredictionModelBase.model_path, model_key)
            if not os.path.exists(ensemble_path):
                raise ex_model.NotFoundModelFileError(model_key, "Ensemble folder not found")
            
            # Ensemble 모델 추가
            ensemble_agent.add_model(model_key=model_key)
            
            # IPCM 실행 요청
            try:
                mt, mn = _split_model_key(model_key)
                res = model_api.post_run_model(mt, mn)
                logger.debug(f"[IPCM RUN] {mt}_{mn} -> code={res.status_code} body={getattr(res, 'text', '')}")
            except Exception as e:
                logger.warning(f"IPCM RUN notify failed: {e}")
                
        else:
            # 단일 모델 처리 (기존 로직)
            model_key = model_agent.parse_versioned_model_key(model_key) # aakr_test_1 -> aakr_test_0001
            if model_agent.check_model_running(model_key):
                return JSONResponse(
                    content={
                        "message": "Requested model is already activated",
                        "detail": {
                            "status": str(ModelRunningStatue.RUNNING),
                            "model_key": model_key,
                        },
                    },
                    status_code=status.HTTP_200_OK,
                )

            model_agent.check_modelfile(model_key)
            model_agent.add_model(model_key=model_key)

            elapsed = time.time() - start_time

            #IPCM 실행 요청
            try:
                mt, mn = _split_model_key(model_key)
                res = model_api.post_run_model(mt, mn)
                logger.debug(f"[IPCM RUN] {mt}_{mn} -> code={res.status_code} body={getattr(res, 'text', '')}")
                
            except Exception as e:
                logger.warning(f"IPCM RUN notify failed: {e}")

        logger.info(f"activate_model: model={model_key}, elapsed={elapsed:.3f}s")

        return JSONResponse(
            content={
                "message": "Requested model has been activated",
                "detail": {
                    "status": str(ModelRunningStatue.RUNNING),
                    "model_key": model_key,
                },
            },
            status_code=status.HTTP_201_CREATED,
        )

    except Exception as err:
        # versioned_key가 정의되지 않은 경우를 대비
        safe_key = locals().get("model_key", model_key)
        logger.exception(err)
        raise ex_api.CanNotActivateModel(safe_key, str(err)) from err



@router.delete("/{model_key}")
async def deactivate_model(model_key: Annotated[str, StringConstraints(to_lower=True)]):
    try:
        # Ensemble 모델 체크
        if is_ensemble_model(model_key):
            # Ensemble 모델은 버전이 없으므로 parse하지 않음
            ensemble_agent.remove_model(model_key)
            
            # IPCM stop 실행 요청
            try:
                mt, mn = _split_model_key(model_key)
                res = model_api.post_stop_model(mt, mn)
                logger.debug(f"[IPCM STOP] {mt}_{mn} -> code={res.status_code} body={getattr(res, 'text', '')}")
            except Exception as e:
                logger.warning(f"IPCM STOP notify failed: {e}")
        else:
            # 단일 모델 처리 (기존 로직)
            model_key = model_agent.parse_versioned_model_key(model_key)
            model_agent.remove_model(model_key)

            # IPCM stop 실행 요청
            try:
                mt, mn = _split_model_key(model_key)
                res = model_api.post_stop_model(mt, mn)
                logger.debug(f"[IPCM STOP] {mt}_{mn} -> code={res.status_code} body={getattr(res, 'text', '')}")
            except Exception as e:
                logger.warning(f"IPCM STOP notify failed: {e}")

    except ex_model.NotFoundActivatedModelError:
        return JSONResponse(
            content={
                "message": "Requested model is already deactivated",
                "detail": {
                    "status": str(ModelRunningStatue.STOPPED),
                    "model_key": model_key,
                },
            },
            status_code=status.HTTP_200_OK,
        )
    except Exception as err:
        raise ex_api.CanNotDeactivateModel(model_key) from err
    return JSONResponse(
        content={
            "message": "Requested model has been deactivated",
            "detail": {
                "status": str(ModelRunningStatue.STOPPED),
                "model_key": model_key,
            },
        },
        status_code=status.HTTP_200_OK,
    )


# @router.post("/{model_key}/setting", summary="Update model alarm setting")
# async def update_model_setting(
#     model_key: Annotated[str, StringConstraints(to_lower=True)],
#     model_setting: ModelSettingAPI,
# ):
#     model_agent.update_model_setting(model_setting.modelType, model_key)

@router.post("/{model_key}/setting", summary="Update model alarm setting")
async def update_model_setting(
    model_key: Annotated[str, StringConstraints(to_lower=True)],
):
    """
    Path: aakr_test1234_1  -> 내부에서 aakr_test1234_0001 로 변환
    Body: 없음 (activate와 동일 구조)
    """
    try:
        # Ensemble 모델 체크
        if is_ensemble_model(model_key):
            # Ensemble 모델 설정 업데이트
            ensemble_agent.update_model_setting(model_key)
        else:
            # 단일 모델 처리 (기존 로직)
            # 버전 정규화
            versioned_key = model_agent.parse_versioned_model_key(model_key)
            modeltype = versioned_key.split("_", 1)[0].lower()

            # 설정 업데이트
            model_agent.update_model_setting(modeltype, versioned_key)

    except Exception as err:
        raise ex_api.CanNotUpdateModelAlarmSetting(model_key, message=str(err))
    
    # 성공 응답
    return JSONResponse(
        content={
            "message": "Model setting updated successfully",
        },
        status_code=status.HTTP_200_OK,
    )                                        



@router.get("/{model_key}/scaler")
async def get_model_scaler(model_key: Annotated[str, StringConstraints(to_lower=True)]):
    if not model_agent.check_model_running(model_key):
        raise ex_api.ModelAlreadyDeactivated(model_key)
    model: Model = model_agent.get_model(model_key)
    scaler: MinMaxScaler = model.pred.scaler
    result = {
        tagname: (d_min, d_max)
        for tagname, d_min, d_max in zip(
            scaler.tagnames_input, scaler.data_input_min, scaler.data_input_max
        )
    }

    return {"message": "Requested model scaler has been sent", "detail": result}


@router.get("/{model_key}/scaler/{tagname}")
async def get_model_scaler_tagname(
    model_key: Annotated[str, StringConstraints(to_lower=True)], tagname: str
):
    if not model_agent.check_model_running(model_key):
        raise ex_api.ModelAlreadyDeactivated(model_key)
    model: Model = model_agent.get_model(model_key)
    scaler: MinMaxScaler = model.pred.scaler

    try:
        try:
            _idx = scaler.tagnames_input.index(tagname)
        except ValueError:
            try:
                _idx = scaler.tagnames_input.index(tagname)
            except ValueError as err:
                raise ex_model.NotFoundTagError(tagname) from err

        data_input_min = scaler.data_input_min[_idx]
        data_input_max = scaler.data_input_max[_idx]
        data_output_min = scaler.data_output_min[_idx]
        data_output_max = scaler.data_output_max[_idx]

    except ex_model.NotFoundTagError as err:
        raise ex_api.CanNotFindTagName(tagname, err.args[0]) from err

    except Exception as err:
        logger.exception(err)

    logger.debug(f"Requested model scaler information has been sent | {tagname=}")
    return {
        "message": "Requested model tag scaler has been sent",
        "detail": {
            "data_input_min": data_input_min,
            "data_input_max": data_input_max,
            "data_output_min": data_output_min,
            "data_output_max": data_output_max,
        },
    }


@router.post(
    "/{model_key}/scaler/{tagname}",
    summary="Update model scaler setting for requested tag",
)
async def update_model_scaler_tagname(
    model_key: Annotated[str, StringConstraints(to_lower=True)],
    tagname: str,
    scaler_setting: dict = Body(None, example=model_examples.scaler_setting),
):
    if not model_agent.check_model_running(model_key):
        raise ex_api.ModelAlreadyDeactivated(model_key)
    model: Model = model_agent.get_model(model_key)
    scaler: MinMaxScaler = model.pred.scaler

    try:
        try:
            _idx = scaler.tagnames_input.index(tagname)
        except ValueError:
            try:
                _idx = scaler.tagnames_input.index(tagname)
            except ValueError as err:
                raise ex_model.NotFoundTagError(tagname) from err

        data_input_min_prev = scaler.data_input_min[_idx]
        data_input_max_prev = scaler.data_input_max[_idx]
        data_output_min_prev = scaler.data_output_min[_idx]
        data_output_max_prev = scaler.data_output_max[_idx]

        scaler.update(
            tagname,
            scaler_setting["data_input_min"],
            scaler_setting["data_input_max"],
            # scaler_setting["data_output_min"],
            scaler_setting["data_input_min"],
            # scaler_setting["data_output_max"],
            scaler_setting["data_input_max"],
        )
        model.pred.save_scaler()

        logger.debug(f"Requested model scaler information has been sent | {tagname=}")
        return {
            "message": "Requested model scaler has been updated",
            "detail": {
                "data_input_min": f"{data_input_min_prev} -> {scaler.data_input_min[_idx]}",
                "data_input_max": f"{data_input_max_prev} -> {scaler.data_input_max[_idx]}",
                "data_output_min": f"{data_output_min_prev} -> {scaler.data_output_min[_idx]}",
                "data_output_max": f"{data_output_max_prev} -> {scaler.data_output_max[_idx]}",
            },
        }

    except ex_model.NotFoundTagError as err:
        raise ex_api.CanNotFindTagName(tagname, err.args[0]) from err

    except Exception as err:
        logger.exception(err)


@router.get("/{model_key}/currentvalue")
async def get_model_values(model_key: Annotated[str, StringConstraints(to_lower=True)]):
    # sourcery skip: inline-immediately-returned-variable
    if not model_agent.check_model_running(model_key):
        raise ex_api.ModelAlreadyDeactivated(model_key)
    model: Model = model_agent.get_model(model_key)
    logger.debug(f"Requested model values are sent | {model_key=}")
    if isinstance(model.pred.learnedvalues, list):
        learnedvalues = model.pred.learnedvalues
    else:
        learnedvalues = model.pred.learnedvalues.tolist()
    res = {
        "message": "Requested model values are sent",
        "detail": {
            "modeltype": model.pred.modeltype,
            "modelname": model.pred.modelname,
            "tagnames": model.pred.tagnames,
            "target_tagnames": model.pred.targettagnames,
            "data": {
                "lastvalues": model.pred.lastvalues[0][-1].tolist(),
                "learnedvalues": learnedvalues,
            },
        },
    }
    return res
