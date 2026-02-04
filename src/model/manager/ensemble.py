from queue import Empty as QueueEmpty
from typing import Optional

import numpy as np
from loguru import logger

from dbinfo.model_info import ModelInfoManager
from dbinfo.tag_value import ModelTagValueQueue
from utils.logger import logging_time
from utils.scheme.singleton import SingletonInstance

# Ensemble 모델의 alarm, index, setting 기능을 위한 import 추가
from model.manager.common import (
    alarm_setting_classes,
    tag_setting_classes,
    get_enabled_alarm_type,
    parse_model_setting,
    model_setting_classes,
)
from model.index.tag._tag_data import TagData
from model.index._types import TAlarmSetting, TModelCalcSetting, TTagIndexSetting
from api_server.models.models import ModelSetting

from model.prediction.ensemble import ENSEMBLE

model_tag_value_queue = ModelTagValueQueue()


class EnsembleAgent(metaclass=SingletonInstance):
    """Ensemble Agent with real-time calculation and Redis auto-restore."""

    def __init__(self) -> None:
        self.models = {}  # {model_key: ENSEMBLE}
        self.cnt_calc = 0

    def add_model(self, model_key: str) -> None:
        """Add ensemble model to the agent."""
        try:
            # Redis에서 모델 정보 가져오기
            model_info = ModelInfoManager.get(model_key)
            
            # 태그 이름들 가져오기
            tagnames = ModelInfoManager.get_tagnames(model_key)
            target_tagnames = ModelInfoManager.get_targettagnames(model_key)
            
            # Redis에서 calc_method 가져오기
            # 우선순위: 1) ModelSetting.calcMethod, 2) modelParameter.calc_method, 3) 기본값 "average"
            calc_method = "average"
            
            # 최우선: ModelSetting의 calcMethod (Redis에서 직접)
            if hasattr(model_info, 'calcMethod'):
                calc_method_value = getattr(model_info, 'calcMethod', None)
                if calc_method_value and isinstance(calc_method_value, str) and calc_method_value.strip():
                    calc_method = calc_method_value.strip()
                    logger.debug(f"Using calc_method from ModelSetting.calcMethod: {calc_method}")
            # 차선: modelParameter의 calc_method
            if calc_method == "average" and hasattr(model_info, 'modelparameter') and model_info.modelparameter:
                if hasattr(model_info.modelparameter, 'calc_method'):
                    param_calc_method = getattr(model_info.modelparameter, 'calc_method', None)
                    if param_calc_method and isinstance(param_calc_method, str) and param_calc_method.strip():
                        calc_method = param_calc_method.strip()
                        logger.debug(f"Using calc_method from modelParameter: {calc_method}")
            
            # ENSEMBLE 객체 생성 (ModelSetting 전달)
            ensemble = ENSEMBLE(
                modelname=model_info.modelname,
                tagnames=tagnames,
                targettagnames=target_tagnames,
                calc_method=calc_method,
                custom_base_path=model_key,  # ensemble 모델의 폴더 경로
                model_info=model_info,  # Redis에서 가져온 ModelSetting 전달
            )
            
            # Ensemble 모델에도 index, setting 설정 추가
            self._setup_ensemble_model_setting(ensemble, model_info)
            
            # 모델 저장
            self.models[model_key] = ensemble
            
            logger.info(f"Ensemble model added: {model_key}")
            
        except Exception as e:
            logger.error(f"Failed to add ensemble model {model_key}: {e}")
            raise

    def _setup_ensemble_model_setting(self, ensemble: ENSEMBLE, model_info: ModelSetting) -> None:
        """Ensemble 모델의 alarm, index, setting 설정"""
        try:
            # Tag index calculators 설정
            index_calculators_tag = self._create_index_calculators_tag(model_info)
            ensemble.attach_index_calculators_tag(index_calculators_tag)
            
            # Model index calculator 설정
            index_calculator_model = self._create_index_calculator_model(model_info)
            ensemble.attach_index_calculator_model(index_calculator_model)
            
            logger.info(f"Ensemble model setting completed for {ensemble.model_key}")
            
        except Exception as e:
            logger.error(f"Failed to setup ensemble model setting: {e}")
            raise

    def _create_index_calculators_tag(self, model_setting: ModelSetting) -> dict[str, TTagIndexSetting]:
        """Tag index calculators 생성 (단일 모델과 동일한 로직)"""
        index_calculators_tag = {}
        for tagname, tag_setting in model_setting.tagsettinglist.items():
            alarm_type, alarm_setting = get_enabled_alarm_type(tag_setting.alarmSetting)
            alarm_setting_data: TAlarmSetting = self._create_alarm_setting_data(
                alarm_type, alarm_setting
            )
            index_calculators_tag[
                tagname
            ]: TTagIndexSetting = self._create_tag_setting_data(
                tagname, alarm_type, alarm_setting_data
            )

        return index_calculators_tag

    def _create_index_calculator_model(self, model_setting: ModelSetting) -> TModelCalcSetting:
        """Model index calculator 생성 (단일 모델과 동일한 로직)"""
        model_index_calc_setting = parse_model_setting(model_setting)

        for _calc_type, _calc_setting in model_index_calc_setting.items():
            index_calculator_model: TModelCalcSetting = (
                model_setting_classes.get_calc_setting(_calc_type, _calc_setting)
            )
            break
        return index_calculator_model

    def _create_alarm_setting_data(
        self, alarm_type, alarm_setting
    ) -> "TAlarmSetting":
        """Alarm setting data 생성 (단일 모델과 동일한 로직)"""
        return alarm_setting_classes.get_alarm_setting(alarm_type, alarm_setting)

    def _create_tag_setting_data(
        self,
        tagname: str,
        alarm_type,
        alarm_setting_data: TAlarmSetting,
    ) -> TTagIndexSetting:
        """Tag setting data 생성 (단일 모델과 동일한 로직)"""
        tag_setting_class: type[TTagIndexSetting] = tag_setting_classes[alarm_type]
        return tag_setting_class(tagname, TagData(), alarm_setting_data)

    def update_model_setting(self, model_key: str) -> None:
        """Ensemble 모델의 setting 업데이트 (Redis에서 최신 정보 읽기)"""
        if model_key not in self.models:
            logger.warning(f"Ensemble model {model_key} not found for setting update")
            return

        ensemble: ENSEMBLE = self.models[model_key]
        
        # Redis에서 최신 ModelSetting 가져오기
        model_info = ModelInfoManager.get(model_key)
        
        # 기존 설정 제거
        ensemble.index_calculators_tag = {}
        ensemble.index_calculator_model = None
        
        # 새로운 설정 적용
        self._setup_ensemble_model_setting(ensemble, model_info)
        
        # model_info도 업데이트
        ensemble.model_info = model_info
        
        logger.info(f"Ensemble model setting updated: {model_key}")

    def remove_model(self, model_key: str) -> None:
        """Remove ensemble model from the agent."""
        if model_key in self.models:
            del self.models[model_key]
            logger.info(f"Ensemble model removed: {model_key}")

    def get_model(self, model_key: str) -> Optional[ENSEMBLE]:
        """Get ensemble model by key."""
        return self.models.get(model_key)

    def predict(self, model_key: str):
        """Predict using ensemble model."""
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"Ensemble model not found: {model_key}")
        return model.predict()

    def get_model_keys(self) -> list:
        """Get all registered model keys."""
        return list(self.models.keys())

    def get_model_count(self) -> int:
        """Get number of registered models."""
        return len(self.models)
    
    def check_model_running(self, model_key: str) -> bool:
        """Check if ensemble model is running."""
        return model_key in self.models

    @logging_time
    def calc_preds(self) -> int:
        """Real-time calculation loop for all ensemble models."""
        self.cnt_calc = 0
        
        for model_key, model in list(self.models.items()):
            is_data_updated = False
            pred_model: ENSEMBLE = model
            
            try:
                # ensemble_key로 queue에서 데이터 pop
                model_tag_values = model_tag_value_queue._pop(model_key)
                
                # ensemble와 sub_models에 데이터 업데이트
                # ENSEMBLE.update_data_raw_pb2()가 sub_models에도 자동으로 업데이트하므로 한 번만 호출
                is_data_updated = pred_model.update_data_raw_pb2(model_tag_values)
                
                # NaN 체크 후 push_data() (단일 모델과 동일 패턴)
                if np.isnan(pred_model.model_tag_data.array_data_raw.values).any():
                    pred_model.model_tag_data.push_data()
                    # sub_models도 push_data (버퍼 초기화)
                    for submodel in pred_model.sub_models:
                        if np.isnan(submodel.model_tag_data.array_data_raw.values).any():
                            submodel.model_tag_data.push_data()
                    
            except (QueueEmpty, KeyError):
                continue
            except Exception as e:
                logger.error(f"Error updating data for ensemble model {model_key}: {e}")
                continue

            if not is_data_updated:
                continue

            # 데이터 처리
            data_raw = pred_model.model_tag_data.array_data_raw
            pred_model.update_max_timestamp(data_raw.timestamps.max())

            try:
                # 예측 실행
                pred_values = pred_model.predict()
                pred_model.update_last_calc_time()
                
                # NaN 체크 (버퍼가 아직 충분하지 않은 경우)
                if np.isnan(pred_values).any():
                    logger.debug(
                        f"Ensemble prediction contains NaN for {model_key} "
                        f"(sub-models may be waiting for buffer to fill)"
                    )
                    continue
                
                # 예측 결과 저장
                pred_model.update_data_pred(
                    max_timestamp=pred_model.max_timestamp,
                    pred_values=pred_values,
                    status_codes=data_raw.statuscodes[:, -1],
                )

                self._calculate_and_update_index(pred_model)

                pred_model.set_calculated()
                self.cnt_calc += 1
                
                # 예측 결과 로그로 저장
                # self._log_prediction_results(model_key, pred_values, pred_model)
                
                logger.trace(f"Ensemble prediction completed for {model_key}: {pred_values}")
                
            except Exception as e:
                logger.exception(f"Error in prediction for {model_key}: {e}")
            finally:
                # 다음 데이터를 위한 버퍼 준비
                pred_model.model_tag_data.push_data()
                for submodel in pred_model.sub_models:
                    submodel.model_tag_data.push_data()
                
        return self.cnt_calc

    def _calculate_and_update_index(self, pred_model: ENSEMBLE) -> None:
        """Index 계산 및 업데이트 (단일 모델과 동일한 방식)"""
        try:
            # Tag index 계산 (단일 모델과 동일한 방식)
            for tagname, index_calculator in pred_model.index_calculators_tag.items():
                try:
                    # 단일 모델과 동일하게 단순히 calc_index()만 호출
                    index_calculator.calc_index()
                    
                    # 계산된 index 값 업데이트
                    if hasattr(index_calculator.data, 'index') and hasattr(index_calculator.data.index, 'values'):
                        if len(index_calculator.data.index.values) > 0:
                            pred_model.update_data_index(
                                tagname=tagname,
                                timestamp=index_calculator.data.index.timestamps[-1],
                                index_value=index_calculator.data.index.values[-1],
                                status_code=index_calculator.data.index.statuscodes[-1],
                            )
                            logger.trace(f"Tag index calculated for {tagname}: {index_calculator.data.index.values[-1]}")
                    
                except Exception as tag_err:
                    logger.error(f"Error calculating tag index for {tagname}: {tag_err}")

            # Model index 계산 (단일 모델과 동일한 방식)
            if pred_model.index_calculator_model is not None:
                try:
                    # ENSEMBLE은 model_setting이 아닌 model_info를 사용
                    model_setting: ModelSetting = pred_model.model_info
                    
                    if model_setting is None:
                        logger.warning(f"Ensemble model {pred_model.model_key} has no model_info, skipping model index calculation")
                        return
                    
                    active_tagnames = [
                        tagname
                        for tagname in pred_model.model_tag_data.tagnames
                        if model_setting.tagsettinglist[tagname].indexalarm
                    ]
                    
                    if not active_tagnames:
                        active_tagnames = list(pred_model.model_tag_data.tagnames)
                    
                    index_ary = np.array([
                        pred_model.model_tag_data.data[tagname].index.values[-1]
                        for tagname in active_tagnames
                    ])
                    status_ary = np.array([
                        pred_model.model_tag_data.data[tagname].index.statuscodes[-1]
                        for tagname in active_tagnames
                    ])
                    
                    pred_model.index_calculator_model.calc_index(
                        index_ary=index_ary,
                        status_ary=status_ary,
                    )
                    
                    # 인덱스 계산 결과 확인 (data.index에 저장됨)
                    model_index = pred_model.index_calculator_model.data.index
                    logger.trace(f"Model index calculated: {model_index}")
                    
                    # 인덱스 값이 None이 아닌 경우에만 로그 출력
                    if model_index is not None and not np.isnan(model_index):
                        logger.trace(f"Ensemble model index successfully calculated: {model_index}")
                    else:
                        logger.warning("Ensemble model index calculation failed or returned NaN")
                    
                except Exception as model_err:
                    logger.error(f"Error calculating model index: {model_err}")
                
            logger.trace("Index calculation completed for ensemble model")
            
        except Exception as e:
            logger.error(f"Error in index calculation: {e}")

    def _log_prediction_results(self, model_key: str, pred_values: np.ndarray, pred_model) -> None:
        try:
            import json
            import os
            from datetime import datetime
            
            # 로그 데이터 구성
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "model_key": str(model_key),
                "prediction_timestamp": int(pred_model.max_timestamp),
                "prediction_values": [float(x) for x in pred_values.tolist()],
                "tagnames": [str(t) for t in pred_model.tagnames],
                "calc_method": str(getattr(pred_model, 'calc_method', '')),
                "sub_models_count": int(len(getattr(pred_model, 'sub_models', []))),
                "sub_model_keys": [str(k) for k in getattr(pred_model, 'sub_model_keys', [])]
            }
            
            # Index 계산 결과 추가
            try:
                actual_values = pred_model.model_tag_data.array_data_raw.values[-1]
                mse = float(np.mean((actual_values - pred_values) ** 2))
                log_data.update({
                    "actual_values": [float(x) for x in actual_values.tolist()],
                    "mse": mse,
                    "prediction_accuracy": float(1 - mse) if mse < 1 else 0.0
                })
                # Tag별 Index 계산
                tag_indices = {}
                for i, tagname in enumerate(pred_model.tagnames):
                    diff = abs(float(actual_values[i]) - float(pred_values[i]))
                    tag_indices[tagname] = {
                        "actual": float(actual_values[i]),
                        "predicted": float(pred_values[i]),
                        "difference": float(diff)
                    }
                log_data["tag_indices"] = tag_indices
            except Exception as index_err:
                log_data["index_calculation_error"] = str(index_err)
            
            # 로그 파일에 저장
            log_file_path = f"logs/ensemble_predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
            
            logger.info(f"Prediction results logged to {log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction results: {e}")

    def create_calc_result_updated_only(self) -> dict:
        """Ensemble 모델 결과를 protobuf 형식으로 생성"""
        try:
            from _protobuf.model_data_pb2 import ToIPCM
            
            response = ToIPCM()
            
            for model_key, pred_model in self.models.items():
                try:
                    if pred_model.calculated:
                        model_data = response.model_data.add()
                        model_data.model_key = model_key
                        model_data.timestamp = pred_model.max_timestamp
                        
                        # 실제 계산된 모델 전체 index 사용
                        model_data.index = float(pred_model.index_calculator_model.data.index)
                        model_data.status_index = int(pred_model.index_calculator_model.data.status_index)
                        
                        for tagname in pred_model.targettagnames:  # targettagnames 사용 (단일 모델과 동일)
                            tag_data = model_data.data.add()
                            tag_data.tagname = tagname
                            tag_data.pred = float(pred_model.model_tag_data.data[tagname].pred.values[-1])
                            tag_data.status_pred = int(pred_model.model_tag_data.data[tagname].pred.statuscodes[-1])
                            tag_data.index = float(pred_model.model_tag_data.data[tagname].index.values[-1])
                            tag_data.status_index = int(pred_model.model_tag_data.data[tagname].index.statuscodes[-1])
                        
                        # calculated 플래그 리셋 (단일 모델과 동일)
                        pred_model.calculated = False
                except Exception as e:
                    logger.warning(f"Error creating ensemble result for {model_key}: {e}")
            return response
        except Exception as e:
            logger.error(f"Error creating ensemble calc result: {e}")
            return None
