from typing import List, Optional

import numpy as np
import os
import orjson
from loguru import logger
from api_server.models.models import ModelSetting

from .base_model import PredictionModelBase, ModelArrayData
from .annoy import ANNOY
from . import algo_calc
from .aakr import AAKR
from .vae import VAE
from .ica import ICA

ALGO_CLASS = {
    "ANNOY": ANNOY,
    "AAKR": AAKR,
    "VAE": VAE,
    "ICA": ICA,
}

class ENSEMBLE(PredictionModelBase):
    def __init__(
        self,
        modelname: str,
        tagnames: List[str],
        targettagnames: Optional[List[str]] = None,
        calc_method: str = "average",
        custom_base_path: str | None = None,
        model_info: Optional[ModelSetting] = None,
    ) -> None:
        # custom_base_path를 먼저 저장 (super().__init__() 전에)
        self._custom_base_path_for_loading = custom_base_path
        
        super().__init__(
            modeltype=self.modeltype,
            modelname=modelname,
            tagnames=tagnames,
            targettagnames=targettagnames or tagnames,
            calc_interval=1,
            custom_base_path=custom_base_path,
        )
        
        # Ensemble 모델은 버전이 없으므로, custom_base_path가 있으면 model_key를 덮어쓰기
        # load_model_setting() 전에 실행되어야 함
        if custom_base_path:
            self.model_key = custom_base_path
            logger.debug(f"Ensemble model_key set to custom_base_path: {self.model_key}")
        else:
            logger.warning(f"Ensemble model custom_base_path is None! model_key will be: {self.model_key}")
        
        self.calc_method = calc_method
        self.sub_model_infos = []
        self.sub_models: list[PredictionModelBase] = []
        self._sub_model_keys: list[str] = []
        self.model_info = model_info

        self.load_model_setting()
        self.setup_model_setting()
        self._create_sub_models()

        if self.sub_models:
            # 모든 sub_model의 window_size가 동일하므로 첫 번째 것을 사용
            self.window_size = self.sub_models[0].window_size
        else:
            self.window_size = 1

        self.create_data_array()

    def __repr__(self) -> str:
        return f"type={self.modeltype} name={self.modelname}"

    @property
    def modeltype(self) -> str:
        return "ENSEMBLE"

    @property
    def model_file_ext(self):
        return None

    @property
    def sub_model_keys(self) -> List[str]:
        return list(self._sub_model_keys)

    def load_model_setting(self) -> None:
        """Ensemble info는 반드시 info.json 파일에서 읽는다.
        
        Redis에는 sub_models 정보가 없으므로, ensemble 폴더의 info.json 파일에서
        sub_models와 calc_method를 읽어야 한다.
        """
        try:
            # info.json 파일 경로 (ensemble 폴더 내)
            # custom_base_path가 있으면 그것을 사용 (ensemble 모델은 버전이 없으므로)
            # _custom_base_path_for_loading을 우선 사용 (__init__에서 전달받은 원본 값)
            base_path = getattr(self, '_custom_base_path_for_loading', None) or getattr(self, 'custom_base_path', None)
            
            if base_path:
                ensemble_folder = base_path
                info_filename = f"{ensemble_folder}_info.json"
                logger.debug(f"Using custom_base_path for info.json: {ensemble_folder}")
            else:
                # fallback: model_key 사용 (하지만 경고)
                ensemble_folder = self.model_key
                info_filename = f"{self.model_key}_info.json"
                logger.warning(f"custom_base_path not found! Using model_key for info.json: {ensemble_folder}")
                logger.warning(f"  - _custom_base_path_for_loading: {getattr(self, '_custom_base_path_for_loading', 'NOT SET')}")
                logger.warning(f"  - custom_base_path: {getattr(self, 'custom_base_path', 'NOT SET')}")
                logger.warning(f"  - model_key: {self.model_key}")
            
            info_path = os.path.join(
                PredictionModelBase.model_path,
                ensemble_folder,
                info_filename,
            )
            
            if not os.path.exists(info_path):
                raise FileNotFoundError(
                    f"Ensemble info file not found: {info_path}. "
                    f"Ensemble model must have info.json file in its folder."
                )
            
            with open(info_path, "r") as f:
                info_json = orjson.loads(f.read())

            # info.json에서 sub_models 추출 (필수)
            if "sub_models" not in info_json:
                raise ValueError(
                    f"Ensemble info.json must contain 'sub_models' list. "
                    f"File: {info_path}"
                )
            
            self.sub_model_infos = info_json["sub_models"]
            logger.info(
                f"Loaded {len(self.sub_model_infos)} sub_models from info.json: {info_path}"
            )
            
            # calc_method 우선순위: 1) __init__에서 전달받은 값 (Redis에서 온 값), 2) info.json, 3) 기본값
            # __init__에서 이미 Redis의 calcMethod 또는 modelParameter.calc_method가 설정되었으므로,
            # info.json의 calc_method는 __init__에서 설정된 값이 없을 때만 사용
            if self.calc_method == "average" and "calc_method" in info_json and info_json["calc_method"]:
                # __init__에서 기본값 "average"로 설정되었고, info.json에 calc_method가 있으면 사용
                self.calc_method = info_json["calc_method"]
                logger.debug(f"Using calc_method from info.json: {self.calc_method}")
            else:
                logger.debug(f"Using calc_method from initialization (Redis or default): {self.calc_method}")
                
        except Exception as e:
            logger.error(f"Error loading ensemble model setting from info.json: {e}")
            raise

    def setup_model_setting(self):
        pass

    def load_model(self):
        pass

    def _create_sub_models(self):
        logger.info(f"Creating sub_models from {len(self.sub_model_infos)} infos")

        for i, info in enumerate(self.sub_model_infos):
            logger.info(f"Processing sub_model {i+1}: {info}")
            algo = str(info.get("algorithm", "")).upper()
            model_class = ALGO_CLASS.get(algo)

            if not model_class:
                logger.error(f"Unknown algorithm: {algo}, cannot create submodel.")
                raise ValueError(f"Unknown algorithm: {algo}")

            sub_name = info.get("modelname")

            if not sub_name:
                logger.error(f"Missing modelname for submodel {i+1}")
                raise ValueError(f"Missing modelname for submodel {i+1}")

            # version은 문자열("0001") 또는 정수(1) 형태로 올 수 있음
            version_raw = info.get("version", "0")
            if isinstance(version_raw, str):
                # "0001" 형태의 문자열인 경우 정수로 변환
                version = int(version_raw) if version_raw.isdigit() else 0
            else:
                # 이미 정수인 경우
                version = int(version_raw)
            
            # model_key 생성 (버전 포함)
            model_key = f"{algo}_{sub_name}_{version:04d}"
            # modelname은 버전 없이 (version 파라미터로 전달)
            modelname_without_version = sub_name

            info_path = os.path.join(
                PredictionModelBase.model_path,
                self.model_key,
                model_key,
                f"{model_key}_info.json",
            )
            
            if not os.path.exists(info_path):
                logger.error(f"Submodel info file not found: {info_path}")
                raise FileNotFoundError(f"Submodel info file not found: {info_path}")

            with open(info_path, "r") as f:
                info_json = orjson.loads(f.read())

            tnames = info_json.get("tagnames", self.tagnames) or self.tagnames
            ttargets = info_json.get("targettagnames", tnames) or tnames

            if not isinstance(tnames, list):
                tnames = [tnames]
            if not isinstance(ttargets, list):
                ttargets = [ttargets]

            # sub_model의 custom_base_path는 ensemble 폴더 경로로 설정
            # base_path가 {model_path}/{custom_base_path}/{model_key}로 생성되므로
            # custom_base_path를 self.model_key로 설정하면 올바른 경로가 됨
            # 예: {model_path}/ensemble_ensemble_test/aakr_ensemble_test_temp_0001
            # 
            # 실제 폴더 구조 확인을 위한 디버그 로그
            expected_path = os.path.join(
                PredictionModelBase.model_path,
                self.model_key,
                model_key
            )
            
            logger.debug(f"Sub-model expected path: {expected_path}")
            logger.debug(f"Sub-model custom_base_path: {self.model_key}")
            logger.debug(f"Sub-model model_key: {model_key}")
            
            # sub_model의 custom_base_path는 ensemble 폴더 경로
            # base_path = {model_path}/{custom_base_path}/{model_key}
            # 따라서 custom_base_path = self.model_key (ensemble 폴더)
            # 예: {model_path}/ensemble_ensemble_test/aakr_ensemble_test_temp_0001
            sub_model_custom_base_path = self.model_key
            
            # 디버그: 실제 base_path가 어떻게 생성되는지 확인
            # sub_model 생성 전에 예상 경로 계산
            test_base_path = os.path.join(
                PredictionModelBase.model_path,
                sub_model_custom_base_path,
                model_key
            )
            logger.debug(f"Sub-model base_path will be: {test_base_path}")
            logger.debug(f"Sub-model save_path_scaler will be: {os.path.join(test_base_path, f'{model_key}_scaler.pkl')}")
            
            # 각 모델의 생성자 시그니처에 맞춰 호출
            # AAKR: (modelname, tagnames, calc_interval=1, custom_base_path=None, is_ensemble=False, version=0)
            # ANNOY: (modelname, tagnames, targettagnames, custom_base_path=None, is_ensemble=False, version=0)
            # VAE: (modelname, tagnames, targettagnames, calc_interval=5, custom_base_path=None, is_ensemble=False, version=0)
            # ICA: (modelname, tagnames, targettagnames, custom_base_path=None, is_ensemble=False, version=0)
            
            if algo == "ANNOY":
                sub_model = ANNOY(
                    modelname=modelname_without_version,
                    tagnames=tnames,
                    targettagnames=ttargets,
                    custom_base_path=sub_model_custom_base_path,
                    is_ensemble=True,
                    version=version,
                )
            elif algo == "AAKR":
                # AAKR은 targettagnames 파라미터가 없음 (내부에서 tagnames로 설정)
                sub_model = AAKR(
                    modelname=modelname_without_version,
                    tagnames=tnames,
                    calc_interval=1,  # AAKR 기본값
                    custom_base_path=sub_model_custom_base_path,
                    is_ensemble=True,
                    version=version,
                )
            elif algo == "ICA":
                sub_model = ICA(
                    modelname=modelname_without_version,
                    tagnames=tnames,
                    targettagnames=ttargets,
                    custom_base_path=sub_model_custom_base_path,
                    is_ensemble=True,
                    version=version,
                )
            elif algo == "VAE":
                sub_model = VAE(
                    modelname=modelname_without_version,
                    tagnames=tnames,
                    targettagnames=ttargets,
                    calc_interval=5,  # VAE 기본값
                    custom_base_path=sub_model_custom_base_path,
                    is_ensemble=True,
                    version=version,
                )
            else:
                logger.error(f"Unknown algorithm: {algo}, cannot create submodel.")
                raise ValueError(f"Unknown algorithm: {algo}")
            
            # 파일 기반 초기화
            if hasattr(sub_model, "load_model"):
                try:
                    sub_model.load_model()
                except Exception as e:
                    logger.error(f"sub_model.load_model() failed: {e}")
                    raise
            
            # scaler 로드 (AAKR은 scaler 파일이 없으므로 스킵)
            # AAKR은 calc_norm_factor()로 차분한 뒤 저장하므로 scaler가 필요 없음
            if algo != "AAKR" and hasattr(sub_model, "load_scaler"):
                try:
                    sub_model.load_scaler()
                except Exception as e:
                    logger.error(f"sub_model.load_scaler() failed: {e}")
                    raise
            elif algo == "AAKR":
                logger.debug("AAKR sub_model does not require scaler file, skipping load_scaler()")

            self.sub_models.append(sub_model)
            self._sub_model_keys.append(model_key)
            logger.trace(f"Successfully created sub_model: {model_key}")

        logger.info(f"Created {len(self.sub_models)} sub_models: {self._sub_model_keys}")

    def update_data_raw_pb2(self, model_tag_values) -> bool:
        """Ensemble와 sub_models에 데이터 업데이트.
        
        EnsembleAgent.calc_preds()에서 한 번만 호출되며,
        내부에서 ensemble 자체와 모든 sub_models를 함께 업데이트합니다.
        """
        updated = super().update_data_raw_pb2(model_tag_values)
        for m in self.sub_models:
            try:
                sub_updated = m.update_data_raw_pb2(model_tag_values)
                updated |= sub_updated
            except Exception as err:
                logger.error(f"Error updating sub_model data: {err}")
        return updated

    def predict(self) -> np.ndarray:
        """Ensemble prediction using sub-models.
        
        각 sub_model의 predict()를 호출하고, VAE/ANNOY 등은 내부에서 
        버퍼 상태를 체크하여 NaN을 반환하므로, 이를 자연스럽게 처리합니다.
        """
        if not self.sub_models:
            logger.warning("No sub_models available, returning NaN")
            return np.array([np.nan] * len(self.targettagnames))
        
        preds = []
        mses = []
        
        # 각 sub_model에 대해 예측 실행
        # VAE/ANNOY는 predict() 내부에서 버퍼 상태를 체크하고 NaN을 반환함
        # AAKR은 Manager와 동일한 방식으로 처리 (predict() 사용 안 함)
        for i, m in enumerate(self.sub_models):
            try:
                # AAKR sub-model인 경우 특별 처리
                if m.modeltype == "AAKR":
                    data_raw: ModelArrayData = m.model_tag_data.array_data_raw
                    
                    # AAKRManager와 동일한 방식으로 처리
                    m.update_targetcols(
                        data_raw.values[-1], data_raw.statuscodes[-1]
                    )
                    if not any(m.targetcols):
                        logger.debug(
                            f"Sub_model {i+1} ({m.model_key}) has no data or bad quality"
                        )
                        continue
                    
                    m.update_input_data_norm(data_raw.values[-1])
                    
                    # algo_calc.aakr() 직접 호출
                    values_pred = algo_calc.aakr(
                        input_data=m.input_data_norm[m.targetcols == True],  # noqa: E712
                        model_data=m.model_data_norm,
                        model_data_for_weight=m.model_data_norm[
                            :, m.targetcols == True  # noqa: E712
                        ],
                    )
                    # inverse_transform 적용
                    pred = m.inverse_transform(values_pred)
                else:
                    # 다른 모델들(VAE, ANNOY 등)은 predict() 사용
                    pred = m.predict()
                
                # NaN 체크 (버퍼가 가득 차지 않았거나 데이터가 없으면 NaN 반환됨)
                if np.isnan(pred).any():
                    logger.debug(
                        f"Sub_model {i+1} ({m.model_key}) returned NaN "
                        f"(buffer may not be full yet)"
                    )
                    continue
                
                preds.append(pred)
                
                # MSE 계산 (예측값과 실제값 비교)
                try:
                    actual_values = m.model_tag_data.array_data_raw.values
                    if len(actual_values.shape) == 3:
                        actual = actual_values[0, -1, :]
                    else:
                        actual = actual_values[-1, :]
                    mse = np.mean((actual - pred) ** 2)
                except Exception:
                    mse = np.inf
                mses.append(mse)
                
            except Exception as e:
                logger.error(f"Error in sub_model {i+1} ({m.model_key}) prediction: {e}")
                # 예외 발생 시 해당 sub_model만 제외하고 계속 진행
        
        # 예측 결과가 없으면 NaN 반환
        if not preds:
            logger.debug(
                "No valid predictions from sub_models "
                "(all returned NaN or errors, may be waiting for buffer to fill)"
            )
            return np.array([np.nan] * len(self.targettagnames))
        
        preds = np.array(preds)
        mses = np.array(mses)
        
        # Ensemble 방법으로 결합
        if self.calc_method == "best":
            if len(mses) == 0 or np.all(np.isinf(mses)):
                logger.debug("No valid MSEs for best method, using average")
                result = preds.mean(axis=0)
            else:
                idx = int(np.argmin(mses))
                result = preds[idx]
            return result
        
        if self.calc_method == "worst":
            if len(mses) == 0 or np.all(np.isinf(mses)):
                logger.debug("No valid MSEs for worst method, using average")
                result = preds.mean(axis=0)
            else:
                idx = int(np.argmax(mses))
                result = preds[idx]
            return result
        
        # average (기본값)
        result = preds.mean(axis=0)
        return result
