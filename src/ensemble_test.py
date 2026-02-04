import numpy as np
import orjson

from model.manager.common import ModelAgent
from model.manager.annoy import ANNOYManager
from model.manager.aakr import AAKRManager
from model.manager.ica import ICAManager
from model.manager.vae import VAEManager
from model.manager.ensemble import EnsembleAgent
from dbinfo.model_info import ModelInfoManager

# Index 계산을 위한 import 추가
from model.index.tag._calculator import _IndexCalcThresholdSingle
from model.index.model._calculator import _ModelIndexCalcMSE

model_agent = ModelAgent()
model_agent.register(ANNOYManager())
model_agent.register(AAKRManager())
model_agent.register(ICAManager())
model_agent.register(VAEManager())

ensemble_key = "ensemble_ensemble_annoy"
window_size = 8
dummy_value = 135
dummy_status = 0
dummy_timestamp = 1234567890

agent = EnsembleAgent()

try:
    # Redis에서 원본 데이터 확인
    print(f"\n=== Redis 원본 데이터 확인 ===")
    raw_data = ModelInfoManager.redis_client.get(ensemble_key)
    if raw_data:
        raw_dict = orjson.loads(raw_data)
        print(f"Raw modelparameter type: {type(raw_dict.get('modelparameter'))}")
        print(f"Raw modelparameter: {raw_dict.get('modelparameter')}")
        
        # modelparameter가 문자열인지 확인
        if isinstance(raw_dict.get('modelparameter'), str):
            parsed_modelparam = orjson.loads(raw_dict.get('modelparameter'))
            print(f"Parsed modelparameter: {parsed_modelparam}")
            print(f"Parsed sub_models: {parsed_modelparam.get('sub_models', [])}")
    
    # 모델 정보 확인 - 상세 정보 출력
    model_info = ModelInfoManager.get(ensemble_key)

    print(f"\n=== Redis에서 가져온 모델 정보 ===")
    print(f"Model name: {model_info.modelname}")
    print(f"Model type: {model_info.modeltype}")
    print(f"Model key: {model_info.modelkey}")
    print("\n=== 모델 상태 정보 ===")
    print(f"Model name: {model_info.modelname}")
    print(f"Model type: {model_info.modeltype}")
    print(f"Model key: {model_info.modelkey}")
    print(f"Running status: {model_info.runningstatus}")
    print(f"Training status: {model_info.trainingstatus}")
    print(f"Description: {model_info.description}")
    print(f"System name: {model_info.systemname}")
    print(f"System index: {model_info.systemidx}")
    print(f"Parent index: {model_info.parentidx}")
    print(f"Index calc method: {model_info.indexcalcmethod}")
    print(f"Index priority: {model_info.indexpriority}")
    print(f"Index weight: {model_info.indexweight}")
    print(f"Index alarm: {model_info.indexalarm}")
    
    # modelparameter 상세 정보
    print(f"\n=== Model Parameters ===")
    print(f"Model params type: {type(model_info.modelparameter.model_params)}")
    print(f"Model params: {model_info.modelparameter.model_params}")
    
    # sub_models 정보 확인
    sub_models = model_info.modelparameter.sub_models or []
    print(f"\n=== Sub Models Info ===")
    print(f"Sub models count: {len(sub_models)}")
    print(f"Sub models: {sub_models}")
    
    # 각 sub model의 상세 정보
    for i, sub_model_info in enumerate(sub_models):
        print(f"\nSub model {i}:")
        print(f"  - Algorithm: {sub_model_info.get('algorithm', 'N/A')}")
        print(f"  - Model name: {sub_model_info.get('modelname', 'N/A')}")
        print(f"  - Version: {sub_model_info.get('version', 'N/A')}")
        
        # 예상 파일 경로 확인
        algo = str(sub_model_info.get("algorithm", "")).upper()
        sub_name = sub_model_info.get("modelname")
        version = int(sub_model_info.get("version", 0))
        model_key = f"{algo}_{sub_name}_{version:04d}"
        print(f"  - Expected model key: {model_key}")
    
    # Ensemble 모델 추가
    print(f"\n=== Ensemble 모델 추가 ===")
    agent.add_model(ensemble_key)
    model = agent.get_model(ensemble_key)

    if model is None:
        print(f"Ensemble model {ensemble_key} 등록 실패")
        exit()

    print(f"Ensemble 모델 등록 완료: {ensemble_key}")
    print(f"Sub models count: {len(model.sub_models)}")
    print(f"Sub model keys: {model.sub_model_keys}")

    # Sub model 상세 정보 출력
    for i, submodel in enumerate(model.sub_models):
        print(f"\nSub model {i}: {model.sub_model_keys[i]}")
        print(f"  - Model type: {submodel.modeltype}")
        print(f"  - Window size: {getattr(submodel, 'window_size', 'N/A')}")
        print(f"  - Tagnames: {submodel.tagnames}")
        print(f"  - Target tagnames: {submodel.targettagnames}")

    # 더미 데이터 주입
    print(f"\n=== 더미 데이터 주입 ===")
    for submodel in model.sub_models:
        tagdata = submodel.model_tag_data
        for i in range(window_size):
            timestamp = dummy_timestamp + i * 5
            for tagname in tagdata.tagnames:
                tag = tagdata.data[tagname].raw
                tag._push_data()
                tag.update_data(timestamp, dummy_value, dummy_status)
            submodel.max_timestamp = timestamp

    print("더미 데이터 주입 완료")
    
    # 각 sub model의 데이터 상태 확인
    print(f"\n--- Sub Model 데이터 상태 확인 ---")
    for i, submodel in enumerate(model.sub_models):
        print(f"\nSub model {i+1} ({model.sub_model_keys[i]}):")
        tagdata = submodel.model_tag_data
        print(f"  - 데이터 shape: {tagdata.array_data_raw.values.shape}")
        print(f"  - 최신 값: {tagdata.array_data_raw.values[-1]}")
        print(f"  - 최신 타임스탬프: {tagdata.array_data_raw.timestamps[-1]}")
        print(f"  - NaN 값 존재: {np.isnan(tagdata.array_data_raw.values).any()}")

    # 예측 실행
    print(f"\n=== 예측 실행 ===")
    try:
        # 각 sub model의 개별 예측 결과 확인
        print(f"\n--- 각 Sub Model의 개별 예측 결과 ---")
        for i, submodel in enumerate(model.sub_models):
            try:
                sub_pred = submodel.predict()
                print(f"Sub model {i+1} ({model.sub_model_keys[i]}): {sub_pred}")
                
                # MSE 계산 시도
                try:
                    actual_values = submodel.model_tag_data.array_data_raw.values[-1]
                    mse = np.mean((actual_values - sub_pred) ** 2)
                    print(f"  - MSE: {mse:.6f}")
                except Exception as mse_err:
                    print(f"  - MSE 계산 실패: {mse_err}")
                    
            except Exception as sub_err:
                print(f"Sub model {i+1} 예측 실패: {sub_err}")
        
        # Ensemble 전체 예측
        print(f"\n--- Ensemble 전체 예측 ---")
        result = model.predict()
        print(f"Ensemble 예측 결과: {result}")
        print(f"예측 결과 타입: {type(result)}")
        print(f"예측 결과 shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        # 계산 방법 확인
        print(f"Ensemble 계산 방법: {model.calc_method}")
        
        # Index 계산 추가
        print(f"\n--- Index 계산 ---")
        try:
            # 실제 값 (더미 데이터의 마지막 값)
            actual_values = np.array([dummy_value] * len(model.tagnames))
            print(f"실제 값: {actual_values}")
            print(f"예측 값: {result}")
            
            # MSE 계산
            mse = np.mean((actual_values - result) ** 2)
            print(f"MSE: {mse:.6f}")
            
            # Tag Index 계산 (각 태그별)
            print(f"\n--- Tag Index 계산 ---")
            tag_index_calc = _IndexCalcThresholdSingle()
            threshold = 5  # 임계값 설정 (실제로는 설정에서 가져와야 함)
            
            for i, tagname in enumerate(model.tagnames):
                try:
                    # 예측값과 실제값의 차이
                    diff = abs(actual_values[i] - result[i])
                    tag_index = tag_index_calc.calc_index(diff, threshold)
                    print(f"Tag {tagname}: 실제={actual_values[i]:.3f}, 예측={result[i]:.3f}, 차이={diff:.6f}, Index={tag_index:.6f}")
                except Exception as tag_err:
                    print(f"Tag {tagname} index 계산 실패: {tag_err}")
            
            # Model Index 계산 (전체 모델)
            print(f"\n--- Model Index 계산 ---")
            model_index_calc = _ModelIndexCalcMSE()
            mse_ref = 0.05  # 참조 MSE (실제로는 설정에서 가져와야 함)
            
            try:
                model_index = model_index_calc.calc_index(mse, mse_ref)
                print(f"Model Index: {model_index:.6f}")
                print(f"  - MSE: {mse:.6f}")
                print(f"  - 참조 MSE: {mse_ref:.6f}")
                print(f"  - Index 공식: 1 - (MSE * (1 - 0.7) / 참조MSE)")
            except Exception as model_err:
                print(f"Model index 계산 실패: {model_err}")
                
        except Exception as index_err:
            print(f"Index 계산 중 오류 발생: {index_err}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # Agent의 predict 메서드 테스트
    print(f"\n=== Agent predict 메서드 테스트 ===")
    try:
        agent_result = agent.predict(ensemble_key)
        print(f"Agent 예측 결과: {agent_result}")
        print(f"Agent 예측 결과 타입: {type(agent_result)}")
        print(f"Agent 예측 결과 shape: {agent_result.shape if hasattr(agent_result, 'shape') else 'N/A'}")
        
        # Agent를 통한 모델 정보 확인
        agent_model = agent.get_model(ensemble_key)
        if agent_model:
            print(f"Agent 모델 계산 방법: {agent_model.calc_method}")
            print(f"Agent 모델 sub model 수: {len(agent_model.sub_models)}")
            
            # Agent 결과에 대한 Index 계산
            print(f"\n--- Agent 결과 Index 계산 ---")
            try:
                actual_values = np.array([dummy_value] * len(agent_model.tagnames))
                mse = np.mean((actual_values - agent_result) ** 2)
                
                # Tag Index 계산
                tag_index_calc = _IndexCalcThresholdSingle()
                threshold = 5
                
                for i, tagname in enumerate(agent_model.tagnames):
                    try:
                        diff = abs(actual_values[i] - agent_result[i])
                        tag_index = tag_index_calc.calc_index(diff, threshold)
                        print(f"Agent Tag {tagname}: 실제={actual_values[i]:.3f}, 예측={agent_result[i]:.3f}, Index={tag_index:.6f}")
                    except Exception as tag_err:
                        print(f"Agent Tag {tagname} index 계산 실패: {tag_err}")
                
                # Model Index 계산
                model_index_calc = _ModelIndexCalcMSE()
                mse_ref = 0.05
                model_index = model_index_calc.calc_index(mse, mse_ref)
                print(f"Agent Model Index: {model_index:.6f} (MSE: {mse:.6f})")
                
            except Exception as agent_index_err:
                print(f"Agent index 계산 실패: {agent_index_err}")
            
    except Exception as e:
        print(f"Agent 예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # 모델 정보 출력
    print(f"\n=== 최종 모델 정보 ===")
    print(f"등록된 모델 수: {agent.get_model_count()}")
    print(f"모델 키들: {agent.get_model_keys()}")

except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()

# ensemble_test.py에서 실행
def check_current_server_status():
    """현재 서버에서 ensemble 모델 상태 확인"""
    print("\n=== 현재 서버 상태 확인 ===")
    
    try:
        # 활성화된 모델 목록 확인
        activated_keys = ModelInfoManager.get_target_activated_model_keys()
        print(f"활성화된 모델 키들: {activated_keys}")
        
        # ensemble 모델 확인
        ensemble_found = False
        for key in activated_keys:
            try:
                model_info = ModelInfoManager.get(key)
                if model_info.modeltype.upper() == "ENSEMBLE":
                    ensemble_found = True
                    print(f"Ensemble 모델 발견: {key}")
                    print(f"   - Running status: {model_info.runningstatus}")
                    print(f"   - Training status: {model_info.trainingstatus}")
            except Exception as e:
                print(f"모델 {key} 확인 중 오류: {e}")
        
        if not ensemble_found:
            print("활성화된 ensemble 모델이 없습니다.")
            print("   모델 상태를 RUNNING으로 변경해야 합니다.")
            
            # Redis에 있는 ensemble 모델 확인
            try:
                model_info = ModelInfoManager.get("ensemble_ensemble_annoy")
                print(f"Redis에 있는 ensemble 모델: ensemble_ensemble_annoy")
                print(f"   - Running status: {model_info.runningstatus}")
                print(f"   - Training status: {model_info.trainingstatus}")
            except Exception as e:
                print(f"Redis에서 ensemble 모델 확인 중 오류: {e}")
        else:
            print("nsemble 모델이 활성화되어 있습니다.")
            
    except Exception as e:
        print(f"상태 확인 중 오류: {e}")

# 실행
check_current_server_status()