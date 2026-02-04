import numpy as np
import time
from model.manager.ensemble import EnsembleAgent, model_tag_value_queue
from dbinfo.model_info import ModelInfoManager
from loguru import logger
from _protobuf.model_data_pb2 import FromIPCM

# 테스트할 ensemble 모델 키
ensemble_key = "ensemble_ensemble_annoy"

# EnsembleAgent 인스턴스
agent = EnsembleAgent()

# 모델이 등록되어 있지 않으면 등록
if ensemble_key not in agent.get_model_keys():
    agent.add_model(ensemble_key)

model = agent.get_model(ensemble_key)
if model is None:
    print(f"Ensemble model {ensemble_key} 등록 실패")
    exit()

print(f"Ensemble 모델 등록 완료: {ensemble_key}")
print(f"Sub models count: {len(model.s)}")
print(f"Sub model keys: {model.sub_model_keys}")

# Index calculator 설정 상태 확인
print(f"\n=== Index Calculator 설정 상태 ===")
print(f"index_calculator_model 존재: {hasattr(model, 'index_calculator_model')}")
if hasattr(model, 'index_calculator_model'):
    print(f"index_calculator_model 값: {model.index_calculator_model}")

print(f"index_calculators_tag 존재: {hasattr(model, 'index_calculators_tag')}")
if hasattr(model, 'index_calculators_tag'):
    print(f"index_calculators_tag 개수: {len(model.index_calculators_tag)}")
    print(f"index_calculators_tag 키들: {list(model.index_calculators_tag.keys())}")

# 모델 정보 확인
try:
    model_info = ModelInfoManager.get(ensemble_key)
    print(f"\n=== 모델 정보 ===")
    print(f"Model type: {model_info.modeltype}")
    print(f"Tag settings 개수: {len(model_info.tagsettinglist) if hasattr(model_info, 'tagsettinglist') else 'N/A'}")
    if hasattr(model_info, 'tagsettinglist'):
        for tagname, tag_setting in model_info.tagsettinglist.items():
            print(f"  - {tagname}: {tag_setting}")
except Exception as e:
    print(f"모델 정보 확인 중 오류: {e}")

# 테스트용 태그명 추출 (첫 submodel 기준)
tag_names = model.sub_models[0].tagnames if model.sub_models else []
window_size = getattr(model, 'window_size', 8)

# 더미 데이터 반복 입력 및 예측
for i in range(10):
    dummy_value = 130 + i * 2  # 값이 130 언저리가 되도록 변경
    dummy_status = 192  # status_code를 192 변경 (good status)
    dummy_timestamp = int(time.time()) + i * 5
    dummy_data = []
    for tag in tag_names:
        tag_pb = FromIPCM.ModelData.Data()
        tag_pb.tagname = tag
        tag_pb.timestamp = dummy_timestamp
        tag_pb.value = dummy_value
        tag_pb.status_code = dummy_status
        tag_pb.is_ignored = False
        dummy_data.append(tag_pb)
    # ensemble_key로만 queue에 데이터 입력 (protobuf 객체)
    model_tag_value_queue.update_data(ensemble_key, dummy_data)
    logger.debug(f"[TEST] {i}회차 더미 데이터 입력: {[{k: getattr(d, k) for k in ['tagname','timestamp','value','status_code','is_ignored']} for d in dummy_data]}")
    print(f"[{i}] 더미 데이터 입력: value={dummy_value}, timestamp={dummy_timestamp}")

    # 예측 시도
    calc_cnt = agent.calc_preds()
    print(f"[{i}] calc_preds() 결과: {calc_cnt}")
    if calc_cnt > 0:
        preds = [model.model_tag_data.data[tag].pred.values[-1] for tag in tag_names]
        print(f"[{i}] 예측값: {preds}")
        
        # 올바른 위치에서 index 값 가져오기
        model_index = None
        if hasattr(model, 'index_calculator_model') and model.index_calculator_model is not None:
            if hasattr(model.index_calculator_model, 'data') and hasattr(model.index_calculator_model.data, 'index'):
                if hasattr(model.index_calculator_model.data.index, 'values') and len(model.index_calculator_model.data.index.values) > 0:
                    model_index = model.index_calculator_model.data.index.values[-1]
        
        # Tag별 index 값도 가져오기
        tag_indices = {}
        for tag in tag_names:
            if tag in model.model_tag_data.data:
                tag_data = model.model_tag_data.data[tag]
                if hasattr(tag_data, 'index') and hasattr(tag_data.index, 'values') and len(tag_data.index.values) > 0:
                    tag_indices[tag] = tag_data.index.values[-1]
                else:
                    tag_indices[tag] = None
        
        max_timestamp = getattr(model, 'max_timestamp', None)
        print(f"[{i}] model index: {model_index}")
        print(f"[{i}] tag indices: {tag_indices}")
        print(f"[{i}] max_timestamp: {max_timestamp}")
    else:
        print(f"[{i}] 예측 미수행 (데이터 부족 또는 조건 미충족)")
    time.sleep(0.5) 