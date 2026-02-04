import time
from kafka import KafkaProducer
from _protobuf.model_data_pb2 import ToIPCM


# Kafka 설정
KAFKA_BROKERS = [
    "192.168.170.10:49091",
    "192.168.170.10:49092",
    "192.168.170.10:49093"
]
KAFKA_TOPIC = "pred-values"


# 모델 key 정보
# model_key = "aakr_mjs_aakr_1"
# model_key = "aakr_test9999_1"
# model_key = "annoy_test9999_2"
# model_key = "annoy_test9999"
model_key = "annoy_test8888"


# 보내고 싶은 횟수 설정 (기본 1)
SEND_COUNT = 10  # ← 여기만 바꾸면 됨

# 태그 및 값 설정
tag_names = [
    "2235-CDP0357",
    "2235-CDP0356",
    "2235-CDP0355",
    "2235-CDP0139",
    "2235-CDF0135"
]
base_pred_values = [101.1, 101.2, 99.8, 102.5, 98.7]
status_codes = [192, 192, 192, 192, 192]

# Kafka 프로듀서 생성
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    acks=0,
    api_version=(2, 5, 0),
    retries=3,
)

# 반복 송신
for i in range(SEND_COUNT):
    msg = ToIPCM()
    model_data = msg.model_data.add()
    timestamp = int(time.time() * 1000)

    model_data.model_key = model_key
    model_data.timestamp = timestamp
    model_data.index = 0.6
    model_data.status_index = 192

    for tag, pred, status in zip(tag_names, base_pred_values, status_codes):
        tag_data = model_data.data.add()
        tag_data.tagname = tag
        tag_data.pred = pred
        tag_data.index = 0.6
        tag_data.status_pred = status
        tag_data.status_index = 192

    producer.send(KAFKA_TOPIC, msg.SerializeToString())
    print(f"[{i+1}/{SEND_COUNT}] Sent prediction for model {model_key} at {timestamp}")
    time.sleep(0.5) if SEND_COUNT > 1 else None  # 1회일 경우 sleep 생략

producer.flush()
print("All messages sent.")
