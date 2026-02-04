import time
from kafka import KafkaProducer
from _protobuf.model_data_pb2 import FromIPCM, ToIPCM
from model.manager.common import ModelAgent
from dbinfo.tag_value import ModelTagValueQueue
from loguru import logger

from model.manager.annoy import ANNOYManager
from model.manager.aakr import AAKRManager
from model.manager.ica import ICAManager
from model.manager.vae import VAEManager

# ===== Settings =====
KAFKA_BROKERS = [
    "192.168.170.10:49091",
    "192.168.170.10:49092",
    "192.168.170.10:49093",
]
KAFKA_TOPIC = "pred-values"
SEND_COUNT = 12

# ===== Test Model =====
model_key = "ica_jihoo_version_0001"
tag_names = ["2235-FWT0015", "2235-FWT0016", "2235-FWT0017"]
base_input_values = [138.1, 137.9, 136.8]
status_code = 192

# ===== Manager registration =====
agent = ModelAgent()
agent.register(ANNOYManager())
agent.register(AAKRManager())
agent.register(ICAManager())
agent.register(VAEManager())

if agent.get_model(model_key) is None:
    agent.add_model(model_key)

model = agent.get_model(model_key).pred
model_type = model.modeltype
window_size = getattr(model, "window_size", 1)
calc_interval = getattr(model, "calc_interval", 5)
logger.info(
    f"\u2705 Model loaded: key={model_key}, type={model_type}, window_size={window_size}, calc_interval={calc_interval}s"
)

# ===== Kafka Producer =====
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    acks=0,
    api_version=(2, 5, 0),
    retries=3,
)

# ===== Test Loop =====
base_timestamp = int(time.time() * 1000)

for i in range(SEND_COUNT):
    timestamp = base_timestamp + i * 1000

    model_data = FromIPCM.ModelData()
    model_data.model_key = model_key

    for tag, base_val in zip(tag_names, base_input_values):
        tag_pb = FromIPCM.ModelData.Data()
        tag_pb.tagname = tag
        tag_pb.timestamp = timestamp
        tag_pb.value = base_val + i * 0.1
        tag_pb.status_code = status_code
        tag_pb.is_ignored = False
        model_data.data.append(tag_pb)

    ModelTagValueQueue().update_data(model_key, model_data.data)

    logger.debug(f"[{i+1}/{SEND_COUNT}] Input pushed: ts={timestamp}, val0={model_data.data[0].value}")

    predicted = False
    for _, manager in agent.get_observers():
        manager.calc_preds()
        if manager.cnt_calc == 0:
            continue

        pred_result: ToIPCM = manager.create_calc_result_updated_only()
        for md in pred_result.model_data:
            if md.data:
                predicted = True
                logger.success(
                    f"\u2705 Predicted index: {md.index}, pred_val: {[d.pred for d in md.data]}"
                )
                producer.send(KAFKA_TOPIC, pred_result.SerializeToString())

    if not predicted:
        logger.info(f"\u274e No prediction at step {i+1} (ts={timestamp})")

    time.sleep(1.0)

producer.flush()
logger.info("\u2705 All predictions sent.")