import orjson
from kafka import KafkaProducer


def json_value_serializer(value):
    return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)


class MessageProducer:
    def __init__(self, broker, topic):
        self.broker = broker
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=self.broker,
            acks=0,
            api_version=(2, 5, 0),
            retries=3,
        )

    def send_message(self, msg):
        try:
            future = self.producer.send(self.topic, msg)
            self.producer.flush()  # 비우는 작업
            future.get(timeout=2)
            return {"status_code": 200, "error": None}
        except Exception as exc:
            raise exc
