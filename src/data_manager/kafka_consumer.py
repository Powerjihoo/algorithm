import time, datetime
from dataclasses import dataclass
from google.protobuf.text_format import MessageToString

from _protobuf.model_data_pb2 import FromIPCM
from config import settings
from dbinfo.tag_value import ModelTagValueQueue
from google import protobuf
from kafka import KafkaConsumer
from model.manager.common import ModelAgent
from utils.logger import logger, logging_time
import re
import os
from resources.constant import CONST

model_tag_value_queue = ModelTagValueQueue()

LOG_FILE_PATH = "./logs/kafka_message.log" 

LOG_DIR = "./logs"
TARGET_MODEL_KEYS = ["aakr_wstest1", "aakr_wstest10", "aakr_wstest100"]

def save_message_to_log(message_pb) -> None:
    """특정 모델키만 각각 별도 로그 파일에 저장"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model_value in message_pb.model_data:
        model_key = model_value.model_key

        # 선택된 모델키가 아니면 스킵
        if model_key not in TARGET_MODEL_KEYS:
            continue

        # 로그 파일 경로 (모델키별 파일)
        log_path = os.path.join(LOG_DIR, f"{model_key}.log")

        # protobuf 내용 문자열로 변환
        msg_txt = MessageToString(model_value)

        # 로그 작성
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== {now_str} =====\n")
            f.write(f"model_key: {model_key}\n")
            f.write(msg_txt)
            f.write("\n")

def save_all_message_to_log(message_pb) -> None:
    """현재 시간 헤더 + 모델 키 개수 + protobuf 내용을 텍스트로 파일에 append"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_count = len(message_pb.model_data)  # 🔹 모델 키 개수

    # protobuf 내용을 문자열로 변환 (가독성 좋게)
    msg_txt = MessageToString(message_pb)

    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n===== {now_str} =====\n")
        f.write(f"총 model_key 개수: {total_count}\n")   # 🔹 모델 키 개수 추가
        f.write(msg_txt)
        f.write("\n")

@dataclass
class TagRawValue:
    timestamp: str
    unixTimestamp: int
    value: float
    statusCodeEnum: int

    @classmethod
    def from_dict(cls, data: dict) -> "TagRawValue":
        return cls(**data)


@dataclass
class ModelTagValue:
    tagName: str
    rawValue: TagRawValue

    @classmethod
    def from_dict(cls, data: dict) -> "ModelTagValue":
        return cls(
            tagName=data["tagName"], rawValue=TagRawValue.from_dict(data["rawValue"])
        )


@dataclass
class IPCMModelData:
    modelType: str
    modelName: str
    algorithmKey: str
    modelTagValueList: list[ModelTagValue]

    @classmethod
    def from_dict(cls, data: dict) -> "IPCMModelData":
        return cls(
            modelType=data["modelType"],
            modelName=data["modelName"],
            algorithmKey=data["algorithmKey"],
            modelTagValueList=[
                ModelTagValue.from_dict(tag_data)
                for tag_data in data["modelTagValueList"]
            ],
        )


class StreamDataCollector:
    def __init__(self, broker, topic, model_agent: ModelAgent, ensemble_agent=None):
        self.broker = broker
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.broker,
            client_id=f"{CONST.PROGRAM_NAME}_{settings.server_id}",
            group_id=f"{CONST.PROGRAM_NAME}_{settings.server_id}",
            #group_id=None,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            api_version=(0, 11, 5),
            consumer_timeout_ms=1000,
            max_poll_records=settings.kafka.max_poll_records, # 원본 코드 2
            session_timeout_ms=300000,
            heartbeat_interval_ms=60000,
            max_poll_interval_ms=600000,
            fetch_min_bytes=0,
            fetch_max_wait_ms=250,
        )
        while not self.consumer.assignment():
            logger.info("Waiting for partition assignment...")
            self.consumer.poll(timeout_ms=100)
            time.sleep(0.1)
        logger.debug("Subscription:", self.consumer.subscription())
        logger.info(f"Partition assigned: {self.consumer.assignment()}")
        self.model_agent: ModelAgent = model_agent
        self.ensemble_agent = ensemble_agent
        

    def close(self) -> None:
        self.consumer.close()

    # @logging_time
    # def receive_message(self) -> None:
    #     logger.trace("polling start")
    #     res = self.consumer.poll(timeout_ms=500)
    #     logger.trace("polling end")
    #     if not res:
    #         print("no data from kafka")
    #         time.sleep(0.5)
    #         return
    #     logger.trace("Get messaged")
    #     message_pb = FromIPCM()
    #     for messages in res.values():
    #         try:
    #             for message in messages:
    #                 message_pb.ParseFromString(message.value)
    #                 logger.trace(
    #                     f"  => Model data in message from IPCM: {len(message_pb.model_data)}"
    #                 )

    #                 for model_value in message_pb.model_data:
    #                     try:
    #                         # [Test]
    #                         model_tag_value_queue.update_data(
    #                             model_value.model_key, model_value.data
    #                         )
    #                     except Exception as e:
    #                         logger.error(e)
    #         except protobuf.message.DecodeError:
    #             continue
    #         except Exception as e:
    #             logger.exception(e)

    @logging_time
    def receive_message(self) -> None:
        logger.info("Start consumer polling loop")
        message_pb = FromIPCM()
        try:
            while True:
                logger.trace("polling start")
                res = self.consumer.poll(timeout_ms=500)
                logger.trace("polling end")

                if not res:
                    print("no data from kafka")
                    time.sleep(0.5)
                    continue

                logger.trace("Get message")
                for messages in res.values():
                    for message in messages:
                        try:
                            message_pb.ParseFromString(message.value)

                            total_count = len(message_pb.model_data)
                            logger.trace(f"총 model_key 개수: {total_count}")

                            # save_message_to_log(message_pb)

                            for model_value in message_pb.model_data:
                                # logger.debug(f"model_key={model_value.model_key}")
                                
                                base_key = model_value.model_key
                                
                                # Ensemble 모델 체크 (model_key 또는 model_type 기반)
                                # ensemble 모델은 버전이 없으므로 base_key 그대로 사용
                                is_ensemble = (
                                    base_key.lower().startswith("ensemble_") or 
                                    (model_value.model_type.upper() == "ENSEMBLE" if model_value.model_type else False)
                                )
                                
                                if is_ensemble:
                                    # Ensemble 모델: base_key 그대로 사용
                                    model_tag_value_queue.update_data(
                                        base_key, model_value.data
                                    )
                                else:
                                    # 단일 모델: 버전 추가하는 로직
                                    versioned_key = self.model_agent.get_versioned_for_base(base_key)
                                    if versioned_key:  # alias에 등록된 모델만 처리
                                        model_tag_value_queue.update_data(
                                            versioned_key, model_value.data
                                        )

                        except protobuf.message.DecodeError:
                            continue
                        except Exception as e:
                            logger.error(e)
        except Exception as e:
            logger.exception("Polling loop crashed: ", e)