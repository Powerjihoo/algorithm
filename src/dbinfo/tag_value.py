import queue
import threading

from _protobuf.model_data_pb2 import FromIPCM
from api_client.apis.tagvalue import tagvalue_api
from dbinfo import exceptions as ex
from utils.logger import logger, logging_time
from utils.scheme.singleton import SingletonInstance


class ModelTagValueQueue(dict, metaclass=SingletonInstance):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        # self.initialize()

    # @logging_time
    # def initialize(self) -> None:
    #     try:
    #         # ! FIXME: 모델별 protobuf 데이터 받아서 initialize 필요
    #         message_pb = FromIPCM()
    #         message = self.load_db_info()
    #         message_pb.ParseFromString(message.value)

    #         for model_value in message_pb.model_data:
    #             self.update_data(
    #                 model_key=model_value.model_key, model_tag_values=model_value.data
    #             )

    #         logger.info(
    #             f"{'Initializing':12} | Model data loaded successfully num_model={len(message_pb)}"
    #         )
    #     except Exception as e:
    #         raise ex.InitializingFailError(
    #             message="Can not load tag value data from IPCM Server"
    #         ) from e

    @logging_time
    def load_db_info(self) -> dict:
        """Return database model info data"""
        result = {}
        __data = tagvalue_api.get_current_values_all()
        if __data.status_code == 200:
            result = __data.json()
        return result

    def update_data(self, model_key, model_tag_values) -> None:
        self._put(model_key, model_tag_values)

    def _put(self, model_key: str, model_tag_values: dict) -> None:
        try:
            self[model_key].put_nowait(model_tag_values)
        except KeyError:
            self.__add_model(model_key)
            self[model_key].put_nowait(model_tag_values)
        except queue.Full:
            self._pop(model_key)
            self[model_key].put_nowait(model_tag_values)
        except Exception as e:
            logger.warning(e)

    def _pop(self, model_key: str) -> None:
        return self[model_key].get_nowait()

    def __add_model(self, model_key: str) -> None:
        self[model_key] = queue.Queue(maxsize=40)
