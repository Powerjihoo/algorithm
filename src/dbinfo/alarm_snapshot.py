from api_client.apis.alarms import alarm_api
from utils.logger import logger, logging_time
from utils.scheme.singleton import SingletonInstance
import threading


class AlarmSnapshot(metaclass=SingletonInstance):
    def __init__(self):
        self._lock = threading.Lock()
        self.data = {}

    @logging_time
    def initialize(self):
        logger.debug(f"{'Initializing':12} | Loading alarm snapshot data...")
        __data = self.load_db_info()
        self.data = {}
        self.update_alarm_info(__data)

    def load_db_info(self) -> dict:
        try:
            res = alarm_api.get_alarm_snapshot()
            return res.json()
        except ValueError:
            logger.error("Could not retrieve alarm snapshot data")
            return {}

    def update_alarm_info(self, alarm_data) -> None:
        with self._lock:
            for model_key, model_alarm_info in alarm_data.items():
                for tagname, alarm_info in model_alarm_info.items():
                    try:
                        self.data[model_key][tagname] = alarm_info["result"]
                    except KeyError:
                        self.data[model_key] = {tagname: alarm_info["result"]}
            logger.debug(f"{'Initializing':12} | Alarm snapshot loaded successfully")

    def get_status_from_tagname(self, model_key, tagname):
        try:
            return self.data[model_key][tagname]
        except KeyError:
            logger.error(f"No {model_key=} {tagname=}")
