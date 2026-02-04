import contextlib
import threading
from time import time as time_current

from api_client.apis.alarms import alarm_api
from dbinfo.alarm_snapshot import AlarmSnapshot
from utils.logger import logger
from utils.scheme.singleton import SingletonInstance

alarm_snapshot = AlarmSnapshot()


class AlarmManager(metaclass=SingletonInstance):
    """Manage alarm data"""

    def __init__(self):
        # self._lock = threading.Lock()
        self.alarm_confirm_buffer = {}
        self.return_confirm_buffer = {}
        self.lock = threading.Lock()

    def clear_alarm_confirm_dict(self, result_code: int):
        self.alarm_confirm_buffer.clear() if result_code.status_code == 200 else None

    def clear_return_confirm_dict(self, result_code: int):
        self.return_confirm_buffer.clear() if result_code.status_code == 200 else None

    def update_alarm_status_alarm(self):
        for model_key, alarms in self.alarm_confirm_buffer.items():
            for tagname in alarms:
                try:
                    alarm_snapshot.data[model_key][tagname] = True
                except KeyError:
                    try:
                        alarm_snapshot.data[model_key] = {tagname: True}
                    except KeyError:
                        alarm_snapshot.data = {model_key: {tagname: True}}

    def update_alarm_status_return(self):
        for model_key, alarms in self.return_confirm_buffer.items():
            for tagname in alarms:
                try:
                    alarm_snapshot.data[model_key][tagname] = False
                except KeyError:
                    try:
                        alarm_snapshot.data[model_key] = {tagname: False}
                    except KeyError:
                        alarm_snapshot.data = {model_key: {tagname: False}}

    def send_alarm(self) -> None:
        # with self._lock:0
        try:
            if self.alarm_confirm_buffer:
                self.update_alarm_status_alarm()
                result_code = alarm_api.post_alarm(self.alarm_confirm_buffer)
                self.clear_alarm_confirm_dict(result_code)

        except AttributeError:
            print("No Alarm Data yet")

    def send_return(self) -> None:
        # with self._lock:
        try:
            if self.return_confirm_buffer:
                self.update_alarm_status_return()
                result_code = alarm_api.post_alarm(self.return_confirm_buffer)
                self.clear_return_confirm_dict(result_code)
        except AttributeError:
            print("No Return Data yet")


class AlarmStaytimeManager(metaclass=SingletonInstance):
    def __init__(self) -> None:
        self.alarm_staytime_dict = {}

    def add_alarm_staytime_checking(
        self,
        alarm_server_ts: float,
        model_key: str,
        tagname: str,
        timestamp: int,
        value: float,
        pred: float,
        residual: float,
        quality: int,
        setpoint: float,
        threshold: float,
        staytime: int,
        state: str = "Predict",
    ) -> None:
        try:
            if tagname not in self.alarm_staytime_dict[model_key]:
                self.alarm_staytime_dict[model_key][tagname] = {
                    "alarm_ts": timestamp + staytime * 1000,
                    "value": value,
                    "pred": pred,
                    "quality": quality,
                    "residual": residual,
                    "state": state,
                    "setpoint": setpoint,
                    "threshold": threshold,
                    "staytime": staytime,
                    "end_time": alarm_server_ts + staytime * 1000,
                }
        except KeyError:
            self.alarm_staytime_dict[model_key] = {
                tagname: {
                    "alarm_ts": timestamp + staytime * 1000,
                    "value": value,
                    "pred": pred,
                    "quality": quality,
                    "residual": residual,
                    "state": state,
                    "setpoint": setpoint,
                    "threshold": threshold,
                    "staytime": staytime,
                    "end_time": alarm_server_ts + staytime * 1000,
                }
            }

    def remove_alarm_staytime_checking(self, model_key: str, tagname: str) -> None:
        with contextlib.suppress(KeyError, TypeError):
            del self.alarm_staytime_dict[model_key][tagname]

    def check_alarm_staytime(self) -> None:
        current_time = time_current() * 1000
        with contextlib.suppress(AttributeError):
            for model_key, v in list(self.alarm_staytime_dict.items()):
                for tagname, items in list(v.items()):
                    if abs(items["value"] - items["pred"]) > items["threshold"]:
                        if current_time > items["end_time"]:
                            self.add_alarm_confirm_dict(model_key, tagname)
                    else:
                        self.remove_alarm_staytime_checking(model_key, tagname)

    def add_alarm_confirm_dict(self, model_key, tagname):
        try:
            manager_alarm.alarm_confirm_buffer[model_key] = {
                tagname: self.alarm_staytime_dict[model_key].pop(tagname)
            }
        except KeyError:
            manager_alarm.alarm_confirm_buffer = {}
            try:
                manager_alarm.alarm_confirm_buffer[model_key] = {
                    tagname: self.alarm_staytime_dict[model_key].pop(tagname)
                }
            except Exception as e:
                logger.error(e)
        except Exception as e:
            logger.error(e)


class AlarmReturnManager(metaclass=SingletonInstance):
    def add_return_confirm_dict(
        self,
        alarm_server_ts: float,
        model_key: str,
        tagname: str,
        timestamp: int,
        value: float,
        pred: float,
        residual: float,
        quality: int,
        setpoint: float,
        threshold: float,
        staytime: int,
        state: str = "RTN",
    ) -> None:
        try:
            if tagname not in manager_alarm.return_confirm_buffer[model_key]:
                manager_alarm.return_confirm_buffer[model_key][tagname] = {
                    "alarm_ts": timestamp,
                    "value": value,
                    "pred": pred,
                    "quality": quality,
                    "residual": residual,
                    "state": state,
                    "setpoint": setpoint,
                    "threshold": threshold,
                    "staytime": staytime,
                    "end_time": alarm_server_ts,
                }
        except KeyError:
            manager_alarm.return_confirm_buffer[model_key] = {
                tagname: {
                    "alarm_ts": timestamp,
                    "value": value,
                    "pred": pred,
                    "quality": quality,
                    "residual": residual,
                    "state": state,
                    "setpoint": setpoint,
                    "threshold": threshold,
                    "staytime": staytime,
                    "end_time": alarm_server_ts,
                }
            }

        except Exception as err:
            print(err)


class AlarmReactivateTimeManager(metaclass=SingletonInstance):
    """
    _summary_

    Args:
        SingletonInstane (_type_): _description_
    """

    def __init__(self) -> None:
        self.alarm_reactivate_dict = {}

    def add_alarm_reactivate_checking(
        self,
        alarm_model,
        alarm_server_ts: float,
        model_key: str,
        tagname: str,
        timestamp: int,
        reactivate_time: int,
    ) -> None:
        # * Need to validate this method
        try:
            if tagname in self.alarm_reactivate_dict[model_key]:
                self.alarm_reactivate_dict[model_key][tagname] = {
                    "alarm_model": alarm_model,
                    "alarm_ts": timestamp + reactivate_time * 1000,
                    "end_time": alarm_server_ts + reactivate_time * 1000,
                }
        except KeyError:
            self.alarm_reactivate_dict[model_key] = {
                tagname: {
                    "alarm_model": alarm_model,
                    "alarm_ts": timestamp + reactivate_time * 1000,
                    "end_time": alarm_server_ts + reactivate_time * 1000,
                }
            }

    def check_alarm_reactivate(self) -> None:
        current_time = time_current() * 1000
        with contextlib.suppress(AttributeError):
            for model_key, v in list(self.alarm_reactivate_dict.items()):
                for tagname, items in list(v.items()):
                    if current_time > items["end_time"]:
                        # TODO: if reactivate satisfied, need to change ignore status (True -> False)
                        self.alarm_reactivate_dict[model_key][tagname][
                            "alarm_model"
                        ].ignore_status = False
                        self.remove_alarm_reactivate_checking(model_key, tagname)

    def remove_alarm_reactivate_checking(self, model_key: str, tagname: str) -> None:
        with contextlib.suppress(KeyError, TypeError):
            del self.alarm_reactivate_dict[model_key][tagname]


manager_alarm = AlarmManager()
manager_alarm_staytime = AlarmStaytimeManager()
manager_alarm_reactivatetime = AlarmReactivateTimeManager()
manager_return = AlarmReturnManager()
