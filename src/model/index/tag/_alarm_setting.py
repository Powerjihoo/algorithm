import inspect
from abc import ABC
from dataclasses import dataclass

from types_ import TAlarmSettingDB, TAlarmType
from utils.scheme.singleton import SingletonInstance


@dataclass
class AlarmSettingBase(ABC):
    @classmethod
    def from_dict(cls, settings: dict):
        return cls(
            **{
                k: v
                for k, v in settings.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class AlarmSettingThreshold(AlarmSettingBase):
    threshold: float
    deadband: float = 0.0


@dataclass
class AlarmSettingSPRT(AlarmSettingBase):
    m0: float
    s0: float
    m: float
    s: float
    alpha: float = 0.0
    beta: float = 0.0


@dataclass
class AlarmSettingSlope(AlarmSettingBase):
    normal_slope: float
    deadband: float = 0.0


class AlarmSettingsClasses(metaclass=SingletonInstance):
    def __init__(self):
        self.Threshold = AlarmSettingThreshold
        self.SPRT = AlarmSettingSPRT
        self.Slope = AlarmSettingSlope

    def __getitem__(self, key):
        return getattr(self, key)

    def get_alarm_setting(
        self, alarm_type: TAlarmType, alarm_setting: TAlarmSettingDB
    ) -> AlarmSettingBase | None:
        if alarm_type is None:
            return None
        alarm_setting_class: AlarmSettingBase = self[alarm_type]
        return alarm_setting_class.from_dict(alarm_setting.__dict__)
