from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from types_ import Array


class _TagArrayData:
    def __init__(self, window_size: int) -> None:
        self.timestamps = np.zeros(shape=window_size, dtype=np.int64)
        self.values = np.full(shape=window_size, fill_value=np.nan, dtype=np.float32)
        self.statuscodes = np.zeros(shape=window_size, dtype=int)
        self.ignore_statuses = np.full(
            shape=window_size,
            dtype=bool,
            fill_value=False,
        )

    def update_data(
        self, timestamp: int, value: float, statuscode: int, ignore_status: bool = False
    ):
        self.timestamps[-1] = timestamp
        self.values[-1] = value
        self.statuscodes[-1] = statuscode
        self.ignore_statuses[-1] = ignore_status

    def _push_data(self) -> None:
        self.timestamps[:-1] = self.timestamps[1:]
        self.statuscodes[:-1] = self.statuscodes[1:]
        self.values[:-1] = self.values[1:]
        self.ignore_statuses[:-1] = self.ignore_statuses[1:]


class TagWindowedData:
    def __init__(self, window_size: int = 1) -> None:
        self.raw = _TagArrayData(window_size)
        self.pred = _TagArrayData(window_size)
        self.index = _TagArrayData(window_size)

    def _push_data(self) -> None:
        self.raw._push_data()
        self.pred._push_data()
        self.index._push_data()


class ModelArrayData:
    def __init__(self, numtag: int, window_size: int = 1) -> None:
        self.timestamps: Array["N,2", int] = np.zeros(
            shape=[window_size, numtag],
            dtype=int,
        )
        self.statuscodes: Array["N,2", int] = np.zeros(
            shape=[window_size, numtag],
            dtype=int,
        )
        self.values: Array["N,2", float] = np.full(
            shape=[1, window_size, numtag],
            fill_value=np.nan,
            dtype=np.float32,
        )
        self.ignore_statuses: Array["N,2", bool] = np.full(
            shape=[window_size, numtag],
            fill_value=False,
            dtype=bool,
        )

    def __repr__(self) -> str:
        return f"ArrayData(shape={self.values.shape})"

    @classmethod
    def from_data(
        cls,
        timestamps: Array["N,2", int],
        statuscodes: Array["N,2", int],
        values: Array["N,2", float],
        ignore_statuses: Array["N,2", bool],
    ) -> object:
        instance = cls(numtag=timestamps.shape[1], window_size=timestamps.shape[0])
        instance.timestamps = timestamps
        instance.statuscodes = statuscodes
        instance.values = values
        instance.ignore_statuses = ignore_statuses
        return instance

    def _push_data(self) -> None:
        self.timestamps[:-1] = self.timestamps[1:]
        self.statuscodes[:-1] = self.statuscodes[1:]
        self.values[0][:-1] = self.values[0][1:]
        self.ignore_statuses[0][:-1] = self.ignore_statuses[0][1:]

    def copy(self):
        return deepcopy(self)


@dataclass
class ModelTagData:
    tagnames: list[str]
    data: dict[str, TagWindowedData]

    @property
    def _max_timestamp(self) -> int:
        timestamps_list = [
            self.data[tagname].raw.timestamps for tagname in self.tagnames
        ]
        return np.concatenate(timestamps_list).max()

    @property
    def array_data_raw(self) -> ModelArrayData:
        # Need reshape to 3-dimension array(?)
        values = [self.data[tagname].raw.values for tagname in self.tagnames]
        values = np.stack(values, axis=-1)
        timestamps = [self.data[tagname].raw.timestamps for tagname in self.tagnames]
        timestamps = np.vstack(timestamps)
        statuscodes = [self.data[tagname].raw.statuscodes for tagname in self.tagnames]
        statuscodes = np.vstack(statuscodes)
        ignore_statuses = [
            self.data[tagname].raw.ignore_statuses for tagname in self.tagnames
        ]
        ignore_statuses = np.vstack(ignore_statuses)

        return ModelArrayData.from_data(
            values=values,
            timestamps=timestamps,
            statuscodes=statuscodes,
            ignore_statuses=ignore_statuses,
        )

    @property
    def last_tag_index_list(self) -> np.array:
        return np.array([
            self.data[tagname].index.values[-1] for tagname in self.tagnames
        ])

    @property
    def last_tag_index_status_list(self) -> np.array:
        return np.array([
            self.data[tagname].index.statuscodes[-1] for tagname in self.tagnames
        ])

    def push_data(self) -> None:
        for tagname in self.tagnames:
            self.data[tagname]._push_data()
