from typing import List

from pydantic import BaseModel


class ModelSettingAAKR(BaseModel):
    tagnames: List[str]


class TestAAKR(BaseModel):
    tagnames: List[str] = ["TagA", "TagB", "TagC"]
    data: List[List[float]] = [
        [10.0, 100.1, 9834.24],
        [10.1, 103.0, 9612.4],
        [10.2, 98.7, 9207.36],
        [9.9, 99.5, 9406.2],
        [10.1, 97.9, 9559.801],
    ]
