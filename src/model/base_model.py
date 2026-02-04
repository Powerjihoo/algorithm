from abc import ABC

from model import exceptions as ex_model


class ModelBase(ABC):
    def __init__(
        self,
        modeltype: str,
        modelname: str,
        tagnames: list[str],
        targettagnames: list[str],
        calc_interval: int = 1,
        version: int = 0,  
    ):
        padded_version = f"{version:04d}"  # 항상 4자리로 패딩
        self.model_key = f"{modeltype}_{modelname}_{padded_version}".lower()
        if not tagnames or not targettagnames:
            raise ex_model.ModelInitializeError(self.model_key, "Empty tagnames")
        if isinstance(tagnames, str):
            tagnames = [tagnames]
        if isinstance(tagnames, str):
            targettagnames = [targettagnames]
        self.tagnames = tagnames
        self.targettagnames = targettagnames
        self.modelname = modelname
        self.calc_interval = calc_interval
        self.numtag = len(self.tagnames)
