import gzip
from base64 import standard_b64decode
from typing import Tuple

import orjson

from api_client.apis.taginfo import taginfo_api
from dbinfo import exceptions as ex
from utils.logger import logger, logging_time
from utils.scheme.dictate import Dictate
from utils.scheme.singleton import SingletonInstance


# FIXME: Do I have to remove dictate???
class TagInfo(Dictate, metaclass=SingletonInstance):
    dict_inequality = {1: "==", 2: "<", 3: "<=", 4: ">", 5: ">="}

    def __init__(self):
        super().__init__({})

    def initialize(self, tagnames: list[str] = None):
        try:
            logger.debug(f"{'Initializing':12} | Loading tag info data...")
            _data = self.load_db_info(tagnames)
            _data = self.add_ignore_settings(_data)
            super().__init__(_data)
        except Exception as err:
            raise ex.InitializingFailError(
                message="Can not load taginfo data from IPCM Server"
            ) from err

    @logging_time
    def load_db_info(self, tagnames: list[str] = None) -> dict:
        """Return database tag info data"""
        if tagnames is None:
            __data = taginfo_api.get_taginfo().json()
        else:
            __data = taginfo_api.get_taginfo_for_tags(tagnames).json()
        logger.info(
            f"{'Initializing':12} | Tag info loaded successfully num={len(__data):,}"
        )
        return __data

    @logging_time
    def load_db_info_compressed(self) -> dict:
        """Return database tag info data"""
        _gzip_data = taginfo_api.get_taginfo_compressed().content
        _gzip_decoded = standard_b64decode(_gzip_data)
        _gzip_decoded_decomp = gzip.decompress(_gzip_decoded)
        _data = orjson.loads(_gzip_decoded_decomp)

        logger.info(
            f"{'Initializing':12} | Tag info loaded successfully num={len(_data):,}"
        )
        return _data

    def update_tag_setting(self, tagname: str, tag_setting: dict):
        self[tagname] = tag_setting
        # logger.debug(f"Updated taginfo {tagname}")

    def add_ignore_settings(self, data: dict) -> Dictate:
        for tagname, v in data.items():
            data[tagname]["ignore_expression"] = ""
            data[tagname]["ignore_tagnames"] = ""

            if v["ignoreEnable"]:
                try:
                    ignore_settings = v["ignoreSetting"]["values"]
                    (
                        data[tagname]["ignore_expression"],
                        data[tagname]["ignore_tagnames"],
                    ) = self.parse_ignore_setting(ignore_settings)
                except ex.IgnoreSettingParsingError as err:
                    data[tagname]["ignore_expression"] = ""
                    ignoreEnable = v["ignoreEnable"]
                    ignore_setting = v["ignoreSetting"]
                    logger.warning(
                        f"{err}\n{'>>> ':>10}{tagname=} {ignoreEnable=} {ignore_setting}"
                    )
                except Exception as err:
                    logger.error(f"{err}{tagname=} {v=}")
        return data

    def parse_ignore_setting(self, ignore_settings) -> Tuple[str]:
        ignore_tagnames = []
        expressions_AND = []
        try:
            for settings in ignore_settings:
                expressions = []

                for setting in settings["values"]:
                    if not setting:
                        continue
                    ignore_tagname = setting["ignoreTagName"]
                    ignore_tagnames.append(ignore_tagname)

                    ignore_tagname_var = f'locals()["{ignore_tagname}"]'
                    expression = " ".join(
                        [
                            "(",
                            ignore_tagname_var,
                            self.dict_inequality[setting["condition"]],
                            setting["value"],
                            ")",
                        ]
                    )

                    expressions.append(expression)
                if expressions:
                    expressions_AND.append(
                        "".join(["(", " and ".join(expressions), ")"])
                    )

            expressions_final = " or ".join(expressions_AND)
            if not expressions_final:
                raise ex.IgnoreSettingParsingError
        except Exception as err:
            raise ex.IgnoreSettingParsingError from err
        return expressions_final, ignore_tagnames
