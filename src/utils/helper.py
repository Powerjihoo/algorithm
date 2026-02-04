from typing import Tuple
from dbinfo import exceptions as ex_db


def parse_ignore_setting(ignore_settings) -> Tuple[str]:
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
                        setting["value"],
                        self.dict_inequality[setting["condition"]],
                        ignore_tagname_var,
                        ")",
                    ]
                )

                expressions.append(expression)
            if expressions:
                expressions_AND.append("".join(["(", " and ".join(expressions), ")"]))

        expressions_final = " or ".join(expressions_AND)
        if not expressions_final:
            raise ex_db.IgnoreSettingParsingError
    except Exception as err:
        raise ex_db.IgnoreSettingParsingError from err
    return expressions_final, ignore_tagnames
