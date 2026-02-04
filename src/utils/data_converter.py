def convert_listdict2dict(data: list, key_name: str) -> dict:
    """Convert list data"""
    return {_listdict.pop(key_name): _listdict for _listdict in data}


def convert_table2dict(columns: list, data: list) -> dict:
    """Convert database table data(2-dimension) to python dictionary

    Parameters
    ----------
    columns : list
        table columns
        thr first column will be the key of the result dictionary
        others will be the key of the each value
    data : list
        table data
    result : dict, optional
        dictionary result, by default {}

    Returns
    -------
    dict
        data that converted to dictionary
    """
    return {str(row[0]): dict(zip(columns[1:], row[1:])) for row in data}


def convert_case_dict_key(data, option="lower"):
    from caseconverter import camelcase, pascalcase

    if option == "lower":
        return {k.lower(): v for k, v in data.items()}
    elif option == "upper":
        return {k.upper(): v for k, v in data.items()}
    elif option == "pascalcase":
        return {pascalcase(k): v for k, v in data.items()}
    elif option == "camelcase":
        return {camelcase(k): v for k, v in data.items()}


def convert_dict_2key_upper(data):
    if isinstance(data, dict):
        return {
            k1: {k2.upper(): v2 for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
    else:
        return data


def convert_dict_key_value_upper(data):
    if isinstance(data, dict):
        return {k.upper(): convert_dict_key_value_upper(v) for k, v in data.items()}
    else:
        return data


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [object_to_dict(elem) for elem in obj]
    elif hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj
