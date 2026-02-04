class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

    # Dictionary-like access / updates
    def __getitem__(self, name):
        value = self.__dict[name]
        if isinstance(value, dict):  # recursively view sub-dicts as objects
            value = Struct(value)
        return value

    def __repr__(self):
        return (
            "{ "
            + str(
                ", ".join(
                    [
                        f"'{k}': {v}"
                        for k, v in [(k, repr(v)) for (k, v) in self.__dict__.items()]
                    ]
                )
            )
            + " }"
        )
