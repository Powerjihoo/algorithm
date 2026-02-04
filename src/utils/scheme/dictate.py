import contextlib


class Dictate(object):
    """Object view of a dict, updating the passed in dict when values are set
    or deleted. "Dictate" the contents of a dict...:"""

    def __init__(self, data):
        # since __setattr__ is overridden, self.__dict = d doesn't work
        object.__setattr__(self, "_Dictate__dict", data)
        with contextlib.suppress(Exception):
            super().__init__(data)

    # Dictionary-like access / updates
    def __getitem__(self, name):
        value = self.__dict[name]
        if isinstance(value, dict):  # recursively view sub-dicts as objects
            value = Dictate(value)
        return value

    def __setitem__(self, name, value):
        self.__dict[name] = value

    def __delitem__(self, name):
        del self.__dict[name]

    # Object-like access / updates
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.__dict)

    def __str__(self):
        return str(self.__dict)

    def __len__(self):
        return len(self._Dictate__dict)


class CaseInsensitiveDictate(object):
    """It is able to access key name by case insensitivity"""

    def __init__(self, data):
        # since __setattr__ is overridden, self.__dict = d doesn't work
        object.__setattr__(self, "_CaseInsensitiveDictate__dict", data)
        object.__setattr__(
            self, "_CaseInsensitiveDictate__dict_casefold", {k.lower(): k for k in data}
        )

    # Dictionary-like access / updates
    def __getitem__(self, name):
        value = self.__dict[self.__dict_casefold[name.lower()]]
        if isinstance(value, dict):  # recursively view sub-dicts as objects
            value = CaseInsensitiveDictate(value)
        return value

    def __setitem__(self, name, value):
        self.__dict[name] = value
        self.__dict_casefold[name.lower()] = name

    def __delitem__(self, name):
        del self.__dict[self.__dict_casefold[name.lower()]]
        del self.__dict_casefold[name.lower()]

    # Object-like access / updates
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.__dict)

    def __str__(self):
        return str(self.__dict)

    def __len__(self):
        return len(self._CaseInsensitiveDictate__dict)
