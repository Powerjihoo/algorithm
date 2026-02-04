class ModelDataCountNotMatchedError(Exception):
    def __init__(
        self,
        num_model_data_col,
        num_db_tag,
        message="The number of model columns is not matched with database number of tags",
    ):
        self.num_model_data_col = num_model_data_col
        self.num_db_tag = num_db_tag
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        # logger.debug(f"{self.message} | {self.num_model_data:} {self.num_db_tag}")
        return f"{'Not Loaded':12} | {__class__.__name__}: {self.message} | {self.num_model_data_col=} {self.num_db_tag=}"


class UndefinedModelTypeError(Exception):
    def __init__(self, modeltype: str, message="Model type is not defiend"):
        self.modeltype = modeltype
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{'Not Loaded':12} | {__class__.__name__}: {self.message} | {self.modeltype=}"


class UndefinedModelSettingError(Exception):
    def __init__(self, modeltype: str, message="Model setting is not defiend"):
        self.modeltype = modeltype
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{'Not Loaded':12} | {__class__.__name__}: {self.message} | {self.modeltype=}"


class NotFoundModelFileError(Exception):
    def __init__(self, model_key: str, message: str = "Can not find model file"):
        self.message = message
        self.model_key = model_key
        super().__init__(self.message)

    def __str__(self):
        message_ln1 = f"{'Not Loaded':12} | {__class__.__name__}: {self.message}"
        message_ln2 = f"{' '*18}{self.model_key=}"
        return f"{message_ln1}\n{message_ln2}"


class NotFoundActivatedModelError(Exception):
    def __init__(
        self, model_key: str, message: str = "Requested model is not activated"
    ):
        self.message = message
        self.model_key = model_key
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} | model_key: {self.model_key}"


class NotFoundTagError(Exception):
    def __init__(self, tagname: str, message: str = "Requested tag is not found"):
        self.message = message
        self.tagname = tagname
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} | tagname: {self.tagname}"


class InvalidSPRTParameterError(Exception):
    def __init__(
        self,
        model_key: str,
        alpha: float,
        beta: float,
        message: str = "Probably can not do log calculation ",
    ):
        self.message = message
        self.model_key = model_key
        self.alpha = alpha
        self.beta = beta
        super().__init__(self.message)

    def __str__(self):
        message_ln1 = f"{'Not Loaded':12} | {__class__.__name__}: {self.message}"
        message_ln2 = f"{' '*18}{self.model_key=} {self.alpha=} {self.beta=}"
        return f"{message_ln1}\n{message_ln2}"


class InvalidRangeError(Exception):
    def __init__(self, range_min, range_max, message: str = "Invalid range"):
        self.range_min = range_min
        self.range_max = range_max
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{self.message} | range_min={self.range_min}, range_max={self.range_max}"
        )


class ModelInitializeError(Exception):
    def __init__(self, model_key: str, message: str = "Can not initialize model"):
        self.model_key = model_key
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        message_ln1 = f"{'Not Loaded':12} | {__class__.__name__}: {self.message}"
        message_ln2 = f"{' '*18}{self.model_key=}"
        return f"{message_ln1}\n{message_ln2}"


class ModelTagSettingError(Exception):
    def __init__(
        self, model_key: str, tagname: str, message: str = "Model tag not found"
    ):
        self.model_key = model_key
        self.tagname = tagname
        self.message = message
        super().__init__(self.message)
