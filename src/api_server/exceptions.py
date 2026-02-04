from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from utils.logger import logger


class APIExeption(Exception):
    status_code: int
    code: str
    msg: str
    detail: str

    def __init__(
        self,
        *,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        code: str = "000000",
        msg: str = None,
        detail: str = None,
        ex: Exception = None,
    ):
        self.status_code = status_code
        self.code = code
        self.msg = msg
        self.detail = detail
        super().__init__(ex)


class ModelAlreadyActivated(Exception):
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.status_code = status.HTTP_406_NOT_ACCEPTABLE
        self.message = "The requested model is already activated"
        self.detail = {"model_key": self.model_key}
        logger.error(
            f"Requested model activation for already activated model | model_key={self.model_key}"
        )


class ModelAlreadyDeactivated(Exception):
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.status_code = status.HTTP_406_NOT_ACCEPTABLE
        self.message = "The requested model is already deactivated"
        self.detail = {"model_key": self.model_key}
        logger.error(
            f"Requested model activation for already deactivated model | model_key={self.model_key}"
        )


class CanNotActivateModel(Exception):
    def __init__(
        self,
        model_key: str,
        message: str = "The requested model could not be activated",
    ):
        self.model_key = model_key
        self.status_code = status.HTTP_406_NOT_ACCEPTABLE
        self.message = message
        self.detail = {"model_key": self.model_key}
        logger.error(
            f"Requested model could not be activated. {self.message} | model_key={self.model_key}"
        )


class CanNotDeactivateModel(Exception):
    def __init__(
        self,
        model_key: str,
        message: str = "The requested model could not be deactivated",
    ):
        self.model_key = model_key
        self.status_code = status.HTTP_406_NOT_ACCEPTABLE
        self.message = message
        self.detail = {"model_key": self.model_key}
        logger.error(
            f"Requested model could not be deactivated | model_key={self.model_key}"
        )


class InvalidRequestBody(Exception):
    def __init__(self, message: str = "Invalid request body"):
        self.status_code = status.HTTP_406_NOT_ACCEPTABLE
        self.message = message
        logger.debug("Requested invalid body")


class CanNotUpdateModelAlarmSetting(Exception):
    def __init__(
        self,
        model_key: str,
        tagname: str = None,
        message: str = "Can not update model alarm setting",
    ):
        self.status_code = status.HTTP_406_NOT_ACCEPTABLE
        self.message = message
        self.model_key = model_key
        self.tagname = tagname
        logger.error(
            f"Requested model alarm setting could not be updated | model_key={self.model_key}, tagname={self.tagname}"
        )


class CanNotUpdateTagSetting(Exception):
    def __init__(self, message: str = " Could not update tag settings"):
        self.status_code = status.HTTP_409_CONFLICT
        self.message = message
        logger.error(self.message)


class SubProcessError(Exception):
    def __init__(
        self, message: str = "Sub processs could not properly process the request."
    ):
        self.status_code = status.HTTP_409_CONFLICT
        self.message = message
        # TODO: Need to add request information
        logger.error(message)


class ProcAgentBusyError(Exception):
    def __init__(
        self, message: str = "Please rquest after a while. API Proxy server is busy."
    ):
        self.status_code = status.HTTP_409_CONFLICT
        self.message = message
        logger.error("Proxy server has blocked request due to treating other task")


async def http_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=exc.status_code, content=str(exc.detail))


async def validation_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(status_code=400, content=str(exc))


async def model_already_activated_exception_handler(
    request: Request, exc: ModelAlreadyActivated
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "detail": exc.detail,
        },
    )


async def model_already_deactivated_exception_handler(
    request: Request, exc: ModelAlreadyDeactivated
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "detail": exc.detail,
        },
    )


async def can_not_activate_model_exception_handler(
    request: Request, exc: CanNotActivateModel
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "detail": exc.detail,
        },
    )


async def can_not_deactivate_model_exception_handler(
    request: Request, exc: CanNotDeactivateModel
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "detail": exc.detail,
        },
    )


async def invalid_request_body(request: Request, exc: InvalidRequestBody):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


async def cannot_update_model_alarm_setting(
    request: Request, exc: CanNotUpdateModelAlarmSetting
):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


async def sub_process_error(request: Request, exc: SubProcessError):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


async def cannot_update_tag_setting(request: Request, exc: CanNotUpdateTagSetting):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


async def proc_agent_busy(request: Request, exc: ProcAgentBusyError):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


def add_exception_handlers(app):
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(
        ModelAlreadyActivated, model_already_activated_exception_handler
    )
    app.add_exception_handler(
        ModelAlreadyDeactivated, model_already_deactivated_exception_handler
    )
    app.add_exception_handler(
        CanNotActivateModel, can_not_activate_model_exception_handler
    )
    app.add_exception_handler(
        CanNotDeactivateModel, can_not_deactivate_model_exception_handler
    )
    app.add_exception_handler(InvalidRequestBody, invalid_request_body)
    app.add_exception_handler(
        CanNotUpdateModelAlarmSetting, cannot_update_model_alarm_setting
    )
    app.add_exception_handler(SubProcessError, sub_process_error)
    app.add_exception_handler(CanNotUpdateTagSetting, cannot_update_tag_setting)
    app.add_exception_handler(ProcAgentBusyError, proc_agent_busy)
