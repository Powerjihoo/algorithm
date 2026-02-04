import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RouterLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, logger: logging.Logger) -> None:
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next):
        # Generate a unique request ID
        request_id = str(uuid4())

        # Log the request
        log_str_request = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client": request.client.host,
            "headers": dict(request.headers),
        }
        self.logger.debug(f"Request: {log_str_request}")

        # Log request body if present
        if request.method != "GET":
            body = await request.body()
            self.logger.trace(f"Request body: {body.decode()}")

        # Call the next middleware or handler
        response = await call_next(request)

        # Log the response
        log_str_response = {
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        self.logger.debug(f"Response: {log_str_response}")

        # Log response body if present
        if hasattr(response, "body"):
            self.logger.trace(f"Response body: {response.body.decode()}")

        return response
