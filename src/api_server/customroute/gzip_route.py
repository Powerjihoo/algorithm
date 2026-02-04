import gzip
from base64 import standard_b64decode
from typing import Callable

import orjson
from fastapi import Request, Response
from fastapi.routing import APIRoute


class GzipRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            _gzip_decoded_decomp = await super().body()
            if "gzip" in self.headers.getlist("Content-Encoding"):
                _gzip_decoded = standard_b64decode(_gzip_decoded_decomp)
                _gzip_decoded_decomp = gzip.decompress(_gzip_decoded)
            self._body = orjson.loads(_gzip_decoded_decomp)
        return self._body


class GzipRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = GzipRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler
