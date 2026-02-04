# etc.py


from typing import Tuple, Union

from api_client.apis.session import APISession


class ETCAPI(APISession):
    def __init__(self):
        self.headers = {"Content-Type": "application/json-patch+json"}
        super().__init__()

    def get_root(self, timeout: Union[None, Tuple, int] = None):
        path = "/"
        url = self.baseurl + path
        return self.request_get(url, timeout=timeout)


etc_api = ETCAPI()
