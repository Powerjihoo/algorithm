# taginfo.py

""" API for requesting basic tag information data to IPCM Server """

from typing import List
from api_client.apis.session import APISession
import orjson

basepath = "/api/Python/taglist"


class TagInfoAPI(APISession):
    def __init__(self):
        super().__init__()
        self.headers = {"Content-Type": "application/json-patch+json"}

    def get_taginfo(self):
        path = ""
        url = self.baseurl + basepath + path
        return self.request_get(url)

    def get_taginfo_for_tags(self, tagnames: list[str]):
        path = ""
        url = self.baseurl + basepath + path
        return self.request_post(url, data=orjson.dumps(tagnames), headers=self.headers)

    def get_taginfo_compressed(self):
        path = "/api/PythonCompress/taglist"
        url = self.baseurl + path
        return self.request_get(url)

    def get_taginfo_tagnames(self, tagnames: List[str]):
        path = f"/{tagnames}"
        url = self.baseurl + basepath + path
        return self.request_get(url)


taginfo_api = TagInfoAPI()
