# model.py

"""API for requesting MODEL information data to IPCM Server"""

from api_client.apis.session import APISession
from config import settings

basepath = "/api/Python/modellist"


class ModelAPI(APISession):
    def __init__(self):
        super().__init__(
            host=settings.servers["ipcm-server"].host,
            port=settings.servers["ipcm-server"].port,
        )

    # def load_modelinfo_for_each_channel(self):
    #     url = f"{self.baseurl}/api/Python/kafka/modellist/running"
    #     return self.request_get(url)
    

    # redis 기능 웹으로 이관되어 사용하는 url 주소 변경, 하드코딩
    # def load_modelinfo_for_each_channel(self):
    #     path = ""
    #     url = f"http://192.168.30.90:50531/api/model/algorithm"
    #     return self.request_get(url)
    
    # yaml 파일에서 불러오도록 변경
    def load_modelinfo_for_each_channel(self):
        web_key = "ipcm-web"
        path = "/api/model/algorithm"
        srv = settings.servers[web_key]
        url = f"http://{srv.host}:{srv.port}{path}"
        return self.request_get(url)

    def load_modelinfo(self):
        path = ""
        url = self.baseurl + basepath + path
        return self.request_get(url)

    def load_modelinfo_model_key(self, model_key: str):
        path = f"/{model_key}"
        url = self.baseurl + basepath + path
        return self.request_get(url)
    
    def post_run_model(self, modeltype: str, modelname: str):
        path = f"/api/Model/model/run/{modeltype}/{modelname}"
        url = self.baseurl + path
        return self.request_post(url)

    def post_stop_model(self, modeltype: str, modelname: str):
        path = f"/api/Model/model/stop/{modeltype}/{modelname}"
        url = self.baseurl + path
        return self.request_post(url)

model_api = ModelAPI()
