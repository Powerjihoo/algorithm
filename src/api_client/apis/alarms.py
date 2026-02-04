# alarms.py

""" API for requesting ALARM information data to IPCM Server """
import orjson
from api_client.apis.session import APISession

basepath = "/api/Python"


class AlarmAPI(APISession):
    def __init__(self):
        self.headers = {"Content-Type": "application/json-patch+json"}
        super().__init__()

    def get_alarm_snapshot(self):
        path = "/alarmsnapshot"
        url = self.baseurl + basepath + path
        return self.request_get(url)

    def post_alarm(self, data: dict):
        path = "/api/Alarm/model/python"
        url = self.baseurl + path

        return self.request_post(
            url=url,
            data=orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode(),
            headers=self.headers,
        )


alarm_api = AlarmAPI()
