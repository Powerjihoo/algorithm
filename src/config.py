import argparse
import os
import sys
from typing import Optional

import yaml

from resources.constant import CONST

parser = argparse.ArgumentParser(prog=CONST.PROGRAM_NAME)
parser.add_argument("--config", type=str, help="config file path")
args = parser.parse_args()

if not (config_path := args.config):
    config_path = "realtime-prediction-server.yaml"


class ServerSettings:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def __repr__(self) -> str:
        return f"{__class__.__name__}(host={self.host}, port={self.port})"


class DatabaseSettings:
    def __init__(
        self,
        host: str,
        port: int,
        database: str = 0,
        username: str = None,
        password: str = None,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}("
            f"host={self.host}, port={self.port}, "
            f"database={self.database}, username={self.username}, password={self.password})"
        )


class LogSettings:
    def __init__(
        self,
        level_file: str,
        level_console: str,
        uvicorn_log_level: str,
        use_api_timing: bool,
        uvicorn_log_level_timing: str,
        logging_router: bool = False,
        logging_request_body: bool = False,
    ):
        self.level_file = level_file
        self.level_console = level_console
        self.uvicorn_log_level = uvicorn_log_level
        self.use_api_timing = use_api_timing
        self.uvicorn_log_level_timing = uvicorn_log_level_timing
        self.logging_router = logging_router
        self.logging_request_body = logging_request_body

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}("
            f"level_file={self.level_file}, level_console={self.level_console}, "
            f"uvicorn_log_level={self.uvicorn_log_level}, "
            f"use_api_timing={self.use_api_timing}, uvicorn_log_level_timing={self.uvicorn_log_level_timing})"
        )


class KafkaSettings:
    def __init__(self, brokers: list, topic_model_values: str, topic_pred_values: str, max_poll_records: int=2):
        self.brokers = brokers
        self.topic_model_values = topic_model_values
        self.topic_pred_values = topic_pred_values
        self.max_poll_records = max_poll_records

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}("
            f"brokers={self.brokers}, topic_model_values={self.topic_model_values}, "
            f"topic_pred_values={self.topic_pred_values})"
            f"max_poll_records={self.max_poll_records})"
        )


class DataSettings:
    def __init__(self, model_folder_window: str, model_folder_linux: str):
        if sys.platform.startswith("win"):
            self.model_path = model_folder_window
        else:
            self.model_path = model_folder_linux

    def __repr__(self) -> str:
        return f"{__class__.__name__}(model_path={self.model_path}"


class SystemSettings:
    def __init__(
        self,
        target_gpu_device_no: str,
        enable_swagger: bool,
        wait_for_model_initializing: bool = True,
        force_ignore_idx_100: bool = False,
    ):
        self.target_gpu_device_no = target_gpu_device_no
        self.enable_swagger = enable_swagger
        self.wait_for_model_initializing = wait_for_model_initializing
        self.force_ignore_idx_100 = force_ignore_idx_100

    def __repr__(self) -> str:
        return f"{__class__.__name__}(target_gpu_device_no={self.target_gpu_device_no})"


class AppConfig:
    def __init__(
        self,
        server_id: str,
        servers: dict,
        databases: dict,
        log: LogSettings,
        data: DataSettings,
        system: SystemSettings,
        kafka: KafkaSettings,
    ):
        self.server_id = server_id
        self.servers = servers
        self.databases = databases
        self.log = log
        self.data = data
        self.system = system
        self.kafka = kafka

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}(servers={self.servers}, "
            f"databases={self.databases}, log={self.log}, data={self.data}, "
            f"gpu={self.system}, kafka={self.kafka})"
        )


def load_app_config_from_yaml(file_path: str) -> Optional[AppConfig]:
    try:
        print(f"Loading config file... {os.getcwd()}/{file_path}")
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        server_id = yaml_data.get("server-id")
        servers = {
            server_name: ServerSettings(**settings)
            for server_name, settings in yaml_data.get("servers", {}).items()
        }
        databases = {
            db_name: DatabaseSettings(**settings)
            for db_name, settings in yaml_data.get("databases", {}).items()
        }
        log = LogSettings(**yaml_data.get("log", {}))
        data = DataSettings(**yaml_data.get("data", {}))
        system = SystemSettings(**yaml_data.get("system", {}))
        kafka = KafkaSettings(**yaml_data.get("kafka", {}))
        return AppConfig(
            server_id=server_id,
            servers=servers,
            databases=databases,
            log=log,
            data=data,
            system=system,
            kafka=kafka,
        )
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None


settings = load_app_config_from_yaml(config_path)
