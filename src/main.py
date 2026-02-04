import multiprocessing as mp
import sys
import time

from config import settings

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        mp.freeze_support()
    else:
        mp.set_start_method("spawn")
    from api_client.apis.etc import etc_api
    from utils.logger import logger

    """Check condition wheter IPMC Server is running or not
     - Check IPCM Server HTTP communication
     - If not connected, write log "Can not connect to IPCM Server,
    Waiting for IPCM Server to be up... (IP, Port)
     - And retry connection checking
    """
    while True:
        try:
            if etc_api.get_root(timeout=10).status_code != 200:
                logger.error(
                    f"Unable to connect to IPCM Server. Waiting for IPCM Server "
                    f"to be up...[{etc_api.host}:{etc_api.port}]"
                )
                time.sleep(10)
            else:
                logger.info("IPCM Server http connection has been established")
                break
        except Exception as e:
            logger.error(
                f"Unable to connect to IPCM Server. Waiting for IPCM Server "
                f"to be up...[{etc_api.host}:{etc_api.port}]"
            )
            logger.error(e)

    from proc_manager import sub_proc_thread

    sub_proc_thread.run_unit_server(
        port=settings.servers["this"].port,
        calc_interval=0.01,
    )
