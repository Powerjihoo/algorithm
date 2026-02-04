import asyncio
import multiprocessing as mp
import sys
import threading
import time
from typing import Union

from api_server import config as api_config
from api_server import exceptions as ex_api
from api_server.middleware.apilogger import RouterLoggingMiddleware
from api_server.middleware.timing import add_timing_middleware
from config import settings
from data_manager.kafka_consumer import StreamDataCollector
from data_manager.kafka_producer import MessageProducer
from dbinfo import exceptions as ex_db
from dbinfo.model_info import ModelInfoManager
from model.manager.aakr import AAKRManager
from model.manager.annoy import ANNOYManager
from model.manager.arima import ARIMAManager
from model.manager.common import ModelAgent, PredModelManagerBase
from model.manager.ensemble import EnsembleAgent
from model.manager.ica import ICAManager
from model.manager.pdmd import PDMDManager
from model.manager.vae import VAEManager
from utils import system as system_util
from utils.logger import logger, logging_time

from resources.constant import CONST

ensemble_agent = EnsembleAgent()
model_agent = ModelAgent()
aakr_manager = AAKRManager()
vae_manager = VAEManager()
# arima_manager = ARIMAManager()
# pdmd_manager = PDMDManager()
annoy_manager = ANNOYManager()
ica_manager = ICAManager()

model_agent.register(aakr_manager)
model_agent.register(vae_manager)
# model_agent.register(arima_manager)
# model_agent.register(pdmd_manager)
model_agent.register(annoy_manager)
model_agent.register(ica_manager)

IS_RUN_APP = False
PROC_NAME = mp.current_process().name


def collect_kafka_modelvalues(initial_sleep: int = 5) -> None:
    logger.info("Starting kafka consumer...")
    time.sleep(initial_sleep)

    consumer = StreamDataCollector(
        broker=settings.kafka.brokers,
        topic=settings.kafka.topic_model_values,
        model_agent=model_agent,
        ensemble_agent=ensemble_agent
    )

    while True:
        try:
            consumer.receive_message()
        except Exception as e:
            logger.error(e)


async def calc_algorithm(interval: int = 0.1) -> None:
    producer = MessageProducer(
        broker=settings.kafka.brokers,
        topic=settings.kafka.topic_pred_values,
    )
    await asyncio.sleep(10)

    async def __calc_algorithm():
        model_manager: Union[
            AAKRManager, VAEManager, ARIMAManager, PDMDManager, ANNOYManager, ICAManager
        ]
        for modeltype, model_manager in model_agent.get_observers():
            try:
                # * Calculate pred value
                if not model_manager:
                    continue

                model_manager.calc_preds()

                # * Send pred values to Kafka
                if (calc_cnt := model_manager.cnt_calc) == 0:
                    continue

                updated_preds = model_manager.create_calc_result_updated_only()
                # 카프카 전송전 버전정보 제거하기
                for model_data in updated_preds.model_data:
                    model_data.model_key = "_".join(model_data.model_key.split("_")[:-1])
                updated_preds_serialized = updated_preds.SerializeToString()
                if len(updated_preds_serialized) == 0:
                    continue
                producer.send_message(updated_preds_serialized)

                logger.trace(
                    f"[{PROC_NAME}] {modeltype} Predict result sent data: {len(updated_preds.model_data)} models"
                )

                logger.trace(
                    f"[{PROC_NAME}] {modeltype} model calculated: "
                    f"{calc_cnt}/{len(model_manager)}"
                )
            except Exception as e:
                logger.error(e)
        
        # Ensemble 모델 처리
        try:
            if ensemble_agent.get_model_count() > 0:
                # Ensemble 모델 예측 실행
                ensemble_agent.calc_preds()
                
                # Ensemble 예측 결과를 Kafka로 전송
                if (calc_cnt := ensemble_agent.cnt_calc) > 0:
                    updated_preds = ensemble_agent.create_calc_result_updated_only()
                    
                    # Ensemble 모델은 버전이 없으므로 model_key 변경 불필요
                    # (ensemble_ensemble_test 형태 그대로 전송)
                    
                    updated_preds_serialized = updated_preds.SerializeToString()
                    if len(updated_preds_serialized) > 0:
                        producer.send_message(updated_preds_serialized)
                        
                        logger.trace(
                            f"[{PROC_NAME}] ENSEMBLE Predict result sent data: {len(updated_preds.model_data)} models"
                        )
                        
                        logger.trace(
                            f"[{PROC_NAME}] ENSEMBLE model calculated: "
                            f"{calc_cnt}/{ensemble_agent.get_model_count()}"
                        )
        except Exception as e:
            logger.error(f"Error in ensemble calculation: {e}")

    while True:
        asyncio.create_task(__calc_algorithm())
        await asyncio.sleep(interval)


@logging_time
def initialize_models():
    logger.info(f"{'Initializing':12} | Initializing models in subprocess...")

    model_keys = ModelInfoManager.get_target_activated_model_keys()
    # model_keys, ensemble_model_keys = ModelInfoManager.get_target_activated_model_keys()

    filtered_model_info = {}
    ensemble_model_keys = []

    # for model_key in model_keys:
    #     try:
    #         _model_info = ModelInfoManager.get(model_key)
    #         filtered_model_info[model_key] = _model_info
    #     except Exception as e:
    #         logger.error(e)

    for model_key in model_keys:
        try:
            _model_info = ModelInfoManager.get(model_key)
            if _model_info.modeltype.upper() == "ENSEMBLE":
                ensemble_model_keys.append(model_key)
                logger.debug(f"Found ensemble model: {model_key}")
            else:
                filtered_model_info[model_key] = _model_info
        except Exception as e:
            logger.error(e)
    

    # * Initialize model
    for model_type, model_manager in model_agent.get_observers():
        model_manager: PredModelManagerBase
        filtered_model_info_by_type = {
            key: value
            for key, value in filtered_model_info.items()
            if value.modeltype == model_type
        }
        if not filtered_model_info_by_type:
            continue
        model_manager.define_models_initial(filtered_model_info_by_type)
        
        for versioned_key in filtered_model_info_by_type.keys():
            try:
                # 매니저에 실제로 올라간 모델만 alias 반영
                if not model_manager.check_model_running(versioned_key):
                    logger.warning(f"[init-alias] not running after init: {versioned_key}")
                    continue

                base_key = "_".join(versioned_key.split("_")[:-1])  # aakr_xxx_0001 -> aakr_xxx
                model_agent._alias_set(base_key, versioned_key)     # copy-on-write, 짧은 락
                logger.debug(f"[init-alias] {base_key} -> {versioned_key}")
            except Exception as e:
                logger.warning(f"[init-alias] failed for {versioned_key}: {e}")

    # ensemble 모델 api로는 호출 불가, redis에서 임의 호출 후 등록
        
    if not ensemble_model_keys:
        # logger.warning("[ENSEMBLE] No ensemble models in activated list, checking Redis...")
        try:
            # Redis에서 ensemble 모델 확인
            redis_ensemble_keys = []
            all_redis_keys = list(ModelInfoManager.redis_client.scan_iter(match="*"))
            
            for key in all_redis_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if 'ensemble' in key_str.lower():
                    try:
                        model_info = ModelInfoManager.get(key_str)
                        if model_info.modeltype.upper() == "ENSEMBLE" and model_info.runningstatus == "RUNNING":
                            redis_ensemble_keys.append(key_str)
                            logger.debug(f"[ENSEMBLE] Found RUNNING ensemble model in Redis: {key_str}")
                    except Exception as e:
                        logger.debug(f"Redis key {key_str} 확인 중 오류: {e}")
            
            ensemble_model_keys = redis_ensemble_keys
            
        except Exception as e:
            logger.error(f"[ENSEMBLE] Redis 확인 중 오류: {e}")

    # ensemble 모델만 별도 처리
    for model_key in ensemble_model_keys:
        try:
            ensemble_agent.add_model(model_key)
            logger.info(f"Ensemble model initialized: {model_key}")
        except Exception as e:
            logger.error(f"Failed to initialize ensemble model {model_key}: {e}")


    logger.info("Model initialized")


def thr_kafka_modelvalues_collector():
    logger.debug(f"{'Initializing':12} | Lastvalue collector is starting...")
    collect_kafka_modelvalues(initial_sleep=5)
    logger.error("Rawvalues collector is terminated")


def thr_pred_calculator(interval: int = 0.1):
    try:
        try:
            logger.debug(f"{'Initializing':12} | Predvalue Calculator is starting...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(calc_algorithm(interval))
            loop.close()
        except Exception as err:
            raise ex_db.InitializingFailError(
                "Can not initialize predvalue calculator thread, "
                "API Server is not running"
            ) from err
    except ex_db.InitializingFailError:
        system_util.kill_process()


def run_api_server(host: str = None, port: int = None) -> None:
    global IS_RUN_APP
    import uvicorn

    _host = api_config.get_api_ip() if host is None else host
    _port = system_util.get_available_port() if port is None else port

    from api_server.unit_server.apis.routes.api import router as unit_api_router

    api_router = unit_api_router

    app = api_config.get_application()
    if not settings.system.enable_swagger:
        to_remove = []
        for route in app.routes:
            if route.name in [
                "openapi",
                "custom_swagger_ui_html",
                "swagger_ui_redirect",
                "redoc_html",
            ]:
                to_remove.append(route)
        for route in to_remove:
            app.routes.remove(route)

    app.include_router(api_router)
    exclude_timing = "health"
    add_timing_middleware(
        app, record=logger.trace, prefix="app", exclude=exclude_timing
    )
    if settings.log.logging_router:
        app.add_middleware(RouterLoggingMiddleware, logger=logger)
    ex_api.add_exception_handlers(app)

    IS_RUN_APP = True
    uvicorn.run(
        app,
        host=_host,
        port=_port,
        log_config=api_config.get_uvicorn_logging_config(),
    )


def run_unit_server(
    port: int,
    calc_interval: int = 0.1,
) -> None:
    try:
        thread_api_server = threading.Thread(
            name="SubThread (API Server)",
            target=run_api_server,
            args=(None, port),
        )

        thread_rawvalues_collector = threading.Thread(
            name="SubThread (Rawvalue Collector)",
            target=thr_kafka_modelvalues_collector,
        )

        thread_pred_calculator = threading.Thread(
            name="Sub Thread (Predvalue Calculator",
            target=thr_pred_calculator,
            args=(calc_interval,),
        )


        thread_model_initializer = threading.Thread(
            name="SubThread (Model Initializer)",
            target=initialize_models,
        )

        thread_model_initializer.start()
        if settings.system.wait_for_model_initializing:
            thread_model_initializer.join()

        threads = [
            thread_api_server,
            thread_rawvalues_collector,
            thread_pred_calculator
        ]
        [thread.start() for thread in threads]

    except Exception:
        logger.exception(f"Failed to strat {CONST.PROGRAM_NAME}")
        sys.exit()
