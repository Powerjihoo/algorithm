from api_server.unit_server.apis.routes import models, system, tags
from fastapi import APIRouter

router = APIRouter()
router.include_router(models.router, tags=["models"], prefix="/models")
router.include_router(tags.router, tags=["tags"], prefix="/tags")
router.include_router(system.router, tags=["system"], prefix="/system")
