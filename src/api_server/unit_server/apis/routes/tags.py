# api_server.apis.routes.tags

from typing import List

from api_server.unit_server.apis.examples import tags as tags_examples
from fastapi import APIRouter, Body, Response, status
from model.manager.common import ModelAgent

router = APIRouter()

model_agent = ModelAgent()


@router.post("/settings", summary="Add(or Update) tag setting (multiple)")
async def update_tag_setting(
    tag_settings: List[dict] = Body(None, example=tags_examples.settings),
):
    # ! Deprecated
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/settings_init", summary="Add tag setting (multiple) for INITILIZE model")
async def add_tag_setting(
    tag_settings: List[dict] = Body(None, example=tags_examples.settings),
):
    # ! Deprecated
    return Response(status_code=status.HTTP_204_NO_CONTENT)
