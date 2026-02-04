from fastapi import APIRouter, Response, status

router = APIRouter()


@router.get(
    "/health_status",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Check API Server is running",
)
async def check_health_status():
    return Response(status_code=status.HTTP_204_NO_CONTENT)
