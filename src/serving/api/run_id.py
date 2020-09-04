from fastapi import APIRouter, Request

from src.serving.schemas.schemas import ModelGetRunIdOutputs

router = APIRouter()


@router.get("", response_model=ModelGetRunIdOutputs, name="get-run-id")
async def get_run_id(request: Request):
    return {"run_id": request.app.state.run_id}
