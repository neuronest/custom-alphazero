from fastapi import APIRouter, Request

from custom_alphazero.utils import best_saved_model

router = APIRouter()


@router.put("/update", name="update-best-model")
async def update_best_model(request: Request):
    request.app.state.best_model = best_saved_model(request.app.state.run_id)
