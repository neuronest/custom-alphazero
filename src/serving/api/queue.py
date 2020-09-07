from fastapi import APIRouter, Body, Request

from src.serving.schemas.schemas import (
    ModelAppendQueueInputs,
    ModelRetrieveQueueOutputs,
    ModelGetQueueSizeOutputs,
)
from src.serving.schemas.example import AppendQueueExample

router = APIRouter()


@router.patch("/append", name="append-queue")
async def append_queue(
    request: Request,
    inputs: ModelAppendQueueInputs = Body(..., example=AppendQueueExample.inputs),
):
    data = inputs.dict()
    for state, policy, value in zip(
        data.pop("states"), data.pop("policies"), data.pop("values")
    ):
        request.app.state.queue.put((state, policy, value))


@router.put(
    "/retrieve", response_model=ModelRetrieveQueueOutputs, name="retrieve-queue"
)
async def retrieve_queue(request: Request):
    states, policies, values = [], [], []
    while not request.app.state.queue.empty():
        state, policy, value = request.app.state.queue.get()
        states.append(state)
        policies.append(policy)
        values.append(value)
    return {
        "states": states,
        "policies": policies,
        "values": values,
    }


@router.get("/size", response_model=ModelGetQueueSizeOutputs, name="get-queue-size")
async def get_queue_size(request: Request):
    return {"queue_size": request.app.state.queue.qsize()}
