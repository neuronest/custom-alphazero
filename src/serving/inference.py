import numpy as np
from fastapi import APIRouter, Body, Request

from src.serving.schema import ModelInferenceInputs, ModelInferenceOutputs
from src.serving.example import InferenceExample

router = APIRouter()


@router.post("", response_model=ModelInferenceOutputs, name="inference")
async def post(
    request: Request,
    inputs: ModelInferenceInputs = Body(..., example=InferenceExample.inputs),
):
    best_model = request.app.state.best_model
    inference_batch = request.app.state.inference_batch
    inference_batch.update_model(best_model)
    data = inputs.dict()
    uid = data.pop("uid")
    state = np.asarray(data.pop("state"))
    concurrency = data.pop("concurrency")
    if concurrency:
        await inference_batch.store(uid, state)
        await inference_batch.predict()
        prediction = inference_batch.get_prediction(uid)
        inference_batch.reset()
    else:
        probabilities, value = best_model(np.expand_dims(state, axis=0))
        probabilities, value = (
            probabilities.numpy().ravel().tolist(),
            value.numpy().item(),
        )
        prediction = {"probabilities": probabilities, "values": value}
    return {"probabilities": prediction["probabilities"], "value": prediction["values"]}
