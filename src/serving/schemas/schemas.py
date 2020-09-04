from typing import List
from pydantic import BaseModel, Field


class ModelInferenceInputs(BaseModel):
    uid: str = Field(..., title="Unique request token")
    state: List[List[List[float]]] = Field(..., title="Board state to predict")
    concurrency: bool = Field(..., title="Enable concurrency inference")


class ModelInferenceOutputs(BaseModel):
    probabilities: List[float] = Field(..., title="Move probabilities")
    value: float = Field(..., title="State value")


class ModelGetRunIdOutputs(BaseModel):
    run_id: str = Field(..., title="Identifier of the run, usually a timestamp")


class ModelRetrieveQueueOutputs(BaseModel):
    states: List[List[List[List[float]]]] = Field(..., title="Board state samples")
    policies: List[List[float]] = Field(..., title="Policies samples")
    values: List[float] = Field(..., title="Values samples")


class ModelAppendQueueInputs(BaseModel):
    states: List[List[List[List[float]]]] = Field(..., title="Board state samples")
    policies: List[List[float]] = Field(..., title="Policies samples")
    values: List[float] = Field(..., title="Values samples")


class ModelGetQueueSizeOutputs(BaseModel):
    queue_size: int = Field(..., title="Queue size")
