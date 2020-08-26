import requests
import json
import uuid
import asyncio
import numpy as np
from typing import Tuple, List, Optional

from src.config import ConfigGeneral, ConfigServing, ConfigPath
from src.model.tensorflow.model import PolicyValueModel

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


class InferenceBatch:
    def __init__(self, model: PolicyValueModel, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.is_not_complete = asyncio.Event()
        self.is_complete = asyncio.Event()
        self.lock = asyncio.Lock()
        self.batch = {}
        self.predictions = {}
        self.is_not_complete.set()

    @staticmethod
    async def event_with_timeout(event, timeout):
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return event.is_set()

    async def store(self, uid: str, state: np.ndarray):
        await self.is_not_complete.wait()
        self.batch[uid] = state
        if len(self.batch) >= self.batch_size:
            self.is_not_complete.clear()
            self.is_complete.set()

    async def predict(self):
        await self.event_with_timeout(
            self.is_complete, ConfigServing.inference_timeout / 10
        )
        await self.lock.acquire()
        if not self.predictions:
            array_batch = np.stack([array for array in self.batch.values()])
            probabilities_batch, values_batch = self.model(array_batch)
            probabilities_batch, values_batch = (
                probabilities_batch.numpy().tolist(),
                values_batch.numpy().ravel().tolist(),
            )
            for uid, probabilities, values in zip(
                self.batch.keys(), probabilities_batch, values_batch
            ):
                self.predictions[uid] = {
                    "probabilities": probabilities,
                    "values": values,
                }
        self.lock.release()

    def get_prediction(self, uid: str):
        self.batch.pop(uid, None)
        return self.predictions.pop(uid, None)

    def reset(self):
        if not len(self.predictions):
            self.is_not_complete.set()
            self.is_complete.clear()


def init_model(path: Optional[str] = None) -> PolicyValueModel:
    all_possible_moves = get_all_possible_moves()
    action_space = len(all_possible_moves)
    input_dim = Board().full_state.shape
    model = PolicyValueModel(input_dim=input_dim, action_space=action_space)
    if path is not None:
        model.load_with_meta(path)
    return model


# class intended to reproduce the same global behavior as request.app.state on fastapi server
class App:
    class State:
        number_samples = 0
        iteration = 0
        model = init_model()


def infer_sample(state: np.ndarray, concurrency: bool) -> Tuple[np.ndarray, float]:
    headers = {"content-type": "application/octet-stream"}
    data = {
        "uid": str(uuid.uuid4()),
        "state": state.tolist(),
        "concurrency": concurrency,
    }
    try:
        response = requests.post(
            url=ConfigServing.serving_address + ConfigPath.inference_path,
            data=json.dumps(data),
            headers=headers,
            timeout=ConfigServing.inference_timeout,
        )
    except requests.Timeout:
        print(
            "Concurrency inference has timed out, falling into regular sample inference..."
        )
        data["concurrency"] = False
        response = requests.post(
            url=ConfigServing.serving_address + ConfigPath.inference_path,
            data=json.dumps(data),
            headers=headers,
        )
    try:
        response_content = json.loads(response.content)
    except json.decoder.JSONDecodeError:
        print(
            f"Internal inference routine error:\nuid:{data['uid']}\nstate:{data['state']}"
        )
        response_content = {
            "probabilities": [0.0] * len(get_all_possible_moves()),
            "value": 0.0,
        }
    return np.asarray(response_content["probabilities"]), response_content["value"]


def train_run_samples_post(
    run_id: str, states: np.ndarray, labels: List[np.ndarray]
) -> Tuple[float, bool, int]:
    headers = {"content-type": "application/octet-stream"}
    policies, values = labels
    data = {
        "run_id": run_id,
        "states": states.tolist(),
        "policies": policies.tolist(),
        "values": values.tolist(),
    }
    response = requests.post(
        url=ConfigServing.serving_address + ConfigPath.training_path,
        data=json.dumps(data),
        headers=headers,
    )
    response_content = json.loads(response.content)
    return (
        response_content["loss"],
        response_content["updated"],
        response_content["iteration"],
    )
