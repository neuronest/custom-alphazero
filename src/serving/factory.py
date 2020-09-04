import requests
import json
import uuid
import numpy as np
from typing import Tuple, List

from src.config import ConfigGeneral, ConfigServing, ConfigPath


if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


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
