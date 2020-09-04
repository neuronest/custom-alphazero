import requests
import json
import uuid
import numpy as np
from typing import Tuple, Optional

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


def get_run_id() -> Optional[str]:
    response = requests.get(url=ConfigServing.serving_address + ConfigPath.run_id_path,)
    try:
        response_content = json.loads(response.content)
    except json.decoder.JSONDecodeError:
        return None
    return response_content.get("run_id")


def append_queue(states: np.ndarray, policies: np.ndarray, values: np.ndarray):
    headers = {"content-type": "application/octet-stream"}
    data = {
        "states": states.tolist(),
        "policies": policies.tolist(),
        "values": values.tolist(),
    }
    requests.patch(
        url=ConfigServing.serving_address + ConfigPath.append_queue_path,
        data=json.dumps(data),
        headers=headers,
    )


def retrieve_queue() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    headers = {"content-type": "application/octet-stream"}
    response = requests.put(
        url=ConfigServing.serving_address + ConfigPath.retrieve_queue_path,
        data={},
        headers=headers,
    )
    try:
        response_content = json.loads(response.content)
    except json.decoder.JSONDecodeError:
        return None
    states, policies, values = (
        np.asarray(response_content.get("states")),
        np.asarray(response_content.get("policies")),
        np.asarray(response_content.get("values")),
    )
    return states, policies, values


def update_best_model():
    headers = {"content-type": "application/octet-stream"}
    requests.put(
        url=ConfigServing.serving_address + ConfigPath.update_best_model_path,
        data={},
        headers=headers,
    )


def get_queue_size() -> Optional[int]:
    response = requests.get(
        url=ConfigServing.serving_address + ConfigPath.size_queue_path,
    )
    try:
        response_content = json.loads(response.content)
    except json.decoder.JSONDecodeError:
        return None
    return response_content.get("queue_size")
