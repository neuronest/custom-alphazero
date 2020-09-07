import asyncio
import numpy as np

from src.config import ConfigServing
from src.model.tensorflow.model import PolicyValueModel


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

    def update_model(self, model: PolicyValueModel):
        self.model = model
