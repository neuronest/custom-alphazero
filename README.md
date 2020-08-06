# Custom AlphaZero

Launch training / inference / evaluation server:
```bash
uvicorn --port 5000 --host 0.0.0.0 src.serving.api:app
```

Launch MCTS search:
```bash
python -m src.run
```

Launch tensorboard server:
```bash
cd src
tensorboard --port 6006 --logdir model/logs/
```

