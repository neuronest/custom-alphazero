# Custom AlphaZero

Launch web server:
```bash
uvicorn --port 5000 --host 0.0.0.0 src.serving.api.main:app
```

Launch self-play processes:
```bash
python -m src.self_play
```

Launch train / evaluation process:
```bash
python -m src.train
```

Launch tensorboard server:
```bash
tensorboard --port 6006 --logdir results/
```

