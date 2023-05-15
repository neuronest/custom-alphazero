# Custom AlphaZero

Launch web server:
```bash
uvicorn --port 5000 --host 0.0.0.0 custom_alphazero.serving.api.main:app
```

Launch self-play processes:
```bash
python -m custom_alphazero.self_play
```

Launch train / evaluation process:
```bash
python -m custom_alphazero.train
```

Launch tensorboard server:
```bash
tensorboard --port 6006 --logdir results/
```

