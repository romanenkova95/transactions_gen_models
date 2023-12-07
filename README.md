# Macro & micro CoLES

## Data
Raw and preprocessed datasets are available [online](https://disk.yandex.ru/d/SzvwAOUhDo6dDg).

## Usage
Run the configuration, specified in config/master.yaml:
```python main.py```

Run in debug-mode (single epoch, single batch, single run, turn off wandb):
```FAST_DEV_RUN=1 python main.py```

Run tests, ~20-30m (v=print running test, b=show stdout&stderr only on error):
```python -m unittest -vb```
note, running COTIC tests requires uncommenting a line in test_backbones.py.
