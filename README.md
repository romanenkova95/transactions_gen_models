# Macro & micro CoLES

## Data
Raw and preprocessed datasets are available [online](https://disk.yandex.ru/d/SzvwAOUhDo6dDg).

## Usage
Run a specific config from config/*yaml:
```python main.py --config-name <CONFIG_NAME>```

Run in debug-mode (single epoch, single batch, single run, turn off wandb):
```FAST_DEV_RUN=1 python main.py --config-name <CONFIG_NAME>```

Run tests, ~10m (v=print running test, b=show stdout&stderr only on error):
```python -m unittest -vb```
