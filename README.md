# Universal representations for financial transactional data: embracing local, global, and external contexts

## Abstract
Effective processing of financial transactions is essential for banking data analysis. However, in this domain, most methods focus on specialized solutions to stand-alone problems instead of constructing universal representations suitable for many problems, including those with a limited amount of labeled data. Adopting a more general approach, this paper presents a representation learning framework that addresses diverse business challenges. We review well-established contrastive approaches to transaction sequence processing and suggest novel generative models that account for data specifics. On top of them, a separate new method integrates external information into a client’s representation, leveraging insights from other customers’ actions. Finally, we suggest a reliable validation methodology providing insights about representation quality from three different standpoints: globally, concerning the entire transaction history; locally, reflecting the client’s current state; and dynamically, capturing representation evolution over time. Our generative approach demonstrates superior performance in local tasks, with an increase in ROC-AUC of up to 14% for the next MCC prediction task and up to 46% for downstream tasks from existing contrastive baselines over multiple financial transaction datasets. Incorporating external information further improves the scores by an additional 20%.

## Data
Open transavtopnal datasets are available [online](https://disk.yandex.ru/d/--KIPMEJ-cB4MA). Folder `raw/` contains the datasets in their initial formats, and the final parquet-files we use are stored in the `preprocessed/` folder.

## Usage
Run the configuration, specified in config/master.yaml:
```python main.py```

Run in debug-mode (single epoch, single batch, single run, turn off wandb):
```FAST_DEV_RUN=1 python main.py```

Run tests, ~20-30m (v=print running test, b=show stdout&stderr only on error):
```python -m unittest -vb```
note, running COTIC tests requires uncommenting a line in test_backbones.py.
