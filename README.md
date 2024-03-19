# Universal representations for financial transactional data: embracing local, global, and external contexts

## Abstract
Effective processing of financial transactions is essential for banking data analysis. However, in this domain, most methods focus on specialized solutions to stand-alone problems instead of constructing universal representations suitable for many problems, including those with a limited amount of labeled data. Adopting a more general approach, this paper presents a representation learning framework that addresses diverse business challenges. We review well-established contrastive approaches to transaction sequence processing and suggest novel generative models that account for data specifics. On top of them, a separate new method integrates external information into a client’s representation, leveraging insights from other customers’ actions. Finally, we suggest a reliable validation methodology providing insights about representation quality from three different standpoints: globally, concerning the entire transaction history; locally, reflecting the client’s current state; and dynamically, capturing representation evolution over time. Our generative approach demonstrates superior performance in local tasks, with an increase in ROC-AUC of up to 14% for the next MCC prediction task and up to 46% for downstream tasks from existing contrastive baselines over multiple financial transaction datasets. Incorporating external information further improves the scores by an additional 20%.

## Data
Open transavtopnal datasets are available [online](https://disk.yandex.ru/d/--KIPMEJ-cB4MA). Folder `raw/` contains the datasets in their initial formats, and the final parquet-files we use are stored in the `preprocessed/` folder. To reproduce the experiments, download the `preprocessed/` folder and put it into the `data/` directory.

Note that you can experiment with custom datasets. To do so, you should prepare a dataframe with transaction records containing columns `'user_id'`, `'amount'`, `'timestamp'`, `'mcc_code'`, `'global_target'`, and, optionally, `'local_target'` and save it as a .parquet file.

## Environment
To reproduce the experiments from the paper, please build and run the docker container, using files from `docker_scripts/`. In particulae, you should
  1) Clone this repository locally;
  2) Fill in the `credentials` file with your user id;
  3) Run the following commands in the terminal:
     ```console
     foo@bar:~/transactions_gen_models/docker_scripts$ bash build_image
     foo@bar:~/transactions_gen_models/docker_scripts$ bash launch_container
     ```

## Usage
To run a full experiment specified in `config/master.yaml`, including model training and validation, run
```console
foo@bar:~/transactions_gen_models$ python main.py
```

Note that you can run the experiment in debug-mode (single epoch, single batch, single run, turn off logging) with
```console
foo@bar:~/transactions_gen_models$ FAST_DEV_RUN=1 python main.py
```

## Configs structure
With use a hydra-based experiment configuration structure. To assemble the set-up you need, you can easily adjust the `config/master.yaml`.
* `- preprocessing:` choose the dataset and the preprocessing you need from the `config/preprocessing/` directory. Note that TPP models (COTIC, NHP, and A-NHP) require time normalization and, thus, have their own configs `tpp_*_nodup.yaml`, while all the other models use configs `*.yaml`.
* `- backbone:` choose the backbone representation learning model from the `config/backbone/` directory. In the paper, we experiment with the following models: CoLES, AE, AR, MLM, TS2Vec, COTIC, NHP, and A-NHP. However, one could add custom models and the corresponding configuration files to the pipeline.
* `- validation:` specify list of validation procedures for the model. By default, we use `['global_validation', 'event_type', 'local_target']` for the Churn, Default anf HSBC datasets. As mentioned in the paper, the Age dataset does not have binary local targets, so, in this case, we only validate the models with `['global_validation', 'event_type']`. The parameters of the validation models are specified in the `config/validation/` folder.
* `- logger:` select wandb, comet or any other logger you prefer.

## Models with external context
In the paper, we propose a representation learning model for transactions that accounts for the external context information of other users. It uses the pretrained CoLES model as a backbone. Please, look to the `notebooks/` directory, to reproduce the experiments with these models. The corresponding configuration files are specified in `config/pooling_*_validation.yaml` files.

## Results summary

## Citation
If you use this repository, please feel free to star it and cite our paper:

TODO.
