# Learning Transactions Representations for Information Management in Banks: Mastering Local, Global, and External Knowledge

The work is [published](https://www.sciencedirect.com/science/article/pii/S2667096825000059) in the International Journal of Information Management Data Insights.

## Abstract
In today's world, banks use artificial intelligence to optimize diverse business processes, aiming to improve customer experience. Most of the customer-related tasks can be categorized into two groups: (1) local ones, which focus on a client’s current state, such as transaction forecasting, and (2) global ones, which consider the general customer behaviour, e.g., predicting successful loan repayment. Unfortunately, maintaining separate models for each task is costly. Therefore, to better facilitate information management, we compared eight state-of-the-art unsupervised methods on 11 tasks in search for a one-size-fits-all solution. Contrastive self-supervised learning methods were demonstrated to excel at global problems, while generative techniques were superior at local tasks. We also introduced a novel approach, which enriches the client’s representation by incorporating external information gathered from other clients. Our method outperforms classical models, boosting accuracy by up to 20%.

## Data
Open transactional datasets are available [online](https://disk.yandex.ru/d/--KIPMEJ-cB4MA). Folder `raw/` contains the datasets in their initial formats, and the final .parquet files we use are stored in the `preprocessed/` folder. To reproduce the experiments, download the `preprocessed/` folder and put it into the `data/` directory.

Note that you can experiment with custom datasets. To do so, you should prepare a dataframe with transaction records containing columns `'user_id'`, `'amount'`, `'timestamp'`, `'mcc_code'`, `'global_target'`, and, optionally, `'local_target'` and save it as a .parquet file.

## Environment
To reproduce the experiments from the paper, please build and run the docker container using files from `docker_scripts/`. In particular, you should:
  1) Clone this repository locally;
  2) Fill in the `credentials` file with your user ID;
  3) Run the following commands in the terminal:
     ```console
     foo@bar:~/transactions_gen_models/docker_scripts$ bash build_image
     foo@bar:~/transactions_gen_models/docker_scripts$ bash launch_container
     ```

## Usage
To conduct a full experiment specified in `config/master.yaml`, including model training and validation, run
```console
foo@bar:~/transactions_gen_models$ python main.py
```

Note that you can run the experiment in debug mode (single epoch, single batch, single run, turn off logging) with
```console
foo@bar:~/transactions_gen_models$ FAST_DEV_RUN=1 python main.py
```

## Configs structure
We use a hydra-based experiment configuration structure. To assemble the set-up you need, you can easily adjust the `config/master.yaml`.
* `- preprocessing:` choose the dataset and the preprocessing you need from the `config/preprocessing/` directory. Note that TPP models (COTIC, NHP, and A-NHP) require time normalization and, thus, have their own configs, `tpp_*_nodup.yaml`, while all the other models use configs `*.yaml`.
* `- backbone:` choose the backbone representation learning model from the `config/backbone/` directory. In the paper, we experiment with the following models: CoLES, AE, AR, MLM, TS2Vec, COTIC, NHP, and A-NHP. However, one could add custom models and the corresponding configuration files to the pipeline.
* `- validation:` specify list of validation procedures for the model. By default, we use `['global_validation', 'event_type', 'local_target']` for the Churn, Default anf HSBC datasets. As mentioned in the paper, the Age dataset does not have binary local targets, so, in this case, we only validate the models with `['global_validation', 'event_type']`. The parameters of the validation models are specified in the `config/validation/` folder.
* `- logger:` select wandb, comet, tensorboard, or any other logger you prefer.
* Set `pretrain: true` if you want to train the representation learning model from scratch, or `pretrain: false` if you would like to load the existing encoder's weights and only validate it.
* Select `seed`.

## Models with external context
In the paper, we propose a representation learning model for transactions that accounts for the external context information of other users. It uses the pretrained CoLES model as a backbone. Please look to the `notebooks/` directory to reproduce the experiments with these models. The `config/pooling_*_validation.yaml` files specify the corresponding configuration files.

## Results summary
We compare representation learning approaches of 3 classes: contrastive SSL (CoLES, TS2Vec), generative SSL (AE, MLM, AR), and Temporal Point processing models (COTIC, NHP, A-NHP). Note that these generative SSL models have been adapted to the transactional data domain for the first time. The approaches are validated in terms of their embeddings' global and local properties. See the paper for details on validation procedures and their motivation. The key findings are highlighted here.

#### General benchmarking
The Figure below shows the trade-off between global validation (x-axis) and local validation (y-axis) results. Thus, the upper and righter the dot is, the better the model it represents. It is seen that ***generative SSL approaches*** (AE, MLM, and AR) come on top in terms of the local patterns while staying competitive with the contrastive methods regarding the global embedding features. Consequently, we recommend this class of models as the optimal choice for representation learning for transactional data.

<p align="center">
  <img width="750" src="https://github.com/romanenkova95/transactions_gen_models/assets/44891804/07c09b0f-6956-45c1-a91c-406aa67589e9">
</p>

#### Models with external information
The similar Figure below presents how existing representation learning approaches (e.g., CoLES) can be improved by using ***external context information*** from all the users in the dataset.

<p align="center">
  <img src="https://github.com/romanenkova95/transactions_gen_models/assets/60795929/2a899520-dac3-4165-a6b9-eeab31c389da" width="750">
</p>

## Citation
If you find this repository useful, please feel free to star it and cite our paper.

```
@article{bazarova2025learning,
  title = {Learning transactions representations for information management in banks: Mastering local, global, and external knowledge},
  journal = {International Journal of Information Management Data Insights},
  volume = {5},
  number = {1},
  pages = {100323},
  year = {2025},
  issn = {2667-0968},
  doi = {https://doi.org/10.1016/j.jjimei.2025.100323},
  url = {https://www.sciencedirect.com/science/article/pii/S2667096825000059},
  author = {Alexandra Bazarova and Maria Kovaleva and Ilya Kuleshov and Evgenia Romanenkova and Alexander Stepikin and Aleksandr Yugay and Dzhambulat Mollaev and Ivan Kireev and Andrey Savchenko and Alexey Zaytsev}
}
```
