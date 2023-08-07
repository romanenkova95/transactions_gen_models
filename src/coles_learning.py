import os
import pickle

from omegaconf import DictConfig

from ptls.preprocessing import PandasDataPreprocessor
from ptls.frames import PtlsDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from sklearn.model_selection import train_test_split

from src.coles import MyCoLES, MyColesDataset
from src.utils.logging_utils import get_logger
from src.temp import preprocessing

logger = get_logger(name=__name__)



def train_coles(
    cfg_preprop: DictConfig, cfg_model: DictConfig, api_token: str
) -> None:
    # My method for data preprocessing
    _, preproc_df = preprocessing(cfg_preprop, True)

    preproc_df.drop(columns=['sample_label', 'target'], inplace=True)

    dir_path    : str = cfg_preprop['dir_path']
    user_column : str = cfg_preprop['user_column']
    dttm_column : str = cfg_preprop['transaction_dttm_column']
    mcc_column  : str = cfg_preprop['mcc_column']
    amt_column  : str = cfg_preprop['transaction_amt_column']


    dir_coles = os.path.join(dir_path, 'coles')

    if not os.path.exists(dir_coles):
        logger.warning('Coles folder does not exist. Creating...')
        os.mkdir(dir_coles)

    if not os.path.exists(os.path.join(dir_coles, 'preprocessor.p')):
        preprocessor = PandasDataPreprocessor(
            user_column,
            dttm_column,
            cols_category=[mcc_column, 'is_income'],
            cols_numerical=[amt_column],
            return_records=True
        )
        logger.info('Fitting CoLES preprocessor')
        dataset = preprocessor.fit_transform(preproc_df)
        with open(os.path.join(dir_coles, 'preprocessor.p'), 'wb') as f:
            pickle.dump(preprocessor, f)
    else:
        with open(os.path.join(dir_coles, 'preprocessor.p'), 'rb') as f:
            dataset = pickle.load(f).transform(preproc_df)

    for i in range(cfg_model['num_iters']):

        train, val = train_test_split(dataset, test_size=.2)

        training_params: DictConfig = cfg_model['learning_params']

        datamodule = PtlsDataModule(
            train_data=MyColesDataset(train, cfg_model),
            train_batch_size=training_params['train_batch_size'],
            train_num_workers=training_params['train_num_workers'],
            valid_data=MyColesDataset(val, cfg_model),
            valid_batch_size=training_params['val_batch_size'],
            valid_num_workers=training_params['val_num_workers']
        )

        model = MyCoLES(cfg_preprop, cfg_model)
        model_name = f'coles_hidden_size_{cfg_model["hidden_size"]}_{i}'

        model_checkpoint = ModelCheckpoint(
            monitor=model.metric_name,
            mode='max',
            dirpath=os.path.join('logs', 'checkpoints', 'coles'),
            filename=model_name
        )

        early_stopping = EarlyStopping(
            model.metric_name,
            training_params['early_stopping']['min_delta'],
            training_params['early_stopping']['patience'],
            True,
            'max'
        )

        comet_logger = CometLogger(
            api_token,
            project_name='coles_diploma',
            experiment_name=model_name
        )

        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=training_params['epochs'],
            log_every_n_steps=20,
            logger=comet_logger,
            callbacks=[model_checkpoint, early_stopping]
        )

        trainer.fit(model, datamodule)
