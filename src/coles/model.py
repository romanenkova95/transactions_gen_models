"""CoLES model"""
from functools import partial

from omegaconf import DictConfig

import torch

from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule


class MyCoLES(CoLESModule):
    """Coles realization"""

    def __init__(
        self,
        data_conf: DictConfig,
        coles_conf: DictConfig,
    ):
        self.data_conf = data_conf
        self.coles_conf = coles_conf

        learning_params: DictConfig = coles_conf['learning_params']

        seq_encoder = self.make_rnn_encoder()
        optimizer_partial = partial(
            torch.optim.Adam,
            lr=learning_params['lr'],
            weight_decay=learning_params['weight_decay']
        )
        lr_scheduler_partial = partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=learning_params['step_size'],
            gamma=learning_params['gamma']
        )

        super().__init__(
            seq_encoder=seq_encoder,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial
        )

    def make_rnn_encoder(self) -> RnnSeqEncoder:
        return RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                embeddings_noise=self.coles_conf['embed_noise'],
                numeric_values={
                    self.data_conf['transaction_amt_column']: 'identity'
                },
                embeddings={
                    self.data_conf['mcc_column']: {
                        'in': self.coles_conf['mcc_vocab_size'],
                        'out': self.coles_conf['mcc_embed_size']
                    }
                },
            ),
            hidden_size=self.coles_conf['hidden_size'],
            type=self.coles_conf['type'],
        )
