from omegaconf import DictConfig
from ptls.nn import TransformerSeqEncoder as TransformerSeqEncoder_
from hydra.utils import instantiate
from ptls.nn.trx_encoder import TrxEncoder


class TransformerSeqEncoder(TransformerSeqEncoder_):
    """This class adds a hidden_size parameter to TransformerSeqEncoder for consistency.
    Done through setting the linear_projection param of TrxEncoder.
    """

    def __init__(
        self,
        trx_encoder: DictConfig,
        hidden_size=1024,
        is_reduce_sequence=True,
        **seq_encoder_params,
    ):
        trx_encoder_instance: TrxEncoder = instantiate(
            trx_encoder, linear_projection_size=hidden_size
        )

        super().__init__(
            trx_encoder=trx_encoder_instance,
            input_size=None,
            is_reduce_sequence=is_reduce_sequence,
            **seq_encoder_params,
        )
