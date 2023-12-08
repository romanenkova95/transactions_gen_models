"""File with the TransformerSeqEncoder class which accepts a hidden_size argument for compatibility."""
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptls.nn import TransformerSeqEncoder as TransformerSeqEncoder_
from ptls.nn.trx_encoder import TrxEncoder


class TransformerSeqEncoder(TransformerSeqEncoder_):
    """Adds a hidden_size parameter to TransformerSeqEncoder for consistency.
    
    Done through setting the linear_projection param of TrxEncoder.
    """

    def __init__(
        self,
        trx_encoder: DictConfig,
        hidden_size=1024,
        is_reduce_sequence=True,
        **seq_encoder_params,
    ):
        """Initialize module internal state.

        Args:
        ----
            trx_encoder (DictConfig): the trx encoder to use.
            hidden_size (int, optional): 
                the output size, i.e. the linear_projection_size of the trx encoder. Defaults to 1024.
            is_reduce_sequence (bool, optional):
                whether to reduce the output sequence. Defaults to True.
            **seq_encoder_params: passed to TransformerSeqEncoder ptls class.
        """
        trx_encoder_instance: TrxEncoder = instantiate(
            trx_encoder, linear_projection_size=hidden_size
        )

        super().__init__(
            trx_encoder=trx_encoder_instance,
            input_size=None,
            is_reduce_sequence=is_reduce_sequence,
            **seq_encoder_params,
        )
