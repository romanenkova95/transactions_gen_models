from torch import nn


class AbsDecoder(nn.Module):
    """Abstract decoder class. Only requirement is to specify the output_size
    Properties:
        output_size (int): size of the channel dimension of the decoder output.
    """
    @property
    def output_size(self) -> int:
        raise NotImplementedError()
