from ptls.nn.seq_encoder.containers import SeqEncoderContainer


class CustomSeqEncoderContainer(SeqEncoderContainer):
    @property
    def hidden_size(self):
        return self.embedding_size
