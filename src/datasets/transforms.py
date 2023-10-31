from ptls.data_load.feature_dict import FeatureDict


class LastTokenTarget(FeatureDict):
    def __init__(self, target_seq_col: str):
        super().__init__()
        self.target_seq_col = target_seq_col

    def __call__(self, x: dict):
        seq_len = self.get_seq_len(x)
        target = x[self.target_seq_col][-1]
        new_x = self.seq_indexing(x, slice(seq_len - 1))
        return new_x, target