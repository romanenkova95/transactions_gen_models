import torch


class BestClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int = None,
        rnn_units: int = 128,  # C_out
        classifier_units: int = 64,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = True,
        num_classes: int = 4,
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout2d(0.5)
        self.gru = self.backbone = torch.nn.GRU(
            input_size=input_size,
            hidden_size=rnn_units,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1

        self.linear1 = torch.nn.Linear(
            in_features=3 * d * rnn_units, out_features=classifier_units
        )
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(
            in_features=classifier_units, out_features=num_classes
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * d * rnn_units, classifier_units),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_units, num_classes),
        )

    def forward(self, input):
        embs = input.payload  # (N, L, C)
        dropout_embs = self.dropout(embs)
        states, h_n = self.backbone(dropout_embs)  # (N, L, 2 * C_out), (2, N, C_out)

        rnn_max_pool = states.max(dim=1)[0]  # (N, 2 * C_out)
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]  # (N, 2 * C_out)
        batch_size = h_n.shape[-2]
        h_n = h_n.view(batch_size, -1)  # (N, 2 * C_out)

        output = torch.cat([rnn_max_pool, rnn_avg_pool, h_n], dim=-1)  # (N, 6 * C_out)
        drop = torch.nn.functional.dropout(output, p=0.5)
        logit = self.mlp(drop)
        probas = torch.nn.functional.softmax(logit, dim=1)

        return probas
