import torch


class RNNClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        num_classes: int = 1,
        logsoftmax: bool = True,
    ):
        super().__init__()

        self.backbone = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1

        if logsoftmax:
            activation = torch.nn.LogSoftmax(dim=-1)
        else:
            activation = torch.nn.Softmax(dim=-1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d * num_layers * hidden_size, num_classes), activation
        )

    def forward(self, input):
        # output, (h_n, c_n) = self.backbone(input.payload)
        _, h_n = self.backbone(input.payload)
        batch_size = h_n.shape[-2]
        h_n = h_n.view(batch_size, -1)
        return self.linear(h_n)
