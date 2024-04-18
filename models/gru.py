import torch.nn as nn


class GRU(nn.Module):
    """Gate Recurrent Unit"""
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int, bidirectional:bool):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional)

        self.layer_norm = nn.LayerNorm(hidden_size)

        if self.bidirectional:
          self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
          self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.layer_norm(x)
        out = self.fc(x)

        return out