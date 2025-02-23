import torch
import torch.nn as nn


class GRUCellStack(nn.Module):
    """
    A stack of GRU cells used within the RSSM to maintain the deterministic state.

    Args:
        input_size (int): Size of the input to the GRU stack.
        hidden_size (int): Total hidden size of the GRU stack.
        n_layers (int): Number of GRU layers in the stack.

    Methods:
        forward(input, state): Processes the input through the GRU stack and returns
                               the concatenated output states.
    """

    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()

        self.n_layers = n_layers
        layer_size = hidden_size // n_layers
        layers = [nn.GRUCell(input_size, layer_size)]
        layers.extend([nn.GRUCell(layer_size, layer_size)
                      for _ in range(n_layers-1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input, state):
        input_states = state.chunk(self.n_layers, -1)
        output_states = []
        x = input
        for i in range(self.n_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)