import torch
import torch.nn as nn
from torch import Tensor

class Encoder(nn.Module):
    def __init__(
        self,
        cfg,
        model: str = "cnn",
    ):
        assert model in ["cnn", "gnn"], f"Unexpected encoder model type `{model}`."
        
        super().__init__()
        
        if model == "cnn":
            self.model = ConvEncoder() # cfg.wm.embed_dim)
        elif model == "gnn":
            raise NotImplementedError()
        
    def forward(self, states: Tensor) -> Tensor:
        return self.model(states)
        
              
class ConvEncoder(nn.Module):
    def __init__(
        self,
        kernel: int = 4,
        stride: int = 2,
        activation: str = "elu"
    ):
        super().__init__()
        
        activation = nn.ELU if activation == "elu" else nn.ReLU
        cnn_depth = 32
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=cnn_depth * 1, kernel_size=kernel, stride=stride),
            activation(),
            
            nn.Conv2d(in_channels=cnn_depth * 1, out_channels=cnn_depth * 2, kernel_size=kernel, stride=stride),
            activation(),
            
            nn.Conv2d(in_channels=cnn_depth * 2, out_channels=cnn_depth * 4, kernel_size=kernel, stride=stride),
            activation(),
            
            nn.Conv2d(in_channels=cnn_depth * 4, out_channels=cnn_depth * 8, kernel_size=kernel, stride=stride),
            activation(),
            
            nn.Flatten(),
        )
        
    def forward(self, states: Tensor) -> Tensor:
        """ Forward Proprogation

        Args:
            states (Tensor): A batch of a sequence of observation, with shape of 
                            (B, T, C, H, W).

        Returns:
            Tensor: Embedded observation, with shape of (B, T, E)
        """
        assert states.dim() == 5, f"{states.dim()}"
        B, T = states.shape[0], states.shape[1]
    
        states = torch.reshape(states, (-1, *states.shape[2:]))
        embed = self.model(states)
        
        embed = torch.reshape(embed, (T, B, -1))
        return embed