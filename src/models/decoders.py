import torch
import torch.nn as nn
from torch import Tensor

class Decoder(nn.Module):
    """
    Abstract class for decoders, responsible for converting the latent representation
    (output of the RSSM) back into a reconstruction of the original input.

    Args:
        cfg: Configuration object.
        model (str): Type of decoder model ("cnn" or "gnn"). Default: "cnn".
        TODO: add gnn decoder.

    Methods:
        forward(feature, states): Decodes the latent features and returns the loss,
                                  image loss, and decoded image.
    """
    def __init__(
        self,
        cfg,
        model: str = "cnn",
    ):
        assert model  in ["cnn", "gnn"], f"Unexpected encoder model type `{model}`."
        
        super().__init__()
        
        feature_dim = cfg.wm.deter_dim + \
            cfg.wm.stoch_dim * cfg.wm.stoch_rank
        
        if model == "cnn":
            self.model = ConvDecoder(feature_dim)
        elif model == "gnn":
            raise NotImplementedError()
            
        self.image_weight = cfg.wm.image_weight
            
    def forward(self, feature, states):
        loss_image, decoded_img = self.model(feature, states)
        loss = self.image_weight * loss_image
        return loss, loss_image, decoded_img
            
            
class ConvDecoder(nn.Module):
    """
    Decoder implementation using a transposed convolutional neural network (CNN).

    Args:
        feature_dim (int): Dimension of the latent features.
        out_channel (int): Number of output channels for the decoded image. Default: 4.
        mlp_layers (int): Number of MLP layers to use. Default: 0.
        activation (str): Activation function to use ("elu" or "relu"). Default: "elu".

    Methods:
        forward(features, states): Decodes the latent features and returns the loss
    """
    def __init__(
        self,
        feature_dim: int,
        out_channel: int = 4,
        mlp_layers: int = 0,
        activation: str = "elu",
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        kernels = (5, 5, 6, 6)
        stride = 2
        cnn_depth = 32
        activation = nn.ELU if activation == "elu" else nn.ReLU
        if mlp_layers == 0:
            layers = [
                nn.Linear(feature_dim, cnn_depth * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = cnn_depth * 32
            norm = nn.LayerNorm
            layers = [
                nn.Linear(feature_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()]
                
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Flatten(0, 1),
            *layers,
            nn.Unflatten(-1, (d * 32, 1, 1)),  # type: ignore
            nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(d * 2, d, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(d, d, kernels[3], stride),
            activation(),
            nn.ConvTranspose2d(d, out_channel, kernel_size=2, stride=2),
        )
        
    def forward(self, features, states):
        decoded_img = self.model(features)
        # assert decoded_img.shape == states.shape, f"{decoded_img.shape} {states.shape}"
        decoded_img = torch.reshape(
            decoded_img, states.shape)
        loss = 0.5 * torch.square(decoded_img -
                                  states).sum(dim=[-1, -2, -3])  # MSE
        B, S = loss.shape
        loss = torch.reshape(loss, (S, B))
        
        return loss, decoded_img
        