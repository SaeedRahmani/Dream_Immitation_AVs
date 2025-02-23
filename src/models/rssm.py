import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor
from omegaconf import DictConfig

from .gru import GRUCellStack


class RSSMCore(nn.Module):
    """
    Manages the recurrent process of the RSSM.

    Args:
        cfg: Configuration object.

    Methods:
        init_state(batch_size): Initializes the deterministic and stochastic states.
        forward(actions, resets, in_state, embeds): Processes the input sequence and
                                                    returns the priors, posteriors,
                                                    samples, features, states, and
                                                    detached final states.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        
        self.cell = RSSMCell(
            embed_dim=cfg.wm.embed_dim, 
            action_dim=cfg.wm.action_dim, 
            deter_dim=cfg.wm.deter_dim, 
            stoch_dim=cfg.wm.stoch_dim, 
            stoch_rank=cfg.wm.stoch_rank, 
            hidden_dim=cfg.wm.hidden_dim,
        )
        
    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)
          
    def forward(
        self, 
        actions: Tensor, 
        resets: Tensor, 
        in_state: Tensor,
        embeds: Tensor = None,
    ) -> Tensor:
        
        T = embeds.shape[0]
        (h, z) = in_state
        
        posts, states_h, samples = [], [], []
        
        for i in range(T):
            post, (h, z) = self.cell.forward(
                action=actions[i],
                reset=None, 
                in_state=(h, z),
                embed=embeds[i])
            posts.append(post)
            states_h.append(h)
            samples.append(z)
        
        posts = torch.stack(posts)
        states_h = torch.stack(states_h)
        samples = torch.stack(samples)
        priors = self.cell.batch_prior(states_h)
        features = self.to_feature(states_h, samples)

        states = (states_h, samples)

        return (
            priors,
            posts,
            samples,
            features,
            states,
            (h.detach(), z.detach())
        )

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)
    

class RSSMCell(nn.Module):
    """
    Performs the computations within a single timestep of the RSSM.

    Args:
        embed_dim (int): Dimension of the embedded observations.
        action_dim (int): Dimension of the actions.
        deter_dim (int): Dimension of the deterministic state.
        stoch_dim (int): Dimension of the stochastic state.
        stoch_rank (int): Rank of the stochastic state.
        hidden_dim (int): Dimension of the hidden layers.
        gru_layers (int): Number of GRU layers in the stack. Default: 2.

    Methods:
        init_state(batch_size): Initializes the deterministic and stochastic states.
        forward(action, reset, in_state, embed): Updates the states and samples from
                                                 the latent distribution.
    """
    
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int,
        stoch_dim: int,
        stoch_rank: int,
        hidden_dim: int,
        gru_layers: int = 2,
    ):
        super().__init__()
        
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_rank = stoch_rank
        
        norm = nn.LayerNorm

        # h: deterministic state
        # z: stochastic state
        self.init_h = nn.Parameter(torch.zeros((self.deter_dim)))
        self.init_z = nn.Parameter(torch.zeros(
            (self.stoch_dim * self.stoch_rank)))
     
        # 
        self.z_mlp = nn.Linear(stoch_dim * (stoch_rank or 1), hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)   
        self.in_norm = norm(hidden_dim, eps=1e-3)

        self.gru = GRUCellStack(
            hidden_dim, deter_dim, gru_layers) 
        
        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(
            hidden_dim, stoch_dim * (stoch_rank or 2))

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(
            hidden_dim, stoch_dim * (stoch_rank or 2))
        
    def init_state(self, batch_size):
        return (
            torch.tile(self.init_h, (batch_size, 1)),
            torch.tile(self.init_z, (batch_size, 1))
        )
        
    def forward(self, action, reset, in_state, embed=None):
                
        in_h, in_z = in_state
        B = action.shape[0]
        
        x = self.z_mlp(in_z) + self.a_mlp(action)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h) 
        
        if embed is not None:
            # concat in original dreamerv2, added in pydreamer
            x = self.post_mlp_h(h) + self.post_mlp_e(embed)
            norm_layer, mlp = self.post_norm, self.post_mlp
        else:
            x = self.prior_mlp_h(h)
            norm_layer, mlp = self.prior_norm, self.prior_mlp

        x = norm_layer(x)
        x = F.elu(x)
        pp = mlp(x)  # posterior or prior
        distr = self.zdistr(pp)
        sample = distr.rsample().reshape(B, -1)

        return pp, (h, sample)

    def batch_prior(self,
                    h: Tensor,     # tensor(T, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = posterior or prior
        logits = pp.reshape(
            pp.shape[:-1] + (self.stoch_dim, self.stoch_rank))
        distr = D.OneHotCategoricalStraightThrough(logits=logits.float())
        distr = D.independent.Independent(distr, 1)
        return distr   
        