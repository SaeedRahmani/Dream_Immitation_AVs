import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
from omegaconf import DictConfig

from .encoders import Encoder
from .decoders import Decoder
from .rssm import RSSMCore


def init_weights(m: nn.Module):
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)    


class WorldModel(nn.Module):
    '''
    This file defines the actual architecture and functionality of the world model. 
    It encapsulates the encoder, the RSSM core, and the decoder, along with the 
    logic for the training step and loss calculation. 
    Think of it as the blueprint for the model. 

    Combines the encoder, RSSM core, and decoder into a single model.
    Args:
        cfg: Configuration object.

    Methods:
        init_state(batch_size): Initializes the deterministic and stochastic states.
        forward(states, actions, resets, in_state): Encodes the states, processes
                                                   them through the RSSM, and returns
                                                   the latent features and final state.
        dream(action, in_state): Performs a "prior prediction" using the RSSM.
        training_step(states, actions, resets, in_state): Performs a single training
                                                         step and returns the batch
                                                         metrics, decoded image, final
                                                         state, and samples.
    '''
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.encoder = Encoder(cfg)
        self.rssm_core = RSSMCore(cfg)
        self.decoder = Decoder(cfg)
        
        # Kullback–Leibler divergence
        self.kl_balance = cfg.wm.kl_balance
        self.kl_weight = cfg.wm.kl_weight
        
        # feature dim [TODO: what is this?]
        self.feature_dim = cfg.wm.deter_dim + \
            cfg.wm.stoch_dim * cfg.wm.stoch_rank
        
        for m in self.modules():
            init_weights(m)
    
    def init_state(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """ Returns:
            Deterministic state, h, Tensor, with shape of B, Deter_dim 
            Stochastic state, z, Tensor, with shape of B, (Stoch_dim*Stoch_rank)
        """
        return self.rssm_core.init_state(batch_size)
        
    def pred_img(self):
        raise NotImplementedError()
    
    def unnormalize(self, img):
        raise NotImplementedError()
        
    def forward(
        self, 
        states: Tensor, 
        actions: Tensor, 
        resets, 
        in_state: tuple[Tensor, Tensor]) -> Tensor:
        """
        Args:
            states (Tensor): _description_
            actions (_type_): _description_
            resets (_type_): _description_
            in_state (_type_): _description_

        Returns:
            features (Tensor): 
            out_states (Tensor):
        """
        # encode the observation.
        embeds = self.encoder(states)

        (prior, post, post_samples,
         features, hidden_states, out_states) = \
            self.rssm_core.forward(
                                   actions,
                                   resets,
                                   in_state,
                                   embeds=embeds)
        return features, out_states        
    
    def dream(self, action, in_state) -> tuple[Tensor, Tensor]:
        """ Prior Predictor """
        _, (h, z) = self.rssm_core.cell.forward(
            action, in_state, embed=None)
        return (h, z)
    
    def training_step(
        self, 
        states,
        actions,
        resets,
        in_state,  
    ):
        embeds = self.encoder(states)
        
        (prior, post, post_samples,
         features, hidden_states, out_states) = \
            self.rssm_core.forward(
                                   actions,
                                   resets,
                                   in_state,
                                   embeds=embeds)
        
        loss_reconstr, loss_image, decoded_img = self.decoder(
            features, states)
                
        d = self.rssm_core.zdistr
        dprior = d(prior)
        dpost = d(post)

        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)
        loss_kl_post = D.kl.kl_divergence(dpost, d(prior.detach()))
        loss_kl_prior = D.kl.kl_divergence(d(post.detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + \
            self.kl_balance * loss_kl_prior

        loss = self.kl_weight * loss_kl + loss_reconstr

        entropy_prior = dprior.entropy()
        entropy_post = dpost.entropy()

        batch_metrics = {"loss": loss, "loss_kl": loss_kl, "loss_kl_exact": loss_kl_exact,
                         "loss_kl_post": loss_kl_post, "loss_kl_prior": loss_kl_prior,
                         "loss_image": loss_image, "entropy_prior": entropy_prior,
                         "entropy_post": entropy_post}
        batch_metrics = {k: v.mean() for k, v in batch_metrics.items()}

        samples = (prior, post_samples, features, states)
        return batch_metrics, decoded_img, out_states, samples