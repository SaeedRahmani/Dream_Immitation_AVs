import hydra
import torch
from omegaconf import DictConfig

from models.encoders import Encoder
from models.rssm import RSSMCore, RSSMCell
from models.world_model import WorldModel
from data.wm_dataset import WorldModelDataset
from trainer.rssm_trainer import RSSMTrainer

@hydra.main(config_name="conf.yaml", config_path="config/", version_base="1.3")
def main(cfg: DictConfig):
    
    trainer = RSSMTrainer(cfg)
    trainer.train()
    
    # wm_dataset = WorldModelDataset(cfg)
    # s, a = wm_dataset[1]
    # print(a.shape, a.dtype)
    # print(s.shape)
    
    # encoder = Encoder(cfg)
    # wm = WorldModel(cfg)
    # # rssm_core = RSSMCore()    

    # states = torch.ones(cfg.wm.batch_size, cfg.wm.seq_length, 3, cfg.wm.img_height, cfg.wm.img_width)
    # actions = torch.ones(cfg.wm.seq_length, cfg.wm.batch_size, cfg.wm.action_dim)
    # resets = torch.ones(cfg.wm.batch_size, cfg.wm.seq_length, 1)
    
    # embed = encoder(states)
    # # embed = wm(states, None, None, None)
    
    # rssm_cell = RSSMCell(
    #     embed_dim=9216,
    #     action_dim=cfg.wm.action_dim,
    #     deter_dim=100,
    #     stoch_dim=32,
    #     stoch_rank=5,
    #     hidden_dim=100,
    # )
    # # init_state = rssm_cell.init_state(cfg.wm.batch_size)
    # # pp, (h, z) = rssm_cell(actions[0], None, init_state, embed)
    
    # # print("Embed:", embed.shape)
    # # print("PP", pp.shape)
    # # print("Deter", h.shape)
    # # print("Stoch", z.shape)
    
    # (h, z) = wm.init_state(cfg.wm.batch_size) 
    # batch_metrics, decoded_img, out_states, samples = wm.training_step(states, actions, resets, (h,z))
    # # print("Feature", _.shape)
    # # print("Deter state", h_.shape)
    # # print("Stoch state", z_.shape)
    # print("Decoded Img:", decoded_img.shape)
    # print(batch_metrics)

if __name__ == "__main__":
    main()