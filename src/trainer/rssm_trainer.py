import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset, Sampler
from torch.amp import GradScaler, autocast
from ..models.world_model import WorldModel
from ..data.wm_dataset import WorldModelDataset

class BatchSamplerSkipSmall(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist()
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.data_source) // self.batch_size

class RSSMTrainer(object):
    '''
    This class is responsible for training the world model.
    It initializes the model, optimizer, and dataloaders, and provides the train method.
    '''
    def __init__(self, cfg: DictConfig, do_val=False):
        self.cfg = cfg
        self.batch_size = self.cfg.wm.batch_size
        
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.cfg.wm.device)
        print(f"Using {self.device}")
        
        self.model = WorldModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.cfg.wm.lr, eps=self.cfg.wm.eps)
        
        self.do_val = do_val
        self.dataloaders: dict = self.build_dataloaders() 
        
        self.scaler = GradScaler(enabled=True)
        
    def split_train_val_dataset(self, dataset: Dataset, split=0.8):
        split_idx = int(len(dataset)*split)
        idxs = torch.arange(len(dataset))
        train_dataset = Subset(dataset, idxs[:split_idx])
        val_dataset = Subset(dataset, idxs[split_idx:])
        # print("set lens:", len(train_dataset), len(val_dataset))
        return train_dataset, val_dataset        
        
    def build_dataloader(self, dataset: Dataset) -> DataLoader:
        # batch_sampler = EquiSampler(
        #     len(dataset), dataset_config.seq_length, dataset_config.batch_size)
        # def transpose_collate(batch):
        #     """
        #     Transposes batch and time dimension
        #     (B, T, ...) -> (T, B, ...)
        #     """
        #     from torch.utils.data._utils.collate import default_collate
        #     return {k: torch.transpose(v, 0, 1) for k, v in default_collate(batch).items()}
        skipSmall_sampler = BatchSamplerSkipSmall(data_source=dataset, batch_size=self.batch_size)

        dataloader = DataLoader(dataset,
                                pin_memory=True,
                                # batch_size=self.batch_size,
                                batch_sampler=skipSmall_sampler,
                                # collat
                                # 
                                # e_fn=custom_collate_fn
                                )
        return dataloader        
                
    def build_dataloaders(self) -> dict[str, DataLoader]:
        dataset = WorldModelDataset(self.cfg)
        
        # split into train and val
        train_dataset, val_dataset = \
            self.split_train_val_dataset(dataset, split=self.cfg.wm.dataset_split)
            
        train_dataloader = self.build_dataloader(train_dataset)
        dataloaders = {"train": train_dataloader}
        
        if self.do_val:
            val_dataloader = self.build_dataloader(val_dataset)
            dataloaders["val"] = val_dataloader
        return dataloaders

    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            print(epoch)
            in_states = self.model.init_state(self.batch_size)
            for idx, batch in enumerate(self.dataloaders["train"]):
                print(epoch, idx)
                states, actions = batch
                states = states.to(self.device)
                actions = actions.permute(1,0,2).to(self.device)
                actions = torch.nan_to_num(actions, nan=0.0)    # replace NaN value by 0
                print(states.shape)
                self.optimizer.zero_grad()
                
                with autocast(enabled=True, device_type=str(self.device)):
                    batch_metrics, decoded_img, out_states, samples = self.model.training_step(
                        states=states, actions=actions, resets=None, in_state=in_states)
                in_states = out_states

                # # put before param updates
                # self.log_stats(batch_metrics, samples, decoded_img)

                self.scaler.scale(batch_metrics["loss"]).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200)
                self.scaler.step(self.optimizer)
                self.scaler.update()

    # def validate(self):
    #     pass
    
    # def log_stats(self, batch_metrics, samples, decoded_img):
    #     steps = self.metrics_helper.step_dict["train"]

    #     if steps % (len(self.dataloaders["train"])) == 0:
    #         self.validate() if self.do_val else None
    #         pred_img = self.model.pred_img(*samples)
    #         self.metrics_helper.log_imgs(
    #             samples[-1], decoded_img, pred_img, "train")

    #     self.metrics_helper.update_stats("train", batch_metrics)

    #     if steps % 25 == 0:
    #         self.pbar.update(25)

    #     if self.do_checkpoint:
    #         if self.c_idx == len(self.checkpoints):
    #             exit()
    #         elif steps > 0 and steps % self.checkpoints[self.c_idx] == 0:
    #             self.write_checkpoint(steps)