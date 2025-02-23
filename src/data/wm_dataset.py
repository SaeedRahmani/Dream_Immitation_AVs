import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset
from omegaconf import DictConfig


class WorldModelDataset(Dataset):
    '''
    This class is responsible for loading and managing the dataset used for training the World Model. 
    It reads data from .npz files, processes the states and actions, and provides them in a format 
    suitable for training the models.
    Attributes:
        seq_length (int): The sequence length for the dataset.
        dataset_path (str): The path to the dataset directory.
        file_list (list): A list of file paths for the dataset files.
        num_scenarios (int): The number of scenarios (files) in the dataset.
    Methods:
        __len__(): Returns the number of scenarios in the dataset.
        __getitem__(index): Loads and returns the states and actions for a given index.
    '''
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.seq_length = cfg.wm.seq_length
        self.dataset_path = cfg.wm.dataset_path
        
        self.file_list = [file for file in Path(self.dataset_path).iterdir() 
                     if file.is_file()]
        self.num_scenarios = len(self.file_list)
        
    def __len__(self):
        return self.num_scenarios
        
    def __getitem__(self, index):
        # NpzFile 'data\\tfrecord-00001-of-01000_307.npz' 
        # with keys: states, actions
        
        # Load the .npz file and extract the states and actions.
        npz = np.load(self.file_list[index], allow_pickle=True)
        # Performs some preprocessing on the states (e.g., cropping images) and actions (e.g., handling missing values).
        states = torch.Tensor(npz["states"]).permute(0,3,1,2)
        states = states[:,:,:128,:128]
        actions = npz["actions"] # torch.tensor(npz["actions"])
        actions = np.where(actions == None, 0, actions)
        actions = torch.Tensor(actions.astype("float32"))
        
        return states, actions
        