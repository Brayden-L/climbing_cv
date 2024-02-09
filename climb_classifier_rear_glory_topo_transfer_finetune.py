# %%
# Imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch_lr_finder import LRFinder
import wandb

import numpy as np
from dotenv import dotenv_values
import os

# %%
# WandB Init
os.environ["WANDB_API_KEY"] = dotenv_values(".env")['WANDB_API_KEY']
wandb.login()
wandb.init(project="climb_classifier_rear_glory_topo_transfer_finetune")
#%%
# Seed freeze
torch.manual_seed(1220)

# %%
# Training device selection
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%
# Data download