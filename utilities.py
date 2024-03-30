import torch
import os
import re

# Function to get the device
def getDevice():
  is_cuda = torch.cuda.is_available()
  return "cuda:0" if is_cuda else "cpu"

def find_latest_epoch_file(path='./'):
    epoch_files = [f for f in os.listdir(path) if re.match(r'transformer_epoch_\d+\_\d+\.pt', f)]
    if epoch_files:
        # Extracting epoch numbers from the files and finding the max
        latest_epoch = max([int(f.split('_')[2]) for f in epoch_files])
        latest_iteration = max([int(f.split('_')[3].split('.')[0]) for f in epoch_files if re.match(rf'transformer_epoch_{latest_epoch}_\d+\.pt', f)])
        return latest_epoch, f"./transformer_epoch_{latest_epoch}_{latest_iteration}.pt"
    else:
        return 0, None

# Function to load the latest epoch file if it exists
def load_latest_checkpoint(model, path='./'):
    latest_epoch, latest_file = find_latest_epoch_file(path)
    if latest_file:
        print(f"Resuming training from epoch {latest_epoch+1}")
        model.load_state_dict(torch.load(latest_file, map_location=torch.device(getDevice())))
    else:
        print("No checkpoint found, starting from beginning")
    return latest_epoch
