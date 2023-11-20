import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
#import intel_extension_for_pytorch as ipex
from torch.optim.lr_scheduler import ExponentialLR
from model.transformer import Transformer
from data.textToSqlDataset import TextToSqlData
import sentencepiece as spm
import tqdm
import constants
import wandbWrapper as wandb
import utilities

wandb.init()

device = utilities.getDevice()
print(f"Device = {device}")

ds = TextToSqlData("train")
dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=ds.collate)

transformer = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # sentencepiece pad_id = 0
optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

transformer.train()

num_of_params = transformer.num_params()

wandb.log({"Total Parameters": num_of_params["gpt_params"], "Embeddings":  num_of_params["emb_params"]})

def getFileName(epoch,idx):
  return f"./transformer_epoch_{epoch+1}_{idx}.pt"

def update_metrics(loss, outputForLoss, labelForLoss, total_loss, total_labels, correct):
    total_loss += loss.item()
    _, predicted = torch.max(outputForLoss, 1)
    total_labels += labelForLoss.size(0)
    correct += (predicted == labelForLoss).sum().item()
    return total_loss, total_labels, correct

def log_result( epoch, idx, correct, total_labels, total_loss):
    torch.save(transformer.state_dict(), getFileName(epoch, idx))
    wandb.log({"acc": correct / total_labels, "total_loss": total_loss, "average_loss": total_loss / len(dl)})
    print(f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}, Loss: {total_loss}")

start_epoch = utilities.load_latest_checkpoint(transformer)

for epoch in range(start_epoch, constants.NUM_OF_EPOCHS):

  total_loss, total_labels, correct = 0, 0, 0
  ds.setMode("train")
  
  for idx, tgt_data in tqdm.tqdm(enumerate(dl), desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    
    optimizer.zero_grad()

    query, label = (x.to(device) for x in tgt_data)
    label = label[:, :-1]   
    output = transformer(query, label)

    outputForLoss = output.to(device).contiguous().view(-1, constants.VOCAB_SIZE)
    labelForLoss = tgt_data[1][:, 1:].to(device).contiguous().view(-1)
    loss = criterion(outputForLoss, labelForLoss)

    loss.backward()
    optimizer.step()
    
    total_loss, total_labels, correct = update_metrics(loss, outputForLoss, labelForLoss, total_loss, total_labels, correct)

    if idx % 100 == 0: 
      print(f"Loss: {loss.item():.4f}")
      scheduler.step()
    
    if idx % 500 == 0: 
      torch.save(transformer.state_dict(), getFileName(epoch, idx))
      log_result(epoch, idx, correct, total_labels, total_loss)

  torch.save(transformer.state_dict(), getFileName(epoch, idx))
  log_result(epoch, idx, correct, total_labels, total_loss)

  total_loss, total_labels, correct = 0, 0, 0
  ds.setMode("validate")
  
  for idx, tgt_data in tqdm.tqdm(enumerate(dl), desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    
    query, label = (x.to(device) for x in tgt_data)
    label = label[:, :-1]   
    output = transformer(query, label)
    
    outputForLoss = output.to(device).contiguous().view(-1, constants.VOCAB_SIZE)
    labelForLoss = tgt_data[1][:, 1:].to(device).contiguous().view(-1)
    loss = criterion(outputForLoss, labelForLoss)
    
    total_loss, total_labels, correct = update_metrics(loss, outputForLoss, labelForLoss, total_loss, total_labels, correct)
  
  log_result(epoch, idx, correct, total_labels, total_loss)

wandb.finish()