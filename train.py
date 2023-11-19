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
import wandb
import utilities

# start a new wandb run to track this script
if constants.WANDB_ON:
  wandb.init(
      # set the wandb project where this run will be logged
      project="Scaling Transformer WPP",
      
      # track hyperparameters and run metadata
      config={
      "learning_rate": constants.LEARNING_RATE,
      "dimensions": constants.DIMENSIONS,
      "dataset": constants.DATASET,
      "vocab_size": constants.VOCAB_SIZE,
      "epochs": constants.NUM_OF_EPOCHS,
      "num_heads" : constants.NUM_HEADS,
      "num_layers" : constants.NUM_LAYERS,
      "d_ff" : constants.D_FF,
      "max_seq_length" : constants.MAX_SEQ_LENGTH,
      "dropout" : constants.DROPOUT
      }
  )

device = utilities.getDevice()
print(f"Device = {device}")

ds = TextToSqlData("train")
dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=ds.collate)

# Generate random sample data
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

transformer = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # sentencepiece pad_id = 0
optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

transformer.train()

num_of_params = transformer.num_params()

if constants.WANDB_ON:
  wandb.log({"Total Parameters": num_of_params["gpt_params"], "Embeddings":  num_of_params["emb_params"]})

def getFileName(epoch,idx):
  return f"./transformer_epoch_{epoch+1}_{idx}.pt"


def print_tensor(tensor, name):
  print("=========/n")
  print(f"{name}/n")
  print("=========/n/n")
  print(f"Shape = {tensor.shape}")
  print(tensor)
  print("/n/n")

start_epoch = utilities.load_latest_checkpoint(transformer)

for epoch in range(start_epoch, constants.NUM_OF_EPOCHS):

  total_loss = 0
  total_labels = 0
  correct = 0
  ds.setMode("train")
  
  for idx, tgt_data in tqdm.tqdm(enumerate(dl), desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    
    optimizer.zero_grad()

    query, context, label = (x.to(device) for x in tgt_data)
    label = label[:, :-1]
    if idx == 0:
      for tensor, name in zip([tgt_data[2], query, context, label], ["label", "query", "context", "adjusted label"]): print_tensor(tensor, name)
    
    output = transformer(query, context, label)
    outputForLoss = output.to(device).contiguous().view(-1, constants.VOCAB_SIZE)
    labelForLoss = tgt_data[2][:, 1:].to(device).contiguous().view(-1)
    if idx == 0:
      for tensor, name in [(output, "output"), (outputForLoss, "outputForLoss"), (labelForLoss, "labelForloss")]: print_tensor(tensor, name)
    loss = criterion(outputForLoss, labelForLoss)
    
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    _, predicted = torch.max(outputForLoss, 1)
    total_labels += labelForLoss.size(0)
    correct += (predicted == labelForLoss).sum().item()

    if idx % 100 == 0: 
      print(f"Loss: {loss.item():.4f}")
      scheduler.step()
    
    if idx % 500 == 0: 
      torch.save(transformer.state_dict(), getFileName(epoch,idx))
      if constants.WANDB_ON:
        wandb.log({"acc": 2, "total_loss": total_loss})
      print(f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}, Loss: {total_loss}")

  average_loss = total_loss / len(dl)
  accuracy = correct / total_labels

  torch.save(transformer.state_dict(), getFileName(epoch,idx))
  print(f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}, Loss: {total_loss}")

  total_loss = 0
  total_labels = 0
  correct = 0
  ds.setMode("validate")
  
  for idx, tgt_data in tqdm.tqdm(enumerate(dl), desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    
    query, context, label = (x.to(device) for x in tgt_data)
    label = label[2][:, :-1]
    output = transformer(query, context, label)
    
    outputForLoss = output.to(device).contiguous().view(-1, constants.VOCAB_SIZE)
    labelForLoss = tgt_data[2][:, 1:].to(device).contiguous().view(-1)
    loss = criterion(outputForLoss, labelForLoss)
    
    total_loss += loss.item()
    _, predicted = torch.max(outputForLoss.data, 1)
    total_labels += label.size(0)
    correct += (predicted == labelForLoss).sum().item()

  average_loss = total_loss / len(dl)
  accuracy = correct / total_labels

  if constants.WANDB_ON:
    wandb.log({"val_acc": accuracy, "val_avg_loss": average_loss})

if constants.WANDB_ON:
  wandb.finish()