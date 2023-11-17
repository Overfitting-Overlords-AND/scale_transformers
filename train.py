import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
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
      project="Scaling Transformer",
      
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
criterion = nn.CrossEntropyLoss(ignore_index=3) # sentencepiece pad_id = 3
optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

num_of_params = transformer.num_params()

if constants.WANDB_ON:
  wandb.log({"Total Parameters": num_of_params["gpt_params"], "Embeddings":  num_of_params["emb_params"]})

def getFileName(epoch,idx):
  return f"./transformer_epoch_{epoch+1}_{idx}.pt"

start_epoch = utilities.load_latest_checkpoint(transformer)

for epoch in range(start_epoch, constants.NUM_OF_EPOCHS):
  total_loss = 0
  for idx, tgt_data in tqdm.tqdm(enumerate(dl), desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    optimizer.zero_grad()
    output = transformer(tgt_data[0].to(device), tgt_data[1].to(device), tgt_data[2][:, :-1].to(device))
    loss = criterion(output.to(device).contiguous().view(-1, constants.VOCAB_SIZE), tgt_data[2][:, 1:].to(device).contiguous().view(-1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if idx % 1000 == 0: print(f"Loss: {loss.item():.4f}")
    if idx % 5000 == 0: torch.save(transformer.state_dict(), getFileName(epoch,idx))
  print(f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}, Loss: {total_loss}")
  torch.save(transformer.state_dict(), getFileName(epoch,idx))
  if constants.WANDB_ON:
    wandb.log({"acc": 2, "total_loss": total_loss})
  
if constants.WANDB_ON:
  wandb.finish()


