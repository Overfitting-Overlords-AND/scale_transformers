import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from model.transformer import Transformer
import sentencepiece as spm
import constants
import utilities
from data.tokenizer import Tokenizer
from data.textToSqlDataset import TextToSqlData, DATASET, PREFIX, corpusWriter

# Generate random sample data
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

device = utilities.getDevice()
print(f"Device = {device}")

def generate(question, context):

  tknz = Tokenizer(DATASET, PREFIX, corpusWriter) 
  question = torch.tensor(tknz.sp.encode_as_ids(question, add_bos=True)).long().unsqueeze(0).to(device)
  print("question shape: ", question.shape)
  context = torch.tensor(tknz.sp.encode_as_ids(context, add_bos=True)).long().unsqueeze(0)
  print("context shape: ", context.shape)
  answer = torch.tensor([[2]]) # <bos>=2
  
  print("answer shape: ", answer.shape)
  transformer = Transformer().to(device)
  transformer.eval()
  utilities.load_latest_checkpoint(transformer)
  
  for _ in range(constants.MAX_SEQ_LENGTH):
    with torch.no_grad():
      logits = transformer(question, context, answer) 
      logits = logits[:, -1, :] / 1.0
      probs = torch.nn.functional.softmax(logits, dim=-1)
      next = torch.multinomial(probs, num_samples=1)
      if next.item() == 3: break # </s>=3
      answer = torch.cat([answer, next], dim=1)

  output = tknz.sp.decode(answer.tolist()[0])
  return { "sql" : output } 
