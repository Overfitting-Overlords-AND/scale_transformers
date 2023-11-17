import torch
from datasets import load_dataset
from data.tokenizer import Tokenizer

DATASET = "b-mc2/sql-create-context"
PREFIX = "text_to_sql"

def corpusWriter(dataset,file):
  for obj in dataset:
    for key, value in obj.items():
      file.write(f"{value}\n")
    file.write("\n")  # Add a newline to separate objects

class TextToSqlData(torch.utils.data.Dataset):
  def __init__(self, mode):
    self.mode = mode
    self.dataset = load_dataset(DATASET,split=self.mode)
    self.tknz = Tokenizer(DATASET, PREFIX, corpusWriter) 

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    record = self.dataset[idx]
    encoded_question = self.tknz.sp.encode_as_ids(record["question"], add_bos=True, add_eos=True)
    encoded_context = self.tknz.sp.encode_as_ids(record["context"], add_bos=True, add_eos=True)
    encoded_answer = self.tknz.sp.encode_as_ids(record["answer"], add_bos=True, add_eos=True)
    return torch.tensor(encoded_question), torch.tensor(encoded_context), torch.tensor(encoded_answer)
  
  def collate(self, batch):
    q = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    c = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    a = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
    return q, c, a
  