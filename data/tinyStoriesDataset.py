import torch
from datasets import load_dataset
from data.tokenizer import Tokenizer

DATASET = "roneneldan/TinyStories"
PREFIX = "tiny_stories"

def corpusWriter(dataset,file):
  for text in dataset["train"]['text']:
    file.write(text)
    
class TinyStoriesData(torch.utils.data.Dataset):
  def __init__(self, mode):
    self.mode = mode
    self.dataset = load_dataset(DATASET,split=self.mode)
    self.tknz = Tokenizer(DATASET, PREFIX, corpusWriter) 

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sentence = self.dataset[idx]["text"]
    encoded =self.tknz.sp.encode_as_ids(sentence, add_bos=True, add_eos=True)
    return torch.tensor(encoded)
  
  def collate(self, batch):
    return torch.nn.utils.rnn.pad_sequence([item for item in batch], batch_first=True, padding_value=3)