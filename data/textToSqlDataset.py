import torch
from datasets import load_dataset
from data.tokenizer import Tokenizer

DATASET = "b-mc2/sql-create-context"
PREFIX = "text_to_sql"

def corpusWriter(dataset,file):
  for obj in dataset["train"]:
    for key, value in obj.items():
      file.write(f"{value}\n")
    file.write("\n")  # Add a newline to separate objects

class TextToSqlData(torch.utils.data.Dataset):
  def __init__(self, mode):
    self.mode = mode
    full_dataset = load_dataset(DATASET,split="train")  # This dataset only has training set
    self.tknz = Tokenizer(DATASET, PREFIX, corpusWriter) 
    # First, split the dataset into train and temp (temp will be further split into validation and test)
    temp_test_dataset = full_dataset.train_test_split(test_size=0.15)  # 15% for temp (validation + test)

    # Now, split the temp dataset into validation and test
    val_test_dataset = temp_test_dataset["test"].train_test_split(test_size=0.33)  # validation 10% and test 5% of total dataset

    self.dataset = {
      "train" : temp_test_dataset["train"],
      "validate" : val_test_dataset["train"],
      "test" : val_test_dataset["test"]
    }
    
  def setMode(self,mode):
    self.mode = mode

  def __len__(self):
    return len(self.dataset[self.mode])

  def __getitem__(self, idx):
    record = self.dataset["train"][idx]
    encoded_question = self.tknz.sp.encode_as_ids(record["question"], add_bos=True, add_eos=True)
    encoded_context = self.tknz.sp.encode_as_ids(record["context"])
    encoded_answer = self.tknz.sp.encode_as_ids(record["answer"], add_bos=True, add_eos=True)
    return torch.tensor(encoded_question), torch.tensor(encoded_context), torch.tensor(encoded_answer)
  
  def collate(self, batch):
    q = torch.nn.utils.rnn.pad_sequence([torch.cat((item[0],item[1]), dim=0) for item in batch], batch_first=True, padding_value=0)
    a = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
    return q, a
  