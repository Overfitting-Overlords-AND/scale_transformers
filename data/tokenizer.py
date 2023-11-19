import sentencepiece as spm
import datasets
import os
import constants

PACKAGE = "data"

def corpusWriter(dataset,file):
  for text in dataset["train"]:
    file.write(text)
 
# let's create a tokenizer class with an encode and decode function, a train function, shifting the ids,
class Tokenizer:
    def __init__(self, dataset, fileName, corpusWriter=corpusWriter):
        self.fileName = fileName
        self.dataset = dataset
        self.modelName = f"{self.fileName}.model"
        self.corpusName = f"{self.fileName}.txt"
        self.corpusWriter = corpusWriter
        self.sp = spm.SentencePieceProcessor()
        if not self.modelExists():
            if not self.corpusExists():
                self.createCorpus()
            self.train()            
        self.sp.load(f"{PACKAGE}/{self.modelName}")
    
    def encode(self, txt):
        return self.sp.encode_as_ids(txt)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def createCorpus(self):
        data = datasets.load_dataset(self.dataset)
        with open(f"{PACKAGE}/{self.corpusName}", 'w', encoding='utf-8') as file:
            self.corpusWriter(data,file)

    def corpusExists(self):
        return os.path.exists(f"{PACKAGE}/{self.corpusName}")

    def modelExists(self):
        return os.path.exists(f"{PACKAGE}/{self.modelName}")

    def train(self):
        spm.SentencePieceTrainer.Train(
            input=f"{PACKAGE}/{self.corpusName}",
            model_type="bpe",
            model_prefix=f"{PACKAGE}/{self.fileName}",
            vocab_size=constants.VOCAB_SIZE,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            bos_piece="[BOS]",
            eos_piece="[EOS]",
            user_defined_symbols=["CREATE", "TABLE", "INTEGER", "SELECT", "COUNT(*)", "COUNT", "FROM", "WHERE", "VARCHAR", "ORDER", "MAX", "MIN", "AVG", "BETWEEN", "DISTINCT", "JOIN", "AS", "ON", "GROUP", "BY", "HAVING", "LIMIT", "ASC", "DESC", "LIKE", "INTERSECT", "EXCEPT"]
        )

        return self

    def vocab_size(self):
        return self.sp.get_piece_size()