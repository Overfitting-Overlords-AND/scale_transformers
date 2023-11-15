import sentencepiece as spm
import datasets
import os

PREFIX = "tiny_stories"
PACKAGE = "data"

# let's create a tokenizer class with an encode and decode function, a train function, shifting the ids,
class TinyTokenizer:
    def __init__(self, mode):
        self.mode = mode
        self.prefix = f"{PREFIX}_{self.mode}"
        self.modelName = f"{self.prefix}.model"
        self.corpusName = f"{self.prefix}.txt"
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
        dataset = datasets.load_dataset("roneneldan/TinyStories")
        with open(self.corpusName,'w',encoding='utf-8') as file:
            for text in dataset[self.mode]['text']:
                file.write(text)

    def corpusExists(self):
        return os.path.exists(f"{PACKAGE}/{self.corpusName}")

    def modelExists(self):
        return os.path.exists(f"{PACKAGE}/{self.modelName}")

    def train(self):
        spm.SentencePieceTrainer.Train(
            input=self.corpusName,
            model_type="bpe",
            model_prefix=self.prefix,
            vocab_size=16000,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            bos_piece="[BOS]",
            eos_piece="[EOS]",
        )

        return self

    def vocab_size(self):
        return self.sp.get_piece_size()