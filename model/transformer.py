import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding
from model.decoder_layer import DecoderLayer
import constants
import utilities

device = utilities.getDevice()

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.decoder_embedding = nn.Embedding(constants.VOCAB_SIZE, constants.DIMENSIONS)
        self.positional_encoding = PositionalEncoding(constants.DIMENSIONS, constants.MAX_SEQ_LENGTH)

        self.decoder_layers = nn.ModuleList([DecoderLayer(constants.DIMENSIONS, constants.NUM_HEADS, constants.D_FF, constants.DROPOUT) for _ in range(constants.NUM_LAYERS)])

        self.fc = nn.Linear(constants.DIMENSIONS, constants.VOCAB_SIZE)
        self.dropout = nn.Dropout(constants.DROPOUT)

    def generate_mask(self, tgt):
        tgt_mask = (tgt != 3).unsqueeze(1).unsqueeze(3) # sentencepiece pad_id = 3
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask
    
    def num_params(self):
      gpt_params = sum(p.numel() for p in self.parameters())  # no. of parameters
      emb_params = (
          self.decoder_embedding.weight.numel()
      )  # no of parameters (weights) in the token embeddings
      print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
      return {"gpt_params": gpt_params, "emb_params": emb_params}

    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output