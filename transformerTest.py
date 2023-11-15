from data.dataset import TinyStoriesData
import constants
import torch
from model.transformer import Transformer

ds = TinyStoriesData("train")
dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=ds.collate_function)

transformer = Transformer(constants.VOCAB_SIZE, constants.DIMENSIONS, constants.NUM_HEADS, constants.NUM_LAYERS, constants.D_FF, constants.MAX_SEQ_LENGTH, constants.DROPOUT)
transformer.eval()

for batch_idx, tgt_data  in enumerate(dl):
    if batch_idx >= 100:
        break
    output = transformer(tgt_data[:, :-1])
    print(f"Shape = {output.shape}")
    print(f"Target Data = {output}")

