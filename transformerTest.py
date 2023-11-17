from data.textToSqlDataset import TextToSqlData
import constants
import torch
from model.transformer import Transformer

ds = TextToSqlData("train")
dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=ds.collate)

transformer = Transformer()
transformer.eval()

for batch_idx, tgt_data  in enumerate(dl):
    if batch_idx >= 100:
        break
    output = transformer(tgt_data[0], tgt_data[1], tgt_data[2][:, :-1])
    print(f"Shape = {tgt_data[0].shape}")
    print(f"Shape = {tgt_data[1].shape}")
    print(f"Shape = {tgt_data[2].shape}")
    print(f"Shape = {output.shape}")
    print(f"Target Data = {output}")

