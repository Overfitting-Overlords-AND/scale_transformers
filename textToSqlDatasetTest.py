from data.textToSqlDataset import TextToSqlData

# Test the TextToSqlData class
ds = TextToSqlData("train")
print('ds.ds', ds.dataset)
print(ds.__getitem__(0))
print('len(ds)', len(ds))
print('ds[362]', ds[362])