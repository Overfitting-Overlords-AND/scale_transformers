from data.dataset import TinyStoriesData

ds = TinyStoriesData("train")
print('ds.ds', ds.dataset)
print('len(ds)', len(ds))
print('ds[362]', ds[362])