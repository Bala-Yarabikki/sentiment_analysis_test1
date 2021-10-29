from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tokenizer_encoder import dataset_train, dataset_val

batch_size = 4

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)