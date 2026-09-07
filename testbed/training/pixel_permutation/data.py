import torch
from torch.utils.data import Dataset


class PermutedData(Dataset):
    def __init__(self, source, indices, permutation, axes, *, augment=False):
        self.source, self.indices = source, torch.as_tensor(indices, dtype=torch.long)
        self.permutation, self.axes, self.augment = permutation, axes, augment

    @property
    def ids(self):
        return self.source.ids[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source.read(int(self.indices[index]), augment=self.augment, permutation=self.permutation, axes=self.axes)
