from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

#data: list of numpy array, num*max_len
class dialogdata(Dataset):
    def __init__(self, data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(1+np.array(self.data[idx])).long()

"""
slice a dialogue with size limit and make example
slide over the whole dialogue, the next sequence share the end of the last sequence
all results are padded with 0s if the length is less than limit
"""
def slice_dialog(dialog, limit=80):
    exs = []
    start = 0
    while ((len(dialogue) - 1) > start):
        length = limit
        if start + limit > len(dialogue):  # padding 0
            length = len(dialogue) - start
            dialogue.extend([-1] * (start + limit - len(dialogue)))
            start += (limit - 1)
            exs.append(dialogue[start:start + limit])
    return exs

def create_array(dialogs):
    d = []
    for dialog in dialogs:
        exs = slice_dialog(dialog)
        d.extend(exs)
    return d
