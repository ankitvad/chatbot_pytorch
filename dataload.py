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
        return torch.from_numpy(np.array(self.data[idx])).long()

"""
slice a dialogue with size limit and make example
slide over the whole dialogue, the next sequence share the end of the last sequence
all results are padded with 0s if the length is less than limit
"""
def slice_dialog(dialogue, limit=80):
    exs = []
    start = 0
    while ((len(dialogue) - 1) > start):
        length = limit
        if start + limit > len(dialogue):  # padding 0
            length = len(dialogue) - start
            dialogue.extend([0] * (start + limit - len(dialogue)))
        exs.append(dialogue[start:start + limit])
        if(len(dialogue[start:start + limit])!=limit):
            print('fuck')
        start += (limit - 1)
    return exs

# dialogs: list of dialogs, each dialog is a list of word indexes
def create_array(dialogs):
    d = []
    for i, dialog in enumerate(dialogs):
        print(i)
        exs = slice_dialog(dialog)
        d.extend(exs)
    return d
