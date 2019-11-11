import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, file_path, sequence_length=50):
        self.piano_midi = pretty_midi.PrettyMIDI(file_path)
        self.piano_roll = self.piano_midi.get_piano_roll()
        self.piano_roll_by_time = self.piano_roll.reshape(-1, 128)
        self.sequence_length = sequence_length

    def __len__(self):
        return self.piano_roll_by_time.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = self.piano_roll_by_time[idx]
        
        # Add padding
        if idx <= self.sequence_length:
            if idx == 0:
                past = np.zeros((self.sequence_length, 128))
            else:
                padding_size = self.sequence_length - (idx - 1)
                padding = np.zeros((padding_size, 128))
                past = self.piano_roll_by_time[0 : idx - 1]
                past = np.concatenate(padding, past)
        else:
            past = self.piano_roll_by_time[idx - self.sequence_length - 1 : idx - 1]
            
        sample = {'target' : target, 'past' : past.flatten()}

        return sample
    
myset = MIDIDataset(r'C:\Users\Kronos\Downloads\maestro-v2.0.0-midi\maestro-v2.0.0\2004\11.midi')

dataloader = DataLoader(myset, batch_size=10, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['target'].size(), sample_batched['past'].size())
    
    if i_batch == 0:
        break
