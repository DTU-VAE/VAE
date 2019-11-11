from os import walk
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, root_path, sequence_length=50):
        self.midi_files = []
        for (dirpath, dirnames, filenames) in walk(root_path):
            ff = [dirpath + "\\" + file for file in filenames if ".midi" in file]
            self.midi_files.extend(ff)
        
        self.midi_end = 0
        self.midi_dict = {} #end idx : file
        self.ordered_keys = [] # for faster lookup
        for file in self.midi_files:
            piano_midi = pretty_midi.PrettyMIDI(file)
            piano_roll = piano_midi.get_piano_roll()
            start = self.midi_end
            self.midi_end += piano_roll.shape[1]
            self.midi_dict[self.midi_end] = (start, file)
            self.ordered_keys.append(self.midi_end)
        
        self.sequence_length = sequence_length

    def __len__(self):
        return self.midi_end

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file
        start = 0
        file = ""
        for i, end_keys in enumerate(self.ordered_keys):
            if idx > start and idx <= end_keys:
                start, file = self.midi_dict[self.ordered_keys[i]]
                break
            start = end_keys + 1
        
        piano_midi = pretty_midi.PrettyMIDI(file)
        piano_roll = piano_midi.get_piano_roll()
        piano_roll_by_time = piano_roll.reshape(-1, 128)
        relative_idx = idx - start
        
        target = piano_roll_by_time[relative_idx]
        
        # Add padding
        if relative_idx <= self.sequence_length:
            if relative_idx == 0:
                past = np.zeros((self.sequence_length, 128))
            else:
                padding_size = self.sequence_length - (relative_idx - 1)
                padding = np.zeros((padding_size, 128))
                past = piano_roll_by_time[0 : relative_idx - 1]
                print(past)
                past = np.concatenate(padding, past)
        else:
            past = piano_roll_by_time[relative_idx - self.sequence_length - 1 : relative_idx - 1]
            
        sample = {'target' : target, 'past' : past.flatten()}

        return sample
    
allMIDI = MIDIDataset(r'C:\Users\Kronos\Downloads\maestro-v2.0.0-midi\maestro-v2.0.0')

dataloader = DataLoader(allMIDI, batch_size=10, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['target'].size(), sample_batched['past'].size())
    
    if i_batch == 0:
        break
