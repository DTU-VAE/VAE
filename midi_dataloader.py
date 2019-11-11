from os import walk, path
import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, root_path, sequence_length=50):
        self.sequence_length = sequence_length
        
        self.midi_files = []
        for (dirpath, dirnames, filenames) in walk(root_path):
            ff = [dirpath + "/" + file for file in filenames if ".midi" in file]
            self.midi_files.extend(ff)
        
        counter = 1
        limit = len(self.midi_files)

        piano_midi = pretty_midi.PrettyMIDI(self.midi_files[0])
        piano_roll = piano_midi.get_piano_roll().astype(np.int8)[21:107, :]

        self.midi_end = piano_roll.shape[1]
        self.midi_array = piano_roll
        self.end_tokens = [self.midi_end]
        for file in self.midi_files[1:]:
            piano_midi = pretty_midi.PrettyMIDI(file)
            piano_roll = piano_midi.get_piano_roll().astype(np.int8)[21:107, :]

            self.midi_end += piano_roll.shape[1]
            self.end_tokens.append(self.midi_end)
            self.midi_array = np.concatenate((self.midi_array, piano_roll), axis=1)
            counter += 1
            if counter % 10 == 0:
                print("loading progress {:.2f}%" .format(counter / limit * 100))
                break
        
        print("Saving training set to binary")
        np.save(root_path + "/training_data.npy", self.midi_array)
        print("Array saved as" + root_path + "/training_data.npy")


    def __len__(self):
        return self.midi_end

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = 0
        for end_keys in self.end_tokens:
            if idx > end_keys:
                break
            start = end_keys + 1
        
        relative_idx = idx - start
                
        # Add padding
        if relative_idx < self.sequence_length:
            if relative_idx == 0:
                sequence = np.zeros((128, self.sequence_length))
            else:
                padding_size = self.sequence_length - relative_idx
                padding = np.zeros((128, padding_size))
                sequence = self.midi_array[start : start + relative_idx]
                sequence = np.concatenate((padding, sequence), axis=1)
        else:
            sequence = self.midi_array[:, start + relative_idx - self.sequence_length : start + relative_idx]
            
        return np.transpose(sequence)
    
allMIDI = MIDIDataset(path.expanduser('~/deep_learning/maestro-v2.0.0'))

dataloader = DataLoader(allMIDI, batch_size=1, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched.size())
    
    if i_batch == 0:
        break
