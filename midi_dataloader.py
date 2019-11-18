from os import walk, path
import os
from pathlib import Path
import numpy as np
import pickle
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

if os.name == 'nt':
    root_path = r'C:\Users\Kronos\Downloads\maestro-v2.0.0-midi'
else:
    root_path = path.expanduser('~/deep_learning/maestro-v2.0.0')


## TODO
# Remove modifiers
# Filter given signature

class MIDIDataset(Dataset):
    def __init__(self, root_path, sequence_length=50, fs=16, year=-1, add_limit_tokens=True, binarize=False, save_pickle=True):
        year = str(year) if not isinstance(year, str) else year
        self.sequence_length = sequence_length
        self.add_limit_tokens = add_limit_tokens
        self.binarize = binarize
        self.save_pickle = save_pickle

        # Load pickle if exists and return
        if add_limit_tokens:
            pickled_file = root_path + "/year_" + year + "_89.pkl"
        else:
            pickled_file = root_path + "/year_" + year + "_88.pkl"
        if Path(pickled_file).is_file():
            with open(pickled_file, 'rb') as f:
                pickle_content = pickle.load(f)
                self.midi_data = pickle_content[0]
                self.dataset_length = pickle_content[1]
                self.end_tokens = pickle_content[2]
                print("Loaded pickled dataset. Size = {}, Path: {}/training_data_year_{}.npy" .format(self.dataset_length, root_path, year))

            return

        # Create dataset
        self.midi_files = []
        for (dirpath, dirnames, filenames) in walk(root_path):  
            if os.name == 'nt':
                ff = [dirpath + "\\" + file for file in filenames if ".midi" in file]
            else:
                ff = [dirpath + "/" + file for file in filenames if ".midi" in file]
            if year == -1 or year in dirpath:
                self.midi_files.extend(ff)

        # Logging
        counter = 0
        self.file_count = len(self.midi_files)

        self.dataset_length = 0
        self.end_tokens = []
        self.midi_data = []
        for idx, file in enumerate(self.midi_files):
            piano_midi = pretty_midi.PrettyMIDI(file)
            piano_roll = piano_midi.get_piano_roll(fs=fs)[21:109, :]
            if self.add_limit_tokens:
                limit_array = np.zeros((1, piano_roll.shape[1]))
                limit_array[:,0] = 1
                limit_array[:,-1] = 1
                piano_roll = np.vstack((piano_roll, limit_array))

            self.midi_data.append(piano_roll)

            self.dataset_length += piano_roll.shape[1] - (self.sequence_length - 1) # Remove uncomplete sequences from choices
            self.end_tokens.append(self.dataset_length-1)
            counter += 1
            if counter % 5 == 0:
                print("loading progress {:.2f}%" .format(counter / self.file_count * 100))

        # Pickle dataset
        if self.save_pickle:
            with open(pickled_file, 'wb') as f:
                pickle.dump((self.midi_data, self.dataset_length, self.end_tokens), f)


    def __len__(self):
        return self.dataset_length


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start, list_idx = 0, 0
        for end_keys in self.end_tokens:
            if idx <= end_keys:
                break
            start = end_keys + 1
            list_idx += 1

        # Adjust for uncomplete sequences
        relative_idx = idx - start + (self.sequence_length - 1)
                
        # Get sequence
        sequence = self.midi_data[list_idx][:, relative_idx - (self.sequence_length - 1) : relative_idx + 1]

        # Binarize if set
        if self.binarize:
            sequence = np.clip(sequence, 0, 1)

        return np.transpose(sequence)
    


allMIDI = MIDIDataset(root_path, fs=16, year=2004, add_limit_tokens=True, binarize=False)

# Split dataset
batch_size = 10
validation_split = .2
shuffle_dataset = True

# Creating data indices for training and validation splits:
dataset_size = len(allMIDI)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(allMIDI, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(allMIDI, batch_size=batch_size, sampler=valid_sampler, drop_last=True, num_workers=0)

dataloader_all = DataLoader(allMIDI, batch_size=10, shuffle=True, num_workers=0, drop_last=True)

for i_batch, sample_batched in enumerate(dataloader_all):
    print(i_batch, sample_batched.shape)
    
    if i_batch == 50:
        break

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched.shape)
    
    if i_batch == 50:
        break

for i_batch, sample_batched in enumerate(validation_loader):
    print(i_batch, sample_batched.shape)
    
    if i_batch == 50:
        break
